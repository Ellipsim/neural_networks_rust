use crate::functions::activation::{relu_derivative, sigmoid_derivative};
use crate::neural_network::layer::Layer;
use crate::neural_network::neuron::NeuronGradients;

pub struct Network {
    pub layers: Vec<Layer>,
}

pub struct ForwardCache {
    pub layer_inputs: Vec<Vec<f32>>,
    pub weighted_sums: Vec<Vec<f32>>,
    pub layer_outputs: Vec<Vec<f32>>,
}

pub struct NetworkGradients {
    pub layers: Vec<Vec<NeuronGradients>>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn from_random(architecture: &[usize], activation: &str) -> Self {
        assert!(architecture.len() >= 2);

        let mut layers = Vec::new();

        for i in 0..architecture.len() - 1 {
            let input_size = architecture[i];
            let neuron_count = architecture[i + 1];

            let layer = Layer::from_random(input_size, neuron_count, activation);

            layers.push(layer);
        }

        Self { layers }
    }

    pub fn forward(&self, inputs: &[f32], activation: &str) -> Vec<f32> {
        let mut outputs = inputs.to_vec();

        for layer in &self.layers {
            outputs = layer.forward(&outputs, activation);
        }

        outputs
    }

    pub fn forward_with_activations(&self, inputs: &[f32], activations: &[&str]) -> Vec<f32> {
        assert_eq!(self.layers.len(), activations.len());

        let mut outputs = inputs.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            outputs = layer.forward_with_activation(&outputs, activations[i]);
        }

        outputs
    }

    pub fn calculate_hidden_delta(
        &self,
        next_layer_weights: &[f32],
        next_layer_deltas: &[f32],
        activation: &str,
        weighted_sum: f32,
    ) -> f32 {
        assert_eq!(next_layer_weights.len(), next_layer_deltas.len());

        let mut error = 0.0;

        for i in 0..next_layer_weights.len() {
            error += next_layer_weights[i] * next_layer_deltas[i];
        }

        let derivative = match activation {
            "relu" => relu_derivative(weighted_sum),
            "sigmoid" => sigmoid_derivative(weighted_sum),
            _ => panic!("Unknown activation function"),
        };

        error * derivative
    }


    pub fn forward_with_cache(
        &self,
        inputs: &[f32],
        activations: &[&str],
    ) -> (Vec<f32>, ForwardCache) {
        assert_eq!(self.layers.len(), activations.len());

        let mut current_outputs = inputs.to_vec();

        let mut layer_inputs = Vec::with_capacity(self.layers.len());
        let mut weighted_sums = Vec::with_capacity(self.layers.len());
        let mut layer_outputs = Vec::with_capacity(self.layers.len());

        for (layer, activation) in self.layers.iter().zip(activations) {
            layer_inputs.push(current_outputs.clone());

            // Calculate weighted sums only once.
            let sums: Vec<f32> = layer
                .neurons
                .iter()
                .map(|neuron| neuron.weighted_sum(&current_outputs))
                .collect();

            // Apply the activation using the calculated sums.
            current_outputs = match *activation {
                "relu" => sums.iter().map(|&x| x.max(0.0)).collect(),

                "sigmoid" => sums
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-x).exp()))
                    .collect(),

                "softmax" => crate::functions::activation::softmax(&sums),

                _ => panic!("Unknown activation function"),
            };

            weighted_sums.push(sums);
            layer_outputs.push(current_outputs.clone());
        }

        let cache = ForwardCache {
            layer_inputs,
            weighted_sums,
            layer_outputs,
        };

        (current_outputs, cache)
    }


    pub fn backward(
        &self,
        predicted: &[f32],
        actual: &[f32],
        cache: &ForwardCache,
        activations: &[&str],
    ) -> NetworkGradients {
        assert_eq!(self.layers.len(), activations.len());
        assert_eq!(self.layers.len(), cache.weighted_sums.len());
        assert_eq!(self.layers.len(), cache.layer_inputs.len());

        let last = self.layers.len() - 1;

        let mut all_gradients: Vec<Option<Vec<NeuronGradients>>> =
            (0..self.layers.len()).map(|_| None).collect();

        // Calculate output-layer gradients.
        let output_gradients = self.layers[last].calculate_output_gradients(
            &cache.layer_inputs[last],
            predicted,
            actual,
        );

        let mut next_deltas: Vec<f32> = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| p - a)
            .collect();

        all_gradients[last] = Some(output_gradients);

        // Calculate hidden-layer gradients, moving backwards.
        for i in (0..last).rev() {
            let gradients = self.layers[i].calculate_hidden_gradients(
                &cache.layer_inputs[i],
                &self.layers[i + 1],
                &next_deltas,
                &cache.weighted_sums[i],
                activations[i],
            );

            next_deltas = gradients.iter().map(|gradient| gradient.bias).collect();

            all_gradients[i] = Some(gradients);
        }

        NetworkGradients {
            layers: all_gradients
                .into_iter()
                .map(|gradients| gradients.unwrap())
                .collect(),
        }
    }

    pub fn update_parameters(&mut self, gradients: &NetworkGradients, learning_rate: f32) {
        assert_eq!(self.layers.len(), gradients.layers.len());

        for (layer, layer_gradients) in self.layers.iter_mut().zip(&gradients.layers) {
            layer.update_parameters(layer_gradients, learning_rate);
        }
    }

    pub fn train_step(
        &mut self,
        inputs: &[f32],
        actual: &[f32],
        activations: &[&str],
        learning_rate: f32,
    ) -> f32 {
        // 1. Forward pass
        let (predicted, cache) = self.forward_with_cache(inputs, activations);

        // 2. Calculate loss
        let loss = crate::functions::loss::cross_entropy_loss(actual, &predicted);

        // 3. Backpropagation
        let gradients = self.backward(&predicted, actual, &cache, activations);

        // 4. Update parameters
        self.update_parameters(&gradients, learning_rate);

        loss
    }
}


impl NetworkGradients {
    pub fn zeros_like(network: &Network) -> Self {
        let layers = network
            .layers
            .iter()
            .map(|layer| {
                layer
                    .neurons
                    .iter()
                    .map(|neuron| NeuronGradients {
                        weights: vec![0.0; neuron.weights.len()],
                        bias: 0.0,
                    })
                    .collect()
            })
            .collect();

        Self { layers }
    }

    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.layers.len(), other.layers.len());

        for (layer, other_layer) in self.layers.iter_mut().zip(&other.layers) {
            assert_eq!(layer.len(), other_layer.len());

            for (neuron, other_neuron) in layer.iter_mut().zip(other_layer) {
                assert_eq!(neuron.weights.len(), other_neuron.weights.len());

                for (weight, other_weight) in
                    neuron.weights.iter_mut().zip(&other_neuron.weights)
                {
                    *weight += other_weight;
                }

                neuron.bias += other_neuron.bias;
            }
        }
    }

    pub fn scale(&mut self, factor: f32) {
        for layer in &mut self.layers {
            for neuron in layer {
                for weight in &mut neuron.weights {
                    *weight *= factor;
                }

                neuron.bias *= factor;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::neuron::Neuron;

    #[test]
    fn test_network_creation() {
        let layer = Layer::new(vec![Neuron::new(vec![0.4, 0.3], 0.1)]);

        let network = Network::new(vec![layer]);

        assert_eq!(network.layers.len(), 1);
    }

    #[test]
    fn test_network_forward() {
        let layer1 = Layer::new(vec![
            Neuron::new(vec![0.4, 0.3], 0.1),
            Neuron::new(vec![0.2, 0.5], 0.2),
        ]);

        let layer2 = Layer::new(vec![Neuron::new(vec![0.5, 0.5], 0.1)]);

        let network = Network::new(vec![layer1, layer2]);

        let inputs = vec![0.2, 0.5];

        let outputs = network.forward(&inputs, "relu");

        assert_eq!(outputs.len(), 1);
        assert!((outputs[0] - 0.51).abs() < 0.0001);
    }

    #[test]
    fn test_network_from_random() {
        let network = Network::from_random(&[784, 128, 64, 10], "relu");

        assert_eq!(network.layers.len(), 3);

        assert_eq!(network.layers[0].neurons.len(), 128);
        assert_eq!(network.layers[1].neurons.len(), 64);
        assert_eq!(network.layers[2].neurons.len(), 10);
    }

    #[test]
    fn test_network_from_random_weights_dimensions() {
        let network = Network::from_random(&[4, 3, 2], "relu");

        for (i, layer) in network.layers.iter().enumerate() {
            assert_eq!(layer.neurons[0].weights.len(), if i == 0 { 4 } else { 3 });
        }
    }

    #[test]
    fn test_network_forward_with_different_activations() {
        let network = Network::from_random(&[4, 3, 2], "relu");

        let outputs = network.forward_with_activations(&[0.5, 0.2, 0.8, 0.1], &["relu", "softmax"]);

        assert_eq!(outputs.len(), 2);

        let sum: f32 = outputs.iter().sum();

        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_hidden_delta_relu() {
        let neuron = Neuron::new(vec![0.2, 0.3], 0.1);

        let next_weights = [0.4, 0.5];
        let next_deltas = [0.2, -0.1];

        let delta = neuron.calculate_hidden_delta(&next_weights, &next_deltas, "relu", 0.5);

        // (0.4 * 0.2 + 0.5 * -0.1) * 1.0 = 0.03
        assert!((delta - 0.03).abs() < 0.0001);
    }

    #[test]
    fn test_forward_with_cache() {
        let layers = vec![
            Layer::new(vec![
                Neuron::new(vec![0.5, 0.3], 0.1),
                Neuron::new(vec![0.2, 0.4], 0.2),
            ]),
            Layer::new(vec![Neuron::new(vec![0.5, 0.5], 0.1)]),
        ];

        let network = Network::new(layers);

        let inputs = [0.2, 0.5];

        let (outputs, cache) = network.forward_with_cache(&inputs, &["relu", "sigmoid"]);

        assert_eq!(outputs.len(), 1);
        assert_eq!(cache.layer_inputs.len(), 2);
        assert_eq!(cache.weighted_sums.len(), 2);
        assert_eq!(cache.layer_outputs.len(), 2);

        assert_eq!(cache.layer_inputs[0], vec![0.2, 0.5]);
        assert_eq!(cache.layer_outputs[0].len(), 2);
        assert_eq!(cache.weighted_sums[1].len(), 1);
    }

    #[test]
    fn test_update_parameters() {
        let layers = vec![Layer::new(vec![Neuron::new(vec![0.5, -0.2], 0.1)])];

        let mut network = Network::new(layers);

        let gradients = NetworkGradients {
            layers: vec![vec![NeuronGradients {
                weights: vec![0.2, -0.4],
                bias: 0.5,
            }]],
        };

        network.update_parameters(&gradients, 0.1);

        let neuron = &network.layers[0].neurons[0];

        assert!((neuron.weights[0] - 0.48).abs() < 1e-6);
        assert!((neuron.weights[1] - (-0.16)).abs() < 1e-6);
        assert!((neuron.bias - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut network = Network::new(vec![
            Layer::new(vec![
                Neuron::new(vec![0.2, -0.3], 0.1),
                Neuron::new(vec![-0.1, 0.4], 0.1),
            ]),
            Layer::new(vec![
                Neuron::new(vec![0.3, -0.2], 0.1),
                Neuron::new(vec![-0.3, 0.2], 0.1),
            ]),
        ]);

        let inputs = [0.5, 0.8];
        let actual = [1.0, 0.0];
        let activations = ["relu", "softmax"];
        let learning_rate = 0.1;

        let (_, cache) = network.forward_with_cache(&inputs, &activations);

        let initial_loss = crate::functions::loss::cross_entropy_loss(
            &actual,
            &network.forward_with_activations(&inputs, &activations),
        );

        for _ in 0..100 {
            network.train_step(&inputs, &actual, &activations, learning_rate);
        }

        let final_predictions = network.forward_with_activations(&inputs, &activations);

        let final_loss = crate::functions::loss::cross_entropy_loss(&actual, &final_predictions);

        assert!(final_loss < initial_loss);
    }
}

