use crate::functions::activation::{relu, sigmoid, softmax};
use crate::neural_network::neuron::Neuron;
use crate::neural_network::neuron::NeuronGradients;
use crate::neural_network::weights::initialize_weights;

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    pub fn from_random(input_size: usize, neuron_count: usize, activation: &str) -> Self {
        let weights = initialize_weights(input_size, neuron_count, activation);

        let neurons = weights
            .into_iter()
            .map(|neuron_weights| Neuron::new(neuron_weights, 0.0))
            .collect();

        Self { neurons }
    }

    pub fn forward(&self, inputs: &[f32], activation: &str) -> Vec<f32> {
        let mut outputs = Vec::new();

        for neuron in &self.neurons {
            let output = neuron.activate(inputs, activation);
            outputs.push(output);
        }

        outputs
    }

    pub fn forward_with_activation(&self, inputs: &[f32], activation: &str) -> Vec<f32> {
        if activation == "softmax" {
            let outputs: Vec<f32> = self
                .neurons
                .iter()
                .map(|neuron| neuron.weighted_sum(inputs))
                .collect();

            return softmax(&outputs);
        }

        self.neurons
            .iter()
            .map(|neuron| {
                let sum = neuron.weighted_sum(inputs);

                match activation {
                    "relu" => relu(sum),
                    "sigmoid" => sigmoid(sum),
                    _ => panic!("Unknown activation function"),
                }
            })
            .collect()
    }

    pub fn calculate_output_gradients(
        &self,
        inputs: &[f32],
        predicted: &[f32],
        actual: &[f32],
    ) -> Vec<NeuronGradients> {
        assert_eq!(self.neurons.len(), predicted.len());
        assert_eq!(self.neurons.len(), actual.len());

        let mut gradients = Vec::new();

        for i in 0..self.neurons.len() {
            let neuron_gradients =
                self.neurons[i].calculate_output_gradients(inputs, predicted[i], actual[i]);

            gradients.push(neuron_gradients);
        }

        gradients
    }

    pub fn update_parameters(&mut self, gradients: &[NeuronGradients], learning_rate: f32) {
        assert_eq!(self.neurons.len(), gradients.len());

        for (neuron, gradient) in self.neurons.iter_mut().zip(gradients) {
            neuron.update_parameters(&gradient.weights, gradient.bias, learning_rate);
        }
    }

    pub fn calculate_hidden_gradients(
        &self,
        inputs: &[f32],
        next_layer: &Layer,
        next_layer_deltas: &[f32],
        weighted_sums: &[f32],
        activation: &str,
    ) -> Vec<NeuronGradients> {
        assert_eq!(self.neurons.len(), weighted_sums.len());
        assert_eq!(next_layer.neurons.len(), next_layer_deltas.len());

        let mut gradients = Vec::new();

        for i in 0..self.neurons.len() {
            // Check that every neuron in the next layer has a weight connecting it to this neuron.
            for neuron in &next_layer.neurons {
                assert!(neuron.weights.len() > i);
            }

            let next_weights: Vec<f32> = next_layer
                .neurons
                .iter()
                .map(|neuron| neuron.weights[i])
                .collect();

            let delta = self.neurons[i].calculate_hidden_delta(
                &next_weights,
                next_layer_deltas,
                activation,
                weighted_sums[i],
            );

            let neuron_gradients = self.neurons[i].calculate_hidden_gradients(inputs, delta);

            gradients.push(neuron_gradients);
        }

        gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let neurons = vec![
            Neuron::new(vec![0.4, 0.3], 0.1),
            Neuron::new(vec![0.2, 0.5], 0.2),
        ];

        let layer = Layer::new(neurons);

        assert_eq!(layer.neurons.len(), 2);
    }

    #[test]
    fn test_layer_forward() {
        let neurons = vec![
            Neuron::new(vec![0.4, 0.3], 0.1),
            Neuron::new(vec![0.2, 0.5], 0.2),
        ];

        let layer = Layer::new(neurons);
        let inputs = vec![0.2, 0.5];

        let outputs = layer.forward(&inputs, "relu");

        assert_eq!(outputs.len(), 2);
        assert!((outputs[0] - 0.33).abs() < 0.0001);
        assert!((outputs[1] - 0.49).abs() < 0.0001);
    }

    #[test]
    fn test_layer_from_random_dimensions() {
        let layer = Layer::from_random(3, 4, "relu");

        assert_eq!(layer.neurons.len(), 4);

        for neuron in &layer.neurons {
            assert_eq!(neuron.weights.len(), 3);
            assert_eq!(neuron.bias, 0.0);
        }
    }

    #[test]
    fn test_layer_from_random_sigmoid() {
        let layer = Layer::from_random(5, 2, "sigmoid");

        assert_eq!(layer.neurons.len(), 2);

        for neuron in &layer.neurons {
            assert_eq!(neuron.weights.len(), 5);
        }
    }

    #[test]
    fn test_layer_calculate_output_gradients() {
        let neurons = vec![
            Neuron::new(vec![0.2, 0.3], 0.1),
            Neuron::new(vec![0.4, 0.5], 0.2),
        ];

        let layer = Layer::new(neurons);

        let inputs = [0.5, 0.8];
        let predicted = [0.2, 0.8];
        let actual = [0.0, 1.0];

        let gradients = layer.calculate_output_gradients(&inputs, &predicted, &actual);

        assert_eq!(gradients.len(), 2);

        // First neuron: delta = 0.2
        assert!((gradients[0].weights[0] - 0.1).abs() < 0.0001);
        assert!((gradients[0].weights[1] - 0.16).abs() < 0.0001);
        assert!((gradients[0].bias - 0.2).abs() < 0.0001);

        // Second neuron: delta = -0.2
        assert!((gradients[1].weights[0] - (-0.1)).abs() < 0.0001);
        assert!((gradients[1].weights[1] - (-0.16)).abs() < 0.0001);
        assert!((gradients[1].bias - (-0.2)).abs() < 0.0001);
    }

    #[test]
    fn test_layer_update_parameters() {
        let neurons = vec![
            Neuron::new(vec![0.5, 0.3], 0.1),
            Neuron::new(vec![0.2, 0.4], 0.2),
        ];

        let mut layer = Layer::new(neurons);

        let gradients = vec![
            NeuronGradients {
                weights: vec![0.2, -0.4],
                bias: 0.1,
            },
            NeuronGradients {
                weights: vec![-0.1, 0.3],
                bias: -0.2,
            },
        ];

        layer.update_parameters(&gradients, 0.1);

        assert!((layer.neurons[0].weights[0] - 0.48).abs() < 0.0001);
        assert!((layer.neurons[0].weights[1] - 0.34).abs() < 0.0001);
        assert!((layer.neurons[0].bias - 0.09).abs() < 0.0001);

        assert!((layer.neurons[1].weights[0] - 0.21).abs() < 0.0001);
        assert!((layer.neurons[1].weights[1] - 0.37).abs() < 0.0001);
        assert!((layer.neurons[1].bias - 0.22).abs() < 0.0001);
    }

    #[test]
    fn test_layer_calculate_hidden_gradients() {
        let current_layer = Layer::new(vec![Neuron::new(vec![0.2, 0.3], 0.1)]);

        let next_layer = Layer::new(vec![
            Neuron::new(vec![0.4], 0.1),
            Neuron::new(vec![0.5], 0.2),
        ]);

        let inputs = [0.5, 0.8];
        let next_deltas = [0.2, -0.1];
        let weighted_sums = [0.5];

        let gradients = current_layer.calculate_hidden_gradients(
            &inputs,
            &next_layer,
            &next_deltas,
            &weighted_sums,
            "relu",
        );

        assert_eq!(gradients.len(), 1);

        // Delta = (0.4 * 0.2 + 0.5 * -0.1) * 1.0 = 0.03
        assert!((gradients[0].weights[0] - 0.015).abs() < 0.0001);
        assert!((gradients[0].weights[1] - 0.024).abs() < 0.0001);
        assert!((gradients[0].bias - 0.03).abs() < 0.0001);
    }
}
