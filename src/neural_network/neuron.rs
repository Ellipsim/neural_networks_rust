use crate::functions::activation::{relu, relu_derivative, sigmoid, sigmoid_derivative};

pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
}

pub struct NeuronGradients {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    pub fn weighted_sum(&self, inputs: &[f32]) -> f32 {
        assert_eq!(self.weights.len(), inputs.len());

        let mut sum = self.bias;

        for i in 0..inputs.len() {
            sum += inputs[i] * self.weights[i];
        }

        sum
    }

    pub fn activate(&self, inputs: &[f32], activation: &str) -> f32 {
        let sum = self.weighted_sum(inputs);

        match activation {
            "relu" => relu(sum),
            "sigmoid" => sigmoid(sum),
            _ => panic!("Unknown activation function"),
        }
    }

    pub fn calculate_output_gradients(
        &self,
        inputs: &[f32],
        predicted: f32,
        actual: f32,
    ) -> NeuronGradients {
        assert_eq!(self.weights.len(), inputs.len());

        let delta = predicted - actual;

        let weight_gradients = inputs.iter().map(|input| delta * input).collect();

        NeuronGradients {
            weights: weight_gradients,
            bias: delta,
        }
    }

    pub fn update_parameters(
        &mut self,
        weight_gradients: &[f32],
        bias_gradient: f32,
        learning_rate: f32,
    ) {
        assert_eq!(self.weights.len(), weight_gradients.len());

        for i in 0..self.weights.len() {
            self.weights[i] -= learning_rate * weight_gradients[i];
        }

        self.bias -= learning_rate * bias_gradient;
    }

    pub fn calculate_hidden_gradients(&self, inputs: &[f32], delta: f32) -> NeuronGradients {
        assert_eq!(self.weights.len(), inputs.len());

        let weight_gradients = inputs.iter().map(|input| delta * input).collect();

        NeuronGradients {
            weights: weight_gradients,
            bias: delta,
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new(vec![0.4, 0.3, 0.6], 0.1);

        assert_eq!(neuron.weights, vec![0.4, 0.3, 0.6]);
        assert_eq!(neuron.bias, 0.1);
    }

    #[test]
    fn test_weighted_sum() {
        let neuron = Neuron::new(vec![0.4, 0.3, 0.6], 0.1);
        let inputs = vec![0.2, 0.5, 0.8];

        let result = neuron.weighted_sum(&inputs);

        // (0.2 * 0.4) + (0.5 * 0.3) + (0.8 * 0.6) + 0.1
        assert!((result - 0.81).abs() < 0.0001);
    }

    #[test]
    fn test_weighted_sum_zero_inputs() {
        let neuron = Neuron::new(vec![0.4, 0.3], 0.1);
        let inputs = vec![0.0, 0.0];

        let result = neuron.weighted_sum(&inputs);

        assert_eq!(result, 0.1);
    }

    #[test]
    fn test_neuron_relu_activation() {
        let neuron = Neuron::new(vec![0.4, 0.3], 0.1);
        let inputs = vec![0.2, 0.5];

        let result = neuron.activate(&inputs, "relu");

        // (0.2 * 0.4) + (0.5 * 0.3) + 0.1 = 0.33
        assert!((result - 0.33).abs() < 0.0001);
    }

    #[test]
    fn test_neuron_sigmoid_activation() {
        let neuron = Neuron::new(vec![0.4, 0.3], 0.1);
        let inputs = vec![0.2, 0.5];

        let result = neuron.activate(&inputs, "sigmoid");

        assert!((result - 0.58176).abs() < 0.0001);
    }

    #[test]
    fn test_neuron_relu_negative_sum() {
        let neuron = Neuron::new(vec![-1.0, -1.0], -0.1);
        let inputs = vec![1.0, 1.0];

        let result = neuron.activate(&inputs, "relu");

        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_calculate_output_gradients() {
        let neuron = Neuron::new(vec![0.2, 0.3], 0.1);

        let inputs = [0.5, 0.8];

        let gradients = neuron.calculate_output_gradients(&inputs, 0.8, 1.0);

        assert!((gradients.weights[0] - (-0.1)).abs() < 0.0001);
        assert!((gradients.weights[1] - (-0.16)).abs() < 0.0001);
        assert!((gradients.bias - (-0.2)).abs() < 0.0001);
    }

    #[test]
    fn test_update_parameters() {
        let mut neuron = Neuron::new(vec![0.5, 0.3], 0.1);

        neuron.update_parameters(&[0.2, -0.4], 0.1, 0.1);

        assert!((neuron.weights[0] - 0.48).abs() < 0.0001);
        assert!((neuron.weights[1] - 0.34).abs() < 0.0001);
        assert!((neuron.bias - 0.09).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_hidden_gradients() {
        let neuron = Neuron::new(vec![0.2, 0.3], 0.1);

        let inputs = [0.5, 0.8];
        let delta = 0.2;

        let gradients = neuron.calculate_hidden_gradients(&inputs, delta);

        assert!((gradients.weights[0] - 0.1).abs() < 0.0001);
        assert!((gradients.weights[1] - 0.16).abs() < 0.0001);
        assert!((gradients.bias - 0.2).abs() < 0.0001);
    }
}
