
use ndarray::Array2;

use crate::neural_network::gradients::NetworkGradients;
use crate::neural_network::network::Network;
use crate::neural_network::optimizers::Optimizer;

#[derive(Clone, Debug)]
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub timestep: usize,

    first_moments: Vec<Array2<f32>>,
    second_moments: Vec<Array2<f32>>,
    first_bias_moments: Vec<Array2<f32>>,
    second_bias_moments: Vec<Array2<f32>>,
}

impl Adam {
    pub fn new(network: &Network, learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,

            first_moments: network.layers.iter()
                .map(|layer| Array2::zeros(layer.weights.dim()))
                .collect(),

            second_moments: network.layers.iter()
                .map(|layer| Array2::zeros(layer.weights.dim()))
                .collect(),

            first_bias_moments: network.layers.iter()
                .map(|layer| Array2::zeros(layer.biases.dim()))
                .collect(),

            second_bias_moments: network.layers.iter()
                .map(|layer| Array2::zeros(layer.biases.dim()))
                .collect(),
        }
    }
}

impl Optimizer for Adam {
    fn update(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) {
        assert_eq!(network.layers.len(), gradients.layers.len());

        self.timestep += 1;

        let correction1 = 1.0 - self.beta1.powi(self.timestep as i32);
        let correction2 = 1.0 - self.beta2.powi(self.timestep as i32);

        for index in 0..network.layers.len() {
            let layer = &mut network.layers[index];
            let gradient = &gradients.layers[index];

            let weight_gradient = &gradient.weights;
            let bias_gradient = &gradient.biases;

            // First moment: exponential moving average of gradients.
            self.first_moments[index].mapv_inplace(|x| x * self.beta1);
            self.first_moments[index] +=
                &(weight_gradient * (1.0 - self.beta1));

            self.first_bias_moments[index].mapv_inplace(|x| x * self.beta1);
            self.first_bias_moments[index] +=
                &(bias_gradient * (1.0 - self.beta1));

            // Second moment: exponential moving average of squared gradients.
            self.second_moments[index].mapv_inplace(|x| x * self.beta2);
            self.second_moments[index] +=
                &(weight_gradient.mapv(|x| x * x) * (1.0 - self.beta2));
            self.second_bias_moments[index].mapv_inplace(|x| x * self.beta2);
            self.second_bias_moments[index] +=
                &(bias_gradient.mapv(|x| x * x) * (1.0 - self.beta2));

            // Bias correction.
            let corrected_first =
                &self.first_moments[index] / correction1;

            let corrected_second =
                &self.second_moments[index] / correction2;

            let corrected_first_bias =
                &self.first_bias_moments[index] / correction1;

            let corrected_second_bias =
                &self.second_bias_moments[index] / correction2;

            // Parameter updates.
            let weight_update = corrected_first /
                (corrected_second.mapv(f32::sqrt) + self.epsilon);

            let bias_update = corrected_first_bias /
                (corrected_second_bias.mapv(f32::sqrt) + self.epsilon);

            layer.weights -= &(weight_update * self.learning_rate);
            layer.biases -= &(bias_update * self.learning_rate);
        }
    }
}
