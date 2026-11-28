
use ndarray::{Array2, Axis};
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;

use crate::neural_network::gradients::{
    LayerGradients,
    NetworkGradients,
};
use crate::neural_network::layer::Layer;

#[derive(Clone, Debug)]
pub struct ForwardCache {
    pub layer_inputs: Vec<Array2<f32>>,
    pub weighted_sums: Vec<Array2<f32>>,
    pub layer_outputs: Vec<Array2<f32>>,
}

#[derive(Clone, Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        assert!(!layers.is_empty());

        for index in 1..layers.len() {
            assert_eq!(
                layers[index - 1].output_size(),
                layers[index].input_size()
            );
        }

        Self { layers }
    }

    pub fn from_random(
        architecture: &[usize],
        activation: &str,
        seed: Option<u64>,
    ) -> Self {
        assert!(architecture.len() >= 2);
        assert!(architecture.iter().all(|&size| size > 0));

        let mut rng = match seed {
            Some(seed) => {
                // A seeded RNG is used for reproducibility.
                StdRng::seed_from_u64(seed)
            }
            None => {
                // A randomly initialized RNG is used otherwise.
                StdRng::from_rng(&mut rand::rng())
            }
        };

        let mut layers = Vec::new();

        for dimensions in architecture.windows(2) {
            let input_size = dimensions[0];
            let output_size = dimensions[1];

            let limit = match activation {
                "relu" => (6.0 / input_size as f32).sqrt(),
                "sigmoid" => {
                    (6.0 / (input_size + output_size) as f32).sqrt()
                }
                _ => panic!("Unknown initialization activation"),
            };

            let weights = Array2::from_shape_fn(
                (input_size, output_size),
                |_| rng.random_range(-limit..limit),
            );

            let biases = Array2::zeros((1, output_size));

            layers.push(Layer::new(weights, biases));
        }

        Self::new(layers)
    }

    fn apply_activation(
        values: &Array2<f32>,
        activation: &str,
    ) -> Array2<f32> {
        match activation {
            "relu" => values.mapv(|x| x.max(0.0)),

            "sigmoid" => {
                values.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            }

            "softmax" => {
                let mut result = values.clone();

                for mut row in result.axis_iter_mut(Axis(0)) {
                    let max = row.iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);

                    let mut sum = 0.0;

                    for value in row.iter_mut() {
                        *value = (*value - max).exp();
                        sum += *value;
                    }

                    for value in row.iter_mut() {
                        *value /= sum;
                    }
                }

                result
            }

            _ => panic!("Unknown activation: {activation}"),
        }
    }

    fn activation_derivative(
        weighted_sums: &Array2<f32>,
        outputs: &Array2<f32>,
        activation: &str,
    ) -> Array2<f32> {
        match activation {
            "relu" => weighted_sums.mapv(|x| {
                if x > 0.0 { 1.0 } else { 0.0 }
            }),

            "sigmoid" => {
                outputs.mapv(|x| x * (1.0 - x))
            }

            _ => panic!(
                "No element-wise derivative for {activation}"
            ),
        }
    }

    pub fn forward(
        &self,
        inputs: &Array2<f32>,
        activations: &[&str],
    ) -> (Array2<f32>, ForwardCache) {
        assert_eq!(self.layers.len(), activations.len());
        assert_eq!(inputs.ncols(), self.layers[0].input_size());

        let mut current = inputs.clone();

        let mut layer_inputs = Vec::new();
        let mut weighted_sums = Vec::new();
        let mut layer_outputs = Vec::new();

        for (layer, activation) in self.layers.iter().zip(activations) {
            layer_inputs.push(current.clone());

            let sums = current.dot(&layer.weights) + &layer.biases;

            current = Self::apply_activation(&sums, activation);

            weighted_sums.push(sums);
            layer_outputs.push(current.clone());
        }

        let cache = ForwardCache {
            layer_inputs,
            weighted_sums,
            layer_outputs,
        };

        (current, cache)
    }

    pub fn backward(
        &self,
        predictions: &Array2<f32>,
        targets: &Array2<f32>,
        cache: &ForwardCache,
        activations: &[&str],
    ) -> NetworkGradients {
        assert_eq!(predictions.dim(), targets.dim());
        assert_eq!(self.layers.len(), activations.len());

        let batch_size = predictions.nrows();
        assert!(batch_size > 0);

        // Softmax + cross-entropy derivative.
        let mut delta = predictions - targets;

        let mut gradients: Vec<Option<LayerGradients>> =
            (0..self.layers.len()).map(|_| None).collect();

        for index in (0..self.layers.len()).rev() {
            let inputs = &cache.layer_inputs[index];

            let weight_gradients =
                inputs.t().dot(&delta) / batch_size as f32;

            let bias_gradients = delta.sum_axis(Axis(0))
                .insert_axis(Axis(0))
                / batch_size as f32;

            gradients[index] = Some(LayerGradients {
                weights: weight_gradients,
                biases: bias_gradients,
            });

            if index > 0 {
                let previous_delta =
                    delta.dot(&self.layers[index].weights.t());

                let derivative = Self::activation_derivative(
                    &cache.weighted_sums[index - 1],
                    &cache.layer_outputs[index - 1],
                    activations[index - 1],
                );

                delta = previous_delta * derivative;
            }
        }

        NetworkGradients {
            layers: gradients.into_iter()
                .map(|gradient| gradient.unwrap())
                .collect(),
        }
    }

    pub fn update_parameters(
        &mut self,
        gradients: &NetworkGradients,
        learning_rate: f32,
    ) {
        assert_eq!(self.layers.len(), gradients.layers.len());

        for (layer, gradient) in
            self.layers.iter_mut().zip(&gradients.layers)
        {
            assert_eq!(layer.weights.dim(), gradient.weights.dim());
            assert_eq!(layer.biases.dim(), gradient.biases.dim());

            layer.weights.zip_mut_with(
                &gradient.weights,
                |weight, grad| {
                    *weight -= learning_rate * grad;
                },
            );

            layer.biases.zip_mut_with(
                &gradient.biases,
                |bias, grad| {
                    *bias -= learning_rate * grad;
                },
            );
        }
    }
}
