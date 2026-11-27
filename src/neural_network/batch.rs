use ndarray::{Array2, Axis};

use crate::functions::optimizer::sgd_update;
use crate::neural_network::network::Network;

pub struct BatchForwardCache {
    pub layer_inputs: Vec<Array2<f32>>,
    pub weighted_sums: Vec<Array2<f32>>,
    pub layer_outputs: Vec<Array2<f32>>,
}

pub struct BatchLayerGradients {
    // Same shape as the layer's weight matrix:
    // (input_size, output_size)
    pub weights: Array2<f32>,

    // Shape: (1, output_size)
    pub biases: Array2<f32>,
}

pub struct BatchGradients {
    pub layers: Vec<BatchLayerGradients>,
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

        "sigmoid" => outputs.mapv(|x| x * (1.0 - x)),

        _ => panic!(
            "No element-wise derivative implemented for: {activation}"
        ),
    }
}

/// Applies an activation function to every element of a matrix.
fn apply_activation(values: &Array2<f32>, activation: &str) -> Array2<f32> {
    match activation {
        "relu" => values.mapv(|x| x.max(0.0)),

        "sigmoid" => {
            values.mapv(|x| 1.0 / (1.0 + (-x).exp()))
        }

        "softmax" => {
            let mut result = values.clone();

            // Apply softmax independently to each image.
            for mut row in result.axis_iter_mut(Axis(0)) {
                let max_value = row.iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut sum = 0.0;

                for value in row.iter_mut() {
                    *value = (*value - max_value).exp();
                    sum += *value;
                }

                for value in row.iter_mut() {
                    *value /= sum;
                }
            }

            result
        }

        _ => panic!("Unknown activation function: {activation}"),
    }
}

/// Performs forward propagation for an entire batch.
///
/// Input shape: (batch_size, input_size)
/// Output shape: (batch_size, output_size)
pub fn forward_batch(
    network: &Network,
    inputs: &Array2<f32>,
    activations: &[&str],
) -> (Array2<f32>, BatchForwardCache) {
    assert_eq!(
        network.layers.len(),
        activations.len(),
        "One activation function is required per layer"
    );

    let mut current_outputs = inputs.clone();

    let mut layer_inputs = Vec::with_capacity(network.layers.len());
    let mut weighted_sums = Vec::with_capacity(network.layers.len());
    let mut layer_outputs = Vec::with_capacity(network.layers.len());

    for (layer, activation) in network.layers.iter().zip(activations) {
        let input_size = layer.neurons[0].weights.len();
        let output_size = layer.neurons.len();

        assert_eq!(
            current_outputs.ncols(),
            input_size,
            "Input size does not match layer weights"
        );

        layer_inputs.push(current_outputs.clone());

        // Convert the existing neuron weights into a matrix.
        //
        // Matrix shape: (input_size, output_size)
        let weights = Array2::from_shape_fn(
            (input_size, output_size),
            |(i, j)| layer.neurons[j].weights[i],
        );

        // Bias shape: (1, output_size)
        let biases = Array2::from_shape_fn(
            (1, output_size),
            |(_, j)| layer.neurons[j].bias,
        );

        // Matrix multiplication:
        // (batch_size, input_size) dot (input_size, output_size)
        // = (batch_size, output_size)
        let sums = current_outputs.dot(&weights) + &biases;

        current_outputs = apply_activation(&sums, activation);

        weighted_sums.push(sums);
        layer_outputs.push(current_outputs.clone());
    }

    let cache = BatchForwardCache {
        layer_inputs,
        weighted_sums,
        layer_outputs,
    };

    (current_outputs, cache)
}


pub fn backward_batch(
    network: &Network,
    cache: &BatchForwardCache,
    targets: &Array2<f32>,
    activations: &[&str],
) -> BatchGradients {
    let layer_count = network.layers.len();
    let batch_size = targets.nrows();

    assert!(batch_size > 0);
    assert_eq!(layer_count, activations.len());

    let predictions = cache.layer_outputs.last().unwrap();

    assert_eq!(predictions.dim(), targets.dim());

    let mut gradients: Vec<Option<BatchLayerGradients>> =
        (0..layer_count).map(|_| None).collect();

    // Output layer gradient: softmax + cross-entropy.
    let mut delta = predictions - targets;

    for layer_index in (0..layer_count).rev() {
        let layer = &network.layers[layer_index];

        let inputs = &cache.layer_inputs[layer_index];

        // dW = X^T * dZ / batch_size
        let weight_gradients =
            inputs.t().dot(&delta) / batch_size as f32;

        // db = sum(dZ, axis=0) / batch_size
        let bias_gradients = delta.sum_axis(Axis(0))
            .insert_axis(Axis(0))
            / batch_size as f32;

        gradients[layer_index] = Some(BatchLayerGradients {
            weights: weight_gradients,
            biases: bias_gradients,
        });

        // Calculate the previous layer's delta.
        if layer_index > 0 {
            let input_size = layer.neurons[0].weights.len();
            let output_size = layer.neurons.len();

            let weights = Array2::from_shape_fn(
                (input_size, output_size),
                |(i, j)| layer.neurons[j].weights[i],
            );

            let previous_delta = delta.dot(&weights.t());

            let previous_activation =
                activations[layer_index - 1];

            let derivative = activation_derivative(
                &cache.weighted_sums[layer_index - 1],
                &cache.layer_outputs[layer_index - 1],
                previous_activation,
            );

            delta = previous_delta * derivative;
        }
    }

    BatchGradients {
        layers: gradients
            .into_iter()
            .map(|gradient| gradient.unwrap())
            .collect(),
    }
}


pub fn update_batch_parameters(
    network: &mut Network,
    gradients: &BatchGradients,
    learning_rate: f32,
) {
    assert_eq!(network.layers.len(), gradients.layers.len());

    for (layer, layer_gradients) in
        network.layers.iter_mut().zip(&gradients.layers)
    {
        assert_eq!(
            layer.neurons.len(),
            layer_gradients.biases.ncols()
        );

        for (neuron_index, neuron) in
            layer.neurons.iter_mut().enumerate()
        {
            assert_eq!(
                neuron.weights.len(),
                layer_gradients.weights.nrows()
            );

            for (weight_index, weight) in
                neuron.weights.iter_mut().enumerate()
            {
                let gradient =
                    layer_gradients.weights[[weight_index, neuron_index]];

                *weight = sgd_update(
                    *weight,
                    gradient,
                    learning_rate,
                );
            }

            let bias_gradient =
                layer_gradients.biases[[0, neuron_index]];

            neuron.bias = sgd_update(
                neuron.bias,
                bias_gradient,
                learning_rate,
            );
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    use crate::neural_network::network::Network;

    #[test]
    fn batch_forward_matches_individual_forward() {
        let architecture = [4, 3, 2];
        let activations = ["relu", "softmax"];

        let network = Network::from_random(&architecture, "relu");

        let inputs = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
            ],
        )
        .unwrap();

        let (batch_outputs, _) =
            forward_batch(&network, &inputs, &activations);

        for (i, row) in inputs.axis_iter(Axis(0)).enumerate() {
            let individual_output =
                network.forward_with_activations(
                    &row.to_vec(),
                    &activations,
                );

            for j in 0..individual_output.len() {
                assert!(
                    (batch_outputs[[i, j]] - individual_output[j]).abs() < 1e-5,
                    "Mismatch at image {i}, output {j}"
                );
            }
        }
    }

    
    #[test]
    fn duplicated_samples_produce_same_averaged_gradients() {
        use crate::neural_network::layer::Layer;
        use crate::neural_network::neuron::Neuron;

        // Deterministic network: 2 inputs -> 2 outputs.
        let network = Network {
            layers: vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            weights: vec![0.2, -0.3],
                            bias: 0.1,
                        },
                        Neuron {
                            weights: vec![-0.4, 0.5],
                            bias: -0.2,
                        },
                    ],
                },
            ],
        };

        let activations = ["softmax"];

        let single_input = Array2::from_shape_vec(
            (1, 2),
            vec![0.6, 0.8],
        )
        .unwrap();

        let single_target = Array2::from_shape_vec(
            (1, 2),
            vec![1.0, 0.0],
        )
        .unwrap();

        let (single_output, single_cache) =
            forward_batch(&network, &single_input, &activations);

        let single_gradients = backward_batch(
            &network,
            &single_cache,
            &single_target,
            &activations,
        );

        // Repeat exactly the same sample.
        let duplicated_input = Array2::from_shape_vec(
            (2, 2),
            vec![
                0.6, 0.8,
                0.6, 0.8,
            ],
        )
        .unwrap();

        let duplicated_targets = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0, 0.0,
                1.0, 0.0,
            ],
        )
        .unwrap();

        let (duplicated_output, duplicated_cache) =
            forward_batch(&network, &duplicated_input, &activations);

        let duplicated_gradients = backward_batch(
            &network,
            &duplicated_cache,
            &duplicated_targets,
            &activations,
        );

        // Both samples should produce identical predictions.
        for j in 0..2 {
            assert!(
                (single_output[[0, j]] - duplicated_output[[0, j]]).abs() < 1e-6
            );
        }

        // Averaged gradients should be identical.
        for j in 0..2 {
            assert!(
                (single_gradients.layers[0].biases[[0, j]]
                    - duplicated_gradients.layers[0].biases[[0, j]])
                    .abs() < 1e-6
            );

            for i in 0..2 {
                assert!(
                    (single_gradients.layers[0].weights[[i, j]]
                        - duplicated_gradients.layers[0].weights[[i, j]])
                        .abs() < 1e-6
                );
            }
        }
    }

    #[test]
    fn batch_update_changes_weights_and_biases() {
        use crate::neural_network::layer::Layer;
        use crate::neural_network::neuron::Neuron;

        let mut network = Network {
            layers: vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            weights: vec![0.2, -0.3],
                            bias: 0.1,
                        },
                        Neuron {
                            weights: vec![-0.4, 0.5],
                            bias: -0.2,
                        },
                    ],
                },
            ],
        };

        let gradients = BatchGradients {
            layers: vec![
                BatchLayerGradients {
                    weights: Array2::from_shape_vec(
                        (2, 2),
                        vec![0.1, 0.2, 0.3, 0.4],
                    )
                    .unwrap(),

                    biases: Array2::from_shape_vec(
                        (1, 2),
                        vec![0.5, 0.6],
                    )
                    .unwrap(),
                },
            ],
        };

        update_batch_parameters(&mut network, &gradients, 0.1);

        assert!((network.layers[0].neurons[0].weights[0] - 0.19).abs() < 1e-6);
        assert!((network.layers[0].neurons[0].weights[1] - (-0.33)).abs() < 1e-6);
        assert!((network.layers[0].neurons[0].bias - 0.05).abs() < 1e-6);

        assert!((network.layers[0].neurons[1].weights[0] - (-0.42)).abs() < 1e-6);
        assert!((network.layers[0].neurons[1].weights[1] - 0.46).abs() < 1e-6);
        assert!((network.layers[0].neurons[1].bias - (-0.26)).abs() < 1e-6);
    }

    #[test]
    fn batch_gradients_match_original_backpropagation() {
        use crate::neural_network::layer::Layer;
        use crate::neural_network::neuron::Neuron;

        let network = Network {
            layers: vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            weights: vec![0.2, -0.3],
                            bias: 0.1,
                        },
                        Neuron {
                            weights: vec![-0.4, 0.5],
                            bias: -0.2,
                        },
                        Neuron {
                            weights: vec![0.3, 0.2],
                            bias: 0.05,
                        },
                    ],
                },
                Layer {
                    neurons: vec![
                        Neuron {
                            weights: vec![0.2, -0.1, 0.4],
                            bias: 0.1,
                        },
                        Neuron {
                            weights: vec![-0.3, 0.5, 0.2],
                            bias: -0.1,
                        },
                    ],
                },
            ],
        };

        let activations = ["relu", "softmax"];

        let input = vec![0.6, 0.8];
        let target = vec![1.0, 0.0];

        // Original implementation.
        let (predicted, original_cache) =
            network.forward_with_cache(&input, &activations);

        let original_gradients = network.backward(
            &predicted,
            &target,
            &original_cache,
            &activations,
        );

        // Vectorized implementation with batch size 1.
        let batch_input = Array2::from_shape_vec(
            (1, 2),
            input,
        ).unwrap();

        let batch_target = Array2::from_shape_vec(
            (1, 2),
            target,
        ).unwrap();

        let (_, batch_cache) =
            forward_batch(&network, &batch_input, &activations);

        let batch_gradients = backward_batch(
            &network,
            &batch_cache,
            &batch_target,
            &activations,
        );

        // Compare every weight and bias gradient.
        for layer_index in 0..network.layers.len() {
            for neuron_index in 0..network.layers[layer_index].neurons.len() {
                let original_neuron =
                    &original_gradients.layers[layer_index][neuron_index];

                let batch_neuron =
                    &batch_gradients.layers[layer_index];

                for weight_index in 0..original_neuron.weights.len() {
                    let difference = (
                        original_neuron.weights[weight_index]
                        - batch_neuron.weights[
                            [weight_index, neuron_index]
                        ]
                    ).abs();

                    assert!(
                        difference < 1e-5,
                        "Weight gradient mismatch: layer {layer_index}, neuron {neuron_index}"
                    );
                }

                let bias_difference = (
                    original_neuron.bias
                    - batch_neuron.biases[[0, neuron_index]]
                ).abs();

                assert!(
                    bias_difference < 1e-5,
                    "Bias gradient mismatch: layer {layer_index}, neuron {neuron_index}"
                );
            }
        }
    }
}
