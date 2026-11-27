use rand::RngExt;

pub fn initialize_weights(
    input_size: usize,
    neuron_count: usize,
    activation: &str,
) -> Vec<Vec<f32>> {
    assert!(input_size > 0);
    assert!(neuron_count > 0);

    let limit = match activation {
        "relu" => (6.0 / input_size as f32).sqrt(),
        "sigmoid" => (6.0 / (input_size + neuron_count) as f32).sqrt(),
        _ => panic!("Unknown activation function"),
    };

    let mut rng = rand::rng();

    let mut weights = Vec::new();

    for _ in 0..neuron_count {
        let mut neuron_weights = Vec::new();

        for _ in 0..input_size {
            let weight = rng.random_range(-limit..limit);
            neuron_weights.push(weight);
        }

        weights.push(neuron_weights);
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_weights_dimensions() {
        let weights = initialize_weights(3, 2, "relu");

        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].len(), 3);
        assert_eq!(weights[1].len(), 3);
    }

    #[test]
    fn test_initialize_weights_relu_range() {
        let weights = initialize_weights(3, 2, "relu");
        let limit = (6.0_f32 / 3.0).sqrt();

        for neuron_weights in &weights {
            for weight in neuron_weights {
                assert!(*weight >= -limit);
                assert!(*weight < limit);
            }
        }
    }

    #[test]
    fn test_initialize_weights_sigmoid_range() {
        let weights = initialize_weights(3, 2, "sigmoid");
        let limit = (6.0_f32 / 5.0).sqrt();

        for neuron_weights in &weights {
            for weight in neuron_weights {
                assert!(*weight >= -limit);
                assert!(*weight < limit);
            }
        }
    }
}
