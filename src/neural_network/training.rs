
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::data::mnist::Mnist;
use crate::neural_network::network::Network;
use crate::neural_network::optimizers::Optimizer;
use crate::neural_network::evaluation::evaluate;


const NUM_CLASSES: usize = 10;

#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub epoch: usize,
    pub average_loss: f32,
    pub correct_predictions: usize,
    pub total_images: usize,
    pub accuracy: f32,
    pub validation_loss: f32,
    pub validation_accuracy: f32,
}

pub fn train<O: Optimizer>(
    network: &mut Network,
    dataset: &Mnist,
    validation_dataset: &Mnist,
    activations: &[&str],
    optimizer: &mut O,
    epochs: usize,
    batch_size: usize,
    seed: u64,
) -> Vec<TrainingResult> {
    assert!(batch_size > 0, "Batch size must be greater than zero");
    assert!(!network.layers.is_empty(), "Network must have at least one layer");

    let image_size = dataset.rows * dataset.cols;
    let input_size = network.layers[0].input_size();
    let output_size = network.layers.last().unwrap().output_size();

    assert_eq!(
        image_size, input_size,
        "Dataset image size must match network input size"
    );

    assert_eq!(
        output_size, NUM_CLASSES,
        "This training implementation expects 10 output classes"
    );

    assert_eq!(
        dataset.images.len(),
        dataset.labels.len() * image_size,
        "Dataset image and label counts do not match"
    );

    let total_images = dataset.labels.len();
    let mut history = Vec::with_capacity(epochs);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..total_images).collect();

    for epoch in 0..epochs {
        indices.shuffle(&mut rng);

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;

        for batch_start in (0..total_images).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_images);
            let current_batch_size = batch_end - batch_start;

            let mut input_values =
                Vec::with_capacity(current_batch_size * input_size);

            let mut target_values =
                Vec::with_capacity(current_batch_size * NUM_CLASSES);

            for position in batch_start..batch_end {
                let dataset_index = indices[position];

                input_values.extend(dataset.get_prepared_image(dataset_index));

                let label = dataset.labels[dataset_index] as usize;
                assert!(label < NUM_CLASSES, "Invalid MNIST label");

                for class in 0..NUM_CLASSES {
                    target_values.push(if class == label { 1.0 } else { 0.0 });
                }
            }

            let inputs = Array2::from_shape_vec(
                (current_batch_size, input_size),
                input_values,
            )
            .expect("Input batch has an invalid shape");

            let targets = Array2::from_shape_vec(
                (current_batch_size, NUM_CLASSES),
                target_values,
            )
            .expect("Target batch has an invalid shape");

            let (predictions, cache) =
                network.forward(&inputs, activations);

            // Calculate predictions and loss.
            for row in 0..current_batch_size {
                let actual = dataset.labels[indices[batch_start + row]] as usize;

                let predicted = predictions
                    .row(row)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(index, _)| index)
                    .unwrap();

                if predicted == actual {
                    correct_predictions += 1;
                }

                let probability = predictions[[row, actual]].max(1e-7);
                total_loss -= probability.ln();
            }

            // Backpropagation.
            let gradients =
                network.backward(&predictions, &targets, &cache, activations);

            // Update parameters using the selected optimizer.
            optimizer.update(network, &gradients);
        }

        let average_loss = if total_images > 0 {
            total_loss / total_images as f32
        } else {
            0.0
        };

        let accuracy = if total_images > 0 {
            correct_predictions as f32 / total_images as f32
        } else {
            0.0
        };

        let validation_result = evaluate(
            network,
            validation_dataset,
            activations,
        );

        history.push(TrainingResult {
            epoch: epoch + 1,
            average_loss,
            correct_predictions,
            total_images,
            accuracy,
            validation_loss: validation_result.average_loss,
            validation_accuracy: validation_result.accuracy,
        });
    }

    history
}
