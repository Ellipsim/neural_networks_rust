
use ndarray::Array2;

use crate::data::mnist::Mnist;
use crate::data::preprocessing::one_hot_encode;
use crate::functions::loss::cross_entropy_loss;
use crate::neural_network::batch::{
    backward_batch,
    forward_batch,
    update_batch_parameters,
};
use crate::neural_network::network::Network;

pub struct TrainingResult {
    pub average_loss: f32,
    pub correct_predictions: usize,
    pub total_images: usize,
    pub accuracy: f32,
}

pub fn train(
    network: &mut Network,
    dataset: &Mnist,
    activations: &[&str],
    learning_rate: f32,
    epochs: usize,
) -> Vec<TrainingResult> {
    train_with_batch_size(
        network,
        dataset,
        activations,
        learning_rate,
        epochs,
        1,
    )
}

pub fn train_with_batch_size(
    network: &mut Network,
    dataset: &Mnist,
    activations: &[&str],
    learning_rate: f32,
    epochs: usize,
    batch_size: usize,
) -> Vec<TrainingResult> {
    assert!(batch_size > 0, "Batch size must be greater than zero");
    assert_eq!(
        network.layers.len(),
        activations.len(),
        "One activation function is required per layer"
    );

    let mut history = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;

        for batch_start in (0..dataset.labels.len()).step_by(batch_size) {
            let batch_end =
                (batch_start + batch_size).min(dataset.labels.len());

            let current_batch_size = batch_end - batch_start;

            // Build the input and target matrices.
            let mut input_data = Vec::new();
            let mut target_data = Vec::with_capacity(current_batch_size * 10);

            for index in batch_start..batch_end {
                let inputs = dataset.get_prepared_image(index);
                let targets = one_hot_encode(dataset.labels[index]);

                input_data.extend_from_slice(&inputs);
                target_data.extend_from_slice(&targets);
            }

            let inputs = Array2::from_shape_vec(
                (current_batch_size, 784),
                input_data,
            )
            .expect("Invalid input batch dimensions");

            let targets = Array2::from_shape_vec(
                (current_batch_size, 10),
                target_data,
            )
            .expect("Invalid target batch dimensions");

            // Forward propagation for the entire batch.
            let (predictions, cache) =
                forward_batch(network, &inputs, activations);

            // Calculate loss and accuracy for each image.
            for row_index in 0..current_batch_size {
                let predicted = predictions.row(row_index).to_vec();
                let actual = targets.row(row_index).to_vec();

                total_loss += cross_entropy_loss(&actual, &predicted);

                let predicted_digit = predicted
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(digit, _)| digit)
                    .unwrap();

                if predicted_digit == dataset.labels[batch_start + row_index] as usize {
                    correct_predictions += 1;
                }
            }

            // Backpropagation for the entire batch.
            let gradients = backward_batch(
                network,
                &cache,
                &targets,
                activations,
            );

            // Update network parameters once per batch.
            update_batch_parameters(
                network,
                &gradients,
                learning_rate,
            );
        }

        let total_images = dataset.labels.len();

        let average_loss = if total_images > 0 {
            total_loss / total_images as f32
        } else {
            0.0
        };

        let accuracy = if total_images > 0 {
            correct_predictions as f32 / total_images as f32 * 100.0
        } else {
            0.0
        };

        let result = TrainingResult {
            average_loss,
            correct_predictions,
            total_images,
            accuracy,
        };

        println!(
            "Epoch {}/{} - Loss: {:.4} - Accuracy: {:.2}%",
            epoch + 1,
            epochs,
            average_loss,
            accuracy
        );

        history.push(result);
    }

    history
}
