use crate::data::mnist::Mnist;
use crate::data::preprocessing::one_hot_encode;
use crate::functions::loss::cross_entropy_loss;
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
    let mut history = Vec::new();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;

        for index in 0..dataset.labels.len() {
            let inputs = dataset.get_prepared_image(index);
            let actual = one_hot_encode(dataset.labels[index]);

            let (predicted, cache) = network.forward_with_cache(&inputs, activations);

            total_loss += cross_entropy_loss(&actual, &predicted);

            let predicted_digit = predicted
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(digit, _)| digit)
                .unwrap();

            if predicted_digit == dataset.labels[index] as usize {
                correct_predictions += 1;
            }

            let gradients = network.backward(&predicted, &actual, &cache, activations);

            network.update_parameters(&gradients, learning_rate);
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
