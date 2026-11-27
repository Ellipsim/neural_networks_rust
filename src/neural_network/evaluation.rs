use crate::data::mnist::Mnist;
use crate::data::preprocessing::one_hot_encode;
use crate::functions::loss::cross_entropy_loss;
use crate::neural_network::network::Network;
use crate::utils::metrics::ConfusionMatrix;

pub struct EvaluationResult {
    pub average_loss: f32,
    pub correct_predictions: usize,
    pub total_images: usize,
    pub accuracy: f32,
    pub confusion_matrix: ConfusionMatrix,
}

pub fn evaluate(network: &Network, dataset: &Mnist, activations: &[&str]) -> EvaluationResult {
    let mut total_loss = 0.0;
    let mut correct_predictions = 0;
    let mut confusion_matrix = ConfusionMatrix::new();

    for index in 0..dataset.labels.len() {
        let inputs = dataset.get_prepared_image(index);
        let actual = one_hot_encode(dataset.labels[index]);

        let predicted = network.forward_with_activations(&inputs, activations);

        total_loss += cross_entropy_loss(&actual, &predicted);

        let predicted_digit = predicted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(digit, _)| digit)
            .unwrap();

        let actual_digit = dataset.labels[index] as usize;

        confusion_matrix.update(actual_digit, predicted_digit);

        if predicted_digit == actual_digit {
            correct_predictions += 1;
        }
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

    EvaluationResult {
        average_loss,
        correct_predictions,
        total_images,
        accuracy,
        confusion_matrix,
    }
}
