
use ndarray::Array2;

use crate::data::mnist::Mnist;
use crate::neural_network::network::Network;

const NUM_CLASSES: usize = 10;
const EVALUATION_BATCH_SIZE: usize = 256;

#[derive(Clone, Debug)]
pub struct ConfusionMatrix {
    pub counts: Vec<Vec<usize>>,
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            counts: vec![vec![0; NUM_CLASSES]; NUM_CLASSES],
        }
    }

    pub fn update(&mut self, actual: usize, predicted: usize) {
        assert!(actual < NUM_CLASSES);
        assert!(predicted < NUM_CLASSES);

        self.counts[actual][predicted] += 1;
    }

    pub fn get(&self, actual: usize, predicted: usize) -> usize {
        self.counts[actual][predicted]
    }
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct EvaluationResult {
    pub average_loss: f32,
    pub correct_predictions: usize,
    pub total_images: usize,
    pub accuracy: f32,
    pub confusion_matrix: ConfusionMatrix,
}

pub fn evaluate(
    network: &Network,
    dataset: &Mnist,
    activations: &[&str],
) -> EvaluationResult {
    let image_size = dataset.rows * dataset.cols;
    let input_size = network.layers[0].input_size();
    let output_size = network.layers.last().unwrap().output_size();

    assert_eq!(image_size, input_size);
    assert_eq!(output_size, NUM_CLASSES);
    assert_eq!(
        dataset.images.len(),
        dataset.labels.len() * image_size,
        "Dataset image and label counts do not match"
    );

    let total_images = dataset.labels.len();
    let mut total_loss = 0.0;
    let mut correct_predictions = 0;
    let mut confusion_matrix = ConfusionMatrix::new();

    for batch_start in (0..total_images).step_by(EVALUATION_BATCH_SIZE) {
        let batch_end =
            (batch_start + EVALUATION_BATCH_SIZE).min(total_images);
        let current_batch_size = batch_end - batch_start;

        let mut input_values =
            Vec::with_capacity(current_batch_size * input_size);

        for index in batch_start..batch_end {
            input_values.extend(dataset.get_prepared_image(index));
        }

        let inputs = Array2::from_shape_vec(
            (current_batch_size, input_size),
            input_values,
        )
        .expect("Evaluation batch has an invalid shape");

        let (predictions, _) = network.forward(&inputs, activations);

        for row in 0..current_batch_size {
            let actual = dataset.labels[batch_start + row] as usize;
            assert!(actual < NUM_CLASSES, "Invalid MNIST label");

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

            confusion_matrix.update(actual, predicted);

            let probability = predictions[[row, actual]].max(1e-7);
            total_loss -= probability.ln();
        }
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

    EvaluationResult {
        average_loss,
        correct_predictions,
        total_images,
        accuracy,
        confusion_matrix,
    }
}
