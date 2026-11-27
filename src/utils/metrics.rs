pub fn precision(matrix: &ConfusionMatrix, digit: usize) -> f32 {
    assert!(digit < 10);

    let true_positives = matrix.values[digit][digit];

    let predicted_positives: usize = (0..10).map(|actual| matrix.values[actual][digit]).sum();

    if predicted_positives == 0 {
        return 0.0;
    }

    true_positives as f32 / predicted_positives as f32
}

pub fn recall(matrix: &ConfusionMatrix, digit: usize) -> f32 {
    assert!(digit < 10);

    let true_positives = matrix.values[digit][digit];
    let actual_positives: usize = matrix.values[digit].iter().sum();

    if actual_positives == 0 {
        return 0.0;
    }

    true_positives as f32 / actual_positives as f32
}

pub fn f1_score(matrix: &ConfusionMatrix, digit: usize) -> f32 {
    let p = precision(matrix, digit);
    let r = recall(matrix, digit);

    if p + r == 0.0 {
        return 0.0;
    }

    2.0 * p * r / (p + r)
}

pub struct ConfusionMatrix {
    pub values: [[usize; 10]; 10],
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            values: [[0; 10]; 10],
        }
    }

    pub fn update(&mut self, actual: usize, predicted: usize) {
        assert!(actual < 10);
        assert!(predicted < 10);

        self.values[actual][predicted] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_update() {
        let mut matrix = ConfusionMatrix::new();

        matrix.update(3, 3);
        matrix.update(3, 5);
        matrix.update(5, 3);

        assert_eq!(matrix.values[3][3], 1);
        assert_eq!(matrix.values[3][5], 1);
        assert_eq!(matrix.values[5][3], 1);
    }

    #[test]
    fn test_precision() {
        let mut matrix = ConfusionMatrix::new();

        matrix.update(3, 3);
        matrix.update(3, 3);
        matrix.update(5, 3);
        matrix.update(7, 3);

        assert!((precision(&matrix, 3) - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_recall() {
        let mut matrix = ConfusionMatrix::new();

        matrix.update(3, 3);
        matrix.update(3, 3);
        matrix.update(3, 5);
        matrix.update(5, 3);

        assert!((recall(&matrix, 3) - (2.0 / 3.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f1_score() {
        let mut matrix = ConfusionMatrix::new();

        matrix.update(3, 3);
        matrix.update(3, 3);
        matrix.update(3, 5);
        matrix.update(5, 3);
        matrix.update(7, 3);

        let expected = 2.0 * 0.5 * (2.0 / 3.0) / (0.5 + (2.0 / 3.0));

        assert!((f1_score(&matrix, 3) - expected).abs() < 0.0001);
    }

    #[test]
    fn test_zero_predictions() {
        let matrix = ConfusionMatrix::new();

        assert_eq!(precision(&matrix, 3), 0.0);
        assert_eq!(recall(&matrix, 3), 0.0);
        assert_eq!(f1_score(&matrix, 3), 0.0);
    }
}
