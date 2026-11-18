pub fn mean_squared_error(actual: &[f32], predicted: &[f32]) -> f32 {
    assert_eq!(actual.len(), predicted.len());

    if actual.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;

    for i in 0..actual.len() {
        let error = actual[i] - predicted[i];
        sum += error * error;
    }

    sum / actual.len() as f32
}

pub fn cross_entropy_loss(actual: &[f32], predicted: &[f32]) -> f32 {
    assert_eq!(actual.len(), predicted.len());

    if actual.is_empty() {
        return 0.0;
    }

    let mut loss = 0.0;

    for i in 0..actual.len() {
        if actual[i] > 0.0 {
            loss -= actual[i] * predicted[i].max(1e-7).ln();
        }
    }

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_squared_error() {
        let actual = vec![1.0, 0.0, 1.0];
        let predicted = vec![0.8, 0.2, 0.9];

        let loss = mean_squared_error(&actual, &predicted);

        assert!((loss - 0.03).abs() < 0.0001);
    }

    #[test]
    fn test_mean_squared_error_perfect_prediction() {
        let actual = vec![1.0, 0.0, 1.0];

        let loss = mean_squared_error(&actual, &actual);

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_mean_squared_error_empty_inputs() {
        let actual: Vec<f32> = vec![];
        let predicted: Vec<f32> = vec![];

        let loss = mean_squared_error(&actual, &predicted);

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let actual = [0.0, 0.0, 1.0, 0.0];
        let predicted = [0.1, 0.2, 0.6, 0.1];

        let loss = cross_entropy_loss(&actual, &predicted);

        assert!((loss - 0.5108256).abs() < 0.0001);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let actual = [1.0, 0.0];
        let predicted = [1.0, 0.0];

        let loss = cross_entropy_loss(&actual, &predicted);

        assert!(loss < 0.0001);
    }

    #[test]
    fn test_cross_entropy_empty_inputs() {
        let loss = cross_entropy_loss(&[], &[]);

        assert_eq!(loss, 0.0);
    }
}
