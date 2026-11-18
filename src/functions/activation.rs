pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    let output = sigmoid(x);
    output * (1.0 - output)
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn softmax(inputs: &[f32]) -> Vec<f32> {
    if inputs.is_empty() {
        return Vec::new();
    }

    let max_value = inputs.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exponentials: Vec<f32> = inputs.iter().map(|x| (x - max_value).exp()).collect();

    let sum: f32 = exponentials.iter().sum();

    exponentials.iter().map(|x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_zero() {
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn test_sigmoid_positive() {
        assert!(sigmoid(10.0) > 0.99);
    }

    #[test]
    fn test_sigmoid_negative() {
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_relu_positive() {
        assert_eq!(relu(5.0), 5.0);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(relu(-5.0), 0.0);
    }

    #[test]
    fn test_relu_zero() {
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_sigmoid_derivative_zero() {
        assert_eq!(sigmoid_derivative(0.0), 0.25);
    }

    #[test]
    fn test_sigmoid_derivative_positive() {
        let derivative = sigmoid_derivative(2.0);

        assert!(derivative > 0.0);
        assert!(derivative < 0.25);
    }

    #[test]
    fn test_relu_derivative_positive() {
        assert_eq!(relu_derivative(5.0), 1.0);
    }

    #[test]
    fn test_relu_derivative_negative() {
        assert_eq!(relu_derivative(-5.0), 0.0);
    }

    #[test]
    fn test_relu_derivative_zero() {
        assert_eq!(relu_derivative(0.0), 0.0);
    }

    #[test]
    fn test_softmax_probabilities_sum_to_one() {
        let outputs = softmax(&[2.0, 1.0, 0.1]);

        let sum: f32 = outputs.iter().sum();

        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_softmax_values_are_positive() {
        let outputs = softmax(&[2.0, 1.0, 0.1]);

        for value in outputs {
            assert!(value > 0.0);
        }
    }

    #[test]
    fn test_softmax_empty_input() {
        let outputs = softmax(&[]);

        assert!(outputs.is_empty());
    }
}
