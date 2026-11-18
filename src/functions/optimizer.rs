pub fn sgd_update(weight: f32, gradient: f32, learning_rate: f32) -> f32 {
    weight - learning_rate * gradient
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update_positive_gradient() {
        let updated = sgd_update(0.8, 0.5, 0.1);

        assert!((updated - 0.75).abs() < 0.0001);
    }

    #[test]
    fn test_sgd_update_negative_gradient() {
        let updated = sgd_update(0.8, -0.5, 0.1);

        assert!((updated - 0.85).abs() < 0.0001);
    }

    #[test]
    fn test_sgd_update_zero_gradient() {
        let updated = sgd_update(0.8, 0.0, 0.1);

        assert_eq!(updated, 0.8);
    }

    #[test]
    fn test_sgd_update_zero_learning_rate() {
        let updated = sgd_update(0.8, 0.5, 0.0);

        assert_eq!(updated, 0.8);
    }
}
