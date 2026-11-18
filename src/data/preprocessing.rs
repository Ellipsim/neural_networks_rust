pub fn normalize_images(images: &[u8]) -> Vec<f32> {
    let mut normalized_images = Vec::new();

    for pixel in images {
        let normalized_pixel = *pixel as f32 / 255.0;
        normalized_images.push(normalized_pixel);
    }

    normalized_images
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_images() {
        let images = vec![0, 64, 128, 255];

        let normalized = normalize_images(&images);

        assert_eq!(normalized.len(), 4);
        assert_eq!(normalized[0], 0.0);
        assert!((normalized[1] - 0.25098).abs() < 0.0001);
        assert!((normalized[2] - 0.50196).abs() < 0.0001);
        assert_eq!(normalized[3], 1.0);
    }

    #[test]
    fn test_normalize_empty_images() {
        let images: Vec<u8> = vec![];

        let normalized = normalize_images(&images);

        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_preserves_original_images() {
        let images = vec![0, 128, 255];

        let _normalized = normalize_images(&images);

        assert_eq!(images, vec![0, 128, 255]);
    }
}