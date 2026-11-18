pub fn normalize_images(images: &[u8]) -> Vec<f32> {
    let mut normalized_images = Vec::new();

    for pixel in images {
        let normalized_pixel = *pixel as f32 / 255.0;
        normalized_images.push(normalized_pixel);
    }

    normalized_images
}

pub fn prepare_image(image: &[u8]) -> Vec<f32> {
    normalize_images(image)
}

pub fn one_hot_encode(label: u8) -> Vec<f32> {
    assert!(label < 10, "MNIST labels must be between 0 and 9");

    let mut encoded = vec![0.0; 10];
    encoded[label as usize] = 1.0;

    encoded
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

    #[test]
    fn test_prepare_image() {
        let image = vec![0, 128, 255];

        let prepared = prepare_image(&image);

        assert_eq!(prepared.len(), 3);
        assert_eq!(prepared[0], 0.0);
        assert!((prepared[1] - 0.50196).abs() < 0.0001);
        assert_eq!(prepared[2], 1.0);
    }

    #[test]
    fn test_prepare_empty_image() {
        let image: Vec<u8> = vec![];

        let prepared = prepare_image(&image);

        assert!(prepared.is_empty());
    }

    #[test]
    fn test_one_hot_encode() {
        let encoded = one_hot_encode(3);

        assert_eq!(
            encoded,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    #[should_panic]
    fn test_one_hot_encode_invalid_label() {
        one_hot_encode(10);
    }
}
