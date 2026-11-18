use std::fs;
use std::io::{Error, ErrorKind};

pub struct Mnist {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Mnist {
    pub fn get_image(&self, index: usize) -> &[u8] {
        let image_size = self.rows * self.cols;

        let start = index * image_size;
        let end = start + image_size;

        &self.images[start..end]
    }

    pub fn get_prepared_image(&self, index: usize) -> Vec<f32> {
        let image = self.get_image(index);
        crate::data::preprocessing::prepare_image(image)
    }
}

pub fn load_dataset(images_path: &str, labels_path: &str) -> Result<Mnist, Error> {
    let (images, rows, cols) = load_images(images_path)?;
    let labels = load_labels(labels_path)?;

    if images.len() != labels.len() * rows * cols {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Number of images and labels does not match",
        ));
    }

    Ok(Mnist {
        images,
        labels,
        rows,
        cols,
    })
}

fn load_images(file_path: &str) -> Result<(Vec<u8>, usize, usize), Error> {
    let data = fs::read(file_path)?;

    // Read header
    if data.len() < 16 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Incomplete image header",
        ));
    }

    let magic_number = u32::from_be_bytes(data[0..4].try_into().unwrap());

    if magic_number != 2051 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid image magic number",
        ));
    }

    let n_images = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(data[12..16].try_into().unwrap()) as usize;

    // Read images
    let images = data[16..].to_vec();

    // Validate dimensions
    if images.len() != n_images * rows * cols {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid image data size",
        ));
    }

    Ok((images, rows, cols))
}

fn load_labels(file_path: &str) -> Result<Vec<u8>, Error> {
    let data = fs::read(file_path)?;

    if data.len() < 8 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Incomplete label header",
        ));
    }

    let magic_number = u32::from_be_bytes(data[0..4].try_into().unwrap());

    if magic_number != 2049 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid label magic number",
        ));
    }

    let n_labels = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;

    let labels = data[8..].to_vec();

    if labels.len() != n_labels {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid label data size",
        ));
    }

    if !labels.iter().all(|&label| label <= 9) {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid label value"));
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_load_images() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("images.idx3-ubyte");

        fs::write(&path, create_test_images()).unwrap();

        let (images, rows, cols) = load_images(path.to_str().unwrap()).unwrap();

        assert_eq!(images.len(), 8);
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
    }

    #[test]
    fn test_load_images_invalid_magic_number() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("images-invalid.idx3-ubyte");

        let mut data = create_test_images();

        // Change magic number
        data[0..4].copy_from_slice(&1234u32.to_be_bytes());

        fs::write(&path, data).unwrap();

        let result = load_images(path.to_str().unwrap());

        assert!(result.is_err());
    }

    #[test]
    fn test_load_images_incomplete_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("images-incomplete.idx3-ubyte");

        fs::write(&path, vec![0, 0, 0, 0]).unwrap();

        let result = load_images(path.to_str().unwrap());

        assert!(result.is_err());
    }

    fn create_test_images() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&2051u32.to_be_bytes());
        data.extend_from_slice(&2u32.to_be_bytes());
        data.extend_from_slice(&2u32.to_be_bytes());
        data.extend_from_slice(&2u32.to_be_bytes());

        // Two images of 2x2 pixels
        data.extend_from_slice(&[0, 100, 150, 255]);
        data.extend_from_slice(&[255, 150, 100, 0]);

        data
    }

    #[test]
    fn test_get_image() {
        let dataset = Mnist {
            images: vec![0, 100, 150, 255, 255, 150, 100, 0],
            labels: vec![3, 7],
            rows: 2,
            cols: 2,
        };

        let image = dataset.get_image(0);

        assert_eq!(image, &[0, 100, 150, 255]);
    }

    #[test]
    fn test_get_second_image() {
        let dataset = Mnist {
            images: vec![0, 100, 150, 255, 255, 150, 100, 0],
            labels: vec![3, 7],
            rows: 2,
            cols: 2,
        };

        let image = dataset.get_image(1);

        assert_eq!(image, &[255, 150, 100, 0]);
    }

    #[test]
    fn test_get_prepared_image() {
        let dataset = Mnist {
            images: vec![0, 128, 255, 64],
            labels: vec![3],
            rows: 2,
            cols: 2,
        };

        let prepared = dataset.get_prepared_image(0);

        assert_eq!(prepared.len(), 4);
        assert_eq!(prepared[0], 0.0);
        assert!((prepared[1] - 0.50196).abs() < 0.0001);
        assert_eq!(prepared[2], 1.0);
        assert!((prepared[3] - 0.25098).abs() < 0.0001);

        // The original image must remain unchanged.
        assert_eq!(dataset.get_image(0), &[0, 128, 255, 64]);
    }
}
