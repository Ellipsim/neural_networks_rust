use neural_networks_rust::data::mnist::load_dataset;
use std::fs;
use tempfile::tempdir;

// Creates a valid MNIST image file with two 2x2 images.
fn create_test_images() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&2051u32.to_be_bytes());
    data.extend_from_slice(&2u32.to_be_bytes());
    data.extend_from_slice(&2u32.to_be_bytes());
    data.extend_from_slice(&2u32.to_be_bytes());

    data.extend_from_slice(&[0, 100, 150, 255]);
    data.extend_from_slice(&[255, 150, 100, 0]);

    data
}

// Creates a valid MNIST label file with two labels.
fn create_test_labels() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&2049u32.to_be_bytes());
    data.extend_from_slice(&2u32.to_be_bytes());

    data.extend_from_slice(&[3, 7]);

    data
}

fn create_test_labels_with_one_label() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&2049u32.to_be_bytes());
    data.extend_from_slice(&1u32.to_be_bytes());

    data.push(3);

    data
}

#[test]
fn test_load_dataset() {
    let dir = tempdir().unwrap();

    let images_path = dir.path().join("images.idx3-ubyte");
    let labels_path = dir.path().join("labels.idx1-ubyte");

    fs::write(&images_path, create_test_images()).unwrap();
    fs::write(&labels_path, create_test_labels()).unwrap();

    let dataset = load_dataset(
        images_path.to_str().unwrap(),
        labels_path.to_str().unwrap(),
    )
    .unwrap();

    assert_eq!(dataset.rows, 2);
    assert_eq!(dataset.cols, 2);
    assert_eq!(dataset.labels, vec![3, 7]);
    assert_eq!(dataset.images.len(), 8);
}

#[test]
fn test_load_dataset_invalid_image_magic_number() {
    let dir = tempdir().unwrap();

    let images_path = dir.path().join("images.idx3-ubyte");
    let labels_path = dir.path().join("labels.idx1-ubyte");

    let mut images = create_test_images();
    images[0..4].copy_from_slice(&1234u32.to_be_bytes());

    fs::write(&images_path, images).unwrap();
    fs::write(&labels_path, create_test_labels()).unwrap();

    let result = load_dataset(
        images_path.to_str().unwrap(),
        labels_path.to_str().unwrap(),
    );

    assert!(result.is_err());
}

#[test]
fn test_load_dataset_incomplete_image_header() {
    let dir = tempdir().unwrap();

    let images_path = dir.path().join("images.idx3-ubyte");
    let labels_path = dir.path().join("labels.idx1-ubyte");

    fs::write(&images_path, vec![0, 0, 0, 0]).unwrap();
    fs::write(&labels_path, create_test_labels()).unwrap();

    let result = load_dataset(
        images_path.to_str().unwrap(),
        labels_path.to_str().unwrap(),
    );

    assert!(result.is_err());
}

#[test]
fn test_load_dataset_invalid_label_magic_number() {
    let dir = tempdir().unwrap();

    let images_path = dir.path().join("images.idx3-ubyte");
    let labels_path = dir.path().join("labels.idx1-ubyte");

    let mut labels = create_test_labels();
    labels[0..4].copy_from_slice(&1234u32.to_be_bytes());

    fs::write(&images_path, create_test_images()).unwrap();
    fs::write(&labels_path, labels).unwrap();

    let result = load_dataset(
        images_path.to_str().unwrap(),
        labels_path.to_str().unwrap(),
    );

    assert!(result.is_err());
}

#[test]
fn test_load_dataset_invalid_label_value() {
    let dir = tempdir().unwrap();

    let images_path = dir.path().join("images.idx3-ubyte");
    let labels_path = dir.path().join("labels.idx1-ubyte");

    let mut labels = create_test_labels();
    labels[8] = 10;

    fs::write(&images_path, create_test_images()).unwrap();
    fs::write(&labels_path, labels).unwrap();

    let result = load_dataset(
        images_path.to_str().unwrap(),
        labels_path.to_str().unwrap(),
    );

    assert!(result.is_err());
}

#[test]
fn test_load_dataset_mismatched_counts() {
    let dir = tempdir().unwrap();

    let images_path = dir.path().join("images.idx3-ubyte");
    let labels_path = dir.path().join("labels.idx1-ubyte");

    fs::write(&images_path, create_test_images()).unwrap();
    fs::write(&labels_path, create_test_labels_with_one_label()).unwrap();

    let result = load_dataset(
        images_path.to_str().unwrap(),
        labels_path.to_str().unwrap(),
    );

    assert!(result.is_err());
}