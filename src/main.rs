use neural_networks_rust::data::mnist::load_dataset;
use neural_networks_rust::neural_network::evaluation::evaluate;
use neural_networks_rust::neural_network::network::Network;
use neural_networks_rust::neural_network::training::train;
use neural_networks_rust::utils::metrics::{f1_score, precision, recall};
use neural_networks_rust::utils::visualization::{
    print_confusion_matrix, print_image, print_training_history,
};

fn main() {
    // 1. Load the training and test dataset
    let dataset = load_dataset(
        "data/train-images-idx3-ubyte/train-images.idx3-ubyte",
        "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte",
    )
    .expect("Failed to load MNIST dataset");

    println!("Loaded {} training images", dataset.labels.len());

    println!("\nExample training image:");
    println!("Label: {}", dataset.labels[0]);

    print_image(dataset.get_image(0), dataset.rows, dataset.cols);

    let test_dataset = load_dataset(
        "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
        "data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte",
    )
    .expect("Failed to load MNIST test dataset");

    println!("Loaded {} test images", test_dataset.labels.len());

    // 2. Define the network architecture
    let architecture = [784, 128, 64, 10];

    // 3. Initialize the network
    let mut network = Network::from_random(&architecture, "relu");

    // 4. Define activation functions
    let activations = ["relu", "relu", "softmax"];

    // 5. Train the network
    let training_history = train(&mut network, &dataset, &activations, 0.01, 5);

    print_training_history(&training_history);

    // 6. Test
    let results = evaluate(&network, &test_dataset, &activations);

    println!("\nEvaluation results:");
    println!("Test loss: {:.4}", results.average_loss);
    println!(
        "Correct predictions: {}/{}",
        results.correct_predictions, results.total_images
    );
    println!("Test accuracy: {:.2}%", results.accuracy);

    println!("\nPer-digit metrics:");

    for digit in 0..10 {
        let p = precision(&results.confusion_matrix, digit);
        let r = recall(&results.confusion_matrix, digit);
        let f1 = f1_score(&results.confusion_matrix, digit);

        println!(
            "Digit {} | Precision: {:.4} | Recall: {:.4} | F1: {:.4}",
            digit, p, r, f1
        );
    }

    print_confusion_matrix(&results.confusion_matrix);
}
