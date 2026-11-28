
use std::time::Instant;

use neural_networks_rust::data::mnist::{load_dataset, split_dataset};
use neural_networks_rust::neural_network::evaluation::evaluate;
use neural_networks_rust::neural_network::network::Network;
use neural_networks_rust::neural_network::optimizers::adam::Adam;
use neural_networks_rust::neural_network::training::{
    train,
    EarlyStopping,
};
use neural_networks_rust::utils::experiment::{
    log_experiment,
    ExperimentConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    let seed = 42;

    // Load the original MNIST training dataset.
    let training_data = load_dataset(
        "data/train-images-idx3-ubyte/train-images.idx3-ubyte",
        "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte",
    )?;

    // Split it into training and validation datasets.
    let (training_dataset, validation_dataset) = split_dataset(
        training_data,
        10_000,
        seed,
    );

    // Load the independent MNIST test dataset.
    let test_data = load_dataset(
        "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
        "data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte",
    )?;

    let architecture = vec![784, 128, 64, 10];

    let activations = vec![
        "relu".to_string(),
        "relu".to_string(),
        "softmax".to_string(),
    ];

    let activation_refs = ["relu", "relu", "softmax"];

    let mut network = Network::from_random(
        &architecture,
        "relu",
        Some(seed),
    );

    let epochs = 30;
    let batch_size = 32;
    let learning_rate = 0.001;

    let mut optimizer = Adam::new(&network, learning_rate);

    let early_stopping = EarlyStopping {
        patience: 5,
        min_delta: 0.001,
    };

    println!("Starting training...");
    println!("Training images: {}", training_dataset.labels.len());
    println!("Validation images: {}", validation_dataset.labels.len());

    let training_start = Instant::now();

    let history = train(
        &mut network,
        &training_dataset,
        &validation_dataset,
        &activation_refs,
        &mut optimizer,
        epochs,
        batch_size,
        seed,
        early_stopping,
    );

    let training_seconds = training_start.elapsed().as_secs_f64();

    for result in &history {
        println!(
            "Epoch {}/{} | Training loss: {:.4} | Training accuracy: {:.2}% | Validation loss: {:.4} | Validation accuracy: {:.2}%",
            result.epoch,
            epochs,
            result.average_loss,
            result.accuracy * 100.0,
            result.validation_loss,
            result.validation_accuracy * 100.0,
        );
    }

    println!("Training time: {:.3} seconds", training_seconds);

    // Evaluate the final network on the independent test dataset.
    println!("\nEvaluating on test data...");

    let evaluation_start = Instant::now();

    let evaluation = evaluate(
        &network,
        &test_data,
        &activation_refs,
    );

    let evaluation_seconds = evaluation_start.elapsed().as_secs_f64();

    println!("Test loss: {:.4}", evaluation.average_loss);
    println!("Test accuracy: {:.2}%", evaluation.accuracy * 100.0);

    println!(
        "Correct predictions: {}/{}",
        evaluation.correct_predictions,
        evaluation.total_images
    );

    // Save experiment results.
    let config = ExperimentConfig {
        architecture,
        activations,
        optimizer: "Adam".to_string(),
        learning_rate,
        batch_size,
        epochs,
        seed,
    };

    let run_path = log_experiment(
        &config,
        &history,
        &evaluation,
        training_seconds,
        evaluation_seconds,
    )?;

    println!("\nExperiment saved:");
    println!("{}", run_path.display());

    println!(
        "Total execution time: {:.3} seconds",
        total_start.elapsed().as_secs_f64()
    );

    Ok(())
}
