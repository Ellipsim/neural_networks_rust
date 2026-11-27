
use std::time::Instant;

use neural_networks_rust::data::mnist::load_dataset;
use neural_networks_rust::neural_network::evaluation::evaluate;
use neural_networks_rust::neural_network::network::Network;
use neural_networks_rust::neural_network::optimizers::adam::Adam;
use neural_networks_rust::neural_network::training::train;
use neural_networks_rust::utils::experiment::{
    log_experiment,
    ExperimentConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    let training_data = load_dataset(
        "data/train-images-idx3-ubyte/train-images.idx3-ubyte",
        "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte",
    )?;

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

    let mut network = Network::from_random(
        &architecture,
        "relu",
    );

    let epochs = 30;
    let batch_size = 32;

    // Learning rate: specified here.
    let learning_rate = 0.001;

    // Select and configure the optimizer.
    let mut optimizer = Adam::new(&network, learning_rate);

    println!("Starting training...");

    let training_start = Instant::now();

    let history = train(
        &mut network,
        &training_data,
        &["relu", "relu", "softmax"],
        &mut optimizer,
        epochs,
        batch_size,
    );

    let training_seconds = training_start.elapsed().as_secs_f64();

    for result in &history {
        println!(
            "Epoch {}/{} | Loss: {:.4} | Accuracy: {:.2}%",
            result.epoch,
            epochs,
            result.average_loss,
            result.accuracy * 100.0,
        );
    }

    println!("Training time: {:.3} seconds", training_seconds);

    println!("\nEvaluating on test data...");

    let evaluation_start = Instant::now();

    let evaluation = evaluate(
        &network,
        &test_data,
        &["relu", "relu", "softmax"],
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
