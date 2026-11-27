use crate::neural_network::training::TrainingResult;
use crate::utils::metrics::ConfusionMatrix;

pub fn print_confusion_matrix(matrix: &ConfusionMatrix) {
    println!("\nConfusion Matrix:");
    println!("Rows = Actual digits | Columns = Predicted digits\n");

    print!("     ");

    for digit in 0..10 {
        print!("{:>6}", digit);
    }

    println!();

    for actual in 0..10 {
        print!("{:>4} ", actual);

        for predicted in 0..10 {
            print!("{:>6}", matrix.values[actual][predicted]);
        }

        println!();
    }
}

pub fn print_image(image: &[u8], rows: usize, cols: usize) {
    assert_eq!(image.len(), rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            let pixel = image[row * cols + col];

            let character = match pixel {
                0..=50 => ' ',
                51..=100 => '.',
                101..=150 => '*',
                151..=200 => 'o',
                _ => '#',
            };

            print!("{}", character);
        }

        println!();
    }
}

pub fn print_training_history(history: &[TrainingResult]) {
    println!("\nTraining History:");
    println!("{:<10} {:<15} {:<15}", "Epoch", "Loss", "Accuracy");

    for (index, result) in history.iter().enumerate() {
        println!(
            "{:<10} {:<15.4} {:<15.2}%",
            index + 1,
            result.average_loss,
            result.accuracy
        );
    }
}
