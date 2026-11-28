
    use std::fs::{self, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::neural_network::evaluation::EvaluationResult;
    use crate::neural_network::training::TrainingResult;

    pub struct ExperimentConfig {
        pub architecture: Vec<usize>,
        pub activations: Vec<String>,
        pub optimizer: String,
        pub learning_rate: f32,
        pub batch_size: usize,
        pub epochs: usize,
        pub seed: u64,
    }

    pub fn log_experiment(
        config: &ExperimentConfig,
        history: &[TrainingResult],
        evaluation: &EvaluationResult,
        training_seconds: f64,
        evaluation_seconds: f64,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let results_dir = PathBuf::from("docs/results/runs");

        fs::create_dir_all(&results_dir)?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis();

        let run_id = format!("run_{}", timestamp);
        let run_path = results_dir.join(format!("{}.csv", run_id));

        // Save epoch-by-epoch training metrics.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&run_path)?;

        let mut writer = BufWriter::new(file);

        writeln!(writer, "# seed,{}", config.seed)?;

        writeln!(
            writer,
            "epoch,training_loss,training_accuracy,correct_predictions,total_images"
        )?;

        for result in history {
            writeln!(
                writer,
                "{},{:.6},{:.6},{},{}",
                result.epoch,
                result.average_loss,
                result.accuracy,
                result.correct_predictions,
                result.total_images
            )?;
        }

        writer.flush()?;

        // Append experiment summary.
        let summary_path = PathBuf::from("docs/results/experiments.csv");

        let summary_exists = summary_path.exists();

        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(summary_path)?;

        let mut writer = BufWriter::new(file);

        if !summary_exists {
            writeln!(
                writer,
                "run_id,architecture,activations,optimizer,learning_rate,batch_size,epochs,seed,training_seconds,evaluation_seconds,test_loss,test_accuracy,correct_predictions,total_images"
            )?;
        }

        let architecture = config
            .architecture
            .iter()
            .map(|size| size.to_string())
            .collect::<Vec<_>>()
            .join("-");

        let activations = config.activations.join("-");

        writeln!(
            writer,
            "{},{},{},{},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{}",
            run_id,
            architecture,
            activations,
            config.optimizer,
            config.learning_rate,
            config.batch_size,
            config.epochs,
            config.seed,
            training_seconds,
            evaluation_seconds,
            evaluation.average_loss,
            evaluation.accuracy,
            evaluation.correct_predictions,
            evaluation.total_images
        )?;

        writer.flush()?;

        Ok(run_path)
    }
