//! AutoTuner Example: Hyperparameter Optimization for GBDT
//!
//! This example demonstrates how to use TreeBoost's AutoTuner to automatically
//! find optimal hyperparameters for gradient boosted decision trees.
//!
//! **Features Shown:**
//! - Setting up the tuner with a base configuration
//! - Configuring the parameter search space
//! - Using different grid strategies (Cartesian, LHS, Random)
//! - Holdout vs K-fold cross-validation
//! - Progress callbacks for real-time monitoring
//! - Extracting and using the best configuration
//!
//! Run with:
//!   cargo run --release --example autotuner

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::BinnedDataset;
use treeboost::tuner::{
    AutoTuner, EvalStrategy, GridStrategy, ModelFormat, ParamBounds, ParameterSpace, SpacePreset,
    TunableParam, TunerConfig,
};

/// Generate a synthetic regression dataset for demonstration
fn create_synthetic_dataset(n: usize, num_features: usize, seed: u64) -> BinnedDataset {
    let mut rng = common::SimpleRng::new(seed);

    let mut features = Vec::with_capacity(n * num_features);

    // Generate features (column-major layout)
    for _f in 0..num_features {
        for _r in 0..n {
            features.push((rng.next_f32() * 255.0) as u8);
        }
    }

    // Generate targets with a known relationship:
    // y = 10*f0 + 5*f1 - 3*f2 + noise
    let targets: Vec<f32> = (0..n)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n + i] as f32 / 255.0;
            let f2 = features[2 * n + i] as f32 / 255.0;
            10.0 * f0 + 5.0 * f1 - 3.0 * f2 + rng.next_f32() * 0.5
        })
        .collect();

    let feature_info = common::create_feature_info(num_features, "feature");
    BinnedDataset::new(n, features, targets, feature_info)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=============================================================");
    println!("TreeBoost AutoTuner Example");
    println!("=============================================================\n");

    // Create a synthetic dataset
    let num_rows = 5000;
    let num_features = 10;
    let seed = 42;

    println!("Generating synthetic dataset...");
    println!("  Rows: {}", num_rows);
    println!("  Features: {}", num_features);

    let dataset = create_synthetic_dataset(num_rows, num_features, seed);

    // =========================================================================
    // Example 1: Basic AutoTuner with default settings
    // =========================================================================
    println!("\n-------------------------------------------------------------");
    println!("Example 1: Basic AutoTuner with Default Settings");
    println!("-------------------------------------------------------------\n");

    // Create base configuration (non-tuned parameters)
    let base_config = GBDTConfig::new().with_num_rounds(50).with_seed(123);

    // Create tuner with default regression parameter space
    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 3 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(true);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config.clone())
        .with_config(tuner_config)
        .with_space(ParameterSpace::with_preset(SpacePreset::Regression))
        .with_seed(42);

    let start = Instant::now();
    let (best_config, history) = tuner.tune(&dataset).expect("Tuning should succeed");
    let elapsed = start.elapsed();

    println!("\n--- Results ---");
    println!("Total trials: {}", history.len());
    println!("Time: {:.2?}", elapsed);

    if let Some(best) = history.best() {
        println!("\nBest trial:");
        println!("  val_metric (MSE): {:.6}", best.val_loss);
        println!("  num_trees: {}", best.num_trees);
        println!("  Parameters:");
        for (name, value) in &best.params {
            println!("    {}: {:.4}", name, value);
        }
    }

    // Train final model with best config
    println!("\nTraining final model with best configuration...");
    let final_model =
        GBDTModel::train_binned(&dataset, best_config.clone()).expect("Training should succeed");

    let predictions = final_model.predict(&dataset);
    let mse: f32 = predictions
        .iter()
        .zip(dataset.targets().iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;
    println!("Final model training MSE: {:.6}", mse);

    // =========================================================================
    // Example 2: Latin Hypercube Sampling with Custom Space
    // =========================================================================
    println!("\n-------------------------------------------------------------");
    println!("Example 2: Latin Hypercube Sampling with Custom Space");
    println!("-------------------------------------------------------------\n");

    // Define custom parameter space
    let custom_space = ParameterSpace::new()
        .with_param(TunableParam::MaxDepth, ParamBounds::discrete(3, 10), 6.0)
        .with_param(
            TunableParam::LearningRate,
            ParamBounds::log_continuous(0.005, 0.5),
            0.05,
        )
        .with_param(
            TunableParam::Subsample,
            ParamBounds::continuous(0.6, 1.0),
            0.8,
        )
        .with_param(TunableParam::Lambda, ParamBounds::continuous(0.0, 5.0), 1.0);

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::LatinHypercube { n_samples: 15 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(true);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config.clone())
        .with_config(tuner_config)
        .with_space(custom_space)
        .with_seed(999);

    let start = Instant::now();
    let (_, history) = tuner.tune(&dataset).expect("LHS tuning should succeed");
    let elapsed = start.elapsed();

    println!("\n--- Results ---");
    println!("Total trials: {}", history.len());
    println!("Time: {:.2?}", elapsed);

    if let Some(best) = history.best() {
        println!("\nBest trial:");
        println!("  val_metric (MSE): {:.6}", best.val_loss);
        println!("  Parameters:");
        for (name, value) in &best.params {
            println!("    {}: {:.4}", name, value);
        }
    }

    // =========================================================================
    // Example 3: K-Fold Cross-Validation with Progress Callback
    // =========================================================================
    println!("\n-------------------------------------------------------------");
    println!("Example 3: K-Fold CV with Progress Callback");
    println!("-------------------------------------------------------------\n");

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2).with_folds(3)) // 3-fold CV
        .with_verbose(false); // We'll use our own progress output

    use std::sync::atomic::{AtomicU32, Ordering};

    // Use atomic to track best seen metric (bit representation of f32)
    let best_seen_bits = std::sync::Arc::new(AtomicU32::new(f32::MAX.to_bits()));
    let best_seen_clone = best_seen_bits.clone();

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config.clone())
        .with_config(tuner_config)
        .with_space(ParameterSpace::with_preset(SpacePreset::Regression))
        .with_seed(777)
        .with_callback(move |trial, current, total| {
            let current_best = f32::from_bits(best_seen_clone.load(Ordering::SeqCst));
            let is_new_best = trial.val_loss < current_best;
            if is_new_best {
                best_seen_clone.store(trial.val_loss.to_bits(), Ordering::SeqCst);
            }
            print!(
                "\r  Trial {}/{}: MSE={:.6} {}",
                current,
                total,
                trial.val_loss,
                if is_new_best { "*NEW BEST*" } else { "" }
            );
            // Flush to ensure output appears immediately
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        });

    println!("Running 3-fold cross-validation tuning...");
    let start = Instant::now();
    let (_, history) = tuner.tune(&dataset).expect("K-fold tuning should succeed");
    let elapsed = start.elapsed();
    println!(); // newline after progress

    println!("\n--- Results ---");
    println!("Total trials: {}", history.len());
    println!("Time: {:.2?}", elapsed);

    if let Some(best) = history.best() {
        println!("\nBest trial (3-fold CV):");
        println!("  mean val_metric (MSE): {:.6}", best.val_loss);
    }

    // =========================================================================
    // Example 4: Random Search with Early Stopping
    // =========================================================================
    println!("\n-------------------------------------------------------------");
    println!("Example 4: Random Search with Early Stopping");
    println!("-------------------------------------------------------------\n");

    // Enable early stopping in base config
    let base_config_es = GBDTConfig::new()
        .with_num_rounds(200) // High max, but early stopping will kick in
        .with_learning_rate(0.1)
        .with_early_stopping(10, 0.2)? // Stop after 10 rounds no improvement
        .with_seed(123);

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Random { n_samples: 10 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_verbose(true);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config_es)
        .with_config(tuner_config)
        .with_space(ParameterSpace::with_preset(SpacePreset::Regression))
        .with_seed(555);

    let start = Instant::now();
    let (_, history) = tuner.tune(&dataset).expect("Random tuning should succeed");
    let elapsed = start.elapsed();

    println!("\n--- Results ---");
    println!("Total trials: {}", history.len());
    println!("Time: {:.2?}", elapsed);

    if let Some(best) = history.best() {
        println!("\nBest trial:");
        println!("  val_metric (MSE): {:.6}", best.val_loss);
        println!(
            "  num_trees: {} (early stopped from max 200)",
            best.num_trees
        );
    }

    // Show tree counts distribution (demonstrates early stopping)
    let tree_counts: Vec<usize> = history.trials().iter().map(|t| t.num_trees).collect();
    let avg_trees: f32 = tree_counts.iter().sum::<usize>() as f32 / tree_counts.len() as f32;
    println!("  avg trees across trials: {:.1}", avg_trees);

    // =========================================================================
    // Example 5: Model Saving with Multiple Formats
    // =========================================================================
    println!("\n-------------------------------------------------------------");
    println!("Example 5: Model Saving with Multiple Formats");
    println!("-------------------------------------------------------------\n");

    // Create output directory for results
    let output_dir = std::path::Path::new("results/autotuner_example");

    let tuner_config = TunerConfig::new()
        .with_iterations(2)
        .with_grid_strategy(GridStrategy::Cartesian { points_per_dim: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_output_dir(output_dir) // Enable logging to directory
        .with_save_model_formats(vec![ModelFormat::Rkyv])
        .with_verbose(true);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config.clone())
        .with_config(tuner_config)
        .with_space(ParameterSpace::with_preset(SpacePreset::Minimal)) // Just max_depth and learning_rate
        .with_seed(12345);

    println!("Tuning with model saving enabled...");
    println!("  Output directory: {}", output_dir.display());
    println!("  Format: rkyv (zero-copy)\n");

    let start = Instant::now();
    let (best_config, history) = tuner
        .tune(&dataset)
        .expect("Tuning with save should succeed");
    let elapsed = start.elapsed();

    println!("\n--- Results ---");
    println!("Total trials: {}", history.len());
    println!("Time: {:.2?}", elapsed);

    if let Some(best) = history.best() {
        println!("\nBest trial:");
        println!("  val_metric (MSE): {:.6}", best.val_loss);
        println!("  num_trees: {}", best.num_trees);
    }

    // Verify saved files exist
    println!("\n--- Saved Files ---");

    // Find the run directory (timestamped)
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            let run_dir = entry.path();
            if run_dir.is_dir()
                && run_dir
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("run_")
            {
                println!("Run directory: {}", run_dir.display());

                // List files in run directory
                if let Ok(files) = std::fs::read_dir(&run_dir) {
                    for file in files.flatten() {
                        let path = file.path();
                        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        println!(
                            "  {} ({} bytes)",
                            path.file_name().unwrap().to_str().unwrap(),
                            size
                        );
                    }
                }

                // Test loading the saved model
                let rkyv_path = run_dir.join("best_model.rkyv");

                if rkyv_path.exists() {
                    println!("\nLoading model from rkyv...");
                    let loaded = treeboost::serialize::load_model(&rkyv_path)
                        .expect("Should load rkyv model");
                    println!("  Loaded {} trees", loaded.num_trees());

                    // Verify predictions match
                    let orig_preds = GBDTModel::train_binned(&dataset, best_config.clone())
                        .unwrap()
                        .predict(&dataset);
                    let loaded_preds = loaded.predict(&dataset);
                    let max_diff: f32 = orig_preds
                        .iter()
                        .zip(loaded_preds.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    println!("  Max prediction diff: {:.6}", max_diff);
                }

                break; // Only process first run directory
            }
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=============================================================");
    println!("Summary");
    println!("=============================================================\n");

    println!("The AutoTuner explored different hyperparameter configurations");
    println!("using iterative grid search with zooming (Auto-Zoom strategy).");
    println!("\nKey features demonstrated:");
    println!("  - Default regression/classification parameter spaces");
    println!("  - Custom parameter spaces with continuous, discrete, log-scale bounds");
    println!("  - Grid strategies: Cartesian, Latin Hypercube, Random");
    println!("  - Evaluation strategies: Holdout split, K-fold cross-validation");
    println!("  - Progress callbacks for real-time monitoring");
    println!("  - Integration with early stopping");
    println!("  - Model saving in rkyv format (zero-copy)");
    println!("  - Streaming CSV trial logs for interrupted runs");
    println!("\nFor production use, consider:");
    println!("  - More iterations (3-5) for better convergence");
    println!("  - Latin Hypercube for high-dimensional spaces");
    println!("  - K-fold CV for smaller datasets");
    println!("  - Early stopping to prevent overfitting");
    println!("  - with_save_rkyv() for fastest model loading");
    println!("  - with_save_all_formats() for flexibility");
    Ok(())
}
