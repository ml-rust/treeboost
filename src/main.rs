//! TreeBoost CLI
//!
//! Command-line interface for training, prediction, model inspection, and incremental updates.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::DatasetLoader;
use treeboost::model::AutoModel;
use treeboost::serialize::{load_model, save_model, TrbReader};
use treeboost::tree::MonotonicConstraint;
use treeboost::Result;

#[derive(Parser)]
#[command(name = "treeboost")]
#[command(about = "High-performance Gradient Boosted Decision Tree engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model
    Train {
        /// Input data file (Parquet or CSV)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Output model path
        #[arg(short, long)]
        output: PathBuf,

        /// Number of boosting rounds
        #[arg(long, default_value_t = 100)]
        rounds: usize,

        /// Maximum tree depth
        #[arg(long, default_value_t = 6)]
        max_depth: usize,

        /// Maximum number of leaves
        #[arg(long, default_value_t = 31)]
        max_leaves: usize,

        /// Learning rate
        #[arg(long, default_value_t = 0.1)]
        learning_rate: f32,

        /// Minimum samples per leaf
        #[arg(long, default_value_t = 20)]
        min_samples_leaf: usize,

        /// L2 regularization (lambda)
        #[arg(long, default_value_t = 1.0)]
        lambda: f32,

        /// Shannon Entropy regularization weight
        #[arg(long, default_value_t = 0.0)]
        entropy_weight: f32,

        /// Row subsampling ratio (0.0 to 1.0)
        #[arg(long, default_value_t = 1.0)]
        subsample: f32,

        /// Column subsampling ratio (0.0 to 1.0)
        #[arg(long, default_value_t = 1.0)]
        colsample: f32,

        /// Early stopping rounds (0 to disable)
        #[arg(long, default_value_t = 0)]
        early_stopping: usize,

        /// Validation ratio for early stopping (e.g., 0.1 for 10%)
        #[arg(long, default_value_t = 0.1)]
        validation_ratio: f32,

        /// Loss function: mse or huber
        #[arg(long, default_value = "mse")]
        loss: String,

        /// Pseudo-Huber delta (only if loss=huber)
        #[arg(long, default_value_t = 1.0)]
        huber_delta: f32,

        /// Enable conformal prediction with calibration ratio
        #[arg(long)]
        conformal: Option<f32>,

        /// Conformal quantile level (e.g., 0.9 for 90% coverage)
        #[arg(long, default_value_t = 0.9)]
        conformal_quantile: f32,

        /// Number of bins for feature discretization
        #[arg(long, default_value_t = 255)]
        num_bins: usize,

        /// Feature columns (comma-separated, or all if not specified)
        #[arg(long)]
        features: Option<String>,

        /// Disable parallel prediction (use sequential instead)
        #[arg(long)]
        no_parallel: bool,

        /// Disable column reordering optimization
        #[arg(long)]
        no_reorder: bool,

        /// Disable 4-bit packed dataset optimization
        #[arg(long)]
        no_pack: bool,

        /// Disable all performance optimizations
        #[arg(long)]
        no_optimizations: bool,

        /// Monotonic constraints (comma-separated: +1=increasing, -1=decreasing, 0=none)
        /// Example: "--monotonic +1,-1,0" for 3 features
        #[arg(long)]
        monotonic: Option<String>,

        /// Feature interaction groups (semicolon-separated groups of comma-separated indices)
        /// Example: "--interactions 0,1,2;3,4" means features 0-2 can interact, features 3-4 can interact
        #[arg(long)]
        interactions: Option<String>,
    },

    /// Make predictions using a trained model
    Predict {
        /// Trained model path
        #[arg(short, long)]
        model: PathBuf,

        /// Input data file (Parquet or CSV)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name (optional, for evaluation)
        #[arg(short, long)]
        target: Option<String>,

        /// Output predictions file (JSON)
        #[arg(short, long)]
        output: PathBuf,

        /// Include prediction intervals (if model was calibrated)
        #[arg(long)]
        intervals: bool,
    },

    /// Display model information
    Info {
        /// Model path (.rkyv or .trb)
        #[arg(short, long)]
        model: PathBuf,

        /// Show feature importance
        #[arg(long)]
        importances: bool,

        /// Force loading despite corrupted updates (loads base only)
        #[arg(long)]
        force: bool,
    },

    /// Update an existing model with new data (incremental learning)
    Update {
        /// Existing model path (.trb format required)
        #[arg(short, long)]
        model: PathBuf,

        /// Input data file (Parquet or CSV)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Number of additional boosting rounds
        #[arg(long, default_value_t = 10)]
        rounds: usize,

        /// Description of this update
        #[arg(long, default_value = "")]
        description: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            data,
            target,
            output,
            rounds,
            max_depth,
            max_leaves,
            learning_rate,
            min_samples_leaf,
            lambda,
            entropy_weight,
            subsample,
            colsample,
            early_stopping,
            validation_ratio,
            loss,
            huber_delta,
            conformal,
            conformal_quantile,
            num_bins,
            features,
            no_parallel,
            no_reorder,
            no_pack,
            no_optimizations,
            monotonic,
            interactions,
        } => {
            println!("Loading data from {:?}...", data);
            let loader = DatasetLoader::new(num_bins);

            let feature_cols: Option<Vec<&str>> = features
                .as_ref()
                .map(|f| f.split(',').map(|s| s.trim()).collect());

            let dataset = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                loader.load_parquet(&data, &target, feature_cols.as_deref())?
            } else {
                loader.load_csv(&data, &target, feature_cols.as_deref())?
            };

            println!(
                "Loaded {} rows, {} features",
                dataset.num_rows(),
                dataset.num_features()
            );

            // Build config
            let mut config = GBDTConfig::new()
                .with_num_rounds(rounds)
                .with_max_depth(max_depth)
                .with_max_leaves(max_leaves)
                .with_learning_rate(learning_rate)
                .with_min_samples_leaf(min_samples_leaf)
                .with_lambda(lambda)
                .with_entropy_weight(entropy_weight)
                .with_subsample(subsample)?
                .with_colsample(colsample)?;

            // Enable early stopping if requested
            if early_stopping > 0 {
                config = config.with_early_stopping(early_stopping, validation_ratio)?;
                println!(
                    "Early stopping enabled: {} rounds, {:.1}% validation",
                    early_stopping,
                    validation_ratio * 100.0
                );
            }

            // Set loss function
            config = match loss.as_str() {
                "mse" => config.with_mse_loss(),
                "huber" => config.with_pseudo_huber_loss(huber_delta),
                _ => {
                    eprintln!("Unknown loss function: {}. Using MSE.", loss);
                    config.with_mse_loss()
                }
            };

            // Enable conformal prediction if requested
            if let Some(calib_ratio) = conformal {
                config = config.with_conformal(calib_ratio, conformal_quantile)?;
                println!(
                    "Conformal prediction enabled: {:.1}% calibration, {:.1}% coverage",
                    calib_ratio * 100.0,
                    conformal_quantile * 100.0
                );
            }

            // Parse and apply monotonic constraints
            if let Some(ref mono_str) = monotonic {
                let constraints: Vec<MonotonicConstraint> = mono_str
                    .split(',')
                    .map(|s| {
                        let s = s.trim();
                        match s {
                            "+1" | "1" => MonotonicConstraint::Increasing,
                            "-1" => MonotonicConstraint::Decreasing,
                            "0" | "" => MonotonicConstraint::None,
                            _ => {
                                eprintln!("Warning: Unknown monotonic value '{}', using None", s);
                                MonotonicConstraint::None
                            }
                        }
                    })
                    .collect();

                let num_constrained = constraints
                    .iter()
                    .filter(|c| **c != MonotonicConstraint::None)
                    .count();
                if num_constrained > 0 {
                    println!(
                        "Monotonic constraints: {} features constrained",
                        num_constrained
                    );
                }
                config = config.with_monotonic_constraints(constraints);
            }

            // Parse and apply interaction constraints
            if let Some(ref interact_str) = interactions {
                let groups: Vec<Vec<usize>> = interact_str
                    .split(';')
                    .filter(|g| !g.trim().is_empty())
                    .map(|group| {
                        group
                            .split(',')
                            .filter_map(|s| s.trim().parse::<usize>().ok())
                            .collect()
                    })
                    .filter(|g: &Vec<usize>| !g.is_empty())
                    .collect();

                if !groups.is_empty() {
                    println!("Interaction constraints: {} groups", groups.len());
                    config = config.with_interaction_groups(groups);
                }
            }

            // Apply optimization opt-outs
            if no_optimizations {
                config = config.without_optimizations();
            } else {
                if no_parallel {
                    config = config.with_parallel_prediction(false);
                }
                if no_reorder {
                    config = config.with_column_reordering(false);
                }
                if no_pack {
                    config = config.with_packed_dataset(false);
                }
            }

            println!("\nTraining configuration:");
            println!("  Rounds: {}", rounds);
            println!("  Max depth: {}", max_depth);
            println!("  Max leaves: {}", max_leaves);
            println!("  Learning rate: {}", learning_rate);
            println!("  Loss: {}", loss);
            if entropy_weight > 0.0 {
                println!("  Entropy weight: {}", entropy_weight);
            }
            if subsample < 1.0 {
                println!("  Row subsampling: {:.0}%", subsample * 100.0);
            }
            if colsample < 1.0 {
                println!("  Column subsampling: {:.0}%", colsample * 100.0);
            }
            if early_stopping > 0 {
                println!(
                    "  Early stopping: {} rounds, {:.0}% validation",
                    early_stopping,
                    validation_ratio * 100.0
                );
            }
            println!("  Optimizations:");
            println!("    Parallel prediction: {}", config.parallel_prediction);
            println!("    Column reordering: {}", config.column_reordering);
            println!("    4-bit packing: {}", config.packed_dataset);

            println!("\nTraining model...");
            let model = GBDTModel::train_binned(&dataset, config)?;

            println!("Training complete: {} trees built", model.num_trees());

            if let Some(q) = model.conformal_quantile() {
                println!("Conformal quantile: {:.4}", q);
            }

            // Determine output format from extension
            let output_ext = output.extension().and_then(|s| s.to_str()).unwrap_or("");

            if output_ext == "trb" {
                // TRB format requires AutoModel/UniversalModel for full incremental support
                // The CLI train command uses GBDTModel directly, so we save as rkyv
                // and inform the user about using the Rust API for TRB format
                eprintln!("Note: CLI train command saves to rkyv format (.rkyv extension).");
                eprintln!("      For TRB incremental format, use the Rust API with AutoModel:");
                eprintln!("        let model = AutoModel::train(&df, \"target\")?;");
                eprintln!("        model.save_trb(\"model.trb\", \"description\")?;");
                eprintln!();

                // Change extension to .rkyv
                let mut rkyv_output = output.clone();
                rkyv_output.set_extension("rkyv");
                println!("Saving model to {:?} (rkyv format)...", rkyv_output);
                save_model(&model, &rkyv_output)?;
                println!("Model saved successfully.");
            } else {
                println!("\nSaving model to {:?} (rkyv format)...", output);
                save_model(&model, &output)?;
                println!("Model saved successfully.");
            }
            Ok(())
        }

        Commands::Predict {
            model,
            data,
            target,
            output,
            intervals,
        } => {
            println!("Loading model from {:?}...", model);
            let model = load_model(&model)?;

            println!(
                "Model loaded: {} trees, {} features",
                model.num_trees(),
                model.num_features()
            );

            println!("Loading data from {:?}...", data);
            let loader = DatasetLoader::new(255);

            // Use model's feature info for consistent binning
            let dataset = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                loader.load_parquet_for_prediction(&data, model.feature_info())?
            } else {
                loader.load_csv_for_prediction(&data, model.feature_info())?
            };

            println!("Loaded {} rows", dataset.num_rows());

            // Load actual targets if specified (for evaluation)
            let actual_targets: Option<Vec<f32>> = if let Some(ref target_col) = target {
                use polars::prelude::*;
                let df = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                    let pl_path = PlRefPath::new(&*data.to_string_lossy());
                    LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?
                } else {
                    CsvReadOptions::default()
                        .try_into_reader_with_file_path(Some(data.clone()))?
                        .finish()?
                };
                let col = df.column(target_col)?;
                let series = col.as_materialized_series();
                let targets: Vec<f32> = series
                    .cast(&DataType::Float32)?
                    .f32()?
                    .iter()
                    .map(|opt| opt.unwrap_or(f32::NAN))
                    .collect();
                Some(targets)
            } else {
                None
            };

            println!("Making predictions...");
            if intervals && model.conformal_quantile().is_some() {
                let (predictions, lower, upper) = model.predict_with_intervals(&dataset);

                // Create JSON output
                let results: Vec<serde_json::Value> = predictions
                    .iter()
                    .zip(lower.iter())
                    .zip(upper.iter())
                    .enumerate()
                    .map(|(i, ((p, l), u))| {
                        serde_json::json!({
                            "row": i,
                            "prediction": p,
                            "lower": l,
                            "upper": u,
                        })
                    })
                    .collect();

                let json = serde_json::to_string_pretty(&results).map_err(std::io::Error::other)?;
                std::fs::write(&output, json)?;
                println!("Predictions with intervals saved to {:?}", output);

                // Compute metrics if target available
                if let Some(ref targets) = actual_targets {
                    let mse: f32 = predictions
                        .iter()
                        .zip(targets.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f32>()
                        / predictions.len() as f32;

                    let coverage: f32 = targets
                        .iter()
                        .zip(lower.iter())
                        .zip(upper.iter())
                        .filter(|((t, l), u)| **t >= **l && **t <= **u)
                        .count() as f32
                        / targets.len() as f32;

                    println!("\nEvaluation:");
                    println!("  MSE: {:.4}", mse);
                    println!("  RMSE: {:.4}", mse.sqrt());
                    println!("  Coverage: {:.2}%", coverage * 100.0);
                }
            } else {
                let predictions = model.predict(&dataset);

                // Create JSON output
                let results: Vec<serde_json::Value> = predictions
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        serde_json::json!({
                            "row": i,
                            "prediction": p,
                        })
                    })
                    .collect();

                let json = serde_json::to_string_pretty(&results).map_err(std::io::Error::other)?;
                std::fs::write(&output, json)?;
                println!("Predictions saved to {:?}", output);

                // Compute metrics if target available
                if let Some(ref targets) = actual_targets {
                    let mse: f32 = predictions
                        .iter()
                        .zip(targets.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f32>()
                        / predictions.len() as f32;

                    println!("\nEvaluation:");
                    println!("  MSE: {:.4}", mse);
                    println!("  RMSE: {:.4}", mse.sqrt());
                }
            }

            Ok(())
        }

        Commands::Info {
            model,
            importances,
            force,
        } => {
            let model_ext = model.extension().and_then(|s| s.to_str()).unwrap_or("");

            if model_ext == "trb" {
                // Handle TRB format with update history
                println!("Loading TRB model from {:?}...", model);

                let mut reader = TrbReader::open(&model)?;
                let header = reader.header().clone();

                println!("\nTRB Model Information:");
                println!("  Format version: {}", header.format_version);
                println!("  Model type: {}", header.model_type);
                println!("  Boosting mode: {}", header.boosting_mode);
                println!("  Number of features: {}", header.num_features);
                println!("  Created: {}", format_timestamp(header.created_at));
                if !header.metadata.is_empty() {
                    println!("  Description: {}", header.metadata);
                }
                println!("  Base model size: {} bytes", header.base_blob_size);

                // Read base blob to validate CRC
                match reader.read_base_blob() {
                    Ok(_) => println!("  Base model CRC: OK"),
                    Err(e) => {
                        eprintln!("  Base model CRC: FAILED - {}", e);
                        if !force {
                            eprintln!(
                                "\nError: Base model is corrupted. Use --force to skip validation."
                            );
                            return Err(e);
                        }
                        eprintln!("  (--force: continuing despite corruption)");
                    }
                }

                // Read updates
                println!("\nUpdate History:");
                let updates = reader.iter_updates()?;
                if updates.is_empty() {
                    println!("  No updates (base model only)");
                } else {
                    let mut total_rows = 0usize;
                    for (i, (update_header, _blob)) in updates.iter().enumerate() {
                        total_rows += update_header.rows_trained;
                        println!("  Update {}:", i + 1);
                        println!("    Type: {:?}", update_header.update_type);
                        println!(
                            "    Created: {}",
                            format_timestamp(update_header.created_at)
                        );
                        println!("    Rows trained: {}", update_header.rows_trained);
                        if !update_header.description.is_empty() {
                            println!("    Description: {}", update_header.description);
                        }
                    }
                    println!("\n  Total updates: {}", updates.len());
                    println!("  Total rows across updates: {}", total_rows);
                }

                // Load the actual model for tree count
                use treeboost::model::UniversalModel;
                match UniversalModel::load_trb(&model) {
                    Ok(loaded_model) => {
                        println!("\nModel State:");
                        println!("  Current tree count: {}", loaded_model.num_trees());
                    }
                    Err(e) => {
                        if !force {
                            eprintln!("\nError loading model: {}", e);
                            return Err(e);
                        }
                        eprintln!("\nWarning: Could not load full model state: {}", e);
                    }
                }

                Ok(())
            } else {
                // Handle rkyv format
                println!("Loading model from {:?}...", model);
                let model = load_model(&model)?;

                println!("\nModel Information:");
                println!("  Number of trees: {}", model.num_trees());
                println!("  Base prediction: {:.4}", model.base_prediction());

                let config = model.config();
                println!("\nTraining Configuration:");
                println!("  Rounds: {}", config.num_rounds);
                println!("  Max depth: {}", config.max_depth);
                println!("  Max leaves: {}", config.max_leaves);
                println!("  Learning rate: {}", config.learning_rate);
                println!("  Lambda: {}", config.lambda);
                println!("  Min samples/leaf: {}", config.min_samples_leaf);
                println!("  Subsample: {}", config.subsample);
                println!("  Loss: {:?}", config.loss_type);

                if config.entropy_weight > 0.0 {
                    println!("  Entropy weight: {}", config.entropy_weight);
                }

                if let Some(q) = model.conformal_quantile() {
                    println!("\nConformal Prediction:");
                    println!("  Quantile: {:.4}", q);
                    println!("  Coverage: {:.1}%", config.conformal_quantile * 100.0);
                }

                if importances {
                    println!("\nFeature Importance:");
                    let imps = model.feature_importance();
                    for (i, imp) in imps.iter().enumerate() {
                        println!("  Feature {}: {:.4}", i, imp);
                    }
                }

                Ok(())
            }
        }

        Commands::Update {
            model,
            data,
            target,
            rounds,
            description,
        } => {
            // Verify .trb extension
            let model_ext = model.extension().and_then(|s| s.to_str()).unwrap_or("");
            if model_ext != "trb" {
                eprintln!(
                    "Error: Update command requires .trb format. Got: {:?}",
                    model
                );
                eprintln!(
                    "Hint: Train with .trb extension: treeboost train ... --output model.trb"
                );
                return Err(treeboost::TreeBoostError::Config(
                    "Update requires .trb format".to_string(),
                ));
            }

            println!("Loading model from {:?}...", model);
            let mut auto_model = AutoModel::load_trb(&model, &target)?;
            let trees_before = auto_model.num_trees();

            println!(
                "Model loaded: {} trees, {} features",
                trees_before,
                auto_model.num_features()
            );

            println!("Loading data from {:?}...", data);
            let df = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                use polars::prelude::*;
                let pl_path = PlRefPath::new(&*data.to_string_lossy());
                LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?
            } else {
                use polars::prelude::*;
                CsvReadOptions::default()
                    .try_into_reader_with_file_path(Some(data.clone()))?
                    .finish()?
            };

            let num_rows = df.height();
            println!("Loaded {} rows", num_rows);

            println!("\nUpdating model with {} additional rounds...", rounds);
            let report = auto_model.update(&df, rounds)?;

            println!("Update complete:");
            println!("  Rows trained: {}", report.rows_trained);
            println!(
                "  Trees: {} -> {} (+{})",
                report.trees_before, report.trees_after, report.trees_added
            );
            println!("  Mode: {:?}", report.mode);

            // Save update to TRB file
            let update_desc = if description.is_empty() {
                format!(
                    "Update: +{} trees from {} rows",
                    report.trees_added, report.rows_trained
                )
            } else {
                description
            };

            println!("\nSaving update to {:?}...", model);
            auto_model.save_trb_update(&model, report.rows_trained, &update_desc)?;

            println!("Update saved successfully.");
            println!(
                "  Model now has {} trees (was {})",
                report.trees_after, report.trees_before
            );

            Ok(())
        }
    }
}

/// Format a Unix timestamp as a human-readable date string
fn format_timestamp(timestamp: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp);
    // Simple ISO-8601-ish format without external crate
    let elapsed = datetime
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let secs = elapsed.as_secs();

    // Calculate date components (simplified, doesn't handle leap seconds perfectly)
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Calculate year/month/day (simplified algorithm)
    let mut year = 1970i32;
    let mut remaining_days = days_since_epoch as i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1;
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for days in days_in_months.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= *days;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}
