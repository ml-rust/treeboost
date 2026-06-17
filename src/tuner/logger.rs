//! Trial logging for hyperparameter tuning
//!
//! Provides streaming CSV output and result export functionality.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::{Result, TreeBoostError};

use super::config::ModelFormat;
use super::history::SearchHistory;
use super::trial::TrialResult;

// =============================================================================
// TrialLogger
// =============================================================================

/// Logger for streaming trial results to CSV files
///
/// Writes each trial immediately after evaluation with flush,
/// so partial results are preserved if tuning is interrupted.
pub(crate) struct TrialLogger {
    run_dir: PathBuf,
    writer: Option<csv::Writer<File>>,
    param_names: Vec<String>,
    /// Extra fixed params to log (learning_rate, num_rounds, etc.)
    extra_param_names: Vec<String>,
}

impl TrialLogger {
    /// Create a new trial logger with timestamped run directory
    pub fn new(output_dir: &Path, param_names: Vec<String>) -> Result<Self> {
        // Create timestamped run directory with milliseconds to avoid collisions
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S%.3f");
        let run_dir = output_dir.join(format!("run_{}", timestamp));
        fs::create_dir_all(&run_dir)
            .map_err(|e| TreeBoostError::Data(format!("Failed to create run directory: {}", e)))?;

        // Fixed params that are always logged (not being tuned)
        let extra_param_names = vec![
            "learning_rate".to_string(),
            "num_rounds".to_string(),
            "colsample".to_string(),
            "min_samples_leaf".to_string(),
            "min_gain".to_string(),
        ];

        Ok(Self {
            run_dir,
            writer: None,
            param_names,
            extra_param_names,
        })
    }

    /// Start a new iteration (creates new CSV file)
    pub fn start_iteration(&mut self, iteration: usize) -> Result<()> {
        // Close previous writer with proper error logging
        self.close()?;

        // Create new CSV file for this iteration
        let path = self
            .run_dir
            .join(format!("iteration_{}.csv", iteration + 1));
        let mut writer = csv::Writer::from_path(&path)
            .map_err(|e| TreeBoostError::Data(format!("Failed to create CSV: {}", e)))?;

        // Write header row using TrialResult's headers
        let mut headers: Vec<String> = TrialResult::csv_headers()
            .iter()
            .map(|s| s.to_string())
            .collect();
        headers.extend(self.param_names.clone());

        // Only add extra params that aren't already in param_names (avoid duplicates)
        let param_names_set: std::collections::HashSet<_> = self.param_names.iter().collect();
        for extra_name in &self.extra_param_names {
            if !param_names_set.contains(extra_name) {
                headers.push(extra_name.clone());
            }
        }

        writer
            .write_record(&headers)
            .map_err(|e| TreeBoostError::Data(format!("Failed to write CSV header: {}", e)))?;
        writer
            .flush()
            .map_err(|e| TreeBoostError::Data(format!("Failed to flush CSV: {}", e)))?;

        self.writer = Some(writer);
        Ok(())
    }

    /// Log a single trial result (immediately flushed)
    pub fn log_trial(&mut self, trial: &TrialResult) -> Result<()> {
        if let Some(ref mut writer) = self.writer {
            // Use TrialResult's CSV conversion
            let mut row = trial.to_csv_row();

            // Add tuned param values in consistent order
            for name in &self.param_names {
                row.push(trial.param_to_csv(name));
            }

            // Add fixed param values that aren't already in param_names (avoid duplicates)
            let param_names_set: std::collections::HashSet<_> = self.param_names.iter().collect();
            for name in &self.extra_param_names {
                if !param_names_set.contains(name) {
                    row.push(trial.param_to_csv(name));
                }
            }

            writer
                .write_record(&row)
                .map_err(|e| TreeBoostError::Data(format!("Failed to write CSV row: {}", e)))?;
            writer
                .flush()
                .map_err(|e| TreeBoostError::Data(format!("Failed to flush CSV: {}", e)))?;
        }
        Ok(())
    }

    /// Close the current CSV writer, flushing any remaining data
    pub fn close(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.writer.take() {
            writer
                .flush()
                .map_err(|e| TreeBoostError::Data(format!("Failed to flush final CSV: {}", e)))?;
        }
        Ok(())
    }

    /// Get the run directory path
    pub fn run_dir(&self) -> &PathBuf {
        &self.run_dir
    }

    /// Export best params to JSON file
    pub fn export_best_params(&self, best: &TrialResult) -> Result<()> {
        let path = self.run_dir.join("best_params.json");
        let json = serde_json::json!({
            "trial_id": best.trial_id,
            "iteration": best.iteration,
            "val_loss": best.val_loss,
            "train_loss": best.train_loss,
            "f1_score": best.f1_score,
            "roc_auc": best.roc_auc,
            "rank_ic": best.rank_ic,
            "num_trees": best.num_trees,
            "params": best.params,
        });
        let file = File::create(&path)
            .map_err(|e| TreeBoostError::Data(format!("Failed to create params file: {}", e)))?;
        serde_json::to_writer_pretty(file, &json)
            .map_err(|e| TreeBoostError::Data(format!("Failed to write params JSON: {}", e)))?;
        Ok(())
    }

    /// Export run summary to JSON file
    pub fn export_summary(&self, history: &SearchHistory, duration_secs: f64) -> Result<()> {
        let path = self.run_dir.join("summary.json");
        let best = history.best();
        let json = serde_json::json!({
            "total_trials": history.len(),
            "duration_secs": duration_secs,
            "best_trial_id": best.map(|b| b.trial_id),
            "best_val_loss": best.map(|b| b.val_loss),
            "best_train_loss": best.map(|b| b.train_loss),
            "best_f1_score": best.and_then(|b| b.f1_score),
            "best_roc_auc": best.and_then(|b| b.roc_auc),
            "best_rank_ic": best.and_then(|b| b.rank_ic),
            "optimization_metric": format!("{:?}", history.optimization_metric()),
        });
        let file = File::create(&path)
            .map_err(|e| TreeBoostError::Data(format!("Failed to create summary file: {}", e)))?;
        serde_json::to_writer_pretty(file, &json)
            .map_err(|e| TreeBoostError::Data(format!("Failed to write summary JSON: {}", e)))?;
        Ok(())
    }

    /// Save the best model in the specified format
    pub fn save_model<M: super::TunableModel>(&self, model: &M, format: ModelFormat) -> Result<()> {
        let path = self.run_dir.join(format.filename());
        match format {
            ModelFormat::Rkyv => model.save_rkyv(&path),
        }
    }

    /// Save the best model in all specified formats
    pub fn save_models<M: super::TunableModel>(
        &self,
        model: &M,
        formats: &[ModelFormat],
    ) -> Result<()> {
        for format in formats {
            self.save_model(model, *format)?;
        }
        Ok(())
    }
}

impl Drop for TrialLogger {
    fn drop(&mut self) {
        // Best effort close on drop
        let _ = self.close();
    }
}

// =============================================================================
// Shared Logger Type and Helper Functions
// =============================================================================

/// Thread-safe logger wrapper for parallel evaluation
pub(crate) type SharedLogger = Arc<Mutex<TrialLogger>>;

/// Initialize a trial logger if output_dir is configured
pub(crate) fn init_logger(
    output_dir: &Option<PathBuf>,
    param_names: Vec<String>,
    verbose: bool,
) -> Result<Option<SharedLogger>> {
    if let Some(ref dir) = output_dir {
        if verbose {
            println!("  Logging to: {}", dir.display());
        }
        Ok(Some(Arc::new(Mutex::new(TrialLogger::new(
            dir,
            param_names,
        )?))))
    } else {
        Ok(None)
    }
}

/// Start a new iteration if logging is enabled
pub(crate) fn start_iteration_logging(
    logger: &Option<SharedLogger>,
    iteration: usize,
) -> Result<()> {
    if let Some(ref l) = logger {
        l.lock()
            .map_err(|e| TreeBoostError::Data(format!("Failed to lock logger: {}", e)))?
            .start_iteration(iteration)?;
    }
    Ok(())
}

/// Log a trial result if logging is enabled (with proper error reporting)
pub(crate) fn log_trial(logger: Option<&SharedLogger>, trial: &TrialResult) {
    if let Some(l) = logger {
        match l.lock() {
            Ok(mut guard) => {
                if let Err(e) = guard.log_trial(trial) {
                    eprintln!("Warning: Failed to log trial {}: {}", trial.trial_id, e);
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to lock logger for trial {}: {}",
                    trial.trial_id, e
                );
            }
        }
    }
}

/// Export final results and optionally save model
pub(crate) fn finalize_logging(
    logger: &Option<SharedLogger>,
    history: &SearchHistory,
    best: &TrialResult,
    duration_secs: f64,
) -> Result<PathBuf> {
    let l = logger
        .as_ref()
        .ok_or_else(|| TreeBoostError::Data("Logger not initialized".into()))?;

    let guard = l
        .lock()
        .map_err(|e| TreeBoostError::Data(format!("Failed to lock logger: {}", e)))?;

    guard.export_best_params(best)?;
    guard.export_summary(history, duration_secs)?;

    Ok(guard.run_dir().clone())
}

/// Save model in specified formats
pub(crate) fn save_model_formats<M: super::TunableModel>(
    logger: &Option<SharedLogger>,
    model: &M,
    formats: &[ModelFormat],
) -> Result<()> {
    if let Some(ref l) = logger {
        let guard = l
            .lock()
            .map_err(|e| TreeBoostError::Data(format!("Failed to lock logger: {}", e)))?;
        guard.save_models(model, formats)?;
    }
    Ok(())
}
