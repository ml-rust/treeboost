//! Trial execution and evaluation orchestration
//!
//! This module handles evaluating individual configurations and orchestrating
//! parallel/sequential candidate evaluation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use polars::prelude::DataFrame;
use rayon::prelude::*;

use crate::dataset::BinnedDataset;
use crate::tuner::logger::{log_trial, SharedLogger};
use crate::tuner::realistic::RealisticModeConfig;
use crate::tuner::traits::TunableModel;
use crate::tuner::trial::TrialResult;
use crate::Result;

use super::types::EvalInput;

/// Evaluate a single candidate configuration (unified for both modes)
///
/// Thread-safe: uses atomic operations for trial ID assignment.
/// Handles both optimistic (pre-binned) and realistic (per-split encoding) modes.
///
/// This is private (only used by evaluate_candidates and evaluate_candidates_internal).
fn evaluate_single<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    input: EvalInput<'_>,
    params: &HashMap<String, f32>,
    iteration: usize,
) -> Result<TrialResult> {
    let trial_id = tuner.next_trial_id_fetch_add(1);
    let start = Instant::now();

    // Dispatch to appropriate strategy based on input mode
    let eval_metrics = match input {
        EvalInput::Optimistic(dataset) => match tuner.eval_strategy() {
            crate::tuner::config::EvalStrategy::Holdout {
                validation_ratio,
                folds,
            } => super::eval_optimistic::evaluate_holdout_with_folds(
                tuner,
                dataset,
                params,
                *validation_ratio,
                *folds,
            )?,
            crate::tuner::config::EvalStrategy::Conformal {
                calibration_ratio,
                quantile,
                folds,
            } => super::eval_optimistic::evaluate_conformal_with_folds(
                tuner,
                dataset,
                params,
                *calibration_ratio,
                *quantile,
                *folds,
            )?,
        },
        EvalInput::Realistic { raw_data, config } => match tuner.eval_strategy() {
            crate::tuner::config::EvalStrategy::Holdout {
                validation_ratio,
                folds,
            } => super::eval_realistic::evaluate_holdout_realistic_with_folds(
                tuner,
                raw_data,
                config,
                params,
                *validation_ratio,
                *folds,
            )?,
            crate::tuner::config::EvalStrategy::Conformal {
                calibration_ratio,
                quantile,
                folds,
            } => super::eval_realistic::evaluate_conformal_realistic_with_folds(
                tuner,
                raw_data,
                config,
                params,
                *calibration_ratio,
                *quantile,
                *folds,
            )?,
        },
    };

    let train_time_ms = start.elapsed().as_millis() as u64;

    // Build full config and store params for CSV logging
    let _full_config = tuner.build_config(params);
    let mut full_params = params.clone();

    // Add learning_rate from base config if not being tuned
    // Note: Other fixed params are model-specific and not logged for generic models
    if !full_params.contains_key("learning_rate") {
        full_params.insert(
            "learning_rate".into(),
            M::get_learning_rate(tuner.base_config()),
        );
    }

    Ok(TrialResult {
        trial_id,
        iteration,
        params: full_params, // Store full params (tuned + fixed)
        val_loss: eval_metrics.val_metric,
        train_loss: eval_metrics.train_metric,
        num_trees: eval_metrics.num_trees,
        train_time_ms,
        f1_score: eval_metrics.f1_score,
        roc_auc: eval_metrics.roc_auc,
        rank_ic: eval_metrics.rank_ic,
    })
}

/// Evaluate candidates using parallel or sequential strategy
///
/// For CPU backends: Uses Rayon for parallel evaluation (faster tuning).
/// For GPU backends: Runs sequentially to avoid CUDA context contention.
/// GPU trials are fast individually (~1-2s), so sequential is acceptable.
///
/// If a logger is provided, results are written immediately after each trial.
pub(super) fn evaluate_candidates<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    dataset: &BinnedDataset,
    candidates: Vec<HashMap<String, f32>>,
    iteration: usize,
    current_trial: &AtomicUsize,
    total_trials: usize,
    logger: Option<&SharedLogger>,
) -> Vec<TrialResult> {
    let use_parallel = tuner.parallel_trials();
    let callback = tuner.callback();

    if use_parallel {
        let results = Mutex::new(Vec::with_capacity(candidates.len()));

        // Closure that evaluates candidates in parallel
        let eval_parallel = || {
            candidates.par_iter().for_each(|params| {
                match evaluate_single(tuner, EvalInput::Optimistic(dataset), params, iteration) {
                    Ok(result) => {
                        let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                        // Call callback (if set)
                        if let Some(ref cb) = callback {
                            cb(&result, trial_num, total_trials);
                        }

                        // Log immediately (streaming write with flush)
                        log_trial(logger, &result);

                        results.lock().unwrap().push(result);
                    }
                    Err(e) => {
                        eprintln!("Trial failed: {}", e);
                    }
                }
            });
        };

        // Use global pool for auto parallelism (n_parallel == 0), otherwise create custom pool
        if tuner.n_parallel() == 0 {
            // Use rayon's global pool directly (no pool creation overhead)
            eval_parallel();
        } else {
            // Create custom pool only when specific parallelism is requested
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(tuner.n_parallel())
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
            pool.install(eval_parallel);
        }

        results.into_inner().unwrap()
    } else {
        // Sequential evaluation (GPU backends or parallel disabled)
        let mut results = Vec::with_capacity(candidates.len());

        for params in candidates {
            match evaluate_single(tuner, EvalInput::Optimistic(dataset), &params, iteration) {
                Ok(result) => {
                    let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                    // Call callback
                    if let Some(ref callback) = callback {
                        callback(&result, trial_num, total_trials);
                    }

                    // Log immediately (streaming write with flush)
                    log_trial(logger, &result);

                    results.push(result);
                }
                Err(e) => {
                    eprintln!("Trial failed: {}", e);
                }
            }
        }

        results
    }
}

/// Evaluate candidates for realistic mode (encoding per split)
///
/// For realistic mode, we cannot parallelize because encoding is stateful.
// reason: kernel/training entry point with many parameters
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_candidates_internal<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    raw_data: &DataFrame,
    realistic_cfg: &RealisticModeConfig,
    candidates: Vec<HashMap<String, f32>>,
    iteration: usize,
    current_trial: &AtomicUsize,
    total_trials: usize,
    use_parallel: bool,
    logger: Option<&SharedLogger>,
) -> Result<Vec<TrialResult>> {
    // Realistic mode cannot be parallelized (encoding is stateful)
    if use_parallel {
        eprintln!(
            "Warning: Parallel mode not supported with realistic tuning, running sequentially"
        );
    }

    // Sequential evaluation
    let mut results = Vec::with_capacity(candidates.len());

    for params in candidates {
        let input = EvalInput::Realistic {
            raw_data,
            config: realistic_cfg,
        };
        match evaluate_single(tuner, input, &params, iteration) {
            Ok(result) => {
                let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                // Call callback
                if let Some(ref callback) = tuner.callback() {
                    callback(&result, trial_num, total_trials);
                }

                // Log immediately (streaming write with flush)
                log_trial(logger, &result);

                results.push(result);
            }
            Err(e) => {
                eprintln!("Trial failed: {}", e);
            }
        }
    }

    Ok(results)
}

/// Check if the backend requires sequential execution
///
/// GPU backends (WGPU, CUDA, ROCm, Metal) cannot run multiple contexts
/// concurrently on a single device, so trials must run sequentially.
/// CPU backends (Scalar, AVX-512, SVE2) can run in parallel.
pub(super) fn is_gpu_backend<M: TunableModel>(base_config: &M::Config) -> bool {
    M::is_gpu_config(base_config)
}

/// Randomize parameter centers to explore a different region
///
/// Called when stuck in a local optimum. Shifts each parameter's center
/// to a random position within its bounds.
pub(super) fn randomize_centers<M: TunableModel>(tuner: &mut crate::tuner::AutoTuner<M>) {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    // Use a seed derived from current iteration count for reproducibility
    let seed = tuner.seed().wrapping_add(tuner.history_len() as u64);
    let mut rng = StdRng::seed_from_u64(seed);

    for param in tuner.space_params_mut() {
        let (min, max) = (param.bounds.min_value(), param.bounds.max_value());

        let new_center = if param.bounds.is_log_scale() {
            // Log-uniform for log-scale parameters
            let log_min = min.max(1e-10).ln();
            let log_max = max.max(1e-10).ln();
            (log_min + rng.random::<f32>() * (log_max - log_min)).exp()
        } else {
            // Uniform for linear parameters
            min + rng.random::<f32>() * (max - min)
        };

        param.set_center(new_center);
    }
}
