//! Hyperparameter tuning methods for AutoBuilder
//!
//! This module contains all tuning logic for different boosting modes:
//! - Tree-based models (PureTree, RandomForest) via AutoTuner
//! - LinearThenTree via LttTuner
//! - Parameter extraction and configuration creation

use crate::analysis::{DataFrameProfile, TaskType};
use crate::booster::GBDTModel;
use crate::dataset::BinnedDataset;
use crate::learner::TreeConfig;
use crate::model::config::{AutoConfig, TreeTunerConfig, TreeTuningResult, TuningLevel};
use crate::model::{BoostingMode, UniversalConfig};
use crate::tuner::ltt::{LttTuner, LttTunerConfig, LttTuningResult};
use crate::tuner::{
    AutoTuner, EvalStrategy, GridStrategy, ParamBounds, ParameterSpace, SearchHistory,
    TaskType as TunerTaskType, TunerConfig, TuningMode,
};
use crate::{Result, TreeBoostError};
use polars::prelude::*;

/// Tune hyperparameters for the selected boosting mode
pub(super) fn tune_hyperparameters(
    config: &AutoConfig,
    dataset: &BinnedDataset,
    validation_dataset: Option<&BinnedDataset>,
    mode: BoostingMode,
    df: &DataFrame,
    target_col: &str,
    profile: &DataFrameProfile,
) -> Result<(
    UniversalConfig,
    Option<LttTuningResult>,
    Option<TreeTuningResult>,
)> {
    // If custom config provided, use it directly (overrides all tuning)
    if let Some(ref custom) = config.custom_config {
        return Ok((custom.clone(), None, None));
    }

    match config.tuning_level {
        TuningLevel::None => {
            // No tuning - use defaults
            // For LTT mode, still need to configure linear component with good defaults
            if matches!(mode, BoostingMode::LinearThenTree) {
                use crate::learner::{LinearConfig, TreeConfig};
                // Use stronger regularization for stability
                let linear_config = LinearConfig::default()
                    .with_preset(crate::learner::LinearPreset::Ridge)
                    .with_lambda(1.0)?
                    .with_shrinkage_factor(0.5)?
                    .with_max_iter(200)?;
                let tree_config = TreeConfig::default().with_max_depth(3)?;
                let univ_config = UniversalConfig::default()
                    .with_mode(mode)
                    .with_linear_config(linear_config)
                    .with_tree_config(tree_config)
                    .with_num_rounds(50)
                    .with_learning_rate(0.3)?;
                Ok((univ_config, None, None))
            } else {
                let univ_config = UniversalConfig::default().with_mode(mode);
                Ok((univ_config, None, None))
            }
        }
        _ => {
            // Tune based on mode
            match mode {
                BoostingMode::LinearThenTree => {
                    let (univ_config, ltt_result) = tune_ltt(config, df, target_col, profile)?;
                    Ok((univ_config, ltt_result, None))
                }
                BoostingMode::PureTree | BoostingMode::RandomForest => {
                    // Run proper AutoTuner for tree-based models
                    let (univ_config, tree_result) =
                        tune_tree_model(config, dataset, validation_dataset, mode, profile)?;
                    Ok((univ_config, None, tree_result))
                }
            }
        }
    }
}

/// Tune tree-based models (PureTree, RandomForest) using AutoTuner
fn tune_tree_model(
    config: &AutoConfig,
    dataset: &BinnedDataset,
    validation_dataset: Option<&BinnedDataset>,
    mode: BoostingMode,
    profile: &DataFrameProfile,
) -> Result<(UniversalConfig, Option<TreeTuningResult>)> {
    if config.verbose.enabled() {
        println!("  [Tuning] Running AutoTuner for {:?} mode...", mode);
    }

    // Get tuning configuration - use custom config if provided, otherwise use preset
    let mut tuner_cfg = if let Some(custom_cfg) = &config.tree_tuner_config {
        custom_cfg.clone()
    } else {
        match config.tuning_level {
            TuningLevel::Quick => {
                TreeTunerConfig::with_preset(crate::model::TreeTunerPreset::Quick)
            }
            TuningLevel::Standard => {
                TreeTunerConfig::with_preset(crate::model::TreeTunerPreset::Standard)
            }
            TuningLevel::Thorough => {
                TreeTunerConfig::with_preset(crate::model::TreeTunerPreset::Thorough)
            }
            TuningLevel::None => return Ok((UniversalConfig::default().with_mode(mode), None)),
        }
    };

    // Auto-configure tuning logs directory: {output_dir}/autotuner
    // Only set if model_output_dir is configured AND tuner_cfg doesn't already have explicit dir
    if let Some(ref output_dir) = config.model_output_dir {
        if tuner_cfg.output.dir.is_none() {
            tuner_cfg.output.dir = Some(output_dir.join("autotuner"));
        }
    }

    // Define parameter space from config
    use crate::tuner::TunableParam;
    let param_space = ParameterSpace::new()
        .with_param(
            TunableParam::MaxDepth,
            ParamBounds::discrete(tuner_cfg.depth.min, tuner_cfg.depth.max),
            6.0,
        )
        .with_param(
            TunableParam::LearningRate,
            ParamBounds::log_continuous(tuner_cfg.learning_rate.min, tuner_cfg.learning_rate.max),
            0.1,
        )
        .with_param(
            TunableParam::Subsample,
            ParamBounds::continuous(0.6, 1.0),
            0.8,
        )
        .with_param(TunableParam::Lambda, ParamBounds::continuous(0.1, 5.0), 1.0)
        .with_param(
            TunableParam::EntropyWeight,
            ParamBounds::continuous(0.0, 0.3),
            0.1,
        );

    // Resolve backend ONCE before tuning (don't re-resolve per trial)
    use crate::backend::{BackendConfig, BackendSelector, BackendType};
    let backend_type = if matches!(config.backend_type, BackendType::Auto) {
        let resolved = BackendSelector::with_config(BackendConfig {
            preferred: BackendType::Auto,
            ..Default::default()
        })
        .select(dataset.num_rows())?;

        // Use type-safe backend identification instead of string matching
        resolved.backend_type()
    } else {
        config.backend_type
    };

    // Base config for tuning - select loss function based on task type
    use crate::booster::GBDTConfig;
    let base_config = match &profile.task_type {
        TaskType::Regression => GBDTConfig::new()
            .with_mse_loss()
            .with_min_samples_leaf(5)
            .with_seed(42)
            .with_backend(backend_type),
        TaskType::BinaryClassification => GBDTConfig::new()
            .with_binary_logloss()
            .with_min_samples_leaf(5)
            .with_seed(42)
            .with_backend(backend_type),
        TaskType::MultiClassification { num_classes } => GBDTConfig::new()
            .with_multiclass_logloss(*num_classes)?
            .with_min_samples_leaf(5)
            .with_seed(42)
            .with_backend(backend_type),
    };

    // Determine task type:
    // 1. If user explicitly set task_type → use it (manual override, no detection)
    // 2. Else if optimization_metric is RankIc → force Regression (RankIc is regression-only)
    // 3. Else → use profile detection
    let tuner_task_type = if let Some(explicit_type) = tuner_cfg.task_type {
        explicit_type
    } else if tuner_cfg.output.metric == crate::tuner::OptimizationMetric::RankIc {
        TunerTaskType::Regression
    } else {
        match &profile.task_type {
            TaskType::Regression => TunerTaskType::Regression,
            TaskType::BinaryClassification => TunerTaskType::BinaryClassification,
            TaskType::MultiClassification { .. } => TunerTaskType::MultiClassClassification,
        }
    };

    if config.verbose.enabled() {
        println!(
            "  [Tuning] Task type: {:?}, Optimization metric: {:?}",
            tuner_task_type, tuner_cfg.output.metric
        );
    }

    // Configure tuner using TreeTunerConfig
    // When custom validation is provided, eval_strategy and early_stopping validation_ratio won't be used
    // (tune_with_validation() bypasses internal splitting), but we still need valid values to pass config validation
    let effective_val_ratio = if validation_dataset.is_some() {
        0.2 // Dummy value (won't be used, but must be non-zero for config validation)
    } else {
        tuner_cfg.stopping.validation_ratio
    };

    let mut tuner_config = TunerConfig::new()
        .with_iterations(tuner_cfg.search.n_iterations)
        .with_grid_strategy(GridStrategy::LatinHypercube {
            n_samples: tuner_cfg.search.n_samples,
        })
        .with_eval_strategy(EvalStrategy::conformal_90(effective_val_ratio))
        .with_tuning_mode(TuningMode::Optimistic) // Use pre-encoded data
        .with_num_rounds(tuner_cfg.max_rounds)
        .with_early_stopping(
            tuner_cfg.stopping.early_stopping_rounds,
            effective_val_ratio,
        )
        .with_improvement_threshold(tuner_cfg.stopping.improvement_threshold)
        .with_min_f1_score(tuner_cfg.stopping.min_f1_score)
        .with_optimization_metric(tuner_cfg.output.metric)
        .with_task_type(tuner_task_type)
        .with_verbose(false); // Quiet internal logging

    // Enable CSV logging if output_dir is specified
    if let Some(ref dir) = tuner_cfg.output.dir {
        tuner_config = tuner_config.with_output_dir(dir);
    }

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(param_space)
        .with_seed(123);

    // Run tuning (use custom validation if provided)
    let (best_gbdt_config, history): (GBDTConfig, SearchHistory) =
        if let Some(val_dataset) = validation_dataset {
            tuner.tune_with_validation(dataset, val_dataset)?
        } else {
            tuner.tune(dataset)?
        };

    // Convert GBDT config to UniversalConfig
    let tree_config = TreeConfig::default()
        .with_max_depth(best_gbdt_config.max_depth)?
        .with_lambda(best_gbdt_config.lambda)?
        .with_entropy_weight(best_gbdt_config.entropy_weight);

    let universal_config = UniversalConfig::default()
        .with_mode(mode)
        .with_num_rounds(best_gbdt_config.num_rounds)
        .with_learning_rate(best_gbdt_config.learning_rate)?
        .with_subsample(best_gbdt_config.subsample)?
        .with_tree_config(tree_config)
        .with_backend(best_gbdt_config.backend_type); // Preserve backend type from tuning!

    // Extract tuning results
    let tuning_result = history.best().map(|best| {
        // Use the actual optimization metric value
        let best_metric = match tuner_cfg.output.metric {
            crate::tuner::OptimizationMetric::ValidationLoss => best.val_loss,
            crate::tuner::OptimizationMetric::F1Score => best.f1_score.unwrap_or(0.0),
            crate::tuner::OptimizationMetric::RocAuc => best.roc_auc.unwrap_or(0.0) as f32,
            crate::tuner::OptimizationMetric::RankIc => best.rank_ic.unwrap_or(0.0) as f32,
        };
        TreeTuningResult {
            num_trials: history.len(),
            best_metric,
            best_params: best.params.clone(),
        }
    });

    if config.verbose.enabled() {
        if let Some(best) = history.best() {
            // Show the optimization metric value with clear labeling
            match tuner_cfg.output.metric {
                crate::tuner::OptimizationMetric::ValidationLoss => {
                    println!("  [Tuning] Best loss (MSE): {:.6}", best.val_loss);
                }
                crate::tuner::OptimizationMetric::F1Score => {
                    if let Some(f1) = best.f1_score {
                        println!("  [Tuning] Best F1 score: {:.4}", f1);
                    }
                }
                crate::tuner::OptimizationMetric::RocAuc => {
                    if let Some(auc) = best.roc_auc {
                        println!("  [Tuning] Best ROC-AUC: {:.6}", auc);
                    }
                }
                crate::tuner::OptimizationMetric::RankIc => {
                    if let Some(ic) = best.rank_ic {
                        println!("  [Tuning] Best Rank IC: {:.6}", ic);
                    }
                    println!("  [Tuning] (val_loss for reference: {:.6})", best.val_loss);
                }
            }
            println!("  [Tuning] Best params: {:?}", best.params);
        }
    }

    Ok((universal_config, tuning_result))
}

/// Tune LinearThenTree mode
fn tune_ltt(
    config: &AutoConfig,
    df: &DataFrame,
    target_col: &str,
    profile: &DataFrameProfile,
) -> Result<(UniversalConfig, Option<LttTuningResult>)> {
    // Extract raw features for linear tuning
    let (features, targets, num_features) = extract_raw_features(df, target_col, profile)?;

    // Compute linear_feature_indices by identifying ENGINEERED features
    // Linear model uses: polynomial features (_squared, _sqrt, _log) + interaction features (_x_)
    // Tree model uses: original features (both numeric and categorical-encoded)
    // This is different from filtering by column TYPE - we filter by feature ORIGIN
    let col_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut linear_feature_indices = Vec::new();

    // Build indices relative to EXTRACTED features (target excluded)
    let mut feature_idx = 0;
    for col_name in &col_names {
        if col_name == target_col {
            continue; // Skip target
        }

        // Identify engineered features by naming convention
        let is_engineered = col_name.ends_with("_squared")
            || col_name.ends_with("_sqrt")
            || col_name.ends_with("_log")
            || col_name.ends_with("_log1p")
            || col_name.contains("_x_") // Interaction features
            || col_name.contains("_ratio_"); // Ratio features

        if is_engineered {
            // Engineered features go to linear model (polynomial/interaction)
            linear_feature_indices.push(feature_idx);
        }

        feature_idx += 1; // Increment for each non-target feature
    }

    // Create tuner config based on tuning level
    let mut tuner_config = match config.tuning_level {
        TuningLevel::Quick => {
            LttTunerConfig::default().with_preset(crate::tuner::ltt::LttTunerPreset::Quick)
        }
        TuningLevel::Standard => LttTunerConfig::default(),
        TuningLevel::Thorough => {
            LttTunerConfig::default().with_preset(crate::tuner::ltt::LttTunerPreset::Thorough)
        }
        TuningLevel::None => {
            LttTunerConfig::default().with_preset(crate::tuner::ltt::LttTunerPreset::Quick)
        } // Should not reach here
    };

    // Set output directory for tuning logs
    if let Some(ref output_dir) = config.model_output_dir {
        tuner_config = tuner_config.with_output_dir(output_dir.clone());
    }

    let tuner = LttTuner::new(tuner_config);
    let result = tuner.tune(&features, num_features, &targets, &linear_feature_indices)?;

    // Build UniversalConfig from tuning result
    // Apply ALL tuned tree parameters (not just max_depth!)
    let tree_config = TreeConfig::default()
        .with_max_depth(result.tree_params.max_depth as usize)?
        .with_min_hessian_leaf(result.tree_params.min_child_weight)?
        .with_colsample(result.tree_params.colsample_bytree)?;

    let univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_learning_rate(result.tree_params.learning_rate)?
        .with_num_rounds(result.tree_params.num_rounds as usize)
        .with_subsample(result.tree_params.subsample)?
        .with_tree_config(tree_config)
        .with_linear_config(result.linear_params.to_config())
        .with_linear_feature_indices(linear_feature_indices); // Store for training and prediction

    Ok((univ_config, Some(result)))
}

/// Extract raw features from DataFrame for linear model tuning
///
/// NOTE: This is a simplified extraction for LTT hyperparameter tuning.
/// - Only numeric features are used (categorical features need encoding first)
/// - Columns marked for dropping in the profile are excluded
/// - For production prediction, use the full DataPipeline with preprocessing
fn extract_raw_features(
    df: &DataFrame,
    target_col: &str,
    _profile: &DataFrameProfile,
) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let num_rows = df.height();

    // Use FeatureExtractor to intelligently select numeric features for linear models
    // This properly handles ID-like columns, categoricals, booleans, etc.
    use crate::dataset::feature_extractor::FeatureExtractor;
    let extractor = FeatureExtractor::new();
    let (features, num_features) = extractor.extract(df, target_col)?;

    if num_features == 0 {
        return Err(TreeBoostError::Data(format!(
            "No numeric features found for linear model tuning. \
                 Dataset has {} columns. \
                 Categorical features need encoding first - consider using DataPipeline.",
            df.width(),
        )));
    }

    // Extract targets
    let target_col_series = df.column(target_col)?;
    let targets: Vec<f32> = (0..num_rows)
        .map(|i| match target_col_series.get(i) {
            Ok(val) => match val {
                AnyValue::Float32(v) => v,
                AnyValue::Float64(v) => v as f32,
                AnyValue::Int32(v) => v as f32,
                AnyValue::Int64(v) => v as f32,
                _ => 0.0,
            },
            Err(_) => 0.0,
        })
        .collect();

    Ok((features, targets, num_features))
}

/// Create default config for non-LTT modes (fallback when tuning is disabled)
pub(super) fn create_config_for_mode(
    mode: BoostingMode,
    tuning_level: TuningLevel,
) -> UniversalConfig {
    let (num_rounds, learning_rate, max_depth) = match tuning_level {
        TuningLevel::Quick => (100, 0.1, 4),
        TuningLevel::Standard => (500, 0.1, 6),
        TuningLevel::Thorough => (1000, 0.05, 8),
        TuningLevel::None => (100, 0.1, 6),
    };

    let tree_config = TreeConfig::default()
        .with_max_depth(max_depth)
        .expect("hardcoded max_depth values are valid");

    UniversalConfig::default()
        .with_mode(mode)
        .with_num_rounds(num_rounds)
        .with_learning_rate(learning_rate)
        .expect("hardcoded learning_rate values are valid")
        .with_tree_config(tree_config)
}
