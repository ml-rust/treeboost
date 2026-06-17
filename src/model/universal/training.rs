//! Training methods for UniversalModel
//!
//! This module contains all training-related methods:
//! - train() and variants (raw_features, linear_feature_selection)
//! - auto() and variants (auto_with_config, auto_with_analysis_config)
//! - train_with_selection()
//! - train_pure_tree(), train_linear_then_tree(), train_random_forest()

use crate::analysis::{AnalysisConfig, DatasetAnalysis};
use crate::booster::GBDTModel;
use crate::dataset::BinnedDataset;
use crate::features::extract_selected_features;
use crate::learner::{LinearBooster, TreeBooster, WeakLearner};
use crate::loss::LossFunction;
use crate::tree::Tree;
use crate::Result;
use rand::SeedableRng;
use rayon::prelude::*;

use super::{BoostingMode, ModeSelection, UniversalConfig, UniversalModel};

impl UniversalModel {
    /// Validate common training configuration parameters
    ///
    /// This is called by all train() variants to ensure consistent validation.
    fn validate_config(config: &UniversalConfig) -> Result<()> {
        if config.num_rounds == 0 {
            return Err(crate::TreeBoostError::Config(
                "UniversalModel configuration error: num_rounds must be greater than 0, got 0. \
                 Specify at least 1 boosting round."
                    .to_string(),
            ));
        }
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "UniversalModel configuration error: learning_rate must be in (0, 1], got {}. \
                 Specify a value greater than 0 and at most 1.0.",
                config.learning_rate
            )));
        }
        if config.subsample <= 0.0 || config.subsample > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "UniversalModel configuration error: subsample must be in (0, 1], got {}. \
                 Specify a value greater than 0 and at most 1.0.",
                config.subsample
            )));
        }
        Ok(())
    }
    /// Train a UniversalModel on binned data
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `config` - Training configuration (fully serializable and persisted)
    /// * `loss_fn` - Loss function for computing gradients during training
    ///
    /// # Important Notes
    ///
    /// - The **loss function is NOT persisted** in the saved model. It is used only
    ///   during training to compute gradients. The trained model is loss-function-agnostic
    ///   and will work correctly with any data processed with the same preprocessing.
    /// - The **config is fully persisted** and can be exported via [`UniversalModel::config()`] or
    ///   saved to JSON for inspection and reuse.
    /// - For reproducibility, store your loss function choice separately if needed.
    ///
    /// # Example
    /// ```ignore
    /// // Train with MSE loss
    /// let model = UniversalModel::train(&dataset, config, &MseLoss)?;
    ///
    /// // Save config and model
    /// let config_json = serde_json::to_string_pretty(model.config())?;
    /// std::fs::write("config.json", config_json)?;
    /// model.save("model.rkyv")?;
    ///
    /// // Later: Load and use (loss function is already baked into the model)
    /// let loaded = UniversalModel::load("model.rkyv")?;
    /// let preds = loaded.predict(&test_dataset)?;  // No need to specify loss again
    /// ```
    pub fn train(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        Self::validate_config(&config)?;

        let feature_extractor = config.feature_extractor.clone();

        // Extract linear_feature_indices from config for LinearThenTree mode
        let linear_indices_opt = config.linear_feature_indices.clone();

        // Extract raw features from dataset if available (for LinearThenTree mode)
        let raw_features_opt = if config.mode == BoostingMode::LinearThenTree {
            dataset.raw_features().map(|r| r.to_vec())
        } else {
            None
        };

        match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, None, feature_extractor)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                raw_features_opt.as_deref(), // Use raw features from dataset if available
                linear_indices_opt.as_deref(), // Pass linear_feature_indices from config
                config,
                loss_fn,
                None,
                feature_extractor,
                None,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, None, feature_extractor)
            }
        }
    }

    /// Train with presplit validation data for early stopping.
    ///
    /// Uses the provided validation dataset for early stopping instead of
    /// creating a random internal split. This is critical for time-series data
    /// where random splits cause temporal leakage.
    pub fn train_with_validation(
        dataset: &BinnedDataset,
        val_dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        Self::validate_config(&config)?;
        let feature_extractor = config.feature_extractor.clone();

        match config.mode {
            BoostingMode::PureTree => Self::train_pure_tree_with_validation(
                dataset,
                val_dataset,
                config,
                loss_fn,
                None,
                feature_extractor,
            ),
            BoostingMode::LinearThenTree => {
                let feature_extractor = config.feature_extractor.clone();
                let linear_indices_opt = config.linear_feature_indices.clone();
                let raw_features_opt = dataset.raw_features().map(|r| r.to_vec());
                Self::train_linear_then_tree(
                    dataset,
                    raw_features_opt.as_deref(),
                    linear_indices_opt.as_deref(),
                    config,
                    loss_fn,
                    None,
                    feature_extractor,
                    Some(val_dataset),
                )
            }
            // For other modes (e.g. RandomForest), validation data is not supported yet
            _ => {
                eprintln!(
                    "Warning: train_with_validation called with {:?} mode, \
                     but validation data is only used for PureTree and LinearThenTree. \
                     Falling back to standard training (validation data ignored).",
                    config.mode
                );
                Self::train(dataset, config, loss_fn)
            }
        }
    }

    pub(super) fn train_pure_tree_with_validation(
        dataset: &BinnedDataset,
        val_dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
    ) -> Result<Self> {
        let num_features = dataset.num_features();

        let (gbdt_model, gbdt_ensemble, stacker_weights, stacker_intercept) =
            if let Some(ref seeds) = config.ensemble_seeds {
                Self::train_gbdt_ensemble(dataset, &config, loss_fn, seeds)?
            } else {
                let mut gbdt_config = Self::to_gbdt_config(&config, loss_fn)?;
                // For presplit validation, set early_stopping_rounds directly
                // (to_gbdt_config skips it when validation_ratio=0)
                if config.early_stopping_rounds > 0 {
                    gbdt_config.early_stopping_rounds = config.early_stopping_rounds;
                } else {
                    // Default to 50 if user didn't set it
                    gbdt_config.early_stopping_rounds = 50;
                }
                let gbdt_model = GBDTModel::train_binned_with_validation(
                    dataset,
                    val_dataset,
                    &[],
                    gbdt_config,
                )?;
                (Some(gbdt_model), None, None, None)
            };

        Ok(Self {
            config,
            gbdt_model,
            gbdt_ensemble,
            stacker_weights,
            stacker_intercept,
            linear_booster: None,
            linear_boosters: None,
            gbdt_per_label: None,
            trees: Vec::new(),
            base_prediction: 0.0,
            base_predictions_multi: None,
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
            feature_extractor,
        })
    }

    /// Train LinearThenTree with raw features (recommended for best accuracy)
    ///
    /// For LinearThenTree mode, passing raw (unbinned) features significantly improves
    /// the linear model's accuracy. Without raw features, LTT uses bin-center approximations
    /// which lose precision that linear models need.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (for tree training)
    /// * `raw_features` - Original features, row-major f32 array (num_rows * num_features)
    /// * `config` - Training configuration
    /// * `loss_fn` - Loss function
    ///
    /// # Example
    /// ```ignore
    /// let model = UniversalModel::train_with_raw_features(
    ///     &binned_dataset,
    ///     &scaled_features,  // Original StandardScaler'd features
    ///     config,
    ///     &MseLoss,
    /// )?;
    /// ```
    pub fn train_with_raw_features(
        dataset: &BinnedDataset,
        raw_features: &[f32],
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        let feature_extractor = config.feature_extractor.clone();
        match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, None, feature_extractor)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                Some(raw_features),
                None,
                config,
                loss_fn,
                None,
                feature_extractor,
                None,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, None, feature_extractor)
            }
        }
    }

    /// Train LinearThenTree with feature selection for linear model
    ///
    /// This allows using a curated subset of features for the linear model
    /// while trees use all features. This can improve linear generalization
    /// by excluding meaningless features (like row IDs) from linear.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (for tree training with all features)
    /// * `raw_features` - All features, row-major f32 array
    /// * `linear_feature_indices` - Which feature indices to use for linear model
    /// * `config` - Training configuration
    /// * `loss_fn` - Loss function
    pub fn train_with_linear_feature_selection(
        dataset: &BinnedDataset,
        raw_features: &[f32],
        linear_feature_indices: &[usize],
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        validation_dataset: Option<&BinnedDataset>,
    ) -> Result<Self> {
        Self::validate_config(&config)?;

        let feature_extractor = config.feature_extractor.clone();
        match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, None, feature_extractor)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                Some(raw_features),
                Some(linear_feature_indices),
                config,
                loss_fn,
                None,
                feature_extractor,
                validation_dataset,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, None, feature_extractor)
            }
        }
    }

    // =========================================================================
    // Automatic Mode Selection
    // =========================================================================

    /// Train with automatic mode selection
    ///
    /// This is TreeBoost's "smart" entry point. It:
    /// 1. Analyzes your dataset (lightweight probes on subsamples)
    /// 2. Picks the best boosting mode with confidence score
    /// 3. Trains the model with optimal settings
    /// 4. Stores the analysis for inspection
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::{UniversalModel, MseLoss};
    ///
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// // See what mode was selected and why
    /// println!("Mode: {:?}", model.mode());
    /// println!("Confidence: {:?}", model.selection_confidence());
    /// println!("{}", model.analysis_report().unwrap());
    /// ```
    ///
    /// # When to Use
    ///
    /// Use `auto()` when:
    /// - You're not sure which mode is best for your data
    /// - You want TreeBoost to explain its decision
    /// - You want a simple one-liner that "just works"
    ///
    /// Use `train()` when:
    /// - You know the best mode for your data
    /// - You need fine-grained control over configuration
    /// - You're running benchmarks and want deterministic mode
    pub fn auto(dataset: &BinnedDataset, loss_fn: &dyn LossFunction) -> Result<Self> {
        Self::auto_with_config(dataset, UniversalConfig::default(), loss_fn)
    }

    /// Train with automatic mode selection and custom configuration
    ///
    /// Like `auto()`, but lets you customize other settings (num_rounds, tree config, etc.).
    /// The mode will be overridden by the analysis recommendation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_num_rounds(200)
    ///     .with_learning_rate(0.05);
    ///
    /// let model = UniversalModel::auto_with_config(&dataset, config, &MseLoss)?;
    /// ```
    pub fn auto_with_config(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        Self::auto_with_analysis_config(dataset, config, AnalysisConfig::default(), loss_fn)
    }

    /// Train with automatic mode selection and custom analysis configuration
    ///
    /// Full control over both model config and analysis settings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UniversalConfig::new().with_num_rounds(200);
    /// let analysis_config = AnalysisConfig::fast(); // Quick analysis
    ///
    /// let model = UniversalModel::auto_with_analysis_config(
    ///     &dataset, config, analysis_config, &MseLoss
    /// )?;
    /// ```
    pub fn auto_with_analysis_config(
        dataset: &BinnedDataset,
        mut config: UniversalConfig,
        analysis_config: AnalysisConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        // Step 1: Analyze the dataset
        let analysis = DatasetAnalysis::analyze_with_config(dataset, analysis_config)?;

        // Step 2: Get the recommended mode
        let recommended_mode = analysis.recommend_mode();

        // Step 3: Update config with recommended mode
        config.mode = recommended_mode;

        // Step 4: Train with the recommended mode
        let model = match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, Some(analysis), None)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                None,
                None,
                config,
                loss_fn,
                Some(analysis),
                None,
                None,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, Some(analysis), None)
            }
        }?;

        Ok(model)
    }

    /// Train using a ModeSelection strategy
    ///
    /// This is the most flexible entry point, supporting:
    /// - `ModeSelection::Auto` - Automatic analysis and selection
    /// - `ModeSelection::AutoWithConfig(config)` - Auto with custom analysis
    /// - `ModeSelection::Fixed(mode)` - Explicit mode specification
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Auto mode
    /// let model = UniversalModel::train_with_selection(
    ///     &dataset, config, ModeSelection::Auto, &MseLoss
    /// )?;
    ///
    /// // Fixed mode
    /// let model = UniversalModel::train_with_selection(
    ///     &dataset, config, ModeSelection::Fixed(BoostingMode::LinearThenTree), &MseLoss
    /// )?;
    /// ```
    pub fn train_with_selection(
        dataset: &BinnedDataset,
        mut config: UniversalConfig,
        selection: ModeSelection,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        match selection {
            ModeSelection::Auto => Self::auto_with_config(dataset, config, loss_fn),
            ModeSelection::AutoWithConfig(analysis_config) => {
                Self::auto_with_analysis_config(dataset, config, analysis_config, loss_fn)
            }
            ModeSelection::Fixed(mode) => {
                config.mode = mode;
                Self::train(dataset, config, loss_fn)
            }
        }
    }

    // =========================================================================
    // PureTree Mode - Delegates to GBDTModel
    // =========================================================================

    pub(super) fn train_pure_tree(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
    ) -> Result<Self> {
        let num_features = dataset.num_features();

        // Check if ensemble training is requested
        let (gbdt_model, gbdt_ensemble, stacker_weights, stacker_intercept) =
            if let Some(ref seeds) = config.ensemble_seeds {
                // Multi-seed ensemble training
                Self::train_gbdt_ensemble(dataset, &config, loss_fn, seeds)?
            } else {
                // Single GBDT training (standard path)
                let gbdt_config = Self::to_gbdt_config(&config, loss_fn)?;
                let gbdt_model = GBDTModel::train_binned(dataset, gbdt_config)?;
                (Some(gbdt_model), None, None, None)
            };

        Ok(Self {
            config,
            gbdt_model,
            gbdt_ensemble,
            stacker_weights,
            stacker_intercept,
            linear_booster: None,
            linear_boosters: None,
            gbdt_per_label: None,
            trees: Vec::new(),
            base_prediction: 0.0, // Not used - GBDTModel handles this
            base_predictions_multi: None,
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
            feature_extractor,
        })
    }

    // =========================================================================
    // LinearThenTree Mode - Linear phase + GBDTModel on residuals
    // =========================================================================
    // Uses Newton-step Coordinate Descent for the linear phase (gradient/hessian).
    // This provides implicit regularization via learning_rate and captures global
    // trends that trees cannot extrapolate.
    //
    // NOTE ON INTENTIONAL CODE DUPLICATION:
    // The linear training loop in this method (Phase 1) is duplicated
    // in `AutoBuilder::train_ltt_ensemble()` (src/model/builder.rs). This duplication
    // is intentional and acceptable because:
    // - This method is part of UniversalModel (core training API)
    // - AutoBuilder::train_ltt_ensemble is high-level AutoML orchestration
    // - Extracting shared logic would require complex trait/closure patterns
    // - The duplication is ~25 lines and stable (rarely changes)
    // - Clear architectural separation outweighs DRY principle
    //
    // See: AutoBuilder::train_ltt_ensemble() in src/model/builder.rs for the other copy.

    // reason: kernel/training entry point with many parameters
    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_linear_then_tree(
        dataset: &BinnedDataset,
        raw_features_opt: Option<&[f32]>,
        linear_feature_indices_opt: Option<&[usize]>,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
        validation_dataset: Option<&BinnedDataset>,
    ) -> Result<Self> {
        let raw_targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Apply target transform if specified (critical for LinearThenTree)
        // Linear model must be trained in the same space as GBDT (logit space for bounded targets)
        let mut transformed_targets = raw_targets.to_vec();
        if let Some(ref pipeline) = config.pipeline {
            if let Some(transform) = pipeline.target_transform() {
                use crate::preprocessing::TargetTransform;
                transform.transform(&mut transformed_targets)?;
            }
        }
        let targets = &transformed_targets;

        // Determine which features to use for linear model
        let linear_indices: Option<Vec<usize>> = linear_feature_indices_opt.map(|v| v.to_vec());

        // Get raw features: use provided ones or extract from bins (lossy fallback)
        let raw_features: Vec<f32> = if let Some(provided) = raw_features_opt {
            provided.to_vec()
        } else {
            // Memory safety check for bin-center extraction fallback
            let estimated_bytes = config.estimate_linear_memory(num_rows, num_features);
            let estimated_mb = estimated_bytes / (1024 * 1024);

            if config.max_linear_memory_mb > 0 && estimated_mb > config.max_linear_memory_mb {
                return Err(crate::TreeBoostError::Config(format!(
                    "LinearThenTree mode would require ~{}MB for raw feature extraction \
                     ({}rows × {}features × 4bytes), exceeding limit of {}MB. \
                     Options: (1) Increase max_linear_memory_mb, (2) Use PureTree mode, \
                     (3) Reduce dataset size, (4) Use fewer features.",
                    estimated_mb, num_rows, num_features, config.max_linear_memory_mb
                )));
            }

            // Warn on very large allocations (>1GB) even without explicit limit
            if estimated_mb > 1024 {
                eprintln!(
                    "Warning: LinearThenTree will allocate ~{}MB for raw features. \
                     Consider setting max_linear_memory_mb to prevent OOM.",
                    estimated_mb
                );
            }

            // Fallback: extract from bins (lossy - linear model will be less accurate)
            Self::extract_raw_features(dataset)
        };

        // Calculate actual number of features in raw_features
        // (may differ from num_features if FeatureExtractor was used)
        let num_raw_features = raw_features
            .len()
            .checked_div(num_rows)
            .unwrap_or(num_features);

        // Determine number of features for linear model
        let num_linear_features = linear_indices
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(num_raw_features);

        // Base prediction (mean target)
        let base_prediction = loss_fn.initial_prediction(targets);

        // =====================================================================
        // Phase 1: Fit Linear Model using Newton-step Coordinate Descent
        // =====================================================================
        // Iterative gradient-based fitting with learning_rate provides implicit
        // regularization. This is more robust than closed-form Ridge for
        // generalization.

        // Extract selected features for linear model if indices are specified
        let linear_features = extract_selected_features(
            &raw_features,
            num_rows,
            num_raw_features,
            linear_indices.as_deref(),
        );

        let mut linear_booster =
            LinearBooster::new(num_linear_features, config.linear_config.clone());

        // Current predictions start from base
        let mut predictions = vec![base_prediction; num_rows];

        // Iteratively fit linear model
        for _round in 0..config.linear_rounds {
            // Compute gradients and hessians
            let mut gradients = vec![0.0f32; num_rows];
            let mut hessians = vec![0.0f32; num_rows];

            for i in 0..num_rows {
                let (g, h) = loss_fn.gradient_hessian(targets[i], predictions[i]);
                gradients[i] = g;
                hessians[i] = h;
            }

            // Fit linear model on gradients (Newton step)
            linear_booster.fit_on_gradients(
                &linear_features,
                num_linear_features,
                &gradients,
                &hessians,
            )?;

            // Update predictions with shrinkage factor (ensemble weighting)
            let shrinkage = config.linear_config.shrinkage_factor;
            for (i, pred) in predictions.iter_mut().enumerate().take(num_rows) {
                let linear_pred =
                    linear_booster.predict_row(&linear_features, num_linear_features, i);
                *pred += shrinkage * linear_pred;
            }
        }

        // =====================================================================
        // Phase 2: Train GBDTModel on Residuals
        // =====================================================================

        // Clone dataset and modify targets to residuals
        let mut residual_dataset = dataset.clone();
        {
            let residual_targets = residual_dataset.targets_mut();
            for i in 0..num_rows {
                residual_targets[i] = targets[i] - predictions[i];
            }
        }

        // Tree component uses ALL features to capture nonlinear patterns in the residuals.
        // Linear captures linear signal, trees capture what's left — but trees need access
        // to the same features to find nonlinear interactions the linear model missed.

        // Compute residuals on validation set for early stopping
        let val_residual_dataset = if let Some(val_data) = validation_dataset {
            let val_num_rows = val_data.num_rows();
            let val_targets = val_data.targets();

            // Compute linear predictions on validation features
            let val_raw_features = extract_selected_features(
                &Self::extract_raw_features(val_data),
                val_num_rows,
                val_data.num_features(),
                linear_indices.as_deref(),
            );
            let val_linear_preds =
                linear_booster.predict_batch(&val_raw_features, num_linear_features);
            let shrinkage = config.linear_config.shrinkage_factor;

            let mut val_residual = val_data.clone();
            {
                let val_res_targets = val_residual.targets_mut();
                for i in 0..val_num_rows {
                    val_res_targets[i] =
                        val_targets[i] - (base_prediction + shrinkage * val_linear_preds[i]);
                }
            }
            Some(val_residual)
        } else {
            None
        };

        // Check if ensemble training is requested
        let (gbdt_model, gbdt_ensemble, stacker_weights, stacker_intercept) =
            if let Some(ref seeds) = config.ensemble_seeds {
                // Multi-seed ensemble training
                Self::train_gbdt_ensemble(&residual_dataset, &config, loss_fn, seeds)?
            } else {
                // Single GBDT training (standard path)
                let mut gbdt_config = Self::to_gbdt_config(&config, loss_fn)?;
                let gbdt_model = if let Some(ref val_res) = val_residual_dataset {
                    // Set early stopping for presplit validation
                    if config.early_stopping_rounds > 0 {
                        gbdt_config.early_stopping_rounds = config.early_stopping_rounds;
                    } else {
                        gbdt_config.early_stopping_rounds = 50;
                    }
                    GBDTModel::train_binned_with_validation(
                        &residual_dataset,
                        val_res,
                        &[],
                        gbdt_config,
                    )?
                } else {
                    GBDTModel::train_binned(&residual_dataset, gbdt_config)?
                };
                (Some(gbdt_model), None, None, None)
            };

        Ok(Self {
            config,
            gbdt_model,
            gbdt_ensemble,
            stacker_weights,
            stacker_intercept,
            linear_booster: Some(linear_booster),
            linear_boosters: None,
            gbdt_per_label: None,
            trees: Vec::new(), // Not used - GBDTModel stores trees
            base_prediction,
            base_predictions_multi: None,
            num_features,
            analysis,
            raw_features_for_linear: Some(raw_features),
            linear_feature_indices: linear_indices.map(|v| v.to_vec()),
            num_linear_features: Some(num_linear_features),
            feature_extractor,
        })
    }

    // =========================================================================
    // RandomForest Mode
    // =========================================================================

    pub(super) fn train_random_forest(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Initial prediction (mean for RF)
        let base_prediction = loss_fn.initial_prediction(targets);

        // RF uses learning_rate = 1.0 (each tree contributes fully)
        let tree_config = config.tree_config.clone().with_learning_rate(1.0)?;

        // Train trees in parallel with bootstrap samples
        let trees: Vec<Tree> = (0..config.num_rounds)
            .into_par_iter()
            .filter_map(|seed_offset| {
                // Bootstrap sample
                let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed + seed_offset as u64);
                let bootstrap_indices: Vec<usize> = (0..num_rows)
                    .map(|_| {
                        use rand::RngExt;
                        rng.random_range(0..num_rows)
                    })
                    .collect();

                // Compute gradients for this bootstrap sample
                // For RF, we fit to residuals from base prediction
                let mut gradients = vec![0.0f32; num_rows];
                let mut hessians = vec![0.0f32; num_rows];

                for &idx in &bootstrap_indices {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], base_prediction);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }

                // Grow tree on bootstrap sample
                let mut booster = TreeBooster::new(tree_config.clone());
                if booster
                    .fit_on_gradients(dataset, &gradients, &hessians, Some(&bootstrap_indices))
                    .is_ok()
                {
                    booster.take_tree()
                } else {
                    None
                }
            })
            .collect();

        Ok(Self {
            config,
            gbdt_model: None, // RandomForest uses self.trees, not GBDTModel
            gbdt_ensemble: None,
            stacker_weights: None,
            stacker_intercept: None,
            linear_booster: None,
            linear_boosters: None,
            gbdt_per_label: None,
            trees,
            base_prediction,
            base_predictions_multi: None,
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
            feature_extractor,
        })
    }

    // =========================================================================
    // Multi-Label LinearThenTree Mode
    // =========================================================================

    /// Train a UniversalModel for multi-label classification
    ///
    /// Uses LinearThenTree mode with:
    /// 1. One LinearBooster per label (independent per-label linear models)
    /// 2. Multi-label GBDTModel on residuals
    ///
    /// # Arguments
    /// * `dataset` - Multi-output binned dataset (targets are row-wise flattened)
    /// * `config` - Training configuration (mode should be LinearThenTree)
    /// * `loss_fn` - Multi-label loss function (e.g., MultiLabelLogLoss)
    pub fn train_multilabel(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        let num_outputs = dataset.num_target_cols();
        if num_outputs <= 1 {
            return Err(crate::TreeBoostError::Config(
                "train_multilabel requires a multi-output dataset (num_target_cols > 1)"
                    .to_string(),
            ));
        }

        // Force LinearThenTree mode for multi-label
        let config = config.with_mode(BoostingMode::LinearThenTree);

        Self::train_multilabel_ltt(dataset, config, loss_fn, num_outputs)
    }

    /// Internal multi-label LinearThenTree training
    fn train_multilabel_ltt(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        num_outputs: usize,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Validate target dimensions
        if targets.len() != num_rows * num_outputs {
            return Err(crate::TreeBoostError::Config(format!(
                "Target dimension mismatch: expected {} ({}rows × {}outputs), got {}",
                num_rows * num_outputs,
                num_rows,
                num_outputs,
                targets.len()
            )));
        }

        // Extract raw features for linear models
        let raw_features = Self::extract_raw_features(dataset);
        let num_raw_features = raw_features
            .len()
            .checked_div(num_rows)
            .unwrap_or(num_features);

        // =====================================================================
        // Phase 1: Fit N LinearBoosters (one per label)
        // =====================================================================

        // Initialize one LinearBooster per label
        let mut linear_boosters: Vec<crate::learner::LinearBooster> = (0..num_outputs)
            .map(|_| {
                crate::learner::LinearBooster::new(num_raw_features, config.linear_config.clone())
            })
            .collect();

        // Per-label base predictions (log-odds from loss function)
        // This is critical: for classification, base predictions are log-odds, not means
        let base_predictions = loss_fn.initial_predictions_multi(targets, num_outputs);

        // Current predictions per label (start from base)
        let mut predictions: Vec<f32> = vec![0.0; num_rows * num_outputs];
        for i in 0..num_rows {
            for k in 0..num_outputs {
                predictions[i * num_outputs + k] = base_predictions[k];
            }
        }

        // Iteratively fit linear models
        for _round in 0..config.linear_rounds {
            for k in 0..num_outputs {
                // Compute per-label gradients and hessians
                let mut gradients = vec![0.0f32; num_rows];
                let mut hessians = vec![0.0f32; num_rows];

                for i in 0..num_rows {
                    let target = targets[i * num_outputs + k];
                    let pred = predictions[i * num_outputs + k];
                    let (g, h) = loss_fn.gradient_hessian(target, pred);
                    gradients[i] = g;
                    hessians[i] = h;
                }

                // Fit this label's linear model on gradients
                linear_boosters[k].fit_on_gradients(
                    &raw_features,
                    num_raw_features,
                    &gradients,
                    &hessians,
                )?;

                // Update predictions for this label
                let shrinkage = config.linear_config.shrinkage_factor;
                for i in 0..num_rows {
                    let linear_pred =
                        linear_boosters[k].predict_row(&raw_features, num_raw_features, i);
                    predictions[i * num_outputs + k] += shrinkage * linear_pred;
                }
            }
        }

        // =====================================================================
        // Phase 2: Train Per-Label GBDTs on Residuals
        // =====================================================================

        // Create GBDT config for training on residuals
        // IMPORTANT: Use MSE loss for residuals, not multi-label log loss!
        // The residuals are continuous values, not binary targets.
        // For LTT: linear fits gradients of original loss, trees fit residuals with MSE
        let gbdt_config = crate::booster::GBDTConfig::new()
            .with_num_rounds(config.num_rounds)
            .with_learning_rate(config.learning_rate)
            .with_max_depth(config.tree_config.max_depth)
            .with_max_leaves(config.tree_config.max_leaves)
            .with_lambda(config.tree_config.lambda);
        // MSE is the default loss, which is correct for residual fitting

        // Get the binned features from original dataset (column-major)
        let features = dataset.features().to_vec();
        let feature_info = dataset.all_feature_info().to_vec();

        // Train one GBDT per label on residuals
        let mut gbdt_per_label = Vec::with_capacity(num_outputs);

        for k in 0..num_outputs {
            // Compute per-label residuals: target - predicted
            let label_residuals: Vec<f32> = (0..num_rows)
                .map(|i| {
                    let target = targets[i * num_outputs + k];
                    let pred = predictions[i * num_outputs + k];
                    target - pred
                })
                .collect();

            // Create single-output dataset for this label
            let label_dataset = crate::dataset::BinnedDataset::new(
                num_rows,
                features.clone(),
                label_residuals,
                feature_info.clone(),
            );

            let label_gbdt =
                crate::booster::GBDTModel::train_binned(&label_dataset, gbdt_config.clone())?;
            gbdt_per_label.push(label_gbdt);
        }

        Ok(Self {
            config,
            gbdt_model: None, // Using per-label GBDTs instead
            gbdt_ensemble: None,
            stacker_weights: None,
            stacker_intercept: None,
            linear_booster: None,
            linear_boosters: Some(linear_boosters),
            gbdt_per_label: Some(gbdt_per_label),
            trees: Vec::new(),
            base_prediction: 0.0, // Not used for multi-output
            base_predictions_multi: Some(base_predictions),
            num_features,
            analysis: None,
            raw_features_for_linear: Some(raw_features),
            linear_feature_indices: None,
            num_linear_features: Some(num_raw_features),
            feature_extractor: None,
        })
    }
}
