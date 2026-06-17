//! UniversalModel core implementation
//!
//! This module contains the UniversalModel struct definition and core accessors.
//! Implementation details are split across:
//! - training.rs: All train*() methods
//! - prediction.rs: All predict*() methods
//! - incremental.rs: update() and TRB save/load
//! - serialization.rs: TunableModel impl and tests

use super::{BoostingMode, UniversalConfig};
use crate::analysis::{Confidence, DatasetAnalysis};
use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::BinnedDataset;
use crate::learner::LinearBooster;
use crate::loss::LossFunction;
use crate::tree::Tree;
use crate::Result;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// UniversalModel
// =============================================================================

/// Use [`UniversalModel::auto()`] to let TreeBoost analyze your data and pick the best mode.
/// The analysis result is stored and can be retrieved with [`UniversalModel::analysis()`].
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct UniversalModel {
    /// Training configuration
    pub(super) config: UniversalConfig,

    /// GBDTModel for PureTree mode (wraps the mature implementation)
    /// Used when ensemble_seeds is None
    pub(super) gbdt_model: Option<GBDTModel>,

    /// Multi-seed GBDT ensemble (when ensemble_seeds is Some)
    /// - For PureTree mode: Multiple GBDTs trained directly
    /// - For LinearThenTree mode: Multiple GBDTs trained on linear residuals
    /// - For RandomForest mode: Not used (RF already uses multiple trees)
    pub(super) gbdt_ensemble: Option<Vec<GBDTModel>>,

    /// Stacker weights for ensemble combination
    /// Only populated when gbdt_ensemble.is_some()
    pub(super) stacker_weights: Option<Vec<f32>>,

    /// Stacker intercept for Ridge stacking
    /// Only populated when using Ridge stacking strategy
    pub(super) stacker_intercept: Option<f32>,

    /// Linear booster (for single-output LinearThenTree mode)
    pub(super) linear_booster: Option<LinearBooster>,

    /// Linear boosters for multi-output LinearThenTree mode (one per output)
    ///
    /// Used when training multi-label or multi-target regression with LinearThenTree.
    /// Each LinearBooster fits one output dimension independently.
    pub(super) linear_boosters: Option<Vec<LinearBooster>>,

    /// Per-label GBDT models for multi-output LinearThenTree mode
    ///
    /// When training multi-label with LTT, we train K separate GBDTs (one per label)
    /// on the residuals using MSE loss. Each GBDT predicts residuals for its label.
    pub(super) gbdt_per_label: Option<Vec<GBDTModel>>,

    /// Ensemble of trained trees (for LinearThenTree and RandomForest modes)
    /// Used when NOT using gbdt_model or gbdt_ensemble
    pub(super) trees: Vec<Tree>,

    /// Base prediction (for single-output LinearThenTree and RandomForest modes)
    pub(super) base_prediction: f32,

    /// Per-label base predictions (for multi-output LinearThenTree mode)
    ///
    /// These are the log-odds of each label, used as starting points for prediction.
    pub(super) base_predictions_multi: Option<Vec<f32>>,

    /// Number of features
    pub(super) num_features: usize,

    /// Analysis result (if auto mode was used)
    ///
    /// Stores the dataset analysis that led to mode selection.
    /// Use `analysis()` to retrieve and `analysis_report()` to get a formatted report.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    pub(super) analysis: Option<DatasetAnalysis>,

    /// Raw features for LinearThenTree prediction (DEPRECATED - now in BinnedDataset)
    ///
    /// **DEPRECATED**: Raw features are now packed into BinnedDataset via
    /// `BinnedDataset::with_raw_features()`. This field is kept for backward compatibility
    /// with old serialized models, but new code should not use it.
    ///
    /// For new code: Use `dataset.raw_features()` to get preprocessed values.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    pub(super) raw_features_for_linear: Option<Vec<f32>>,

    /// Feature indices to use for linear model (optional)
    ///
    /// When set, only these feature indices from raw_features are used for
    /// the linear model. Trees use the complementary set (all features NOT in this list).
    /// This is CRITICAL for LinearThenTree mode - must be serialized with the model!
    pub(super) linear_feature_indices: Option<Vec<usize>>,

    /// Number of features used by linear model (may differ from num_features)
    pub(super) num_linear_features: Option<usize>,

    /// Feature extractor for LinearThenTree inference (optional)
    ///
    /// Stores feature extraction configuration (which columns to exclude,
    /// auto-exclusion settings) used during training. This ensures consistent
    /// feature extraction during inference for linear component.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    pub(super) feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
}

impl UniversalModel {
    /// Set feature extractor for LinearThenTree inference
    ///
    /// This is called after training to store the feature extraction configuration
    /// (which columns to exclude, auto-exclusion settings). This ensures consistent
    /// feature extraction during inference for the linear component.
    pub fn with_feature_extractor(
        mut self,
        extractor: crate::dataset::feature_extractor::FeatureExtractor,
    ) -> Self {
        self.feature_extractor = Some(extractor);
        self
    }

    /// Get the feature extractor (if set)
    pub fn feature_extractor(
        &self,
    ) -> Option<&crate::dataset::feature_extractor::FeatureExtractor> {
        self.feature_extractor.as_ref()
    }

    /// Set the preprocessing pipeline on the model's config
    ///
    /// This is called after training to store the Pipeline, which includes:
    /// - Feature engineering steps (polynomial, interactions, ratios)
    /// - Categorical encoding (target encoding, frequency encoding)
    /// - Numeric binning (quantile boundaries)
    /// - Target transformation (log, logit, power)
    /// - Linear feature extraction indices
    ///
    /// The pipeline is serialized with the model (in model.rkyv) and used
    /// during inference to apply the same transformations as training.
    pub fn set_pipeline(&mut self, pipeline: crate::model::Pipeline) {
        self.config.pipeline = Some(pipeline);
    }

    /// Set the pipeline state for inference binning/encoding
    pub fn set_pipeline_state(&mut self, state: crate::dataset::PipelineState) {
        self.config.pipeline_state = Some(state);
    }

    /// Apply linear model shrinkage factor to predictions (batch)
    ///
    /// In LinearThenTree mode, the linear model's contribution to the ensemble
    /// is weighted by `shrinkage_factor`:
    ///
    /// ```text
    /// ensemble_pred = base + shrinkage_factor * linear_pred + tree_pred
    /// ```
    ///
    /// This shrinkage factor controls ensemble weighting (not optimization step size).
    /// See `LinearConfig::shrinkage_factor` for details.
    ///
    /// # Arguments
    /// * `predictions` - Mutable slice of predictions to update
    /// * `linear_preds` - Linear model predictions to add (with shrinkage applied)
    #[inline]
    pub(super) fn apply_linear_shrinkage(&self, predictions: &mut [f32], linear_preds: &[f32]) {
        let shrinkage = self.config.linear_config.shrinkage_factor;
        let len = predictions.len().min(linear_preds.len());
        for i in 0..len {
            predictions[i] += shrinkage * linear_preds[i];
        }
    }

    /// Apply linear model shrinkage factor to a single prediction
    ///
    /// Applies the same ensemble weighting logic as `apply_linear_shrinkage`
    /// but for a single scalar value.
    ///
    /// # Arguments
    /// * `linear_pred` - Raw linear prediction
    ///
    /// # Returns
    /// Linear prediction scaled by shrinkage factor
    #[inline]
    pub(super) fn apply_linear_shrinkage_scalar(&self, linear_pred: f32) -> f32 {
        self.config.linear_config.shrinkage_factor * linear_pred
    }

    // =========================================================================
    // Config Conversion
    // =========================================================================

    /// Convert UniversalConfig to GBDTConfig for delegation to GBDTModel
    pub(super) fn to_gbdt_config(
        config: &UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<GBDTConfig> {
        let mut gbdt_config = GBDTConfig::new()
            .with_num_rounds(config.num_rounds)
            .with_learning_rate(config.learning_rate)
            .with_max_depth(config.tree_config.max_depth)
            .with_max_leaves(config.tree_config.max_leaves)
            .with_lambda(config.tree_config.lambda)
            .with_entropy_weight(config.tree_config.entropy_weight) // Pass entropy regularization
            .with_subsample(config.subsample)?
            .with_colsample(config.colsample)?
            .with_backend(config.backend_type) // Pass backend type
            .with_seed(config.seed);

        // Early stopping
        if config.early_stopping_rounds > 0 && config.validation_ratio > 0.0 {
            gbdt_config = gbdt_config
                .with_early_stopping(config.early_stopping_rounds, config.validation_ratio)?;
        }

        // Conformal calibration
        if config.calibration_ratio > 0.0 {
            gbdt_config.calibration_ratio = config.calibration_ratio;
            gbdt_config.conformal_quantile = config.conformal_quantile;
        }

        // Set loss type based on loss_fn type name (best effort)
        let loss_name = std::any::type_name_of_val(loss_fn);
        if loss_name.contains("PseudoHuber") {
            gbdt_config = gbdt_config.with_pseudo_huber_loss(1.0);
        } else if loss_name.contains("BinaryLogLoss") || loss_name.contains("LogLoss") {
            gbdt_config = gbdt_config.with_binary_logloss();
        }
        // Default is MSE which is already the default

        Ok(gbdt_config)
    }

    /// Extract raw feature values from BinnedDataset using bin-center approximation
    ///
    /// **⚠️ Note**: This is a fallback method with accuracy limitations. For best results,
    /// use `FeatureExtractor` with the original DataFrame instead. See the
    /// `feature_extractor` module documentation for a detailed comparison.
    ///
    /// This method approximates raw values by using the midpoint of bin boundaries.
    /// While functional, this loses precision compared to extracting from the original
    /// DataFrame with `FeatureExtractor`.
    ///
    /// Returns row-major f32 array for linear model training.
    ///
    /// # When to Use
    /// - Only when you have a BinnedDataset but not the original DataFrame
    /// - When using UniversalModel directly (advanced usage)
    /// - When slight accuracy loss is acceptable
    ///
    /// # Alternative (Recommended)
    /// ```rust,ignore
    /// use treeboost::dataset::feature_extractor::FeatureExtractor;
    ///
    /// let extractor = FeatureExtractor::new();
    /// let (raw_features, num_features) = extractor.extract(&df, "target")?;
    /// // Use raw_features with train_with_raw_features()
    /// ```
    pub fn extract_raw_features(dataset: &BinnedDataset) -> Vec<f32> {
        dataset.extract_raw_features_from_bins()
    }

    /// Extract linear features for LinearThenTree mode
    ///
    /// Combines raw feature extraction with optional feature selection.
    /// Returns (linear_features, num_linear_features).
    pub(super) fn get_linear_features(&self, dataset: &BinnedDataset) -> (Vec<f32>, usize) {
        let num_rows = dataset.num_rows();

        // Get raw features (from stored or extract from bins)
        let raw_features = self
            .raw_features_for_linear
            .clone()
            .unwrap_or_else(|| Self::extract_raw_features(dataset));

        let num_raw_features = raw_features
            .len()
            .checked_div(num_rows)
            .unwrap_or(self.num_features);

        // Extract selected features if feature selection was used
        let linear_features = crate::features::extract_selected_features(
            &raw_features,
            num_rows,
            num_raw_features,
            self.linear_feature_indices.as_deref(),
        );

        let num_linear_features = self.num_linear_features.unwrap_or(num_raw_features);

        (linear_features, num_linear_features)
    }

    /// Extract raw feature values for a single row
    ///
    /// More efficient than extract_raw_features() when only predicting one row.
    pub(super) fn extract_raw_features_row(dataset: &BinnedDataset, row_idx: usize) -> Vec<f32> {
        let feature_info = dataset.all_feature_info();
        let mut raw_features = Vec::with_capacity(feature_info.len());

        for (f, info) in feature_info.iter().enumerate() {
            let boundaries = &info.bin_boundaries;
            let bin = dataset.get_bin(row_idx, f) as usize;

            // Convert bin back to approximate raw value using bin center
            let raw_value = if boundaries.is_empty() {
                bin as f32
            } else if bin == 0 {
                boundaries.first().copied().unwrap_or(0.0) as f32
            } else if bin >= boundaries.len() {
                boundaries.last().copied().unwrap_or(0.0) as f32
            } else {
                // Midpoint between bin boundaries
                ((boundaries[bin - 1] + boundaries[bin.min(boundaries.len() - 1)]) / 2.0) as f32
            };

            raw_features.push(raw_value);
        }

        raw_features
    }

    /// Predict using GBDT ensemble with stacker weights
    pub(super) fn predict_ensemble(
        &self,
        dataset: &BinnedDataset,
        ensemble: &[GBDTModel],
    ) -> Vec<f32> {
        let num_rows = dataset.num_rows();

        // Get predictions from all ensemble members
        let member_predictions: Vec<Vec<f32>> = ensemble
            .iter()
            .map(|model| model.predict(dataset))
            .collect();

        // Combine using stacker weights
        if let Some(ref weights) = self.stacker_weights {
            let mut combined = vec![0.0; num_rows];

            // Weighted sum of member predictions
            for (pred, &weight) in member_predictions.iter().zip(weights.iter()) {
                for i in 0..num_rows {
                    combined[i] += pred[i] * weight;
                }
            }

            // Add intercept if present (Ridge stacking)
            if let Some(intercept) = self.stacker_intercept {
                for value in combined.iter_mut() {
                    *value += intercept;
                }
            }

            combined
        } else {
            // Fallback to equal-weight averaging if weights not available
            let mut combined = vec![0.0; num_rows];
            let scale = 1.0 / ensemble.len() as f32;

            for pred in &member_predictions {
                for i in 0..num_rows {
                    combined[i] += pred[i] * scale;
                }
            }

            combined
        }
    }

    /// Train multi-seed GBDT ensemble and fit stacker
    ///
    /// Returns (None, Some(ensemble), Some(weights), Some(intercept))
    #[allow(clippy::type_complexity)] // reason: complex return tuple kept inline
    pub(super) fn train_gbdt_ensemble(
        dataset: &BinnedDataset,
        config: &UniversalConfig,
        loss_fn: &dyn LossFunction,
        seeds: &[u64],
    ) -> Result<(
        Option<GBDTModel>,
        Option<Vec<GBDTModel>>,
        Option<Vec<f32>>,
        Option<f32>,
    )> {
        use crate::ensemble::{RidgeStacker, Stacker, StackingConfig};

        // Train multiple GBDTs with different seeds
        let mut models = Vec::with_capacity(seeds.len());
        let mut oof_predictions = Vec::with_capacity(seeds.len());

        for &seed in seeds {
            let mut gbdt_config = Self::to_gbdt_config(config, loss_fn)?;
            gbdt_config.seed = seed;

            let model = GBDTModel::train_binned(dataset, gbdt_config)?;

            // Get OOF predictions for stacking
            let preds = model.predict(dataset);
            oof_predictions.push(preds);

            models.push(model);
        }

        // Fit stacker on OOF predictions
        let (weights, intercept) = match &config.stacking_strategy {
            super::config::StackingStrategy::Ridge {
                alpha,
                rank_transform,
                fit_intercept,
                min_weight,
            } => {
                let stacking_config = StackingConfig {
                    alpha: *alpha,
                    rank_transform: *rank_transform,
                    fit_intercept: *fit_intercept,
                    min_weight: *min_weight,
                };

                let mut stacker = RidgeStacker::new(stacking_config);
                stacker.fit(&oof_predictions, dataset.targets());

                let weights = stacker.weights().map(|w| w.to_vec());
                let intercept = if *fit_intercept {
                    Some(stacker.intercept())
                } else {
                    None
                };

                (weights, intercept)
            }
            super::config::StackingStrategy::Average => {
                // Equal weights for all models
                let weights = vec![1.0 / seeds.len() as f32; seeds.len()];
                (Some(weights), None)
            }
        };

        Ok((None, Some(models), weights, intercept))
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the boosting mode
    pub fn mode(&self) -> BoostingMode {
        self.config.mode
    }

    /// Get training configuration
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Save the model to a file using zero-copy rkyv serialization
    ///
    /// # Example
    /// ```ignore
    /// model.save("model.rkyv")?;
    /// let loaded = UniversalModel::load("model.rkyv")?;
    /// ```
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        crate::serialize::save_universal_model(self, path)
    }

    /// Load a model from a file using zero-copy rkyv deserialization
    ///
    /// # Example
    /// ```ignore
    /// let model = UniversalModel::load("model.rkyv")?;
    /// let predictions = model.predict(&dataset);
    /// ```
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        crate::serialize::load_universal_model(path)
    }

    /// Get number of trees
    pub fn num_trees(&self) -> usize {
        // Check ensemble first
        if let Some(ref ensemble) = self.gbdt_ensemble {
            // Sum trees from all ensemble models
            ensemble.iter().map(|m| m.num_trees()).sum()
        } else if let Some(ref gbdt) = self.gbdt_model {
            // Single GBDT model
            gbdt.num_trees()
        } else {
            // RandomForest uses self.trees
            self.trees.len()
        }
    }

    /// Get base prediction
    ///
    /// For PureTree, this delegates to GBDTModel.
    /// For LinearThenTree, returns the original base prediction (GBDTModel was trained on residuals).
    /// For RandomForest, returns the stored base prediction.
    pub fn base_prediction(&self) -> f32 {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| m.base_prediction())
                .unwrap_or(self.base_prediction),
            // LinearThenTree and RandomForest: use stored base_prediction
            // (GBDTModel in LinearThenTree was trained on residuals, so its base is ~0)
            BoostingMode::LinearThenTree | BoostingMode::RandomForest => self.base_prediction,
        }
    }

    /// Check if model has linear component (single or multi-output)
    pub fn has_linear(&self) -> bool {
        self.linear_booster.is_some() || self.linear_boosters.is_some()
    }

    /// Get linear booster reference (if present, for single-output mode)
    pub fn linear_booster(&self) -> Option<&LinearBooster> {
        self.linear_booster.as_ref()
    }

    /// Get linear boosters reference (for multi-output mode)
    pub fn linear_boosters(&self) -> Option<&[LinearBooster]> {
        self.linear_boosters.as_deref()
    }

    /// Get number of linear boosters (0 for single-output or no linear, N for multi-output)
    pub fn num_linear_boosters(&self) -> usize {
        self.linear_boosters.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get underlying GBDTModel (for PureTree and single-output LinearThenTree modes)
    pub fn gbdt_model(&self) -> Option<&GBDTModel> {
        self.gbdt_model.as_ref()
    }

    /// Get per-label GBDT models (for multi-output LinearThenTree mode)
    ///
    /// Returns K GBDTs, one per label. Each GBDT was trained on residuals for its label.
    pub fn gbdt_per_label(&self) -> Option<&[GBDTModel]> {
        self.gbdt_per_label.as_deref()
    }

    /// Get number of per-label GBDTs
    pub fn num_gbdt_per_label(&self) -> usize {
        self.gbdt_per_label.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get trees (only for RandomForest mode; PureTree/LinearThenTree use GBDTModel)
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    // =========================================================================
    // Analysis and Mode Selection Info
    // =========================================================================

    /// Get the dataset analysis that led to mode selection (if auto mode was used)
    ///
    /// Returns `Some(analysis)` if the model was trained with `auto()` or `train_with_selection(Auto)`.
    /// Returns `None` if a fixed mode was specified.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// if let Some(analysis) = model.analysis() {
    ///     println!("Linear R²: {:.2}", analysis.linear_r2);
    ///     println!("Tree gain: {:.2}", analysis.tree_gain);
    ///     println!("Noise floor: {:.2}", analysis.noise_floor);
    /// }
    /// ```
    pub fn analysis(&self) -> Option<&DatasetAnalysis> {
        self.analysis.as_ref()
    }

    /// Get the confidence in the mode selection (if auto mode was used)
    ///
    /// Returns the confidence level from the analysis:
    /// - `High`: Very clear signal, strongly recommend this mode
    /// - `Medium`: Reasonable signal, this mode is likely best
    /// - `Low`: Weak signal, consider validating with cross-validation
    ///
    /// Returns `None` if a fixed mode was specified.
    pub fn selection_confidence(&self) -> Option<Confidence> {
        self.analysis.as_ref().map(|a| a.confidence())
    }

    /// Check if the mode was automatically selected
    pub fn was_auto_selected(&self) -> bool {
        self.analysis.is_some()
    }

    /// Get a formatted analysis report (if auto mode was used)
    ///
    /// Returns a human-readable report explaining:
    /// - Dataset characteristics (linear signal, tree gain, noise floor)
    /// - Mode scores for each option
    /// - Why the selected mode was chosen
    /// - Alternative modes to consider
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// if let Some(report) = model.analysis_report() {
    ///     println!("{}", report);
    /// }
    /// ```
    pub fn analysis_report(&self) -> Option<crate::analysis::AnalysisReport<'_>> {
        self.analysis.as_ref().map(|a| a.report())
    }

    /// Get a compact single-line summary of the analysis (if auto mode was used)
    ///
    /// Useful for logging or progress output.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// if let Some(summary) = model.analysis_summary() {
    ///     log::info!("{}", summary);
    /// }
    /// ```
    pub fn analysis_summary(&self) -> Option<String> {
        self.analysis.as_ref().map(crate::analysis::compact_summary)
    }

    /// Check if model is multi-class
    pub fn is_multiclass(&self) -> bool {
        self.gbdt_model.as_ref().is_some_and(|m| m.is_multiclass())
    }

    /// Get number of classes (1 for regression, 2+ for classification)
    pub fn get_num_classes(&self) -> usize {
        self.gbdt_model
            .as_ref()
            .map(|m| m.get_num_classes())
            .unwrap_or(1)
    }

    /// Get mutable reference to underlying GBDTModel (for advanced manipulation)
    pub fn gbdt_model_mut(&mut self) -> Option<&mut GBDTModel> {
        self.gbdt_model.as_mut()
    }

    /// Get mutable reference to linear booster (for advanced manipulation)
    pub fn linear_booster_mut(&mut self) -> Option<&mut LinearBooster> {
        self.linear_booster.as_mut()
    }

    /// Get mutable reference to trees (for RandomForest mode)
    pub fn trees_mut(&mut self) -> &mut Vec<Tree> {
        &mut self.trees
    }

    /// Check if model is compatible with dataset for incremental update
    pub fn is_compatible_for_update(&self, dataset: &BinnedDataset) -> bool {
        self.num_features == dataset.num_features()
    }
}

/// Report from an incremental training update
#[derive(Debug, Clone)]
pub struct IncrementalUpdateReport {
    /// Number of rows in the new training data
    pub rows_trained: usize,
    /// Number of trees before update
    pub trees_before: usize,
    /// Number of trees after update
    pub trees_after: usize,
    /// Number of trees added
    pub trees_added: usize,
    /// Boosting mode
    pub mode: BoostingMode,
}

impl std::fmt::Display for IncrementalUpdateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Incremental Update: {} rows, {} trees added ({} -> {}), mode={:?}",
            self.rows_trained, self.trees_added, self.trees_before, self.trees_after, self.mode
        )
    }
}
