//! Predictor Abstraction for Unified Prediction Interface
//!
//! This module provides a fully rkyv-serializable predictor system that enables:
//! - Formula-based predictions (domain knowledge, Kaggle-winning patterns)
//! - ML model predictions (UniversalModel + Pipeline)
//! - Gated predictions (formula + classifier for boundary cases)
//! - Ensemble predictions (weighted combination)
//!
//! # Design Philosophy
//!
//! **White-box, fully replicable**: Load model → predict(df). Everything serializes.
//!
//! Uses concrete types (not trait objects or recursive enums) for full rkyv compatibility.
//!
//! # Example
//!
//! ```ignore
//! // Formula predictor (Kaggle-winning pattern)
//! let formula = FormulaBuilder::new("score_formula")
//!     .add_term("study_hours", 6.0)
//!     .add_lut("sleep_quality", &[("good", 5.0), ("poor", -5.0)])
//!     .build();
//! let predictor = FormulaPredictor::new(formula);
//!
//! // Gated predictor (formula + boundary classifier)
//! let gated = GatedPredictor::new(
//!     formula,
//!     boundary_classifier,
//!     GatingConfig::new(0.8, 0.8, 19.6, 100.0),
//! );
//!
//! // Predict
//! let predictions = gated.predict(&df)?;
//!
//! // Save/load (fully serializable)
//! gated.save("predictor.rkyv")?;
//! let loaded = GatedPredictor::load("predictor.rkyv")?;
//! ```

use crate::booster::GBDTModel;
use crate::dataset::core::{BinnedDataset, FeatureInfo, FeatureType};
use crate::model::{CustomFeature, Pipeline, UniversalModel};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

// =============================================================================
// FormulaPredictor
// =============================================================================

/// Predictor that evaluates a custom formula directly
///
/// Wraps a `CustomFeature` (built via `FormulaBuilder`) for direct prediction.
/// No ML model - just domain knowledge encoded as weighted features + LUT mappings.
///
/// # Example
///
/// ```ignore
/// let formula = FormulaBuilder::new("exam_score")
///     .add_term("study_hours", 6.0)
///     .add_term("attendance", 0.35)
///     .add_lut("sleep_quality", &[("good", 5.0), ("average", 0.0), ("poor", -5.0)])
///     .build();
///
/// let predictor = FormulaPredictor::new(formula);
/// let scores = predictor.predict(&df)?;
/// ```
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct FormulaPredictor {
    /// The formula to evaluate
    pub formula: CustomFeature,
}

impl FormulaPredictor {
    /// Create a new formula predictor
    pub fn new(formula: CustomFeature) -> Self {
        Self { formula }
    }

    /// Predict using the formula
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f64>> {
        self.formula.evaluate(df)
    }

    /// Predict with uncertainty (formula has no uncertainty - returns zeros)
    pub fn predict_with_uncertainty(&self, df: &DataFrame) -> Result<(Vec<f64>, Vec<f64>)> {
        let preds = self.predict(df)?;
        let uncertainties = vec![0.0; preds.len()];
        Ok((preds, uncertainties))
    }

    /// Save predictor to file using rkyv
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize: {}", e)))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load predictor from file using rkyv
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let archived = rkyv::access::<rkyv::Archived<Self>, rkyv::rancor::Error>(&bytes)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to access: {}", e)))?;
        rkyv::deserialize::<Self, rkyv::rancor::Error>(archived)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to deserialize: {}", e)))
    }
}

// =============================================================================
// ModelPredictor
// =============================================================================

/// Predictor that wraps a UniversalModel with its Pipeline
///
/// The standard ML prediction workflow:
/// 1. Pipeline transforms raw DataFrame
/// 2. Model predicts on transformed data
/// 3. Target transform inverse is applied (if configured)
///
/// # Example
///
/// ```ignore
/// let builder = AutoBuilder::new().fit(&df, "target")?;
/// let (model, pipeline) = builder.into_parts();
/// let predictor = ModelPredictor::new(model, pipeline);
/// let predictions = predictor.predict(&test_df)?;
/// ```
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct ModelPredictor {
    /// The trained model
    pub model: UniversalModel,
    /// The preprocessing pipeline
    pub pipeline: Pipeline,
}

impl ModelPredictor {
    /// Create a new model predictor
    pub fn new(model: UniversalModel, pipeline: Pipeline) -> Self {
        Self { model, pipeline }
    }

    /// Predict using the model
    ///
    /// This method creates a temporary model with pipeline attached and delegates
    /// to `UniversalModel::predict_df()` which handles all the complexity of
    /// pipeline transformation, binning, and target transform inverse.
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f64>> {
        // Clone model and attach pipeline for predict_df
        let mut model_with_pipeline = self.model.clone();
        model_with_pipeline.set_pipeline(self.pipeline.clone());

        // Delegate to predict_df which handles everything
        let preds_f32 = model_with_pipeline.predict_df(df)?;

        Ok(preds_f32.into_iter().map(|p| p as f64).collect())
    }

    /// Predict with uncertainty (uses conformal intervals if available)
    pub fn predict_with_uncertainty(&self, df: &DataFrame) -> Result<(Vec<f64>, Vec<f64>)> {
        // For now, just return predictions with zero uncertainty
        // TODO: Integrate conformal prediction intervals
        let preds = self.predict(df)?;
        let uncertainties = vec![0.0; preds.len()];
        Ok((preds, uncertainties))
    }

    /// Save predictor to file using rkyv
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize: {}", e)))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load predictor from file using rkyv
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let archived = rkyv::access::<rkyv::Archived<Self>, rkyv::rancor::Error>(&bytes)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to access: {}", e)))?;
        rkyv::deserialize::<Self, rkyv::rancor::Error>(archived)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to deserialize: {}", e)))
    }
}

// =============================================================================
// GatedPredictor
// =============================================================================

/// Configuration for gated prediction
///
/// Defines thresholds and boundary values for the gating mechanism.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Probability threshold for LOW class (class 0)
    pub threshold_low: f64,
    /// Probability threshold for HIGH class (class 2)
    pub threshold_high: f64,
    /// Value to use when gating to LOW
    pub min_value: f64,
    /// Value to use when gating to HIGH
    pub max_value: f64,
}

impl GatingConfig {
    /// Create a new gating configuration
    ///
    /// # Arguments
    /// - `threshold_low`: Probability threshold for LOW class detection
    /// - `threshold_high`: Probability threshold for HIGH class detection
    /// - `min_value`: Value to output when LOW is detected
    /// - `max_value`: Value to output when HIGH is detected
    pub fn new(threshold_low: f64, threshold_high: f64, min_value: f64, max_value: f64) -> Self {
        Self {
            threshold_low,
            threshold_high,
            min_value,
            max_value,
        }
    }

    /// Create symmetric thresholds (same for low and high)
    pub fn symmetric(threshold: f64, min_value: f64, max_value: f64) -> Self {
        Self::new(threshold, threshold, min_value, max_value)
    }
}

/// Gated predictor: formula + classifier for boundary cases
///
/// Implements the winning Kaggle pattern:
/// 1. Formula handles most cases (domain knowledge)
/// 2. Classifier detects boundary cases (very low/high values)
/// 3. Gating logic overrides formula with min/max when classifier is confident
///
/// # Example
///
/// ```ignore
/// // Train boundary classifier (3 classes: LOW, MID, HIGH)
/// let classifier = train_boundary_classifier(&train_df, min_split, max_split)?;
///
/// // Create gated predictor
/// let gated = GatedPredictor::new(
///     formula,
///     classifier,
///     GatingConfig::new(0.8, 0.8, 19.6, 100.0),
/// );
///
/// // Predictions use formula, but extreme cases get overridden
/// let predictions = gated.predict(&test_df)?;
/// ```
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct GatedPredictor {
    /// Base formula for normal cases
    pub formula: CustomFeature,

    /// Multiclass classifier (3 classes: LOW=0, MID=1, HIGH=2)
    pub classifier: GBDTModel,

    /// Gating configuration
    pub config: GatingConfig,
}

impl GatedPredictor {
    /// Create a new gated predictor
    pub fn new(formula: CustomFeature, classifier: GBDTModel, config: GatingConfig) -> Self {
        Self {
            formula,
            classifier,
            config,
        }
    }

    /// Predict with gating logic
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f64>> {
        // Get base predictions from formula
        let mut predictions = self.formula.evaluate(df)?;

        // Get classifier probabilities
        let class_probs = self.get_class_probabilities(df)?;

        // Apply gating
        for (i, probs) in class_probs.iter().enumerate() {
            if probs[0] >= self.config.threshold_low {
                // HIGH probability of LOW class -> override with min_value
                predictions[i] = self.config.min_value;
            } else if probs[2] >= self.config.threshold_high {
                // HIGH probability of HIGH class -> override with max_value
                predictions[i] = self.config.max_value;
            }
            // Otherwise keep formula prediction
        }

        Ok(predictions)
    }

    /// Predict with uncertainty
    pub fn predict_with_uncertainty(&self, df: &DataFrame) -> Result<(Vec<f64>, Vec<f64>)> {
        let preds = self.predict(df)?;

        // Uncertainty is higher for gated predictions (classifier uncertainty)
        let class_probs = self.get_class_probabilities(df)?;
        let uncertainties: Vec<f64> = class_probs
            .iter()
            .map(|probs| {
                // Entropy-based uncertainty: -sum(p * log(p))
                let entropy: f64 = probs
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| -p * p.ln())
                    .sum();
                entropy / 3.0_f64.ln() // Normalize by max entropy (log(3) for 3 classes)
            })
            .collect();

        Ok((preds, uncertainties))
    }

    /// Save predictor to file using rkyv
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize: {}", e)))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load predictor from file using rkyv
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let archived = rkyv::access::<rkyv::Archived<Self>, rkyv::rancor::Error>(&bytes)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to access: {}", e)))?;
        rkyv::deserialize::<Self, rkyv::rancor::Error>(archived)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to deserialize: {}", e)))
    }

    /// Get class probabilities from classifier
    fn get_class_probabilities(&self, df: &DataFrame) -> Result<Vec<[f64; 3]>> {
        let num_rows = df.height();
        let num_features = df.width();

        // Extract features and create BinnedDataset for classifier
        let mut binned = Vec::with_capacity(num_rows * num_features);
        let mut feature_info = Vec::with_capacity(num_features);

        for col in df.columns() {
            let col_name = col.name().to_string();

            // Extract column values as f32 with proper error handling
            let vals: Result<Vec<f32>> = (0..num_rows)
                .map(|r| {
                    if let Ok(ca) = col.f64() {
                        ca.get(r)
                            .ok_or_else(|| {
                                TreeBoostError::Pipeline(format!(
                                    "Null/missing value in column '{}' at row {}",
                                    col_name, r
                                ))
                            })
                            .map(|v| v as f32)
                    } else if let Ok(ca) = col.f32() {
                        ca.get(r).ok_or_else(|| {
                            TreeBoostError::Pipeline(format!(
                                "Null/missing value in column '{}' at row {}",
                                col_name, r
                            ))
                        })
                    } else if let Ok(ca) = col.i64() {
                        ca.get(r)
                            .ok_or_else(|| {
                                TreeBoostError::Pipeline(format!(
                                    "Null/missing value in column '{}' at row {}",
                                    col_name, r
                                ))
                            })
                            .map(|v| v as f32)
                    } else if let Ok(ca) = col.i32() {
                        ca.get(r)
                            .ok_or_else(|| {
                                TreeBoostError::Pipeline(format!(
                                    "Null/missing value in column '{}' at row {}",
                                    col_name, r
                                ))
                            })
                            .map(|v| v as f32)
                    } else {
                        Err(TreeBoostError::Pipeline(format!(
                            "Unsupported column type for '{}'. Expected numeric type (f64, f32, i64, i32)",
                            col_name
                        )))
                    }
                })
                .collect();
            let vals = vals?;

            // Compute min/max for uniform binning
            let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = (max_val - min_val).max(1e-9);

            // Create 255 uniform bins
            let boundaries: Vec<f64> = (1..=254)
                .map(|i| (min_val + range * (i as f32) / 255.0) as f64)
                .collect();

            // Bin values
            for &val in &vals {
                let bin = ((val - min_val) / range * 254.0).clamp(0.0, 254.0) as u8;
                binned.push(bin);
            }

            feature_info.push(FeatureInfo {
                name: col_name,
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: boundaries,
                impute_value: 0.0,
            });
        }

        // Transpose from column-major to row-major format
        let mut binned_row_major = vec![0u8; num_rows * num_features];
        for row in 0..num_rows {
            for feat in 0..num_features {
                binned_row_major[row * num_features + feat] = binned[feat * num_rows + row];
            }
        }

        // Create BinnedDataset with dummy targets
        let dummy_targets = vec![0.0f32; num_rows];
        let dataset = BinnedDataset::new(num_rows, binned_row_major, dummy_targets, feature_info);

        // Get raw probabilities from classifier
        let raw_probs = self.classifier.predict_proba_multiclass(&dataset);

        // Convert to [f64; 3] arrays
        let probs: Vec<[f64; 3]> = raw_probs
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();

        Ok(probs)
    }
}

// =============================================================================
// EnsemblePredictor
// =============================================================================

/// Strategy for combining ensemble predictions
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Weighted average of predictions
    WeightedAverage,
    /// Median of predictions (robust to outliers)
    Median,
}

/// Ensemble predictor: combines formula and model predictions
///
/// Combines predictions from a formula and an ML model using weighted averaging.
/// This is a common pattern: formula captures domain knowledge, model captures residual patterns.
///
/// # Example
///
/// ```ignore
/// let ensemble = EnsemblePredictor::new(
///     formula,
///     model,
///     pipeline,
///     0.3,  // 30% formula weight
///     0.7,  // 70% model weight
/// );
/// let predictions = ensemble.predict(&df)?;
/// ```
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct EnsemblePredictor {
    /// Formula predictor
    pub formula: CustomFeature,
    /// ML model
    pub model: UniversalModel,
    /// Pipeline for model
    pub pipeline: Pipeline,
    /// Weight for formula (0.0 to 1.0)
    pub formula_weight: f64,
    /// Weight for model (0.0 to 1.0)
    pub model_weight: f64,
    /// Strategy for combining predictions
    pub strategy: EnsembleStrategy,
}

impl EnsemblePredictor {
    /// Create a new ensemble predictor
    pub fn new(
        formula: CustomFeature,
        model: UniversalModel,
        pipeline: Pipeline,
        formula_weight: f64,
        model_weight: f64,
    ) -> Self {
        Self {
            formula,
            model,
            pipeline,
            formula_weight,
            model_weight,
            strategy: EnsembleStrategy::WeightedAverage,
        }
    }

    /// Create with equal weights
    pub fn equal_weights(
        formula: CustomFeature,
        model: UniversalModel,
        pipeline: Pipeline,
    ) -> Self {
        Self::new(formula, model, pipeline, 0.5, 0.5)
    }

    /// Set combination strategy
    pub fn with_strategy(mut self, strategy: EnsembleStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Predict using ensemble
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f64>> {
        // Get formula predictions
        let formula_preds = self.formula.evaluate(df)?;

        // Get model predictions
        let mut model_with_pipeline = self.model.clone();
        model_with_pipeline.set_pipeline(self.pipeline.clone());
        let model_preds: Vec<f64> = model_with_pipeline
            .predict_df(df)?
            .into_iter()
            .map(|p| p as f64)
            .collect();

        let n_samples = formula_preds.len();

        match self.strategy {
            EnsembleStrategy::WeightedAverage => {
                let combined: Vec<f64> = formula_preds
                    .iter()
                    .zip(model_preds.iter())
                    .map(|(&f, &m)| f * self.formula_weight + m * self.model_weight)
                    .collect();
                Ok(combined)
            }
            EnsembleStrategy::Median => {
                let combined: Vec<f64> = (0..n_samples)
                    .map(|i| {
                        let mut values = [formula_preds[i], model_preds[i]];
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        (values[0] + values[1]) / 2.0 // Median of 2 = average
                    })
                    .collect();
                Ok(combined)
            }
        }
    }

    /// Predict with uncertainty (disagreement between predictors)
    pub fn predict_with_uncertainty(&self, df: &DataFrame) -> Result<(Vec<f64>, Vec<f64>)> {
        // Get formula predictions
        let formula_preds = self.formula.evaluate(df)?;

        // Get model predictions
        let mut model_with_pipeline = self.model.clone();
        model_with_pipeline.set_pipeline(self.pipeline.clone());
        let model_preds: Vec<f64> = model_with_pipeline
            .predict_df(df)?
            .into_iter()
            .map(|p| p as f64)
            .collect();

        // Calculate combined predictions and uncertainty (disagreement)
        let combined: Vec<f64> = formula_preds
            .iter()
            .zip(model_preds.iter())
            .map(|(&f, &m)| f * self.formula_weight + m * self.model_weight)
            .collect();

        let uncertainties: Vec<f64> = formula_preds
            .iter()
            .zip(model_preds.iter())
            .map(|(&f, &m)| (f - m).abs()) // Disagreement as uncertainty
            .collect();

        Ok((combined, uncertainties))
    }

    /// Save predictor to file using rkyv
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize: {}", e)))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load predictor from file using rkyv
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let archived = rkyv::access::<rkyv::Archived<Self>, rkyv::rancor::Error>(&bytes)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to access: {}", e)))?;
        rkyv::deserialize::<Self, rkyv::rancor::Error>(archived)
            .map_err(|e| TreeBoostError::Serialization(format!("Failed to deserialize: {}", e)))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FormulaBuilder;

    #[test]
    fn test_formula_predictor_creation() {
        // Build a simple linear formula: 2.0 * x
        let formula = FormulaBuilder::new("test").add_term("x", 2.0).build();

        let predictor = FormulaPredictor::new(formula);
        assert_eq!(predictor.formula.name, "test");
    }

    #[test]
    fn test_gating_config() {
        let config = GatingConfig::symmetric(0.8, 0.0, 100.0);
        assert_eq!(config.threshold_low, 0.8);
        assert_eq!(config.threshold_high, 0.8);
        assert_eq!(config.min_value, 0.0);
        assert_eq!(config.max_value, 100.0);
    }

    #[test]
    fn test_ensemble_strategy() {
        assert!(matches!(
            EnsembleStrategy::WeightedAverage,
            EnsembleStrategy::WeightedAverage
        ));
        assert!(matches!(EnsembleStrategy::Median, EnsembleStrategy::Median));
    }
}
