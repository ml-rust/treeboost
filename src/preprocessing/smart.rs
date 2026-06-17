//! Smart Preprocessing Engine
//!
//! Automatically infers optimal preprocessing based on column profiles and model type.
//! Acts as a "Smart Data Engineer" that prescribes preprocessing with full transparency.
//!
//! # Design Philosophy
//!
//! **Different models need different preprocessing:**
//! - Linear models: NEED scaling (StandardScaler), benefit from YeoJohnson for skew
//! - Tree models: DON'T need scaling, prefer FrequencyEncoder over OneHot
//! - LTT mode: TWO separate preprocessing plans (scaled for linear, raw for trees)
//!
//! # Example
//!
//! ```ignore
//! use treeboost::analysis::profiler::DataFrameProfile;
//! use treeboost::preprocessing::smart::{SmartPreprocessor, ModelType};
//!
//! let profile = DataFrameProfile::analyze(&df, "target")?;
//! let plan = SmartPreprocessor::infer(&profile, ModelType::Tree);
//!
//! println!("Preprocessing Plan:");
//! for reason in &plan.reasoning {
//!     println!("  - {}", reason);
//! }
//! ```

use crate::analysis::profiler::{ColumnDataType, ColumnProfile, DataFrameProfile};
use crate::defaults::preprocessing as preprocessing_defaults;
use crate::preprocessing::{
    FrequencyEncoder, LabelEncoder, MinMaxScaler, OneHotEncoder, Preprocessor, RobustScaler,
    SimpleImputer, StandardScaler, YeoJohnsonTransform,
};
use std::collections::{HashMap, HashSet};

/// Target model type - determines preprocessing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Linear models (Ridge, LASSO, ElasticNet)
    /// - REQUIRE StandardScaler
    /// - Benefit from YeoJohnson for skewed features
    /// - Prefer OneHotEncoder for categoricals
    Linear,

    /// Tree-based models (GBDT, Random Forest)
    /// - DON'T need scaling (scale-invariant)
    /// - Prefer FrequencyEncoder (faster than OneHot)
    /// - Handle missing values via bin 0
    Tree,

    /// LinearThenTree hybrid mode
    /// - Generates TWO preprocessing plans
    /// - Linear phase: scaled features
    /// - Tree phase: raw features
    LinearThenTree,
}

/// Step in a preprocessing plan
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessingStep {
    /// Column name this step applies to
    pub column: String,
    /// The preprocessor to apply
    pub preprocessor: Preprocessor,
    /// Human-readable reason for this choice
    pub reason: String,
}

/// Complete preprocessing plan for a dataset
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessingPlan {
    /// Ordered list of preprocessing steps
    pub steps: Vec<PreprocessingStep>,
    /// Columns to drop (ID-like, constant, text)
    pub drop_columns: Vec<String>,
    /// Human-readable reasoning for all decisions
    pub reasoning: Vec<String>,
}

impl PreprocessingPlan {
    /// Create an empty plan
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            drop_columns: Vec::new(),
            reasoning: Vec::new(),
        }
    }

    /// Add a preprocessing step
    pub fn add_step(&mut self, column: String, preprocessor: Preprocessor, reason: String) {
        self.reasoning.push(format!("{}: {}", column, reason));
        self.steps.push(PreprocessingStep {
            column,
            preprocessor,
            reason,
        });
    }

    /// Mark a column for dropping
    pub fn drop_column(&mut self, column: String, reason: String) {
        self.reasoning.push(format!("DROP {}: {}", column, reason));
        self.drop_columns.push(column);
    }

    /// Get step for a specific column
    pub fn get_step(&self, column: &str) -> Option<&PreprocessingStep> {
        self.steps.iter().find(|s| s.column == column)
    }
}

impl Default for PreprocessingPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// LTT-specific preprocessing plan with separate phases
#[derive(Debug, Clone)]
pub struct LttPreprocessingPlan {
    /// Preprocessing for linear phase (scaled features)
    pub linear_plan: PreprocessingPlan,
    /// Preprocessing for tree phase (raw features)
    pub tree_plan: PreprocessingPlan,
    /// Shared steps applied to both phases (imputation, basic encoding)
    pub shared_steps: Vec<PreprocessingStep>,
    /// Combined reasoning
    pub reasoning: Vec<String>,
}

impl LttPreprocessingPlan {
    /// Create empty LTT plan
    pub fn new() -> Self {
        Self {
            linear_plan: PreprocessingPlan::new(),
            tree_plan: PreprocessingPlan::new(),
            shared_steps: Vec::new(),
            reasoning: Vec::new(),
        }
    }
}

impl Default for LttPreprocessingPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for smart preprocessing decisions
#[derive(Debug, Clone)]
pub struct SmartPreprocessConfig {
    /// Cardinality threshold for "high cardinality" (default: 50)
    pub high_cardinality_threshold: usize,
    /// Skewness threshold for YeoJohnson (default: 2.0)
    pub skewness_threshold: f32,
    /// Missing ratio threshold for indicator imputer (default: 0.05)
    pub missing_indicator_threshold: f32,
    /// Force specific encodings for columns
    pub force_encodings: HashMap<String, EncodingType>,
    /// Force specific scalers for columns
    pub force_scalers: HashMap<String, ScalerType>,
    /// Skip preprocessing for these columns
    pub skip_columns: HashSet<String>,
}

/// Presets for smart preprocessing thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartPreprocessPreset {
    /// Standard thresholds.
    Standard,
    /// Higher cardinality threshold, looser skewness.
    Permissive,
    /// Lower thresholds, more aggressive encoding.
    Strict,
}

impl Default for SmartPreprocessConfig {
    fn default() -> Self {
        Self {
            high_cardinality_threshold: preprocessing_defaults::HIGH_CARDINALITY_THRESHOLD,
            skewness_threshold: preprocessing_defaults::SKEWNESS_THRESHOLD,
            missing_indicator_threshold: preprocessing_defaults::MISSING_INDICATOR_THRESHOLD,
            force_encodings: HashMap::new(),
            force_scalers: HashMap::new(),
            skip_columns: HashSet::new(),
        }
    }
}

impl SmartPreprocessConfig {
    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: SmartPreprocessPreset) -> Self {
        match preset {
            SmartPreprocessPreset::Standard => {}
            SmartPreprocessPreset::Permissive => {
                self.high_cardinality_threshold =
                    preprocessing_defaults::PERMISSIVE_HIGH_CARDINALITY_THRESHOLD;
                self.skewness_threshold = preprocessing_defaults::PERMISSIVE_SKEWNESS_THRESHOLD;
                self.missing_indicator_threshold =
                    preprocessing_defaults::PERMISSIVE_MISSING_INDICATOR_THRESHOLD;
            }
            SmartPreprocessPreset::Strict => {
                self.high_cardinality_threshold =
                    preprocessing_defaults::STRICT_HIGH_CARDINALITY_THRESHOLD;
                self.skewness_threshold = preprocessing_defaults::STRICT_SKEWNESS_THRESHOLD;
                self.missing_indicator_threshold =
                    preprocessing_defaults::STRICT_MISSING_INDICATOR_THRESHOLD;
            }
        }
        self
    }
}

/// Encoding type override
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    Frequency,
    Label,
    OneHot,
    Target,
}

/// Scaler type override
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalerType {
    Standard,
    MinMax,
    Robust,
    None,
}

/// Smart Preprocessing Engine
///
/// Analyzes column profiles and model type to generate optimal preprocessing plans.
#[derive(Debug, Clone)]
pub struct SmartPreprocessor {
    /// Configuration for preprocessing decisions
    pub config: SmartPreprocessConfig,
}

impl SmartPreprocessor {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: SmartPreprocessConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SmartPreprocessConfig) -> Self {
        Self { config }
    }

    /// Infer optimal preprocessing plan based on profile and model type
    ///
    /// # Decision Matrix
    ///
    /// | Column Type | Condition | Model=Linear | Model=Tree |
    /// |-------------|-----------|--------------|------------|
    /// | Numeric | Normal | StandardScaler | None |
    /// | Numeric | Skewed (>2.0) | YeoJohnson + StandardScaler | None |
    /// | Numeric | Missing | Imputer + Indicator | Imputer + Indicator |
    /// | Categorical | Low card (<50) | OneHot | FrequencyEncoder |
    /// | Categorical | High card (>=50) | TargetEncoder | TargetEncoder |
    /// | ID-like | Monotonic + unique | DROP | DROP |
    /// | Constant | Zero variance | DROP | DROP |
    pub fn infer(profile: &DataFrameProfile, model_type: ModelType) -> PreprocessingPlan {
        let config = SmartPreprocessConfig::default();
        Self::infer_with_config(profile, model_type, &config)
    }

    /// Infer with custom configuration
    pub fn infer_with_config(
        profile: &DataFrameProfile,
        model_type: ModelType,
        config: &SmartPreprocessConfig,
    ) -> PreprocessingPlan {
        let mut plan = PreprocessingPlan::new();

        // Add drops from profile
        for dropped in &profile.drop_columns {
            plan.drop_column(dropped.name.clone(), dropped.reason.to_string());
        }

        // Process each column
        for col_profile in &profile.columns {
            // Skip if in skip list
            if config.skip_columns.contains(&col_profile.name) {
                plan.reasoning
                    .push(format!("{}: Skipped (user override)", col_profile.name));
                continue;
            }

            Self::infer_column_preprocessing(&mut plan, col_profile, model_type, config);
        }

        plan
    }

    /// Infer preprocessing for a single column
    fn infer_column_preprocessing(
        plan: &mut PreprocessingPlan,
        col: &ColumnProfile,
        model_type: ModelType,
        config: &SmartPreprocessConfig,
    ) {
        match col.dtype {
            ColumnDataType::Numeric => {
                Self::infer_numeric_preprocessing(plan, col, model_type, config);
            }
            ColumnDataType::Categorical => {
                Self::infer_categorical_preprocessing(plan, col, model_type, config);
            }
            ColumnDataType::Boolean => {
                // Booleans are already 0/1 - no preprocessing needed
                plan.reasoning
                    .push(format!("{}: Boolean (no preprocessing needed)", col.name));
            }
            ColumnDataType::DateTime => {
                // DateTime columns should have SeasonalGenerator applied
                // For now, mark for feature engineering rather than preprocessing
                plan.reasoning.push(format!(
                    "{}: DateTime (defer to feature engineering)",
                    col.name
                ));
            }
            ColumnDataType::Text => {
                // Text columns should be dropped (no NLP support in v1)
                plan.drop_column(col.name.clone(), "Text column (requires NLP)".to_string());
            }
        }
    }

    /// Infer preprocessing for numeric columns
    fn infer_numeric_preprocessing(
        plan: &mut PreprocessingPlan,
        col: &ColumnProfile,
        model_type: ModelType,
        config: &SmartPreprocessConfig,
    ) {
        let mut steps_for_col: Vec<(Preprocessor, String)> = Vec::new();

        // Check for forced scaler
        if let Some(scaler_type) = config.force_scalers.get(&col.name) {
            let (scaler, reason) = match scaler_type {
                ScalerType::Standard => (
                    Preprocessor::Standard(StandardScaler::new()),
                    "StandardScaler (forced)".to_string(),
                ),
                ScalerType::MinMax => (
                    Preprocessor::MinMax(MinMaxScaler::new()),
                    "MinMaxScaler (forced)".to_string(),
                ),
                ScalerType::Robust => (
                    Preprocessor::Robust(RobustScaler::new()),
                    "RobustScaler (forced)".to_string(),
                ),
                ScalerType::None => {
                    plan.reasoning
                        .push(format!("{}: No scaling (forced)", col.name));
                    return;
                }
            };
            plan.add_step(col.name.clone(), scaler, reason);
            return;
        }

        // 1. Handle missing values first
        if col.missing_ratio > 0.0 {
            steps_for_col.push((
                Preprocessor::Imputer(SimpleImputer::median()),
                format!(
                    "Median imputation ({:.1}% missing)",
                    col.missing_ratio * 100.0
                ),
            ));

            // Add indicator if high missing ratio
            if col.missing_ratio > config.missing_indicator_threshold {
                plan.reasoning.push(format!(
                    "{}: Consider adding missing indicator ({:.1}% > {:.1}%)",
                    col.name,
                    col.missing_ratio * 100.0,
                    config.missing_indicator_threshold * 100.0
                ));
            }
        }

        // 2. Handle skewness (only for linear models)
        if model_type == ModelType::Linear || model_type == ModelType::LinearThenTree {
            if let Some(skew) = col.skewness {
                if skew.abs() > config.skewness_threshold && !col.has_negative {
                    steps_for_col.push((
                        Preprocessor::YeoJohnson(YeoJohnsonTransform::new()),
                        format!(
                            "YeoJohnson (skewness={:.2} > {:.1})",
                            skew, config.skewness_threshold
                        ),
                    ));
                } else if skew.abs() > config.skewness_threshold && col.has_negative {
                    plan.reasoning.push(format!(
                        "{}: High skew ({:.2}) but has negatives - skipping YeoJohnson",
                        col.name, skew
                    ));
                }
            }
        }

        // 3. Apply scaling based on model type
        match model_type {
            ModelType::Linear => {
                // Linear models REQUIRE scaling
                steps_for_col.push((
                    Preprocessor::Standard(StandardScaler::new()),
                    "StandardScaler (required for linear models)".to_string(),
                ));
            }
            ModelType::Tree => {
                // Trees don't need scaling
                if steps_for_col.is_empty() {
                    plan.reasoning
                        .push(format!("{}: No scaling needed (tree model)", col.name));
                }
            }
            ModelType::LinearThenTree => {
                // For LTT, this creates the LINEAR plan - scaling is required
                steps_for_col.push((
                    Preprocessor::Standard(StandardScaler::new()),
                    "StandardScaler (for linear phase of LTT)".to_string(),
                ));
            }
        }

        // Add all steps for this column
        for (preprocessor, reason) in steps_for_col {
            plan.add_step(col.name.clone(), preprocessor, reason);
        }
    }

    /// Infer preprocessing for categorical columns
    fn infer_categorical_preprocessing(
        plan: &mut PreprocessingPlan,
        col: &ColumnProfile,
        model_type: ModelType,
        config: &SmartPreprocessConfig,
    ) {
        // Check for forced encoding
        if let Some(encoding_type) = config.force_encodings.get(&col.name) {
            let (encoder, reason) = match encoding_type {
                EncodingType::Frequency => (
                    Preprocessor::Frequency(FrequencyEncoder::new()),
                    "FrequencyEncoder (forced)".to_string(),
                ),
                EncodingType::Label => (
                    Preprocessor::Label(LabelEncoder::new()),
                    "LabelEncoder (forced)".to_string(),
                ),
                EncodingType::OneHot => (
                    Preprocessor::OneHot(OneHotEncoder::new()),
                    "OneHotEncoder (forced)".to_string(),
                ),
                EncodingType::Target => {
                    // Target encoding would be handled separately
                    plan.reasoning
                        .push(format!("{}: TargetEncoder (forced)", col.name));
                    return;
                }
            };
            plan.add_step(col.name.clone(), encoder, reason);
            return;
        }

        let is_high_cardinality = col.cardinality > config.high_cardinality_threshold;

        match model_type {
            ModelType::Linear => {
                if is_high_cardinality {
                    // High cardinality: use TargetEncoder (not OneHot - too many columns)
                    plan.reasoning.push(format!(
                        "{}: High cardinality ({}) - recommend TargetEncoder",
                        col.name, col.cardinality
                    ));
                    // Fall back to FrequencyEncoder since TargetEncoder needs target
                    plan.add_step(
                        col.name.clone(),
                        Preprocessor::Frequency(FrequencyEncoder::new()),
                        format!(
                            "FrequencyEncoder (high cardinality {} > {})",
                            col.cardinality, config.high_cardinality_threshold
                        ),
                    );
                } else {
                    // Low cardinality: use OneHot for linear models
                    plan.add_step(
                        col.name.clone(),
                        Preprocessor::OneHot(OneHotEncoder::new()),
                        format!(
                            "OneHotEncoder (low cardinality {} for linear model)",
                            col.cardinality
                        ),
                    );
                }
            }
            ModelType::Tree | ModelType::LinearThenTree => {
                // Trees prefer FrequencyEncoder (faster, no column explosion)
                if is_high_cardinality {
                    plan.add_step(
                        col.name.clone(),
                        Preprocessor::Frequency(FrequencyEncoder::new()),
                        format!(
                            "FrequencyEncoder (high cardinality {} > {})",
                            col.cardinality, config.high_cardinality_threshold
                        ),
                    );
                } else {
                    plan.add_step(
                        col.name.clone(),
                        Preprocessor::Frequency(FrequencyEncoder::new()),
                        format!(
                            "FrequencyEncoder (optimal for trees, {} categories)",
                            col.cardinality
                        ),
                    );
                }
            }
        }
    }

    /// Create separate preprocessing plans for LTT mode
    ///
    /// Returns two plans:
    /// - `linear_plan`: Scaled features for linear phase
    /// - `tree_plan`: Raw features for tree phase (on residuals)
    pub fn infer_ltt(profile: &DataFrameProfile) -> LttPreprocessingPlan {
        let config = SmartPreprocessConfig::default();
        Self::infer_ltt_with_config(profile, &config)
    }

    /// Create LTT plans with custom configuration
    pub fn infer_ltt_with_config(
        profile: &DataFrameProfile,
        config: &SmartPreprocessConfig,
    ) -> LttPreprocessingPlan {
        let mut ltt_plan = LttPreprocessingPlan::new();

        ltt_plan
            .reasoning
            .push("=== LTT Dual-Phase Preprocessing ===".to_string());
        ltt_plan
            .reasoning
            .push("Phase 1 (Linear): Scaled features with StandardScaler".to_string());
        ltt_plan
            .reasoning
            .push("Phase 2 (Tree): Raw features, no scaling".to_string());

        // Add drops from profile to both plans
        for dropped in &profile.drop_columns {
            ltt_plan
                .linear_plan
                .drop_column(dropped.name.clone(), dropped.reason.to_string());
            ltt_plan
                .tree_plan
                .drop_column(dropped.name.clone(), dropped.reason.to_string());
        }

        // Process each column
        for col_profile in &profile.columns {
            if config.skip_columns.contains(&col_profile.name) {
                continue;
            }

            // Shared: Imputation (both phases need non-missing data)
            if col_profile.missing_ratio > 0.0 && col_profile.dtype == ColumnDataType::Numeric {
                ltt_plan.shared_steps.push(PreprocessingStep {
                    column: col_profile.name.clone(),
                    preprocessor: Preprocessor::Imputer(SimpleImputer::median()),
                    reason: format!(
                        "Median imputation ({:.1}% missing) - shared",
                        col_profile.missing_ratio * 100.0
                    ),
                });
            }

            match col_profile.dtype {
                ColumnDataType::Numeric => {
                    // LINEAR PHASE: StandardScaler + optional YeoJohnson
                    if let Some(skew) = col_profile.skewness {
                        if skew.abs() > config.skewness_threshold && !col_profile.has_negative {
                            ltt_plan.linear_plan.add_step(
                                col_profile.name.clone(),
                                Preprocessor::YeoJohnson(YeoJohnsonTransform::new()),
                                format!("YeoJohnson (skew={:.2}) for linear phase", skew),
                            );
                        }
                    }
                    ltt_plan.linear_plan.add_step(
                        col_profile.name.clone(),
                        Preprocessor::Standard(StandardScaler::new()),
                        "StandardScaler (required for linear phase)".to_string(),
                    );

                    // TREE PHASE: No scaling needed
                    ltt_plan.tree_plan.reasoning.push(format!(
                        "{}: No scaling (trees are scale-invariant)",
                        col_profile.name
                    ));
                }
                ColumnDataType::Categorical => {
                    let is_high_card = col_profile.cardinality > config.high_cardinality_threshold;

                    // LINEAR PHASE: OneHot for low cardinality, Frequency for high
                    if is_high_card {
                        ltt_plan.linear_plan.add_step(
                            col_profile.name.clone(),
                            Preprocessor::Frequency(FrequencyEncoder::new()),
                            format!(
                                "FrequencyEncoder (high cardinality {} for linear)",
                                col_profile.cardinality
                            ),
                        );
                    } else {
                        ltt_plan.linear_plan.add_step(
                            col_profile.name.clone(),
                            Preprocessor::OneHot(OneHotEncoder::new()),
                            format!(
                                "OneHotEncoder ({} categories for linear phase)",
                                col_profile.cardinality
                            ),
                        );
                    }

                    // TREE PHASE: FrequencyEncoder (optimal for trees)
                    ltt_plan.tree_plan.add_step(
                        col_profile.name.clone(),
                        Preprocessor::Frequency(FrequencyEncoder::new()),
                        format!(
                            "FrequencyEncoder (optimal for trees, {} categories)",
                            col_profile.cardinality
                        ),
                    );
                }
                ColumnDataType::Boolean => {
                    // Both phases: no preprocessing needed
                    ltt_plan
                        .reasoning
                        .push(format!("{}: Boolean (no preprocessing)", col_profile.name));
                }
                ColumnDataType::DateTime => {
                    ltt_plan.reasoning.push(format!(
                        "{}: DateTime (defer to feature engineering)",
                        col_profile.name
                    ));
                }
                ColumnDataType::Text => {
                    // Already dropped
                }
            }
        }

        ltt_plan
    }

    /// Generate human-readable summary of preprocessing plan
    pub fn summarize(plan: &PreprocessingPlan) -> String {
        let mut summary = String::new();

        summary.push_str("Preprocessing Plan:\n");
        summary.push_str(&format!("  Columns to drop: {}\n", plan.drop_columns.len()));
        summary.push_str(&format!("  Preprocessing steps: {}\n", plan.steps.len()));
        summary.push_str("\nDecisions:\n");

        for reason in &plan.reasoning {
            summary.push_str(&format!("  - {}\n", reason));
        }

        summary
    }
}

impl Default for SmartPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_profile() -> DataFrameProfile {
        let df = DataFrame::new(
            5,
            vec![
                Series::new("numeric_normal".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
                Series::new("numeric_skewed".into(), vec![1.0f64, 1.0, 1.0, 10.0, 100.0]).into(),
                Series::new("categorical_low".into(), vec!["A", "B", "A", "B", "A"]).into(),
                Series::new("constant".into(), vec![1.0f64, 1.0, 1.0, 1.0, 1.0]).into(),
                Series::new("target".into(), vec![0.0f64, 1.0, 0.0, 1.0, 0.0]).into(),
            ],
        )
        .unwrap();

        DataFrameProfile::analyze(&df, "target").unwrap()
    }

    #[test]
    fn test_infer_linear_model() {
        let profile = create_test_profile();
        let plan = SmartPreprocessor::infer(&profile, ModelType::Linear);

        // Should have StandardScaler for numeric columns
        assert!(plan
            .steps
            .iter()
            .any(|s| s.column == "numeric_normal"
                && matches!(s.preprocessor, Preprocessor::Standard(_))));

        // Should drop constant column
        assert!(plan.drop_columns.contains(&"constant".to_string()));
    }

    #[test]
    fn test_infer_tree_model() {
        let profile = create_test_profile();
        let plan = SmartPreprocessor::infer(&profile, ModelType::Tree);

        // Should NOT have StandardScaler for numeric columns (trees don't need it)
        let has_scaler_for_normal = plan.steps.iter().any(|s| {
            s.column == "numeric_normal" && matches!(s.preprocessor, Preprocessor::Standard(_))
        });
        assert!(!has_scaler_for_normal);

        // Should have FrequencyEncoder for categorical
        assert!(plan.steps.iter().any(|s| s.column == "categorical_low"
            && matches!(s.preprocessor, Preprocessor::Frequency(_))));
    }

    #[test]
    fn test_infer_ltt() {
        let profile = create_test_profile();
        let ltt_plan = SmartPreprocessor::infer_ltt(&profile);

        // Linear plan should have StandardScaler
        assert!(ltt_plan
            .linear_plan
            .steps
            .iter()
            .any(|s| s.column == "numeric_normal"
                && matches!(s.preprocessor, Preprocessor::Standard(_))));

        // Tree plan should NOT have StandardScaler for numeric
        let tree_has_scaler = ltt_plan.tree_plan.steps.iter().any(|s| {
            s.column == "numeric_normal" && matches!(s.preprocessor, Preprocessor::Standard(_))
        });
        assert!(!tree_has_scaler);
    }

    #[test]
    fn test_force_encoding() {
        let profile = create_test_profile();
        let mut config = SmartPreprocessConfig::default();
        config
            .force_encodings
            .insert("categorical_low".to_string(), EncodingType::Label);

        let plan = SmartPreprocessor::infer_with_config(&profile, ModelType::Linear, &config);

        // Should have LabelEncoder instead of OneHot
        assert!(plan
            .steps
            .iter()
            .any(|s| s.column == "categorical_low"
                && matches!(s.preprocessor, Preprocessor::Label(_))));
    }

    #[test]
    fn test_skip_columns() {
        let profile = create_test_profile();
        let mut config = SmartPreprocessConfig::default();
        config.skip_columns.insert("numeric_normal".to_string());

        let plan = SmartPreprocessor::infer_with_config(&profile, ModelType::Linear, &config);

        // Should not have any steps for skipped column
        assert!(!plan.steps.iter().any(|s| s.column == "numeric_normal"));
    }
}
