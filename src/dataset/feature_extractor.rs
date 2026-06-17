//! Feature Extraction for Linear Models in LinearThenTree Mode
//!
//! # Overview
//!
//! TreeBoost uses **two complementary feature extraction strategies** depending on
//! whether you have access to the original DataFrame or only a BinnedDataset:
//!
//! ## 1. FeatureExtractor (this module) - **Preferred Method**
//! - **Input**: Polars DataFrame (original raw data)
//! - **Output**: `Vec<f32>` with true raw numeric feature values
//! - **Purpose**: Extract accurate raw values for linear model training
//! - **Usage**: During AutoModel training and prediction (recommended path)
//! - **Advantages**: Most accurate for linear models, properly handles categoricals
//! - **Location**: `src/dataset/feature_extractor.rs`
//!
//! ## 2. Bin-Center Extraction - **Fallback Method**
//! - **Input**: BinnedDataset (quantized/discretized data)
//! - **Output**: `Vec<f32>` with approximated values from bin centers
//! - **Purpose**: Fallback when original DataFrame is not available
//! - **Usage**: Direct UniversalModel usage without AutoModel wrapper
//! - **Disadvantages**: Lossy approximation, less accurate for linear models
//! - **Location**: `UniversalModel::extract_raw_features()` in `src/model/universal.rs`
//!
//! # When to Use Which
//!
//! **Use FeatureExtractor (this module)** when:
//! - Training via AutoModel (recommended path - highest accuracy)
//! - You have access to the original DataFrame
//! - Accuracy is critical for the linear component
//! - You want proper handling of categorical/ID-like columns
//!
//! **Use bin-center extraction** when:
//! - Using UniversalModel directly (advanced usage only)
//! - You only have a BinnedDataset available
//! - Slight accuracy loss is acceptable
//! - You're working with pre-binned data
//!
//! # Architecture Note
//!
//! The separation between DataFrame extraction (this module) and BinnedDataset
//! extraction (in UniversalModel) is intentional:
//! - **FeatureExtractor** operates on high-level DataFrame with type information
//! - **Bin-center extraction** operates on low-level binned representation
//! - Both serve different use cases in the LinearThenTree architecture
//! - The two methods are NOT duplicates - they have different inputs and purposes
//!
//! # How FeatureExtractor Works
//!
//! This module extracts raw numeric features from DataFrame for linear models,
//! intelligently excluding problematic columns:
//! - **Categoricals**: Excluded (need encoding first, handled by trees)
//! - **ID-like columns**: Excluded (high cardinality + monotonic, e.g., "id", "year")
//! - **Constant columns**: Excluded (zero variance, no predictive power)
//! - **Boolean/DateTime**: Configurable exclusion
//! - **User-specified**: Manual exclusion via configuration
//!
//! This is critical for LinearThenTree mode, where the linear phase needs:
//! - Raw numeric values (not binned, no bin-center approximation)
//! - Only numeric columns (no categoricals)
//! - Consistent extraction for training and inference
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use treeboost::dataset::feature_extractor::FeatureExtractor;
//! use treeboost::model::{AutoModel, AutoConfig, BoostingMode};
//!
//! // Automatic usage via AutoModel (recommended)
//! let config = AutoConfig::new().with_mode(BoostingMode::LinearThenTree);
//! let model = AutoModel::train_with_config(&df, "target", config)?;
//!
//! // Manual usage with UniversalModel (advanced)
//! let extractor = FeatureExtractor::new();
//! let (raw_features, num_features) = extractor.extract(&df, "target")?;
//!
//! let model = UniversalModel::train_with_raw_features(
//!     &dataset,
//!     &raw_features,
//!     config,
//!     &loss,
//!     Some(extractor), // Store extractor for inference
//! )?;
//! ```
//!
//! # Key Features
//!
//! - **Auto-exclusion**: Automatically detects and excludes problematic columns
//! - **User override**: All auto-decisions can be overridden via LinearFeatureConfig
//! - **Serialization**: Stores feature extraction logic with model for consistent inference
//! - **Transparency**: Generates reports explaining which features were included/excluded
//! - **Type-safe**: Uses Polars type system for robust column type detection

use polars::prelude::*;
use rkyv::Archive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::defaults::features::linear_feature as linear_feature_defaults;
use crate::Result;
use crate::TreeBoostError;

/// Column type classification for feature exclusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub enum ColumnType {
    /// Numeric types (int, float)
    Numeric,
    /// Categorical strings (low cardinality)
    Categorical,
    /// High-cardinality text
    Text,
    /// Boolean
    Boolean,
    /// DateTime types
    DateTime,
    /// ID-like (high cardinality + monotonic)
    IdLike,
    /// Constant (zero variance)
    Constant,
}

impl ColumnType {
    /// Detect column type from Polars DataType and statistics
    pub fn detect(
        dtype: &DataType,
        cardinality_ratio: f32,
        is_constant: bool,
        is_monotonic: bool,
    ) -> Self {
        match dtype {
            DataType::Boolean => ColumnType::Boolean,
            DataType::String | DataType::Categorical(_, _) => {
                if cardinality_ratio > 0.9 && is_monotonic {
                    ColumnType::IdLike
                } else if cardinality_ratio > 0.5 {
                    ColumnType::Text
                } else {
                    ColumnType::Categorical
                }
            }
            DataType::Date | DataType::Datetime(_, _) | DataType::Time | DataType::Duration(_) => {
                ColumnType::DateTime
            }
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => {
                if is_constant {
                    ColumnType::Constant
                } else if cardinality_ratio > 0.9 && is_monotonic {
                    ColumnType::IdLike
                } else {
                    ColumnType::Numeric
                }
            }
            _ => ColumnType::Text, // Unknown types treated as text
        }
    }

    /// Detect from a Polars Series
    pub fn from_series(series: &Series) -> Self {
        let dtype = series.dtype();
        let len = series.len();

        // Skip single-column detection for edge cases
        if len == 0 {
            return ColumnType::Numeric;
        }

        // Calculate cardinality ratio
        let unique_count = series.n_unique().unwrap_or(1);
        let cardinality_ratio = unique_count as f32 / len as f32;

        // Check if constant (all values same)
        let is_constant = unique_count <= 1;

        // Check if monotonic (for ID detection)
        let is_monotonic = Self::is_monotonic(series);

        Self::detect(dtype, cardinality_ratio, is_constant, is_monotonic)
    }

    /// Check if series is monotonically increasing
    fn is_monotonic(series: &Series) -> bool {
        match series.dtype() {
            DataType::Int32 => {
                if let Ok(ca) = series.i32() {
                    let vals: Vec<Option<i32>> = ca.iter().collect();
                    for i in 1..vals.len() {
                        if let (Some(a), Some(b)) = (vals[i - 1], vals[i]) {
                            if b <= a {
                                return false;
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            DataType::Int64 => {
                if let Ok(ca) = series.i64() {
                    let vals: Vec<Option<i64>> = ca.iter().collect();
                    for i in 1..vals.len() {
                        if let (Some(a), Some(b)) = (vals[i - 1], vals[i]) {
                            if b <= a {
                                return false;
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            DataType::UInt32 => {
                if let Ok(ca) = series.u32() {
                    let vals: Vec<Option<u32>> = ca.iter().collect();
                    for i in 1..vals.len() {
                        if let (Some(a), Some(b)) = (vals[i - 1], vals[i]) {
                            if b <= a {
                                return false;
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// Configuration for linear model feature extraction
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct LinearFeatureConfig {
    /// Columns to exclude from linear model (user override)
    pub exclude_columns: HashSet<String>,

    /// Exclude categoricals from linear model
    pub exclude_categorical: bool,

    /// Exclude ID-like columns (high cardinality + monotonic)
    pub exclude_id: bool,

    /// Exclude constant columns (zero variance)
    pub exclude_constant: bool,

    /// Exclude boolean columns
    pub exclude_boolean: bool,

    /// Exclude datetime columns
    pub exclude_datetime: bool,

    /// Exclude text columns (high cardinality strings)
    pub exclude_text: bool,
}

impl Default for LinearFeatureConfig {
    fn default() -> Self {
        // IMPORTANT: Sensible defaults for linear models
        // Exclude problematic column types that hurt linear model performance
        Self {
            exclude_columns: HashSet::new(),
            exclude_categorical: linear_feature_defaults::EXCLUDE_CATEGORICAL, // Categoricals already encoded by DataPipeline
            exclude_id: linear_feature_defaults::EXCLUDE_ID, // ID columns have no predictive value
            exclude_constant: linear_feature_defaults::EXCLUDE_CONSTANT, // Zero variance columns are useless
            exclude_boolean: linear_feature_defaults::EXCLUDE_BOOLEAN, // Booleans can be useful (0/1)
            exclude_datetime: linear_feature_defaults::EXCLUDE_DATETIME, // DateTime needs feature engineering first
            exclude_text: linear_feature_defaults::EXCLUDE_TEXT, // High-cardinality text needs encoding
        }
    }
}

impl LinearFeatureConfig {
    /// Create default config
    pub fn new() -> Self {
        Self::default()
    }

    /// Exclude specific columns
    pub fn with_exclude_columns(mut self, columns: &[&str]) -> Self {
        self.exclude_columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a column to exclude
    pub fn exclude_column(mut self, column: &str) -> Self {
        self.exclude_columns.insert(column.to_string());
        self
    }

    /// Enable/disable categorical auto-exclusion
    pub fn with_exclude_categorical(mut self, enable: bool) -> Self {
        self.exclude_categorical = enable;
        self
    }

    /// Enable/disable ID auto-exclusion
    pub fn with_exclude_id(mut self, enable: bool) -> Self {
        self.exclude_id = enable;
        self
    }

    /// Enable/disable constant auto-exclusion
    pub fn with_exclude_constant(mut self, enable: bool) -> Self {
        self.exclude_constant = enable;
        self
    }

    /// Enable/disable boolean auto-exclusion
    pub fn with_exclude_boolean(mut self, enable: bool) -> Self {
        self.exclude_boolean = enable;
        self
    }

    /// Enable/disable datetime auto-exclusion
    pub fn with_exclude_datetime(mut self, enable: bool) -> Self {
        self.exclude_datetime = enable;
        self
    }

    /// Enable/disable text auto-exclusion
    pub fn with_exclude_text(mut self, enable: bool) -> Self {
        self.exclude_text = enable;
        self
    }
}

/// Result of feature extraction with detailed report
#[derive(Debug, Clone)]
pub struct FeatureExtractionResult {
    /// Raw features in row-major layout: features[row * num_features + col]
    pub features: Vec<f32>,

    /// Number of features extracted
    pub num_features: usize,

    /// Names of features included
    pub feature_names: Vec<String>,

    /// Detailed report of what was excluded and why
    pub report: FeatureExtractionReport,
}

/// Report explaining feature selection decisions
#[derive(Debug, Clone)]
pub struct FeatureExtractionReport {
    /// Total columns in DataFrame
    pub total_columns: usize,

    /// Columns excluded by type
    pub excluded_by_type: HashMap<ColumnType, Vec<String>>,

    /// Columns excluded by user
    pub excluded_by_user: Vec<String>,

    /// Target column (always excluded)
    pub target_column: String,

    /// Final features included
    pub final_features: Vec<String>,
}

impl FeatureExtractionReport {
    /// Generate human-readable report
    pub fn format(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Feature Extraction Report ===\n");
        output.push_str(&format!("Total columns: {}\n", self.total_columns));
        output.push_str(&format!("Target column: {}\n", self.target_column));
        output.push_str(&format!(
            "Final features: {}\n\n",
            self.final_features.len()
        ));

        // User exclusions
        if !self.excluded_by_user.is_empty() {
            output.push_str("Excluded by user:\n");
            for col in &self.excluded_by_user {
                output.push_str(&format!("  - {}\n", col));
            }
            output.push('\n');
        }

        // Type-based exclusions
        for (col_type, cols) in &self.excluded_by_type {
            if !cols.is_empty() {
                output.push_str(&format!("Excluded as {:?}:\n", col_type));
                for col in cols {
                    output.push_str(&format!("  - {}\n", col));
                }
                output.push('\n');
            }
        }

        // Final features
        output.push_str("Final features for linear model:\n");
        for (idx, col) in self.final_features.iter().enumerate() {
            output.push_str(&format!("  [{}] {}\n", idx, col));
        }

        output
    }
}

/// Feature extractor for linear models
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct FeatureExtractor {
    config: LinearFeatureConfig,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create new extractor with default config
    pub fn new() -> Self {
        Self {
            config: LinearFeatureConfig::default(),
        }
    }

    /// Create extractor with custom config
    pub fn with_config(config: LinearFeatureConfig) -> Self {
        Self { config }
    }

    /// Extract raw numeric features from DataFrame for linear models
    ///
    /// Returns row-major `Vec<f32>` where:
    /// - features[row * num_features + col] = value of feature col for row
    ///
    /// Excludes:
    /// - Target column (always)
    /// - User-specified columns (if any)
    /// - Categoricals (if auto_exclude_categorical)
    /// - ID-like columns (if auto_exclude_id)
    /// - Constant columns (if auto_exclude_constant)
    /// - Boolean columns (if auto_exclude_boolean)
    /// - DateTime columns (if auto_exclude_datetime)
    /// - Text columns (if auto_exclude_text)
    pub fn extract_numeric_features(
        &self,
        df: &DataFrame,
        target_col: &str,
    ) -> Result<FeatureExtractionResult> {
        let num_rows = df.height();
        let total_columns = df.width();

        if num_rows == 0 {
            return Ok(FeatureExtractionResult {
                features: vec![],
                num_features: 0,
                feature_names: vec![],
                report: FeatureExtractionReport {
                    total_columns,
                    excluded_by_type: HashMap::new(),
                    excluded_by_user: vec![],
                    target_column: target_col.to_string(),
                    final_features: vec![],
                },
            });
        }

        // Track exclusions
        let mut excluded_by_type: HashMap<ColumnType, Vec<String>> = HashMap::new();
        let mut excluded_by_user: Vec<String> = vec![];

        // Find numeric columns to include
        let mut feature_names: Vec<String> = vec![];

        for col_name in df.get_column_names() {
            let col_name_str = col_name.as_str();

            // Skip target column
            if col_name_str == target_col {
                continue;
            }

            // Skip user-specified exclusions
            if self.config.exclude_columns.contains(col_name_str) {
                excluded_by_user.push(col_name_str.to_string());
                continue;
            }

            // Get column series
            let col = df.column(col_name).map_err(|e| {
                TreeBoostError::Data(format!("Column '{}' not found: {}", col_name, e))
            })?;

            // Detect column type
            let series = col.as_materialized_series();
            let col_type = ColumnType::from_series(series);

            // Check if we should exclude this type
            let should_exclude = match col_type {
                ColumnType::Numeric => false, // Always include numeric
                ColumnType::Boolean => self.config.exclude_boolean,
                ColumnType::Categorical => self.config.exclude_categorical,
                ColumnType::IdLike => self.config.exclude_id,
                ColumnType::Constant => self.config.exclude_constant,
                ColumnType::DateTime => self.config.exclude_datetime,
                ColumnType::Text => self.config.exclude_text,
            };

            if should_exclude {
                excluded_by_type
                    .entry(col_type)
                    .or_default()
                    .push(col_name_str.to_string());
                continue;
            }

            // Include this feature (config-based exclusion already applied above)
            feature_names.push(col_name_str.to_string());
        }

        // Extract features in row-major layout
        let num_features = feature_names.len();
        let mut features = Vec::with_capacity(num_rows * num_features);

        for row_idx in 0..num_rows {
            for col_name in &feature_names {
                let col = df.column(col_name).map_err(|e| {
                    TreeBoostError::Data(format!(
                        "Column '{}' not found during extraction: {}",
                        col_name, e
                    ))
                })?;

                let series = col.as_materialized_series();
                let val = series.get(row_idx)?;
                let f_val = self.anyvalue_to_f32(val, row_idx, col_name);
                features.push(f_val);
            }
        }

        // Build report
        let report = FeatureExtractionReport {
            total_columns,
            excluded_by_type: excluded_by_type.clone(),
            excluded_by_user: excluded_by_user.clone(),
            target_column: target_col.to_string(),
            final_features: feature_names.clone(),
        };

        Ok(FeatureExtractionResult {
            features,
            num_features,
            feature_names,
            report,
        })
    }

    /// Extract features and return raw `Vec<f32>` (simplified API)
    pub fn extract(&self, df: &DataFrame, target_col: &str) -> Result<(Vec<f32>, usize)> {
        let result = self.extract_numeric_features(df, target_col)?;
        Ok((result.features, result.num_features))
    }

    /// Convert AnyValue to f32 with error handling
    fn anyvalue_to_f32(&self, val: AnyValue, _row_idx: usize, _col_name: &str) -> f32 {
        match val {
            AnyValue::Null => {
                // Fill NaN with 0 for linear model
                0.0
            }
            AnyValue::Int8(v) => v as f32,
            AnyValue::Int16(v) => v as f32,
            AnyValue::Int32(v) => v as f32,
            AnyValue::Int64(v) => v as f32,
            AnyValue::UInt8(v) => v as f32,
            AnyValue::UInt16(v) => v as f32,
            AnyValue::UInt32(v) => v as f32,
            AnyValue::UInt64(v) => {
                // May overflow on very large values
                v.min(u32::MAX as u64) as f32
            }
            AnyValue::Float32(v) if v.is_finite() => v,
            AnyValue::Float64(v) if v.is_finite() => v as f32,
            AnyValue::Boolean(v) if v => 1.0,
            _ => 0.0,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &LinearFeatureConfig {
        &self.config
    }

    /// Check if a column should be excluded
    pub fn should_exclude_column(&self, df: &DataFrame, col_name: &str, target_col: &str) -> bool {
        if col_name == target_col {
            return true;
        }

        if self.config.exclude_columns.contains(col_name) {
            return true;
        }

        if let Ok(col) = df.column(col_name) {
            let series = col.as_materialized_series();
            let col_type = ColumnType::from_series(series);

            match col_type {
                ColumnType::Numeric => false,
                ColumnType::Boolean => self.config.exclude_boolean,
                ColumnType::Categorical => self.config.exclude_categorical,
                ColumnType::IdLike => self.config.exclude_id,
                ColumnType::Constant => self.config.exclude_constant,
                ColumnType::DateTime => self.config.exclude_datetime,
                ColumnType::Text => self.config.exclude_text,
            }
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataframe() -> DataFrame {
        df!(
            "id" => &[1i32, 2, 3, 4, 5],
            "year" => &[2020i32, 2021, 2022, 2023, 2024],
            "rank" => &[1i32, 2, 3, 4, 5],
            "numeric1" => &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            "numeric2" => &[10.0f32, 20.0, 30.0, 40.0, 50.0],
            "constant" => &[5i32, 5, 5, 5, 5],
            "category" => &["A", "B", "A", "B", "A"],
            "target" => &[1.0f32, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn test_exclude_target_column() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        assert!(!result.feature_names.contains(&"target".to_string()));
        assert!(result.num_features > 0);
    }

    #[test]
    fn test_auto_exclude_categorical() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        assert!(!result.feature_names.contains(&"category".to_string()));
        assert!(result
            .report
            .excluded_by_type
            .contains_key(&ColumnType::Categorical));
    }

    #[test]
    fn test_auto_exclude_constant() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        assert!(!result.feature_names.contains(&"constant".to_string()));
        assert!(result
            .report
            .excluded_by_type
            .contains_key(&ColumnType::Constant));
    }

    #[test]
    fn test_include_numeric() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        assert!(result.feature_names.contains(&"numeric1".to_string()));
        assert!(result.feature_names.contains(&"numeric2".to_string()));
    }

    #[test]
    fn test_user_exclude_override() {
        let df = create_test_dataframe();
        let config = LinearFeatureConfig::new().with_exclude_columns(&["numeric1"]);
        let extractor = FeatureExtractor::with_config(config);
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        assert!(!result.feature_names.contains(&"numeric1".to_string()));
        assert!(result.feature_names.contains(&"numeric2".to_string()));
        assert!(result
            .report
            .excluded_by_user
            .contains(&"numeric1".to_string()));
    }

    #[test]
    fn test_row_major_layout() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        let num_rows = df.height();
        let num_features = result.num_features;

        assert_eq!(result.features.len(), num_rows * num_features);

        // Verify row-major ordering
        if num_features >= 2 {
            let row_0_feat_0 = result.features[0]; // row 0, feature 0
            let row_0_feat_1 = result.features[1]; // row 0, feature 1
            let row_1_feat_0 = result.features[num_features];

            // Values should match the DataFrame
            assert_eq!(row_0_feat_0, 1.0); // numeric1 row 0
            assert_eq!(row_0_feat_1, 10.0); // numeric2 row 0
            assert_eq!(row_1_feat_0, 2.0); // numeric1 row 1
        }
    }

    #[test]
    fn test_auto_exclude_id_like() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        // 'id' is monotonic with high cardinality ratio, should be excluded
        assert!(!result.feature_names.contains(&"id".to_string()));
    }

    #[test]
    fn test_report_formatting() {
        let df = create_test_dataframe();
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        let report_str = result.report.format();
        assert!(report_str.contains("Feature Extraction Report"));
        assert!(report_str.contains("Total columns"));
        assert!(report_str.contains("Final features"));
    }

    #[test]
    fn test_boolean_handling() {
        let df = df!(
            "feature" => &[1.0f32, 2.0, 3.0],
            "flag" => &[true, false, true],
            "target" => &[1.0f32, 2.0, 3.0],
        )
        .unwrap();

        // By default, booleans are not auto-excluded
        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();
        assert!(result.feature_names.contains(&"flag".to_string()));

        // But can be excluded
        let config = LinearFeatureConfig::new().with_exclude_boolean(true);
        let extractor = FeatureExtractor::with_config(config);
        let result = extractor.extract_numeric_features(&df, "target").unwrap();
        assert!(!result.feature_names.contains(&"flag".to_string()));
    }

    #[test]
    fn test_nan_handling() {
        let df = df!(
            "feature" => &[Some(1.0f32), None, Some(3.0)],
            "target" => &[Some(1.0f32), Some(2.0), Some(3.0)],
        )
        .unwrap();

        let extractor = FeatureExtractor::new();
        let result = extractor.extract_numeric_features(&df, "target").unwrap();

        // NaN should be filled with 0
        assert_eq!(result.features[result.num_features], 0.0);
    }

    #[test]
    fn test_column_type_detection() {
        // Test id-like (monotonic + high cardinality ratio = 1.0)
        let s_id = Series::new("test".into(), &[1i32, 2, 3, 4, 5]);
        assert_eq!(ColumnType::from_series(&s_id), ColumnType::IdLike);

        // Test numeric (non-monotonic)
        let s_numeric = Series::new("test".into(), &[1i32, 2, 1, 3, 2]);
        assert_eq!(ColumnType::from_series(&s_numeric), ColumnType::Numeric);

        // Test constant
        let s_constant = Series::new("test".into(), &[1i32, 1, 1, 1, 1]);
        assert_eq!(ColumnType::from_series(&s_constant), ColumnType::Constant);

        // Test categorical
        let s_cat = Series::new("test".into(), &["A", "B", "A", "B", "A"]);
        assert_eq!(ColumnType::from_series(&s_cat), ColumnType::Categorical);

        // Test boolean
        let s_bool = Series::new("test".into(), &[true, false, true, false, true]);
        assert_eq!(ColumnType::from_series(&s_bool), ColumnType::Boolean);
    }

    #[test]
    #[ignore] // TODO: Update for rkyv 0.8 API - requires proper Serialize/Deserialize trait implementations
    fn test_serialization_roundtrip() {
        let config = LinearFeatureConfig::new()
            .with_exclude_columns(&["col1", "col2"])
            .with_exclude_categorical(false);

        let extractor = FeatureExtractor::with_config(config.clone());

        // TODO: Implement proper rkyv 0.8 serialization
        // The serialization test needs to be updated for rkyv 0.8's new trait system

        // Verify config is created correctly
        assert_eq!(
            extractor.config.exclude_categorical,
            config.exclude_categorical
        );
        assert_eq!(extractor.config.exclude_columns, config.exclude_columns);
    }
}
