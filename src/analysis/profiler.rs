//! Column and DataFrame Profiling for AutoML
//!
//! This module provides the "blood test" for each column - analyzing data characteristics
//! to inform smart preprocessing, feature engineering, and model selection decisions.
//!
//! # Design Philosophy
//!
//! **Two-Pass Analysis for Efficiency:**
//! 1. **Pass 1 (Cheap)**: Use Polars native stats (`null_count`, `n_unique`, `min`, `max`)
//!    to identify columns to DROP (constant, ID-like, text) - ~1ms per column
//! 2. **Pass 2 (Expensive)**: Only for kept columns, compute skewness, kurtosis,
//!    target correlation - these require full scans
//!
//! # Example
//!
//! ```ignore
//! use treeboost::analysis::profiler::{DataFrameProfile, TaskType};
//!
//! let profile = DataFrameProfile::analyze(&df, "target")?;
//! println!("{}", profile.report());
//!
//! // Check what to drop
//! for col in &profile.drop_columns {
//!     println!("Dropping: {} (reason: {})", col.name, col.reason);
//! }
//!
//! // Get task type
//! match profile.task_type {
//!     TaskType::Regression => println!("Regression task"),
//!     TaskType::BinaryClassification => println!("Binary classification"),
//!     TaskType::MultiClassification { num_classes } => println!("{}-class classification", num_classes),
//! }
//! ```

use crate::preprocessing::polars_ext::{is_categorical, is_numeric};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::collections::HashSet;

// Use alias to avoid conflict with Polars DataType
use polars::prelude::DataType as PolarsDataType;

/// Check if a column name suggests it's a date/time column
fn is_date_column_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower == "date"
        || lower == "time"
        || lower == "datetime"
        || lower == "timestamp"
        || lower == "dt"
        || lower.ends_with("_date")
        || lower.ends_with("_time")
        || lower.ends_with("_dt")
        || lower.starts_with("date_")
        || lower.starts_with("time_")
}

/// Time granularity detected from date/datetime differences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeGranularity {
    /// Sub-hourly data (minutes, seconds)
    SubHourly,
    /// Hourly observations
    Hourly,
    /// Daily observations
    Daily,
    /// Weekly observations
    Weekly,
    /// Monthly observations
    Monthly,
    /// Yearly observations
    Yearly,
    /// Could not determine granularity
    Unknown,
}

impl TimeGranularity {
    /// Get suggested lag periods for this granularity
    pub fn suggested_lag_periods(&self) -> Vec<usize> {
        match self {
            TimeGranularity::SubHourly => vec![1, 5, 10, 30, 60],
            TimeGranularity::Hourly => vec![1, 6, 12, 24],
            TimeGranularity::Daily => vec![1, 3, 7, 14],
            TimeGranularity::Weekly => vec![1, 4, 12],
            TimeGranularity::Monthly => vec![1, 3, 6, 12],
            TimeGranularity::Yearly => vec![1, 2, 5],
            TimeGranularity::Unknown => vec![1, 3, 7],
        }
    }

    /// Get suggested rolling windows for this granularity
    pub fn suggested_rolling_windows(&self) -> Vec<usize> {
        match self {
            TimeGranularity::SubHourly => vec![10, 30, 60, 120],
            TimeGranularity::Hourly => vec![12, 24, 48],
            TimeGranularity::Daily => vec![7, 14, 28],
            TimeGranularity::Weekly => vec![4, 12, 26],
            TimeGranularity::Monthly => vec![3, 6, 12],
            TimeGranularity::Yearly => vec![2, 5, 10],
            TimeGranularity::Unknown => vec![7, 14, 28],
        }
    }
}

impl std::fmt::Display for TimeGranularity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeGranularity::SubHourly => write!(f, "sub-hourly"),
            TimeGranularity::Hourly => write!(f, "hourly"),
            TimeGranularity::Daily => write!(f, "daily"),
            TimeGranularity::Weekly => write!(f, "weekly"),
            TimeGranularity::Monthly => write!(f, "monthly"),
            TimeGranularity::Yearly => write!(f, "yearly"),
            TimeGranularity::Unknown => write!(f, "unknown"),
        }
    }
}

/// Panel data structure information
///
/// Panel data (also called longitudinal data) has observations across multiple
/// entities over time (e.g., daily stock prices for multiple stocks).
#[derive(Debug, Clone)]
pub struct PanelDataInfo {
    /// Name of the group/entity column (e.g., "stock_code", "user_id")
    pub group_column: String,
    /// Name of the date/time column
    pub date_column: String,
    /// Number of unique groups/entities
    pub num_groups: usize,
    /// Average number of observations per group
    pub avg_observations_per_group: f32,
    /// Minimum observations in any group
    pub min_observations_per_group: usize,
    /// Maximum observations in any group
    pub max_observations_per_group: usize,
    /// Confidence score for this detection [0, 1]
    pub confidence: f32,
    /// Detected time granularity
    pub time_granularity: TimeGranularity,
}

impl std::fmt::Display for PanelDataInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Panel data: {} groups by '{}', ordered by '{}' ({} granularity, {:.1} avg obs/group)",
            self.num_groups,
            self.group_column,
            self.date_column,
            self.time_granularity,
            self.avg_observations_per_group
        )
    }
}

/// Data type classification for a column
///
/// Text vs Categorical distinction is critical:
/// - Categorical: Low cardinality strings that can be encoded (city names, categories)
/// - Text: High cardinality + long strings that need embeddings (reviews, descriptions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnDataType {
    /// Numeric column (int or float)
    Numeric,
    /// Categorical column - low cardinality strings that can be encoded
    Categorical,
    /// Text column - high cardinality + long strings (DROP for v1, needs NLP)
    Text,
    /// DateTime column - can extract seasonal features
    DateTime,
    /// Boolean column - binary indicator
    Boolean,
}

impl ColumnDataType {
    /// Detect data type from a Polars column
    ///
    /// Heuristic for Text vs Categorical:
    /// - String + cardinality > 100 + avg_length > 20 → Text
    /// - String + cardinality <= 100 OR avg_length <= 20 → Categorical
    pub fn detect(column: &Column, num_rows: usize) -> Self {
        let dtype = column.dtype();

        // Check for boolean first
        if matches!(dtype, PolarsDataType::Boolean) {
            return ColumnDataType::Boolean;
        }

        // Check for datetime
        if matches!(
            dtype,
            PolarsDataType::Date
                | PolarsDataType::Datetime(_, _)
                | PolarsDataType::Time
                | PolarsDataType::Duration(_)
        ) {
            return ColumnDataType::DateTime;
        }

        // Check for numeric
        if is_numeric(column) {
            return ColumnDataType::Numeric;
        }

        // Check for string/categorical
        if is_categorical(column) {
            // Detect Text vs Categorical
            let cardinality = column.n_unique().unwrap_or(0);

            // For high cardinality, check average string length
            if cardinality > 100 {
                let avg_len = Self::estimate_avg_string_length(column);
                if avg_len > 20.0 {
                    return ColumnDataType::Text;
                }
            }

            // Also check if cardinality ratio is very high (potential ID)
            let cardinality_ratio = cardinality as f32 / num_rows.max(1) as f32;
            if cardinality_ratio > 0.9 && cardinality > 50 {
                // High cardinality ratio suggests ID-like or text
                let avg_len = Self::estimate_avg_string_length(column);
                if avg_len > 15.0 {
                    return ColumnDataType::Text;
                }
            }

            return ColumnDataType::Categorical;
        }

        // Default to categorical for unknown types
        ColumnDataType::Categorical
    }

    /// Estimate average string length by sampling
    fn estimate_avg_string_length(column: &Column) -> f32 {
        let Ok(str_col) = column.str() else {
            return 0.0;
        };

        let mut total_len = 0usize;
        let mut count = 0usize;
        let sample_size = 100.min(str_col.len());

        for i in 0..sample_size {
            if let Some(s) = str_col.get(i) {
                total_len += s.len();
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_len as f32 / count as f32
        }
    }
}

/// Reason why a column should be dropped
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropReason {
    /// Column has only one unique value
    Constant,
    /// Column appears to be an ID (high cardinality + monotonic)
    IdLike,
    /// Column is text that requires NLP (not supported in v1)
    Text,
    /// Column is the target (excluded from features)
    Target,
}

impl std::fmt::Display for DropReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DropReason::Constant => write!(f, "constant (single value)"),
            DropReason::IdLike => write!(f, "ID-like (unique per row)"),
            DropReason::Text => write!(f, "text (requires NLP, not supported)"),
            DropReason::Target => write!(f, "target column"),
        }
    }
}

/// Column marked for dropping with reason
#[derive(Debug, Clone)]
pub struct DroppedColumn {
    pub name: String,
    pub reason: DropReason,
}

/// Profile of a single column
///
/// Contains both cheap stats (Pass 1) and expensive stats (Pass 2).
/// Expensive stats are `Option` because they're only computed for kept columns.
#[derive(Debug, Clone)]
pub struct ColumnProfile {
    /// Column name
    pub name: String,
    /// Detected data type
    pub dtype: ColumnDataType,

    // === Pass 1: Cheap stats (Polars native) ===
    /// Fraction of missing values [0, 1]
    pub missing_ratio: f32,
    /// Number of unique values
    pub cardinality: usize,
    /// Cardinality / total rows
    pub cardinality_ratio: f32,
    /// True if column has only one unique value
    pub is_constant: bool,
    /// True if column appears to be an ID (high cardinality + monotonic)
    pub is_id_like: bool,

    // === Pass 2: Expensive stats (only for kept columns) ===
    /// Skewness (only for Numeric)
    pub skewness: Option<f32>,
    /// Kurtosis (only for Numeric)
    pub kurtosis: Option<f32>,
    /// Minimum value (only for Numeric)
    pub min: Option<f64>,
    /// Maximum value (only for Numeric)
    pub max: Option<f64>,
    /// True if column contains negative values
    pub has_negative: bool,
    /// True if column has exactly 2 unique values
    pub is_binary: bool,
    /// Correlation with target (computed with target)
    pub target_correlation: Option<f32>,
}

impl ColumnProfile {
    /// Run Pass 1 analysis (cheap stats using Polars native functions)
    fn analyze_pass1(column: &Column, num_rows: usize) -> Self {
        let name = column.name().to_string();
        let dtype = ColumnDataType::detect(column, num_rows);

        // Cheap stats using Polars native
        let null_count = column.null_count();
        let missing_ratio = null_count as f32 / num_rows.max(1) as f32;

        let cardinality = column.n_unique().unwrap_or(0);
        let cardinality_ratio = cardinality as f32 / num_rows.max(1) as f32;

        let is_constant = cardinality <= 1;

        // ID-like detection: Use precise row ID check for numeric columns
        // This avoids false positives where valid continuous features (e.g., [0.0, 0.1, 0.2, ...])
        // were incorrectly classified as IDs just because they were monotonic and high-cardinality
        let is_id_like = if dtype == ColumnDataType::Numeric && cardinality_ratio > 0.9 {
            // Check both data pattern AND column name to avoid false positives
            Self::check_is_row_id(column) && Self::has_id_like_name(&name)
        } else {
            cardinality_ratio > 0.95 // For non-numeric, just use very high cardinality
        };

        // Quick binary check
        let is_binary = cardinality == 2;

        // Quick negative check for numeric
        let has_negative = if dtype == ColumnDataType::Numeric {
            Self::extract_scalar_f64(&column.min_reduce().ok()).is_some_and(|v| v < 0.0)
        } else {
            false
        };

        // Min/max from Polars (cheap)
        let (min, max) = if dtype == ColumnDataType::Numeric {
            let min_val = Self::extract_scalar_f64(&column.min_reduce().ok());
            let max_val = Self::extract_scalar_f64(&column.max_reduce().ok());
            (min_val, max_val)
        } else {
            (None, None)
        };

        Self {
            name,
            dtype,
            missing_ratio,
            cardinality,
            cardinality_ratio,
            is_constant,
            is_id_like,
            skewness: None,
            kurtosis: None,
            min,
            max,
            has_negative,
            is_binary,
            target_correlation: None,
        }
    }

    /// Extract f64 from Polars Scalar
    fn extract_scalar_f64(scalar: &Option<Scalar>) -> Option<f64> {
        scalar.as_ref().and_then(|s| {
            // Get the AnyValue from the scalar and convert to f64
            let av = s.value();
            match av {
                AnyValue::Float64(v) => Some(*v),
                AnyValue::Float32(v) => Some(*v as f64),
                AnyValue::Int64(v) => Some(*v as f64),
                AnyValue::Int32(v) => Some(*v as f64),
                AnyValue::Int16(v) => Some(*v as f64),
                AnyValue::Int8(v) => Some(*v as f64),
                AnyValue::UInt64(v) => Some(*v as f64),
                AnyValue::UInt32(v) => Some(*v as f64),
                AnyValue::UInt16(v) => Some(*v as f64),
                AnyValue::UInt8(v) => Some(*v as f64),
                _ => None,
            }
        })
    }

    /// Run Pass 2 analysis (expensive stats - only for kept columns)
    fn analyze_pass2(&mut self, column: &Column, target: Option<&[f32]>) {
        if self.dtype != ColumnDataType::Numeric {
            return;
        }

        // Compute skewness and kurtosis
        let values: Vec<f64> = column
            .cast(&PolarsDataType::Float64)
            .ok()
            .and_then(|c| c.f64().ok().map(|ca| ca.iter().flatten().collect()))
            .unwrap_or_default();

        if values.len() < 3 {
            return;
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std > 1e-10 {
            // Skewness: E[(X - μ)³] / σ³
            let m3 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>()
                / n;
            self.skewness = Some(m3 as f32);

            // Kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
            let m4 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f64>()
                / n;
            self.kurtosis = Some((m4 - 3.0) as f32);
        }

        // Compute target correlation if target provided
        if let Some(target) = target {
            if values.len() == target.len() {
                self.target_correlation =
                    Some(crate::analysis::compute_correlation_mixed(&values, target));
            }
        }
    }

    /// Check if column name suggests it's an ID column
    ///
    /// Common patterns: id, ID, index, idx, row_id, _id, etc.
    fn has_id_like_name(name: &str) -> bool {
        let lower = name.to_lowercase();

        // Exact matches
        if lower == "id"
            || lower == "index"
            || lower == "idx"
            || lower == "row"
            || lower == "row_id"
            || lower == "rowid"
            || lower == "row_num"
            || lower == "row_number"
        {
            return true;
        }

        // Patterns
        if lower.ends_with("_id")
            || lower.ends_with("id")
            || lower.starts_with("id_")
            || lower.ends_with("_index")
        {
            return true;
        }

        false
    }

    /// Check if column is a row ID (sequential integers starting from 0 or 1)
    ///
    /// True row IDs have these properties:
    /// - Integer values (or floats with .0)
    /// - Start from 0 or 1
    /// - Increment by exactly 1
    /// - Strongly correlated with row index
    fn check_is_row_id(column: &Column) -> bool {
        let Ok(values) = column.cast(&PolarsDataType::Float64) else {
            return false;
        };
        let Ok(ca) = values.f64() else {
            return false;
        };

        let mut expected_val = None;
        let mut is_sequential = true;
        let mut sample_count = 0;
        let max_samples = 1000;

        for (row_idx, opt_val) in ca.iter().enumerate() {
            let row_idx = row_idx as i64;
            if sample_count >= max_samples {
                break;
            }

            if let Some(val) = opt_val {
                // Check if value is an integer (allow small floating point error)
                if (val - val.round()).abs() > 1e-9 {
                    return false; // Not an integer, can't be a row ID
                }

                let int_val = val.round() as i64;

                // Initialize expected value on first sample
                if expected_val.is_none() {
                    // Row IDs typically start from 0 or 1
                    if int_val != row_idx && int_val != row_idx + 1 {
                        return false;
                    }
                    expected_val = Some(int_val);
                }

                // Check if value matches expected sequential value
                if let Some(expected) = expected_val {
                    let expected_at_this_row = expected + row_idx;
                    if int_val != expected_at_this_row {
                        is_sequential = false;
                        break;
                    }
                }

                sample_count += 1;
            }
        }

        is_sequential && sample_count > 10
    }

    /// Check if this column should be dropped
    pub fn should_drop(&self) -> Option<DropReason> {
        if self.is_constant {
            Some(DropReason::Constant)
        } else if self.is_id_like {
            Some(DropReason::IdLike)
        } else if self.dtype == ColumnDataType::Text {
            Some(DropReason::Text)
        } else {
            None
        }
    }
}

/// Task type detected from target column
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Continuous target variable
    Regression,
    /// Binary classification (2 classes)
    BinaryClassification,
    /// Multi-class classification
    MultiClassification { num_classes: usize },
}

impl TaskType {
    /// Detect task type from target values
    ///
    /// Heuristic:
    /// - 2 unique values that are close to integers → Binary classification
    /// - 3-20 unique values that are close to integers → Multi-class classification
    /// - Otherwise → Regression
    pub fn detect(target: &[f32]) -> Self {
        // Check if values are integer-like (for classification)
        // A value is "integer-like" if it's very close to an integer (within 0.01)
        let integer_like_count = target
            .iter()
            .filter(|&&v| (v - v.round()).abs() < 0.01)
            .count();

        let integer_like_ratio = integer_like_count as f32 / target.len() as f32;

        // If most values are NOT integer-like, it's clearly regression
        if integer_like_ratio < 0.9 {
            return TaskType::Regression;
        }

        // Count unique integer values for classification
        let unique_ints: HashSet<i32> = target
            .iter()
            .filter_map(|&v| {
                if (v - v.round()).abs() < 0.01 {
                    Some(v.round() as i32)
                } else {
                    None
                }
            })
            .collect();

        match unique_ints.len() {
            2 => TaskType::BinaryClassification,
            3..=20 => TaskType::MultiClassification {
                num_classes: unique_ints.len(),
            },
            _ => TaskType::Regression,
        }
    }

    /// Get default loss function name for this task type
    pub fn default_loss_name(&self) -> &'static str {
        match self {
            TaskType::Regression => "mse",
            TaskType::BinaryClassification => "logloss",
            TaskType::MultiClassification { .. } => "softmax",
        }
    }

    /// Get default evaluation metric name for this task type
    pub fn default_metric_name(&self) -> &'static str {
        match self {
            TaskType::Regression => "rmse",
            TaskType::BinaryClassification => "auc",
            TaskType::MultiClassification { .. } => "mlogloss",
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Regression => write!(f, "Regression"),
            TaskType::BinaryClassification => write!(f, "Binary Classification"),
            TaskType::MultiClassification { num_classes } => {
                write!(f, "{}-Class Classification", num_classes)
            }
        }
    }
}

/// Complete profile of a DataFrame
#[derive(Debug, Clone)]
pub struct DataFrameProfile {
    /// Profiles for all columns (excluding dropped)
    pub columns: Vec<ColumnProfile>,
    /// Columns marked for dropping
    pub drop_columns: Vec<DroppedColumn>,
    /// Total number of rows
    pub num_rows: usize,
    /// Number of numeric features (after drops)
    pub num_numeric: usize,
    /// Number of categorical features (after drops)
    pub num_categorical: usize,
    /// Profile of target column
    pub target_profile: ColumnProfile,
    /// Detected task type
    pub task_type: TaskType,
    /// Target column name
    pub target_name: String,
}

impl DataFrameProfile {
    /// Analyze a DataFrame with two-pass strategy
    ///
    /// Pass 1: Cheap stats to identify drops (~1ms per column)
    /// Pass 2: Expensive stats only for kept columns
    pub fn analyze(df: &DataFrame, target_col: &str) -> Result<Self> {
        Self::analyze_with_options(df, target_col, false)
    }

    pub fn analyze_with_options(
        df: &DataFrame,
        target_col: &str,
        skip_correlations: bool,
    ) -> Result<Self> {
        let num_rows = df.height();

        // Get target column and values
        let target_column = df.column(target_col).map_err(|e| {
            TreeBoostError::Data(format!("Target column '{}' not found: {}", target_col, e))
        })?;

        let target_values: Vec<f32> = target_column
            .cast(&PolarsDataType::Float64)
            .map_err(|e| TreeBoostError::Data(format!("Cannot convert target to numeric: {}", e)))?
            .f64()
            .map_err(|e| TreeBoostError::Data(format!("Target column error: {}", e)))?
            .iter()
            .map(|opt| opt.map(|v| v as f32).unwrap_or(f32::NAN))
            .collect();

        // Detect task type
        let task_type = TaskType::detect(&target_values);

        // Profile target column
        let mut target_profile = ColumnProfile::analyze_pass1(target_column, num_rows);
        target_profile.analyze_pass2(target_column, None);

        // === Pass 1: Cheap stats for all columns ===
        let mut columns = Vec::new();
        let mut drop_columns = Vec::new();

        for name in df.get_column_names() {
            let name_str = name.to_string();

            // Skip target column
            if name_str == target_col {
                drop_columns.push(DroppedColumn {
                    name: name_str,
                    reason: DropReason::Target,
                });
                continue;
            }

            let column = df.column(name).unwrap();
            let profile = ColumnProfile::analyze_pass1(column, num_rows);

            // Check if should drop
            if let Some(reason) = profile.should_drop() {
                drop_columns.push(DroppedColumn {
                    name: name_str,
                    reason,
                });
            } else {
                columns.push(profile);
            }
        }

        // === Pass 2: Expensive stats for kept columns ===
        // OPTIMIZATION: Skip O(n*cols) correlation computation when not needed for mode selection
        for profile in &mut columns {
            let column = df.column(&profile.name).unwrap();
            if skip_correlations {
                profile.analyze_pass2(column, None); // Skip correlations
            } else {
                profile.analyze_pass2(column, Some(&target_values)); // Include correlations
            }
        }

        // Count by type
        let num_numeric = columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Numeric)
            .count();
        let num_categorical = columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Categorical)
            .count();

        Ok(Self {
            columns,
            drop_columns,
            num_rows,
            num_numeric,
            num_categorical,
            target_profile,
            task_type,
            target_name: target_col.to_string(),
        })
    }

    /// Get names of columns to keep (not dropped)
    pub fn kept_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|p| p.name.as_str()).collect()
    }

    /// Get names of numeric columns
    pub fn numeric_column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Numeric)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Get names of categorical columns
    pub fn categorical_column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Categorical)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Get column profile by name
    pub fn get_column(&self, name: &str) -> Option<&ColumnProfile> {
        self.columns.iter().find(|p| p.name == name)
    }

    /// Get columns with high skewness (candidates for YeoJohnson)
    pub fn high_skew_columns(&self, threshold: f32) -> Vec<&ColumnProfile> {
        self.columns
            .iter()
            .filter(|p| p.skewness.map(|s| s.abs() > threshold).unwrap_or(false))
            .collect()
    }

    /// Get columns with high missing ratio
    pub fn high_missing_columns(&self, threshold: f32) -> Vec<&ColumnProfile> {
        self.columns
            .iter()
            .filter(|p| p.missing_ratio > threshold)
            .collect()
    }

    /// Get columns strongly correlated with target
    pub fn correlated_columns(&self, threshold: f32) -> Vec<&ColumnProfile> {
        self.columns
            .iter()
            .filter(|p| {
                p.target_correlation
                    .map(|c| c.abs() > threshold)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Get names of datetime columns
    pub fn datetime_column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::DateTime)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Detect panel data structure in the dataset
    ///
    /// Panel data has observations across multiple entities over time.
    /// This method looks for:
    /// 1. A datetime column (the time dimension)
    /// 2. A categorical column with moderate cardinality (the entity/group dimension)
    ///
    /// Returns `Some(PanelDataInfo)` if panel structure is detected with sufficient confidence.
    pub fn detect_panel_structure(&self, df: &DataFrame) -> Option<PanelDataInfo> {
        // Step 1: Find datetime columns (DateTime dtype OR numeric/categorical columns with date-like names)
        let datetime_cols: Vec<&ColumnProfile> = self
            .columns
            .iter()
            .filter(|p| {
                if p.dtype == ColumnDataType::DateTime {
                    true
                } else if is_date_column_name(&p.name) {
                    // Accept Numeric (includes i64 dates)
                    if p.dtype == ColumnDataType::Numeric {
                        return true;
                    }
                    // For Categorical: only accept if cardinality > MIN_DATE_CARDINALITY
                    // (low cardinality = categorical time period like "morning/afternoon", not timestamps)
                    if p.dtype == ColumnDataType::Categorical {
                        return p.cardinality > crate::defaults::analysis::MIN_DATE_CARDINALITY;
                    }
                    false
                } else {
                    false
                }
            })
            .collect();

        if datetime_cols.is_empty() {
            return None;
        }

        // Step 2: Find candidate group columns (categorical with moderate cardinality)
        // Good candidates: cardinality 2-10000, cardinality_ratio < 0.5
        // Increased threshold to 10000 to support large stock datasets (thousands of tickers)
        let group_candidates: Vec<&ColumnProfile> = self
            .columns
            .iter()
            .filter(|p| {
                p.dtype == ColumnDataType::Categorical
                    && p.cardinality >= 2
                    && p.cardinality <= 10000  // Increased from 1000 to support stock data
                    && p.cardinality_ratio < 0.5
            })
            .collect();

        if group_candidates.is_empty() {
            return None;
        }

        // Step 3: Score each (group_col, date_col) combination
        let mut best_score = 0.0f32;
        let mut best_info: Option<PanelDataInfo> = None;

        for date_col in &datetime_cols {
            for group_col in &group_candidates {
                if let Some(info) =
                    self.score_panel_candidate(df, &group_col.name, &date_col.name, group_col)
                {
                    if info.confidence > best_score {
                        best_score = info.confidence;
                        best_info = Some(info);
                    }
                }
            }
        }

        // Only return if confidence > 0.3 (lowered from 0.5 to support stock data)
        // Stock/financial panel data often has irregular observations per group,
        // leading to lower confidence scores (0.4-0.5 range)
        if let Some(ref info) = best_info {
            if info.confidence <= 0.3 {
                return None;
            }
        }

        best_info
    }

    /// Score a panel data candidate (group_col, date_col pair)
    fn score_panel_candidate(
        &self,
        df: &DataFrame,
        group_col: &str,
        date_col: &str,
        group_profile: &ColumnProfile,
    ) -> Option<PanelDataInfo> {
        // Get the columns
        let group_series = df.column(group_col).ok()?;
        let date_series = df.column(date_col).ok()?;

        let num_rows = df.height();
        let num_groups = group_profile.cardinality;

        if num_groups == 0 {
            return None;
        }

        // Compute observations per group
        let avg_obs = num_rows as f32 / num_groups as f32;

        // Compute min/max observations per group by counting
        let (min_obs, max_obs) = Self::compute_group_obs_range(group_series, num_groups);

        // Detect time granularity
        let time_granularity = Self::detect_time_granularity(date_series);

        // Compute confidence score based on multiple factors:
        // 1. Avg observations per group (more = better, up to a point)
        // 2. Balance across groups (min_obs / max_obs)
        // 3. Reasonable cardinality ratio

        let obs_score = (avg_obs.min(100.0) / 100.0).sqrt(); // 0-1, favors >=100 obs/group
        let balance_score = if max_obs > 0 {
            (min_obs as f32 / max_obs as f32).sqrt()
        } else {
            0.0
        };
        let cardinality_score = 1.0 - group_profile.cardinality_ratio; // Lower ratio = better

        // Weight the scores
        let confidence = 0.4 * obs_score + 0.3 * balance_score + 0.3 * cardinality_score;

        Some(PanelDataInfo {
            group_column: group_col.to_string(),
            date_column: date_col.to_string(),
            num_groups,
            avg_observations_per_group: avg_obs,
            min_observations_per_group: min_obs,
            max_observations_per_group: max_obs,
            confidence,
            time_granularity,
        })
    }

    /// Compute min/max observations per group
    fn compute_group_obs_range(group_series: &Column, num_groups: usize) -> (usize, usize) {
        use std::collections::HashMap;

        // Count observations per group
        let mut counts: HashMap<String, usize> = HashMap::with_capacity(num_groups);

        // Sample for efficiency if large dataset
        let sample_size = group_series.len().min(100_000);
        let step = group_series.len() / sample_size.max(1);

        for i in (0..group_series.len()).step_by(step.max(1)) {
            if let Ok(val) = group_series.get(i) {
                let key = format!("{:?}", val);
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        if counts.is_empty() {
            return (0, 0);
        }

        let min_obs = *counts.values().min().unwrap_or(&0);
        let max_obs = *counts.values().max().unwrap_or(&0);

        // Scale back if we sampled
        let scale = group_series.len() as f32 / sample_size as f32;
        (
            (min_obs as f32 * scale) as usize,
            (max_obs as f32 * scale) as usize,
        )
    }

    /// Detect time granularity from a datetime column
    fn detect_time_granularity(date_series: &Column) -> TimeGranularity {
        // Try to compute median time difference between consecutive observations
        let dtype = date_series.dtype();

        // Extract timestamps as f64 (seconds or days depending on type)
        let timestamps: Vec<f64> = match dtype {
            PolarsDataType::Date => {
                // Date is days since epoch
                date_series
                    .cast(&PolarsDataType::Int32)
                    .ok()
                    .and_then(|c| {
                        c.i32()
                            .ok()
                            .map(|ca| ca.iter().flatten().map(|v| v as f64).collect())
                    })
                    .unwrap_or_default()
            }
            PolarsDataType::Datetime(_, _) => {
                // Datetime is microseconds since epoch, convert to days
                date_series
                    .cast(&PolarsDataType::Int64)
                    .ok()
                    .and_then(|c| {
                        c.i64().ok().map(|ca| {
                            ca.iter()
                                .flatten()
                                .map(|v| v as f64 / (1_000_000.0 * 86400.0)) // microseconds to days
                                .collect()
                        })
                    })
                    .unwrap_or_default()
            }
            _ => return TimeGranularity::Unknown,
        };

        if timestamps.len() < 2 {
            return TimeGranularity::Unknown;
        }

        // Sort timestamps and compute differences
        let mut sorted = timestamps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute differences (sample for efficiency)
        let mut diffs: Vec<f64> = Vec::new();
        let step = (sorted.len() / 1000).max(1);
        for i in (1..sorted.len()).step_by(step) {
            let diff = sorted[i] - sorted[i - 1];
            if diff > 0.0 {
                diffs.push(diff);
            }
        }

        if diffs.is_empty() {
            return TimeGranularity::Unknown;
        }

        // Get median difference (in days)
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_diff_days = diffs[diffs.len() / 2];

        // Classify based on median difference
        // Note: These are in days
        if median_diff_days < 1.0 / 24.0 {
            // < 1 hour
            TimeGranularity::SubHourly
        } else if median_diff_days < 0.5 {
            // < 12 hours
            TimeGranularity::Hourly
        } else if median_diff_days < 3.0 {
            // < 3 days
            TimeGranularity::Daily
        } else if median_diff_days < 14.0 {
            // < 2 weeks
            TimeGranularity::Weekly
        } else if median_diff_days < 60.0 {
            // < 2 months
            TimeGranularity::Monthly
        } else {
            TimeGranularity::Yearly
        }
    }

    /// Generate human-readable report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("┌─────────────────────────────────────────────────────────────────┐\n");
        report.push_str("│                    DataFrame Profile                            │\n");
        report.push_str("├─────────────────────────────────────────────────────────────────┤\n");

        // Summary
        report.push_str(&format!(
            "│ Rows: {:>10}    Features: {:>3} ({} numeric, {} categorical)    │\n",
            self.num_rows,
            self.columns.len(),
            self.num_numeric,
            self.num_categorical
        ));
        report.push_str(&format!(
            "│ Target: {} ({})                                     │\n",
            self.target_name, self.task_type
        ));

        // Dropped columns
        if !self.drop_columns.is_empty() {
            report
                .push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report
                .push_str("│ Dropped Columns                                                 │\n");
            for col in &self.drop_columns {
                if col.reason != DropReason::Target {
                    report.push_str(&format!(
                        "│   • {} ({})                              │\n",
                        col.name, col.reason
                    ));
                }
            }
        }

        // High skewness warning
        let skewed = self.high_skew_columns(2.0);
        if !skewed.is_empty() {
            report
                .push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ High Skewness (consider YeoJohnson)                            │\n");
            for col in skewed.iter().take(5) {
                report.push_str(&format!(
                    "│   • {} (skew={:.2})                                     │\n",
                    col.name,
                    col.skewness.unwrap_or(0.0)
                ));
            }
        }

        // High missing warning
        let missing = self.high_missing_columns(0.05);
        if !missing.is_empty() {
            report
                .push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ High Missing Values (>5%)                                      │\n");
            for col in missing.iter().take(5) {
                report.push_str(&format!(
                    "│   • {} ({:.1}% missing)                                 │\n",
                    col.name,
                    col.missing_ratio * 100.0
                ));
            }
        }

        // Top correlated features
        let mut correlated: Vec<_> = self
            .columns
            .iter()
            .filter(|p| p.target_correlation.is_some())
            .collect();
        correlated.sort_by(|a, b| {
            b.target_correlation
                .unwrap_or(0.0)
                .abs()
                .partial_cmp(&a.target_correlation.unwrap_or(0.0).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if !correlated.is_empty() {
            report
                .push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ Top Correlated Features                                        │\n");
            for col in correlated.iter().take(5) {
                report.push_str(&format!(
                    "│   • {} (r={:.3})                                       │\n",
                    col.name,
                    col.target_correlation.unwrap_or(0.0)
                ));
            }
        }

        report.push_str("└─────────────────────────────────────────────────────────────────┘\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_detect_numeric() {
        let col: Column = Series::new("test".into(), vec![1.0f64, 2.0, 3.0]).into();
        assert_eq!(ColumnDataType::detect(&col, 3), ColumnDataType::Numeric);
    }

    #[test]
    fn test_data_type_detect_categorical() {
        let col: Column = Series::new("test".into(), vec!["a", "b", "c", "a", "b"]).into();
        assert_eq!(ColumnDataType::detect(&col, 5), ColumnDataType::Categorical);
    }

    #[test]
    fn test_task_type_regression() {
        let target = vec![1.5, 2.3, 3.1, 4.7, 5.2, 6.8, 7.1, 8.9, 9.0, 10.1];
        let mut more_values = target.clone();
        for i in 11..100 {
            more_values.push(i as f32 * 0.1);
        }
        assert_eq!(TaskType::detect(&more_values), TaskType::Regression);
    }

    #[test]
    fn test_task_type_binary() {
        let target = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(TaskType::detect(&target), TaskType::BinaryClassification);
    }

    #[test]
    fn test_task_type_multiclass() {
        let target = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            TaskType::detect(&target),
            TaskType::MultiClassification { num_classes: 5 }
        );
    }

    #[test]
    fn test_column_profile_constant() {
        let col: Column = Series::new("test".into(), vec![5.0f64, 5.0, 5.0, 5.0]).into();
        let profile = ColumnProfile::analyze_pass1(&col, 4);
        assert!(profile.is_constant);
        assert_eq!(profile.should_drop(), Some(DropReason::Constant));
    }

    #[test]
    fn test_column_profile_id_like() {
        let col: Column = Series::new("id".into(), vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10]).into();
        let profile = ColumnProfile::analyze_pass1(&col, 10);
        // High cardinality ratio + monotonic = ID-like
        assert!(profile.cardinality_ratio > 0.9);
    }

    #[test]
    fn test_dataframe_profile() {
        let df = DataFrame::new(
            5,
            vec![
                Series::new("feature1".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
                Series::new("feature2".into(), vec!["a", "b", "a", "b", "a"]).into(),
                Series::new("constant".into(), vec![1.0f64, 1.0, 1.0, 1.0, 1.0]).into(),
                Series::new("target".into(), vec![0.0f64, 1.0, 0.0, 1.0, 0.0]).into(),
            ],
        )
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();

        // Should have 2 features (constant dropped, target excluded)
        assert_eq!(profile.columns.len(), 2);
        assert_eq!(profile.num_numeric, 1);
        assert_eq!(profile.num_categorical, 1);

        // Task type should be binary classification
        assert_eq!(profile.task_type, TaskType::BinaryClassification);

        // Constant column should be dropped
        assert!(profile
            .drop_columns
            .iter()
            .any(|d| d.name == "constant" && d.reason == DropReason::Constant));
    }

    #[test]
    fn test_pearson_correlation() {
        use crate::analysis::compute_correlation_mixed;

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];
        let r = compute_correlation_mixed(&x, &y);
        assert!((r - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let y_neg = vec![10.0f32, 8.0, 6.0, 4.0, 2.0];
        let r_neg = compute_correlation_mixed(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_time_granularity_suggestions() {
        // Daily granularity
        let lags = TimeGranularity::Daily.suggested_lag_periods();
        assert!(lags.contains(&1));
        assert!(lags.contains(&7));

        let windows = TimeGranularity::Daily.suggested_rolling_windows();
        assert!(windows.contains(&7));
        assert!(windows.contains(&14));
    }

    #[test]
    fn test_panel_data_detection() {
        use chrono::{Datelike, NaiveDate};

        // Create panel-like data: 2 stocks × 5 days = 10 rows
        let dates: Vec<i32> = [
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
        ]
        .iter()
        .map(|d| d.num_days_from_ce())
        .collect();

        let df = DataFrame::new(
            10,
            vec![
                Series::new(
                    "stock_code".into(),
                    vec![
                        "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL", "GOOGL",
                        "GOOGL",
                    ],
                )
                .into(),
                Series::new("date".into(), dates)
                    .cast(&PolarsDataType::Date)
                    .unwrap()
                    .into(),
                Series::new(
                    "price".into(),
                    vec![
                        150.0f64, 151.0, 152.0, 151.5, 153.0, 2800.0, 2810.0, 2805.0, 2820.0,
                        2830.0,
                    ],
                )
                .into(),
                Series::new(
                    "target".into(),
                    vec![
                        0.01f64, 0.02, -0.01, 0.01, 0.02, 0.01, -0.01, 0.02, 0.01, 0.01,
                    ],
                )
                .into(),
            ],
        )
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();

        // Detect panel structure
        let panel_info = profile.detect_panel_structure(&df);

        assert!(panel_info.is_some(), "Should detect panel structure");
        let info = panel_info.unwrap();

        assert_eq!(info.group_column, "stock_code");
        assert_eq!(info.date_column, "date");
        assert_eq!(info.num_groups, 2);
        assert!((info.avg_observations_per_group - 5.0).abs() < 0.1);
        assert!(info.confidence > 0.5);
    }

    #[test]
    fn test_no_panel_without_datetime() {
        // Data without datetime column - should not detect panel
        let df = DataFrame::new(
            5,
            vec![
                Series::new("category".into(), vec!["A", "A", "B", "B", "C"]).into(),
                Series::new("value".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
                Series::new("target".into(), vec![0.0f64, 1.0, 0.0, 1.0, 0.0]).into(),
            ],
        )
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();
        let panel_info = profile.detect_panel_structure(&df);

        assert!(
            panel_info.is_none(),
            "Should not detect panel without datetime"
        );
    }

    #[test]
    fn test_no_panel_without_group() {
        use chrono::{Datelike, NaiveDate};

        // Data with datetime but no group column
        let dates: Vec<i32> = (0..5)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, i + 1)
                    .unwrap()
                    .num_days_from_ce()
            })
            .collect();

        let df = DataFrame::new(
            5,
            vec![
                Series::new("date".into(), dates)
                    .cast(&PolarsDataType::Date)
                    .unwrap()
                    .into(),
                Series::new("value".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
                Series::new("target".into(), vec![0.0f64, 1.0, 0.0, 1.0, 0.0]).into(),
            ],
        )
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();
        let panel_info = profile.detect_panel_structure(&df);

        assert!(
            panel_info.is_none(),
            "Should not detect panel without group column"
        );
    }
}
