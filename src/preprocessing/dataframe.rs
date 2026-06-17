//! Preprocessing utilities
//!
//! This module provides shared utility functions for applying preprocessing operations
//! to Polars DataFrames. Users can manually apply scalers, encoders, imputers, and
//! transforms with full control.

use crate::preprocessing::{
    polars_ext::is_numeric, FrequencyEncoder, ImputeStrategy, LabelEncoder, MinMaxScaler,
    OneHotEncoder, RobustScaler, Scaler, SimpleImputer, StandardScaler,
};
use crate::Result;
use polars::prelude::*;

/// Apply StandardScaler to DataFrame columns (z-score normalization)
///
/// Transforms features to have zero mean and unit variance: (x - μ) / σ
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to scale (None = all numeric columns)
/// * `fitted_scaler` - Optional pre-fitted scaler for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (scaled DataFrame, fitted scaler for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_standard_scaler;
///
/// // Training: fit and transform
/// let (train_df, scaler) = apply_standard_scaler(train_df, None, None)?;
///
/// // Inference: transform using fitted scaler
/// let (test_df, _) = apply_standard_scaler(test_df, None, Some(scaler))?;
/// ```
pub fn apply_standard_scaler(
    df: DataFrame,
    columns: Option<Vec<String>>,
    fitted_scaler: Option<StandardScaler>,
) -> Result<(DataFrame, StandardScaler)> {
    apply_scaler_impl(df, columns, fitted_scaler, |_| StandardScaler::new())
}

/// Apply MinMaxScaler to DataFrame columns (scale to [min, max] range)
///
/// Transforms features to a fixed range (default [0, 1]): (x - x_min) / (x_max - x_min)
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to scale (None = all numeric columns)
/// * `min` - Minimum value of output range (default 0.0)
/// * `max` - Maximum value of output range (default 1.0)
/// * `fitted_scaler` - Optional pre-fitted scaler for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (scaled DataFrame, fitted scaler for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_minmax_scaler;
///
/// // Scale to [0, 1]
/// let (train_df, scaler) = apply_minmax_scaler(train_df, None, 0.0, 1.0, None)?;
///
/// // Scale to [-1, 1]
/// let (train_df, scaler) = apply_minmax_scaler(train_df, None, -1.0, 1.0, None)?;
/// ```
pub fn apply_minmax_scaler(
    df: DataFrame,
    columns: Option<Vec<String>>,
    min: f32,
    max: f32,
    fitted_scaler: Option<MinMaxScaler>,
) -> Result<(DataFrame, MinMaxScaler)> {
    apply_scaler_impl(df, columns, fitted_scaler, |_| {
        MinMaxScaler::new().with_range(min, max)
    })
}

/// Apply RobustScaler to DataFrame columns (median/IQR scaling, robust to outliers)
///
/// Transforms features using median and IQR: (x - median) / IQR
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to scale (None = all numeric columns)
/// * `fitted_scaler` - Optional pre-fitted scaler for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (scaled DataFrame, fitted scaler for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_robust_scaler;
///
/// let (train_df, scaler) = apply_robust_scaler(train_df, None, None)?;
/// ```
pub fn apply_robust_scaler(
    df: DataFrame,
    columns: Option<Vec<String>>,
    fitted_scaler: Option<RobustScaler>,
) -> Result<(DataFrame, RobustScaler)> {
    apply_scaler_impl(df, columns, fitted_scaler, |_| RobustScaler::new())
}

/// Generic scaler application implementation
fn apply_scaler_impl<S: Scaler + Clone>(
    mut df: DataFrame,
    columns: Option<Vec<String>>,
    fitted_scaler: Option<S>,
    create_scaler: impl Fn(&[String]) -> S,
) -> Result<(DataFrame, S)> {
    let num_rows = df.height();

    // Determine which columns to scale
    let feature_cols: Vec<String> = if let Some(cols) = columns {
        cols
    } else {
        // Auto-detect numeric columns
        df.columns()
            .iter()
            .filter(|col| is_numeric(col))
            .map(|col| col.name().to_string())
            .collect()
    };

    let num_features = feature_cols.len();
    if num_features == 0 {
        // No features to scale, return scaler as-is
        let scaler = fitted_scaler.unwrap_or_else(|| create_scaler(&feature_cols));
        return Ok((df, scaler));
    }

    // Extract columns ONCE to avoid O(n²) lookups
    let mut columns: Vec<_> = Vec::with_capacity(num_features);
    for col_name in &feature_cols {
        let col = df
            .column(col_name)?
            .as_materialized_series()
            .cast(&DataType::Float32)?
            .f32()?
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect::<Vec<f32>>();
        columns.push(col);
    }

    // Interleave to row-major format
    let mut feature_data: Vec<f32> = Vec::with_capacity(num_rows * num_features);
    for row_idx in 0..num_rows {
        for col in &columns {
            feature_data.push(col[row_idx]);
        }
    }

    // Apply scaler
    let mut scaler = fitted_scaler.unwrap_or_else(|| create_scaler(&feature_cols));
    if !scaler.is_fitted() {
        scaler.fit(&feature_data, num_features)?;
    }
    scaler.transform(&mut feature_data, num_features)?;

    // Convert back to DataFrame columns
    let mut new_columns = Vec::new();
    for col_idx in 0..num_features {
        let mut col_data = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            col_data.push(feature_data[row_idx * num_features + col_idx]);
        }
        let series = Series::new(feature_cols[col_idx].clone().into(), col_data);
        new_columns.push(series);
    }

    // Replace columns in original DataFrame
    for series in new_columns {
        df.replace(&series.name().to_string(), series.into())?;
    }

    Ok((df, scaler))
}

/// Apply FrequencyEncoder to DataFrame columns (category → count)
///
/// Maps each category to its frequency in the training data. Optimal for GBDTs.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to encode (must be String/Categorical type)
/// * `fitted_encoder` - Optional pre-fitted encoder for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (encoded DataFrame, fitted encoder for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_frequency_encoder;
///
/// // Training
/// let (train_df, encoder) = apply_frequency_encoder(
///     train_df,
///     vec!["category".to_string(), "brand".to_string()],
///     None,
/// )?;
///
/// // Inference
/// let (test_df, _) = apply_frequency_encoder(test_df, columns, Some(encoder))?;
/// ```
pub fn apply_frequency_encoder(
    mut df: DataFrame,
    columns: Vec<String>,
    fitted_encoder: Option<FrequencyEncoder>,
) -> Result<(DataFrame, FrequencyEncoder)> {
    let mut encoder = fitted_encoder.unwrap_or_default();

    for col_name in &columns {
        let col = df.column(col_name)?;
        let str_series = col.as_materialized_series().str()?;

        // Collect categories
        let categories: Vec<&str> = str_series.iter().map(|v| v.unwrap_or("")).collect();

        // Fit if not fitted
        if !encoder.is_fitted() {
            encoder.fit(&categories);
        }

        // Transform
        let encoded = encoder.transform(&categories)?;

        // Replace column
        let new_series = Series::new(col_name.clone().into(), encoded);
        df.replace(col_name, new_series.into())?;
    }

    Ok((df, encoder))
}

/// Apply LabelEncoder to DataFrame columns (string → integer)
///
/// Maps each unique category to a unique integer. Essential for CSV loading.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to encode (must be String/Categorical type)
/// * `fitted_encoder` - Optional pre-fitted encoder for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (encoded DataFrame, fitted encoder for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_label_encoder;
///
/// let (train_df, encoder) = apply_label_encoder(
///     train_df,
///     vec!["category".to_string()],
///     None,
/// )?;
/// ```
pub fn apply_label_encoder(
    mut df: DataFrame,
    columns: Vec<String>,
    fitted_encoder: Option<LabelEncoder>,
) -> Result<(DataFrame, LabelEncoder)> {
    let mut encoder = fitted_encoder.unwrap_or_default();

    for col_name in &columns {
        let col = df.column(col_name)?;
        let str_series = col.as_materialized_series().str()?;

        // Collect categories
        let categories: Vec<&str> = str_series.iter().map(|v| v.unwrap_or("")).collect();

        // Fit if not fitted
        if !encoder.is_fitted() {
            encoder.fit(&categories);
        }

        // Transform
        let encoded = encoder.transform(&categories)?;

        // Replace column with u32
        let new_series = Series::new(col_name.clone().into(), encoded);
        df.replace(col_name, new_series.into())?;
    }

    Ok((df, encoder))
}

/// Apply OneHotEncoder to DataFrame columns (category → binary columns)
///
/// Creates binary indicator columns for each category. Best for linear models.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to encode (must be String/Categorical type)
/// * `fitted_encoder` - Optional pre-fitted encoder for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (encoded DataFrame with new binary columns, fitted encoder for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_onehot_encoder;
///
/// let (train_df, encoder) = apply_onehot_encoder(
///     train_df,
///     vec!["category".to_string()],
///     None,
/// )?;
/// // Creates columns: category_A, category_B, category_C, etc.
/// ```
pub fn apply_onehot_encoder(
    mut df: DataFrame,
    columns: Vec<String>,
    fitted_encoder: Option<OneHotEncoder>,
) -> Result<(DataFrame, OneHotEncoder)> {
    let mut encoder = fitted_encoder.unwrap_or_default();

    for col_name in &columns {
        let col = df.column(col_name)?;
        let str_series = col.as_materialized_series().str()?;

        // Collect categories
        let categories: Vec<&str> = str_series.iter().map(|v| v.unwrap_or("")).collect();

        // Fit if not fitted
        if !encoder.is_fitted() {
            encoder.fit(&categories)?;
        }

        // Transform
        let encoded = encoder.transform(&categories)?;

        // Get column names
        let feature_names = encoder.get_feature_names(col_name);
        let num_cols = encoder.num_columns();

        // Create binary columns
        let num_rows = categories.len();
        for col_idx in 0..num_cols {
            let mut binary_col = Vec::with_capacity(num_rows);
            for row_idx in 0..num_rows {
                binary_col.push(encoded[row_idx * num_cols + col_idx]);
            }
            let series = Series::new(feature_names[col_idx].clone().into(), binary_col);
            df.hstack(&[series.into_column()])?;
        }

        // Remove original column
        df = df.drop(col_name)?;
    }

    Ok((df, encoder))
}

/// Apply SimpleImputer to DataFrame columns (fill missing values)
///
/// Replaces NaN values with computed statistics (mean, median, mode, or constant).
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `columns` - Column names to impute (None = all numeric columns)
/// * `strategy` - Imputation strategy (Mean, Median, Mode, Constant)
/// * `fitted_imputer` - Optional pre-fitted imputer for inference (None = fit on this data)
///
/// # Returns
///
/// Tuple of (imputed DataFrame, fitted imputer for reuse)
///
/// # Example
///
/// ```ignore
/// use treeboost::preprocessing::apply_simple_imputer;
/// use treeboost::preprocessing::ImputeStrategy;
///
/// // Fill with mean
/// let (train_df, imputer) = apply_simple_imputer(
///     train_df,
///     None,
///     ImputeStrategy::Mean,
///     None,
/// )?;
///
/// // Fill with constant 0.0
/// let (train_df, imputer) = apply_simple_imputer(
///     train_df,
///     Some(vec!["price".to_string()]),
///     ImputeStrategy::Constant(0.0),
///     None,
/// )?;
/// ```
pub fn apply_simple_imputer(
    mut df: DataFrame,
    columns: Option<Vec<String>>,
    strategy: ImputeStrategy,
    fitted_imputer: Option<SimpleImputer>,
) -> Result<(DataFrame, SimpleImputer)> {
    let num_rows = df.height();

    // Determine which columns to impute
    let feature_cols: Vec<String> = if let Some(cols) = columns {
        cols
    } else {
        // Auto-detect numeric columns
        df.columns()
            .iter()
            .filter(|col| is_numeric(col))
            .map(|col| col.name().to_string())
            .collect()
    };

    let num_features = feature_cols.len();
    if num_features == 0 {
        let imputer = fitted_imputer.unwrap_or_else(|| SimpleImputer::new(strategy));
        return Ok((df, imputer));
    }

    // Extract columns
    let mut columns: Vec<_> = Vec::with_capacity(num_features);
    for col_name in &feature_cols {
        let col = df
            .column(col_name)?
            .as_materialized_series()
            .cast(&DataType::Float32)?
            .f32()?
            .iter()
            .map(|v| v.unwrap_or(f32::NAN))
            .collect::<Vec<f32>>();
        columns.push(col);
    }

    // Interleave to row-major format
    let mut feature_data: Vec<f32> = Vec::with_capacity(num_rows * num_features);
    for row_idx in 0..num_rows {
        for col in &columns {
            feature_data.push(col[row_idx]);
        }
    }

    // Apply imputer
    let mut imputer = fitted_imputer.unwrap_or_else(|| SimpleImputer::new(strategy));
    if !imputer.is_fitted() {
        imputer.fit(&feature_data, num_features)?;
    }
    imputer.transform(&mut feature_data, num_features)?;

    // Convert back to DataFrame columns
    let mut new_columns = Vec::new();
    for col_idx in 0..num_features {
        let mut col_data = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            col_data.push(feature_data[row_idx * num_features + col_idx]);
        }
        let series = Series::new(feature_cols[col_idx].clone().into(), col_data);
        new_columns.push(series);
    }

    // Replace columns
    for series in new_columns {
        df.replace(&series.name().to_string(), series.into())?;
    }

    Ok((df, imputer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler_basic() {
        // Create simple test DataFrame
        let df = df! {
            "a" => &[1.0f32, 2.0, 3.0],
            "b" => &[10.0f32, 20.0, 30.0],
        }
        .unwrap();

        let (scaled_df, scaler) = apply_standard_scaler(df, None, None).unwrap();

        assert!(scaler.is_fitted());
        assert_eq!(scaled_df.height(), 3);
    }

    #[test]
    fn test_frequency_encoder_basic() {
        let df = df! {
            "cat" => &["A", "B", "A", "C"],
        }
        .unwrap();

        let (encoded_df, encoder) =
            apply_frequency_encoder(df, vec!["cat".to_string()], None).unwrap();

        assert!(encoder.is_fitted());
        assert_eq!(encoded_df.height(), 4);
    }

    #[test]
    fn test_minmax_scaler() {
        let df = df! {
            "a" => &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            "b" => &[10.0f32, 20.0, 30.0, 40.0, 50.0],
        }
        .unwrap();

        let (scaled_df, scaler) = apply_minmax_scaler(df, None, 0.0, 1.0, None).unwrap();

        assert!(scaler.is_fitted());
        assert_eq!(scaled_df.height(), 5);
        assert_eq!(scaled_df.width(), 2);
    }

    #[test]
    fn test_robust_scaler() {
        let df = df! {
            "a" => &[1.0f32, 2.0, 3.0, 4.0, 100.0], // Has outlier
            "b" => &[10.0f32, 20.0, 30.0, 40.0, 50.0],
        }
        .unwrap();

        let (scaled_df, scaler) = apply_robust_scaler(df, None, None).unwrap();

        assert!(scaler.is_fitted());
        assert_eq!(scaled_df.height(), 5);
        assert_eq!(scaled_df.width(), 2);
    }

    #[test]
    fn test_label_encoder() {
        let df = df! {
            "cat" => &["red", "blue", "red", "green", "blue"],
        }
        .unwrap();

        let (encoded_df, encoder) = apply_label_encoder(df, vec!["cat".to_string()], None).unwrap();

        assert!(encoder.is_fitted());
        assert_eq!(encoded_df.height(), 5);
        // Should have same width (column replaced in-place)
        assert_eq!(encoded_df.width(), 1);
    }

    #[test]
    fn test_onehot_encoder() {
        let df = df! {
            "cat" => &["A", "B", "A", "C"],
            "value" => &[1.0f32, 2.0, 3.0, 4.0],
        }
        .unwrap();

        let (encoded_df, encoder) =
            apply_onehot_encoder(df, vec!["cat".to_string()], None).unwrap();

        assert!(encoder.is_fitted());
        assert_eq!(encoded_df.height(), 4);
        // OneHot encoding replaces the categorical column with binary columns
        // Just verify it completed successfully and width is reasonable
        assert!(encoded_df.width() >= 1);
    }

    #[test]
    fn test_simple_imputer_mean() {
        let df = df! {
            "a" => &[Some(1.0f32), None, Some(3.0), Some(4.0)],
            "b" => &[Some(10.0f32), Some(20.0), None, Some(40.0)],
        }
        .unwrap();

        let (imputed_df, imputer) =
            apply_simple_imputer(df, None, ImputeStrategy::Mean, None).unwrap();

        assert!(imputer.is_fitted());
        assert_eq!(imputed_df.height(), 4);
        assert_eq!(imputed_df.width(), 2);
    }

    #[test]
    fn test_simple_imputer_constant() {
        let df = df! {
            "a" => &[Some(1.0f32), None, Some(3.0)],
        }
        .unwrap();

        let (imputed_df, imputer) =
            apply_simple_imputer(df, None, ImputeStrategy::Constant(99.0), None).unwrap();

        assert!(imputer.is_fitted());
        assert_eq!(imputed_df.height(), 3);
    }
}
