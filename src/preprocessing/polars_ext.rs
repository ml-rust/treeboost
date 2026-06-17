//! Polars integration helpers for preprocessing
//!
//! This module provides ergonomic conversion helpers for working with
//! Polars DataFrames and Series in preprocessing pipelines.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::preprocessing::{StandardScaler, polars_ext::*};
//! use polars::prelude::*;
//!
//! let df = df!("a" => [1.0, 2.0, 3.0], "b" => [4.0, 5.0, 6.0])?;
//!
//! // Convert DataFrame columns to row-major f32 matrix
//! let (data, num_features) = df_to_features(&df, &["a", "b"])?;
//!
//! // Fit and transform
//! let mut scaler = StandardScaler::new();
//! let mut data = data;
//! scaler.fit_transform(&mut data, num_features)?;
//!
//! // Convert back to DataFrame
//! let scaled_df = features_to_df(&data, num_features, &["a_scaled", "b_scaled"])?;
//! ```

use crate::TreeBoostError;
use polars::prelude::*;

/// Convert a Polars Column to `Vec<f32>`
///
/// Handles numeric types by casting to f64 then f32.
/// NaN values are preserved.
///
/// # Arguments
/// * `column` - The Polars Column to convert
///
/// # Returns
/// `Vec<f32>` with the column values
///
/// # Errors
/// Returns error if column cannot be cast to numeric type
pub fn column_to_f32(column: &Column) -> Result<Vec<f32>, TreeBoostError> {
    let f64_col = column
        .cast(&DataType::Float64)
        .map_err(|e| TreeBoostError::Data(format!("Cannot convert column to f64: {}", e)))?;

    let ca = f64_col
        .f64()
        .map_err(|e| TreeBoostError::Data(format!("Cannot get f64 ChunkedArray: {}", e)))?;

    Ok(ca
        .iter()
        .map(|opt| opt.map(|v| v as f32).unwrap_or(f32::NAN))
        .collect())
}

/// Convert a Polars Column to `Vec<String>`
///
/// Handles string types. Null values become empty strings.
///
/// # Arguments
/// * `column` - The Polars Column to convert
///
/// # Returns
/// `Vec<String>` with the column values
///
/// # Errors
/// Returns error if column cannot be cast to string type
pub fn column_to_strings(column: &Column) -> Result<Vec<String>, TreeBoostError> {
    let str_col = column
        .cast(&DataType::String)
        .map_err(|e| TreeBoostError::Data(format!("Cannot convert column to string: {}", e)))?;

    let ca = str_col
        .str()
        .map_err(|e| TreeBoostError::Data(format!("Cannot get str ChunkedArray: {}", e)))?;

    Ok(ca.iter().map(|opt| opt.unwrap_or("").to_string()).collect())
}

/// Convert a Polars Series to `Vec<f32>`
///
/// Handles numeric types by casting to f64 then f32.
/// NaN values are preserved.
///
/// # Arguments
/// * `series` - The Polars Series to convert
///
/// # Returns
/// `Vec<f32>` with the series values
///
/// # Errors
/// Returns error if series cannot be cast to numeric type
pub fn series_to_f32(series: &Series) -> Result<Vec<f32>, TreeBoostError> {
    let col: Column = series.clone().into();
    column_to_f32(&col)
}

/// Convert a Polars Series to `Vec<String>`
///
/// Handles string types. Null values become empty strings.
///
/// # Arguments
/// * `series` - The Polars Series to convert
///
/// # Returns
/// `Vec<String>` with the series values
///
/// # Errors
/// Returns error if series cannot be cast to string type
pub fn series_to_strings(series: &Series) -> Result<Vec<String>, TreeBoostError> {
    let col: Column = series.clone().into();
    column_to_strings(&col)
}

/// Convert DataFrame columns to row-major f32 matrix
///
/// Extracts specified columns and converts them to a flat `Vec<f32>`
/// in row-major order: [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
///
/// # Arguments
/// * `df` - The Polars DataFrame
/// * `columns` - Column names to extract
///
/// # Returns
/// Tuple of (data, num_features) where data is row-major f32 matrix
///
/// # Errors
/// Returns error if columns don't exist or cannot be converted to f32
pub fn df_to_features(
    df: &DataFrame,
    columns: &[&str],
) -> Result<(Vec<f32>, usize), TreeBoostError> {
    if columns.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let num_features = columns.len();
    let num_rows = df.height();

    // Collect all columns as f32 vecs
    let cols: Vec<Vec<f32>> = columns
        .iter()
        .map(|&name| {
            let column = df
                .column(name)
                .map_err(|e| TreeBoostError::Data(format!("Column '{}' not found: {}", name, e)))?;
            column_to_f32(column)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Convert to row-major
    let mut data = Vec::with_capacity(num_rows * num_features);
    for row in 0..num_rows {
        for col in &cols {
            data.push(col[row]);
        }
    }

    Ok((data, num_features))
}

/// Convert row-major f32 matrix back to DataFrame
///
/// # Arguments
/// * `data` - Row-major f32 matrix
/// * `num_features` - Number of features (columns)
/// * `column_names` - Names for the columns
///
/// # Returns
/// New DataFrame with the specified columns
///
/// # Errors
/// Returns error if data length doesn't match num_features * num_rows
pub fn features_to_df(
    data: &[f32],
    num_features: usize,
    column_names: &[&str],
) -> Result<DataFrame, TreeBoostError> {
    if num_features == 0 {
        return Ok(DataFrame::empty());
    }

    if !data.len().is_multiple_of(num_features) {
        return Err(TreeBoostError::Data(format!(
            "Data length {} not divisible by num_features {}",
            data.len(),
            num_features
        )));
    }

    if column_names.len() != num_features {
        return Err(TreeBoostError::Data(format!(
            "Expected {} column names, got {}",
            num_features,
            column_names.len()
        )));
    }

    let num_rows = data.len() / num_features;

    // Extract each column and convert Series to Column
    let columns: Vec<Column> = (0..num_features)
        .map(|f| {
            let col_data: Vec<f32> = (0..num_rows).map(|r| data[r * num_features + f]).collect();
            let series = Series::new(column_names[f].into(), col_data);
            series.into()
        })
        .collect();

    DataFrame::new_infer_height(columns)
        .map_err(|e| TreeBoostError::Data(format!("Failed to create DataFrame: {}", e)))
}

/// Extract target column as `Vec<f32>`
///
/// Convenience wrapper around column_to_f32 for target extraction.
///
/// # Arguments
/// * `df` - The Polars DataFrame
/// * `target_col` - Name of the target column
///
/// # Returns
/// `Vec<f32>` with target values
pub fn df_to_target(df: &DataFrame, target_col: &str) -> Result<Vec<f32>, TreeBoostError> {
    let column = df.column(target_col).map_err(|e| {
        TreeBoostError::Data(format!("Target column '{}' not found: {}", target_col, e))
    })?;
    column_to_f32(column)
}

/// Get column names from DataFrame
///
/// # Arguments
/// * `df` - The Polars DataFrame
///
/// # Returns
/// Vec of column names as Strings
pub fn df_column_names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Check if a column is numeric (can be converted to f32)
///
/// # Arguments
/// * `column` - The Polars Column to check
///
/// # Returns
/// true if the column is a numeric type
pub fn is_numeric(column: &Column) -> bool {
    matches!(
        column.dtype(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Check if a column is categorical/string
///
/// # Arguments
/// * `column` - The Polars Column to check
///
/// # Returns
/// true if the column is a string or categorical type
pub fn is_categorical(column: &Column) -> bool {
    matches!(
        column.dtype(),
        DataType::String | DataType::Categorical(_, _)
    )
}

/// Separate DataFrame columns into numeric and categorical
///
/// # Arguments
/// * `df` - The Polars DataFrame
/// * `exclude` - Column names to exclude (e.g., target column)
///
/// # Returns
/// Tuple of (numeric_columns, categorical_columns)
pub fn split_by_dtype(df: &DataFrame, exclude: &[&str]) -> (Vec<String>, Vec<String>) {
    let mut numeric = Vec::new();
    let mut categorical = Vec::new();

    for name in df.get_column_names() {
        let name_str = name.to_string();
        if exclude.contains(&name_str.as_str()) {
            continue;
        }

        if let Ok(column) = df.column(name) {
            if is_numeric(column) {
                numeric.push(name_str);
            } else if is_categorical(column) {
                categorical.push(name_str);
            }
        }
    }

    (numeric, categorical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_to_f32_int() {
        let series = Series::new("test".into(), vec![1i32, 2, 3, 4]);
        let result = series_to_f32(&series).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_series_to_f32_float() {
        let series = Series::new("test".into(), vec![1.5f64, 2.5, 3.5]);
        let result = series_to_f32(&series).unwrap();
        assert_eq!(result, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_series_to_f32_with_null() {
        let series = Series::new("test".into(), vec![Some(1.0f64), None, Some(3.0)]);
        let result = series_to_f32(&series).unwrap();
        assert_eq!(result[0], 1.0);
        assert!(result[1].is_nan());
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_series_to_strings() {
        let series = Series::new("test".into(), vec!["a", "b", "c"]);
        let result = series_to_strings(&series).unwrap();
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_df_to_features() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), vec![1.0f64, 2.0, 3.0]).into(),
                Series::new("b".into(), vec![4.0f64, 5.0, 6.0]).into(),
            ],
        )
        .unwrap();

        let (data, num_features) = df_to_features(&df, &["a", "b"]).unwrap();

        assert_eq!(num_features, 2);
        // Row-major: [1, 4, 2, 5, 3, 6]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_features_to_df() {
        let data = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let df = features_to_df(&data, 2, &["a", "b"]).unwrap();

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);

        let col_a = df.column("a").unwrap().f32().unwrap();
        assert_eq!(col_a.get(0), Some(1.0));
        assert_eq!(col_a.get(1), Some(2.0));
        assert_eq!(col_a.get(2), Some(3.0));
    }

    #[test]
    fn test_df_to_target() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("feature".into(), vec![1.0f64, 2.0]).into(),
                Series::new("target".into(), vec![10.0f64, 20.0]).into(),
            ],
        )
        .unwrap();

        let target = df_to_target(&df, "target").unwrap();
        assert_eq!(target, vec![10.0, 20.0]);
    }

    #[test]
    fn test_split_by_dtype() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("num1".into(), vec![1.0f64, 2.0]).into(),
                Series::new("num2".into(), vec![3i32, 4]).into(),
                Series::new("cat1".into(), vec!["a", "b"]).into(),
                Series::new("target".into(), vec![1.0f64, 2.0]).into(),
            ],
        )
        .unwrap();

        let (numeric, categorical) = split_by_dtype(&df, &["target"]);

        assert!(numeric.contains(&"num1".to_string()));
        assert!(numeric.contains(&"num2".to_string()));
        assert!(!numeric.contains(&"target".to_string()));
        assert!(categorical.contains(&"cat1".to_string()));
    }

    #[test]
    fn test_is_numeric() {
        let int_col: Column = Series::new("test".into(), vec![1i32, 2, 3]).into();
        let float_col: Column = Series::new("test".into(), vec![1.0f64, 2.0]).into();
        let str_col: Column = Series::new("test".into(), vec!["a", "b"]).into();

        assert!(is_numeric(&int_col));
        assert!(is_numeric(&float_col));
        assert!(!is_numeric(&str_col));
    }

    #[test]
    fn test_is_categorical() {
        let str_col: Column = Series::new("test".into(), vec!["a", "b"]).into();
        let int_col: Column = Series::new("test".into(), vec![1i32, 2, 3]).into();

        assert!(is_categorical(&str_col));
        assert!(!is_categorical(&int_col));
    }

    #[test]
    fn test_roundtrip() {
        // DataFrame -> features -> DataFrame roundtrip
        let original = DataFrame::new(
            3,
            vec![
                Series::new("x".into(), vec![1.0f64, 2.0, 3.0]).into(),
                Series::new("y".into(), vec![4.0f64, 5.0, 6.0]).into(),
            ],
        )
        .unwrap();

        let (data, num_features) = df_to_features(&original, &["x", "y"]).unwrap();
        let restored = features_to_df(&data, num_features, &["x", "y"]).unwrap();

        assert_eq!(original.height(), restored.height());
        assert_eq!(original.width(), restored.width());
    }
}
