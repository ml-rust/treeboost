//! Feature selection and extraction utilities
//!
//! This module provides shared utility functions for feature selection and extraction
//! that are used across different parts of the codebase (AutoBuilder, UniversalModel, etc.).

use crate::analysis::PanelDataInfo;
use crate::features::{
    FeatureGenerator, GroupedTimeSeriesConfig, GroupedTimeSeriesGenerator, InteractionGenerator,
    NaNStrategy, PolynomialGenerator, RatioGenerator, TimeSeriesFeaturePlan,
};
use crate::preprocessing::polars_ext::is_numeric;
use crate::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Extract selected features from raw feature array based on feature indices.
///
/// This utility is used in LinearThenTree mode to select a subset of features for
/// the linear model while trees use all features.
///
/// # Arguments
///
/// * `raw_features` - Flat array of features in row-major order `[row0_feat0, row0_feat1, ..., row1_feat0, ...]`
/// * `num_rows` - Number of rows in the dataset
/// * `num_raw_features` - Total number of features per row in raw_features
/// * `indices` - Optional feature indices to select. If None, returns all features.
///
/// # Returns
///
/// A new feature array containing only the selected features in row-major order.
///
/// # Examples
///
/// ```ignore
/// let raw = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];  // 2 rows, 3 features
/// let selected = extract_selected_features(&raw, 2, 3, Some(&[0, 2]));
/// // Result: [1.0, 3.0,  4.0, 6.0]  // Features 0 and 2 for each row
/// ```
pub fn extract_selected_features(
    raw_features: &[f32],
    num_rows: usize,
    num_raw_features: usize,
    indices: Option<&[usize]>,
) -> Vec<f32> {
    if let Some(indices) = indices {
        let mut selected = Vec::with_capacity(num_rows * indices.len());
        for row in 0..num_rows {
            let row_offset = row * num_raw_features;
            for &feat_idx in indices {
                selected.push(raw_features[row_offset + feat_idx]);
            }
        }
        selected
    } else {
        raw_features.to_vec()
    }
}

/// Apply time-series features to a DataFrame (panel data with groups × time)
///
/// This is a public utility that can be used by AutoBuilder, examples, or user code.
/// Generates lag, rolling, and EWMA features within each group using the efficient
/// GroupedTimeSeriesGenerator.
///
/// # Arguments
///
/// * `df` - Input DataFrame with panel data (groups × time)
/// * `ts_plan` - Time-series feature configuration (lag periods, rolling windows, etc.)
/// * `panel_info` - Panel structure info (group column, date column)
/// * `fast_mode` - If true, uses reduced features for large datasets (>1M rows)
///
/// # Returns
///
/// New DataFrame with original columns + generated time-series features
///
/// # Example
///
/// ```ignore
/// use treeboost::features::apply_timeseries_features;
///
/// let df_with_ts = apply_timeseries_features(
///     df,
///     &ts_plan,
///     &panel_info,
///     false, // Use full feature set
/// )?;
/// ```
pub fn apply_timeseries_features(
    df: DataFrame,
    ts_plan: &TimeSeriesFeaturePlan,
    panel_info: &PanelDataInfo,
    fast_mode: bool,
) -> Result<DataFrame> {
    let num_rows = df.height();

    // Extract group IDs (convert string codes to u32 indices)
    let code_col = df.column(&panel_info.group_column)?;
    let code_series = code_col.as_materialized_series();

    // Build code -> group_id mapping
    let mut code_to_id: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0u32;

    let group_ids: Vec<u32> = code_series
        .str()?
        .iter()
        .map(|v| {
            let code = v.unwrap_or("");
            *code_to_id.entry(code.to_string()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    // Extract timestamps (convert date column to f64)
    let date_col = df.column(&panel_info.date_column)?;
    let timestamps: Vec<f64> = date_col
        .as_materialized_series()
        .cast(&DataType::Float64)?
        .f64()?
        .iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();

    // Get numeric feature columns to transform
    let feature_cols: Vec<String> = ts_plan.lag_columns.clone();
    let num_features = feature_cols.len();

    if num_features == 0 {
        return Ok(df);
    }

    // OPTIMIZED: Extract columns ONCE, not millions of times!
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

    // Create and configure the generator
    let config = if fast_mode && num_rows > 1_000_000 {
        // Fast mode for large datasets: fewer features
        GroupedTimeSeriesConfig {
            lag_periods: vec![1, 7],
            rolling_windows: vec![7],
            rolling_stats: vec![crate::features::RollingStat::Mean],
            ewma_alphas: vec![0.1],
            momentum_periods: vec![1, 7],
            nan_strategy: NaNStrategy::Keep,
            min_periods: 1,
        }
    } else {
        // Full feature set
        GroupedTimeSeriesConfig {
            lag_periods: ts_plan.lag_periods.clone(),
            rolling_windows: ts_plan.rolling_windows.clone(),
            rolling_stats: ts_plan.rolling_stats.clone(),
            ewma_alphas: ts_plan.ewma_alphas.clone(),
            momentum_periods: ts_plan.momentum_periods.clone(),
            nan_strategy: NaNStrategy::Keep,
            min_periods: 1,
        }
    };

    let mut generator = GroupedTimeSeriesGenerator::new(config);

    // Fit and transform
    generator.fit(
        &group_ids,
        &timestamps,
        (0..num_features).collect(),
        feature_cols,
    )?;
    let (new_features, new_names) = generator.transform(&feature_data, num_features)?;

    if new_features.is_empty() {
        return Ok(df);
    }

    let num_new_features = new_names.len();

    // OPTIMIZED: Build all Series at once, then horizontal stack
    let mut all_series: Vec<Series> = Vec::with_capacity(num_new_features);

    for feat_idx in 0..num_new_features {
        let mut col_data = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            col_data.push(new_features[row * num_new_features + feat_idx]);
        }
        let series = Series::new(new_names[feat_idx].clone().into(), col_data);
        all_series.push(series);
    }

    // Create new DataFrame with all time-series features
    let columns: Vec<_> = all_series.into_iter().map(|s| s.into_column()).collect();
    let ts_df = DataFrame::new_infer_height(columns)?;

    // Horizontal stack - much faster than repeated with_column()
    let new_df = df.hstack(ts_df.columns())?;

    Ok(new_df)
}

/// Apply polynomial features to a DataFrame (x², x³, √x, log(x+1))
///
/// This is a public utility that can be used by AutoBuilder, examples, or user code.
/// Generates polynomial transformations of numeric columns using PolynomialGenerator.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `generator` - Configured PolynomialGenerator (use PolynomialGenerator::new(), .all(), etc.)
/// * `columns` - Column names to transform (None = all numeric columns)
///
/// # Returns
///
/// New DataFrame with original columns + generated polynomial features
///
/// # Example
///
/// ```ignore
/// use treeboost::features::apply_polynomial_features;
/// use treeboost::features::PolynomialGenerator;
///
/// let poly = PolynomialGenerator::new() // x² and √x
///     .with_cube()                      // Add x³
///     .with_log1p();                    // Add log(x+1)
///
/// let df_with_poly = apply_polynomial_features(
///     df,
///     &poly,
///     Some(vec!["price".to_string(), "volume".to_string()]), // Transform only these
/// )?;
/// ```
pub fn apply_polynomial_features(
    df: DataFrame,
    generator: &PolynomialGenerator,
    columns: Option<Vec<String>>,
) -> Result<DataFrame> {
    let num_rows = df.height();

    // Determine which columns to transform
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
        return Ok(df);
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

    // Generate polynomial features
    let (new_features, new_names) = generator.generate(&feature_data, num_features, &feature_cols);

    if new_features.is_empty() {
        return Ok(df);
    }

    let num_new_features = new_names.len();

    // Build all Series at once, then horizontal stack
    let mut all_series: Vec<Series> = Vec::with_capacity(num_new_features);

    for feat_idx in 0..num_new_features {
        let mut col_data = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            col_data.push(new_features[row * num_new_features + feat_idx]);
        }
        let series = Series::new(new_names[feat_idx].clone().into(), col_data);
        all_series.push(series);
    }

    // Create new DataFrame with polynomial features
    let columns: Vec<_> = all_series.into_iter().map(|s| s.into_column()).collect();
    let poly_df = DataFrame::new_infer_height(columns)?;

    // Horizontal stack - O(n) operation
    let new_df = df.hstack(poly_df.columns())?;

    Ok(new_df)
}

/// Apply ratio features to a DataFrame (x_i / x_j)
///
/// This is a public utility that can be used by AutoBuilder, examples, or user code.
/// Generates ratio features for pairs of numeric columns using RatioGenerator.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `generator` - Configured RatioGenerator (use RatioGenerator::from_pairs(), .auto_select(), etc.)
/// * `columns` - Column names to use for ratios (None = all numeric columns)
///
/// # Returns
///
/// New DataFrame with original columns + generated ratio features
///
/// # Example
///
/// ```ignore
/// use treeboost::features::apply_ratio_features;
/// use treeboost::features::RatioGenerator;
///
/// // Option 1: Explicit pairs
/// let ratio = RatioGenerator::from_pairs(vec![(0, 1), (1, 2)]);
///
/// // Option 2: Auto-select based on correlation
/// let ratio = RatioGenerator::auto_select(&data, num_features, 3);
///
/// let df_with_ratios = apply_ratio_features(df, &ratio, None)?;
/// ```
pub fn apply_ratio_features(
    df: DataFrame,
    generator: &RatioGenerator,
    columns: Option<Vec<String>>,
) -> Result<DataFrame> {
    let num_rows = df.height();

    // Determine which columns to use
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
        return Ok(df);
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

    // Generate ratio features
    let (new_features, new_names) = generator.generate(&feature_data, num_features, &feature_cols);

    if new_features.is_empty() {
        return Ok(df);
    }

    let num_new_features = new_names.len();

    // Build all Series at once, then horizontal stack
    let mut all_series: Vec<Series> = Vec::with_capacity(num_new_features);

    for feat_idx in 0..num_new_features {
        let mut col_data = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            col_data.push(new_features[row * num_new_features + feat_idx]);
        }
        let series = Series::new(new_names[feat_idx].clone().into(), col_data);
        all_series.push(series);
    }

    // Create new DataFrame with ratio features
    let columns: Vec<_> = all_series.into_iter().map(|s| s.into_column()).collect();
    let ratio_df = DataFrame::new_infer_height(columns)?;

    // Horizontal stack - O(n) operation
    let new_df = df.hstack(ratio_df.columns())?;

    Ok(new_df)
}

/// Apply interaction features to a DataFrame (x_i × x_j, x_i + x_j, etc.)
///
/// This is a public utility that can be used by AutoBuilder, examples, or user code.
/// Generates interaction features for pairs of numeric columns using InteractionGenerator.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `generator` - Configured InteractionGenerator (use InteractionGenerator::from_pairs(), .top_correlated(), etc.)
/// * `columns` - Column names to use for interactions (None = all numeric columns)
///
/// # Returns
///
/// New DataFrame with original columns + generated interaction features
///
/// # Example
///
/// ```ignore
/// use treeboost::features::apply_interaction_features;
/// use treeboost::features::{InteractionGenerator, InteractionType};
///
/// // Option 1: Explicit pairs with multiple interaction types
/// let inter = InteractionGenerator::from_pairs(vec![(0, 1), (1, 2)])
///     .with_types(vec![InteractionType::Multiply, InteractionType::Add]);
///
/// // Option 2: Auto-select top correlated pairs
/// let inter = InteractionGenerator::top_correlated(20);
///
/// let df_with_interactions = apply_interaction_features(df, &inter, None)?;
/// ```
pub fn apply_interaction_features(
    df: DataFrame,
    generator: &InteractionGenerator,
    columns: Option<Vec<String>>,
) -> Result<DataFrame> {
    let num_rows = df.height();

    // Determine which columns to use
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
        return Ok(df);
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

    // Generate interaction features
    let (new_features, new_names) = generator.generate(&feature_data, num_features, &feature_cols);

    if new_features.is_empty() {
        return Ok(df);
    }

    let num_new_features = new_names.len();

    // Build all Series at once, then horizontal stack
    let mut all_series: Vec<Series> = Vec::with_capacity(num_new_features);

    for feat_idx in 0..num_new_features {
        let mut col_data = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            col_data.push(new_features[row * num_new_features + feat_idx]);
        }
        let series = Series::new(new_names[feat_idx].clone().into(), col_data);
        all_series.push(series);
    }

    // Create new DataFrame with interaction features
    let columns: Vec<_> = all_series.into_iter().map(|s| s.into_column()).collect();
    let interaction_df = DataFrame::new_infer_height(columns)?;

    // Horizontal stack - O(n) operation
    let new_df = df.hstack(interaction_df.columns())?;

    Ok(new_df)
}

/// Apply cross-sectional ranking features to panel data
///
/// Transforms numeric features to be relative to their cross-section (e.g., all stocks on same date).
/// Critical for ranking models where relative position matters (Rank IC).
///
/// # Arguments
///
/// * `df` - Input DataFrame with panel data
/// * `group_col` - Column defining cross-sections (e.g., "date")
/// * `exclude_cols` - Columns to exclude (IDs, targets)
///
/// # Transformations
///
/// For each numeric column, generates:
/// - `{col}_rank`: Percentile rank [0, 1]
/// - `{col}_zscore`: Standardized value within cross-section
/// - `{col}_vs_median`: Distance from cross-sectional median
///
/// # Example
///
/// ```ignore
/// let df = apply_crosssectional_features(&df, "date", &["code", "date", "y"])?;
/// ```
pub fn apply_crosssectional_features(
    df: &DataFrame,
    group_col: &str,
    exclude_cols: &[&str],
) -> Result<DataFrame> {
    crate::features::crosssectional::apply_crosssectional_features(df, group_col, exclude_cols)
        .map_err(|e| crate::TreeBoostError::Data(e.to_string()))
}

/// Apply cross-sectional features to specific columns only
///
/// Same as `apply_crosssectional_features` but allows specifying which columns to include.
/// Useful when you only want cross-sectional features for original features, not derived ones.
///
/// # Example
///
/// ```ignore
/// // Only apply to original features f_0-f_6, not to lag/rolling features
/// let original_features = vec!["f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6"];
/// let df = apply_crosssectional_features_selective(
///     &df,
///     "date",
///     &["code", "date", "y"],
///     Some(&original_features),
/// )?;
/// ```
pub fn apply_crosssectional_features_selective(
    df: &DataFrame,
    group_col: &str,
    exclude_cols: &[&str],
    include_cols: Option<&[&str]>,
) -> Result<DataFrame> {
    crate::features::crosssectional::apply_crosssectional_features_with_include(
        df,
        group_col,
        exclude_cols,
        include_cols,
    )
    .map_err(|e| crate::TreeBoostError::Data(e.to_string()))
}

/// Apply a complete feature engineering plan to a DataFrame.
///
/// This is the canonical function for applying feature engineering. It consolidates all
/// feature generation logic in one place to ensure consistency between training and inference.
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `feature_plan` - Feature engineering plan containing polynomial, ratio, interaction, and time-series features
/// * `panel_info` - Optional panel data information (for time-series features only)
///
/// # Returns
///
/// Transformed DataFrame with all engineered features added.
///
/// # Example
///
/// ```ignore
/// // During training: AutoBuilder discovers the plan
/// let plan = feature_plan.clone();
///
/// // During inference: Apply the same plan to new data
/// let test_df = apply_feature_plan(test_df, Some(&plan), None)?;
/// ```
pub fn apply_feature_plan(
    mut df: DataFrame,
    feature_plan: Option<&crate::features::FeaturePlan>,
    panel_info: Option<&PanelDataInfo>,
) -> Result<DataFrame> {
    // Early return if no plan
    let plan = match feature_plan {
        Some(p) if !p.is_empty() => p,
        _ => return Ok(df),
    };

    // 1. Apply polynomial features (x², sqrt, log)
    if !plan.polynomial_features.is_empty() {
        let poly_gen = PolynomialGenerator::all();
        df = apply_polynomial_features(df, &poly_gen, Some(plan.polynomial_features.clone()))?;
        tracing::debug!(
            num_features = plan.polynomial_features.len(),
            "Applied polynomial features"
        );
    }

    // 2. Apply ratio features (x_i / x_j)
    if !plan.ratio_pairs.is_empty() {
        // Get numeric columns for name-to-index mapping
        let numeric_cols: Vec<String> = df
            .columns()
            .iter()
            .filter(|col| col.dtype().is_numeric())
            .map(|col| col.name().to_string())
            .collect();

        // Convert named pairs to index pairs
        let mut index_pairs = Vec::new();
        for (col_a, col_b) in &plan.ratio_pairs {
            if let (Some(idx_a), Some(idx_b)) = (
                numeric_cols.iter().position(|c| c == col_a),
                numeric_cols.iter().position(|c| c == col_b),
            ) {
                index_pairs.push((idx_a, idx_b));
            }
        }

        if !index_pairs.is_empty() {
            let ratio_gen = RatioGenerator::from_pairs(index_pairs);
            df = apply_ratio_features(df, &ratio_gen, None)?;
            tracing::debug!(
                num_ratios = plan.ratio_pairs.len(),
                "Applied ratio features"
            );
        }
    }

    // 3. Apply interaction features (x_i * x_j)
    if !plan.interaction_pairs.is_empty() {
        // Get numeric columns for name-to-index mapping
        let numeric_cols: Vec<String> = df
            .columns()
            .iter()
            .filter(|col| col.dtype().is_numeric())
            .map(|col| col.name().to_string())
            .collect();

        // Convert named pairs to index pairs
        let mut index_pairs = Vec::new();
        for (col_a, col_b) in &plan.interaction_pairs {
            if let (Some(idx_a), Some(idx_b)) = (
                numeric_cols.iter().position(|c| c == col_a),
                numeric_cols.iter().position(|c| c == col_b),
            ) {
                index_pairs.push((idx_a, idx_b));
            }
        }

        if !index_pairs.is_empty() {
            let interaction_gen = InteractionGenerator::from_pairs(index_pairs);
            df = apply_interaction_features(df, &interaction_gen, None)?;
            tracing::debug!(
                num_interactions = plan.interaction_pairs.len(),
                "Applied interaction features"
            );
        }
    }

    // 4. Apply time-series features (lags, rolling, EWMA)
    if let Some(ts_plan) = &plan.timeseries_features {
        // Require panel_info for time-series features
        let panel_info = panel_info.ok_or_else(|| {
            crate::TreeBoostError::Data(
                "Time-series features require panel_info (group and date columns)".to_string(),
            )
        })?;

        df = apply_timeseries_features(df, ts_plan, panel_info, true /* fast_mode */)?;

        // Fill nulls created by lags with 0 (safe for tree models)
        // Nulls appear for rows without sufficient history (e.g., first rows of each group)
        // Only fill nulls/NaN on numeric columns (string columns don't support fill_nan)
        let numeric_fill: Vec<_> = df
            .columns()
            .iter()
            .filter(|c| c.dtype().is_numeric())
            .map(|c| col(c.name().clone()).fill_null(lit(0)).fill_nan(lit(0.0)))
            .collect();
        if !numeric_fill.is_empty() {
            df = df
                .lazy()
                .with_columns(numeric_fill)
                .collect()
                .map_err(|e| {
                    crate::TreeBoostError::Data(format!(
                        "Failed to fill nulls after feature engineering: {}",
                        e
                    ))
                })?;
        }

        tracing::debug!(
            num_lags = ts_plan.lag_periods.len(),
            num_rolling = ts_plan.rolling_windows.len(),
            num_ewma = ts_plan.ewma_alphas.len(),
            "Applied time-series features"
        );
    }

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_all_features() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = extract_selected_features(&raw, 2, 3, None);
        assert_eq!(result, raw);
    }

    #[test]
    fn test_extract_selected_features() {
        // 2 rows, 3 features per row
        let raw = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];

        // Select features 0 and 2
        let indices = vec![0, 2];
        let result = extract_selected_features(&raw, 2, 3, Some(&indices));

        assert_eq!(
            result,
            vec![
                1.0, 3.0, // row 0: features 0, 2
                4.0, 6.0, // row 1: features 0, 2
            ]
        );
    }

    #[test]
    fn test_extract_single_feature() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![1];
        let result = extract_selected_features(&raw, 2, 3, Some(&indices));
        assert_eq!(result, vec![2.0, 5.0]); // Feature 1 from each row
    }

    #[test]
    fn test_extract_reordered_features() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![2, 0]; // Reverse order
        let result = extract_selected_features(&raw, 2, 3, Some(&indices));
        assert_eq!(result, vec![3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn test_apply_polynomial_features() {
        let df = df! {
            "x" => &[1.0f32, 2.0, 4.0],
            "y" => &[10.0f32, 20.0, 30.0],
        }
        .unwrap();

        let gen = PolynomialGenerator::new(); // Default: x², sqrt
        let result =
            apply_polynomial_features(df.clone(), &gen, Some(vec!["x".to_string()])).unwrap();

        // Should have original 2 columns + 2 new polynomial features for 'x'
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 3);
        let col_names: Vec<String> = result
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(col_names.contains(&"x_sq".to_string()));
        assert!(col_names.contains(&"x_sqrt".to_string()));
    }

    #[test]
    fn test_apply_ratio_features() {
        let df = df! {
            "a" => &[10.0f32, 20.0, 30.0],
            "b" => &[2.0f32, 4.0, 5.0],
        }
        .unwrap();

        let gen = RatioGenerator::from_pairs(vec![(0, 1)]); // a / b
        let result = apply_ratio_features(df.clone(), &gen, None).unwrap();

        // Should have original 2 columns + 1 ratio column
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        let col_names: Vec<String> = result
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(col_names.contains(&"a_div_b".to_string()));
    }

    #[test]
    fn test_apply_interaction_features() {
        let df = df! {
            "a" => &[1.0f32, 2.0, 3.0],
            "b" => &[4.0f32, 5.0, 6.0],
        }
        .unwrap();

        let gen = InteractionGenerator::from_pairs(vec![(0, 1)]); // a * b by default
        let result = apply_interaction_features(df.clone(), &gen, None).unwrap();

        // Should have original 2 columns + 1 interaction column
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        let col_names: Vec<String> = result
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(col_names.contains(&"a_mul_b".to_string()));
    }
}
