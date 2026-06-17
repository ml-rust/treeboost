//! Cross-sectional features for panel data
//!
//! Transforms features to be relative to their cross-section (e.g., all stocks on the same date).
//! Essential for ranking models where relative position matters more than absolute values.

use polars::prelude::*;

/// Apply cross-sectional transformations to numeric columns
///
/// For panel data (e.g., stocks × dates), transforms features to be relative
/// to their cross-section. This is critical for ranking models (Rank IC).
///
/// # Transformations
///
/// For each numeric column, generates:
/// - `{col}_rank`: Percentile rank within cross-section [0, 1]
/// - `{col}_zscore`: Standardized value (mean=0, std=1) within cross-section
/// - `{col}_vs_median`: Distance from cross-sectional median
///
/// # Arguments
///
/// * `df` - Input DataFrame with panel data
/// * `group_col` - Column defining cross-sections (e.g., "date")
/// * `exclude_cols` - Columns to skip (e.g., IDs, targets)
///
/// # Example
///
/// ```ignore
/// // For stock data: transform features relative to same-day distribution
/// let df = apply_crosssectional_features(
///     &df,
///     "date",
///     &["code", "date", "y"]
/// )?;
///
/// // Now each stock's features are ranked against other stocks that day
/// // f_0_rank: where does this stock's f_0 rank among all stocks today?
/// // f_0_zscore: how many std devs from today's mean?
/// ```
pub fn apply_crosssectional_features(
    df: &DataFrame,
    group_col: &str,
    exclude_cols: &[&str],
) -> PolarsResult<DataFrame> {
    apply_crosssectional_features_with_include(df, group_col, exclude_cols, None)
}

/// Apply cross-sectional transformations to specific columns
///
/// Same as `apply_crosssectional_features` but allows specifying which columns to include.
///
/// # Arguments
///
/// * `df` - Input DataFrame with panel data
/// * `group_col` - Column defining cross-sections (e.g., "date")
/// * `exclude_cols` - Columns to skip (e.g., IDs, targets)
/// * `include_cols` - Optional list of columns to process. If None, processes all numeric columns.
pub fn apply_crosssectional_features_with_include(
    df: &DataFrame,
    group_col: &str,
    exclude_cols: &[&str],
    include_cols: Option<&[&str]>,
) -> PolarsResult<DataFrame> {
    // Get numeric columns
    let numeric_cols: Vec<String> = if let Some(includes) = include_cols {
        // If include_cols specified, use only those
        includes
            .iter()
            .filter(|&&name| {
                if let Ok(col) = df.column(name) {
                    col.dtype().is_numeric() && name != group_col && !exclude_cols.contains(&name)
                } else {
                    false
                }
            })
            .map(|&s| s.to_string())
            .collect()
    } else {
        // Otherwise, process all numeric columns (excluding group and exclusion list)
        df.columns()
            .iter()
            .filter(|col| {
                let name = col.name().as_str();
                col.dtype().is_numeric() && name != group_col && !exclude_cols.contains(&name)
            })
            .map(|col| col.name().to_string())
            .collect()
    };

    if numeric_cols.is_empty() {
        return Ok(df.clone());
    }

    // OPTIMIZED: Use Polars native groupby operations instead of manual iteration
    // This is O(n log n) instead of O(features × groups × rows)
    let mut result = df.clone();

    // HYBRID APPROACH: Use Polars window functions where possible
    // For each numeric column, compute transformations
    for col_name_str in &numeric_cols {
        let col_name = col_name_str.as_str();

        // Use Polars LazyFrame for optimized window functions
        let lazy = result.clone().lazy();

        // Compute group-wise statistics using window functions (FAST!)
        let mean_expr = col(col_name).mean().over([col(group_col)])?;
        let std_expr = col(col_name).std(1).over([col(group_col)])?;
        let median_expr = col(col_name).median().over([col(group_col)])?;

        // Z-score: (x - mean) / std
        let zscore_expr = when(std_expr.clone().gt(lit(0.0)))
            .then((col(col_name) - mean_expr.clone()) / std_expr)
            .otherwise(lit(0.0))
            .alias(format!("{}_zscore", col_name));

        // Median difference: x - median
        let median_diff_expr =
            (col(col_name) - median_expr).alias(format!("{}_vs_median", col_name));

        // Apply window functions
        let result_with_stats = lazy
            .with_columns([zscore_expr, median_diff_expr])
            .collect()?;

        // For ranking, use Polars native rank().over() window function
        // This provides optimal performance for grouped percentile ranking
        let rank_col = compute_group_ranks(&result_with_stats, col_name, group_col)?;

        result = result_with_stats;
        result.with_column(rank_col.into())?;
    }

    Ok(result)
}

/// Compute percentile ranks within groups using Polars window functions
/// Uses native rank() with .over() for optimal performance
fn compute_group_ranks(df: &DataFrame, feature_col: &str, group_col: &str) -> PolarsResult<Series> {
    // RankOptions and RankMethod are re-exported in polars::prelude
    // (already imported at module level with use polars::prelude::*)

    // Use lazy evaluation with window function for rank
    let lazy = df.clone().lazy();

    // Compute rank within each group using window functions (FAST!)
    let result = lazy
        .with_columns([
            // Step 1: Compute 1-based rank within each group
            col(feature_col)
                .rank(
                    RankOptions {
                        method: RankMethod::Average,
                        descending: false,
                    },
                    None,
                )
                .over([col(group_col)])?
                .alias("__temp_rank__"),
            // Step 2: Get group size
            col(feature_col)
                .count()
                .over([col(group_col)])?
                .alias("__group_size__"),
        ])
        .with_columns([
            // Step 3: Convert rank to percentile [0, 1]
            // rank() gives 1-based ranks, convert to 0-1 scale
            when(col("__group_size__").gt(lit(1)))
                .then(
                    (col("__temp_rank__") - lit(1.0))
                        / (col("__group_size__").cast(DataType::Float64) - lit(1.0)),
                )
                .otherwise(lit(0.5)) // Single-element groups get 0.5
                .alias(format!("{}_rank", feature_col)),
        ])
        .select([col(format!("{}_rank", feature_col))])
        .collect()?;

    Ok(result
        .column(&format!("{}_rank", feature_col))?
        .as_materialized_series()
        .clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosssectional_features() {
        // Create sample panel data (3 stocks × 2 dates)
        let df = df!(
            "date" => &[1, 1, 1, 2, 2, 2],
            "stock" => &[1, 2, 3, 1, 2, 3],
            "price" => &[100.0, 200.0, 150.0, 110.0, 190.0, 160.0],
        )
        .unwrap();

        let result = apply_crosssectional_features(&df, "date", &["stock", "date"]).unwrap();

        // Check new columns exist
        assert!(result.column("price_rank").is_ok());
        assert!(result.column("price_zscore").is_ok());
        assert!(result.column("price_vs_median").is_ok());

        // Check ranks for date=1: [100, 200, 150] -> ranks [0.0, 1.0, 0.5]
        let ranks = result
            .column("price_rank")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .take(3)
            .collect::<Vec<_>>();

        // 100 < 150 < 200, so ranks: 0/2, 2/2, 1/2 = 0.0, 1.0, 0.5
        assert!((ranks[0].unwrap() - 0.0).abs() < 0.01);
        assert!((ranks[1].unwrap() - 1.0).abs() < 0.01);
        assert!((ranks[2].unwrap() - 0.5).abs() < 0.01);
    }
}
