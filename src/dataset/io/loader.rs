//! Data loading utilities for Polars DataFrames
//!
//! Provides integration with Polars for loading and preprocessing tabular data.

use crate::dataset::binning::{DatasetBinner, QuantileBinner};
use crate::dataset::core::{BinnedDataset, FeatureInfo, FeatureType};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::path::Path;

/// Dataset loader for converting Polars DataFrames to BinnedDataset
pub struct DatasetLoader {
    binner: DatasetBinner,
}

impl DatasetLoader {
    /// Create a new dataset loader with specified number of bins
    pub fn new(num_bins: usize) -> Self {
        Self {
            binner: DatasetBinner::new(num_bins),
        }
    }

    /// Load a Parquet file into a BinnedDataset
    pub fn load_parquet(
        &self,
        path: impl AsRef<Path>,
        target_column: &str,
        feature_columns: Option<&[&str]>,
    ) -> Result<BinnedDataset> {
        let pl_path = PlRefPath::new(&*path.as_ref().to_string_lossy());
        let df = LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?;
        self.from_dataframe(df, target_column, feature_columns)
    }

    /// Load a CSV file into a BinnedDataset
    pub fn load_csv(
        &self,
        path: impl AsRef<Path>,
        target_column: &str,
        feature_columns: Option<&[&str]>,
    ) -> Result<BinnedDataset> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        self.from_dataframe(df, target_column, feature_columns)
    }

    /// Convert a Polars DataFrame to BinnedDataset
    pub fn from_dataframe(
        &self,
        df: DataFrame,
        target_column: &str,
        feature_columns: Option<&[&str]>,
    ) -> Result<BinnedDataset> {
        let num_rows = df.height();

        // Extract target column
        let target_col = df.column(target_column).map_err(|e| {
            TreeBoostError::Data(format!(
                "Target column '{}' not found: {}",
                target_column, e
            ))
        })?;
        let target_series = target_col.as_materialized_series();
        let targets = self.series_to_f32(target_series)?;

        // Determine feature columns
        let feature_names: Vec<String> = match feature_columns {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => df
                .get_column_names()
                .into_iter()
                .filter(|name| *name != target_column)
                .map(|s| s.to_string())
                .collect(),
        };

        // Process each feature column
        let mut all_binned: Vec<Vec<u8>> = Vec::with_capacity(feature_names.len());
        let mut all_info: Vec<FeatureInfo> = Vec::with_capacity(feature_names.len());

        for name in &feature_names {
            let col = df.column(name.as_str()).map_err(|e| {
                TreeBoostError::Data(format!("Feature column '{}' not found: {}", name, e))
            })?;
            let series = col.as_materialized_series();

            let (binned, info) = self.process_series(name.clone(), series)?;
            all_binned.push(binned);
            all_info.push(info);
        }

        // Flatten to column-major layout
        let mut features = Vec::with_capacity(num_rows * all_binned.len());
        for binned_col in all_binned {
            features.extend(binned_col);
        }

        Ok(BinnedDataset::new(num_rows, features, targets, all_info))
    }

    /// Convert a Polars DataFrame to BinnedDataset with era indices (for panel data)
    ///
    /// # Arguments
    /// * `df` - The DataFrame to convert
    /// * `target_column` - Name of the target column
    /// * `feature_columns` - Optional list of feature column names (None = all except target and era)
    /// * `era_column` - Name of the column containing era/time information
    ///
    /// Era values will be mapped to sequential u16 indices (0, 1, 2, ...)
    pub fn from_dataframe_with_eras(
        &self,
        df: DataFrame,
        target_column: &str,
        feature_columns: Option<&[&str]>,
        era_column: &str,
    ) -> Result<BinnedDataset> {
        let num_rows = df.height();

        // Extract target column
        let target_col = df.column(target_column).map_err(|e| {
            TreeBoostError::Data(format!(
                "Target column '{}' not found: {}",
                target_column, e
            ))
        })?;
        let target_series = target_col.as_materialized_series();
        let targets = self.series_to_f32(target_series)?;

        // Extract era column and map to sequential indices
        let era_col = df.column(era_column).map_err(|e| {
            TreeBoostError::Data(format!("Era column '{}' not found: {}", era_column, e))
        })?;
        let era_series = era_col.as_materialized_series();
        let era_indices = self.extract_era_indices(era_series)?;

        // Determine feature columns (exclude both target and era)
        let feature_names: Vec<String> = match feature_columns {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => df
                .get_column_names()
                .into_iter()
                .filter(|name| *name != target_column && *name != era_column)
                .map(|s| s.to_string())
                .collect(),
        };

        // Process each feature column
        let mut all_binned: Vec<Vec<u8>> = Vec::with_capacity(feature_names.len());
        let mut all_info: Vec<FeatureInfo> = Vec::with_capacity(feature_names.len());

        for name in &feature_names {
            let col = df.column(name.as_str()).map_err(|e| {
                TreeBoostError::Data(format!("Feature column '{}' not found: {}", name, e))
            })?;
            let series = col.as_materialized_series();

            let (binned, info) = self.process_series(name.clone(), series)?;
            all_binned.push(binned);
            all_info.push(info);
        }

        // Flatten to column-major layout
        let mut features = Vec::with_capacity(num_rows * all_binned.len());
        for binned_col in all_binned {
            features.extend(binned_col);
        }

        Ok(BinnedDataset::new_with_eras(
            num_rows,
            features,
            targets,
            all_info,
            era_indices,
        ))
    }

    /// Extract era indices from a series and map to sequential u16 values
    ///
    /// Handles both numeric and categorical era columns.
    /// Returns a vector where `era_indices[i]` is the era index for row `i`.
    fn extract_era_indices(&self, series: &Series) -> Result<Vec<u16>> {
        use std::collections::HashMap;

        // Convert to string for generic handling
        let str_series = series.cast(&DataType::String).map_err(|e| {
            TreeBoostError::Data(format!("Failed to cast era column to string: {}", e))
        })?;

        let str_ca = str_series.str().map_err(|e| {
            TreeBoostError::Data(format!(
                "Failed to get string chunked from era column: {}",
                e
            ))
        })?;

        // Build mapping from unique era values to sequential indices
        let mut era_map: HashMap<String, u16> = HashMap::new();
        let mut next_idx = 0u16;

        let indices: Vec<u16> = str_ca
            .iter()
            .map(|opt| {
                opt.map_or(0, |s| {
                    *era_map.entry(s.to_string()).or_insert_with(|| {
                        let idx = next_idx;
                        next_idx = next_idx.saturating_add(1);
                        if next_idx == 0 {
                            // Overflow - u16 max is 65535
                            panic!("Era count exceeds u16::MAX (65535). Use fewer eras.");
                        }
                        idx
                    })
                })
            })
            .collect();

        Ok(indices)
    }

    /// Process a single series into binned values
    fn process_series(&self, name: String, series: &Series) -> Result<(Vec<u8>, FeatureInfo)> {
        match series.dtype() {
            DataType::Float64
            | DataType::Float32
            | DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8 => {
                let values = self.series_to_f64(series)?;
                self.binner.process_numeric_column(name, &values)
            }
            DataType::String | DataType::Categorical(_, _) => {
                // For now, treat categoricals as numeric after encoding
                // Full Ordered Target Encoding will be in encoding module
                let values = self.categorical_to_numeric(series)?;
                self.binner.process_numeric_column(name, &values)
            }
            dtype => Err(TreeBoostError::Data(format!(
                "Unsupported dtype for column '{}': {:?}",
                name, dtype
            ))),
        }
    }

    /// Convert numeric series to f64 vec
    fn series_to_f64(&self, series: &Series) -> Result<Vec<f64>> {
        series
            .cast(&DataType::Float64)
            .map_err(|e| TreeBoostError::Data(format!("Failed to cast to f64: {}", e)))?
            .f64()
            .map_err(|e| TreeBoostError::Data(format!("Failed to get f64 chunked: {}", e)))?
            .iter()
            .map(|opt| Ok(opt.unwrap_or(f64::NAN)))
            .collect()
    }

    /// Convert numeric series to f32 vec
    fn series_to_f32(&self, series: &Series) -> Result<Vec<f32>> {
        series
            .cast(&DataType::Float32)
            .map_err(|e| TreeBoostError::Data(format!("Failed to cast to f32: {}", e)))?
            .f32()
            .map_err(|e| TreeBoostError::Data(format!("Failed to get f32 chunked: {}", e)))?
            .iter()
            .map(|opt| Ok(opt.unwrap_or(f32::NAN)))
            .collect()
    }

    /// Convert categorical to numeric (simple ordinal encoding)
    /// Full Ordered Target Encoding is in encoding module
    fn categorical_to_numeric(&self, series: &Series) -> Result<Vec<f64>> {
        // Simple ordinal encoding using a hash map
        use std::collections::HashMap;

        let str_series = series
            .cast(&DataType::String)
            .map_err(|e| TreeBoostError::Data(format!("Failed to cast to string: {}", e)))?;

        let str_ca = str_series
            .str()
            .map_err(|e| TreeBoostError::Data(format!("Failed to get string chunked: {}", e)))?;

        // Build mapping from unique values to indices
        let mut mapping: HashMap<String, u32> = HashMap::new();
        let mut next_idx = 0u32;

        let values: Vec<f64> = str_ca
            .iter()
            .map(|opt| match opt {
                Some(s) => {
                    let idx = *mapping.entry(s.to_string()).or_insert_with(|| {
                        let idx = next_idx;
                        next_idx += 1;
                        idx
                    });
                    idx as f64
                }
                None => f64::NAN,
            })
            .collect();

        Ok(values)
    }

    /// Load a Parquet file for prediction using existing bin boundaries
    pub fn load_parquet_for_prediction(
        &self,
        path: impl AsRef<Path>,
        feature_info: &[FeatureInfo],
    ) -> Result<BinnedDataset> {
        let pl_path = PlRefPath::new(&*path.as_ref().to_string_lossy());
        let df = LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?;
        self.from_dataframe_for_prediction(df, feature_info)
    }

    /// Load a CSV file for prediction using existing bin boundaries
    pub fn load_csv_for_prediction(
        &self,
        path: impl AsRef<Path>,
        feature_info: &[FeatureInfo],
    ) -> Result<BinnedDataset> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        self.from_dataframe_for_prediction(df, feature_info)
    }

    /// Convert a Polars DataFrame for prediction using existing bin boundaries
    pub fn from_dataframe_for_prediction(
        &self,
        df: DataFrame,
        feature_info: &[FeatureInfo],
    ) -> Result<BinnedDataset> {
        let num_rows = df.height();

        // Process each feature using training bin boundaries
        let mut all_binned: Vec<Vec<u8>> = Vec::with_capacity(feature_info.len());
        let mut all_info: Vec<FeatureInfo> = Vec::with_capacity(feature_info.len());

        for info in feature_info {
            let col = df.column(&info.name).map_err(|e| {
                TreeBoostError::Data(format!("Feature column '{}' not found: {}", info.name, e))
            })?;
            let series = col.as_materialized_series();

            let binned = self.bin_with_boundaries(series, info)?;
            all_binned.push(binned);
            all_info.push(info.clone());
        }

        // Flatten to column-major layout
        let mut features = Vec::with_capacity(num_rows * all_binned.len());
        for binned_col in all_binned {
            features.extend(binned_col);
        }

        // Create dummy targets (not used for prediction)
        let targets = vec![0.0f32; num_rows];

        Ok(BinnedDataset::new(num_rows, features, targets, all_info))
    }

    /// Bin a series using pre-computed boundaries
    fn bin_with_boundaries(&self, series: &Series, info: &FeatureInfo) -> Result<Vec<u8>> {
        match info.feature_type {
            FeatureType::Numeric => {
                let values = self.series_to_f64(series)?;
                Ok(values
                    .iter()
                    .map(|&v| QuantileBinner::bin_value(v, &info.bin_boundaries))
                    .collect())
            }
            FeatureType::Categorical => {
                // For categoricals, use same ordinal encoding
                let values = self.categorical_to_numeric(series)?;
                Ok(values
                    .iter()
                    .map(|&v| QuantileBinner::bin_value(v, &info.bin_boundaries))
                    .collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_from_dataframe() {
        let df = df! {
            "feature1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2" => &[10.0, 20.0, 30.0, 40.0, 50.0],
            "target" => &[100.0, 200.0, 300.0, 400.0, 500.0]
        }
        .unwrap();

        let loader = DatasetLoader::new(4);
        let dataset = loader.from_dataframe(df, "target", None).unwrap();

        assert_eq!(dataset.num_rows(), 5);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.targets().len(), 5);
    }
}
