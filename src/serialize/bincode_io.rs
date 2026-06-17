//! Bincode-based model serialization
//!
//! Provides compact binary serialization via serde + bincode.

use crate::booster::GBDTModel;
use crate::{Result, TreeBoostError};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Save a model to a bincode file
pub fn save_model_bincode(model: &GBDTModel, path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let config = bincode::config::standard();

    bincode::serde::encode_into_std_write(model, &mut writer, config).map_err(|e| {
        TreeBoostError::Serialization(format!("Failed to serialize bincode: {}", e))
    })?;

    Ok(())
}

/// Load a model from a bincode file
pub fn load_model_bincode(path: impl AsRef<Path>) -> Result<GBDTModel> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let config = bincode::config::standard();

    let model: GBDTModel =
        bincode::serde::decode_from_std_read(&mut reader, config).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to deserialize bincode: {}", e))
        })?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::booster::GBDTConfig;
    use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};
    use tempfile::tempdir;

    fn create_test_dataset() -> BinnedDataset {
        let num_rows = 100;
        let num_features = 2;

        let features: Vec<u8> = (0..num_rows * num_features)
            .map(|i| (i % 256) as u8)
            .collect();
        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.1).collect();
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
                impute_value: 0.0,
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
                impute_value: 0.0,
            },
        ];

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_save_load_model_bincode() {
        let dataset = create_test_dataset();
        let config = GBDTConfig::new().with_num_rounds(5).with_max_depth(3);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();

        // Save to temp file
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.bincode");

        save_model_bincode(&model, &path).unwrap();

        // Load back
        let loaded = load_model_bincode(&path).unwrap();

        // Verify
        assert_eq!(loaded.num_trees(), model.num_trees());
        assert_eq!(loaded.base_prediction(), model.base_prediction());

        // Compare predictions
        let orig_preds = model.predict(&dataset);
        let loaded_preds = loaded.predict(&dataset);

        for (a, b) in orig_preds.iter().zip(loaded_preds.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
