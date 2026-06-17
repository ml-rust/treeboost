//! TunableModel implementation and tests for UniversalModel
//!
//! This module contains:
//! - TunableModel trait implementation
//! - All unit and integration tests

use std::collections::HashMap;

use crate::dataset::BinnedDataset;
use crate::loss::MseLoss;
use crate::tuner::{ParamValue, TunableModel};

use super::{BoostingMode, UniversalConfig, UniversalModel};

// =============================================================================
// TunableModel Implementation
// =============================================================================

impl TunableModel for UniversalModel {
    type Config = UniversalConfig;

    fn train(dataset: &BinnedDataset, config: &Self::Config) -> crate::Result<Self> {
        // Create a default MSE loss for tuning (loss type could be parameterized later)
        let loss_fn = MseLoss::new();
        UniversalModel::train(dataset, config.clone(), &loss_fn)
    }

    fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        UniversalModel::predict(self, dataset)
    }

    fn num_trees(&self) -> usize {
        // Delegate to UniversalModel::num_trees() which handles all modes correctly
        UniversalModel::num_trees(self)
    }

    fn apply_params(config: &mut Self::Config, params: &HashMap<String, ParamValue>) {
        for (name, value) in params {
            match (name.as_str(), value) {
                // Categorical: boosting mode
                ("mode", ParamValue::Categorical(v)) => {
                    config.mode = match v.as_str() {
                        "PureTree" => BoostingMode::PureTree,
                        "LinearThenTree" => BoostingMode::LinearThenTree,
                        "RandomForest" => BoostingMode::RandomForest,
                        _ => BoostingMode::PureTree, // Default fallback
                    };
                }
                // Numeric parameters
                ("num_rounds", ParamValue::Numeric(v)) => config.num_rounds = *v as usize,
                ("learning_rate", ParamValue::Numeric(v)) => config.learning_rate = *v,
                ("subsample", ParamValue::Numeric(v)) => config.subsample = *v,
                ("validation_ratio", ParamValue::Numeric(v)) => config.validation_ratio = *v,
                ("early_stopping_rounds", ParamValue::Numeric(v)) => {
                    config.early_stopping_rounds = *v as usize
                }
                ("linear_rounds", ParamValue::Numeric(v)) => config.linear_rounds = *v as usize,
                // Tree config parameters (prefixed with tree_)
                ("tree_max_depth", ParamValue::Numeric(v)) => {
                    config.tree_config = config
                        .tree_config
                        .clone()
                        .with_max_depth(*v as usize)
                        .expect("valid max_depth")
                }
                ("tree_max_leaves", ParamValue::Numeric(v)) => {
                    config.tree_config = config
                        .tree_config
                        .clone()
                        .with_max_leaves(*v as usize)
                        .expect("valid max_leaves")
                }
                ("tree_lambda", ParamValue::Numeric(v)) => {
                    config.tree_config = config
                        .tree_config
                        .clone()
                        .with_lambda(*v)
                        .expect("valid lambda")
                }
                // Linear config parameters (prefixed with linear_)
                ("linear_lambda", ParamValue::Numeric(v)) => {
                    config.linear_config = config
                        .linear_config
                        .clone()
                        .with_lambda(*v)
                        .expect("valid lambda")
                }
                ("linear_max_iter", ParamValue::Numeric(v)) => {
                    config.linear_config = config
                        .linear_config
                        .clone()
                        .with_max_iter(*v as usize)
                        .expect("valid max_iter")
                }
                _ => {} // Unknown params are ignored
            }
        }
    }

    fn valid_params() -> &'static [&'static str] {
        &[
            // Categorical
            "mode",
            // Numeric
            "num_rounds",
            "learning_rate",
            "subsample",
            "validation_ratio",
            "early_stopping_rounds",
            "linear_rounds",
            // Tree config
            "tree_max_depth",
            "tree_max_leaves",
            "tree_lambda",
            // Linear config
            "linear_lambda",
            "linear_max_iter",
        ]
    }

    fn default_config() -> Self::Config {
        UniversalConfig::default()
    }

    fn get_learning_rate(config: &Self::Config) -> f32 {
        config.learning_rate
    }

    fn configure_validation(
        config: &mut Self::Config,
        validation_ratio: f32,
        early_stopping_rounds: usize,
    ) {
        config.validation_ratio = validation_ratio;
        config.early_stopping_rounds = early_stopping_rounds;
    }

    fn set_num_rounds(config: &mut Self::Config, num_rounds: usize) {
        config.num_rounds = num_rounds;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};
    use crate::loss::MseLoss;
    use rkyv::rancor::Error as RkyvError;

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        // Linear relationship with some noise
        let targets: Vec<f32> = (0..num_rows)
            .map(|i| (i as f32) * 0.1 + (i % 10) as f32 * 0.01)
            .collect();

        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: (0..255).map(|b| b as f64).collect(),
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    // ========================================
    // Test Helper Functions
    // ========================================

    /// Test helper: Verify serde serialization roundtrip for BoostingMode
    fn assert_serde_roundtrip_mode(mode: &BoostingMode) {
        let json = serde_json::to_string(mode).expect("Failed to serialize");
        assert!(!json.is_empty(), "Serialized JSON should not be empty");

        let loaded: BoostingMode = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(loaded, *mode, "Deserialized value should match original");
    }

    /// Test helper: Verify model predictions match after serialization
    fn assert_model_predictions_match(
        original: &UniversalModel,
        loaded: &UniversalModel,
        dataset: &BinnedDataset,
        tolerance: f32,
    ) {
        let original_preds = original.predict(dataset);
        let loaded_preds = loaded.predict(dataset);

        assert_eq!(
            original_preds.len(),
            loaded_preds.len(),
            "Prediction count mismatch"
        );

        for (i, (o, l)) in original_preds.iter().zip(loaded_preds.iter()).enumerate() {
            assert!(
                (o - l).abs() < tolerance,
                "Prediction mismatch at index {}: {} vs {}",
                i,
                o,
                l
            );
        }
    }

    fn train_test_model(mode: BoostingMode, num_rounds: usize) -> (UniversalModel, BinnedDataset) {
        let dataset = create_test_dataset(100, 5);
        let config = UniversalConfig::default()
            .with_mode(mode)
            .with_num_rounds(num_rounds);

        let loss_fn = MseLoss::new();
        let model =
            UniversalModel::train(&dataset, config, &loss_fn).expect("Failed to train model");

        (model, dataset)
    }

    // ========================================
    // Configuration Tests
    // ========================================

    #[test]
    fn test_universal_config_defaults() {
        let config = UniversalConfig::default();
        assert_eq!(config.mode, BoostingMode::PureTree);
        assert_eq!(config.num_rounds, 100);
        assert!(config.learning_rate > 0.0 && config.learning_rate <= 1.0);
    }

    #[test]
    fn test_universal_config_builder() {
        let config = UniversalConfig::default()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(200);

        assert_eq!(config.mode, BoostingMode::LinearThenTree);
        assert_eq!(config.num_rounds, 200);
    }

    // ========================================
    // Training Tests
    // ========================================

    #[test]
    fn test_pure_tree_training() {
        let (model, _) = train_test_model(BoostingMode::PureTree, 5);

        assert_eq!(model.mode(), BoostingMode::PureTree);
        assert_eq!(model.num_trees(), 5);
        assert!(model.gbdt_model().is_some());
        assert!(!model.has_linear());
    }

    #[test]
    fn test_pure_tree_prediction() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 5);

        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), dataset.num_rows());
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_linear_then_tree_training() {
        let (model, _) = train_test_model(BoostingMode::LinearThenTree, 5);

        assert_eq!(model.mode(), BoostingMode::LinearThenTree);
        assert_eq!(model.num_trees(), 5);
        assert!(model.has_linear());
        assert!(model.linear_booster().is_some());
    }

    #[test]
    fn test_linear_then_tree_prediction() {
        let (model, dataset) = train_test_model(BoostingMode::LinearThenTree, 5);

        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), dataset.num_rows());
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_random_forest_training() {
        let (model, _) = train_test_model(BoostingMode::RandomForest, 5);

        assert_eq!(model.mode(), BoostingMode::RandomForest);
        assert_eq!(model.num_trees(), 5);
        assert!(!model.trees().is_empty());
    }

    #[test]
    fn test_random_forest_prediction() {
        let (model, dataset) = train_test_model(BoostingMode::RandomForest, 5);

        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), dataset.num_rows());
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_single_row_prediction_matches_batch() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 5);

        let batch_preds = model.predict(&dataset);

        for (row_idx, &batch_pred) in batch_preds.iter().take(10).enumerate() {
            let single_pred = model.predict_row(&dataset, row_idx);
            assert!(
                (single_pred - batch_pred).abs() < 1e-5,
                "Single-row and batch predictions should match"
            );
        }
    }

    // ========================================
    // Auto Selection Tests
    // ========================================

    #[test]
    fn test_auto_selects_mode_and_trains() {
        let dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();

        let model = UniversalModel::auto(&dataset, &loss_fn).expect("Auto training failed");

        assert!(model.was_auto_selected());
        assert!(model.analysis().is_some());
        assert!(model.num_trees() > 0);
    }

    #[test]
    fn test_auto_with_config() {
        let dataset = create_test_dataset(100, 5);
        let config = UniversalConfig::default().with_num_rounds(10);
        let loss_fn = MseLoss::new();

        let model =
            UniversalModel::auto_with_config(&dataset, config, &loss_fn).expect("Auto failed");

        assert!(model.was_auto_selected());
        assert_eq!(model.num_trees(), 10);
    }

    // ========================================
    // Analysis Report Tests
    // ========================================

    #[test]
    fn test_analysis_report_generation() {
        let dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();

        let model = UniversalModel::auto(&dataset, &loss_fn).expect("Auto training failed");

        let report = model.analysis_report();
        assert!(
            report.is_some(),
            "Auto-trained model should have analysis report"
        );
    }

    #[test]
    fn test_analysis_summary() {
        let dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();

        let model = UniversalModel::auto(&dataset, &loss_fn).expect("Auto training failed");

        let summary = model.analysis_summary();
        assert!(summary.is_some(), "Auto-trained model should have summary");
        assert!(!summary.unwrap().is_empty(), "Summary should not be empty");
    }

    // ========================================
    // Serialization Tests
    // ========================================

    #[test]
    fn test_universal_config_serde_serialization() {
        let config = UniversalConfig::default()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(50);

        let json = serde_json::to_string(&config).expect("Serialization failed");
        let loaded: UniversalConfig = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(config.mode, loaded.mode);
        assert_eq!(config.num_rounds, loaded.num_rounds);
    }

    #[test]
    fn test_boosting_mode_serde_serialization() {
        assert_serde_roundtrip_mode(&BoostingMode::PureTree);
        assert_serde_roundtrip_mode(&BoostingMode::LinearThenTree);
        assert_serde_roundtrip_mode(&BoostingMode::RandomForest);
    }

    #[test]
    fn test_puretree_model_serde_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 3);

        let json = serde_json::to_string(&model).expect("Serialization failed");
        let loaded: UniversalModel = serde_json::from_str(&json).expect("Deserialization failed");

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-5);
    }

    #[test]
    fn test_linear_then_tree_model_serde_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::LinearThenTree, 3);

        let json = serde_json::to_string(&model).expect("Serialization failed");
        let loaded: UniversalModel = serde_json::from_str(&json).expect("Deserialization failed");

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-5);
    }

    #[test]
    fn test_random_forest_model_serde_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::RandomForest, 3);

        let json = serde_json::to_string(&model).expect("Serialization failed");
        let loaded: UniversalModel = serde_json::from_str(&json).expect("Deserialization failed");

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-5);
    }

    #[test]
    fn test_universal_config_rkyv_serialization() {
        let config = UniversalConfig::default()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(50);

        let bytes = rkyv::to_bytes::<RkyvError>(&config).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalConfig = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode, BoostingMode::LinearThenTree);
        assert_eq!(loaded.num_rounds, 50);
    }

    #[test]
    fn test_puretree_model_rkyv_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 3);

        let bytes = rkyv::to_bytes::<RkyvError>(&model).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalModel = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::PureTree);
        assert_eq!(loaded.num_features(), 5);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_linear_then_tree_model_rkyv_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::LinearThenTree, 3);

        let bytes = rkyv::to_bytes::<RkyvError>(&model).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalModel = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::LinearThenTree);
        assert_eq!(loaded.num_features(), 5);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_random_forest_model_rkyv_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::RandomForest, 3);

        let bytes = rkyv::to_bytes::<RkyvError>(&model).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalModel = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::RandomForest);
        assert_eq!(loaded.num_features(), 5);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    // ========================================
    // Incremental Update Tests
    // ========================================

    #[test]
    fn test_puretree_incremental_update() {
        let (mut model, _) = train_test_model(BoostingMode::PureTree, 3);
        let initial_trees = model.num_trees();

        let new_dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();

        let report = model
            .update(&new_dataset, &loss_fn, 2)
            .expect("Update failed");

        assert_eq!(report.trees_added, 2);
        assert_eq!(model.num_trees(), initial_trees + 2);
    }

    #[test]
    fn test_linear_then_tree_incremental_update() {
        let (mut model, _) = train_test_model(BoostingMode::LinearThenTree, 3);
        let initial_trees = model.num_trees();

        let new_dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();

        let report = model
            .update(&new_dataset, &loss_fn, 2)
            .expect("Update failed");

        assert_eq!(report.trees_added, 2);
        assert_eq!(model.num_trees(), initial_trees + 2);
    }

    #[test]
    fn test_random_forest_incremental_update() {
        let (mut model, _) = train_test_model(BoostingMode::RandomForest, 3);
        let initial_trees = model.num_trees();

        let new_dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();

        let report = model
            .update(&new_dataset, &loss_fn, 2)
            .expect("Update failed");

        assert_eq!(report.trees_added, 2);
        assert_eq!(model.num_trees(), initial_trees + 2);
    }

    #[test]
    fn test_incremental_update_report_display() {
        use crate::model::universal::IncrementalUpdateReport;
        let report = IncrementalUpdateReport {
            rows_trained: 100,
            trees_before: 5,
            trees_after: 7,
            trees_added: 2,
            mode: BoostingMode::PureTree,
        };

        let display_str = format!("{}", report);
        assert!(display_str.contains("100 rows"));
        assert!(display_str.contains("2 trees"));
    }

    #[test]
    fn test_incremental_update_feature_mismatch() {
        let (mut model, _) = train_test_model(BoostingMode::PureTree, 3);

        // Create dataset with different number of features
        let mismatched_dataset = create_test_dataset(100, 10);
        let loss_fn = MseLoss::new();

        let result = model.update(&mismatched_dataset, &loss_fn, 2);
        assert!(
            result.is_err(),
            "Update with mismatched features should fail"
        );
    }

    #[test]
    fn test_is_compatible_for_update() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 3);

        // Compatible dataset (same features)
        assert!(model.is_compatible_for_update(&dataset));

        // Incompatible dataset (different features)
        let mismatched = create_test_dataset(50, 10);
        assert!(!model.is_compatible_for_update(&mismatched));
    }

    // ========================================
    // File Serialization Tests
    // ========================================

    #[test]
    fn test_trb_save_and_load() {
        use tempfile::TempDir;

        let (model, dataset) = train_test_model(BoostingMode::PureTree, 3);
        let original_preds = model.predict(&dataset);

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test_model.trb");

        model
            .save_trb(&path, "Test model")
            .expect("Failed to save TRB");
        assert!(path.exists(), "TRB file should exist");

        let loaded = UniversalModel::load_trb(&path).expect("Failed to load TRB");
        let loaded_preds = loaded.predict(&dataset);

        assert_eq!(original_preds.len(), loaded_preds.len());
    }

    #[test]
    fn test_trb_save_update_and_load() {
        use tempfile::TempDir;

        let (mut model, _dataset) = train_test_model(BoostingMode::PureTree, 3);

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test_model_update.trb");

        model
            .save_trb(&path, "Initial")
            .expect("Failed to save TRB");

        let new_dataset = create_test_dataset(100, 5);
        let loss_fn = MseLoss::new();
        model
            .update(&new_dataset, &loss_fn, 2)
            .expect("Update failed");

        model
            .save_trb_update(&path, 100, "Update 1")
            .expect("Failed to save TRB update");

        let loaded = UniversalModel::load_trb(&path).expect("Failed to load TRB");
        assert_eq!(loaded.num_trees(), 5, "Should have 5 trees after load");
    }
}
