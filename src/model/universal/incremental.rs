//! Incremental learning and model persistence for UniversalModel
//!
//! This module contains:
//! - update() and mode-specific update methods
//! - TRB format save/load (incremental binary format)

use crate::booster::GBDTModel;
use crate::dataset::BinnedDataset;
use crate::learner::WeakLearner;
use crate::loss::LossFunction;
use crate::Result;
use rand::SeedableRng;
use rayon::prelude::*;

use super::{BoostingMode, UniversalModel};

impl UniversalModel {
    // =========================================================================
    // Incremental Learning Support
    // =========================================================================

    /// Update the model with new training data (incremental learning)
    ///
    /// This method continues training from the current model state:
    /// - **PureTree**: Appends new trees trained on residuals
    /// - **LinearThenTree**: Updates linear weights (warm_fit) + appends new trees
    /// - **RandomForest**: Appends new bootstrap trees
    ///
    /// # Arguments
    /// * `dataset` - New data to train on (must have same feature schema)
    /// * `loss_fn` - Loss function (should match original training)
    /// * `additional_rounds` - Number of new boosting rounds (trees) to add
    ///
    /// # Example
    /// ```ignore
    /// // Train on January data
    /// let model = UniversalModel::train(&jan_data, config, &MseLoss)?;
    ///
    /// // Update with February data (10 more trees)
    /// model.update(&feb_data, &MseLoss, 10)?;
    /// ```
    pub fn update(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<super::core::IncrementalUpdateReport> {
        // Validate feature compatibility
        if dataset.num_features() != self.num_features {
            return Err(crate::TreeBoostError::Config(format!(
                "Feature count mismatch: model has {} features, dataset has {}",
                self.num_features,
                dataset.num_features()
            )));
        }

        let num_rows = dataset.num_rows();
        let trees_before = self.num_trees();

        match self.config.mode {
            BoostingMode::PureTree => {
                self.update_pure_tree(dataset, loss_fn, additional_rounds)?;
            }
            BoostingMode::LinearThenTree => {
                self.update_linear_then_tree(dataset, loss_fn, additional_rounds)?;
            }
            BoostingMode::RandomForest => {
                self.update_random_forest(dataset, loss_fn, additional_rounds)?;
            }
        }

        let trees_after = self.num_trees();

        Ok(super::core::IncrementalUpdateReport {
            rows_trained: num_rows,
            trees_before,
            trees_after,
            trees_added: trees_after - trees_before,
            mode: self.config.mode,
        })
    }

    /// Update PureTree model by appending trees trained on residuals
    fn update_pure_tree(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<()> {
        let gbdt = self.gbdt_model.as_mut().ok_or_else(|| {
            crate::TreeBoostError::Config("No GBDTModel available for update".to_string())
        })?;

        // Compute current predictions
        let predictions = gbdt.predict(dataset);

        // Create residual targets (avoids cloning targets only to overwrite)
        let residual_targets: Vec<f32> = dataset
            .targets()
            .iter()
            .zip(predictions.iter())
            .map(|(target, pred)| target - pred)
            .collect();

        // Create residual dataset with new targets
        let residual_dataset = dataset.with_targets(residual_targets);

        // Train new trees on residuals using same config
        let mut update_config = self.config.clone();
        update_config.num_rounds = additional_rounds;
        let gbdt_config = Self::to_gbdt_config(&update_config, loss_fn)?;
        let new_model = GBDTModel::train_binned(&residual_dataset, gbdt_config)?;

        // Append new trees
        gbdt.append_ensemble_trees(new_model.trees().to_vec());

        Ok(())
    }

    /// Update LinearThenTree model
    fn update_linear_then_tree(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<()> {
        use crate::learner::incremental::IncrementalLearner;

        let num_rows = dataset.num_rows();
        let targets = dataset.targets();

        // Compute current predictions BEFORE borrowing linear_booster mutably
        let current_preds = self.predict(dataset);

        // Extract linear features once (avoids duplication)
        let (linear_features, num_linear_features) = self.get_linear_features(dataset);

        // Update linear booster with warm_fit if present
        if let Some(ref mut linear_booster) = self.linear_booster {
            // Compute gradients
            let (gradients, hessians): (Vec<f32>, Vec<f32>) = targets
                .iter()
                .zip(current_preds.iter())
                .map(|(t, p)| loss_fn.gradient_hessian(*t, *p))
                .unzip();

            // Warm fit linear booster
            linear_booster.warm_fit(
                &linear_features,
                num_linear_features,
                &gradients,
                &hessians,
            )?;
        }

        // Now update trees on residuals from full ensemble
        if let Some(ref mut gbdt) = self.gbdt_model {
            // Get linear predictions (using already-extracted features)
            let linear_preds = if let Some(ref linear_booster) = self.linear_booster {
                linear_booster.predict_batch(&linear_features, num_linear_features)
            } else {
                vec![0.0; num_rows]
            };

            // Get existing tree predictions
            let tree_preds = gbdt.predict(dataset);

            // Compute residual targets: target - base - shrinkage*linear - tree
            let shrinkage = self.config.linear_config.shrinkage_factor;
            let residual_targets: Vec<f32> = targets
                .iter()
                .zip(linear_preds.iter())
                .zip(tree_preds.iter())
                .map(|((t, lp), tp)| t - self.base_prediction - shrinkage * lp - tp)
                .collect();

            // Create residual dataset with new targets (avoids full clone)
            let residual_dataset = dataset.with_targets(residual_targets);

            // Train new trees on residuals
            let mut update_config = self.config.clone();
            update_config.num_rounds = additional_rounds;
            let gbdt_config = Self::to_gbdt_config(&update_config, loss_fn)?;
            let new_model = GBDTModel::train_binned(&residual_dataset, gbdt_config)?;

            gbdt.append_ensemble_trees(new_model.trees().to_vec());
        }

        Ok(())
    }

    /// Update RandomForest by adding more bootstrap trees
    fn update_random_forest(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<()> {
        use crate::learner::TreeBooster;

        let num_rows = dataset.num_rows();
        let targets = dataset.targets();

        // RF trees are trained on bootstrap samples from base prediction
        let tree_config = self.config.tree_config.clone().with_learning_rate(1.0)?;

        // Use next seed offset based on current tree count
        let seed_offset_start = self.trees.len() as u64;

        let new_trees: Vec<crate::tree::Tree> = (0..additional_rounds)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    self.config.seed + seed_offset_start + i as u64,
                );

                // Bootstrap sample
                let bootstrap_indices: Vec<usize> = (0..num_rows)
                    .map(|_| {
                        use rand::RngExt;
                        rng.random_range(0..num_rows)
                    })
                    .collect();

                // Compute gradients for bootstrap sample
                let mut gradients = vec![0.0f32; num_rows];
                let mut hessians = vec![0.0f32; num_rows];

                for &idx in &bootstrap_indices {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], self.base_prediction);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }

                // Grow tree
                let mut booster = TreeBooster::new(tree_config.clone());
                if booster
                    .fit_on_gradients(dataset, &gradients, &hessians, Some(&bootstrap_indices))
                    .is_ok()
                {
                    booster.take_tree()
                } else {
                    None
                }
            })
            .collect();

        self.trees.extend(new_trees);

        Ok(())
    }

    // =========================================================================
    // TRB Format Save/Load
    // =========================================================================

    /// Save model to TRB (TreeBoost) incremental format
    ///
    /// TRB format supports incremental updates without rewriting the entire file.
    /// Use this format when you plan to update the model with new data.
    ///
    /// # Example
    /// ```ignore
    /// model.save_trb("model.trb", "Initial training on January data")?;
    ///
    /// // Later, after updating:
    /// model.save_trb_update("model.trb", 1000, "February update")?;
    /// ```
    pub fn save_trb(&self, path: impl AsRef<std::path::Path>, description: &str) -> Result<()> {
        use crate::serialize::{TrbHeader, TrbWriter, FORMAT_VERSION};
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let header = TrbHeader {
            format_version: FORMAT_VERSION,
            model_type: "universal".to_string(),
            created_at: timestamp,
            boosting_mode: format!("{:?}", self.config.mode),
            num_features: self.num_features,
            base_blob_size: 0, // Will be filled by TrbWriter
            metadata: description.to_string(),
        };

        // Serialize the model to rkyv bytes
        let blob = crate::serialize::rkyv_io::serialize_universal_model(self)?;

        TrbWriter::new(path, header, &blob)?;

        Ok(())
    }

    /// Append an update to an existing TRB file
    ///
    /// This appends a new segment without rewriting the base model.
    /// The model must be loaded with `load_trb()` before calling this.
    ///
    /// # Arguments
    /// * `path` - Path to existing .trb file
    /// * `rows_trained` - Number of rows used in this update
    /// * `description` - Description of this update
    pub fn save_trb_update(
        &self,
        path: impl AsRef<std::path::Path>,
        rows_trained: usize,
        description: &str,
    ) -> Result<()> {
        use crate::serialize::{open_for_append, TrbUpdateHeader, UpdateType};
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Use Snapshot type for all modes - stores full model state
        // This is simpler and more robust than incremental tree-only updates
        let update_type = UpdateType::Snapshot;

        let header = TrbUpdateHeader {
            update_type,
            created_at: timestamp,
            rows_trained,
            description: description.to_string(),
        };

        // Serialize the updated model
        let blob = crate::serialize::rkyv_io::serialize_universal_model(self)?;

        // Open and append
        let mut writer = open_for_append(path)?;
        writer.append_update(&header, &blob)?;

        Ok(())
    }

    /// Load model from TRB format
    ///
    /// Loads the base model and applies all update segments.
    ///
    /// # Example
    /// ```ignore
    /// let model = UniversalModel::load_trb("model.trb")?;
    /// ```
    pub fn load_trb(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use crate::serialize::{TrbReader, TrbSegment, UpdateType};

        let mut reader = TrbReader::open(path)?;

        // Load all segments
        let segments = reader.load_all_segments()?;

        // The last snapshot-type segment (or base if no snapshots) is the current state
        let mut model: Option<Self> = None;

        for segment in segments {
            match segment {
                TrbSegment::Base { blob, .. } => {
                    model = Some(crate::serialize::rkyv_io::deserialize_universal_model(
                        &blob,
                    )?);
                }
                TrbSegment::Update { header, blob } => {
                    match header.update_type {
                        UpdateType::Snapshot => {
                            // Replace with snapshot
                            model = Some(crate::serialize::rkyv_io::deserialize_universal_model(
                                &blob,
                            )?);
                        }
                        UpdateType::Trees => {
                            // For trees-only updates, we'd need to deserialize and merge
                            // For now, we just use snapshots for simplicity
                            if model.is_none() {
                                model = Some(
                                    crate::serialize::rkyv_io::deserialize_universal_model(&blob)?,
                                );
                            }
                        }
                        UpdateType::Linear | UpdateType::Preprocessor => {
                            // These would need specialized handling
                            // For now, fall through
                        }
                    }
                }
            }
        }

        model.ok_or_else(|| {
            crate::TreeBoostError::Serialization("No valid model found in TRB file".to_string())
        })
    }
}
