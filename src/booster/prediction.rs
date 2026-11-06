//! GBDT prediction implementations
//!
//! Contains all prediction-related methods for GBDTModel:
//! - Regression predictions (raw, with intervals)
//! - Binary classification (probabilities, classes)
//! - Multi-class classification (probabilities, classes, raw scores)
//! - Batch and row-wise variants

use super::GBDTModel;
use crate::dataset::BinnedDataset;
use crate::loss::{sigmoid, softmax};
use rayon::prelude::*;

impl GBDTModel {
    /// Predict for a single row
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        let mut pred = self.base_predictions[0];
        for tree in &self.trees {
            pred += tree.as_scalar().predict_row(dataset, row_idx);
        }
        pred
    }

    /// Predict for all rows using tree-wise batch prediction
    ///
    /// This approach traverses one tree for ALL rows before moving to the next tree,
    /// which is more cache-friendly than row-wise traversal.
    ///
    /// Routes to parallel or sequential based on config.parallel_prediction
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        if self.config.parallel_prediction {
            self.predict_parallel(dataset)
        } else {
            self.predict_sequential(dataset)
        }
    }

    /// Single-threaded tree-wise batch prediction
    ///
    /// Traverses each tree for all rows before moving to the next tree.
    /// More cache-friendly than row-wise traversal.
    pub fn predict_sequential(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();

        // Initialize predictions with base value
        let mut predictions = vec![self.base_predictions[0]; num_rows];

        // Tree-wise: traverse each tree for all rows
        for tree in &self.trees {
            tree.as_scalar()
                .predict_batch_add(dataset, &mut predictions);
        }

        predictions
    }

    /// Parallel tree-wise batch prediction
    ///
    /// Splits rows into chunks and processes each chunk in parallel.
    /// Each chunk uses tree-wise traversal internally.
    pub fn predict_parallel(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();

        // For small datasets, use sequential
        if num_rows < 1000 || self.trees.is_empty() {
            return self.predict_sequential(dataset);
        }

        // Initialize predictions with base value
        let mut predictions = vec![self.base_predictions[0]; num_rows];

        // Determine chunk size for parallelism (target ~4 chunks per thread)
        let num_threads = rayon::current_num_threads();
        let chunk_size = (num_rows / (num_threads * 4)).max(256);

        // Process chunks in parallel, each chunk does tree-wise traversal
        predictions
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_row = chunk_idx * chunk_size;

                // For each tree, process this chunk of rows
                for tree in &self.trees {
                    let scalar_tree = tree.as_scalar();
                    for (i, pred) in chunk.iter_mut().enumerate() {
                        let row_idx = start_row + i;
                        *pred += scalar_tree.predict(|f| dataset.get_bin(row_idx, f));
                    }
                }
            });

        predictions
    }

    /// Legacy row-wise prediction (kept for comparison/testing)
    #[doc(hidden)]
    pub fn predict_row_wise(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        let mut predictions = Vec::with_capacity(num_rows);
        let mut row_bins = vec![0u8; num_features];

        for row_idx in 0..num_rows {
            // Cache all bins for this row
            for (f, bin) in row_bins.iter_mut().enumerate() {
                *bin = dataset.get_bin(row_idx, f);
            }

            // Traverse all trees with cached bins
            let mut pred = self.base_predictions[0];
            for tree in &self.trees {
                pred += tree.as_scalar().predict(|f| row_bins[f]);
            }
            predictions.push(pred);
        }

        predictions
    }

    /// Predict with conformal intervals
    ///
    /// Returns (predictions, lower_bounds, upper_bounds)
    pub fn predict_with_intervals(
        &self,
        dataset: &BinnedDataset,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let predictions = self.predict(dataset);

        let q = self.conformal_q.unwrap_or(0.0);
        let lower: Vec<f32> = predictions.iter().map(|&p| p - q).collect();
        let upper: Vec<f32> = predictions.iter().map(|&p| p + q).collect();

        (predictions, lower, upper)
    }

    // ============================================================================
    // Classification prediction methods
    // ============================================================================

    /// Predict class probabilities for binary classification
    ///
    /// Applies sigmoid to raw predictions to get probabilities in [0, 1].
    /// Only meaningful when trained with `with_binary_logloss()`.
    ///
    /// # Returns
    /// Vector of probabilities (probability of class 1)
    pub fn predict_proba(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let raw = self.predict(dataset);
        raw.iter().map(|&r| sigmoid(r)).collect()
    }

    /// Predict class labels for binary classification
    ///
    /// Applies sigmoid to raw predictions and thresholds at 0.5 (or custom threshold).
    /// Only meaningful when trained with `with_binary_logloss()`.
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset to predict on
    /// * `threshold` - Classification threshold (default 0.5)
    ///
    /// # Returns
    /// Vector of class labels (0 or 1)
    pub fn predict_class(&self, dataset: &BinnedDataset, threshold: f32) -> Vec<u32> {
        let proba = self.predict_proba(dataset);
        proba
            .iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect()
    }

    // ============================================================================
    // Multi-class classification prediction methods
    // ============================================================================

    /// Check if this is a multi-class model
    pub fn is_multiclass(&self) -> bool {
        matches!(self.output_type(), crate::booster::OutputType::MultiClass)
    }

    /// Check if this is a multi-label model
    pub fn is_multilabel(&self) -> bool {
        matches!(self.output_type(), crate::booster::OutputType::MultiLabel)
    }

    /// Get number of classes (0 for regression/binary/multi-label)
    pub fn get_num_classes(&self) -> usize {
        if matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            self.num_outputs()
        } else {
            0
        }
    }

    // ============================================================================
    // Multi-label classification prediction methods
    // ============================================================================

    /// Predict raw scores for multi-label classification
    ///
    /// Supports two storage modes:
    /// - **Vector Trees (new)**: One VectorTree per round, each leaf has `Vec<f32>` values
    /// - **Scalar Trees (legacy)**: N scalar trees per round, stored round-major
    ///
    /// # Returns
    /// Vector of vectors: `result[sample][label]`
    pub fn predict_multilabel(&self, dataset: &BinnedDataset) -> Vec<Vec<f32>> {
        if !self.is_multilabel() {
            // Fall back to single-output prediction wrapped in Vec
            return self.predict(dataset).into_iter().map(|p| vec![p]).collect();
        }

        // Check if we're using unified vector trees or legacy scalar trees
        if self.uses_vector_trees() {
            self.predict_multilabel_vector(dataset)
        } else {
            self.predict_multilabel_scalar(dataset)
        }
    }

    /// Predict using unified vector trees (new approach)
    ///
    /// Each tree contributes to all labels simultaneously.
    fn predict_multilabel_vector(&self, dataset: &BinnedDataset) -> Vec<Vec<f32>> {
        let num_rows = dataset.num_rows();
        let num_outputs = self.num_outputs();

        // Initialize with base predictions (flat layout: row-major)
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_outputs);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add vector tree contributions
        for tree in &self.trees {
            let vector_tree = tree.as_vector();
            vector_tree.predict_batch_add(
                |sample_idx, feature_idx| dataset.get_bin(sample_idx, feature_idx),
                num_rows,
                &mut raw_preds,
            );
        }

        // Convert to Vec<Vec<f32>>
        raw_preds
            .chunks_exact(num_outputs)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Predict using legacy scalar trees (one tree per label per round)
    fn predict_multilabel_scalar(&self, dataset: &BinnedDataset) -> Vec<Vec<f32>> {
        let num_rows = dataset.num_rows();
        let num_outputs = self.num_outputs();
        let num_rounds = self.trees.len() / num_outputs;

        // Initialize with base predictions
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_outputs);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add tree contributions (trees stored round-major)
        for round in 0..num_rounds {
            for label_idx in 0..num_outputs {
                let tree_idx = round * num_outputs + label_idx;
                let tree = self.trees[tree_idx].as_scalar();

                for row_idx in 0..num_rows {
                    let delta = tree.predict(|f| dataset.get_bin(row_idx, f));
                    raw_preds[row_idx * num_outputs + label_idx] += delta;
                }
            }
        }

        // Convert to Vec<Vec<f32>>
        raw_preds
            .chunks_exact(num_outputs)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Predict probabilities for multi-label classification
    ///
    /// Applies sigmoid to raw predictions to get probabilities for each label.
    /// Only meaningful when trained with `with_multilabel_logloss()` or `with_multilabel_focal_loss()`.
    ///
    /// # Returns
    /// Vector of probability vectors: `result[sample][label]` where each value is in [0, 1]
    pub fn predict_proba_multilabel(&self, dataset: &BinnedDataset) -> Vec<Vec<f32>> {
        let raw = self.predict_multilabel(dataset);
        raw.into_iter()
            .map(|row| row.into_iter().map(sigmoid).collect())
            .collect()
    }

    /// Predict binary labels for multi-label classification using default threshold
    ///
    /// Uses threshold 0.5 for all labels.
    ///
    /// # Returns
    /// Vector of boolean vectors: `result[sample][label]`
    pub fn predict_labels(&self, dataset: &BinnedDataset) -> Vec<Vec<bool>> {
        self.predict_labels_with_threshold(dataset, 0.5)
    }

    /// Predict binary labels for multi-label classification with custom threshold
    ///
    /// # Arguments
    /// * `dataset` - Input data
    /// * `threshold` - Probability threshold for positive class (same for all labels)
    ///
    /// # Returns
    /// Vector of boolean vectors: `result[sample][label]`
    pub fn predict_labels_with_threshold(
        &self,
        dataset: &BinnedDataset,
        threshold: f32,
    ) -> Vec<Vec<bool>> {
        let proba = self.predict_proba_multilabel(dataset);
        proba
            .into_iter()
            .map(|row| row.into_iter().map(|p| p >= threshold).collect())
            .collect()
    }

    /// Predict binary labels using per-label thresholds
    ///
    /// # Arguments
    /// * `dataset` - Input data
    /// * `thresholds` - Probability threshold per label (must match num_outputs)
    ///
    /// # Returns
    /// Vector of boolean vectors: `result[sample][label]`
    pub fn predict_labels_with_thresholds(
        &self,
        dataset: &BinnedDataset,
        thresholds: &[f32],
    ) -> Vec<Vec<bool>> {
        debug_assert_eq!(
            thresholds.len(),
            self.num_outputs(),
            "thresholds length must match num_outputs"
        );

        let proba = self.predict_proba_multilabel(dataset);
        proba
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(thresholds.iter())
                    .map(|(p, &t)| p >= t)
                    .collect()
            })
            .collect()
    }

    // ============================================================================
    // Multi-label raw prediction methods (from raw features, no binning needed)
    // ============================================================================

    /// Predict raw scores from raw features for multi-label classification
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    ///
    /// # Returns
    /// Vector of vectors: `result[sample][label]`
    pub fn predict_multilabel_raw(&self, features: &[f64]) -> Vec<Vec<f32>> {
        if !self.is_multilabel() {
            return self
                .predict_raw(features)
                .into_iter()
                .map(|p| vec![p])
                .collect();
        }

        // Check if we're using unified vector trees or legacy scalar trees
        if self.uses_vector_trees() {
            self.predict_multilabel_vector_raw(features)
        } else {
            self.predict_multilabel_scalar_raw(features)
        }
    }

    /// Predict using unified vector trees with raw features
    fn predict_multilabel_vector_raw(&self, features: &[f64]) -> Vec<Vec<f32>> {
        let num_features = self.num_features();
        if num_features == 0 {
            return vec![];
        }

        let num_rows = features.len() / num_features;
        let num_outputs = self.num_outputs();

        // Initialize with base predictions
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_outputs);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add vector tree contributions
        for tree in &self.trees {
            let vector_tree = tree.as_vector();
            vector_tree.predict_batch_add_raw(features, num_features, &mut raw_preds);
        }

        // Convert to Vec<Vec<f32>>
        raw_preds
            .chunks_exact(num_outputs)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Predict using legacy scalar trees with raw features
    fn predict_multilabel_scalar_raw(&self, features: &[f64]) -> Vec<Vec<f32>> {
        let num_features = self.num_features();
        if num_features == 0 {
            return vec![];
        }

        let num_rows = features.len() / num_features;
        let num_outputs = self.num_outputs();
        let num_rounds = self.trees.len() / num_outputs;

        // Initialize with base predictions
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_outputs);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add tree contributions
        for round in 0..num_rounds {
            for label_idx in 0..num_outputs {
                let tree_idx = round * num_outputs + label_idx;
                let tree = self.trees[tree_idx].as_scalar();

                for row_idx in 0..num_rows {
                    let row_offset = row_idx * num_features;
                    let delta = tree.predict_raw(|f| features[row_offset + f]);
                    raw_preds[row_idx * num_outputs + label_idx] += delta;
                }
            }
        }

        // Convert to Vec<Vec<f32>>
        raw_preds
            .chunks_exact(num_outputs)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Predict probabilities from raw features for multi-label classification
    ///
    /// # Returns
    /// Vector of probability vectors: `result[sample][label]`
    pub fn predict_proba_multilabel_raw(&self, features: &[f64]) -> Vec<Vec<f32>> {
        let raw = self.predict_multilabel_raw(features);
        raw.into_iter()
            .map(|row| row.into_iter().map(sigmoid).collect())
            .collect()
    }

    /// Predict binary labels from raw features for multi-label classification
    ///
    /// Uses threshold 0.5 for all labels.
    pub fn predict_labels_raw(&self, features: &[f64]) -> Vec<Vec<bool>> {
        self.predict_labels_with_threshold_raw(features, 0.5)
    }

    /// Predict binary labels from raw features with custom threshold
    pub fn predict_labels_with_threshold_raw(
        &self,
        features: &[f64],
        threshold: f32,
    ) -> Vec<Vec<bool>> {
        let proba = self.predict_proba_multilabel_raw(features);
        proba
            .into_iter()
            .map(|row| row.into_iter().map(|p| p >= threshold).collect())
            .collect()
    }

    /// Predict class probabilities for multi-class classification
    ///
    /// Applies softmax to raw predictions to get probabilities for each class.
    /// Only meaningful when trained with `with_multiclass_logloss()`.
    ///
    /// # Returns
    /// Vector of probability vectors: `result[sample][class]`
    pub fn predict_proba_multiclass(&self, dataset: &BinnedDataset) -> Vec<Vec<f32>> {
        if !matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            // Not a multi-class model, fall back to binary
            return self
                .predict_proba(dataset)
                .into_iter()
                .map(|p| vec![1.0 - p, p])
                .collect();
        }

        let num_rows = dataset.num_rows();
        let num_classes = self.num_outputs();
        let num_rounds = self.trees.len() / num_classes;

        // Initialize raw predictions with base values
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_classes);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add tree predictions
        // Trees are stored as: [round0_class0, round0_class1, ..., round0_classK, round1_class0, ...]
        for round in 0..num_rounds {
            for class_idx in 0..num_classes {
                let tree_idx = round * num_classes + class_idx;
                let tree = &self.trees[tree_idx];

                for row_idx in 0..num_rows {
                    let delta = tree.predict(|f| dataset.get_bin(row_idx, f));
                    raw_preds[row_idx * num_classes + class_idx] += delta;
                }
            }
        }

        // Apply softmax to each row
        let mut result = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let row_preds = &raw_preds[row_idx * num_classes..(row_idx + 1) * num_classes];
            result.push(softmax(row_preds));
        }

        result
    }

    /// Predict class labels for multi-class classification
    ///
    /// Returns the class with highest probability (argmax of softmax).
    /// Only meaningful when trained with `with_multiclass_logloss()`.
    ///
    /// # Returns
    /// Vector of class labels (0, 1, 2, ..., K-1)
    pub fn predict_class_multiclass(&self, dataset: &BinnedDataset) -> Vec<u32> {
        let proba = self.predict_proba_multiclass(dataset);
        proba
            .iter()
            .map(|p| {
                p.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict raw scores for multi-class classification (before softmax)
    ///
    /// Returns raw predictions for each class (not probabilities).
    /// Shape: `result[sample][class]`
    pub fn predict_raw_multiclass(&self, dataset: &BinnedDataset) -> Vec<Vec<f32>> {
        if !matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            // Not a multi-class model
            return self.predict(dataset).into_iter().map(|p| vec![p]).collect();
        }

        let num_rows = dataset.num_rows();
        let num_classes = self.num_outputs();
        let num_rounds = self.trees.len() / num_classes;

        // Initialize raw predictions with base values
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_classes);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add tree predictions
        for round in 0..num_rounds {
            for class_idx in 0..num_classes {
                let tree_idx = round * num_classes + class_idx;
                let tree = &self.trees[tree_idx];

                for row_idx in 0..num_rows {
                    let delta = tree.predict(|f| dataset.get_bin(row_idx, f));
                    raw_preds[row_idx * num_classes + class_idx] += delta;
                }
            }
        }

        // Convert to Vec<Vec<f32>>
        let mut result = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let row_preds = &raw_preds[row_idx * num_classes..(row_idx + 1) * num_classes];
            result.push(row_preds.to_vec());
        }

        result
    }

    // ============================================================================
    // Raw prediction methods (no binning required)
    // ============================================================================

    /// Predict using raw feature values (no binning needed)
    ///
    /// This is the primary prediction method for external use (e.g., Python bindings).
    /// Uses the split_value stored in tree nodes to compare directly against raw values,
    /// avoiding the overhead of binning on every prediction call.
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: features[row * num_features + feature]
    ///   Shape: (num_rows, num_features)
    ///
    /// # Returns
    /// Vector of predictions for each row
    pub fn predict_raw(&self, features: &[f64]) -> Vec<f32> {
        let num_features = self.num_features();
        if num_features == 0 {
            return vec![];
        }

        let num_rows = features.len() / num_features;
        debug_assert_eq!(features.len(), num_rows * num_features);

        if self.config.parallel_prediction && num_rows >= 1000 {
            self.predict_raw_parallel(features, num_features)
        } else {
            self.predict_raw_sequential(features, num_features)
        }
    }

    /// Single-threaded raw prediction using tree-wise traversal
    fn predict_raw_sequential(&self, features: &[f64], num_features: usize) -> Vec<f32> {
        let num_rows = features.len() / num_features;

        // Initialize predictions with base value
        let mut predictions = vec![self.base_predictions[0]; num_rows];

        // Tree-wise: traverse each tree for all rows
        for tree in &self.trees {
            tree.predict_batch_add_raw(features, num_features, &mut predictions);
        }

        predictions
    }

    /// Parallel raw prediction using tree-wise traversal
    fn predict_raw_parallel(&self, features: &[f64], num_features: usize) -> Vec<f32> {
        let num_rows = features.len() / num_features;

        // For small datasets, use sequential
        if num_rows < 1000 || self.trees.is_empty() {
            return self.predict_raw_sequential(features, num_features);
        }

        // Initialize predictions with base value
        let mut predictions = vec![self.base_predictions[0]; num_rows];

        // Determine chunk size for parallelism
        let num_threads = rayon::current_num_threads();
        let chunk_size = (num_rows / (num_threads * 4)).max(256);

        // Process chunks in parallel
        predictions
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_row = chunk_idx * chunk_size;
                let chunk_features_start = start_row * num_features;

                // Each thread processes its chunk through all trees
                for tree in &self.trees {
                    for (i, pred) in chunk.iter_mut().enumerate() {
                        let row_offset = chunk_features_start + i * num_features;
                        *pred += tree.predict_raw(|f| features[row_offset + f]);
                    }
                }
            });

        predictions
    }

    /// Predict raw with conformal intervals
    ///
    /// Returns (predictions, lower_bounds, upper_bounds)
    pub fn predict_raw_with_intervals(&self, features: &[f64]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let predictions = self.predict_raw(features);

        let q = self.conformal_q.unwrap_or(0.0);
        let lower: Vec<f32> = predictions.iter().map(|&p| p - q).collect();
        let upper: Vec<f32> = predictions.iter().map(|&p| p + q).collect();

        (predictions, lower, upper)
    }

    /// Predict class probabilities from raw features (for binary classification)
    ///
    /// Applies sigmoid to raw predictions to get probabilities in [0, 1].
    /// Only meaningful when trained with `with_binary_logloss()`.
    pub fn predict_proba_raw(&self, features: &[f64]) -> Vec<f32> {
        let raw = self.predict_raw(features);
        raw.iter().map(|&r| sigmoid(r)).collect()
    }

    /// Predict class labels from raw features (for binary classification)
    ///
    /// Applies sigmoid to raw predictions and thresholds.
    /// Only meaningful when trained with `with_binary_logloss()`.
    pub fn predict_class_raw(&self, features: &[f64], threshold: f32) -> Vec<u32> {
        let proba = self.predict_proba_raw(features);
        proba
            .iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect()
    }

    // ============================================================================
    // Multi-class raw prediction methods (from raw features, no binning needed)
    // ============================================================================

    /// Predict class probabilities from raw features (for multi-class classification)
    ///
    /// Uses the split_value stored in tree nodes to compare directly against raw values.
    /// Applies softmax to raw predictions to get probabilities for each class.
    /// Only meaningful when trained with `with_multiclass_logloss()`.
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    ///
    /// # Returns
    /// Vector of probability vectors: `result[sample][class]`
    pub fn predict_proba_multiclass_raw(&self, features: &[f64]) -> Vec<Vec<f32>> {
        if !matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            // Not a multi-class model, fall back to binary
            return self
                .predict_proba_raw(features)
                .into_iter()
                .map(|p| vec![1.0 - p, p])
                .collect();
        }

        let num_features = self.num_features();
        if num_features == 0 {
            return vec![];
        }

        let num_rows = features.len() / num_features;
        let num_classes = self.num_outputs();
        let num_rounds = self.trees.len() / num_classes;

        // Initialize raw predictions with base values
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_classes);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add tree predictions
        // Trees are stored as: [round0_class0, round0_class1, ..., round0_classK, round1_class0, ...]
        for round in 0..num_rounds {
            for class_idx in 0..num_classes {
                let tree_idx = round * num_classes + class_idx;
                let tree = &self.trees[tree_idx];

                for row_idx in 0..num_rows {
                    let row_offset = row_idx * num_features;
                    let delta = tree.predict_raw(|f| features[row_offset + f]);
                    raw_preds[row_idx * num_classes + class_idx] += delta;
                }
            }
        }

        // Apply softmax to each row
        let mut result = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let row_preds = &raw_preds[row_idx * num_classes..(row_idx + 1) * num_classes];
            result.push(softmax(row_preds));
        }

        result
    }

    /// Predict class labels from raw features (for multi-class classification)
    ///
    /// Returns the class with highest probability (argmax of softmax).
    /// Only meaningful when trained with `with_multiclass_logloss()`.
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: features[row * num_features + feature]
    ///
    /// # Returns
    /// Vector of class labels (0, 1, 2, ..., K-1)
    pub fn predict_class_multiclass_raw(&self, features: &[f64]) -> Vec<u32> {
        let proba = self.predict_proba_multiclass_raw(features);
        proba
            .iter()
            .map(|p| {
                p.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict raw scores from raw features (for multi-class, before softmax)
    ///
    /// Returns raw predictions for each class (not probabilities).
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    ///
    /// # Returns
    /// Vector of raw score vectors: `result[sample][class]`
    pub fn predict_raw_multiclass_raw(&self, features: &[f64]) -> Vec<Vec<f32>> {
        if !matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            // Not a multi-class model
            return self
                .predict_raw(features)
                .into_iter()
                .map(|p| vec![p])
                .collect();
        }

        let num_features = self.num_features();
        if num_features == 0 {
            return vec![];
        }

        let num_rows = features.len() / num_features;
        let num_classes = self.num_outputs();
        let num_rounds = self.trees.len() / num_classes;

        // Initialize raw predictions with base values
        let mut raw_preds: Vec<f32> = Vec::with_capacity(num_rows * num_classes);
        for _ in 0..num_rows {
            raw_preds.extend_from_slice(&self.base_predictions);
        }

        // Add tree predictions
        for round in 0..num_rounds {
            for class_idx in 0..num_classes {
                let tree_idx = round * num_classes + class_idx;
                let tree = &self.trees[tree_idx];

                for row_idx in 0..num_rows {
                    let row_offset = row_idx * num_features;
                    let delta = tree.predict_raw(|f| features[row_offset + f]);
                    raw_preds[row_idx * num_classes + class_idx] += delta;
                }
            }
        }

        // Convert to Vec<Vec<f32>>
        let mut result = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let row_preds = &raw_preds[row_idx * num_classes..(row_idx + 1) * num_classes];
            result.push(row_preds.to_vec());
        }

        result
    }
}
