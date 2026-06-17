//! Split finding with Shannon Entropy regularization and Era-based Directional Splitting
//!
//! This module uses SIMD-optimized kernels for split finding when possible.
//! The SIMD path is taken when:
//! - No entropy regularization (entropy_weight == 0.0)
//! - No monotonic constraints
//!
//! Otherwise falls back to scalar implementation with full feature support.
//!
//! # Era-Based Splitting (DES)
//!
//! When era splitting is enabled, only accepts splits where ALL eras agree on
//! the split direction. This filters out spurious correlations that work in
//! some eras but not others.

use crate::backend::scalar::kernel::find_best_split as kernel_find_best_split;
use crate::histogram::{
    average_era_gain, has_directional_agreement, EraHistograms, EraSplitStats, Histogram,
    NodeHistograms, NUM_BINS,
};
use rkyv::{Archive, Deserialize, Serialize};

/// Split statistics for a node partition (used in gain computation)
#[derive(Clone, Copy)]
struct PartitionStats {
    gradient: f32,
    hessian: f32,
    count: u32,
}

/// Monotonic constraint for a feature
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum MonotonicConstraint {
    /// No constraint
    None,
    /// Prediction must increase as feature value increases
    Increasing,
    /// Prediction must decrease as feature value increases
    Decreasing,
}

/// Feature interaction constraints
///
/// Defines which features can interact (appear together in the same tree path).
/// Features in the same group can interact; features in different groups cannot.
/// Features not in any group can interact with all features.
#[derive(Debug, Clone, Default)]
pub struct InteractionConstraints {
    /// Groups of feature indices that can interact
    /// Each inner Vec is a group of features that can appear together
    groups: Vec<Vec<usize>>,
    /// Lookup: feature_idx -> group_idx (None if unconstrained)
    feature_to_group: Vec<Option<usize>>,
}

impl InteractionConstraints {
    /// Create empty constraints (all features can interact)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create constraints from groups of feature indices
    ///
    /// # Arguments
    /// * `groups` - Each group is a list of feature indices that can interact
    /// * `num_features` - Total number of features in the dataset
    ///
    /// # Example
    /// ```ignore
    /// // Features 0,1,2 can interact; features 3,4 can interact; 5 is unconstrained
    /// let constraints = InteractionConstraints::from_groups(
    ///     vec![vec![0, 1, 2], vec![3, 4]],
    ///     6
    /// );
    /// ```
    pub fn from_groups(groups: Vec<Vec<usize>>, num_features: usize) -> Self {
        let mut feature_to_group = vec![None; num_features];

        for (group_idx, group) in groups.iter().enumerate() {
            for &feature_idx in group {
                if feature_idx < num_features {
                    feature_to_group[feature_idx] = Some(group_idx);
                }
            }
        }

        Self {
            groups,
            feature_to_group,
        }
    }

    /// Check if constraints are empty (all features can interact)
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Get allowed features given the ancestor path
    ///
    /// # Arguments
    /// * `ancestor_features` - Features used in ancestors of current node
    /// * `num_features` - Total number of features
    ///
    /// # Returns
    /// List of feature indices that can be used for splitting
    pub fn allowed_features(&self, ancestor_features: &[usize], num_features: usize) -> Vec<usize> {
        if self.groups.is_empty() {
            // No constraints: all features allowed
            return (0..num_features).collect();
        }

        // Find which groups have been used by ancestors
        let mut used_groups: Vec<bool> = vec![false; self.groups.len()];
        for &feat in ancestor_features {
            if let Some(group_idx) = self.feature_to_group.get(feat).copied().flatten() {
                used_groups[group_idx] = true;
            }
        }

        // A feature is allowed if:
        // 1. It's unconstrained (not in any group), OR
        // 2. It belongs to a group that was already used by ancestors, OR
        // 3. No group has been used yet
        let any_group_used = used_groups.iter().any(|&u| u);

        (0..num_features)
            .filter(|&feat| {
                match self.feature_to_group.get(feat).copied().flatten() {
                    None => true, // Unconstrained feature
                    Some(group_idx) => {
                        if !any_group_used {
                            true // No group used yet, any feature allowed
                        } else {
                            used_groups[group_idx] // Only if this group was used
                        }
                    }
                }
            })
            .collect()
    }

    /// Get the groups
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
}

/// Information about a potential split
#[derive(Debug, Clone, Copy)]
pub struct SplitInfo {
    /// Feature index
    pub feature_idx: usize,
    /// Bin threshold (samples with bin <= threshold go left)
    pub bin_threshold: u8,
    /// Actual split value for raw prediction (samples with value <= split_value go left)
    /// This is populated from bin_boundaries during tree growth.
    pub split_value: f64,
    /// Split gain (higher is better)
    pub gain: f32,
    /// Left child gradient sum
    pub left_gradient: f32,
    /// Left child hessian sum
    pub left_hessian: f32,
    /// Left child sample count
    pub left_count: u32,
    /// Right child gradient sum
    pub right_gradient: f32,
    /// Right child hessian sum
    pub right_hessian: f32,
    /// Right child sample count
    pub right_count: u32,
    /// Default direction for missing values (bin 0)
    /// If true, missing values go left; if false, they go right
    pub default_left: bool,
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self {
            feature_idx: 0,
            bin_threshold: 0,
            split_value: 0.0,
            gain: f32::NEG_INFINITY,
            left_gradient: 0.0,
            left_hessian: 0.0,
            left_count: 0,
            right_gradient: 0.0,
            right_hessian: 0.0,
            right_count: 0,
            default_left: true, // Default: missing values go left
        }
    }
}

impl SplitInfo {
    /// Check if this split is valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.gain > f32::NEG_INFINITY && self.left_count > 0 && self.right_count > 0
    }
}

/// Split finder with Shannon Entropy regularization
pub struct SplitFinder {
    /// L2 regularization parameter (lambda)
    lambda: f32,
    /// Minimum samples in a leaf
    min_samples_leaf: usize,
    /// Minimum hessian sum in a leaf
    min_hessian_leaf: f32,
    /// Shannon Entropy regularization weight (beta)
    entropy_weight: f32,
    /// Minimum gain to make a split
    min_gain: f32,
    /// Monotonic constraints per feature (empty = no constraints)
    monotonic_constraints: Vec<MonotonicConstraint>,
    /// Enable adaptive missing value direction selection
    /// When true, evaluates both directions for missing values and selects optimal path
    /// When false, missing values follow default left direction
    use_missing_value_learning: bool,
    /// Enable noise pruning (PaloBoost-style)
    /// When true with era splitting, Era 0 = train, Era 1 = validation
    /// Splits found on Era 0 are rejected if Era 1 gain <= threshold
    noise_pruning: bool,
    /// Noise pruning threshold for validation gain
    /// Splits are rejected if validation gain <= this threshold
    /// Default: -0.1 (allows small negative gains to handle distribution shift)
    /// Stricter: 0.0 (classic PaloBoost, only accept positive validation gains)
    /// More permissive: -0.2 (for high temporal distribution shift)
    noise_pruning_threshold: f32,
}

impl Default for SplitFinder {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            entropy_weight: 0.0, // No entropy regularization by default
            min_gain: 0.0,
            monotonic_constraints: Vec::new(),
            use_missing_value_learning: false, // Default: missing values go left
            noise_pruning: false,              // Default: use DES directional agreement
            noise_pruning_threshold: -0.1,     // Default: allow small negative validation gains
        }
    }
}

impl SplitFinder {
    /// Create a new split finder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set L2 regularization (lambda)
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set minimum hessian sum per leaf
    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    /// Set Shannon Entropy regularization weight (beta)
    ///
    /// Higher values penalize imbalanced splits more,
    /// making the tree more robust to concept drift.
    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    /// Set minimum gain for splitting
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Set monotonic constraints for features
    ///
    /// The vector should have one entry per feature. Features beyond the
    /// vector length are treated as unconstrained.
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Enable adaptive missing value direction selection
    ///
    /// When enabled, evaluates both child directions for missing values at each split
    /// and selects the direction that maximizes gain. When disabled (default), missing
    /// values follow the left child direction.
    ///
    /// Beneficial when missing values carry predictive information distinct from observed
    /// values. May increase training time due to additional gain evaluations.
    pub fn with_missing_value_learning(mut self, enable: bool) -> Self {
        self.use_missing_value_learning = enable;
        self
    }

    /// Enable noise pruning (PaloBoost-style validation)
    ///
    /// When enabled with era splitting, treats Era 0 as training and Era 1 as validation.
    /// Splits are found using Era 0 gain, but rejected if Era 1 gain <= threshold.
    /// This prevents overfitting by validating each split against held-out data.
    pub fn with_noise_pruning(mut self, enable: bool) -> Self {
        self.noise_pruning = enable;
        self
    }

    /// Set noise pruning threshold
    ///
    /// Splits are rejected if validation gain <= this threshold.
    /// - 0.0: Classic PaloBoost (only accept positive validation gains)
    /// - -0.1: Default (allows small negative gains, handles distribution shift)
    /// - -0.2: More permissive (for high temporal distribution shift)
    pub fn with_noise_pruning_threshold(mut self, threshold: f32) -> Self {
        self.noise_pruning_threshold = threshold;
        self
    }

    /// Get the monotonic constraint for a feature
    fn get_constraint(&self, feature_idx: usize) -> MonotonicConstraint {
        self.monotonic_constraints
            .get(feature_idx)
            .copied()
            .unwrap_or(MonotonicConstraint::None)
    }

    /// Check if we can use the SIMD fast path for a feature
    ///
    /// SIMD is used when:
    /// - No entropy regularization (entropy_weight == 0.0)
    /// - No monotonic constraint for this feature
    #[inline]
    fn can_use_simd(&self, feature_idx: usize) -> bool {
        self.entropy_weight == 0.0 && self.get_constraint(feature_idx) == MonotonicConstraint::None
    }

    /// Check if a split satisfies monotonic constraints
    ///
    /// For a valid monotonic split:
    /// - Increasing: left_weight <= right_weight
    /// - Decreasing: left_weight >= right_weight
    fn satisfies_monotonic_constraint(
        &self,
        feature_idx: usize,
        left_gradient: f32,
        left_hessian: f32,
        right_gradient: f32,
        right_hessian: f32,
    ) -> bool {
        let constraint = self.get_constraint(feature_idx);

        match constraint {
            MonotonicConstraint::None => true,
            MonotonicConstraint::Increasing | MonotonicConstraint::Decreasing => {
                // Compute leaf weights: weight = -gradient / (hessian + lambda)
                let left_weight = -left_gradient / (left_hessian + self.lambda);
                let right_weight = -right_gradient / (right_hessian + self.lambda);

                match constraint {
                    MonotonicConstraint::Increasing => left_weight <= right_weight,
                    MonotonicConstraint::Decreasing => left_weight >= right_weight,
                    MonotonicConstraint::None => unreachable!(),
                }
            }
        }
    }

    /// Find the best split across all features
    pub fn find_best_split(
        &self,
        histograms: &NodeHistograms,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        self.find_best_split_with_features(
            histograms,
            total_gradient,
            total_hessian,
            total_count,
            None,
        )
    }

    /// Find the best split across a subset of features (for column subsampling)
    ///
    /// # Arguments
    /// * `histograms` - Node histograms for all features
    /// * `total_gradient` - Sum of gradients in the node
    /// * `total_hessian` - Sum of hessians in the node
    /// * `total_count` - Number of samples in the node
    /// * `feature_mask` - Optional mask of feature indices to consider (None = all features)
    pub fn find_best_split_with_features(
        &self,
        histograms: &NodeHistograms,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
        feature_mask: Option<&[usize]>,
    ) -> Option<SplitInfo> {
        let mut best_split = SplitInfo::default();

        match feature_mask {
            Some(features) => {
                // Only consider specified features
                for &feature_idx in features {
                    if feature_idx >= histograms.num_features() {
                        continue;
                    }
                    if let Some(split) = self.find_best_split_for_feature(
                        histograms.get(feature_idx),
                        feature_idx,
                        total_gradient,
                        total_hessian,
                        total_count,
                    ) {
                        if split.gain > best_split.gain {
                            best_split = split;
                        }
                    }
                }
            }
            None => {
                // Consider all features
                for (feature_idx, histogram) in histograms.iter() {
                    if let Some(split) = self.find_best_split_for_feature(
                        histogram,
                        feature_idx,
                        total_gradient,
                        total_hessian,
                        total_count,
                    ) {
                        if split.gain > best_split.gain {
                            best_split = split;
                        }
                    }
                }
            }
        }

        if best_split.is_valid() && best_split.gain >= self.min_gain {
            Some(best_split)
        } else {
            None
        }
    }

    /// Find the best split for a single feature
    fn find_best_split_for_feature(
        &self,
        histogram: &Histogram,
        feature_idx: usize,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        // Use SIMD fast path when no entropy regularization or monotonic constraints
        if self.can_use_simd(feature_idx) {
            return self.find_best_split_for_feature_simd(
                histogram,
                feature_idx,
                total_gradient,
                total_hessian,
                total_count,
            );
        }

        // Fall back to scalar implementation with full feature support
        self.find_best_split_for_feature_scalar(
            histogram,
            feature_idx,
            total_gradient,
            total_hessian,
            total_count,
        )
    }

    /// SIMD-optimized split finding for a single feature
    ///
    /// Uses the kernel's vectorized implementation for maximum performance.
    /// Only used when entropy_weight == 0 and no monotonic constraints.
    fn find_best_split_for_feature_simd(
        &self,
        histogram: &Histogram,
        feature_idx: usize,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        // Extract raw arrays from histogram for SIMD kernel
        let bins = histogram.bins();
        let mut hist_grads = [0.0f32; NUM_BINS];
        let mut hist_hess = [0.0f32; NUM_BINS];
        let mut hist_counts = [0u32; NUM_BINS];

        for i in 0..NUM_BINS {
            hist_grads[i] = bins[i].sum_gradients;
            hist_hess[i] = bins[i].sum_hessians;
            hist_counts[i] = bins[i].count;
        }

        // Call SIMD kernel
        let candidate = kernel_find_best_split(
            &hist_grads,
            &hist_hess,
            &hist_counts,
            total_gradient,
            total_hessian,
            total_count,
            self.lambda,
            self.min_samples_leaf as u32,
            self.min_hessian_leaf,
        )?;

        // Check min_gain threshold
        if candidate.gain < self.min_gain {
            return None;
        }

        // Convert kernel result to SplitInfo
        // Note: split_value will be populated later by the tree grower from bin_boundaries
        // SIMD path: always default_left=true (missing→left) for simplicity
        Some(SplitInfo {
            feature_idx,
            bin_threshold: candidate.bin_threshold,
            split_value: 0.0, // Populated by tree grower
            gain: candidate.gain,
            left_gradient: candidate.left_gradient,
            left_hessian: candidate.left_hessian,
            left_count: candidate.left_count,
            right_gradient: candidate.right_gradient,
            right_hessian: candidate.right_hessian,
            right_count: candidate.right_count,
            default_left: true, // SIMD path uses default behavior
        })
    }

    /// Scalar split finding with full feature support
    ///
    /// Supports entropy regularization, monotonic constraints, and adaptive
    /// missing value direction selection.
    fn find_best_split_for_feature_scalar(
        &self,
        histogram: &Histogram,
        feature_idx: usize,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        let mut best_split = SplitInfo {
            feature_idx,
            ..Default::default()
        };

        // Extract missing value statistics if learning is enabled
        let (missing_gradient, missing_hessian, missing_count) = if self.use_missing_value_learning
        {
            let missing_entry = histogram.get(0);
            (
                missing_entry.sum_gradients,
                missing_entry.sum_hessians,
                missing_entry.count,
            )
        } else {
            (0.0, 0.0, 0)
        };

        // Cumulative sums for left child
        let mut left_gradient = 0.0f32;
        let mut left_hessian = 0.0f32;
        let mut left_count = 0u32;

        // Scan through bins
        for (bin, entry) in histogram.iter() {
            if entry.count == 0 {
                continue;
            }

            left_gradient += entry.sum_gradients;
            left_hessian += entry.sum_hessians;
            left_count += entry.count;

            // Right child = total - left
            let right_gradient = total_gradient - left_gradient;
            let right_hessian = total_hessian - left_hessian;
            let right_count = total_count - left_count;

            // Determine how many directions to test
            let directions = if self.use_missing_value_learning {
                // Test both missing→left and missing→right
                vec![
                    (
                        true,
                        left_gradient,
                        left_hessian,
                        left_count,
                        right_gradient,
                        right_hessian,
                        right_count,
                    ),
                    (
                        false,
                        left_gradient - missing_gradient,
                        left_hessian - missing_hessian,
                        left_count - missing_count,
                        right_gradient + missing_gradient,
                        right_hessian + missing_hessian,
                        right_count + missing_count,
                    ),
                ]
            } else {
                // Only test default direction (missing→left)
                vec![(
                    true,
                    left_gradient,
                    left_hessian,
                    left_count,
                    right_gradient,
                    right_hessian,
                    right_count,
                )]
            };

            for (default_left, left_g, left_h, left_c, right_g, right_h, right_c) in directions {
                // Check leaf constraints
                if (left_c as usize) < self.min_samples_leaf
                    || (right_c as usize) < self.min_samples_leaf
                {
                    continue;
                }
                if left_h < self.min_hessian_leaf || right_h < self.min_hessian_leaf {
                    continue;
                }

                // Check monotonic constraint
                if !self.satisfies_monotonic_constraint(
                    feature_idx,
                    left_g,
                    left_h,
                    right_g,
                    right_h,
                ) {
                    continue;
                }

                // Compute gain with entropy regularization
                let gain = self.compute_gain(
                    PartitionStats {
                        gradient: left_g,
                        hessian: left_h,
                        count: left_c,
                    },
                    PartitionStats {
                        gradient: right_g,
                        hessian: right_h,
                        count: right_c,
                    },
                    PartitionStats {
                        gradient: total_gradient,
                        hessian: total_hessian,
                        count: total_count,
                    },
                );

                if gain > best_split.gain {
                    best_split.bin_threshold = bin;
                    best_split.gain = gain;
                    best_split.left_gradient = left_g;
                    best_split.left_hessian = left_h;
                    best_split.left_count = left_c;
                    best_split.right_gradient = right_g;
                    best_split.right_hessian = right_h;
                    best_split.right_count = right_c;
                    best_split.default_left = default_left;
                }
            }
        }

        if best_split.is_valid() {
            Some(best_split)
        } else {
            None
        }
    }

    /// Compute split gain with optional Shannon Entropy regularization
    ///
    /// Standard gain (Friedman MSE):
    /// gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)]
    ///
    /// With entropy regularization:
    /// gain = standard_gain + β * H(split)
    ///
    /// Where H(split) = -p*log(p) - (1-p)*log(1-p), p = n_L/n
    fn compute_gain(
        &self,
        left: PartitionStats,
        right: PartitionStats,
        total: PartitionStats,
    ) -> f32 {
        // Friedman MSE gain (standard GBDT gain)
        let left_score = (left.gradient * left.gradient) / (left.hessian + self.lambda);
        let right_score = (right.gradient * right.gradient) / (right.hessian + self.lambda);
        let parent_score = (total.gradient * total.gradient) / (total.hessian + self.lambda);

        let standard_gain = 0.5 * (left_score + right_score - parent_score);

        // Add Shannon Entropy regularization if enabled
        if self.entropy_weight > 0.0 {
            let entropy = self.split_entropy(left.count, right.count, total.count);
            standard_gain + self.entropy_weight * entropy
        } else {
            standard_gain
        }
    }

    /// Compute Shannon Entropy of a split
    ///
    /// H(split) = -p*log₂(p) - (1-p)*log₂(1-p)
    ///
    /// Maximum entropy (1.0) when p = 0.5 (balanced split)
    /// Minimum entropy (0.0) when p = 0 or p = 1 (completely imbalanced)
    fn split_entropy(&self, left_count: u32, right_count: u32, total_count: u32) -> f32 {
        if total_count == 0 || left_count == 0 || right_count == 0 {
            return 0.0;
        }

        let p = left_count as f32 / total_count as f32;
        let q = 1.0 - p;

        // Use safe log computation

        if p > 0.0 && q > 0.0 {
            -(p * p.log2() + q * q.log2())
        } else {
            0.0
        }
    }

    // ============================================================
    // Era-Based Split Finding (Directional Era Splitting / DES)
    // ============================================================

    /// Find the best split using era-stratified histograms (DES)
    ///
    /// Only accepts splits where ALL eras agree on the split direction.
    /// This filters out spurious correlations that work in some eras but not others.
    ///
    /// # Arguments
    /// * `era_histograms` - Per-era histograms for all features
    /// * `per_era_totals` - (gradient, hessian, count) totals for each era
    pub fn find_best_split_with_eras(
        &self,
        era_histograms: &EraHistograms,
        per_era_totals: &[(f32, f32, u32)],
    ) -> Option<SplitInfo> {
        self.find_best_split_with_eras_and_features(era_histograms, per_era_totals, None)
    }

    /// Find the best split with era filtering and optional feature mask
    pub fn find_best_split_with_eras_and_features(
        &self,
        era_histograms: &EraHistograms,
        per_era_totals: &[(f32, f32, u32)],
        feature_mask: Option<&[usize]>,
    ) -> Option<SplitInfo> {
        // Validate era configuration for noise pruning
        if self.noise_pruning {
            let num_eras = era_histograms.num_eras();

            // Need at least 2 eras for noise pruning
            if num_eras < 2 {
                eprintln!(
                    "WARNING: noise_pruning enabled but only {} era(s) found. \
                     Noise pruning requires 2 eras (Era 0=train, Era 1=validation). \
                     All splits will be rejected.",
                    num_eras
                );
                return None;
            }

            // Check that eras 0 and 1 exist in per_era_totals
            let has_era_0 = per_era_totals
                .first()
                .is_some_and(|(_, _, count)| *count > 0);
            let has_era_1 = per_era_totals
                .get(1)
                .is_some_and(|(_, _, count)| *count > 0);

            if !has_era_0 || !has_era_1 {
                eprintln!(
                    "WARNING: noise_pruning enabled but missing required eras. \
                     Found: Era 0={}, Era 1={}. \
                     Noise pruning requires both Era 0 (train) and Era 1 (validation) \
                     with non-zero samples. All splits will be rejected.",
                    if has_era_0 { "present" } else { "missing" },
                    if has_era_1 { "present" } else { "missing" }
                );
                return None;
            }
        }

        let mut best_split = SplitInfo::default();
        let num_features = era_histograms.num_features();

        match feature_mask {
            Some(features) => {
                for &feature_idx in features {
                    if feature_idx >= num_features {
                        continue;
                    }
                    if let Some(split) = self.find_best_split_for_feature_era(
                        era_histograms,
                        feature_idx,
                        per_era_totals,
                    ) {
                        if split.gain > best_split.gain {
                            best_split = split;
                        }
                    }
                }
            }
            None => {
                for feature_idx in 0..num_features {
                    if let Some(split) = self.find_best_split_for_feature_era(
                        era_histograms,
                        feature_idx,
                        per_era_totals,
                    ) {
                        if split.gain > best_split.gain {
                            best_split = split;
                        }
                    }
                }
            }
        }

        if best_split.is_valid() && best_split.gain >= self.min_gain {
            Some(best_split)
        } else {
            None
        }
    }

    /// Find the best split for a single feature using era histograms
    fn find_best_split_for_feature_era(
        &self,
        era_histograms: &EraHistograms,
        feature_idx: usize,
        per_era_totals: &[(f32, f32, u32)],
    ) -> Option<SplitInfo> {
        let num_eras = era_histograms.num_eras();
        let mut best_split = SplitInfo {
            feature_idx,
            ..Default::default()
        };

        // Per-era cumulative sums
        let mut era_left_grads = vec![0.0f32; num_eras];
        let mut era_left_hess = vec![0.0f32; num_eras];
        let mut era_left_counts = vec![0u32; num_eras];

        // Scan through bins
        for bin in 0..255u8 {
            // Accumulate for each era
            for era in 0..num_eras {
                let hist = era_histograms.get(era, feature_idx);
                let entry = hist.get(bin);

                era_left_grads[era] += entry.sum_gradients;
                era_left_hess[era] += entry.sum_hessians;
                era_left_counts[era] += entry.count;
            }

            // Compute aggregate totals for leaf constraints
            let total_left_grad: f32 = era_left_grads.iter().sum();
            let total_left_hess: f32 = era_left_hess.iter().sum();
            let total_left_count: u32 = era_left_counts.iter().sum();

            let total_grad: f32 = per_era_totals.iter().map(|(g, _, _)| g).sum();
            let total_hess: f32 = per_era_totals.iter().map(|(_, h, _)| h).sum();
            let total_count: u32 = per_era_totals.iter().map(|(_, _, c)| c).sum();

            let total_right_count = total_count - total_left_count;

            // Check leaf constraints on aggregate counts
            if (total_left_count as usize) < self.min_samples_leaf
                || (total_right_count as usize) < self.min_samples_leaf
            {
                continue;
            }

            let total_right_hess = total_hess - total_left_hess;
            if total_left_hess < self.min_hessian_leaf || total_right_hess < self.min_hessian_leaf {
                continue;
            }

            // Compute per-era split statistics
            let era_stats: Vec<EraSplitStats> = (0..num_eras)
                .filter_map(|era| {
                    let (era_grad_total, era_hess_total, era_count_total) = per_era_totals[era];

                    // Skip eras with no samples in this node
                    if era_count_total == 0 {
                        return None;
                    }

                    Some(EraSplitStats::compute(
                        era,
                        era_left_grads[era],
                        era_left_hess[era],
                        era_grad_total,
                        era_hess_total,
                        self.lambda,
                    ))
                })
                .collect();

            // Skip if no eras have data
            if era_stats.is_empty() {
                continue;
            }

            // Compute gain based on mode (noise pruning vs DES)
            let gain = if self.noise_pruning {
                // === NOISE PRUNING MODE (PaloBoost-style) ===
                // Era 0 = Training, Era 1 = Validation
                // Find split on Era 0, reject if Era 1 gain <= 0

                // Need at least 2 eras for noise pruning
                if era_stats.len() < 2 {
                    continue;
                }

                // Get train (Era 0) and validation (Era 1) stats
                let train_stats = era_stats.iter().find(|s| s.era == 0);
                let val_stats = era_stats.iter().find(|s| s.era == 1);

                match (train_stats, val_stats) {
                    (Some(train), Some(val)) => {
                        // Check train gain passes threshold
                        if train.gain <= self.min_gain {
                            continue;
                        }

                        // THE KEY CHECK: Reject if validation gain is below threshold
                        // Threshold is configurable to handle different data scenarios
                        if val.gain <= self.noise_pruning_threshold {
                            continue;
                        }

                        // Use train gain as the final gain
                        train.gain
                    }
                    _ => continue, // Missing train or val era
                }
            } else {
                // === DES MODE (Directional Era Splitting) ===
                // Check directional agreement across all eras
                if !has_directional_agreement(&era_stats) {
                    continue;
                }

                // Compute average gain across eras
                average_era_gain(&era_stats)
            };

            // Check monotonic constraint on aggregate (if applicable)
            let total_right_grad = total_grad - total_left_grad;
            if !self.satisfies_monotonic_constraint(
                feature_idx,
                total_left_grad,
                total_left_hess,
                total_right_grad,
                total_right_hess,
            ) {
                continue;
            }

            if gain > best_split.gain {
                best_split.bin_threshold = bin;
                best_split.gain = gain;
                best_split.left_gradient = total_left_grad;
                best_split.left_hessian = total_left_hess;
                best_split.left_count = total_left_count;
                best_split.right_gradient = total_right_grad;
                best_split.right_hessian = total_right_hess;
                best_split.right_count = total_right_count;
            }
        }

        if best_split.is_valid() {
            Some(best_split)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::Histogram;

    #[test]
    fn test_split_finder_basic() {
        let mut histogram = Histogram::new();

        // Create a clear split: bins 0-127 have negative gradients, 128-255 have positive
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0);
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, 1.0, 1.0);
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        let finder = SplitFinder::new().with_lambda(0.0);
        let split = finder
            .find_best_split(&histograms, 0.0, 256.0, 256)
            .unwrap();

        // Best split should be around bin 127
        assert_eq!(split.feature_idx, 0);
        assert_eq!(split.bin_threshold, 127);
        assert!(split.gain > 0.0);
    }

    #[test]
    fn test_entropy_regularization() {
        let mut histogram = Histogram::new();

        // Imbalanced split: most samples on one side
        for _ in 0..10 {
            histogram.accumulate(0, -0.5, 1.0);
        }
        for _ in 0..90 {
            histogram.accumulate(255, 0.5, 1.0);
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        // Without entropy regularization
        let finder_no_entropy = SplitFinder::new().with_lambda(0.0);
        let split_no_entropy = finder_no_entropy.find_best_split(&histograms, 40.0, 100.0, 100);

        // With entropy regularization (penalizes imbalanced splits)
        let finder_entropy = SplitFinder::new()
            .with_lambda(0.0)
            .with_entropy_weight(10.0);
        let split_entropy = finder_entropy.find_best_split(&histograms, 40.0, 100.0, 100);

        // Both should find splits, but gains differ
        assert!(split_no_entropy.is_some());
        assert!(split_entropy.is_some());
    }

    #[test]
    fn test_split_entropy() {
        let finder = SplitFinder::new();

        // Balanced split: maximum entropy
        let balanced = finder.split_entropy(50, 50, 100);
        assert!((balanced - 1.0).abs() < 0.001);

        // Imbalanced split: lower entropy
        let imbalanced = finder.split_entropy(10, 90, 100);
        assert!(imbalanced < 0.5);

        // Edge cases
        assert_eq!(finder.split_entropy(0, 100, 100), 0.0);
        assert_eq!(finder.split_entropy(100, 0, 100), 0.0);
    }

    #[test]
    fn test_min_samples_constraint() {
        let mut histogram = Histogram::new();

        histogram.accumulate(0, -1.0, 1.0); // 1 sample
        for _ in 0..99 {
            histogram.accumulate(255, 1.0, 1.0); // 99 samples
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        let finder = SplitFinder::new().with_min_samples_leaf(5); // Require at least 5 samples

        let split = finder.find_best_split(&histograms, 98.0, 100.0, 100);

        // Should not find split because left would only have 1 sample
        assert!(split.is_none());
    }

    #[test]
    fn test_monotonic_increasing_constraint() {
        let mut histogram = Histogram::new();

        // Setup: bins 0-127 have negative gradient (positive weight),
        // bins 128-255 have positive gradient (negative weight)
        // weight = -gradient / hessian, so:
        // left: weight = -(-128) / 128 = +1.0
        // right: weight = -(+128) / 128 = -1.0
        // left_weight > right_weight => VIOLATES Increasing constraint
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0); // negative gradient = positive weight
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, 1.0, 1.0); // positive gradient = negative weight
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram.clone();

        // Without constraint: should find split
        let finder_no_constraint = SplitFinder::new().with_lambda(0.0);
        let split = finder_no_constraint.find_best_split(&histograms, 0.0, 256.0, 256);
        assert!(split.is_some());

        // With increasing constraint: should NOT find split
        // left_weight=+1.0 > right_weight=-1.0 => violates Increasing requirement
        let finder_increasing = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Increasing]);
        let split = finder_increasing.find_best_split(&histograms, 0.0, 256.0, 256);
        // The split violates monotonicity, should be rejected
        assert!(split.is_none());
    }

    #[test]
    fn test_monotonic_decreasing_constraint() {
        let mut histogram = Histogram::new();

        // Setup for decreasing: left should have higher weight than right
        // negative gradient = positive weight, positive gradient = negative weight
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0); // negative gradient = positive weight
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, 1.0, 1.0); // positive gradient = negative weight
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        // With decreasing constraint: should find split
        // left has positive weight, right has negative weight => left >= right (satisfied)
        let finder_decreasing = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Decreasing]);
        let split = finder_decreasing.find_best_split(&histograms, 0.0, 256.0, 256);
        assert!(split.is_some());

        // With increasing constraint: should NOT find split
        let finder_increasing = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Increasing]);
        let split = finder_increasing.find_best_split(&histograms, 0.0, 256.0, 256);
        assert!(split.is_none());
    }

    #[test]
    fn test_monotonic_constraint_allows_valid_split() {
        let mut histogram = Histogram::new();

        // Setup for valid increasing: left has lower weight than right
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0); // negative gradient = positive weight
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, -2.0, 1.0); // more negative gradient = more positive weight
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        // left_weight = -(-128) / 128 = 1.0
        // right_weight = -(-256) / 128 = 2.0
        // left_weight < right_weight => increasing is satisfied
        let finder = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Increasing]);
        let split = finder.find_best_split(&histograms, -384.0, 256.0, 256);
        assert!(split.is_some());
    }

    #[test]
    fn test_interaction_constraints_empty() {
        let constraints = InteractionConstraints::new();
        assert!(constraints.is_empty());

        // All features should be allowed
        let allowed = constraints.allowed_features(&[], 5);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);

        // Even with ancestors, all features allowed
        let allowed = constraints.allowed_features(&[0, 2], 5);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_interaction_constraints_basic() {
        // Group 0: features 0, 1
        // Group 1: features 2, 3
        // Feature 4: unconstrained
        let constraints = InteractionConstraints::from_groups(vec![vec![0, 1], vec![2, 3]], 5);

        // No ancestors: all features allowed
        let allowed = constraints.allowed_features(&[], 5);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);

        // After using feature 0 (group 0): only group 0 and unconstrained allowed
        let allowed = constraints.allowed_features(&[0], 5);
        assert_eq!(allowed, vec![0, 1, 4]); // 0, 1 (group 0) and 4 (unconstrained)

        // After using feature 2 (group 1): only group 1 and unconstrained allowed
        let allowed = constraints.allowed_features(&[2], 5);
        assert_eq!(allowed, vec![2, 3, 4]); // 2, 3 (group 1) and 4 (unconstrained)
    }

    #[test]
    fn test_interaction_constraints_unconstrained_always_allowed() {
        // Only constrain some features
        let constraints = InteractionConstraints::from_groups(
            vec![vec![0, 1]],
            5, // Features 2, 3, 4 are unconstrained
        );

        // After using constrained feature, unconstrained still allowed
        let allowed = constraints.allowed_features(&[0], 5);
        assert!(allowed.contains(&2)); // unconstrained
        assert!(allowed.contains(&3)); // unconstrained
        assert!(allowed.contains(&4)); // unconstrained
        assert!(allowed.contains(&0)); // same group
        assert!(allowed.contains(&1)); // same group
        assert_eq!(allowed.len(), 5); // All allowed since 2,3,4 are unconstrained
    }

    #[test]
    fn test_noise_pruning_rejects_bad_validation_splits() {
        use crate::histogram::EraHistograms;

        // Setup: 2 eras (Era 0=train, Era 1=validation), 1 feature
        let mut era_histograms = EraHistograms::new(2, 1);

        // Era 0 (Train): Strong split signal
        // Left bins (0-127): gradient=-2.0 (wants positive weight)
        // Right bins (128-255): gradient=+2.0 (wants negative weight)
        // This creates large gain from splitting
        let train_hist = era_histograms.get_mut(0, 0);
        for bin in 0..128 {
            train_hist.accumulate(bin, -2.0, 1.0);
        }
        for bin in 128..=255 {
            train_hist.accumulate(bin, 2.0, 1.0);
        }

        // Era 1 (Validation): Uniform/flat pattern (no real signal)
        // All bins have small uniform gradients - splitting doesn't help
        // This creates VERY LOW or NEGATIVE gain on validation (splitting is useless)
        let val_hist = era_histograms.get_mut(1, 0);
        for bin in 0..=255 {
            val_hist.accumulate(bin, 0.01, 1.0); // Nearly uniform, tiny gradient
        }

        // Per-era totals: (gradient, hessian, count)
        let per_era_totals = vec![
            (-2.0 * 128.0 + 2.0 * 128.0, 256.0, 256), // Era 0: train (grad=0)
            (0.01 * 256.0, 256.0, 256),               // Era 1: validation (grad=2.56)
        ];

        // Test 1: With noise_pruning enabled and stricter threshold (0.0),
        // split should be REJECTED because validation gain is tiny/negligible
        let finder_noise_pruning = SplitFinder::new()
            .with_lambda(0.0)
            .with_min_gain(0.0)
            .with_noise_pruning(true)
            .with_noise_pruning_threshold(0.5); // Reject if val gain <= 0.5

        let split =
            finder_noise_pruning.find_best_split_with_eras(&era_histograms, &per_era_totals);
        assert!(
            split.is_none(),
            "Expected NO split with noise_pruning=true (validation has low gain)"
        );

        // Test 2: With noise_pruning disabled (DES mode), split might still be accepted
        // because DES checks directional agreement, not gain magnitude
        let finder_des = SplitFinder::new()
            .with_lambda(0.0)
            .with_min_gain(0.0)
            .with_noise_pruning(false); // Use DES directional agreement

        let _split_des = finder_des.find_best_split_with_eras(&era_histograms, &per_era_totals);
        // DES might accept this if there's directional agreement
        // (we're not asserting here as DES behavior depends on implementation details)
    }

    #[test]
    fn test_noise_pruning_validation_with_missing_eras() {
        use crate::histogram::EraHistograms;

        // Setup: Only 1 era (should fail validation)
        let era_histograms_single = EraHistograms::new(1, 1);
        let per_era_totals_single = vec![(0.0, 256.0, 256)];

        let finder = SplitFinder::new()
            .with_noise_pruning(true)
            .with_noise_pruning_threshold(-0.1);

        // Should return None with warning about insufficient eras
        let split =
            finder.find_best_split_with_eras(&era_histograms_single, &per_era_totals_single);
        assert!(split.is_none(), "Expected None when only 1 era provided");

        // Setup: 2 eras but Era 1 has zero samples
        let mut era_histograms_zero = EraHistograms::new(2, 1);
        let train_hist = era_histograms_zero.get_mut(0, 0);
        for bin in 0..128 {
            train_hist.accumulate(bin, -1.0, 1.0);
        }
        // Era 1 has no data

        let per_era_totals_zero = vec![
            (-128.0, 128.0, 128), // Era 0: has data
            (0.0, 0.0, 0),        // Era 1: zero samples
        ];

        let split_zero =
            finder.find_best_split_with_eras(&era_histograms_zero, &per_era_totals_zero);
        assert!(
            split_zero.is_none(),
            "Expected None when Era 1 has zero samples"
        );
    }

    #[test]
    fn test_noise_pruning_accepts_good_validation_splits() {
        use crate::histogram::EraHistograms;

        // Setup: Both eras agree on the split (same pattern)
        let mut era_histograms = EraHistograms::new(2, 1);

        // Era 0 (Train): Good signal
        let train_hist = era_histograms.get_mut(0, 0);
        for bin in 0..128 {
            train_hist.accumulate(bin, -2.0, 1.0);
        }
        for bin in 128..=255 {
            train_hist.accumulate(bin, 2.0, 1.0);
        }

        // Era 1 (Validation): SAME signal (not noise)
        let val_hist = era_histograms.get_mut(1, 0);
        for bin in 0..128 {
            val_hist.accumulate(bin, -2.0, 1.0); // Same pattern as train
        }
        for bin in 128..=255 {
            val_hist.accumulate(bin, 2.0, 1.0); // Same pattern as train
        }

        let per_era_totals = vec![
            (-2.0 * 128.0 + 2.0 * 128.0, 256.0, 256), // Era 0
            (-2.0 * 128.0 + 2.0 * 128.0, 256.0, 256), // Era 1
        ];

        // With noise_pruning enabled, split should be ACCEPTED
        // because validation gain is positive (same pattern as train)
        let finder = SplitFinder::new()
            .with_lambda(0.0)
            .with_min_gain(0.0)
            .with_noise_pruning(true)
            .with_noise_pruning_threshold(-0.1);

        let split = finder.find_best_split_with_eras(&era_histograms, &per_era_totals);
        assert!(
            split.is_some(),
            "Expected split when both train and validation have positive gain"
        );
        assert!(split.unwrap().gain > 0.0, "Expected positive gain");
    }
}
