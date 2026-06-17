//! Distribution shift detector for inference-time drift detection

use super::metrics::{DistributionMetric, PSI};
use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};

/// Alert level for distribution shift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    /// No significant shift detected
    None,
    /// Moderate shift - investigate
    Warning,
    /// Significant shift - action required
    Critical,
}

/// Result of a distribution shift check
#[derive(Debug, Clone)]
pub struct ShiftResult {
    /// Per-feature drift scores (feature_name, score)
    pub feature_scores: Vec<(String, f32)>,
    /// Overall drift score (aggregated)
    pub overall_score: f32,
    /// Alert level based on thresholds
    pub alert: AlertLevel,
    /// Features flagged as drifted (above warning threshold)
    pub drifted_features: Vec<String>,
    /// Features with critical drift
    pub critical_features: Vec<String>,
}

impl ShiftResult {
    /// Check if any drift was detected
    pub fn has_drift(&self) -> bool {
        self.alert != AlertLevel::None
    }

    /// Get the number of drifted features
    pub fn n_drifted(&self) -> usize {
        self.drifted_features.len()
    }

    /// Get the number of critically drifted features
    pub fn n_critical(&self) -> usize {
        self.critical_features.len()
    }
}

/// Distribution shift detector
///
/// Compares feature distributions between training data and new inference data.
/// Uses PSI by default, but supports custom metrics.
///
/// Applies sample-size-aware threshold adjustment: when inference samples are smaller
/// than training samples, thresholds are automatically relaxed to account for higher
/// variance in the empirical distribution.
///
/// # Example
///
/// ```ignore
/// let detector = ShiftDetector::from_dataset(&train_data)
///     .with_thresholds(0.1, 0.25);
///
/// let result = detector.check(&inference_data);
/// if result.alert == AlertLevel::Critical {
///     println!("Drift detected in: {:?}", result.critical_features);
/// }
/// ```
/// Maximum number of histogram bins supported (matches u8 range from BinnedDataset)
///
/// Used as an upper bound for validation. Actual allocations use per-feature `num_bins`
/// from FeatureInfo for memory efficiency.
pub const MAX_HISTOGRAM_BINS: usize = 256;

pub struct ShiftDetector {
    /// Reference feature histograms (bin counts per feature)
    ///
    /// **Memory-efficient design**: Each histogram is dynamically sized based on
    /// `feature_info[i].num_bins`, not fixed at 256. This saves significant memory
    /// for datasets with fewer bins (e.g., 32 bins saves 224 × 4 = 896 bytes/feature).
    reference_histograms: Vec<Vec<u32>>,
    /// Actual number of bins per feature (cached from FeatureInfo for fast access)
    bins_per_feature: Vec<usize>,
    /// Reference sample count per feature (for sample-size-aware threshold adjustment)
    reference_counts: Vec<usize>,
    /// Feature information (names and bin boundaries)
    feature_info: Vec<FeatureInfo>,
    /// Distribution metric to use
    metric: Box<dyn DistributionMetric>,
    /// Warning threshold (base value, adjusted per feature based on sample size)
    warning_threshold: f32,
    /// Critical threshold (base value, adjusted per feature based on sample size)
    critical_threshold: f32,
}

impl ShiftDetector {
    /// Create a detector from a training BinnedDataset
    ///
    /// Pre-computes histograms from the training data for efficient
    /// comparison with inference data.
    ///
    /// Uses dynamic histogram sizing based on each feature's actual `num_bins`
    /// to minimize memory usage (vs fixed 256-bin arrays).
    pub fn from_dataset(dataset: &BinnedDataset) -> Self {
        let num_features = dataset.num_features();
        let num_rows = dataset.num_rows();
        let feature_info = dataset.all_feature_info().to_vec();

        // Extract actual bin counts per feature (dynamic sizing)
        let bins_per_feature: Vec<usize> = feature_info
            .iter()
            .map(|info| {
                // Use num_bins from FeatureInfo, defaulting to MAX_HISTOGRAM_BINS if 0
                // (0 indicates unknown/unbinned data)
                let bins = info.num_bins as usize;
                if bins == 0 {
                    MAX_HISTOGRAM_BINS
                } else {
                    bins
                }
            })
            .collect();

        // Allocate histograms with exact size needed per feature
        let mut reference_histograms: Vec<Vec<u32>> = bins_per_feature
            .iter()
            .map(|&bins| vec![0u32; bins])
            .collect();

        // Build histograms
        #[allow(clippy::needless_range_loop)] // Direct indexing is optimal for histogram building
        for row in 0..num_rows {
            for feature in 0..num_features {
                let bin = dataset.get_bin(row, feature) as usize;
                // Safety: bin values are guaranteed to be < num_bins by BinnedDataset
                if bin < reference_histograms[feature].len() {
                    reference_histograms[feature][bin] += 1;
                }
            }
        }

        let reference_counts = vec![num_rows; num_features];

        let metric = Box::new(PSI::default());
        let warning_threshold = metric.warning_threshold();
        let critical_threshold = metric.critical_threshold();

        Self {
            reference_histograms,
            bins_per_feature,
            reference_counts,
            feature_info,
            metric,
            warning_threshold,
            critical_threshold,
        }
    }

    /// Create from raw feature data
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix
    /// * `num_features` - Number of features
    /// * `feature_names` - Optional feature names
    ///
    /// Note: Raw data uses MAX_HISTOGRAM_BINS (256) since actual bin count is unknown.
    /// For memory efficiency with binned data, use `from_dataset` instead.
    pub fn from_raw(
        features: &[f32],
        num_features: usize,
        feature_names: Option<&[String]>,
    ) -> Self {
        let num_rows = features.len().checked_div(num_features).unwrap_or(0);

        // For raw data without binning info, use MAX_HISTOGRAM_BINS as default
        let bins_per_feature = vec![MAX_HISTOGRAM_BINS; num_features];

        // Store raw values for each feature (simplified - no binning)
        // For raw data, we'll use empty histograms and store feature stats
        let reference_histograms = vec![vec![0u32; MAX_HISTOGRAM_BINS]; num_features];
        let reference_counts = vec![num_rows; num_features];

        let feature_info: Vec<FeatureInfo> = (0..num_features)
            .map(|i| {
                let name = feature_names
                    .and_then(|names| names.get(i).cloned())
                    .unwrap_or_else(|| format!("feature_{}", i));
                FeatureInfo {
                    name,
                    feature_type: FeatureType::Numeric,
                    num_bins: 0,
                    bin_boundaries: vec![],
                    impute_value: 0.0,
                }
            })
            .collect();

        let metric = Box::new(PSI::default());
        let warning_threshold = metric.warning_threshold();
        let critical_threshold = metric.critical_threshold();

        Self {
            reference_histograms,
            bins_per_feature,
            reference_counts,
            feature_info,
            metric,
            warning_threshold,
            critical_threshold,
        }
    }

    /// Set custom thresholds
    pub fn with_thresholds(mut self, warning: f32, critical: f32) -> Self {
        self.warning_threshold = warning;
        self.critical_threshold = critical;
        self
    }

    /// Set a custom distribution metric
    pub fn with_metric<M: DistributionMetric + 'static>(mut self, metric: M) -> Self {
        self.warning_threshold = metric.warning_threshold();
        self.critical_threshold = metric.critical_threshold();
        self.metric = Box::new(metric);
        self
    }

    /// Compute sample-size-aware threshold adjustment factor
    ///
    /// Returns a multiplier for thresholds based on the ratio of inference to reference samples.
    /// Smaller inference samples get more lenient (higher) thresholds to account for higher variance.
    ///
    /// Uses the formula: threshold_multiplier = sqrt((n_ref + n_inf) / n_inf)
    /// - When n_inf << n_ref: multiplier → ∞ (very lenient for small samples)
    /// - When n_inf = n_ref: multiplier ≈ 1.41 (41% more lenient)
    /// - When n_inf >> n_ref: multiplier → 1.0 (strict for large samples)
    ///
    /// Rationale: Smaller samples have higher sampling variance, so we need looser
    /// thresholds to avoid false positives from noise.
    fn compute_threshold_multiplier(&self, feature_idx: usize, inference_count: usize) -> f32 {
        debug_assert!(
            feature_idx < self.reference_counts.len(),
            "Feature index {} out of bounds (max: {})",
            feature_idx,
            self.reference_counts.len()
        );

        let ref_count = self.reference_counts[feature_idx];
        if ref_count == 0 || inference_count == 0 {
            return 1.0; // No adjustment for empty samples
        }

        // Sample-size-aware adjustment: smaller inference samples get larger multipliers (more lenient)
        ((ref_count + inference_count) as f32 / inference_count as f32).sqrt()
    }

    /// Check inference data for distribution shift
    ///
    /// Compares each feature's distribution in the inference data against
    /// the reference (training) distribution. Automatically adjusts thresholds
    /// based on sample size differences.
    ///
    /// Uses dynamic histogram sizing matching reference histograms for memory efficiency.
    pub fn check(&self, inference_data: &BinnedDataset) -> ShiftResult {
        let num_features = self.feature_info.len().min(inference_data.num_features());
        let num_rows = inference_data.num_rows();

        if num_rows == 0 || num_features == 0 {
            return ShiftResult {
                feature_scores: Vec::new(),
                overall_score: 0.0,
                alert: AlertLevel::None,
                drifted_features: Vec::new(),
                critical_features: Vec::new(),
            };
        }

        // Compute inference histograms with dynamic sizing (matching reference)
        let mut inference_histograms: Vec<Vec<u32>> = self
            .bins_per_feature
            .iter()
            .take(num_features)
            .map(|&bins| vec![0u32; bins])
            .collect();

        #[allow(clippy::needless_range_loop)] // Direct indexing is optimal for histogram building
        for row in 0..num_rows {
            for feature in 0..num_features {
                let bin = inference_data.get_bin(row, feature) as usize;
                // Safety: bin values should be within range, but clamp to be safe
                if bin < inference_histograms[feature].len() {
                    inference_histograms[feature][bin] += 1;
                }
            }
        }

        // Compare each feature
        let mut feature_scores = Vec::with_capacity(num_features);
        let mut drifted_features = Vec::new();
        let mut critical_features = Vec::new();
        let mut total_score = 0.0f32;

        for (i, feature_info) in self.feature_info.iter().take(num_features).enumerate() {
            // Normalize histograms to probability distributions
            let ref_total: f32 = self.reference_histograms[i].iter().map(|&c| c as f32).sum();
            let inf_total: f32 = inference_histograms[i].iter().map(|&c| c as f32).sum();

            // Skip features with no data
            if ref_total == 0.0 || inf_total == 0.0 {
                feature_scores.push((feature_info.name.clone(), 0.0));
                continue;
            }

            // Convert to probability distributions (normalized histograms)
            // Only iterate over actual bins, not fixed 256
            let ref_probs: Vec<f32> = self.reference_histograms[i]
                .iter()
                .map(|&c| c as f32 / ref_total)
                .collect();
            let inf_probs: Vec<f32> = inference_histograms[i]
                .iter()
                .map(|&c| c as f32 / inf_total)
                .collect();

            // Compute metric score on probabilities
            let score = self.metric.compute(&ref_probs, &inf_probs);
            total_score += score;

            feature_scores.push((feature_info.name.clone(), score));

            // Apply sample-size-aware threshold adjustment
            let threshold_multiplier = self.compute_threshold_multiplier(i, num_rows);
            let adjusted_warning = self.warning_threshold * threshold_multiplier;
            let adjusted_critical = self.critical_threshold * threshold_multiplier;

            if score >= adjusted_critical {
                drifted_features.push(feature_info.name.clone());
                critical_features.push(feature_info.name.clone());
            } else if score >= adjusted_warning {
                drifted_features.push(feature_info.name.clone());
            }
        }

        // Compute overall score (mean)
        let overall_score = total_score / num_features as f32;

        // Determine alert level (use base thresholds for overall score)
        let alert = if !critical_features.is_empty() || overall_score >= self.critical_threshold {
            AlertLevel::Critical
        } else if !drifted_features.is_empty() || overall_score >= self.warning_threshold {
            AlertLevel::Warning
        } else {
            AlertLevel::None
        };

        ShiftResult {
            feature_scores,
            overall_score,
            alert,
            drifted_features,
            critical_features,
        }
    }

    /// Get reference feature info
    pub fn feature_info(&self) -> &[FeatureInfo] {
        &self.feature_info
    }

    /// Get the warning threshold
    pub fn warning_threshold(&self) -> f32 {
        self.warning_threshold
    }

    /// Get the critical threshold
    pub fn critical_threshold(&self) -> f32 {
        self.critical_threshold
    }

    /// Get the metric name
    pub fn metric_name(&self) -> &'static str {
        self.metric.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_level() {
        assert_eq!(AlertLevel::None, AlertLevel::None);
        assert_ne!(AlertLevel::None, AlertLevel::Warning);
        assert_ne!(AlertLevel::Warning, AlertLevel::Critical);
    }

    #[test]
    fn test_shift_result() {
        let result = ShiftResult {
            feature_scores: vec![("a".to_string(), 0.1), ("b".to_string(), 0.3)],
            overall_score: 0.2,
            alert: AlertLevel::Warning,
            drifted_features: vec!["a".to_string(), "b".to_string()],
            critical_features: vec!["b".to_string()],
        };

        assert!(result.has_drift());
        assert_eq!(result.n_drifted(), 2);
        assert_eq!(result.n_critical(), 1);
    }

    #[test]
    fn test_shift_result_no_drift() {
        let result = ShiftResult {
            feature_scores: vec![("a".to_string(), 0.01)],
            overall_score: 0.01,
            alert: AlertLevel::None,
            drifted_features: vec![],
            critical_features: vec![],
        };

        assert!(!result.has_drift());
        assert_eq!(result.n_drifted(), 0);
    }

    #[test]
    fn test_threshold_multiplier() {
        use crate::dataset::FeatureType;

        // Create a simple detector
        let feature_info = vec![FeatureInfo {
            name: "test".to_string(),
            feature_type: FeatureType::Numeric,
            num_bins: 2,
            bin_boundaries: vec![0.0, 0.5, 1.0],
            impute_value: 0.0,
        }];

        let detector = ShiftDetector {
            reference_histograms: vec![vec![0; 2]], // Dynamic sizing: only 2 bins
            bins_per_feature: vec![2],
            reference_counts: vec![1000], // 1000 training samples
            feature_info,
            metric: Box::new(PSI::default()),
            warning_threshold: 0.1,
            critical_threshold: 0.25,
        };

        // Test: Equal sample sizes (1000 ref, 1000 inf)
        let mult_equal = detector.compute_threshold_multiplier(0, 1000);
        assert!(
            (mult_equal - 1.414).abs() < 0.01,
            "Expected ~1.414, got {}",
            mult_equal
        );

        // Test: Small inference sample (1000 ref, 100 inf) - should be MORE lenient (larger multiplier)
        let mult_small = detector.compute_threshold_multiplier(0, 100);
        assert!(
            mult_small > mult_equal,
            "Small samples should have larger multipliers (more lenient)"
        );
        assert!(
            (mult_small - 3.32).abs() < 0.01,
            "Expected ~3.32, got {}",
            mult_small
        );

        // Test: Large inference sample (1000 ref, 10000 inf) - should be STRICTER (smaller multiplier)
        let mult_large = detector.compute_threshold_multiplier(0, 10000);
        assert!(
            mult_large < mult_equal,
            "Large samples should have smaller multipliers (stricter)"
        );
        assert!(
            (mult_large - 1.049).abs() < 0.01,
            "Expected ~1.049, got {}",
            mult_large
        );

        // Verify: Small samples get MORE lenient thresholds (larger multiplier = higher threshold)
        // This prevents false positives from sampling noise in small batches
        assert!(
            mult_small > mult_large,
            "Smaller inference samples should have LARGER multipliers (more lenient thresholds). Got small={}, large={}",
            mult_small, mult_large
        );

        // Verify ordering: small > equal > large
        assert!(
            mult_small > mult_equal && mult_equal > mult_large,
            "Expected: small ({}) > equal ({}) > large ({})",
            mult_small,
            mult_equal,
            mult_large
        );
    }
}
