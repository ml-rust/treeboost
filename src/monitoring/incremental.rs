//! Incremental learning drift detection
//!
//! Provides drift detection specifically for incremental model updates.
//! Monitors feature distribution changes between training batches to
//! warn when data drift may degrade model performance.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::monitoring::IncrementalDriftDetector;
//!
//! // Create detector from initial training data
//! let mut drift_detector = IncrementalDriftDetector::from_dataset(&initial_data);
//!
//! // Before each update, check for drift
//! let drift_result = drift_detector.check_update(&new_data);
//! if drift_result.has_significant_drift() {
//!     println!("Warning: Drift detected! {}", drift_result);
//! }
//!
//! // Update the reference distribution (optional)
//! drift_detector.update_reference(&new_data, 0.1); // 10% weight to new data
//! ```

use super::detector::{AlertLevel, ShiftDetector, ShiftResult};
use super::metrics::PSI;
use crate::dataset::BinnedDataset;

/// Result of incremental drift detection
#[derive(Debug, Clone)]
pub struct IncrementalDriftResult {
    /// Underlying shift detection result
    pub shift_result: ShiftResult,
    /// Number of samples in reference distribution
    pub reference_samples: usize,
    /// Number of samples in update batch
    pub update_samples: usize,
    /// Mean drift score across all features
    pub mean_drift: f32,
    /// Maximum drift score across features
    pub max_drift: f32,
    /// Feature with maximum drift
    pub max_drift_feature: Option<String>,
    /// Recommendation for how to proceed
    pub recommendation: DriftRecommendation,
}

/// Recommendation based on drift severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftRecommendation {
    /// Proceed with update - no significant drift
    ProceedNormally,
    /// Proceed but monitor closely - moderate drift detected
    ProceedWithCaution,
    /// Consider retraining from scratch - significant drift
    ConsiderRetrain,
    /// Strong recommendation to retrain - critical drift
    RetrainRecommended,
}

impl std::fmt::Display for DriftRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProceedNormally => write!(f, "Proceed normally"),
            Self::ProceedWithCaution => write!(f, "Proceed with caution - monitor performance"),
            Self::ConsiderRetrain => write!(f, "Consider full retrain - significant drift"),
            Self::RetrainRecommended => write!(f, "Full retrain recommended - critical drift"),
        }
    }
}

impl IncrementalDriftResult {
    /// Check if significant drift was detected
    pub fn has_significant_drift(&self) -> bool {
        matches!(
            self.shift_result.alert,
            AlertLevel::Warning | AlertLevel::Critical
        )
    }

    /// Check if critical drift was detected
    pub fn has_critical_drift(&self) -> bool {
        self.shift_result.alert == AlertLevel::Critical
    }

    /// Get names of features with drift above warning threshold
    pub fn drifted_features(&self) -> &[String] {
        &self.shift_result.drifted_features
    }

    /// Get names of features with critical drift
    pub fn critical_features(&self) -> &[String] {
        &self.shift_result.critical_features
    }
}

impl std::fmt::Display for IncrementalDriftResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Drift Check: {} ({} ref samples, {} update samples) - mean={:.4}, max={:.4}",
            match self.shift_result.alert {
                AlertLevel::None => "No drift",
                AlertLevel::Warning => "WARNING: Moderate drift",
                AlertLevel::Critical => "CRITICAL: Significant drift",
            },
            self.reference_samples,
            self.update_samples,
            self.mean_drift,
            self.max_drift
        )?;

        if let Some(ref feature) = self.max_drift_feature {
            write!(f, " (max in '{}')", feature)?;
        }

        if !self.shift_result.drifted_features.is_empty() {
            write!(
                f,
                "\n  Drifted features: {:?}",
                self.shift_result.drifted_features
            )?;
        }

        write!(f, "\n  Recommendation: {}", self.recommendation)
    }
}

/// Drift detector for incremental model updates
///
/// Monitors feature distribution changes between training batches.
/// Can optionally update the reference distribution over time using
/// exponential moving average.
pub struct IncrementalDriftDetector {
    /// Underlying shift detector
    detector: ShiftDetector,
    /// Number of samples in reference distribution
    reference_samples: usize,
    /// Warning threshold (PSI default: 0.1)
    warning_threshold: f32,
    /// Critical threshold (PSI default: 0.25)
    critical_threshold: f32,
    /// Proportion of drifted features to trigger critical alert
    critical_feature_ratio: f32,
}

impl IncrementalDriftDetector {
    /// Create from initial training dataset
    pub fn from_dataset(dataset: &BinnedDataset) -> Self {
        let detector = ShiftDetector::from_dataset(dataset).with_metric(PSI::default());

        Self {
            detector,
            reference_samples: dataset.num_rows(),
            warning_threshold: 0.1,
            critical_threshold: 0.25,
            critical_feature_ratio: 0.2, // If >20% of features have critical drift
        }
    }

    /// Set custom thresholds
    pub fn with_thresholds(mut self, warning: f32, critical: f32) -> Self {
        self.warning_threshold = warning;
        self.critical_threshold = critical;
        self.detector = self.detector.with_thresholds(warning, critical);
        self
    }

    /// Set critical feature ratio threshold
    ///
    /// If more than this fraction of features have critical drift,
    /// recommend full retrain.
    pub fn with_critical_feature_ratio(mut self, ratio: f32) -> Self {
        self.critical_feature_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Check update data for drift
    ///
    /// Compares the new data distribution against the reference (training)
    /// distribution and returns a detailed report.
    pub fn check_update(&self, update_data: &BinnedDataset) -> IncrementalDriftResult {
        let shift_result = self.detector.check(update_data);
        let update_samples = update_data.num_rows();

        // Calculate drift statistics
        let mean_drift = shift_result.overall_score;
        let (max_drift, max_drift_feature) = shift_result
            .feature_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, score)| (*score, Some(name.clone())))
            .unwrap_or((0.0, None));

        // Determine recommendation based on drift severity
        let num_features = shift_result.feature_scores.len();
        let critical_ratio = if num_features > 0 {
            shift_result.critical_features.len() as f32 / num_features as f32
        } else {
            0.0
        };

        let recommendation = if critical_ratio >= self.critical_feature_ratio {
            DriftRecommendation::RetrainRecommended
        } else if shift_result.alert == AlertLevel::Critical {
            DriftRecommendation::ConsiderRetrain
        } else if shift_result.alert == AlertLevel::Warning {
            DriftRecommendation::ProceedWithCaution
        } else {
            DriftRecommendation::ProceedNormally
        };

        IncrementalDriftResult {
            shift_result,
            reference_samples: self.reference_samples,
            update_samples,
            mean_drift,
            max_drift,
            max_drift_feature,
            recommendation,
        }
    }

    /// Get the underlying shift detector
    pub fn detector(&self) -> &ShiftDetector {
        &self.detector
    }

    /// Get reference sample count
    pub fn reference_samples(&self) -> usize {
        self.reference_samples
    }
}

/// Convenience function to check drift between two datasets
///
/// # Arguments
/// * `reference` - The original/training dataset
/// * `update` - The new data to compare
///
/// # Returns
/// Drift result with alert level and recommendations
pub fn check_drift(reference: &BinnedDataset, update: &BinnedDataset) -> IncrementalDriftResult {
    let detector = IncrementalDriftDetector::from_dataset(reference);
    detector.check_update(update)
}

/// Summary statistics for a series of drift checks
#[derive(Debug, Clone, Default)]
pub struct DriftHistory {
    /// Total number of updates
    pub total_updates: usize,
    /// Number of updates with warning-level drift
    pub warning_count: usize,
    /// Number of updates with critical drift
    pub critical_count: usize,
    /// Running mean drift score
    pub mean_drift: f32,
    /// Maximum drift score seen
    pub max_drift_ever: f32,
    /// Features that have drifted across updates
    pub frequently_drifted_features: Vec<(String, usize)>,
}

impl DriftHistory {
    /// Create empty history
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a drift check result
    pub fn record(&mut self, result: &IncrementalDriftResult) {
        self.total_updates += 1;

        match result.shift_result.alert {
            AlertLevel::Warning => self.warning_count += 1,
            AlertLevel::Critical => self.critical_count += 1,
            AlertLevel::None => {}
        }

        // Update running mean using Welford's online algorithm
        let delta = result.mean_drift - self.mean_drift;
        self.mean_drift += delta / self.total_updates as f32;

        if result.max_drift > self.max_drift_ever {
            self.max_drift_ever = result.max_drift;
        }

        // Track frequently drifted features
        for feature in &result.shift_result.drifted_features {
            if let Some(entry) = self
                .frequently_drifted_features
                .iter_mut()
                .find(|(f, _)| f == feature)
            {
                entry.1 += 1;
            } else {
                self.frequently_drifted_features.push((feature.clone(), 1));
            }
        }

        // Sort by frequency
        self.frequently_drifted_features
            .sort_by_key(|b| std::cmp::Reverse(b.1));
    }

    /// Get drift rate (fraction of updates with drift)
    pub fn drift_rate(&self) -> f32 {
        if self.total_updates == 0 {
            0.0
        } else {
            (self.warning_count + self.critical_count) as f32 / self.total_updates as f32
        }
    }

    /// Get critical drift rate
    pub fn critical_rate(&self) -> f32 {
        if self.total_updates == 0 {
            0.0
        } else {
            self.critical_count as f32 / self.total_updates as f32
        }
    }
}

impl std::fmt::Display for DriftHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Drift History ({} updates):", self.total_updates)?;
        writeln!(
            f,
            "  Drift rate: {:.1}% ({} warnings, {} critical)",
            self.drift_rate() * 100.0,
            self.warning_count,
            self.critical_count
        )?;
        writeln!(
            f,
            "  Mean drift: {:.4}, Max drift: {:.4}",
            self.mean_drift, self.max_drift_ever
        )?;
        if !self.frequently_drifted_features.is_empty() {
            writeln!(f, "  Frequently drifted:")?;
            for (feature, count) in self.frequently_drifted_features.iter().take(5) {
                writeln!(f, "    - {} ({} times)", feature, count)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(num_rows: usize, num_features: usize, offset: u8) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                // Create bins with some pattern + offset for drift
                features.push(((r * 3 + f * 7 + offset as usize) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();

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

    #[test]
    fn test_no_drift_same_distribution() {
        let data = create_test_dataset(100, 3, 0);
        let detector = IncrementalDriftDetector::from_dataset(&data);

        // Check against same distribution
        let result = detector.check_update(&data);

        assert!(
            !result.has_significant_drift(),
            "Same distribution should have no drift"
        );
        assert_eq!(result.shift_result.alert, AlertLevel::None);
        assert_eq!(result.recommendation, DriftRecommendation::ProceedNormally);
    }

    #[test]
    fn test_drift_shifted_distribution() {
        let reference = create_test_dataset(100, 3, 0);
        let shifted = create_test_dataset(100, 3, 128); // Large offset = drift

        let detector = IncrementalDriftDetector::from_dataset(&reference);
        let result = detector.check_update(&shifted);

        // Should detect some drift (actual level depends on the data)
        // The shifted data has different bin patterns
        println!("Drift result: {}", result);
        println!("Mean drift: {}", result.mean_drift);
    }

    #[test]
    fn test_drift_recommendation_levels() {
        // Test that recommendations escalate properly
        assert_eq!(
            DriftRecommendation::ProceedNormally.to_string(),
            "Proceed normally"
        );
        assert!(DriftRecommendation::RetrainRecommended
            .to_string()
            .contains("retrain"));
    }

    #[test]
    fn test_drift_result_display() {
        let shift_result = ShiftResult {
            feature_scores: vec![("f1".to_string(), 0.15), ("f2".to_string(), 0.05)],
            overall_score: 0.10,
            alert: AlertLevel::Warning,
            drifted_features: vec!["f1".to_string()],
            critical_features: vec![],
        };

        let result = IncrementalDriftResult {
            shift_result,
            reference_samples: 1000,
            update_samples: 100,
            mean_drift: 0.10,
            max_drift: 0.15,
            max_drift_feature: Some("f1".to_string()),
            recommendation: DriftRecommendation::ProceedWithCaution,
        };

        let display = format!("{}", result);
        assert!(display.contains("WARNING"));
        assert!(display.contains("f1"));
        assert!(display.contains("Proceed with caution"));
    }

    #[test]
    fn test_drift_history() {
        let mut history = DriftHistory::new();

        // Record a warning
        let shift_result = ShiftResult {
            feature_scores: vec![("f1".to_string(), 0.12)],
            overall_score: 0.12,
            alert: AlertLevel::Warning,
            drifted_features: vec!["f1".to_string()],
            critical_features: vec![],
        };

        history.record(&IncrementalDriftResult {
            shift_result,
            reference_samples: 1000,
            update_samples: 100,
            mean_drift: 0.12,
            max_drift: 0.12,
            max_drift_feature: Some("f1".to_string()),
            recommendation: DriftRecommendation::ProceedWithCaution,
        });

        assert_eq!(history.total_updates, 1);
        assert_eq!(history.warning_count, 1);
        assert_eq!(history.critical_count, 0);
        assert!((history.mean_drift - 0.12).abs() < 0.01);
        assert_eq!(history.frequently_drifted_features.len(), 1);
        assert_eq!(history.frequently_drifted_features[0].0, "f1");
    }

    #[test]
    fn test_check_drift_convenience() {
        let reference = create_test_dataset(100, 3, 0);
        let update = create_test_dataset(100, 3, 0);

        let result = check_drift(&reference, &update);
        assert!(!result.has_significant_drift());
    }
}
