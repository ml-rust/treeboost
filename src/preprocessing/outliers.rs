//! Outlier detection and handling
//!
//! Provides methods to detect and handle outliers in feature data.
//! Complements the Pseudo-Huber loss which handles outliers during training.
//!
//! # Detection Methods
//!
//! | Method | Formula | Best For |
//! |--------|---------|----------|
//! | IQR | Outside [Q1 - k×IQR, Q3 + k×IQR] | General use, robust |
//! | Z-score | \|x - μ\| / σ > threshold | Normal distributions |
//!
//! # Actions
//!
//! | Action | Description | Data Shape |
//! |--------|-------------|------------|
//! | Cap | Clip to bounds | Unchanged |
//! | Flag | Add indicator columns | +N features |
//! | Remove | Delete outlier rows | Fewer rows |
//!
//! # Example
//!
//! ```ignore
//! use treeboost::preprocessing::{OutlierDetector, OutlierMethod, OutlierAction};
//!
//! // IQR-based detection with capping
//! let mut detector = OutlierDetector::new(OutlierMethod::Iqr { k: 1.5 })
//!     .with_action(OutlierAction::Cap);
//!
//! detector.fit(&data, num_features)?;
//! let cleaned = detector.transform(&data, num_features, &names)?;
//!
//! // Z-score detection with flagging
//! let mut detector = OutlierDetector::new(OutlierMethod::ZScore { threshold: 3.0 })
//!     .with_action(OutlierAction::Flag);
//! ```

use crate::{Result, TreeBoostError};

/// Outlier detection method
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OutlierMethod {
    /// IQR-based detection: outlier if x < Q1 - k×IQR or x > Q3 + k×IQR
    /// Default k = 1.5 (standard Tukey fence)
    Iqr {
        /// Multiplier for IQR (default: 1.5)
        k: f32,
    },
    /// Z-score detection: outlier if |x - μ| / σ > threshold
    /// Default threshold = 3.0
    ZScore {
        /// Z-score threshold (default: 3.0)
        threshold: f32,
    },
}

impl OutlierMethod {
    /// Standard IQR method with k=1.5 (Tukey fence)
    pub fn iqr() -> Self {
        Self::Iqr { k: 1.5 }
    }

    /// IQR method with custom k value
    pub fn iqr_with_k(k: f32) -> Self {
        Self::Iqr { k }
    }

    /// Standard Z-score method with threshold=3.0
    pub fn zscore() -> Self {
        Self::ZScore { threshold: 3.0 }
    }

    /// Z-score method with custom threshold
    pub fn zscore_with_threshold(threshold: f32) -> Self {
        Self::ZScore { threshold }
    }
}

impl Default for OutlierMethod {
    fn default() -> Self {
        Self::iqr()
    }
}

/// Action to take when outliers are detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OutlierAction {
    /// Cap outliers to the boundary values (winsorization)
    /// Data shape unchanged, extreme values clipped
    #[default]
    Cap,
    /// Add binary indicator columns for each feature
    /// Returns (original_data, indicator_data, indicator_names)
    Flag,
    /// Remove rows containing any outlier
    /// Returns only non-outlier rows
    Remove,
}

/// Per-feature outlier bounds learned during fit
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureBounds {
    /// Lower bound (values below are outliers)
    pub lower: f32,
    /// Upper bound (values above are outliers)
    pub upper: f32,
}

/// Outlier detector with configurable method and action
///
/// Follows the fit-transform pattern:
/// 1. `fit()` learns outlier thresholds from training data
/// 2. `transform()` applies the configured action
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{OutlierDetector, OutlierMethod, OutlierAction};
///
/// let mut data = vec![
///     1.0, 10.0,   // row 0
///     2.0, 20.0,   // row 1
///     3.0, 30.0,   // row 2
///     100.0, 40.0, // row 3 (outlier in feature 0)
/// ];
///
/// let mut detector = OutlierDetector::new(OutlierMethod::iqr())
///     .with_action(OutlierAction::Cap);
///
/// detector.fit(&data, 2)?;
/// let result = detector.transform(&mut data, 2, &["f0".into(), "f1".into()])?;
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutlierDetector {
    /// Detection method
    method: OutlierMethod,
    /// Action to take on outliers
    action: OutlierAction,
    /// Per-feature bounds (learned during fit)
    bounds: Vec<FeatureBounds>,
    /// Whether fit() has been called
    fitted: bool,
}

impl OutlierDetector {
    /// Create a new outlier detector with the specified method
    pub fn new(method: OutlierMethod) -> Self {
        Self {
            method,
            action: OutlierAction::default(),
            bounds: Vec::new(),
            fitted: false,
        }
    }

    /// Set the action to take on outliers
    pub fn with_action(mut self, action: OutlierAction) -> Self {
        self.action = action;
        self
    }

    /// Get the detection method
    pub fn method(&self) -> OutlierMethod {
        self.method
    }

    /// Get the configured action
    pub fn action(&self) -> OutlierAction {
        self.action
    }

    /// Check if the detector has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the learned bounds for each feature
    pub fn bounds(&self) -> &[FeatureBounds] {
        &self.bounds
    }

    /// Fit the detector to learn outlier thresholds
    ///
    /// # Arguments
    /// * `data` - Row-major feature matrix (num_rows × num_features)
    /// * `num_features` - Number of features (columns)
    pub fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "OutlierDetector::fit() requires num_features > 0, got 0".into(),
            ));
        }

        if data.is_empty() {
            return Err(TreeBoostError::Data(
                "OutlierDetector::fit() received empty dataset. Provide data with at least 1 row."
                    .into(),
            ));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows < 4 {
            return Err(TreeBoostError::Data(
                "Need at least 4 rows to compute quartiles".into(),
            ));
        }

        self.bounds = Vec::with_capacity(num_features);

        match self.method {
            OutlierMethod::Iqr { k } => {
                self.fit_iqr(data, num_features, num_rows, k)?;
            }
            OutlierMethod::ZScore { threshold } => {
                self.fit_zscore(data, num_features, num_rows, threshold)?;
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Fit using IQR method
    fn fit_iqr(
        &mut self,
        data: &[f32],
        num_features: usize,
        num_rows: usize,
        k: f32,
    ) -> Result<()> {
        for feat in 0..num_features {
            // Extract and sort feature column
            let mut column: Vec<f32> = (0..num_rows)
                .map(|row| data[row * num_features + feat])
                .filter(|v| v.is_finite())
                .collect();

            if column.is_empty() {
                // All NaN - use infinite bounds (no outliers)
                self.bounds.push(FeatureBounds {
                    lower: f32::NEG_INFINITY,
                    upper: f32::INFINITY,
                });
                continue;
            }

            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Q1 (25th percentile)
            let q1 = percentile(&column, 0.25);

            // Q3 (75th percentile)
            let q3 = percentile(&column, 0.75);

            // IQR = Q3 - Q1
            let iqr = q3 - q1;

            // Bounds
            let lower = q1 - k * iqr;
            let upper = q3 + k * iqr;

            self.bounds.push(FeatureBounds { lower, upper });
        }

        Ok(())
    }

    /// Fit using Z-score method
    fn fit_zscore(
        &mut self,
        data: &[f32],
        num_features: usize,
        num_rows: usize,
        threshold: f32,
    ) -> Result<()> {
        for feat in 0..num_features {
            // Extract feature column (excluding NaN)
            let column: Vec<f32> = (0..num_rows)
                .map(|row| data[row * num_features + feat])
                .filter(|v| v.is_finite())
                .collect();

            if column.is_empty() {
                // All NaN - use infinite bounds
                self.bounds.push(FeatureBounds {
                    lower: f32::NEG_INFINITY,
                    upper: f32::INFINITY,
                });
                continue;
            }

            let n = column.len() as f32;

            // Mean
            let mean: f32 = column.iter().sum::<f32>() / n;

            // Standard deviation
            let variance: f32 = column.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let std = variance.sqrt().max(1e-10);

            // Bounds: mean ± threshold * std
            let lower = mean - threshold * std;
            let upper = mean + threshold * std;

            self.bounds.push(FeatureBounds { lower, upper });
        }

        Ok(())
    }

    /// Check if a value is an outlier for a given feature
    pub fn is_outlier(&self, value: f32, feature_idx: usize) -> bool {
        if !self.fitted || feature_idx >= self.bounds.len() {
            return false;
        }

        let bounds = &self.bounds[feature_idx];
        value < bounds.lower || value > bounds.upper
    }

    /// Detect outliers and return their indices
    ///
    /// Returns a vector of (row_idx, feature_idx) pairs for each outlier.
    pub fn detect(&self, data: &[f32], num_features: usize) -> Result<Vec<(usize, usize)>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "OutlierDetector not fitted. Call fit() first.".into(),
            ));
        }

        if num_features != self.bounds.len() {
            return Err(TreeBoostError::Data(format!(
                "num_features mismatch: fit with {}, detect with {}",
                self.bounds.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;
        let mut outliers = Vec::new();

        for row in 0..num_rows {
            for feat in 0..num_features {
                let value = data[row * num_features + feat];
                if value.is_finite() && self.is_outlier(value, feat) {
                    outliers.push((row, feat));
                }
            }
        }

        Ok(outliers)
    }

    /// Transform data according to the configured action
    ///
    /// # Returns
    /// `TransformResult` containing the transformed data and any additional outputs
    pub fn transform(
        &self,
        data: &mut [f32],
        num_features: usize,
        feature_names: &[String],
    ) -> Result<TransformResult> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "OutlierDetector not fitted. Call fit() first.".into(),
            ));
        }

        if num_features != self.bounds.len() {
            return Err(TreeBoostError::Data(format!(
                "num_features mismatch: fit with {}, transform with {}",
                self.bounds.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        match self.action {
            OutlierAction::Cap => self.transform_cap(data, num_features, num_rows),
            OutlierAction::Flag => self.transform_flag(data, num_features, num_rows, feature_names),
            OutlierAction::Remove => {
                self.transform_remove(data, num_features, num_rows, feature_names)
            }
        }
    }

    /// Cap outliers to boundary values (winsorization)
    fn transform_cap(
        &self,
        data: &mut [f32],
        num_features: usize,
        num_rows: usize,
    ) -> Result<TransformResult> {
        let mut outlier_count = 0;

        for row in 0..num_rows {
            for feat in 0..num_features {
                let idx = row * num_features + feat;
                let value = data[idx];

                if !value.is_finite() {
                    continue;
                }

                let bounds = &self.bounds[feat];
                if value < bounds.lower {
                    data[idx] = bounds.lower;
                    outlier_count += 1;
                } else if value > bounds.upper {
                    data[idx] = bounds.upper;
                    outlier_count += 1;
                }
            }
        }

        Ok(TransformResult::Capped { outlier_count })
    }

    /// Add indicator columns for outliers
    fn transform_flag(
        &self,
        data: &[f32],
        num_features: usize,
        num_rows: usize,
        feature_names: &[String],
    ) -> Result<TransformResult> {
        let mut indicators = vec![0.0f32; num_rows * num_features];
        let mut indicator_names = Vec::with_capacity(num_features);

        for feat in 0..num_features {
            let name = feature_names
                .get(feat)
                .cloned()
                .unwrap_or_else(|| format!("f{}", feat));
            indicator_names.push(format!("{}_outlier", name));

            let bounds = &self.bounds[feat];

            for row in 0..num_rows {
                let value = data[row * num_features + feat];
                if value.is_finite() && (value < bounds.lower || value > bounds.upper) {
                    indicators[row * num_features + feat] = 1.0;
                }
            }
        }

        Ok(TransformResult::Flagged {
            indicators,
            indicator_names,
        })
    }

    /// Remove rows containing outliers
    fn transform_remove(
        &self,
        data: &[f32],
        num_features: usize,
        num_rows: usize,
        _feature_names: &[String],
    ) -> Result<TransformResult> {
        // Find rows with outliers
        let mut outlier_rows = vec![false; num_rows];

        for row in 0..num_rows {
            for feat in 0..num_features {
                let value = data[row * num_features + feat];
                if value.is_finite() && self.is_outlier(value, feat) {
                    outlier_rows[row] = true;
                    break;
                }
            }
        }

        // Collect non-outlier rows
        let kept_indices: Vec<usize> = (0..num_rows).filter(|&row| !outlier_rows[row]).collect();

        let mut cleaned_data = Vec::with_capacity(kept_indices.len() * num_features);

        for &row in &kept_indices {
            for feat in 0..num_features {
                cleaned_data.push(data[row * num_features + feat]);
            }
        }

        let removed_count = num_rows - kept_indices.len();

        Ok(TransformResult::Removed {
            cleaned_data,
            kept_indices,
            removed_count,
        })
    }

    /// Count outliers per feature
    pub fn outlier_counts(&self, data: &[f32], num_features: usize) -> Result<Vec<usize>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "OutlierDetector not fitted. Call fit() first.".into(),
            ));
        }

        let num_rows = data.len() / num_features;
        let mut counts = vec![0usize; num_features];

        for row in 0..num_rows {
            for feat in 0..num_features {
                let value = data[row * num_features + feat];
                if value.is_finite() && self.is_outlier(value, feat) {
                    counts[feat] += 1;
                }
            }
        }

        Ok(counts)
    }
}

impl Default for OutlierDetector {
    fn default() -> Self {
        Self::new(OutlierMethod::default())
    }
}

/// Result of outlier transformation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TransformResult {
    /// Values were capped to boundaries
    Capped {
        /// Number of values that were capped
        outlier_count: usize,
    },
    /// Indicator columns were created
    Flagged {
        /// Binary indicator data (num_rows × num_features)
        indicators: Vec<f32>,
        /// Names for indicator columns
        indicator_names: Vec<String>,
    },
    /// Outlier rows were removed
    Removed {
        /// Cleaned data without outlier rows
        cleaned_data: Vec<f32>,
        /// Indices of kept rows (from original data)
        kept_indices: Vec<usize>,
        /// Number of rows removed
        removed_count: usize,
    },
}

impl TransformResult {
    /// Get the number of outliers handled
    pub fn outlier_count(&self) -> usize {
        match self {
            Self::Capped { outlier_count } => *outlier_count,
            Self::Flagged { indicators, .. } => indicators.iter().filter(|&&v| v > 0.0).count(),
            Self::Removed { removed_count, .. } => *removed_count,
        }
    }
}

/// Compute percentile using linear interpolation
fn percentile(sorted_data: &[f32], p: f32) -> f32 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let n = sorted_data.len();
    if n == 1 {
        return sorted_data[0];
    }

    // Linear interpolation method
    let idx = p * (n - 1) as f32;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f32;

    if upper >= n {
        sorted_data[n - 1]
    } else if lower == upper {
        sorted_data[lower]
    } else {
        sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // OutlierMethod Tests
    // ========================================

    #[test]
    fn test_outlier_method_defaults() {
        let iqr = OutlierMethod::iqr();
        assert!(matches!(iqr, OutlierMethod::Iqr { k } if (k - 1.5).abs() < 1e-6));

        let zscore = OutlierMethod::zscore();
        assert!(
            matches!(zscore, OutlierMethod::ZScore { threshold } if (threshold - 3.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_outlier_method_custom() {
        let iqr = OutlierMethod::iqr_with_k(2.0);
        assert!(matches!(iqr, OutlierMethod::Iqr { k } if (k - 2.0).abs() < 1e-6));

        let zscore = OutlierMethod::zscore_with_threshold(2.5);
        assert!(
            matches!(zscore, OutlierMethod::ZScore { threshold } if (threshold - 2.5).abs() < 1e-6)
        );
    }

    // ========================================
    // IQR Detection Tests
    // ========================================

    #[test]
    fn test_iqr_detection_basic() {
        // Data with clear outlier
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,  // normal values
            100.0, // outlier
        ];

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 1).unwrap();

        assert!(detector.is_outlier(100.0, 0));
        assert!(!detector.is_outlier(5.0, 0));
    }

    #[test]
    fn test_iqr_bounds_computation() {
        // Simple dataset: 1, 2, 3, 4, 5, 6, 7, 8
        // Using linear interpolation:
        // Q1 (25th): idx = 0.25 * 7 = 1.75 → lerp(2, 3, 0.75) = 2.75
        // Q3 (75th): idx = 0.75 * 7 = 5.25 → lerp(6, 7, 0.25) = 6.25
        // IQR = 6.25 - 2.75 = 3.5
        // Lower = 2.75 - 1.5*3.5 = -2.5
        // Upper = 6.25 + 1.5*3.5 = 11.5
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 1).unwrap();

        assert!(!detector.is_outlier(-2.0, 0)); // Within bounds
        assert!(detector.is_outlier(-3.0, 0)); // Below lower
        assert!(!detector.is_outlier(11.0, 0)); // Within bounds
        assert!(detector.is_outlier(12.0, 0)); // Above upper
    }

    #[test]
    fn test_iqr_multifeature() {
        // 4 rows × 2 features
        let data = vec![
            1.0, 100.0, // row 0
            2.0, 200.0, // row 1
            3.0, 300.0, // row 2
            4.0, 400.0, // row 3
        ];

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 2).unwrap();

        // Both features should have their own bounds
        assert_eq!(detector.bounds.len(), 2);
    }

    // ========================================
    // Z-score Detection Tests
    // ========================================

    #[test]
    fn test_zscore_detection_basic() {
        // Mean = 5.5, Std ≈ 2.87
        // 3σ bounds ≈ [-3.1, 14.1]
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();

        let mut detector = OutlierDetector::new(OutlierMethod::zscore());
        detector.fit(&data, 1).unwrap();

        assert!(!detector.is_outlier(5.0, 0)); // Near mean
        assert!(detector.is_outlier(20.0, 0)); // Far from mean
        assert!(detector.is_outlier(-10.0, 0)); // Far below mean
    }

    #[test]
    fn test_zscore_custom_threshold() {
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();

        // Stricter threshold (2σ)
        let mut detector = OutlierDetector::new(OutlierMethod::zscore_with_threshold(2.0));
        detector.fit(&data, 1).unwrap();

        // Values that would pass 3σ should fail 2σ
        assert!(detector.is_outlier(12.0, 0));
    }

    // ========================================
    // Cap Action Tests
    // ========================================

    #[test]
    fn test_cap_action() {
        let mut data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,   // normal
            100.0, // outlier (high)
            -50.0, // outlier (low)
        ];

        let mut detector =
            OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Cap);
        detector.fit(&data, 1).unwrap();

        let result = detector.transform(&mut data, 1, &["f0".into()]).unwrap();

        // Outliers should be capped
        assert!(data[8] < 100.0); // Was capped down
        assert!(data[9] > -50.0); // Was capped up

        if let TransformResult::Capped { outlier_count } = result {
            assert_eq!(outlier_count, 2);
        } else {
            panic!("Expected Capped result");
        }
    }

    // ========================================
    // Flag Action Tests
    // ========================================

    #[test]
    fn test_flag_action() {
        let mut data = vec![
            1.0, 10.0, // row 0: normal
            2.0, 20.0, // row 1: normal
            3.0, 30.0, // row 2: normal
            4.0, 40.0, // row 3: normal
            100.0, 50.0, // row 4: outlier in f0
        ];

        let mut detector =
            OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Flag);
        detector.fit(&data, 2).unwrap();

        let names = vec!["f0".into(), "f1".into()];
        let result = detector.transform(&mut data, 2, &names).unwrap();

        if let TransformResult::Flagged {
            indicators,
            indicator_names,
        } = result
        {
            assert_eq!(indicator_names.len(), 2);
            assert_eq!(indicator_names[0], "f0_outlier");
            assert_eq!(indicator_names[1], "f1_outlier");

            // Row 4, feature 0 should be flagged
            assert!((indicators[4 * 2] - 1.0).abs() < 1e-6);
            // Row 4, feature 1 should not be flagged
            assert!((indicators[4 * 2 + 1] - 0.0).abs() < 1e-6);
        } else {
            panic!("Expected Flagged result");
        }
    }

    // ========================================
    // Remove Action Tests
    // ========================================

    #[test]
    fn test_remove_action() {
        let mut data = vec![
            1.0, 10.0, // row 0: normal
            2.0, 20.0, // row 1: normal
            3.0, 30.0, // row 2: normal
            4.0, 40.0, // row 3: normal
            100.0, 50.0, // row 4: outlier in f0
        ];

        let mut detector =
            OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Remove);
        detector.fit(&data, 2).unwrap();

        let names = vec!["f0".into(), "f1".into()];
        let result = detector.transform(&mut data, 2, &names).unwrap();

        if let TransformResult::Removed {
            cleaned_data,
            kept_indices,
            removed_count,
        } = result
        {
            assert_eq!(removed_count, 1);
            assert_eq!(kept_indices.len(), 4);
            assert_eq!(cleaned_data.len(), 8); // 4 rows × 2 features
            assert!(!kept_indices.contains(&4)); // Row 4 was removed
        } else {
            panic!("Expected Removed result");
        }
    }

    // ========================================
    // Edge Case Tests
    // ========================================

    #[test]
    fn test_no_outliers() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut detector =
            OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Cap);
        detector.fit(&data, 1).unwrap();

        let result = detector.transform(&mut data, 1, &["f0".into()]).unwrap();

        if let TransformResult::Capped { outlier_count } = result {
            assert_eq!(outlier_count, 0);
        }
    }

    #[test]
    fn test_nan_handling() {
        let data = vec![1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 1).unwrap();

        // NaN should not be detected as outlier
        assert!(!detector.is_outlier(f32::NAN, 0));
    }

    #[test]
    fn test_detect_method() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,   // normal
            100.0, // outlier
        ];

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 1).unwrap();

        let outliers = detector.detect(&data, 1).unwrap();
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], (8, 0)); // Row 8, Feature 0
    }

    #[test]
    fn test_outlier_counts() {
        let data = vec![
            1.0, 10.0, // row 0
            2.0, 20.0, // row 1
            3.0, 30.0, // row 2
            4.0, 40.0, // row 3
            100.0, 1000.0, // row 4: both features are outliers
        ];

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 2).unwrap();

        let counts = detector.outlier_counts(&data, 2).unwrap();
        assert_eq!(counts[0], 1); // 1 outlier in feature 0
        assert_eq!(counts[1], 1); // 1 outlier in feature 1
    }

    #[test]
    fn test_not_fitted_error() {
        let detector = OutlierDetector::new(OutlierMethod::iqr());

        let result = detector.detect(&[1.0, 2.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_mismatch_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut detector = OutlierDetector::new(OutlierMethod::iqr());
        detector.fit(&data, 1).unwrap();

        let result = detector.detect(&[1.0, 2.0, 3.0, 4.0], 2);
        assert!(result.is_err());
    }

    // ========================================
    // Percentile Tests
    // ========================================

    #[test]
    fn test_percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-6);
        assert!((percentile(&data, 0.5) - 3.0).abs() < 1e-6);
        assert!((percentile(&data, 1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_percentile_interpolation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Q1 = 25th percentile
        let q1 = percentile(&data, 0.25);
        assert!((q1 - 1.75).abs() < 1e-6);

        // Q3 = 75th percentile
        let q3 = percentile(&data, 0.75);
        assert!((q3 - 3.25).abs() < 1e-6);
    }

    // ========================================
    // Serialization Tests
    // ========================================

    #[test]
    fn test_outlier_detector_serialization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut detector =
            OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Cap);
        detector.fit(&data, 1).unwrap();

        // Serialize
        let json = serde_json::to_string(&detector).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let loaded: OutlierDetector = serde_json::from_str(&json).unwrap();
        assert!(loaded.is_fitted());
        assert_eq!(loaded.bounds.len(), 1);
        assert_eq!(loaded.method(), OutlierMethod::iqr());
        assert_eq!(loaded.action(), OutlierAction::Cap);
    }

    #[test]
    fn test_outlier_method_serialization() {
        let methods = vec![
            OutlierMethod::iqr(),
            OutlierMethod::iqr_with_k(2.0),
            OutlierMethod::zscore(),
            OutlierMethod::zscore_with_threshold(2.5),
        ];

        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let loaded: OutlierMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded, method);
        }
    }

    #[test]
    fn test_outlier_action_serialization() {
        let actions = vec![
            OutlierAction::Cap,
            OutlierAction::Flag,
            OutlierAction::Remove,
        ];

        for action in actions {
            let json = serde_json::to_string(&action).unwrap();
            let loaded: OutlierAction = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded, action);
        }
    }

    #[test]
    fn test_transform_result_serialization() {
        // Test Capped variant
        let result1 = TransformResult::Capped { outlier_count: 5 };
        let json1 = serde_json::to_string(&result1).unwrap();
        let loaded1: TransformResult = serde_json::from_str(&json1).unwrap();
        assert_eq!(loaded1.outlier_count(), 5);

        // Test Flagged variant
        let result2 = TransformResult::Flagged {
            indicators: vec![0.0, 1.0, 0.0],
            indicator_names: vec!["f0_outlier".into(), "f1_outlier".into()],
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let loaded2: TransformResult = serde_json::from_str(&json2).unwrap();
        assert_eq!(loaded2.outlier_count(), 1);

        // Test Removed variant
        let result3 = TransformResult::Removed {
            cleaned_data: vec![1.0, 2.0, 3.0],
            kept_indices: vec![0, 1, 2],
            removed_count: 2,
        };
        let json3 = serde_json::to_string(&result3).unwrap();
        let loaded3: TransformResult = serde_json::from_str(&json3).unwrap();
        assert_eq!(loaded3.outlier_count(), 2);
    }
}
