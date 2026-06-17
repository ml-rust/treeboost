//! Scaling transformations for numerical features
//!
//! Scalers normalize feature distributions to improve model performance:
//! - **StandardScaler**: Zero mean, unit variance (most common)
//! - **MinMaxScaler**: Scale to fixed range [min, max]
//! - **RobustScaler**: Use median/IQR (robust to outliers)
//!
//! # Incremental Support
//!
//! StandardScaler and MinMaxScaler support incremental fitting via the
//! `IncrementalScaler` trait, allowing updates with new data batches:
//!
//! ```ignore
//! use treeboost::preprocessing::{StandardScaler, Scaler};
//! use treeboost::preprocessing::incremental::IncrementalScaler;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.partial_fit(&batch1, num_features)?;
//! scaler.partial_fit(&batch2, num_features)?;
//! // scaler now has statistics from both batches
//! ```
//!
//! # Example
//!
//! ```rust
//! use treeboost::preprocessing::{StandardScaler, Scaler};
//!
//! let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
//! let num_features = 3;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(&data, num_features);
//! scaler.transform(&mut data, num_features);
//!
//! // data is now standardized: (x - mean) / std
//! ```

use crate::preprocessing::incremental::{IncrementalScaler, WelfordState};
use crate::{Result, TreeBoostError};

// =============================================================================
// Scaler Trait
// =============================================================================

/// Trait for all scalers (fit-transform pattern)
///
/// All scalers must implement:
/// - `fit()`: Learn parameters from training data
/// - `transform()`: Apply learned parameters to data
/// - Serialization for train/test consistency
pub trait Scaler {
    /// Fit scaler on training data (row-major: num_rows × num_features)
    ///
    /// # Arguments
    /// - `data`: Row-major flat array (row0_feat0, row0_feat1, ..., row1_feat0, ...)
    /// - `num_features`: Number of features per row
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()>;

    /// Transform data in-place using fitted parameters
    ///
    /// # Arguments
    /// - `data`: Row-major flat array to transform in-place
    /// - `num_features`: Number of features per row (must match fit)
    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()>;

    /// Fit and transform in one step (convenience)
    fn fit_transform(&mut self, data: &mut [f32], num_features: usize) -> Result<()> {
        self.fit(data, num_features)?;
        self.transform(data, num_features)?;
        Ok(())
    }

    /// Check if scaler has been fitted
    fn is_fitted(&self) -> bool;
}

// =============================================================================
// StandardScaler
// =============================================================================

/// StandardScaler: (x - μ) / σ
///
/// Transforms features to have zero mean and unit variance.
///
/// # Why it helps GBDTs
/// Even though trees are scale-invariant, scaling improves:
/// - **Regularization fairness**: L1/L2 penalties applied uniformly
/// - **Binning uniformity**: Quantiles distributed evenly
/// - **Numerical stability**: Gradient/Hessian calculations
/// - **Mixed ensembles**: Combining linear + tree models
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{StandardScaler, Scaler};
///
/// let mut train = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]; // 3 rows × 2 features
/// let mut test = vec![1.5, 15.0, 2.5, 25.0]; // 2 rows × 2 features
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(&train, 2)?;
/// scaler.transform(&mut train, 2)?;
/// scaler.transform(&mut test, 2)?; // Use same mean/std from training
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StandardScaler {
    /// Mean of each feature (learned during fit)
    pub means: Vec<f32>,
    /// Standard deviation of each feature (learned during fit)
    pub stds: Vec<f32>,
    /// Whether fit() has been called
    fitted: bool,
    /// Welford state for incremental updates (one per feature)
    #[serde(default)]
    welford_states: Vec<WelfordState>,
    /// Optional forget factor (alpha) for EMA-based rolling window updates (0.0 to 1.0)
    ///
    /// When set, statistics are updated using exponential moving average:
    /// `new_stat = (1 - alpha) * old_stat + alpha * batch_stat`
    ///
    /// - `alpha=0.0`: Ignores new batches entirely (not useful)
    /// - `alpha=0.1`: 10% blend from new batch per update (slow adaptation)
    /// - `alpha=0.5`: Equal blend of old and new statistics each update
    /// - `alpha=1.0`: Completely replace with new batch statistics
    ///
    /// **Decay behavior**: After N batches, the first batch's influence is `(1-alpha)^N`.
    /// Example with alpha=0.1: after 10 batches, first batch retains ~35% influence.
    ///
    /// Use small values (0.05-0.2) for gradual adaptation to distribution drift.
    #[serde(default)]
    forget_factor: Option<f32>,
}

impl StandardScaler {
    /// Create a new unfitted StandardScaler
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
            welford_states: Vec::new(),
            forget_factor: None,
        }
    }

    /// Create a StandardScaler with EMA-based rolling window updates
    ///
    /// # Arguments
    /// * `forget_factor` - Alpha value between 0.0 and 1.0 (clamped if out of range)
    ///
    /// # Example
    /// ```ignore
    /// // Create scaler with alpha=0.1 (10% blend from each new batch)
    /// let mut scaler = StandardScaler::with_forget_factor(0.1);
    /// scaler.partial_fit(&batch1, num_features)?;  // 100% batch1
    /// scaler.partial_fit(&batch2, num_features)?;  // 90% batch1, 10% batch2
    /// scaler.partial_fit(&batch3, num_features)?;  // 81% batch1, 9% batch2, 10% batch3
    /// ```
    pub fn with_forget_factor(forget_factor: f32) -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
            welford_states: Vec::new(),
            forget_factor: Some(forget_factor.clamp(0.0, 1.0)),
        }
    }

    /// Set the forget factor for EMA-based updates
    ///
    /// # Arguments
    /// * `factor` - Value between 0.0 and 1.0, or None to disable EMA mode
    pub fn set_forget_factor(&mut self, factor: Option<f32>) {
        self.forget_factor = factor.map(|f| f.clamp(0.0, 1.0));
    }

    /// Get the current forget factor
    pub fn forget_factor(&self) -> Option<f32> {
        self.forget_factor
    }

    /// Get the means (only valid after fit)
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// Get the standard deviations (only valid after fit)
    pub fn stds(&self) -> &[f32] {
        &self.stds
    }

    /// Inverse transform: Convert standardized data back to original scale
    ///
    /// # Arguments
    /// * `data` - Mutable slice of standardized data in row-major order
    /// * `num_features` - Number of features per row
    ///
    /// # Formula
    /// ```text
    /// x = y * std + mean
    /// ```
    ///
    /// # Example
    /// ```
    /// use treeboost::preprocessing::{StandardScaler, Scaler};
    ///
    /// let mut scaler = StandardScaler::new();
    /// let mut data = vec![10.0, 50.0, 90.0];
    ///
    /// scaler.fit(&data, 1).unwrap();
    /// scaler.transform(&mut data, 1).unwrap(); // Standardized
    /// scaler.inverse_transform(&mut data, 1).unwrap(); // Back to original
    ///
    /// assert!((data[0] - 10.0).abs() < 0.01);
    /// assert!((data[1] - 50.0).abs() < 0.01);
    /// assert!((data[2] - 90.0).abs() < 0.01);
    /// ```
    pub fn inverse_transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "StandardScaler not fitted. Call fit() first before inverse_transform().".into(),
            ));
        }

        if num_features != self.means.len() {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::inverse_transform() feature count mismatch: fitted with {} features, \
                 but inverse_transform() called with {} features.",
                self.means.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::inverse_transform() received invalid data layout: {} elements not divisible by {} features.",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        // Inverse standardization: x = y * std + mean
        for feat in 0..num_features {
            let mean = self.means[feat];
            let std = self.stds[feat];

            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = data[idx] * std + mean;
            }
        }

        Ok(())
    }

    /// Sync means/stds from Welford states (internal helper)
    fn sync_from_welford(&mut self) {
        let num_features = self.welford_states.len();
        self.means.resize(num_features, 0.0);
        self.stds.resize(num_features, 1.0);

        for (i, state) in self.welford_states.iter().enumerate() {
            self.means[i] = state.mean as f32;
            let std = state.std() as f32;
            // Handle zero-variance features (constant column)
            self.stds[i] = if std < 1e-8 { 1.0 } else { std };
        }
    }

    /// Compute mean and variance for a batch (helper for EMA updates)
    fn compute_batch_stats(data: &[f32], num_features: usize) -> Vec<(f64, f64)> {
        let num_rows = data.len() / num_features;
        let mut stats = vec![(0.0f64, 0.0f64); num_features];

        if num_rows == 0 {
            return stats;
        }

        // Compute means
        for feat in 0..num_features {
            let mut sum = 0.0f64;
            for row in 0..num_rows {
                sum += data[row * num_features + feat] as f64;
            }
            stats[feat].0 = sum / num_rows as f64;
        }

        // Compute variances
        for feat in 0..num_features {
            let mean = stats[feat].0;
            let mut variance = 0.0f64;
            for row in 0..num_rows {
                let x = data[row * num_features + feat] as f64;
                variance += (x - mean).powi(2);
            }
            stats[feat].1 = variance / num_rows as f64;
        }

        stats
    }

    /// EMA-based partial fit for rolling window updates
    ///
    /// Uses exponential moving average to decay old statistics:
    /// new_mean = (1 - alpha) * old_mean + alpha * batch_mean
    /// new_var = (1 - alpha) * old_var + alpha * batch_var
    ///
    /// This allows the scaler to adapt to distribution drift over time.
    fn partial_fit_ema(&mut self, data: &[f32], num_features: usize, alpha: f32) -> Result<()> {
        let num_rows = data.len() / num_features;
        if num_rows == 0 {
            return Ok(());
        }

        // Compute batch statistics
        let batch_stats = Self::compute_batch_stats(data, num_features);

        // First batch: just use batch stats directly
        if self.means.is_empty() || !self.fitted {
            self.means = vec![0.0; num_features];
            self.stds = vec![1.0; num_features];
            self.welford_states = vec![WelfordState::new(); num_features];

            for (feat, &(mean, var)) in batch_stats.iter().enumerate() {
                self.means[feat] = mean as f32;
                let std = var.sqrt() as f32;
                self.stds[feat] = if std < 1e-8 { 1.0 } else { std };

                // Also initialize Welford state for sample counting
                self.welford_states[feat].n = num_rows as u64;
                self.welford_states[feat].mean = mean;
                self.welford_states[feat].m2 = var * num_rows as f64;
            }
            self.fitted = true;
            return Ok(());
        }

        // Check feature count consistency
        if self.means.len() != num_features {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::partial_fit_ema() feature count mismatch: previously initialized with {} features, \
                 but partial_fit_ema() called with {} features. All EMA updates must have consistent feature count.",
                self.means.len(),
                num_features
            )));
        }

        // EMA update: new = (1 - alpha) * old + alpha * batch
        let alpha_64 = alpha as f64;
        let decay = 1.0 - alpha_64;

        for (feat, &(batch_mean, batch_var)) in batch_stats.iter().enumerate() {
            // Update mean via EMA
            let old_mean = self.means[feat] as f64;
            let new_mean = decay * old_mean + alpha_64 * batch_mean;
            self.means[feat] = new_mean as f32;

            // Update variance via EMA
            // Note: This is approximate for variance, but works well in practice
            let old_var = (self.stds[feat] as f64).powi(2);
            let new_var = decay * old_var + alpha_64 * batch_var;
            let new_std = new_var.sqrt() as f32;
            self.stds[feat] = if new_std < 1e-8 { 1.0 } else { new_std };

            // Update sample count (approximate effective samples)
            self.welford_states[feat].n += num_rows as u64;
        }

        Ok(())
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for StandardScaler {
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "StandardScaler::fit() requires num_features > 0, got 0".into(),
            ));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::fit() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows == 0 {
            return Err(TreeBoostError::Data(
                "StandardScaler::fit() received empty dataset (0 rows). Provide data with at least 1 row.".into(),
            ));
        }

        self.means = vec![0.0; num_features];
        self.stds = vec![0.0; num_features];

        // Compute means
        for feat in 0..num_features {
            let mut sum = 0.0;
            for row in 0..num_rows {
                sum += data[row * num_features + feat];
            }
            self.means[feat] = sum / num_rows as f32;
        }

        // Compute standard deviations
        for feat in 0..num_features {
            let mean = self.means[feat];
            let mut variance = 0.0;
            for row in 0..num_rows {
                let x = data[row * num_features + feat];
                variance += (x - mean).powi(2);
            }
            let std = (variance / num_rows as f32).sqrt();

            // Handle zero-variance features (constant column)
            self.stds[feat] = if std < 1e-8 { 1.0 } else { std };
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "StandardScaler not fitted. Call fit() first to learn scaling parameters.".into(),
            ));
        }

        if num_features != self.means.len() {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::transform() feature count mismatch: fitted with {} features, \
                 but transform() called with {} features. Ensure transform data has same feature count as training data.",
                self.means.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::transform() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        // Apply standardization: (x - mean) / std
        for feat in 0..num_features {
            let mean = self.means[feat];
            let std = self.stds[feat];
            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - mean) / std;
            }
        }

        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl IncrementalScaler for StandardScaler {
    fn partial_fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "StandardScaler::partial_fit() requires num_features > 0, got 0".into(),
            ));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::partial_fit() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;
        if num_rows == 0 {
            return Ok(()); // Nothing to do
        }

        // If forget_factor is set, use EMA-based updates
        if let Some(alpha) = self.forget_factor {
            return self.partial_fit_ema(data, num_features, alpha);
        }

        // Standard Welford-based incremental update (cumulative, no decay)

        // Initialize Welford states if this is the first call
        if self.welford_states.is_empty() {
            self.welford_states = vec![WelfordState::new(); num_features];
        } else if self.welford_states.len() != num_features {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::partial_fit() feature count mismatch: previously initialized with {} features, \
                 but partial_fit() called with {} features. All partial_fit() calls must have consistent feature count.",
                self.welford_states.len(),
                num_features
            )));
        }

        // Update Welford states with new data
        for row in 0..num_rows {
            for feat in 0..num_features {
                let x = data[row * num_features + feat] as f64;
                if x.is_finite() {
                    self.welford_states[feat].update(x);
                }
            }
        }

        // Sync mean/std from Welford states
        self.sync_from_welford();
        self.fitted = true;

        Ok(())
    }

    fn n_samples(&self) -> u64 {
        self.welford_states.first().map(|s| s.n).unwrap_or(0)
    }

    fn merge(&mut self, other: &Self) -> Result<()> {
        if self.welford_states.is_empty() {
            // Copy from other
            self.welford_states = other.welford_states.clone();
            self.sync_from_welford();
            self.fitted = other.fitted;
            return Ok(());
        }

        if other.welford_states.is_empty() {
            return Ok(()); // Nothing to merge
        }

        if self.welford_states.len() != other.welford_states.len() {
            return Err(TreeBoostError::Data(format!(
                "StandardScaler::merge() feature count mismatch: left scaler has {} features, \
                 right scaler has {} features. Both scalers must be initialized with the same feature count.",
                self.welford_states.len(),
                other.welford_states.len()
            )));
        }

        // Merge Welford states using Chan's parallel algorithm
        for (self_state, other_state) in self.welford_states.iter_mut().zip(&other.welford_states) {
            self_state.merge(other_state);
        }

        self.sync_from_welford();
        Ok(())
    }
}

// =============================================================================
// MinMaxScaler
// =============================================================================

/// MinMaxScaler: (x - min) / (max - min) * (b - a) + a
///
/// Scales features to a fixed range [a, b] (default [0, 1]).
///
/// # Use cases
/// - When you need features in a specific range (e.g., [0, 1] for neural nets)
/// - When you know the expected min/max bounds
///
/// # Warning
/// - Sensitive to outliers (one extreme value affects entire scale)
/// - Consider RobustScaler if outliers are present
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{MinMaxScaler, Scaler};
///
/// let mut data = vec![1.0, 5.0, 2.0, 10.0, 3.0, 15.0]; // 3 rows × 2 features
///
/// let mut scaler = MinMaxScaler::new().with_range(0.0, 1.0);
/// scaler.fit(&data, 2)?;
/// scaler.transform(&mut data, 2)?;
///
/// // data is now in [0, 1] range
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinMaxScaler {
    /// Minimum of each feature (learned during fit)
    pub mins: Vec<f32>,
    /// Maximum of each feature (learned during fit)
    pub maxs: Vec<f32>,
    /// Output range (a, b)
    pub feature_range: (f32, f32),
    /// Whether fit() has been called
    fitted: bool,
    /// Number of samples seen (for incremental fitting)
    #[serde(default)]
    n_samples: u64,
}

impl MinMaxScaler {
    /// Create a new unfitted MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            feature_range: (0.0, 1.0),
            fitted: false,
            n_samples: 0,
        }
    }

    /// Set the output range (default is [0, 1])
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.feature_range = (min, max);
        self
    }

    /// Inverse transform: Convert scaled data back to original range
    ///
    /// # Arguments
    /// * `data` - Mutable slice of scaled data in row-major order (num_rows × num_features)
    /// * `num_features` - Number of features per row
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if scaler not fitted or data dimensions don't match
    ///
    /// # Formula
    /// Given scaled value `y` in range [a, b], recover original `x`:
    /// ```text
    /// x = (y - a) / (b - a) * (max - min) + min
    /// ```
    ///
    /// # Example
    /// ```
    /// use treeboost::preprocessing::{MinMaxScaler, Scaler};
    ///
    /// let mut scaler = MinMaxScaler::new().with_range(0.0, 1.0);
    /// let mut data = vec![10.0, 50.0, 90.0]; // Original: [10, 50, 90]
    ///
    /// scaler.fit(&data, 1).unwrap();
    /// scaler.transform(&mut data, 1).unwrap(); // Scaled: [0.0, 0.5, 1.0]
    ///
    /// scaler.inverse_transform(&mut data, 1).unwrap(); // Back to: [10, 50, 90]
    /// assert!((data[0] - 10.0).abs() < 0.01);
    /// assert!((data[1] - 50.0).abs() < 0.01);
    /// assert!((data[2] - 90.0).abs() < 0.01);
    /// ```
    pub fn inverse_transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "MinMaxScaler not fitted. Call fit() first before inverse_transform().".into(),
            ));
        }

        if num_features != self.mins.len() {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::inverse_transform() feature count mismatch: fitted with {} features, \
                 but inverse_transform() called with {} features.",
                self.mins.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::inverse_transform() received invalid data layout: {} elements not divisible by {} features.",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;
        let (a, b) = self.feature_range;
        let scale = b - a;

        // Inverse scaling: x = (y - a) / (b - a) * (max - min) + min
        for feat in 0..num_features {
            let min = self.mins[feat];
            let max = self.maxs[feat];
            let data_range = max - min;

            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - a) / scale * data_range + min;
            }
        }

        Ok(())
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for MinMaxScaler {
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "MinMaxScaler::fit() requires num_features > 0, got 0".into(),
            ));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::fit() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows == 0 {
            return Err(TreeBoostError::Data(
                "MinMaxScaler::fit() received empty dataset (0 rows). Provide data with at least 1 row.".into(),
            ));
        }

        self.mins = vec![f32::INFINITY; num_features];
        self.maxs = vec![f32::NEG_INFINITY; num_features];

        // Find min and max for each feature
        for feat in 0..num_features {
            for row in 0..num_rows {
                let val = data[row * num_features + feat];
                self.mins[feat] = self.mins[feat].min(val);
                self.maxs[feat] = self.maxs[feat].max(val);
            }

            // Handle constant features (min == max)
            if (self.maxs[feat] - self.mins[feat]).abs() < 1e-8 {
                self.maxs[feat] = self.mins[feat] + 1.0;
            }
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "MinMaxScaler not fitted. Call fit() first to learn min/max bounds.".into(),
            ));
        }

        if num_features != self.mins.len() {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::transform() feature count mismatch: fitted with {} features, \
                 but transform() called with {} features. Ensure transform data has same feature count as training data.",
                self.mins.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::transform() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;
        let (a, b) = self.feature_range;

        // Apply scaling: (x - min) / (max - min) * (b - a) + a
        for feat in 0..num_features {
            let min = self.mins[feat];
            let max = self.maxs[feat];
            let scale = b - a;

            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - min) / (max - min) * scale + a;

                // Clip to range (handles out-of-bound values in test set)
                data[idx] = data[idx].clamp(a, b);
            }
        }

        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl IncrementalScaler for MinMaxScaler {
    fn partial_fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "MinMaxScaler::partial_fit() requires num_features > 0, got 0".into(),
            ));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::partial_fit() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;
        if num_rows == 0 {
            return Ok(()); // Nothing to do
        }

        // Initialize mins/maxs if this is the first call
        if self.mins.is_empty() {
            self.mins = vec![f32::INFINITY; num_features];
            self.maxs = vec![f32::NEG_INFINITY; num_features];
        } else if self.mins.len() != num_features {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::partial_fit() feature count mismatch: previously initialized with {} features, \
                 but partial_fit() called with {} features. All partial_fit() calls must have consistent feature count.",
                self.mins.len(),
                num_features
            )));
        }

        // Update min/max with new data (monotonic expansion)
        for row in 0..num_rows {
            for feat in 0..num_features {
                let val = data[row * num_features + feat];
                if val.is_finite() {
                    self.mins[feat] = self.mins[feat].min(val);
                    self.maxs[feat] = self.maxs[feat].max(val);
                }
            }
        }

        // Handle constant features
        for feat in 0..num_features {
            if (self.maxs[feat] - self.mins[feat]).abs() < 1e-8 {
                self.maxs[feat] = self.mins[feat] + 1.0;
            }
        }

        self.n_samples += num_rows as u64;
        self.fitted = true;

        Ok(())
    }

    fn n_samples(&self) -> u64 {
        self.n_samples
    }

    fn merge(&mut self, other: &Self) -> Result<()> {
        if self.mins.is_empty() {
            // Copy from other
            self.mins = other.mins.clone();
            self.maxs = other.maxs.clone();
            self.n_samples = other.n_samples;
            self.fitted = other.fitted;
            return Ok(());
        }

        if other.mins.is_empty() {
            return Ok(()); // Nothing to merge
        }

        if self.mins.len() != other.mins.len() {
            return Err(TreeBoostError::Data(format!(
                "MinMaxScaler::merge() feature count mismatch: left scaler has {} features, \
                 right scaler has {} features. Both scalers must be initialized with the same feature count.",
                self.mins.len(),
                other.mins.len()
            )));
        }

        // Merge min/max (take min of mins, max of maxs)
        for i in 0..self.mins.len() {
            self.mins[i] = self.mins[i].min(other.mins[i]);
            self.maxs[i] = self.maxs[i].max(other.maxs[i]);
        }

        self.n_samples += other.n_samples;
        Ok(())
    }
}

// =============================================================================
// RobustScaler
// =============================================================================

/// RobustScaler: (x - median) / IQR
///
/// Scales features using statistics robust to outliers:
/// - Center: median (instead of mean)
/// - Scale: IQR = Q3 - Q1 (instead of std)
///
/// # Use cases
/// - Data with outliers or heavy-tailed distributions
/// - When mean/std are unreliable due to extreme values
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{RobustScaler, Scaler};
///
/// let mut data = vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0]; // 3 rows × 2 features (outlier: 100)
///
/// let mut scaler = RobustScaler::new();
/// scaler.fit(&data, 2)?;
/// scaler.transform(&mut data, 2)?;
///
/// // Outlier (100) has less impact than with StandardScaler
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RobustScaler {
    /// Median of each feature (learned during fit)
    pub medians: Vec<f32>,
    /// IQR (Q3 - Q1) of each feature (learned during fit)
    pub iqrs: Vec<f32>,
    /// Whether fit() has been called
    fitted: bool,
}

impl RobustScaler {
    /// Create a new unfitted RobustScaler
    pub fn new() -> Self {
        Self {
            medians: Vec::new(),
            iqrs: Vec::new(),
            fitted: false,
        }
    }

    /// Inverse transform: Convert scaled data back to original scale
    ///
    /// # Arguments
    /// * `data` - Mutable slice of scaled data in row-major order
    /// * `num_features` - Number of features per row
    ///
    /// # Formula
    /// ```text
    /// x = y * IQR + median
    /// ```
    ///
    /// # Example
    /// ```
    /// use treeboost::preprocessing::{RobustScaler, Scaler};
    ///
    /// let mut scaler = RobustScaler::new();
    /// let mut data = vec![10.0, 50.0, 90.0];
    ///
    /// scaler.fit(&data, 1).unwrap();
    /// scaler.transform(&mut data, 1).unwrap(); // Scaled
    /// scaler.inverse_transform(&mut data, 1).unwrap(); // Back to original
    ///
    /// assert!((data[0] - 10.0).abs() < 0.01);
    /// assert!((data[1] - 50.0).abs() < 0.01);
    /// assert!((data[2] - 90.0).abs() < 0.01);
    /// ```
    pub fn inverse_transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "RobustScaler not fitted. Call fit() first before inverse_transform().".into(),
            ));
        }

        if num_features != self.medians.len() {
            return Err(TreeBoostError::Data(format!(
                "RobustScaler::inverse_transform() feature count mismatch: fitted with {} features, \
                 but inverse_transform() called with {} features.",
                self.medians.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "RobustScaler::inverse_transform() received invalid data layout: {} elements not divisible by {} features.",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        // Inverse robust scaling: x = y * IQR + median
        for feat in 0..num_features {
            let median = self.medians[feat];
            let iqr = self.iqrs[feat];

            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = data[idx] * iqr + median;
            }
        }

        Ok(())
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for RobustScaler {
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "RobustScaler::fit() requires num_features > 0, got 0".into(),
            ));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "RobustScaler::fit() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows == 0 {
            return Err(TreeBoostError::Data(
                "RobustScaler::fit() received empty dataset (0 rows). Provide data with at least 1 row.".into(),
            ));
        }

        self.medians = vec![0.0; num_features];
        self.iqrs = vec![0.0; num_features];

        // Use T-Digest for O(n) quantile estimation instead of O(n log n) sorting
        // This is critical for large datasets (100M+ rows)
        use tdigest::TDigest;

        // Compute median and IQR for each feature using approximate quantiles
        for feat in 0..num_features {
            // Build T-Digest for this feature column
            let mut digest = TDigest::new_with_size(100); // 100 centroids is accurate enough

            for row in 0..num_rows {
                let value = data[row * num_features + feat] as f64;
                if value.is_finite() {
                    digest = digest.merge_unsorted(vec![value]);
                }
            }

            // Get approximate quantiles from T-Digest
            let q1 = digest.estimate_quantile(0.25) as f32;
            let median = digest.estimate_quantile(0.50) as f32;
            let q3 = digest.estimate_quantile(0.75) as f32;

            self.medians[feat] = median;

            // IQR = Q3 - Q1
            let iqr = q3 - q1;

            // Handle zero IQR (all values in Q1-Q3 range are same)
            self.iqrs[feat] = if iqr < 1e-8 { 1.0 } else { iqr };
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "RobustScaler not fitted. Call fit() first to learn median/IQR statistics.".into(),
            ));
        }

        if num_features != self.medians.len() {
            return Err(TreeBoostError::Data(format!(
                "RobustScaler::transform() feature count mismatch: fitted with {} features, \
                 but transform() called with {} features. Ensure transform data has same feature count as training data.",
                self.medians.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "RobustScaler::transform() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features = {} × {}",
                data.len(),
                num_features,
                data.len() / num_features.max(1),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        // Apply robust scaling: (x - median) / IQR
        for feat in 0..num_features {
            let median = self.medians[feat];
            let iqr = self.iqrs[feat];
            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - median) / iqr;
            }
        }

        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 2 rows × 3 features
        // Row 0: [1.0, 2.0, 3.0]
        // Row 1: [4.0, 5.0, 6.0]
        let num_features = 3;

        let mut scaler = StandardScaler::new();
        assert!(!scaler.is_fitted());

        scaler.fit(&data, num_features).unwrap();
        assert!(scaler.is_fitted());

        // Check means (column averages)
        // Feature 0: (1.0 + 4.0) / 2 = 2.5
        // Feature 1: (2.0 + 5.0) / 2 = 3.5
        // Feature 2: (3.0 + 6.0) / 2 = 4.5
        assert_eq!(scaler.means(), &[2.5, 3.5, 4.5]);

        scaler.transform(&mut data, num_features).unwrap();

        // After standardization, mean should be ~0, std should be ~1
    }

    #[test]
    fn test_standard_scaler_zero_variance() {
        let mut data = vec![5.0, 1.0, 2.0, 5.0, 3.0, 4.0];
        // 2 rows × 3 features
        // Row 0: [5.0, 1.0, 2.0]
        // Row 1: [5.0, 3.0, 4.0]
        // Feature 0 is constant: [5.0, 5.0]
        let num_features = 3;

        let mut scaler = StandardScaler::new();
        scaler.fit(&data, num_features).unwrap();

        // Zero-variance feature should have std = 1.0 (fallback)
        assert_eq!(scaler.stds[0], 1.0);
        assert_eq!(scaler.means[0], 5.0);

        // Transform should not panic
        scaler.transform(&mut data, num_features).unwrap();
    }

    #[test]
    fn test_minmax_scaler_basic() {
        let mut data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]; // 3 rows × 2 features
        let num_features = 2;

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&data, num_features).unwrap();

        assert_eq!(scaler.mins, vec![1.0, 10.0]);
        assert_eq!(scaler.maxs, vec![3.0, 30.0]);

        scaler.transform(&mut data, num_features).unwrap();

        // First feature: [1, 2, 3] → [0.0, 0.5, 1.0]
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.5).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);

        // Second feature: [10, 20, 30] → [0.0, 0.5, 1.0]
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[3] - 0.5).abs() < 1e-6);
        assert!((data[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_scaler_custom_range() {
        let mut data = vec![1.0, 2.0, 3.0]; // 3 rows × 1 feature
        let num_features = 1;

        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        scaler.fit(&data, num_features).unwrap();
        scaler.transform(&mut data, num_features).unwrap();

        // [1, 2, 3] → [-1.0, 0.0, 1.0]
        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_robust_scaler_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 100.0]; // 2 rows × 2 features (outlier: 100)
        let num_features = 2;

        let mut scaler = RobustScaler::new();
        scaler.fit(&data, num_features).unwrap();

        // Medians: [2.0, 51.0] (avg of middle two values)
        assert!((scaler.medians[0] - 2.0).abs() < 1e-6);

        scaler.transform(&mut data, num_features).unwrap();

        // Check that outlier doesn't dominate (median-based scaling)
    }

    #[test]
    fn test_scaler_not_fitted_error() {
        let mut data = vec![1.0, 2.0, 3.0];
        let scaler = StandardScaler::new();

        let result = scaler.transform(&mut data, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not fitted"));
    }

    #[test]
    fn test_scaler_feature_mismatch_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut scaler = StandardScaler::new();

        scaler.fit(&data, 2).unwrap(); // Fit with 2 features

        let mut test_data = vec![5.0, 6.0, 7.0];
        let result = scaler.transform(&mut test_data, 3); // Try to transform with 3 features

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    // =========================================================================
    // Incremental Scaler Tests
    // =========================================================================

    #[test]
    fn test_standard_scaler_incremental_equivalence() {
        // Test that partial_fit on chunks equals fit on all data
        let all_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let num_features = 1;

        // Scaler A: fit() on all data
        let mut scaler_a = StandardScaler::new();
        scaler_a.fit(&all_data, num_features).unwrap();

        // Scaler B: partial_fit() on 10 chunks of 100
        let mut scaler_b = StandardScaler::new();
        for chunk in all_data.chunks(100) {
            scaler_b.partial_fit(chunk, num_features).unwrap();
        }

        // Verify equivalence (within f32 epsilon)
        assert!(
            (scaler_a.means[0] - scaler_b.means[0]).abs() < 1e-3,
            "Means differ: {} vs {}",
            scaler_a.means[0],
            scaler_b.means[0]
        );
        assert!(
            (scaler_a.stds[0] - scaler_b.stds[0]).abs() < 1e-3,
            "Stds differ: {} vs {}",
            scaler_a.stds[0],
            scaler_b.stds[0]
        );

        // Verify n_samples tracked correctly
        assert_eq!(scaler_b.n_samples(), 1000);
    }

    #[test]
    fn test_standard_scaler_welford_stability() {
        // Test numerical stability with large offset
        let offset = 1e8_f32;
        let data: Vec<f32> = (0..100).map(|i| offset + i as f32).collect();
        let num_features = 1;

        let mut scaler = StandardScaler::new();
        scaler.partial_fit(&data, num_features).unwrap();

        // Mean should be offset + 49.5
        let expected_mean = offset + 49.5;
        assert!(
            (scaler.means[0] - expected_mean).abs() < 1.0,
            "Mean with large offset: got {}, expected {}",
            scaler.means[0],
            expected_mean
        );
    }

    #[test]
    fn test_standard_scaler_merge() {
        let num_features = 2;

        // Scaler A: [1, 2, 3, 4] for 2 features
        let mut scaler_a = StandardScaler::new();
        scaler_a
            .partial_fit(&[1.0, 10.0, 2.0, 20.0], num_features)
            .unwrap();

        // Scaler B: [3, 4, 5, 6] for 2 features
        let mut scaler_b = StandardScaler::new();
        scaler_b
            .partial_fit(&[3.0, 30.0, 4.0, 40.0], num_features)
            .unwrap();

        // Merge B into A
        scaler_a.merge(&scaler_b).unwrap();

        // Should be equivalent to fitting on all 4 rows
        assert_eq!(scaler_a.n_samples(), 4);

        // Feature 0: [1, 2, 3, 4] → mean = 2.5
        assert!((scaler_a.means[0] - 2.5).abs() < 1e-5);

        // Feature 1: [10, 20, 30, 40] → mean = 25.0
        assert!((scaler_a.means[1] - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_minmax_scaler_incremental() {
        let num_features = 2;

        let mut scaler = MinMaxScaler::new();

        // Batch 1: data in [0, 50] for feat 0, [0, 100] for feat 1
        scaler
            .partial_fit(&[0.0, 0.0, 50.0, 100.0], num_features)
            .unwrap();
        assert_eq!(scaler.mins, vec![0.0, 0.0]);
        assert_eq!(scaler.maxs, vec![50.0, 100.0]);

        // Batch 2: data in [25, 100] for feat 0, [50, 200] for feat 1
        scaler
            .partial_fit(&[25.0, 50.0, 100.0, 200.0], num_features)
            .unwrap();

        // Min/max should expand monotonically
        assert_eq!(scaler.mins, vec![0.0, 0.0]); // min stays at 0
        assert_eq!(scaler.maxs, vec![100.0, 200.0]); // max expands

        assert_eq!(scaler.n_samples(), 4);
    }

    #[test]
    fn test_minmax_scaler_merge() {
        let num_features = 1;

        let mut scaler_a = MinMaxScaler::new();
        scaler_a.partial_fit(&[10.0, 20.0], num_features).unwrap();

        let mut scaler_b = MinMaxScaler::new();
        scaler_b.partial_fit(&[5.0, 30.0], num_features).unwrap();

        scaler_a.merge(&scaler_b).unwrap();

        // Should have min=5, max=30 (union of ranges)
        assert_eq!(scaler_a.mins, vec![5.0]);
        assert_eq!(scaler_a.maxs, vec![30.0]);
        assert_eq!(scaler_a.n_samples(), 4);
    }

    // =========================================================================
    // Rolling Window / EMA Tests
    // =========================================================================

    #[test]
    fn test_standard_scaler_forget_factor_creation() {
        let scaler = StandardScaler::with_forget_factor(0.1);
        assert_eq!(scaler.forget_factor(), Some(0.1));

        let mut scaler2 = StandardScaler::new();
        assert_eq!(scaler2.forget_factor(), None);

        scaler2.set_forget_factor(Some(0.5));
        assert_eq!(scaler2.forget_factor(), Some(0.5));

        scaler2.set_forget_factor(None);
        assert_eq!(scaler2.forget_factor(), None);
    }

    #[test]
    fn test_standard_scaler_forget_factor_clamping() {
        let scaler = StandardScaler::with_forget_factor(-0.5);
        assert_eq!(scaler.forget_factor(), Some(0.0));

        let scaler2 = StandardScaler::with_forget_factor(1.5);
        assert_eq!(scaler2.forget_factor(), Some(1.0));
    }

    #[test]
    fn test_standard_scaler_ema_single_batch() {
        // First batch should set stats directly
        let num_features = 1;
        let data = vec![10.0, 20.0, 30.0, 40.0];

        let mut scaler = StandardScaler::with_forget_factor(0.1);
        scaler.partial_fit(&data, num_features).unwrap();

        assert!(scaler.is_fitted());
        // Mean should be 25.0
        assert!((scaler.means()[0] - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_standard_scaler_ema_decay() {
        let num_features = 1;

        // Batch 1: mean=10
        let batch1 = vec![8.0, 10.0, 12.0];
        // Batch 2: mean=100 (shifted distribution)
        let batch2 = vec![98.0, 100.0, 102.0];

        let mut scaler = StandardScaler::with_forget_factor(0.3);
        scaler.partial_fit(&batch1, num_features).unwrap();

        let mean_after_batch1 = scaler.means()[0];
        assert!((mean_after_batch1 - 10.0).abs() < 0.01);

        // After batch 2 with alpha=0.3:
        // new_mean = 0.7 * 10 + 0.3 * 100 = 7 + 30 = 37
        scaler.partial_fit(&batch2, num_features).unwrap();

        let mean_after_batch2 = scaler.means()[0];
        assert!(
            (mean_after_batch2 - 37.0).abs() < 0.5,
            "Expected ~37, got {}",
            mean_after_batch2
        );
    }

    #[test]
    fn test_standard_scaler_ema_vs_cumulative() {
        let num_features = 1;

        // Batch 1: mean=10
        let batch1 = vec![8.0, 10.0, 12.0];
        // Batch 2: mean=100 (shifted distribution)
        let batch2 = vec![98.0, 100.0, 102.0];

        // Cumulative (no forget factor)
        let mut cumulative = StandardScaler::new();
        cumulative.partial_fit(&batch1, num_features).unwrap();
        cumulative.partial_fit(&batch2, num_features).unwrap();

        // EMA with high forget factor
        let mut ema = StandardScaler::with_forget_factor(0.5);
        ema.partial_fit(&batch1, num_features).unwrap();
        ema.partial_fit(&batch2, num_features).unwrap();

        // Cumulative mean: (10 + 100) / 2 = 55 (equal weight to all samples)
        let cumulative_mean = cumulative.means()[0];

        // EMA mean: 0.5 * 10 + 0.5 * 100 = 55 (with alpha=0.5)
        let ema_mean = ema.means()[0];

        // With alpha=0.5, both should be similar but EMA weights batch means, not sample means
        assert!((cumulative_mean - 55.0).abs() < 1.0);
        assert!((ema_mean - 55.0).abs() < 1.0);
    }

    #[test]
    fn test_standard_scaler_ema_adapts_to_drift() {
        let num_features = 1;

        // Start with mean=10
        let batch1 = vec![8.0, 10.0, 12.0];

        // Series of batches with drifting mean
        let batch2 = vec![28.0, 30.0, 32.0]; // mean=30
        let batch3 = vec![48.0, 50.0, 52.0]; // mean=50
        let batch4 = vec![68.0, 70.0, 72.0]; // mean=70
        let batch5 = vec![88.0, 90.0, 92.0]; // mean=90

        let mut scaler = StandardScaler::with_forget_factor(0.5); // 50% weight to new batch

        scaler.partial_fit(&batch1, num_features).unwrap();
        assert!((scaler.means()[0] - 10.0).abs() < 1.0);

        scaler.partial_fit(&batch2, num_features).unwrap();
        // 0.5 * 10 + 0.5 * 30 = 20
        assert!(
            (scaler.means()[0] - 20.0).abs() < 1.0,
            "Expected ~20, got {}",
            scaler.means()[0]
        );

        scaler.partial_fit(&batch3, num_features).unwrap();
        // 0.5 * 20 + 0.5 * 50 = 35
        assert!(
            (scaler.means()[0] - 35.0).abs() < 1.0,
            "Expected ~35, got {}",
            scaler.means()[0]
        );

        scaler.partial_fit(&batch4, num_features).unwrap();
        // 0.5 * 35 + 0.5 * 70 = 52.5
        assert!(
            (scaler.means()[0] - 52.5).abs() < 1.0,
            "Expected ~52.5, got {}",
            scaler.means()[0]
        );

        scaler.partial_fit(&batch5, num_features).unwrap();
        // 0.5 * 52.5 + 0.5 * 90 = 71.25
        assert!(
            (scaler.means()[0] - 71.25).abs() < 1.5,
            "Expected ~71.25, got {}",
            scaler.means()[0]
        );
    }

    #[test]
    fn test_standard_scaler_ema_variance_decay() {
        let num_features = 1;

        // Low variance batch
        let batch1 = vec![9.9, 10.0, 10.1]; // std ≈ 0.08

        // High variance batch
        let batch2 = vec![0.0, 10.0, 20.0]; // std ≈ 8.16

        let mut scaler = StandardScaler::with_forget_factor(0.3);

        scaler.partial_fit(&batch1, num_features).unwrap();
        let std_after_batch1 = scaler.stds()[0];
        assert!(
            std_after_batch1 < 1.0,
            "Std should be small after low-variance batch"
        );

        scaler.partial_fit(&batch2, num_features).unwrap();
        let std_after_batch2 = scaler.stds()[0];

        // Std should increase (EMA blend of low and high variance)
        assert!(
            std_after_batch2 > std_after_batch1,
            "Std should increase after high-variance batch"
        );
    }
}
