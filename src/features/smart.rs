//! Smart Feature Engineering
//!
//! Automatically infers optimal feature generation based on data characteristics
//! and model analysis. Acts as a "Smart Feature Engineer" that prescribes features.
//!
//! # Design Philosophy
//!
//! **Different models benefit from different features:**
//! - Linear models: Polynomial features (x², x³) help capture non-linearity
//! - Tree models: Interaction features (x_i * x_j) capture combinations trees struggle with
//! - LTT mode: Polynomial for linear phase, interactions for tree phase (on residuals)
//!
//! # Example
//!
//! ```ignore
//! use treeboost::analysis::{DataFrameProfile, DatasetAnalysis};
//! use treeboost::features::smart::{SmartFeatureEngine, SmartFeatureConfig};
//!
//! let profile = DataFrameProfile::analyze(&df, "target")?;
//! let analysis = DatasetAnalysis::analyze(&dataset);
//!
//! let plan = SmartFeatureEngine::infer(&profile, Some(&analysis));
//! println!("Feature Plan:\n{}", SmartFeatureEngine::summarize(&plan));
//! ```

use crate::analysis::profiler::{ColumnDataType, ColumnProfile, DataFrameProfile};
use crate::analysis::{DatasetAnalysis, PanelDataInfo};
use crate::defaults::features::smart as feature_defaults;
use crate::features::timeseries::RollingStat;
use std::collections::HashSet;

/// Time-series feature engineering plan
///
/// Defines lag, rolling, and EWMA features to generate for panel/time-series data.
/// All features are computed within groups to prevent data leakage.
///
/// # Serialization
///
/// This struct is serializable via serde to ensure reproducibility:
/// - Saved to `UniversalConfig.feature_plan` in `config.json`
/// - Used during inference to apply the same features to new data
/// - Format is stable for backward compatibility
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimeSeriesFeaturePlan {
    /// Name of the group/entity column (e.g., "stock_code")
    pub group_column: String,
    /// Name of the date/time column
    pub date_column: String,
    /// Columns to generate lag features for
    pub lag_columns: Vec<String>,
    /// Lag periods to use (e.g., [1, 3, 7, 14])
    pub lag_periods: Vec<usize>,
    /// Columns to generate rolling features for
    pub rolling_columns: Vec<String>,
    /// Rolling window sizes (e.g., [7, 14, 28])
    pub rolling_windows: Vec<usize>,
    /// Rolling statistics to compute
    pub rolling_stats: Vec<RollingStat>,
    /// Columns to generate EWMA features for
    pub ewma_columns: Vec<String>,
    /// EWMA alpha values (e.g., [0.1, 0.3])
    pub ewma_alphas: Vec<f32>,
    /// Columns to generate momentum/return features for
    pub momentum_columns: Vec<String>,
    /// Momentum periods (e.g., [1, 3, 7, 14] for 1-day, 3-day, 7-day, 14-day returns)
    pub momentum_periods: Vec<usize>,
    /// Estimated number of features to generate
    pub estimated_features: usize,
    /// Reasoning for feature selections
    pub reasoning: Vec<String>,
}

impl TimeSeriesFeaturePlan {
    /// Create an empty plan
    pub fn new(group_column: String, date_column: String) -> Self {
        Self {
            group_column,
            date_column,
            lag_columns: Vec::new(),
            lag_periods: Vec::new(),
            rolling_columns: Vec::new(),
            rolling_windows: Vec::new(),
            rolling_stats: Vec::new(),
            ewma_columns: Vec::new(),
            ewma_alphas: Vec::new(),
            momentum_columns: Vec::new(),
            momentum_periods: Vec::new(),
            estimated_features: 0,
            reasoning: Vec::new(),
        }
    }

    /// Check if plan is empty (no features to generate)
    pub fn is_empty(&self) -> bool {
        self.lag_columns.is_empty()
            && self.rolling_columns.is_empty()
            && self.ewma_columns.is_empty()
            && self.momentum_columns.is_empty()
    }

    /// Compute estimated feature count
    pub fn compute_estimated_features(&mut self) {
        let lag_count = self.lag_columns.len() * self.lag_periods.len();
        let rolling_count =
            self.rolling_columns.len() * self.rolling_windows.len() * self.rolling_stats.len();
        let ewma_count = self.ewma_columns.len() * self.ewma_alphas.len();
        let momentum_count = self.momentum_columns.len() * self.momentum_periods.len();

        self.estimated_features = lag_count + rolling_count + ewma_count + momentum_count;
    }
}

/// Feature generation plan
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeaturePlan {
    /// Columns to apply polynomial transforms (x², sqrt, log)
    pub polynomial_features: Vec<String>,
    /// Column pairs for ratio features (x_i / x_j)
    pub ratio_pairs: Vec<(String, String)>,
    /// Column pairs for interaction features (x_i * x_j)
    pub interaction_pairs: Vec<(String, String)>,
    /// DateTime columns for seasonal feature extraction
    pub time_features: Vec<(String, TimeFeatureType)>,
    /// Time-series feature plan (lag, rolling, EWMA) - group-aware
    pub timeseries_features: Option<TimeSeriesFeaturePlan>,
    /// Human-readable reasoning for decisions
    pub reasoning: Vec<String>,
}

impl FeaturePlan {
    /// Create an empty plan
    pub fn new() -> Self {
        Self {
            polynomial_features: Vec::new(),
            ratio_pairs: Vec::new(),
            interaction_pairs: Vec::new(),
            time_features: Vec::new(),
            timeseries_features: None,
            reasoning: Vec::new(),
        }
    }

    /// Check if plan is empty (no features to generate)
    pub fn is_empty(&self) -> bool {
        self.polynomial_features.is_empty()
            && self.ratio_pairs.is_empty()
            && self.interaction_pairs.is_empty()
            && self.time_features.is_empty()
            && self
                .timeseries_features
                .as_ref()
                .map(|ts| ts.is_empty())
                .unwrap_or(true)
    }

    /// Total number of features to be generated (estimate)
    pub fn estimated_feature_count(&self) -> usize {
        // Each polynomial column generates: square, sqrt (if positive), log (if positive)
        let poly_count = self.polynomial_features.len() * 2; // Conservative estimate
        let ratio_count = self.ratio_pairs.len();
        let interaction_count = self.interaction_pairs.len();
        let time_count = self.time_features.len() * 4; // Typical seasonal components
        let timeseries_count = self
            .timeseries_features
            .as_ref()
            .map(|ts| ts.estimated_features)
            .unwrap_or(0);

        poly_count + ratio_count + interaction_count + time_count + timeseries_count
    }

    /// Check if time-series features are planned
    pub fn has_timeseries_features(&self) -> bool {
        self.timeseries_features
            .as_ref()
            .map(|ts| !ts.is_empty())
            .unwrap_or(false)
    }
}

impl Default for FeaturePlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Time feature types to extract from DateTime columns
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TimeFeatureType {
    /// Hour of day (0-23)
    Hour,
    /// Day of week (0-6)
    DayOfWeek,
    /// Day of month (1-31)
    DayOfMonth,
    /// Month (1-12)
    Month,
    /// Year
    Year,
    /// Is weekend
    IsWeekend,
    /// Cyclical hour (sin/cos)
    CyclicalHour,
    /// Cyclical day of week
    CyclicalDayOfWeek,
    /// Cyclical month
    CyclicalMonth,
}

/// LTT-specific feature plan with separate phases
#[derive(Debug, Clone)]
pub struct LttFeaturePlan {
    /// Features for linear phase (polynomial focus)
    pub linear_features: FeaturePlan,
    /// Features for tree phase on residuals (interaction focus)
    pub tree_features: FeaturePlan,
    /// Features used by both phases
    pub shared_features: Vec<String>,
    /// Combined reasoning
    pub reasoning: Vec<String>,
}

impl LttFeaturePlan {
    /// Create empty LTT feature plan
    pub fn new() -> Self {
        Self {
            linear_features: FeaturePlan::new(),
            tree_features: FeaturePlan::new(),
            shared_features: Vec::new(),
            reasoning: Vec::new(),
        }
    }
}

impl Default for LttFeaturePlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for smart feature engineering
#[derive(Debug, Clone)]
pub struct SmartFeatureConfig {
    /// Enable polynomial feature generation
    pub enable_polynomial: bool,
    /// Enable ratio feature generation
    pub enable_ratios: bool,
    /// Enable interaction feature generation
    pub enable_interactions: bool,
    /// Enable time feature extraction
    pub enable_time_features: bool,
    /// Enable time-series feature generation (lag, rolling, EWMA)
    pub enable_timeseries: bool,
    /// Maximum new features to generate
    pub max_new_features: usize,
    /// Linear R² threshold below which to add interactions
    pub low_linear_r2_threshold: f32,
    /// Correlation threshold for ratio features
    pub ratio_correlation_threshold: f32,
    /// Top N features for polynomial generation
    pub top_n_polynomial: usize,
    /// Top N pairs for interaction generation
    pub top_n_interactions: usize,
    /// Top N columns for time-series features (by target correlation)
    pub top_n_timeseries: usize,
    /// Override group column detection (None = auto-detect)
    pub group_column: Option<String>,
    /// Override date column detection (None = auto-detect)
    pub date_column: Option<String>,
    /// Custom lag periods (None = use granularity-based defaults)
    pub custom_lag_periods: Option<Vec<usize>>,
    /// Custom rolling windows (None = use granularity-based defaults)
    pub custom_rolling_windows: Option<Vec<usize>>,
    /// Features to skip
    pub skip_features: HashSet<String>,
}

/// Presets for smart feature engineering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartFeaturePreset {
    /// All feature types enabled, 50 max.
    Standard,
    /// Polynomial + ratios only, 20 max.
    Minimal,
    /// All types, 100 max, lower thresholds.
    Aggressive,
}

impl Default for SmartFeatureConfig {
    fn default() -> Self {
        Self {
            enable_polynomial: true,
            enable_ratios: true,
            enable_interactions: true,
            enable_time_features: true,
            enable_timeseries: true, // Auto-enabled when panel data detected
            max_new_features: feature_defaults::DEFAULT_MAX_NEW_FEATURES,
            low_linear_r2_threshold: feature_defaults::LOW_LINEAR_R2_THRESHOLD,
            ratio_correlation_threshold: feature_defaults::RATIO_CORRELATION_THRESHOLD,
            top_n_polynomial: feature_defaults::TOP_N_POLYNOMIAL,
            top_n_interactions: feature_defaults::TOP_N_INTERACTIONS,
            top_n_timeseries: 10, // Default: top 10 columns by correlation
            group_column: None,
            date_column: None,
            custom_lag_periods: None,
            custom_rolling_windows: None,
            skip_features: HashSet::new(),
        }
    }
}

impl SmartFeatureConfig {
    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: SmartFeaturePreset) -> Self {
        match preset {
            SmartFeaturePreset::Standard => {}
            SmartFeaturePreset::Minimal => {
                self.enable_polynomial = true;
                self.enable_ratios = true;
                self.enable_interactions = false;
                self.enable_time_features = false;
                self.enable_timeseries = false;
                self.max_new_features = feature_defaults::MINIMAL_MAX_NEW_FEATURES;
            }
            SmartFeaturePreset::Aggressive => {
                self.enable_polynomial = true;
                self.enable_ratios = true;
                self.enable_interactions = true;
                self.enable_time_features = true;
                self.enable_timeseries = true;
                self.max_new_features = feature_defaults::AGGRESSIVE_MAX_NEW_FEATURES;
                self.low_linear_r2_threshold = feature_defaults::AGGRESSIVE_LOW_LINEAR_R2_THRESHOLD;
                self.ratio_correlation_threshold =
                    feature_defaults::AGGRESSIVE_RATIO_CORRELATION_THRESHOLD;
                self.top_n_polynomial = feature_defaults::AGGRESSIVE_TOP_N_POLYNOMIAL;
                self.top_n_interactions = feature_defaults::AGGRESSIVE_TOP_N_INTERACTIONS;
                self.top_n_timeseries = 15; // More columns for aggressive mode
            }
        }
        self
    }

    /// Set group column override
    pub fn with_group_column(mut self, col: impl Into<String>) -> Self {
        self.group_column = Some(col.into());
        self
    }

    /// Set date column override
    pub fn with_date_column(mut self, col: impl Into<String>) -> Self {
        self.date_column = Some(col.into());
        self
    }

    /// Set custom lag periods
    pub fn with_lag_periods(mut self, periods: Vec<usize>) -> Self {
        self.custom_lag_periods = Some(periods);
        self
    }

    /// Set custom rolling windows
    pub fn with_rolling_windows(mut self, windows: Vec<usize>) -> Self {
        self.custom_rolling_windows = Some(windows);
        self
    }
}

/// Smart Feature Engineering Engine
///
/// Analyzes data characteristics and model analysis to generate optimal features.
#[derive(Debug, Clone)]
pub struct SmartFeatureEngine {
    /// Configuration
    pub config: SmartFeatureConfig,
}

impl SmartFeatureEngine {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: SmartFeatureConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SmartFeatureConfig) -> Self {
        Self { config }
    }

    /// Infer optimal feature generation plan
    ///
    /// # Decision Matrix
    ///
    /// | Condition | Action |
    /// |-----------|--------|
    /// | Linear R² < 0.3 | Generate interactions (trees need help) |
    /// | Numeric + skewed | Add log/sqrt transforms |
    /// | DateTime column | Add cyclical (sin/cos) + components |
    /// | Correlated numerics (r > 0.5) | Add ratio features |
    /// | Too many features (>500) | Apply FeatureSelector |
    pub fn infer(profile: &DataFrameProfile, analysis: Option<&DatasetAnalysis>) -> FeaturePlan {
        let config = SmartFeatureConfig::default();
        Self::infer_with_config(profile, analysis, &config)
    }

    /// Infer with custom configuration
    pub fn infer_with_config(
        profile: &DataFrameProfile,
        analysis: Option<&DatasetAnalysis>,
        config: &SmartFeatureConfig,
    ) -> FeaturePlan {
        let mut plan = FeaturePlan::new();

        // Get numeric columns sorted by target correlation
        let mut numeric_cols: Vec<&ColumnProfile> = profile
            .columns
            .iter()
            .filter(|c| c.dtype == ColumnDataType::Numeric)
            .filter(|c| !config.skip_features.contains(&c.name))
            .collect();

        // Sort by target correlation (absolute value, descending)
        numeric_cols.sort_by(|a, b| {
            let corr_a = a.target_correlation.map(|c| c.abs()).unwrap_or(0.0);
            let corr_b = b.target_correlation.map(|c| c.abs()).unwrap_or(0.0);
            corr_b
                .partial_cmp(&corr_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 1. Polynomial features for top correlated columns
        if config.enable_polynomial {
            Self::add_polynomial_features(&mut plan, &numeric_cols, config);
        }

        // 2. Ratio features for correlated pairs
        if config.enable_ratios {
            Self::add_ratio_features(&mut plan, &numeric_cols, config);
        }

        // 3. Interaction features if linear R² is low
        if config.enable_interactions {
            let linear_r2 = analysis.map(|a| a.linear_r2).unwrap_or(0.0);
            if linear_r2 < config.low_linear_r2_threshold {
                Self::add_interaction_features(&mut plan, &numeric_cols, config);
                plan.reasoning.push(format!(
                    "Adding interactions: Linear R²={:.3} < {:.2} threshold",
                    linear_r2, config.low_linear_r2_threshold
                ));
            } else {
                plan.reasoning.push(format!(
                    "Skipping interactions: Linear R²={:.3} >= {:.2} threshold",
                    linear_r2, config.low_linear_r2_threshold
                ));
            }
        }

        // 4. Time features for DateTime columns
        if config.enable_time_features {
            Self::add_time_features(&mut plan, profile, config);
        }

        // Check feature count limit
        if plan.estimated_feature_count() > config.max_new_features {
            plan.reasoning.push(format!(
                "Warning: Estimated {} features exceeds max {} - consider reducing",
                plan.estimated_feature_count(),
                config.max_new_features
            ));
        }

        plan
    }

    /// Add polynomial features for top correlated numeric columns
    fn add_polynomial_features(
        plan: &mut FeaturePlan,
        numeric_cols: &[&ColumnProfile],
        config: &SmartFeatureConfig,
    ) {
        let top_n = config.top_n_polynomial.min(numeric_cols.len());

        for col in numeric_cols.iter().take(top_n) {
            // Skip if has negatives (can't do sqrt/log safely)
            if col.has_negative {
                plan.reasoning.push(format!(
                    "{}: Skip polynomial (has negative values)",
                    col.name
                ));
                continue;
            }

            plan.polynomial_features.push(col.name.clone());
            let corr = col.target_correlation.unwrap_or(0.0);
            plan.reasoning.push(format!(
                "{}: Add polynomial (x², sqrt, log) - correlation={:.3}",
                col.name, corr
            ));
        }
    }

    /// Add ratio features for correlated numeric pairs
    fn add_ratio_features(
        plan: &mut FeaturePlan,
        numeric_cols: &[&ColumnProfile],
        config: &SmartFeatureConfig,
    ) {
        // Find pairs with high correlation (both with target)
        let high_corr_cols: Vec<&ColumnProfile> = numeric_cols
            .iter()
            .filter(|c| {
                c.target_correlation
                    .map(|r| r.abs() > config.ratio_correlation_threshold)
                    .unwrap_or(false)
            })
            .copied()
            .collect();

        // Generate ratio pairs (avoid division by potentially zero columns)
        for (i, col_a) in high_corr_cols.iter().enumerate() {
            for col_b in high_corr_cols.iter().skip(i + 1) {
                // Check if denominator column has values away from zero
                if col_b.min.map(|v| v.abs() > 0.01).unwrap_or(false) {
                    plan.ratio_pairs
                        .push((col_a.name.clone(), col_b.name.clone()));
                    plan.reasoning.push(format!(
                        "Ratio: {} / {} (both highly correlated with target)",
                        col_a.name, col_b.name
                    ));

                    // Limit pairs
                    if plan.ratio_pairs.len() >= config.top_n_interactions {
                        break;
                    }
                }
            }
            if plan.ratio_pairs.len() >= config.top_n_interactions {
                break;
            }
        }
    }

    /// Add interaction features for top numeric pairs
    fn add_interaction_features(
        plan: &mut FeaturePlan,
        numeric_cols: &[&ColumnProfile],
        config: &SmartFeatureConfig,
    ) {
        let max_pairs = if numeric_cols.len() >= 2 {
            numeric_cols.len() * (numeric_cols.len() - 1) / 2
        } else {
            0
        };
        let top_n = config.top_n_interactions.min(max_pairs);
        let mut pair_count = 0;

        // Generate interaction pairs from top correlated columns
        for (i, col_a) in numeric_cols.iter().enumerate() {
            for col_b in numeric_cols.iter().skip(i + 1) {
                plan.interaction_pairs
                    .push((col_a.name.clone(), col_b.name.clone()));
                plan.reasoning.push(format!(
                    "Interaction: {} × {} (top correlated features)",
                    col_a.name, col_b.name
                ));

                pair_count += 1;
                if pair_count >= top_n {
                    break;
                }
            }
            if pair_count >= top_n {
                break;
            }
        }
    }

    /// Add time features for DateTime columns
    fn add_time_features(
        plan: &mut FeaturePlan,
        profile: &DataFrameProfile,
        _config: &SmartFeatureConfig,
    ) {
        for col in &profile.columns {
            if col.dtype == ColumnDataType::DateTime {
                // Add standard time components
                plan.time_features
                    .push((col.name.clone(), TimeFeatureType::Hour));
                plan.time_features
                    .push((col.name.clone(), TimeFeatureType::DayOfWeek));
                plan.time_features
                    .push((col.name.clone(), TimeFeatureType::Month));
                plan.time_features
                    .push((col.name.clone(), TimeFeatureType::IsWeekend));
                plan.time_features
                    .push((col.name.clone(), TimeFeatureType::CyclicalHour));
                plan.time_features
                    .push((col.name.clone(), TimeFeatureType::CyclicalDayOfWeek));

                plan.reasoning.push(format!(
                    "{}: Add time features (hour, day_of_week, month, is_weekend, cyclical)",
                    col.name
                ));
            }
        }
    }

    /// Infer feature plan with panel data awareness
    ///
    /// When panel data info is provided, this method also generates time-series
    /// features (lags, rolling, EWMA) that respect group boundaries.
    pub fn infer_with_panel(
        profile: &DataFrameProfile,
        analysis: Option<&DatasetAnalysis>,
        panel_info: Option<&PanelDataInfo>,
    ) -> FeaturePlan {
        let config = SmartFeatureConfig::default();
        Self::infer_with_panel_and_config(profile, analysis, panel_info, &config)
    }

    /// Infer feature plan with panel data awareness and custom configuration
    pub fn infer_with_panel_and_config(
        profile: &DataFrameProfile,
        analysis: Option<&DatasetAnalysis>,
        panel_info: Option<&PanelDataInfo>,
        config: &SmartFeatureConfig,
    ) -> FeaturePlan {
        // Start with standard feature inference
        let mut plan = Self::infer_with_config(profile, analysis, config);

        // Add time-series features if panel data detected and enabled
        if config.enable_timeseries {
            if let Some(info) = panel_info {
                let ts_plan = Self::plan_timeseries_features(profile, info, config);
                if !ts_plan.is_empty() {
                    plan.reasoning.push(format!(
                        "Panel data detected: {} groups by '{}', ordered by '{}'",
                        info.num_groups, info.group_column, info.date_column
                    ));
                    plan.reasoning.push(format!(
                        "Adding {} time-series features (lag, rolling, EWMA)",
                        ts_plan.estimated_features
                    ));
                    plan.timeseries_features = Some(ts_plan);
                }
            }
        }

        plan
    }

    /// Plan time-series features for panel data
    ///
    /// Selects top N numeric columns by target correlation and generates:
    /// - Lag features at granularity-appropriate periods
    /// - Rolling statistics (mean, std) at granularity-appropriate windows
    /// - EWMA features with multiple alpha values
    pub fn plan_timeseries_features(
        profile: &DataFrameProfile,
        panel_info: &PanelDataInfo,
        config: &SmartFeatureConfig,
    ) -> TimeSeriesFeaturePlan {
        let mut plan = TimeSeriesFeaturePlan::new(
            config
                .group_column
                .clone()
                .unwrap_or_else(|| panel_info.group_column.clone()),
            config
                .date_column
                .clone()
                .unwrap_or_else(|| panel_info.date_column.clone()),
        );

        // Get numeric columns sorted by target correlation (exclude group/date)
        let mut numeric_cols: Vec<&ColumnProfile> = profile
            .columns
            .iter()
            .filter(|c| c.dtype == ColumnDataType::Numeric)
            .filter(|c| !config.skip_features.contains(&c.name))
            .filter(|c| c.name != panel_info.group_column && c.name != panel_info.date_column)
            .collect();

        numeric_cols.sort_by(|a, b| {
            let corr_a = a.target_correlation.map(|c| c.abs()).unwrap_or(0.0);
            let corr_b = b.target_correlation.map(|c| c.abs()).unwrap_or(0.0);
            corr_b
                .partial_cmp(&corr_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top N columns
        let top_cols: Vec<String> = numeric_cols
            .iter()
            .take(config.top_n_timeseries)
            .map(|c| c.name.clone())
            .collect();

        if top_cols.is_empty() {
            plan.reasoning
                .push("No numeric columns available for time-series features".to_string());
            return plan;
        }

        // Configure lag periods based on granularity or custom
        plan.lag_periods = config
            .custom_lag_periods
            .clone()
            .unwrap_or_else(|| panel_info.time_granularity.suggested_lag_periods());

        // Configure rolling windows based on granularity or custom
        plan.rolling_windows = config
            .custom_rolling_windows
            .clone()
            .unwrap_or_else(|| panel_info.time_granularity.suggested_rolling_windows());

        // Default rolling stats: Mean, Std, Min, Max (captures more signal)
        plan.rolling_stats = vec![
            RollingStat::Mean,
            RollingStat::Std,
            RollingStat::Min,
            RollingStat::Max,
        ];

        // EWMA alphas for signal capture across different timescales
        plan.ewma_alphas = vec![0.1, 0.2, 0.3, 0.5];

        // Momentum periods - same as lag periods (percentage changes over these periods)
        plan.momentum_periods = plan.lag_periods.clone();

        // Assign columns to each feature type
        plan.lag_columns = top_cols.clone();
        plan.rolling_columns = top_cols.clone();
        plan.ewma_columns = top_cols.clone();
        plan.momentum_columns = top_cols;

        // Compute estimated features
        plan.compute_estimated_features();

        // Add reasoning
        plan.reasoning.push(format!(
            "Time granularity: {} -> lag periods {:?}, rolling windows {:?}",
            panel_info.time_granularity, plan.lag_periods, plan.rolling_windows
        ));
        plan.reasoning.push(format!(
            "Selected top {} columns by target correlation for time-series features",
            plan.lag_columns.len()
        ));
        plan.reasoning.push(format!(
            "Estimated {} time-series features ({} lag + {} rolling + {} EWMA + {} momentum)",
            plan.estimated_features,
            plan.lag_columns.len() * plan.lag_periods.len(),
            plan.rolling_columns.len() * plan.rolling_windows.len() * plan.rolling_stats.len(),
            plan.ewma_columns.len() * plan.ewma_alphas.len(),
            plan.momentum_columns.len() * plan.momentum_periods.len()
        ));

        plan
    }

    /// Create separate feature plans for LTT mode
    ///
    /// # LTT Feature Matrix
    ///
    /// | Phase | Feature Type | When to Add |
    /// |-------|-------------|-------------|
    /// | Linear | Polynomial (x², x³) | Top 5 correlated features |
    /// | Linear | Log/sqrt transforms | Skewed positives |
    /// | Linear | Interaction terms | Only if linear R² < 0.5 |
    /// | Tree | Interaction (x_i * x_j) | Top 10 feature pairs |
    /// | Tree | Ratios (x_i / x_j) | Correlated pairs (scale-free) |
    /// | Tree | NO polynomial | Trees capture non-linearity natively |
    pub fn infer_ltt(
        profile: &DataFrameProfile,
        analysis: Option<&DatasetAnalysis>,
    ) -> LttFeaturePlan {
        let config = SmartFeatureConfig::default();
        Self::infer_ltt_with_config(profile, analysis, &config)
    }

    /// Create LTT feature plans with custom configuration
    pub fn infer_ltt_with_config(
        profile: &DataFrameProfile,
        analysis: Option<&DatasetAnalysis>,
        config: &SmartFeatureConfig,
    ) -> LttFeaturePlan {
        let mut ltt_plan = LttFeaturePlan::new();

        ltt_plan
            .reasoning
            .push("=== LTT Dual-Phase Feature Engineering ===".to_string());
        ltt_plan
            .reasoning
            .push("Phase 1 (Linear): Polynomial features extend linear model's reach".to_string());
        ltt_plan.reasoning.push(
            "Phase 2 (Tree): Interaction features capture what trees struggle with".to_string(),
        );

        // Get numeric columns sorted by target correlation
        let mut numeric_cols: Vec<&ColumnProfile> = profile
            .columns
            .iter()
            .filter(|c| c.dtype == ColumnDataType::Numeric)
            .filter(|c| !config.skip_features.contains(&c.name))
            .collect();

        numeric_cols.sort_by(|a, b| {
            let corr_a = a.target_correlation.map(|c| c.abs()).unwrap_or(0.0);
            let corr_b = b.target_correlation.map(|c| c.abs()).unwrap_or(0.0);
            corr_b
                .partial_cmp(&corr_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // === LINEAR PHASE FEATURES ===
        // Polynomial features help linear models capture non-linearity
        for col in numeric_cols.iter().take(config.top_n_polynomial) {
            if !col.has_negative {
                ltt_plan
                    .linear_features
                    .polynomial_features
                    .push(col.name.clone());
                ltt_plan.linear_features.reasoning.push(format!(
                    "{}: Polynomial for linear phase (correlation={:.3})",
                    col.name,
                    col.target_correlation.unwrap_or(0.0)
                ));
            }
        }

        // Only add interactions to linear if R² is very low
        let linear_r2 = analysis.map(|a| a.linear_r2).unwrap_or(0.0);
        if linear_r2 < 0.5 && config.enable_interactions {
            // Limited interactions for linear
            for (i, col_a) in numeric_cols.iter().enumerate().take(3) {
                for col_b in numeric_cols.iter().skip(i + 1).take(3) {
                    ltt_plan
                        .linear_features
                        .interaction_pairs
                        .push((col_a.name.clone(), col_b.name.clone()));
                }
            }
            ltt_plan.linear_features.reasoning.push(format!(
                "Adding limited interactions: Linear R²={:.3} < 0.5",
                linear_r2
            ));
        }

        // === TREE PHASE FEATURES (on residuals) ===
        // Trees benefit from interactions they can't easily learn
        // NO polynomial - trees handle non-linearity natively

        // Interaction features
        let top_n = config
            .top_n_interactions
            .min(numeric_cols.len() * (numeric_cols.len() - 1) / 2);
        let mut pair_count = 0;
        for (i, col_a) in numeric_cols.iter().enumerate() {
            for col_b in numeric_cols.iter().skip(i + 1) {
                ltt_plan
                    .tree_features
                    .interaction_pairs
                    .push((col_a.name.clone(), col_b.name.clone()));
                pair_count += 1;
                if pair_count >= top_n {
                    break;
                }
            }
            if pair_count >= top_n {
                break;
            }
        }
        ltt_plan.tree_features.reasoning.push(format!(
            "Added {} interaction pairs for tree phase",
            ltt_plan.tree_features.interaction_pairs.len()
        ));

        // Ratio features (scale-free, good for residuals)
        let high_corr_cols: Vec<&ColumnProfile> = numeric_cols
            .iter()
            .filter(|c| {
                c.target_correlation
                    .map(|r| r.abs() > config.ratio_correlation_threshold)
                    .unwrap_or(false)
            })
            .copied()
            .collect();

        for (i, col_a) in high_corr_cols.iter().enumerate().take(5) {
            for col_b in high_corr_cols.iter().skip(i + 1).take(5) {
                if col_b.min.map(|v| v.abs() > 0.01).unwrap_or(false) {
                    ltt_plan
                        .tree_features
                        .ratio_pairs
                        .push((col_a.name.clone(), col_b.name.clone()));
                }
            }
        }
        if !ltt_plan.tree_features.ratio_pairs.is_empty() {
            ltt_plan.tree_features.reasoning.push(format!(
                "Added {} ratio pairs for tree phase (scale-free)",
                ltt_plan.tree_features.ratio_pairs.len()
            ));
        }

        // Time features are shared
        for col in &profile.columns {
            if col.dtype == ColumnDataType::DateTime {
                ltt_plan.shared_features.push(col.name.clone());
                ltt_plan.reasoning.push(format!(
                    "{}: DateTime features shared between phases",
                    col.name
                ));
            }
        }

        ltt_plan
    }

    /// Generate human-readable summary of feature plan
    pub fn summarize(plan: &FeaturePlan) -> String {
        let mut summary = String::new();

        summary.push_str("Feature Generation Plan:\n");
        summary.push_str(&format!(
            "  Polynomial features: {}\n",
            plan.polynomial_features.len()
        ));
        summary.push_str(&format!("  Ratio pairs: {}\n", plan.ratio_pairs.len()));

        // Time-series features summary
        if let Some(ref ts) = plan.timeseries_features {
            summary.push_str(&format!(
                "  Time-series features: {} (group='{}', date='{}')\n",
                ts.estimated_features, ts.group_column, ts.date_column
            ));
            summary.push_str(&format!(
                "    - Lag: {} columns × {} periods\n",
                ts.lag_columns.len(),
                ts.lag_periods.len()
            ));
            summary.push_str(&format!(
                "    - Rolling: {} columns × {} windows × {} stats\n",
                ts.rolling_columns.len(),
                ts.rolling_windows.len(),
                ts.rolling_stats.len()
            ));
            summary.push_str(&format!(
                "    - EWMA: {} columns × {} alphas\n",
                ts.ewma_columns.len(),
                ts.ewma_alphas.len()
            ));
        }
        summary.push_str(&format!(
            "  Interaction pairs: {}\n",
            plan.interaction_pairs.len()
        ));
        summary.push_str(&format!("  Time features: {}\n", plan.time_features.len()));
        summary.push_str(&format!(
            "  Estimated total: {} new features\n",
            plan.estimated_feature_count()
        ));

        if !plan.reasoning.is_empty() {
            summary.push_str("\nDecisions:\n");
            for reason in &plan.reasoning {
                summary.push_str(&format!("  - {}\n", reason));
            }
        }

        summary
    }

    /// Generate summary for LTT plan
    pub fn summarize_ltt(plan: &LttFeaturePlan) -> String {
        let mut summary = String::new();

        summary.push_str("=== LTT Feature Engineering Plan ===\n\n");

        summary.push_str("Linear Phase Features:\n");
        summary.push_str(&format!(
            "  Polynomial: {}\n",
            plan.linear_features.polynomial_features.len()
        ));
        summary.push_str(&format!(
            "  Interactions: {}\n",
            plan.linear_features.interaction_pairs.len()
        ));

        summary.push_str("\nTree Phase Features:\n");
        summary.push_str(&format!(
            "  Interactions: {}\n",
            plan.tree_features.interaction_pairs.len()
        ));
        summary.push_str(&format!(
            "  Ratios: {}\n",
            plan.tree_features.ratio_pairs.len()
        ));

        summary.push_str(&format!(
            "\nShared Features: {}\n",
            plan.shared_features.len()
        ));

        if !plan.reasoning.is_empty() {
            summary.push_str("\nDecisions:\n");
            for reason in &plan.reasoning {
                summary.push_str(&format!("  - {}\n", reason));
            }
        }

        summary
    }
}

impl Default for SmartFeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_profile() -> DataFrameProfile {
        let df = DataFrame::new(
            5,
            vec![
                Series::new("feature1".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
                Series::new("feature2".into(), vec![2.0f64, 4.0, 6.0, 8.0, 10.0]).into(),
                Series::new("feature3".into(), vec![5.0f64, 4.0, 3.0, 2.0, 1.0]).into(),
                Series::new("target".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
            ],
        )
        .unwrap();

        DataFrameProfile::analyze(&df, "target").unwrap()
    }

    #[test]
    fn test_infer_feature_plan() {
        let profile = create_test_profile();
        let plan = SmartFeatureEngine::infer(&profile, None);

        // Should have polynomial features for top correlated
        assert!(!plan.polynomial_features.is_empty());
    }

    #[test]
    fn test_infer_ltt_plan() {
        let profile = create_test_profile();
        let ltt_plan = SmartFeatureEngine::infer_ltt(&profile, None);

        // Linear should have polynomial (no interactions without low R²)
        assert!(!ltt_plan.linear_features.polynomial_features.is_empty());

        // Tree should have interactions
        assert!(!ltt_plan.tree_features.interaction_pairs.is_empty());

        // Tree should NOT have polynomial
        assert!(ltt_plan.tree_features.polynomial_features.is_empty());
    }

    #[test]
    fn test_feature_plan_estimation() {
        let mut plan = FeaturePlan::new();
        plan.polynomial_features.push("f1".to_string());
        plan.polynomial_features.push("f2".to_string());
        plan.interaction_pairs
            .push(("f1".to_string(), "f2".to_string()));

        // 2 polynomial columns * 2 features each + 1 interaction = 5
        assert!(plan.estimated_feature_count() >= 4);
    }

    #[test]
    fn test_skip_negative_polynomial() {
        // Create profile with negative values
        let df = DataFrame::new(
            5,
            vec![
                Series::new(
                    "negative_feature".into(),
                    vec![-1.0f64, -2.0, 3.0, 4.0, 5.0],
                )
                .into(),
                Series::new("positive_feature".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
                Series::new("target".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
            ],
        )
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();
        let plan = SmartFeatureEngine::infer(&profile, None);

        // Should NOT have negative feature in polynomial (can't do sqrt/log)
        assert!(!plan
            .polynomial_features
            .contains(&"negative_feature".to_string()));
    }

    #[test]
    fn test_timeseries_feature_plan() {
        use crate::analysis::{PanelDataInfo, TimeGranularity};

        let profile = create_test_profile();

        // Create mock panel data info
        let panel_info = PanelDataInfo {
            group_column: "group".to_string(),
            date_column: "date".to_string(),
            num_groups: 5,
            avg_observations_per_group: 100.0,
            min_observations_per_group: 80,
            max_observations_per_group: 120,
            confidence: 0.9,
            time_granularity: TimeGranularity::Daily,
        };

        let ts_plan = SmartFeatureEngine::plan_timeseries_features(
            &profile,
            &panel_info,
            &SmartFeatureConfig::default(),
        );

        // Should have generated time-series features
        assert!(!ts_plan.is_empty());

        // Should have daily-appropriate lag periods
        assert!(ts_plan.lag_periods.contains(&1));
        assert!(ts_plan.lag_periods.contains(&7));

        // Should have daily-appropriate rolling windows
        assert!(ts_plan.rolling_windows.contains(&7));
        assert!(ts_plan.rolling_windows.contains(&14));

        // Should have estimated features count
        assert!(ts_plan.estimated_features > 0);

        // Should use the correct group/date columns
        assert_eq!(ts_plan.group_column, "group");
        assert_eq!(ts_plan.date_column, "date");
    }

    #[test]
    fn test_infer_with_panel() {
        use crate::analysis::{PanelDataInfo, TimeGranularity};

        let profile = create_test_profile();

        // Create mock panel data info
        let panel_info = PanelDataInfo {
            group_column: "group".to_string(),
            date_column: "date".to_string(),
            num_groups: 5,
            avg_observations_per_group: 100.0,
            min_observations_per_group: 80,
            max_observations_per_group: 120,
            confidence: 0.9,
            time_granularity: TimeGranularity::Daily,
        };

        let plan = SmartFeatureEngine::infer_with_panel(&profile, None, Some(&panel_info));

        // Should have time-series features
        assert!(plan.has_timeseries_features());

        // Should also have regular features
        assert!(!plan.polynomial_features.is_empty());

        // Check that time-series features count is in the total
        let ts_count = plan
            .timeseries_features
            .as_ref()
            .map(|ts| ts.estimated_features)
            .unwrap_or(0);
        assert!(ts_count > 0);
        assert!(plan.estimated_feature_count() > ts_count);
    }

    #[test]
    fn test_timeseries_feature_plan_empty_without_numerics() {
        use crate::analysis::{PanelDataInfo, TimeGranularity};

        // Profile with only categorical (no numerics)
        let df = DataFrame::new(
            5,
            vec![
                Series::new("cat1".into(), vec!["a", "b", "c", "d", "e"]).into(),
                Series::new("target".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
            ],
        )
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();

        let panel_info = PanelDataInfo {
            group_column: "group".to_string(),
            date_column: "date".to_string(),
            num_groups: 5,
            avg_observations_per_group: 100.0,
            min_observations_per_group: 80,
            max_observations_per_group: 120,
            confidence: 0.9,
            time_granularity: TimeGranularity::Daily,
        };

        let ts_plan = SmartFeatureEngine::plan_timeseries_features(
            &profile,
            &panel_info,
            &SmartFeatureConfig::default(),
        );

        // Should be empty (no numeric columns)
        assert!(ts_plan.is_empty());
    }
}
