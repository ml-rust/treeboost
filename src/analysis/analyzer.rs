//! Dataset analysis and intelligent mode recommendation
//!
//! The brain of TreeBoost's automatic mode selection.

use crate::dataset::BinnedDataset;
use crate::defaults::analysis as analysis_defaults;
use crate::model::BoostingMode;
use crate::Result;
use polars::prelude::*;

use super::probes::{run_combined_probe, CombinedProbeResult};
use super::stats::{
    compute_correlation, compute_monotonicity, detect_discrete_target, estimate_noise_floor,
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for dataset analysis
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisConfig {
    /// Maximum rows to sample for analysis (for speed)
    pub max_sample_rows: usize,

    /// Maximum iterations for linear probe
    pub linear_max_iter: usize,

    /// Maximum depth for tree probe
    pub tree_max_depth: usize,

    /// Number of top features to analyze in detail
    pub top_features_to_analyze: usize,

    /// Random seed for sampling
    pub seed: u64,
}

/// Presets for analysis configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisPreset {
    Fast,
    Standard,
    Thorough,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_sample_rows: analysis_defaults::DEFAULT_MAX_SAMPLE_ROWS,
            linear_max_iter: analysis_defaults::DEFAULT_LINEAR_MAX_ITER,
            tree_max_depth: analysis_defaults::DEFAULT_TREE_MAX_DEPTH,
            top_features_to_analyze: analysis_defaults::DEFAULT_TOP_FEATURES,
            seed: analysis_defaults::DEFAULT_ANALYSIS_SEED,
        }
    }
}

impl AnalysisConfig {
    /// Create config from preset.
    pub fn with_preset(preset: AnalysisPreset) -> Self {
        match preset {
            AnalysisPreset::Fast => Self::fast(),
            AnalysisPreset::Standard => Self::default(),
            AnalysisPreset::Thorough => Self::thorough(),
        }
    }
    pub fn fast() -> Self {
        Self {
            max_sample_rows: analysis_defaults::FAST_MAX_SAMPLE_ROWS,
            linear_max_iter: analysis_defaults::FAST_LINEAR_MAX_ITER,
            tree_max_depth: analysis_defaults::FAST_TREE_MAX_DEPTH,
            top_features_to_analyze: analysis_defaults::FAST_TOP_FEATURES,
            seed: analysis_defaults::DEFAULT_ANALYSIS_SEED,
        }
    }

    pub fn thorough() -> Self {
        Self {
            max_sample_rows: analysis_defaults::THOROUGH_MAX_SAMPLE_ROWS,
            linear_max_iter: analysis_defaults::THOROUGH_LINEAR_MAX_ITER,
            tree_max_depth: analysis_defaults::THOROUGH_TREE_MAX_DEPTH,
            top_features_to_analyze: analysis_defaults::THOROUGH_TOP_FEATURES,
            seed: analysis_defaults::DEFAULT_ANALYSIS_SEED,
        }
    }
}

// =============================================================================
// Confidence Level
// =============================================================================

/// Confidence level in the recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Very clear signal - highly recommend this mode
    High,
    /// Reasonable signal - this mode is likely best
    Medium,
    /// Weak signal - consider validating with CV
    Low,
}

impl Confidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            Confidence::High => "High",
            Confidence::Medium => "Medium",
            Confidence::Low => "Low",
        }
    }

    pub fn as_bar(&self) -> &'static str {
        match self {
            Confidence::High => "████████████████████",
            Confidence::Medium => "████████████░░░░░░░░",
            Confidence::Low => "████████░░░░░░░░░░░░",
        }
    }

    /// Downgrade confidence from High to Medium, keep Medium and Low unchanged
    pub fn downgrade_to_medium(&self) -> Self {
        match self {
            Confidence::High => Confidence::Medium,
            other => *other,
        }
    }
}

// =============================================================================
// Recommendation
// =============================================================================

/// Mode recommendation with reasoning
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// The recommended boosting mode
    pub mode: BoostingMode,

    /// Confidence in this recommendation
    pub confidence: Confidence,

    /// Human-readable explanation of why this mode was chosen
    pub reasoning: String,

    /// Alternative modes the user might consider
    pub alternatives: Vec<(BoostingMode, String)>,

    /// Numeric score for each mode (higher = better fit for this data)
    pub mode_scores: ModeScores,
}

/// Numeric scores for each mode based on data characteristics
#[derive(Debug, Clone, Default)]
pub struct ModeScores {
    pub pure_tree: f32,
    pub linear_then_tree: f32,
    pub random_forest: f32,
}

impl ModeScores {
    pub fn best_mode(&self) -> BoostingMode {
        if self.linear_then_tree >= self.pure_tree && self.linear_then_tree >= self.random_forest {
            BoostingMode::LinearThenTree
        } else if self.random_forest >= self.pure_tree {
            BoostingMode::RandomForest
        } else {
            BoostingMode::PureTree
        }
    }

    pub fn best_score(&self) -> f32 {
        self.pure_tree
            .max(self.linear_then_tree)
            .max(self.random_forest)
    }

    pub fn score_gap(&self) -> f32 {
        let best = self.best_score();
        let second = if best == self.pure_tree {
            self.linear_then_tree.max(self.random_forest)
        } else if best == self.linear_then_tree {
            self.pure_tree.max(self.random_forest)
        } else {
            self.pure_tree.max(self.linear_then_tree)
        };
        best - second
    }
}

// =============================================================================
// Dataset Analysis
// =============================================================================

/// Complete analysis of a dataset for mode selection
#[derive(Debug, Clone)]
pub struct DatasetAnalysis {
    // --- Dataset Info ---
    /// Number of rows in the dataset
    pub num_rows: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of categorical features
    pub num_categorical: usize,
    /// Number of numeric features
    pub num_numeric: usize,

    // --- Linear Signal ---
    /// R² from quick linear regression (0-1)
    /// High value indicates strong linear signal
    pub linear_r2: f32,
    /// Top feature correlations with target (absolute values)
    pub top_correlations: Vec<(usize, f32)>,
    /// Average monotonicity score across features (0-1)
    pub avg_monotonicity: f32,

    // --- Non-linear Structure ---
    /// R² of tree on linear residuals (0-1)
    /// High value indicates trees can capture what linear missed
    pub tree_gain: f32,
    /// Relative MSE improvement from adding trees
    pub tree_relative_improvement: f32,
    /// Combined R² (linear + tree)
    pub combined_r2: f32,

    // --- Data Characteristics ---
    /// Ratio of categorical features (0-1)
    pub categorical_ratio: f32,
    /// Estimated noise floor (0-1)
    /// High value indicates irreducible error
    pub noise_floor: f32,
    /// Is target likely discrete (classification)?
    pub target_is_discrete: bool,
    /// Number of unique target values (if discrete)
    pub target_unique_values: usize,

    // --- Derived Scores ---
    /// Mode-specific scores
    pub mode_scores: ModeScores,
    /// The recommendation
    pub recommendation: Recommendation,

    // --- Probe Results (for detailed inspection) ---
    /// Raw probe results (optional, for debugging)
    pub probe_result: Option<CombinedProbeResult>,
}

impl DatasetAnalysis {
    /// Analyze a raw DataFrame and produce mode recommendation (PREFERRED)
    ///
    /// This analyzes the raw data before binning, preserving numeric vs categorical
    /// distinctions and providing accurate mode selection.
    /// This is the DEFAULT and RECOMMENDED entry point.
    pub fn analyze_from_dataframe(
        df: &DataFrame,
        target_col: &str,
        feature_cols: Option<&[&str]>,
    ) -> Result<Self> {
        Self::analyze_from_dataframe_with_config(
            df,
            target_col,
            feature_cols,
            AnalysisConfig::default(),
        )
    }

    /// Analyze a raw DataFrame with custom configuration
    pub fn analyze_from_dataframe_with_config(
        df: &DataFrame,
        target_col: &str,
        feature_cols: Option<&[&str]>,
        config: AnalysisConfig,
    ) -> Result<Self> {
        // Extract target column
        let target_series = df.column(target_col)?;
        let targets: Vec<f32> = target_series
            .cast(&DataType::Float32)?
            .f32()?
            .iter()
            .map(|v| v.unwrap_or(f32::NAN))
            .collect();

        // Determine which features to use
        let feature_names: Vec<String> = if let Some(cols) = feature_cols {
            cols.iter().map(|s| s.to_string()).collect()
        } else {
            df.get_column_names()
                .iter()
                .filter(|name| name.as_str() != target_col)
                .map(|s| s.to_string())
                .collect()
        };

        let num_rows = df.height();
        let num_features = feature_names.len();

        // Extract features as f32 arrays
        let mut raw_features = vec![f32::NAN; num_rows * num_features];
        let mut num_numeric = 0;
        let mut num_categorical = 0;

        for (f_idx, fname) in feature_names.iter().enumerate() {
            let col = df.column(fname)?;
            let dtype = col.dtype();

            // Check actual data type
            let is_numeric = matches!(
                dtype,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Float32
                    | DataType::Float64
            );

            if is_numeric {
                num_numeric += 1;
                // Safe to cast numeric types
                if let Ok(numeric_series) = col.cast(&DataType::Float32) {
                    if let Ok(vals) = numeric_series.f32() {
                        for (r_idx, val) in vals.iter().enumerate() {
                            raw_features[r_idx * num_features + f_idx] = val.unwrap_or(f32::NAN);
                        }
                    }
                }
            } else {
                // Treat as categorical (String, Categorical, etc.)
                num_categorical += 1;
                if let Ok(str_series) = col.str() {
                    use std::collections::HashMap;
                    let mut category_map: HashMap<&str, f32> = HashMap::new();
                    let mut next_code = 0f32;

                    for (r_idx, val) in str_series.iter().enumerate() {
                        if let Some(s) = val {
                            let code = *category_map.entry(s).or_insert_with(|| {
                                let c = next_code;
                                next_code += 1.0;
                                c
                            });
                            raw_features[r_idx * num_features + f_idx] = code;
                        } else {
                            raw_features[r_idx * num_features + f_idx] = -1.0;
                        }
                    }
                }
            }
        }

        let categorical_ratio = if num_features > 0 {
            num_categorical as f32 / num_features as f32
        } else {
            0.0
        };

        // --- Sample indices for efficiency ---
        let sample_indices: Option<Vec<usize>> = if num_rows > config.max_sample_rows {
            use rand::RngExt;
            use rand::SeedableRng;

            // OPTIMIZED: Reservoir sampling instead of shuffle
            let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
            let mut indices: Vec<usize> = Vec::with_capacity(config.max_sample_rows);

            // Reservoir sampling: uniform random sample in single pass
            for i in 0..config.max_sample_rows.min(num_rows) {
                indices.push(i);
            }

            for i in config.max_sample_rows..num_rows {
                let j = rng.random_range(0..=i);
                if j < config.max_sample_rows {
                    indices[j] = i;
                }
            }

            indices.sort(); // Keep sorted for cache efficiency
            Some(indices)
        } else {
            None
        };

        let _sample_refs = sample_indices.as_deref();

        // Extract sample for analysis
        let (sample_features, sample_targets) = if let Some(indices) = &sample_indices {
            let mut sf = vec![f32::NAN; indices.len() * num_features];
            let st: Vec<f32> = indices.iter().map(|&i| targets[i]).collect();
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                for f in 0..num_features {
                    sf[new_idx * num_features + f] = raw_features[old_idx * num_features + f];
                }
            }
            (sf, st)
        } else {
            (raw_features.clone(), targets.clone())
        };

        let num_samples = sample_targets.len();

        // --- Compute correlations on raw data (filtering NaN) ---
        let mut correlations: Vec<(usize, f32)> = Vec::with_capacity(num_features);
        for f in 0..num_features.min(config.top_features_to_analyze) {
            // Extract feature, removing NaN values
            let mut feature_col: Vec<f32> = Vec::new();
            let mut target_col: Vec<f32> = Vec::new();

            for r in 0..num_samples {
                let feat_val = sample_features[r * num_features + f];
                let targ_val = sample_targets[r];
                // Only include rows where both feature and target are not NaN
                if !feat_val.is_nan() && !targ_val.is_nan() {
                    feature_col.push(feat_val);
                    target_col.push(targ_val);
                }
            }

            // Only compute correlation if we have enough valid samples
            if !feature_col.is_empty() && feature_col.len() > 1 {
                // Temporarily use target_col for correlation computation
                let corr = if !feature_col.is_empty() {
                    // Compute Pearson correlation on clean data
                    let n = feature_col.len() as f32;
                    let mean_x = feature_col.iter().sum::<f32>() / n;
                    let mean_y = target_col.iter().sum::<f32>() / n;

                    let mut num = 0.0f32;
                    let mut sum_x2 = 0.0f32;
                    let mut sum_y2 = 0.0f32;

                    for (x, y) in feature_col.iter().zip(target_col.iter()) {
                        let dx = x - mean_x;
                        let dy = y - mean_y;
                        num += dx * dy;
                        sum_x2 += dx * dx;
                        sum_y2 += dy * dy;
                    }

                    let denom = (sum_x2 * sum_y2).sqrt();
                    if denom > 0.0 {
                        (num / denom).abs()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                if !corr.is_nan() && corr > 0.0 {
                    correlations.push((f, corr));
                }
            }
        }
        correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_correlations: Vec<(usize, f32)> = correlations.into_iter().take(10).collect();
        let top_correlation = if !top_correlations.is_empty() {
            top_correlations.first().map(|(_, c)| *c).unwrap_or(0.0)
        } else {
            0.0
        };

        // --- Monotonicity on raw data ---
        let mut monotonicity_sum = 0.0f32;
        let features_to_check = num_features.min(20);
        for f in 0..features_to_check {
            let feature_col: Vec<f32> = (0..num_samples)
                .map(|r| sample_features[r * num_features + f])
                .collect();
            monotonicity_sum += compute_monotonicity(&feature_col, &sample_targets);
        }
        let avg_monotonicity = if features_to_check > 0 {
            monotonicity_sum / features_to_check as f32
        } else {
            0.5
        };

        // --- Noise estimation on raw data ---
        let best_feature_idx = top_correlations.first().map(|(idx, _)| *idx).unwrap_or(0);
        let noise_floor = estimate_noise_floor(
            &sample_features,
            &sample_targets,
            num_features,
            best_feature_idx,
        );

        // --- Target analysis ---
        let (target_is_discrete, target_unique_values) = detect_discrete_target(&targets);

        // --- Simple linear R² estimation on raw data ---
        // For raw data, we can estimate linear R² directly from correlations
        // Use the sum of squared correlations as a proxy for multivariate R²
        let linear_r2_from_corr = top_correlations
            .iter()
            .take(5)
            .map(|(_, c)| c * c)
            .sum::<f32>()
            .min(1.0);

        // For tree gain, estimate as residual variance if linear had captured the signal
        let tree_gain = if linear_r2_from_corr > 0.3 {
            (1.0 - linear_r2_from_corr).sqrt() * 0.5 // Rough estimate
        } else {
            0.43 // Conservative default for data with weak linear signal
        };

        // --- Compute mode scores ---
        let characteristics = DataCharacteristics {
            linear_r2: linear_r2_from_corr,
            tree_gain,
            tree_relative_improvement: tree_gain,
            categorical_ratio,
            noise_floor,
            avg_monotonicity,
            num_features,
            top_correlation,
        };
        let mode_scores = compute_mode_scores(&characteristics);

        // --- Generate recommendation ---
        let recommendation = generate_recommendation(
            &mode_scores,
            linear_r2_from_corr,
            tree_gain,
            categorical_ratio,
            noise_floor,
            avg_monotonicity,
        );

        // Combined R² estimate
        let combined_r2 = combined_r2(linear_r2_from_corr, tree_gain);

        Ok(DatasetAnalysis {
            num_rows,
            num_features,
            num_numeric,
            num_categorical,
            categorical_ratio,
            top_correlations,
            avg_monotonicity,
            noise_floor,
            linear_r2: linear_r2_from_corr,
            tree_gain,
            tree_relative_improvement: tree_gain,
            combined_r2,
            target_is_discrete,
            target_unique_values,
            mode_scores,
            recommendation,
            probe_result: None, // No detailed probes for raw data analysis
        })
    }

    /// Analyze a binned dataset and produce mode recommendation (LEGACY)
    ///
    /// This analyzes binned data. Prefer `analyze_from_dataframe` for raw data analysis.
    pub fn analyze(dataset: &BinnedDataset) -> Result<Self> {
        Self::analyze_with_config(dataset, AnalysisConfig::default())
    }

    /// Analyze with custom configuration
    pub fn analyze_with_config(dataset: &BinnedDataset, config: AnalysisConfig) -> Result<Self> {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();
        let feature_info = dataset.all_feature_info();
        let targets = dataset.targets();

        // --- Dataset Info ---
        let num_categorical = feature_info
            .iter()
            .filter(|f| matches!(f.feature_type, crate::dataset::FeatureType::Categorical))
            .count();
        let num_numeric = num_features - num_categorical;
        let categorical_ratio = num_categorical as f32 / num_features.max(1) as f32;

        // --- Sample indices for efficiency ---
        let sample_indices: Option<Vec<usize>> = if num_rows > config.max_sample_rows {
            use rand::RngExt;
            use rand::SeedableRng;

            // OPTIMIZED: Direct random sampling instead of shuffle
            // Old: O(n) create + O(n log n) shuffle for 3.48M elements
            // New: O(n) reservoir sampling with O(k) memory for k=20k samples
            let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
            let mut indices: Vec<usize> = Vec::with_capacity(config.max_sample_rows);

            // Reservoir sampling: uniform random sample in single pass
            for i in 0..config.max_sample_rows.min(num_rows) {
                indices.push(i);
            }

            for i in config.max_sample_rows..num_rows {
                let j = rng.random_range(0..=i);
                if j < config.max_sample_rows {
                    indices[j] = i;
                }
            }

            indices.sort(); // Keep sorted for cache efficiency
            Some(indices)
        } else {
            None
        };

        let sample_refs = sample_indices.as_deref();

        // --- Run probes ---
        let probe_result = run_combined_probe(
            dataset,
            sample_refs,
            config.linear_max_iter,
            config.tree_max_depth,
        )?;

        let linear_r2 = probe_result.linear.r2;
        let tree_gain = probe_result.tree.r2_on_residuals;
        let tree_relative_improvement = probe_result.tree.relative_improvement;
        let combined_r2 = probe_result.combined_r2;

        // --- Feature correlations ---
        let (raw_features, sample_targets) = if let Some(indices) = &sample_indices {
            let features = extract_sample_features(dataset, indices);
            let targets: Vec<f32> = indices.iter().map(|&i| targets[i]).collect();
            (features, targets)
        } else {
            let features = extract_all_features(dataset);
            (features, targets.to_vec())
        };

        let num_samples = sample_targets.len();
        let mut correlations: Vec<(usize, f32)> = Vec::with_capacity(num_features);

        for f in 0..num_features.min(config.top_features_to_analyze) {
            let feature_col: Vec<f32> = (0..num_samples)
                .map(|r| raw_features[r * num_features + f])
                .collect();
            let corr = compute_correlation(&feature_col, &sample_targets).abs();
            correlations.push((f, corr));
        }
        correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_correlations: Vec<(usize, f32)> = correlations.into_iter().take(10).collect();

        // --- Monotonicity ---
        let mut monotonicity_sum = 0.0f32;
        let features_to_check = num_features.min(20);
        for f in 0..features_to_check {
            let feature_col: Vec<f32> = (0..num_samples)
                .map(|r| raw_features[r * num_features + f])
                .collect();
            monotonicity_sum += compute_monotonicity(&feature_col, &sample_targets);
        }
        let avg_monotonicity = if features_to_check > 0 {
            monotonicity_sum / features_to_check as f32
        } else {
            0.5
        };

        // --- Noise estimation ---
        // Use the most correlated feature for binning (already computed above)
        let best_feature_idx = top_correlations.first().map(|(idx, _)| *idx).unwrap_or(0);
        let noise_floor = estimate_noise_floor(
            &raw_features,
            &sample_targets,
            num_features,
            best_feature_idx,
        );

        // --- Target analysis ---
        let (target_is_discrete, target_unique_values) = detect_discrete_target(targets);

        // --- Compute mode scores ---
        // Get top correlation value (important for LinearThenTree detection)
        let top_correlation = top_correlations.first().map(|(_, c)| *c).unwrap_or(0.0);

        let characteristics = DataCharacteristics {
            linear_r2,
            tree_gain,
            tree_relative_improvement,
            categorical_ratio,
            noise_floor,
            avg_monotonicity,
            num_features,
            top_correlation,
        };
        let mode_scores = compute_mode_scores(&characteristics);

        // --- Generate recommendation ---
        let recommendation = generate_recommendation(
            &mode_scores,
            linear_r2,
            tree_gain,
            categorical_ratio,
            noise_floor,
            avg_monotonicity,
        );

        Ok(Self {
            num_rows,
            num_features,
            num_categorical,
            num_numeric,
            linear_r2,
            top_correlations,
            avg_monotonicity,
            tree_gain,
            tree_relative_improvement,
            combined_r2,
            categorical_ratio,
            noise_floor,
            target_is_discrete,
            target_unique_values,
            mode_scores,
            recommendation,
            probe_result: Some(probe_result),
        })
    }

    /// Get the recommended mode
    pub fn recommend_mode(&self) -> BoostingMode {
        self.recommendation.mode
    }

    /// Get confidence in the recommendation
    pub fn confidence(&self) -> Confidence {
        self.recommendation.confidence
    }

    /// Generate a human-readable report
    pub fn report(&self) -> super::report::AnalysisReport<'_> {
        super::report::AnalysisReport::from_analysis(self)
    }
}

// =============================================================================
// Mode Scoring Logic
// =============================================================================

/// Compute scores for each mode based on data characteristics
/// Data characteristics used for mode scoring
struct DataCharacteristics {
    linear_r2: f32,
    tree_gain: f32,
    tree_relative_improvement: f32,
    categorical_ratio: f32,
    noise_floor: f32,
    avg_monotonicity: f32,
    num_features: usize,
    top_correlation: f32,
}

///
/// This is the CORE decision logic. Each mode gets a score based on
/// how well-suited the data is for that approach.
fn compute_mode_scores(chars: &DataCharacteristics) -> ModeScores {
    let DataCharacteristics {
        linear_r2,
        tree_gain,
        tree_relative_improvement,
        categorical_ratio,
        noise_floor,
        avg_monotonicity,
        num_features,
        top_correlation,
    } = *chars;
    // --- Effective Linear Signal ---
    // The multivariate linear_r2 can be low due to multicollinearity,
    // but if a single feature has very high correlation, linear models
    // can still be effective (univariate signal is strong).
    // Use the MAX of multivariate R² and squared top correlation
    let effective_linear_signal = linear_r2.max(top_correlation.powi(2) * 0.9);

    // --- PureTree Score ---
    // Favored when:
    // - Weak linear signal (trees can do better)
    // - High categorical ratio (trees handle categoricals natively)
    // - Low monotonicity (complex non-monotonic relationships)
    // - Complex interactions (many features)
    // - MODERATE noise (not too high - RF better for very noisy data)
    let pure_tree_score = {
        let weak_linear_bonus = (1.0 - effective_linear_signal).powf(0.5) * 0.3;
        let categorical_bonus = categorical_ratio * 0.3;
        let complexity_bonus = (num_features as f32 / 100.0).min(0.2);
        let non_monotonic_bonus = (1.0 - avg_monotonicity) * 0.2;

        // High noise penalty - when noise is very high, RF's variance reduction helps
        // PureTree can overfit to noise
        let high_noise_penalty = if noise_floor > 0.8 {
            (noise_floor - 0.8) * 1.5 // 0.8→0, 1.0→0.3
        } else {
            0.0
        };

        // Linear dominance penalty - when linear is strong AND trees add little,
        // LTT is better because the linear component does the heavy lifting
        let linear_dominance_penalty = if effective_linear_signal > 0.5 && tree_gain < 0.1 {
            // The stronger linear is and the less trees add, the more we penalize PT
            effective_linear_signal * (1.0 - tree_gain.min(0.1) * 10.0) * 0.25
        } else {
            0.0
        };

        // Base score for PureTree (it's a safe default)
        (0.5 + weak_linear_bonus + categorical_bonus + complexity_bonus + non_monotonic_bonus
            - high_noise_penalty
            - linear_dominance_penalty)
            .max(0.3)
    };

    // --- LinearThenTree Score ---
    // Favored when:
    // - Strong linear signal (linear model captures trend)
    // - Low categorical ratio (numeric-heavy data)
    // - KEY INSIGHT: When linear_r2 is high AND tree_gain is low, linear IS the answer!
    //   Trees add almost nothing → LTT's linear component dominates → LTT wins
    let linear_then_tree_score = {
        // Strong univariate correlation bonus
        let univariate_signal = if top_correlation > 0.7 {
            // 0.7→0.15, 0.85→0.35, 1.0→0.5
            0.15 + (top_correlation - 0.7) / 0.3 * 0.35
        } else if top_correlation > 0.5 {
            // 0.5→0.0, 0.7→0.15
            (top_correlation - 0.5) / 0.2 * 0.15
        } else {
            0.0
        };

        // Strong multivariate signal bonus (R² from linear regression)
        let multivariate_signal = if linear_r2 > 0.5 {
            // Strong: 0.5→0.2, 0.8→0.4
            0.2 + (linear_r2 - 0.5) / 0.3 * 0.2
        } else if linear_r2 > 0.2 {
            // Moderate: 0.2→0.05, 0.5→0.2
            0.05 + (linear_r2 - 0.2) / 0.3 * 0.15
        } else {
            0.0
        };

        // KEY: When linear_r2 is high AND tree_gain is low, this CONFIRMS
        // that linear is doing all the work. This is the ideal case for LTT!
        // The linear component captures the signal, trees just fine-tune.
        let linear_dominance_bonus = if linear_r2 > 0.5 && tree_gain < 0.1 {
            // Linear explains >50% and trees add <10% more → linear is dominant
            // Higher linear_r2 + lower tree_gain = stronger bonus
            let dominance = linear_r2 * (1.0 - tree_gain.min(0.1) * 10.0);
            dominance * 0.3 // Up to 0.3 bonus
        } else {
            0.0
        };

        let linear_signal_score = univariate_signal.max(multivariate_signal);
        let numeric_bonus = (1.0 - categorical_ratio) * 0.1;

        // Base score scales with signal strength
        let base = if linear_r2 > 0.5 || top_correlation > 0.7 {
            0.3
        } else if linear_r2 > 0.2 || top_correlation > 0.5 {
            0.2
        } else {
            0.1
        };

        // Only penalize if signal is truly weak
        let weak_signal_penalty = if linear_r2 < 0.1 && top_correlation < 0.4 {
            0.2
        } else {
            0.0
        };

        (base + linear_signal_score + linear_dominance_bonus + numeric_bonus - weak_signal_penalty)
            .max(0.0)
    };

    // --- RandomForest Score ---
    // Favored when:
    // - High noise floor (variance reduction helps)
    // - Trees don't add much over linear (data is noisy/random)
    // - Need robustness
    // - Very high noise (>0.8) makes RF clearly better than PureTree
    let random_forest_score = {
        // Noise bonus scales with noise level
        // Higher bonus for very high noise where RF's bagging really helps
        let noise_bonus = if noise_floor > 0.8 {
            // Very high noise: strong bonus
            0.3 + (noise_floor - 0.8) * 1.5 // 0.8→0.3, 1.0→0.6
        } else if noise_floor > 0.5 {
            // Moderate-high noise
            noise_floor * 0.5 // 0.5→0.25, 0.8→0.4
        } else {
            noise_floor * 0.3
        };

        let robustness_bonus = if tree_relative_improvement < 0.1 && effective_linear_signal < 0.3 {
            0.3 // When nothing works well, RF provides robustness
        } else {
            0.0
        };

        // RF is rarely the best choice for clean data with strong signal
        let combined = combined_r2(effective_linear_signal, tree_gain);
        let clean_data_penalty = if noise_floor < 0.2 && combined > 0.7 {
            0.3
        } else {
            0.0
        };

        (0.3 + noise_bonus + robustness_bonus - clean_data_penalty).max(0.0)
    };

    ModeScores {
        pure_tree: pure_tree_score,
        linear_then_tree: linear_then_tree_score,
        random_forest: random_forest_score,
    }
}

fn combined_r2(linear_r2: f32, tree_gain: f32) -> f32 {
    (linear_r2 + (1.0 - linear_r2) * tree_gain).clamp(0.0, 1.0)
}

// =============================================================================
// Recommendation Generation
// =============================================================================

fn generate_recommendation(
    scores: &ModeScores,
    linear_r2: f32,
    tree_gain: f32,
    categorical_ratio: f32,
    noise_floor: f32,
    avg_monotonicity: f32,
) -> Recommendation {
    let mode = scores.best_mode();
    let score_gap = scores.score_gap();

    // Determine confidence based on how clear the winner is
    let mut confidence = if score_gap > 0.3 {
        Confidence::High
    } else if score_gap > 0.15 {
        Confidence::Medium
    } else {
        Confidence::Low
    };

    // DOWNGRADE confidence for unreliable data characteristics
    // All-categorical data (binned representation) can mislead linear detection
    if categorical_ratio >= 0.99 {
        // Analysis was done on binned/categorical data - unreliable for linear assessment
        confidence = confidence.downgrade_to_medium();
    }

    // High noise makes any decision uncertain
    if noise_floor > 0.8 {
        confidence = confidence.downgrade_to_medium();
    }

    // Close score gap means validation would be valuable
    if score_gap < 0.25 {
        confidence = Confidence::Low;
    }

    // Generate reasoning
    let reasoning = match mode {
        BoostingMode::LinearThenTree => {
            let mut reasons = Vec::new();

            // Key case: linear dominance (high R², low tree gain)
            if linear_r2 > 0.5 && tree_gain < 0.1 {
                reasons.push(format!(
                    "Linear dominance detected (R²={:.2}, tree gain={:.3})",
                    linear_r2, tree_gain
                ));
                reasons.push(
                    "Linear model captures most signal, trees add minimal improvement".to_string(),
                );
            } else if linear_r2 > 0.3 {
                reasons.push(format!(
                    "Strong linear signal detected (R²={:.2})",
                    linear_r2
                ));
            } else if linear_r2 > 0.1 {
                reasons.push(format!(
                    "Moderate linear signal detected (R²={:.2})",
                    linear_r2
                ));
            }

            if tree_gain > 0.1 {
                reasons.push(format!(
                    "Trees capture additional structure (gain={:.2})",
                    tree_gain
                ));
            }

            if reasons.is_empty() {
                reasons.push(
                    "Hybrid approach balances linear trend and non-linear patterns".to_string(),
                );
            }

            format!(
                "LinearThenTree recommended. {}. Linear model captures the global trend, \
                 trees capture residual non-linearities.",
                reasons.join(". ")
            )
        }

        BoostingMode::PureTree => {
            let mut reasons = Vec::new();

            if linear_r2 < 0.2 {
                reasons.push(format!("Weak linear signal (R²={:.2})", linear_r2));
            }

            if categorical_ratio > 0.3 {
                reasons.push(format!(
                    "Categorical-heavy data ({:.0}% categorical)",
                    categorical_ratio * 100.0
                ));
            }

            if avg_monotonicity < 0.55 {
                reasons.push("Non-monotonic relationships detected".to_string());
            }

            if reasons.is_empty() {
                reasons.push("Standard GBDT is well-suited for this data".to_string());
            }

            format!(
                "PureTree (GBDT) recommended. {}. Trees can capture complex \
                 non-linear patterns and feature interactions.",
                reasons.join(". ")
            )
        }

        BoostingMode::RandomForest => {
            let mut reasons = Vec::new();

            if noise_floor > 0.3 {
                reasons.push(format!(
                    "High noise detected ({:.0}% noise floor)",
                    noise_floor * 100.0
                ));
            }

            reasons.push("Bagging provides variance reduction and robustness".to_string());

            format!(
                "RandomForest recommended. {}. Ensemble averaging reduces \
                 overfitting risk.",
                reasons.join(". ")
            )
        }
    };

    // Generate alternatives
    let mut alternatives = Vec::new();

    if mode != BoostingMode::PureTree {
        alternatives.push((
            BoostingMode::PureTree,
            "Safe default for most tabular data".to_string(),
        ));
    }

    if mode != BoostingMode::LinearThenTree && linear_r2 > 0.1 {
        alternatives.push((
            BoostingMode::LinearThenTree,
            format!("Consider if data has trends (linear R²={:.2})", linear_r2),
        ));
    }

    if mode != BoostingMode::RandomForest && noise_floor > 0.2 {
        alternatives.push((
            BoostingMode::RandomForest,
            "Consider if robustness is a priority".to_string(),
        ));
    }

    Recommendation {
        mode,
        confidence,
        reasoning,
        alternatives,
        mode_scores: scores.clone(),
    }
}

// =============================================================================
// Feature Extraction Helpers
// =============================================================================

fn extract_sample_features(dataset: &BinnedDataset, indices: &[usize]) -> Vec<f32> {
    let num_features = dataset.num_features();
    let feature_info = dataset.all_feature_info();
    let mut features = vec![0.0f32; indices.len() * num_features];

    // OPTIMIZED: Pre-compute lookup tables (same as extract_features_for_probe)
    let bin_tables: Vec<Vec<f32>> = feature_info
        .iter()
        .map(|info| {
            (0..256)
                .map(|bin| bin_to_raw(bin, &info.bin_boundaries))
                .collect()
        })
        .collect();

    for (out_idx, &row_idx) in indices.iter().enumerate() {
        for f in 0..num_features {
            let bin = dataset.get_bin(row_idx, f) as usize;
            features[out_idx * num_features + f] = bin_tables[f][bin];
        }
    }

    features
}

fn extract_all_features(dataset: &BinnedDataset) -> Vec<f32> {
    let num_rows = dataset.num_rows();
    let num_features = dataset.num_features();
    let feature_info = dataset.all_feature_info();
    let mut features = vec![0.0f32; num_rows * num_features];

    // OPTIMIZED: Pre-compute lookup tables
    let bin_tables: Vec<Vec<f32>> = feature_info
        .iter()
        .map(|info| {
            (0..256)
                .map(|bin| bin_to_raw(bin, &info.bin_boundaries))
                .collect()
        })
        .collect();

    // Fast lookup: O(1) per cell instead of O(branches)
    for r in 0..num_rows {
        for f in 0..num_features {
            let bin = dataset.get_bin(r, f) as usize;
            features[r * num_features + f] = bin_tables[f][bin];
        }
    }

    features
}

fn bin_to_raw(bin: usize, boundaries: &[f64]) -> f32 {
    if boundaries.is_empty() {
        bin as f32
    } else if bin == 0 {
        boundaries.first().copied().unwrap_or(0.0) as f32
    } else if bin >= boundaries.len() {
        boundaries.last().copied().unwrap_or(0.0) as f32
    } else {
        ((boundaries[bin - 1] + boundaries[bin.min(boundaries.len() - 1)]) / 2.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(n: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(n * num_features);
        for f in 0..num_features {
            for r in 0..n {
                features.push(((r * 17 + f * 31) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();

        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: (0..255).map(|b| b as f64).collect(),
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(n, features, targets, feature_info)
    }

    #[test]
    fn test_analysis_runs() {
        let dataset = create_test_dataset(500, 5);
        let analysis = DatasetAnalysis::analyze(&dataset).unwrap();

        assert!(analysis.linear_r2 >= 0.0 && analysis.linear_r2 <= 1.0);
        assert!(analysis.tree_gain >= 0.0 && analysis.tree_gain <= 1.0);
    }

    #[test]
    fn test_recommendation_has_reasoning() {
        let dataset = create_test_dataset(500, 5);
        let analysis = DatasetAnalysis::analyze(&dataset).unwrap();

        assert!(!analysis.recommendation.reasoning.is_empty());
    }

    #[test]
    fn test_mode_scores_sum_reasonably() {
        let dataset = create_test_dataset(500, 5);
        let analysis = DatasetAnalysis::analyze(&dataset).unwrap();

        // At least one mode should have a decent score
        assert!(analysis.mode_scores.best_score() > 0.2);
    }
}
