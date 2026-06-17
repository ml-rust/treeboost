//! Grid generation for parameter search
//!
//! This module handles the generation of candidate parameter configurations
//! using different grid strategies (Cartesian, Latin Hypercube, Random).

use std::collections::HashMap;

use crate::tuner::config::{GridStrategy, ParamBounds, ParamDef, ParameterSpace};

/// Generate a grid of candidate configurations around current centers
///
/// Dispatches to the appropriate strategy (Cartesian, LHS, or Random) based on configuration.
pub(super) fn generate_grid(
    space: &ParameterSpace,
    grid_strategy: &GridStrategy,
    seed: u64,
    spread: f32,
) -> Vec<HashMap<String, f32>> {
    match grid_strategy {
        GridStrategy::Cartesian { points_per_dim } => {
            generate_cartesian_grid(space, spread, *points_per_dim)
        }
        GridStrategy::LatinHypercube { n_samples } => {
            generate_lhs_grid(space, spread, *n_samples, seed)
        }
        GridStrategy::Random { n_samples } => generate_random_grid(space, spread, *n_samples, seed),
    }
}

/// Generate Cartesian grid
pub(crate) fn generate_cartesian_grid(
    space: &ParameterSpace,
    spread: f32,
    points_per_dim: usize,
) -> Vec<HashMap<String, f32>> {
    let params = space.params();

    if params.is_empty() {
        return vec![HashMap::new()];
    }

    // Generate values for each parameter
    let param_values: Vec<Vec<f32>> = params
        .iter()
        .map(|p| generate_param_values(p, spread, points_per_dim))
        .collect();

    // Cartesian product
    let mut candidates = Vec::new();
    let mut indices = vec![0usize; params.len()];

    loop {
        // Build candidate from current indices
        let mut candidate = HashMap::new();
        for (i, param) in params.iter().enumerate() {
            candidate.insert(param.name.to_string().into(), param_values[i][indices[i]]);
        }
        candidates.push(candidate);

        // Increment indices (like a multi-digit counter)
        let mut carry = true;
        for i in (0..params.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] >= param_values[i].len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            break; // All combinations exhausted
        }
    }

    // Dedup candidates (in case multiple parameter combinations produce identical configs)
    // This can happen when discrete parameters with small spread all round to the same value
    candidates.sort_by(|a, b| {
        for param in params {
            let name_str = param.name.to_string();
            let va = a.get(name_str).unwrap_or(&0.0);
            let vb = b.get(name_str).unwrap_or(&0.0);
            match va.partial_cmp(vb) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => continue,
            }
        }
        std::cmp::Ordering::Equal
    });
    candidates.dedup();

    candidates
}

/// Generate values for a single parameter
pub(crate) fn generate_param_values(param: &ParamDef, spread: f32, points: usize) -> Vec<f32> {
    let center = param.center;
    let (min, max) = (param.bounds.min_value(), param.bounds.max_value());

    if points == 1 {
        return vec![center];
    }

    match &param.bounds {
        ParamBounds::Continuous { log_scale, .. } if *log_scale => {
            // Log-scale sampling
            let log_center = center.ln();
            let log_min = min.ln();
            let log_max = max.ln();
            let range = log_max - log_min;
            let half_span = range * spread / 2.0;

            let low = (log_center - half_span).max(log_min);
            let high = (log_center + half_span).min(log_max);

            (0..points)
                .map(|i| {
                    let t = i as f32 / (points - 1) as f32;
                    (low + t * (high - low)).exp()
                })
                .collect()
        }
        ParamBounds::Continuous { .. } => {
            // Linear sampling
            let range = max - min;
            let half_span = range * spread / 2.0;

            let low = (center - half_span).max(min);
            let high = (center + half_span).min(max);

            (0..points)
                .map(|i| {
                    let t = i as f32 / (points - 1) as f32;
                    low + t * (high - low)
                })
                .collect()
        }
        ParamBounds::Discrete { step, .. } => {
            // Discrete sampling
            let range = max - min;
            let half_span = range * spread / 2.0;

            let low = ((center - half_span).max(min) as usize).max(*step);
            let high = (center + half_span).min(max) as usize;

            // Round to step boundaries
            let low = (low / step) * step;
            let high = high.div_ceil(*step) * step;

            let mut values: Vec<f32> = (low..=high).step_by(*step).map(|v| v as f32).collect();

            // Limit to points_per_dim values, evenly spaced
            if values.len() > points {
                let step_size = values.len() / points;
                values = values.into_iter().step_by(step_size).take(points).collect();
            }

            // Ensure center is included
            let center_val = param.bounds.clamp(center);
            if !values.contains(&center_val) {
                // Replace closest value with center
                if let Some(idx) = values
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - center_val)
                            .abs()
                            .partial_cmp(&(*b - center_val).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    values[idx] = center_val;
                }
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();
            values
        }
        ParamBounds::Categorical { values } => {
            // Return indices for each category
            // The index will be converted to the actual category string when applying params
            (0..values.len()).map(|i| i as f32).collect()
        }
    }
}

/// Generate Latin Hypercube Sampling grid
///
/// LHS ensures good space-filling by dividing each parameter's range into n equal strata
/// and sampling exactly once from each stratum. This provides better coverage than
/// pure random sampling with the same number of samples.
pub(crate) fn generate_lhs_grid(
    space: &ParameterSpace,
    spread: f32,
    n_samples: usize,
    seed: u64,
) -> Vec<HashMap<String, f32>> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};

    if n_samples == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let params = space.params();
    let n_params = params.len();

    if n_params == 0 {
        return vec![HashMap::new(); n_samples];
    }

    // Create permutation for each parameter dimension
    // Each column gets a shuffled list of strata indices [0, 1, ..., n_samples-1]
    let mut strata_permutations: Vec<Vec<usize>> = Vec::with_capacity(n_params);
    for _ in 0..n_params {
        let mut perm: Vec<usize> = (0..n_samples).collect();
        perm.shuffle(&mut rng);
        strata_permutations.push(perm);
    }

    // Generate samples - iterate by sample index, accessing each param's permutation
    let mut candidates = Vec::with_capacity(n_samples);
    #[allow(clippy::needless_range_loop)]
    for sample_idx in 0..n_samples {
        let mut candidate = HashMap::new();

        for (param_idx, param) in params.iter().enumerate() {
            let stratum = strata_permutations[param_idx][sample_idx];

            // Compute the effective bounds based on spread around center
            let center = param.center;
            let (min, max) = (param.bounds.min_value(), param.bounds.max_value());
            let range = max - min;
            let half_span = range * spread / 2.0;
            let low = (center - half_span).max(min);
            let high = (center + half_span).min(max);

            // Sample uniformly within this stratum
            // Stratum boundaries: [stratum/n_samples, (stratum+1)/n_samples] of the [low, high] range
            let stratum_low = stratum as f32 / n_samples as f32;
            let stratum_high = (stratum + 1) as f32 / n_samples as f32;
            let u: f32 = rng.random_range(stratum_low..stratum_high);

            let value = if param.bounds.is_log_scale() {
                // Log-uniform sampling within stratum
                let log_low = low.max(1e-10).ln();
                let log_high = high.max(1e-10).ln();
                (log_low + u * (log_high - log_low)).exp()
            } else {
                // Linear interpolation within stratum
                low + u * (high - low)
            };

            candidate.insert(param.name.to_string().into(), param.bounds.clamp(value));
        }

        candidates.push(candidate);
    }

    candidates
}

/// Generate random sampling grid with proper deterministic PRNG
pub(crate) fn generate_random_grid(
    space: &ParameterSpace,
    spread: f32,
    n_samples: usize,
    seed: u64,
) -> Vec<HashMap<String, f32>> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    if n_samples == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let params = space.params();

    if params.is_empty() {
        return vec![HashMap::new(); n_samples];
    }

    let mut candidates = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut candidate = HashMap::new();

        for param in params {
            // Compute the effective bounds based on spread around center
            let center = param.center;
            let (min, max) = (param.bounds.min_value(), param.bounds.max_value());
            let range = max - min;
            let half_span = range * spread / 2.0;
            let low = (center - half_span).max(min);
            let high = (center + half_span).min(max);

            // Sample uniformly in [0, 1)
            let u: f32 = rng.random();

            let value = if param.bounds.is_log_scale() {
                // Log-uniform sampling
                let log_low = low.max(1e-10).ln();
                let log_high = high.max(1e-10).ln();
                (log_low + u * (log_high - log_low)).exp()
            } else {
                // Linear interpolation
                low + u * (high - low)
            };

            candidate.insert(param.name.to_string().into(), param.bounds.clamp(value));
        }

        candidates.push(candidate);
    }

    candidates
}
