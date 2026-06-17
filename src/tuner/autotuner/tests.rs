//! Unit tests for AutoTuner

use std::collections::HashMap;

use crate::backend::BackendType;
use crate::booster::{GBDTConfig, GBDTModel};
use crate::tuner::config::{
    ParamBounds, ParamDef, ParameterSpace, SpacePreset, TunerConfig, TuningMode,
};
use crate::tuner::history::SearchHistory;
use crate::tuner::trial::TrialResult;

use super::AutoTuner;

#[test]
fn test_trial_result() {
    let mut params = HashMap::new();
    params.insert("max_depth".into(), 6.0);
    params.insert("learning_rate".into(), 0.1);

    let result = TrialResult {
        trial_id: 0,
        iteration: 0,
        params,
        val_loss: 0.5,
        train_loss: 0.4,
        num_trees: 100,
        train_time_ms: 1000,
        f1_score: None,
        roc_auc: None,
        rank_ic: None,
    };

    assert_eq!(result.trial_id, 0);
    assert_eq!(result.val_loss, 0.5);
}

#[test]
fn test_search_history() {
    let mut history = SearchHistory::new();
    assert!(history.is_empty());

    // Add first trial
    let mut params1 = HashMap::new();
    params1.insert("max_depth".into(), 6.0);

    history.add(TrialResult {
        trial_id: 0,
        iteration: 0,
        params: params1,
        val_loss: 0.5,
        train_loss: 0.4,
        num_trees: 100,
        train_time_ms: 1000,
        f1_score: None,
        roc_auc: None,
        rank_ic: None,
    });

    assert_eq!(history.len(), 1);
    assert_eq!(history.best().unwrap().trial_id, 0);

    // Add better trial
    let mut params2 = HashMap::new();
    params2.insert("max_depth".into(), 8.0);

    history.add(TrialResult {
        trial_id: 1,
        iteration: 0,
        params: params2,
        val_loss: 0.3, // Better
        train_loss: 0.25,
        num_trees: 100,
        train_time_ms: 1000,
        f1_score: None,
        roc_auc: None,
        rank_ic: None,
    });

    assert_eq!(history.len(), 2);
    assert_eq!(history.best().unwrap().trial_id, 1);
}

#[test]
fn test_search_history_to_json() {
    let mut history = SearchHistory::new();
    let mut params = HashMap::new();
    params.insert("max_depth".into(), 6.0);

    history.add(TrialResult {
        trial_id: 0,
        iteration: 0,
        params,
        val_loss: 0.5,
        train_loss: 0.4,
        num_trees: 100,
        train_time_ms: 1000,
        f1_score: None,
        roc_auc: None,
        rank_ic: None,
    });

    let json = history.to_json();
    assert!(json.contains("\"trial_id\": 0"));
    assert!(json.contains("\"val_metric\": 0.5"));
    assert!(json.contains("\"best_trial_id\": 0"));
}

#[test]
fn test_autotuner_generate_param_values() {
    use crate::tuner::config::TunableParam;

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default());

    // Test continuous parameter
    let param = ParamDef::new(
        TunableParam::Subsample,
        ParamBounds::continuous(0.0, 1.0),
        0.5,
    );
    let values = tuner.generate_param_values(&param, 0.5, 3);
    assert_eq!(values.len(), 3);
    assert!(values[0] < values[1]);
    assert!(values[1] < values[2]);

    // Test discrete parameter
    let param = ParamDef::new(TunableParam::MaxDepth, ParamBounds::discrete(2, 10), 6.0);
    let values = tuner.generate_param_values(&param, 0.5, 3);
    assert!(!values.is_empty());
    assert!(values.iter().all(|&v| (2.0..=10.0).contains(&v)));
}

#[test]
fn test_autotuner_generate_cartesian_grid() {
    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(ParameterSpace::with_preset(SpacePreset::Minimal));

    let grid = tuner.generate_cartesian_grid(0.5, 3);
    // 2 parameters, 3 points each = 9 candidates
    assert_eq!(grid.len(), 9);

    for candidate in &grid {
        assert!(candidate.contains_key("max_depth"));
        assert!(candidate.contains_key("learning_rate"));
    }
}

#[test]
fn test_autotuner_build_config() {
    let base = GBDTConfig::default();
    let tuner = AutoTuner::<GBDTModel>::new(base.clone());

    let mut params = HashMap::new();
    params.insert("max_depth".into(), 8.0);
    params.insert("learning_rate".into(), 0.05);

    let config = tuner.build_config(&params);
    assert_eq!(config.max_depth, 8);
    assert_eq!(config.learning_rate, 0.05);
}

#[test]
fn test_discrete_grid_dedup() {
    // Test that discrete parameters with small spread don't produce duplicates
    // If center=6 and spread is tiny, all 3 points should round to 6
    // After dedup, we should have only 1 unique value
    use crate::tuner::config::TunableParam;

    let space =
        ParameterSpace::new().with_param(TunableParam::MaxDepth, ParamBounds::discrete(2, 10), 6.0);

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default()).with_space(space);

    // Very small spread - all values should round to 6
    let values = tuner.generate_param_values(
        tuner.config.space.get(TunableParam::MaxDepth).unwrap(),
        0.01, // 1% spread around center 6
        3,
    );

    // After dedup, there should be no duplicate values
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup();
    assert_eq!(
        values.len(),
        sorted.len(),
        "Discrete values should be unique after dedup"
    );
}

#[test]
fn test_grid_level_dedup() {
    // Test that the grid itself has no duplicate candidates
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new()
        .with_param(TunableParam::MaxDepth, ParamBounds::discrete(2, 10), 6.0)
        .with_param(
            TunableParam::MinSamplesLeaf,
            ParamBounds::discrete(1, 10),
            5.0,
        );

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default()).with_space(space);

    // Small spread - may cause duplicates before dedup
    let grid = tuner.generate_cartesian_grid(0.05, 3);

    // Check no duplicate candidates
    let mut seen = std::collections::HashSet::new();
    for candidate in &grid {
        let key = format!("{:?}", candidate);
        assert!(seen.insert(key), "Grid should have no duplicate candidates");
    }
}

#[test]
fn test_lhs_determinism() {
    // Same seed should produce identical samples
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new()
        .with_param(
            TunableParam::LearningRate,
            ParamBounds::log_continuous(0.01, 0.5),
            0.1,
        )
        .with_param(TunableParam::MaxDepth, ParamBounds::discrete(2, 12), 6.0);

    let tuner1 = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space.clone())
        .with_seed(42);

    let tuner2 = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(42);

    let grid1 = tuner1.generate_lhs_grid(0.5, 10);
    let grid2 = tuner2.generate_lhs_grid(0.5, 10);

    assert_eq!(grid1.len(), grid2.len());
    for (c1, c2) in grid1.iter().zip(grid2.iter()) {
        for key in c1.keys() {
            assert!(
                (c1[key] - c2[key]).abs() < 1e-6,
                "LHS should be deterministic with same seed"
            );
        }
    }
}

#[test]
fn test_lhs_sample_count() {
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new()
        .with_param(
            TunableParam::LearningRate,
            ParamBounds::continuous(0.01, 0.5),
            0.1,
        )
        .with_param(
            TunableParam::Subsample,
            ParamBounds::continuous(0.5, 1.0),
            0.8,
        );

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(123);

    // Request 20 samples
    let grid = tuner.generate_lhs_grid(1.0, 20);
    assert_eq!(grid.len(), 20, "LHS should return exactly n_samples");

    // Edge case: 0 samples
    let empty = tuner.generate_lhs_grid(1.0, 0);
    assert!(empty.is_empty(), "LHS with n_samples=0 should be empty");
}

#[test]
fn test_lhs_bounds_respected() {
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new()
        .with_param(
            TunableParam::LearningRate,
            ParamBounds::continuous(0.01, 0.5),
            0.1,
        )
        .with_param(TunableParam::MaxDepth, ParamBounds::discrete(2, 12), 6.0);

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(999);

    let grid = tuner.generate_lhs_grid(1.0, 50);

    for candidate in &grid {
        let lr = candidate["learning_rate"];
        assert!(
            (0.01..=0.5).contains(&lr),
            "learning_rate {} out of bounds [0.01, 0.5]",
            lr
        );

        let depth = candidate["max_depth"];
        assert!(
            (2.0..=12.0).contains(&depth),
            "max_depth {} out of bounds [2, 12]",
            depth
        );
    }
}

#[test]
fn test_lhs_stratification() {
    // LHS should have good space-filling property
    // Each stratum should be sampled exactly once
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new().with_param(
        TunableParam::Subsample,
        ParamBounds::continuous(0.0, 1.0),
        0.5,
    );

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(12345);

    let n_samples = 10;
    let grid = tuner.generate_lhs_grid(1.0, n_samples);

    // Extract values and check stratum coverage
    let values: Vec<f32> = grid.iter().map(|c| c["subsample"]).collect();

    // Count how many samples fall into each stratum
    let mut stratum_counts = vec![0; n_samples];
    for &v in &values {
        let stratum = (v * n_samples as f32).floor() as usize;
        let stratum = stratum.min(n_samples - 1); // Handle edge case of v = 1.0
        stratum_counts[stratum] += 1;
    }

    // Each stratum should have exactly one sample
    for (i, &count) in stratum_counts.iter().enumerate() {
        assert_eq!(
            count, 1,
            "Stratum {} should have exactly 1 sample, got {}",
            i, count
        );
    }
}

#[test]
fn test_random_determinism() {
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new()
        .with_param(
            TunableParam::LearningRate,
            ParamBounds::log_continuous(0.01, 0.5),
            0.1,
        )
        .with_param(
            TunableParam::Lambda,
            ParamBounds::continuous(0.0, 10.0),
            1.0,
        );

    let tuner1 = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space.clone())
        .with_seed(42);

    let tuner2 = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(42);

    let grid1 = tuner1.generate_random_grid(0.5, 15);
    let grid2 = tuner2.generate_random_grid(0.5, 15);

    assert_eq!(grid1.len(), grid2.len());
    for (c1, c2) in grid1.iter().zip(grid2.iter()) {
        for key in c1.keys() {
            assert!(
                (c1[key] - c2[key]).abs() < 1e-6,
                "Random sampling should be deterministic with same seed"
            );
        }
    }
}

#[test]
fn test_random_sample_count() {
    let space = ParameterSpace::with_preset(SpacePreset::Minimal);
    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(777);

    let grid = tuner.generate_random_grid(1.0, 25);
    assert_eq!(grid.len(), 25, "Random should return exactly n_samples");

    let empty = tuner.generate_random_grid(1.0, 0);
    assert!(empty.is_empty(), "Random with n_samples=0 should be empty");
}

#[test]
fn test_random_bounds_respected() {
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new()
        .with_param(
            TunableParam::Subsample,
            ParamBounds::continuous(0.5, 1.0),
            0.8,
        )
        .with_param(
            TunableParam::EntropyWeight,
            ParamBounds::continuous(0.0, 0.5),
            0.1,
        );

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(888);

    let grid = tuner.generate_random_grid(1.0, 100);

    for candidate in &grid {
        let ss = candidate["subsample"];
        assert!(
            (0.5..=1.0).contains(&ss),
            "subsample {} out of bounds [0.5, 1.0]",
            ss
        );

        let ew = candidate["entropy_weight"];
        assert!(
            (0.0..=0.5).contains(&ew),
            "entropy_weight {} out of bounds [0.0, 0.5]",
            ew
        );
    }
}

#[test]
fn test_different_seeds_produce_different_results() {
    let space = ParameterSpace::with_preset(SpacePreset::Minimal);

    let tuner1 = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space.clone())
        .with_seed(1);

    let tuner2 = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(2);

    let grid1 = tuner1.generate_lhs_grid(1.0, 5);
    let grid2 = tuner2.generate_lhs_grid(1.0, 5);

    // At least one value should differ
    let mut all_same = true;
    for (c1, c2) in grid1.iter().zip(grid2.iter()) {
        for key in c1.keys() {
            if (c1[key] - c2[key]).abs() > 1e-6 {
                all_same = false;
                break;
            }
        }
    }
    assert!(
        !all_same,
        "Different seeds should produce different results"
    );
}

#[test]
fn test_log_scale_sampling() {
    // Verify log-scale parameters are sampled uniformly in log space
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new().with_param(
        TunableParam::LearningRate,
        ParamBounds::log_continuous(0.001, 1.0),
        0.1,
    );

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(42);

    let grid = tuner.generate_random_grid(1.0, 1000);
    let values: Vec<f32> = grid.iter().map(|c| c["learning_rate"]).collect();

    // Count how many are below vs above geometric mean
    let geo_mean = (0.001_f32 * 1.0).sqrt(); // ~0.0316
    let below = values.iter().filter(|&&v| v < geo_mean).count();
    let above = values.iter().filter(|&&v| v >= geo_mean).count();

    // Should be roughly 50/50 in log space
    let ratio = below as f32 / (below + above) as f32;
    assert!(
        ratio > 0.4 && ratio < 0.6,
        "Log-scale sampling should be balanced: ratio = {}",
        ratio
    );
}

#[test]
fn test_spread_affects_range() {
    use crate::tuner::config::TunableParam;

    let space = ParameterSpace::new().with_param(
        TunableParam::Colsample,
        ParamBounds::continuous(0.0, 1.0),
        0.5,
    );

    let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default())
        .with_space(space)
        .with_seed(42);

    // Wide spread
    let wide = tuner.generate_random_grid(1.0, 100);
    let wide_range: f32 = wide
        .iter()
        .map(|c| c["colsample"])
        .fold(0.0_f32, |a, b| a.max(b))
        - wide
            .iter()
            .map(|c| c["colsample"])
            .fold(1.0_f32, |a, b| a.min(b));

    // Narrow spread
    let narrow = tuner.generate_random_grid(0.1, 100);
    let narrow_range: f32 = narrow
        .iter()
        .map(|c| c["colsample"])
        .fold(0.0_f32, |a, b| a.max(b))
        - narrow
            .iter()
            .map(|c| c["colsample"])
            .fold(1.0_f32, |a, b| a.min(b));

    assert!(
        wide_range > narrow_range,
        "Larger spread should produce wider range: wide={}, narrow={}",
        wide_range,
        narrow_range
    );
}

#[test]
fn test_is_gpu_backend() {
    // Test GPU backends are detected
    let mut config = GBDTConfig {
        backend_type: BackendType::Auto,
        ..Default::default()
    };

    let tuner = AutoTuner::<GBDTModel>::new(config.clone());
    assert!(
        tuner.is_gpu_backend(),
        "Auto should be treated as GPU (conservative)"
    );

    config.backend_type = BackendType::Wgpu;
    let tuner = AutoTuner::<GBDTModel>::new(config.clone());
    assert!(tuner.is_gpu_backend(), "WGPU is a GPU backend");

    config.backend_type = BackendType::Cuda;
    let tuner = AutoTuner::<GBDTModel>::new(config.clone());
    assert!(tuner.is_gpu_backend(), "CUDA is a GPU backend");

    // Test CPU backends are not GPU
    config.backend_type = BackendType::Scalar;
    let tuner = AutoTuner::<GBDTModel>::new(config.clone());
    assert!(!tuner.is_gpu_backend(), "Scalar is a CPU backend");

    config.backend_type = BackendType::Avx512;
    let tuner = AutoTuner::<GBDTModel>::new(config.clone());
    assert!(!tuner.is_gpu_backend(), "AVX-512 is a CPU backend");

    config.backend_type = BackendType::Sve2;
    let tuner = AutoTuner::<GBDTModel>::new(config);
    assert!(!tuner.is_gpu_backend(), "SVE2 is a CPU backend");
}

#[test]
fn test_parallel_config_respected() {
    // Test that parallel_trials setting is respected
    let config = GBDTConfig {
        backend_type: BackendType::Scalar, // CPU backend
        ..Default::default()
    };

    let tuner_config = TunerConfig::new().with_parallel(true).with_n_parallel(4);

    let tuner = AutoTuner::<GBDTModel>::new(config).with_config(tuner_config);

    // Verify settings are applied
    assert!(tuner.config().parallel_trials);
    assert_eq!(tuner.config().n_parallel, 4);
}

// ==========================================================================
// Realistic Mode Tests
// ==========================================================================

#[test]
fn test_tuning_mode_variants() {
    // Test that TuningMode enum works correctly
    let optimistic = TuningMode::Optimistic;
    let realistic = TuningMode::Realistic;

    // Verify they are distinct variants
    assert!(matches!(optimistic, TuningMode::Optimistic));
    assert!(matches!(realistic, TuningMode::Realistic));
}
