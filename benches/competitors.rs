//! Benchmark comparing TreeBoost against competitor GBDT implementations
//!
//! Compares training and prediction performance of:
//! - TreeBoost (this crate)
//! - gbdt-rs (Baidu's pure-Rust GBDT)
//! - forust (modern histogram-based GBDT)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::hint::black_box;
use std::time::Duration;

// TreeBoost imports
use treeboost::booster::{GBDTConfig, GBDTModel};

// Competitor imports
use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};
use gbdt::config::Config;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;

/// Generate synthetic regression dataset
/// Returns (features_flat, targets) where features_flat is row-major
fn generate_regression_data(
    num_rows: usize,
    num_features: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut features = Vec::with_capacity(num_rows * num_features);
    let mut targets = Vec::with_capacity(num_rows);

    for _ in 0..num_rows {
        let mut row_sum = 0.0;
        for f in 0..num_features {
            let val: f64 = rng.random_range(0.0..10.0);
            features.push(val);
            // Target is weighted sum of features with some noise
            row_sum += val * (f as f64 + 1.0) * 0.1;
        }
        let noise: f64 = rng.random_range(-1.0..1.0);
        targets.push(row_sum + noise);
    }

    (features, targets)
}

/// Convert f64 features to f32 for TreeBoost (raw, no binning)
fn to_treeboost_features(features: &[f64], targets: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let features_f32: Vec<f32> = features.iter().map(|&f| f as f32).collect();
    let targets_f32: Vec<f32> = targets.iter().map(|&t| t as f32).collect();
    (features_f32, targets_f32)
}

/// Convert to gbdt-rs format
fn to_gbdt_data(
    features: &[f64],
    targets: &[f64],
    num_rows: usize,
    num_features: usize,
) -> Vec<Data> {
    (0..num_rows)
        .map(|r| {
            let row_features: Vec<f32> = (0..num_features)
                .map(|f| features[r * num_features + f] as f32)
                .collect();
            Data::new_training_data(row_features, 1.0, targets[r] as f32, None)
        })
        .collect()
}

/// Convert to forust format (column-major f64)
fn to_forust_matrix(features: &[f64], num_rows: usize, num_features: usize) -> Vec<f64> {
    // Forust uses column-major layout
    let mut col_major = vec![0.0; num_rows * num_features];
    for r in 0..num_rows {
        for f in 0..num_features {
            col_major[f * num_rows + r] = features[r * num_features + f];
        }
    }
    col_major
}

fn benchmark_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("Training");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    // Test different dataset sizes (smaller default for faster iteration)
    for &num_rows in &[1_000, 10_000, 100_000] {
        let num_features = 10;
        let num_rounds = 50;
        let max_depth = 6;

        let (features, targets) = generate_regression_data(num_rows, num_features, 42);

        group.throughput(Throughput::Elements(num_rows as u64));

        // Pre-compute TreeBoost features (raw, no binning - same as Python)
        let (treeboost_features, treeboost_targets) = to_treeboost_features(&features, &targets);
        group.bench_with_input(
            BenchmarkId::new("TreeBoost", num_rows),
            &(&treeboost_features, &treeboost_targets),
            |b, (feats, targs)| {
                b.iter(|| {
                    let config = GBDTConfig::new()
                        .with_num_rounds(num_rounds)
                        .with_max_depth(max_depth)
                        .with_learning_rate(0.1)
                        .with_min_samples_leaf(5);
                    black_box(GBDTModel::train(feats, num_features, targs, config, None).unwrap())
                });
            },
        );

        // Pre-compute gbdt-rs data once
        let gbdt_data = to_gbdt_data(&features, &targets, num_rows, num_features);
        group.bench_with_input(
            BenchmarkId::new("gbdt-rs", num_rows),
            &gbdt_data,
            |b, data| {
                b.iter(|| {
                    let mut data_clone = data.clone();
                    let mut cfg = Config::new();
                    cfg.set_feature_size(num_features);
                    cfg.set_max_depth(max_depth as u32);
                    cfg.set_iterations(num_rounds);
                    cfg.set_shrinkage(0.1);
                    cfg.set_loss("SquaredError");
                    cfg.set_min_leaf_size(5);

                    let mut gbdt = GBDT::new(&cfg);
                    gbdt.fit(&mut data_clone);
                    black_box(gbdt)
                });
            },
        );

        // Pre-compute forust data once
        let forust_features = to_forust_matrix(&features, num_rows, num_features);
        let forust_targets: Vec<f64> = targets.clone();
        group.bench_with_input(
            BenchmarkId::new("forust", num_rows),
            &(&forust_features, &forust_targets),
            |b, (feats, targs)| {
                b.iter(|| {
                    let matrix = Matrix::new(feats, num_rows, num_features);
                    let mut model = GradientBooster::default()
                        .set_objective_type(ObjectiveType::SquaredLoss)
                        .set_iterations(num_rounds)
                        .set_max_depth(max_depth)
                        .set_learning_rate(0.1)
                        .set_min_leaf_weight(5.0)
                        .set_parallel(false); // Fair single-threaded comparison

                    model.fit_unweighted(&matrix, targs, None).unwrap();
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Prediction");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let num_features = 10;
    let num_rounds = 50;
    let max_depth = 6;
    let train_rows = 10_000;

    // Generate training data and train models once
    let (train_features, train_targets) = generate_regression_data(train_rows, num_features, 42);

    // Train TreeBoost model (raw float data)
    let (treeboost_features, treeboost_targets) =
        to_treeboost_features(&train_features, &train_targets);
    let treeboost_config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(max_depth)
        .with_learning_rate(0.1)
        .with_min_samples_leaf(5);
    let treeboost_model = GBDTModel::train(
        &treeboost_features,
        num_features,
        &treeboost_targets,
        treeboost_config,
        None,
    )
    .unwrap();

    // Train gbdt-rs model
    let mut gbdt_train = to_gbdt_data(&train_features, &train_targets, train_rows, num_features);
    let mut gbdt_cfg = Config::new();
    gbdt_cfg.set_feature_size(num_features);
    gbdt_cfg.set_max_depth(max_depth as u32);
    gbdt_cfg.set_iterations(num_rounds);
    gbdt_cfg.set_shrinkage(0.1);
    gbdt_cfg.set_loss("SquaredError");
    gbdt_cfg.set_min_leaf_size(5);
    let mut gbdt_model = GBDT::new(&gbdt_cfg);
    gbdt_model.fit(&mut gbdt_train);

    // Train forust model
    let forust_train_features = to_forust_matrix(&train_features, train_rows, num_features);
    let forust_train_matrix = Matrix::new(&forust_train_features, train_rows, num_features);
    let mut forust_model = GradientBooster::default()
        .set_objective_type(ObjectiveType::SquaredLoss)
        .set_iterations(num_rounds)
        .set_max_depth(max_depth)
        .set_learning_rate(0.1)
        .set_min_leaf_weight(5.0)
        .set_parallel(false);
    forust_model
        .fit_unweighted(&forust_train_matrix, &train_targets, None)
        .unwrap();

    // Benchmark prediction on different test sizes
    for &num_rows in &[100, 1_000, 10_000] {
        let (test_features, test_targets) = generate_regression_data(num_rows, num_features, 123);

        group.throughput(Throughput::Elements(num_rows as u64));

        // TreeBoost prediction (raw float data)
        let (treeboost_test_features, _) = to_treeboost_features(&test_features, &test_targets);
        // Convert f32 back to f64 for predict_raw API
        let treeboost_test_f64: Vec<f64> =
            treeboost_test_features.iter().map(|&f| f as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("TreeBoost", num_rows),
            &treeboost_test_f64,
            |b, feats| {
                b.iter(|| black_box(treeboost_model.predict_raw(feats)));
            },
        );

        // gbdt-rs prediction
        let gbdt_test = to_gbdt_data(&test_features, &test_targets, num_rows, num_features);
        group.bench_with_input(
            BenchmarkId::new("gbdt-rs", num_rows),
            &gbdt_test,
            |b, data| {
                b.iter(|| black_box(gbdt_model.predict(data)));
            },
        );

        // forust prediction
        let forust_test_features = to_forust_matrix(&test_features, num_rows, num_features);
        group.bench_with_input(
            BenchmarkId::new("forust", num_rows),
            &forust_test_features,
            |b, feats| {
                let matrix = Matrix::new(feats, num_rows, num_features);
                b.iter(|| black_box(forust_model.predict(&matrix, false)));
            },
        );
    }

    group.finish();
}

fn benchmark_parallel_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("ParallelTraining");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let num_features = 10;
    let num_rounds = 100;
    let max_depth = 6;
    let num_rows = 100_000;

    let (features, targets) = generate_regression_data(num_rows, num_features, 42);

    group.throughput(Throughput::Elements(num_rows as u64));

    // TreeBoost (uses Rayon internally, raw float data)
    let (treeboost_features, treeboost_targets) = to_treeboost_features(&features, &targets);
    group.bench_function("TreeBoost", |b| {
        b.iter(|| {
            let config = GBDTConfig::new()
                .with_num_rounds(num_rounds)
                .with_max_depth(max_depth)
                .with_learning_rate(0.1)
                .with_min_samples_leaf(5);
            black_box(
                GBDTModel::train(
                    &treeboost_features,
                    num_features,
                    &treeboost_targets,
                    config,
                    None,
                )
                .unwrap(),
            )
        });
    });

    // forust with parallelism enabled
    let forust_features = to_forust_matrix(&features, num_rows, num_features);
    group.bench_function("forust-parallel", |b| {
        b.iter(|| {
            let matrix = Matrix::new(&forust_features, num_rows, num_features);
            let mut model = GradientBooster::default()
                .set_objective_type(ObjectiveType::SquaredLoss)
                .set_iterations(num_rounds)
                .set_max_depth(max_depth)
                .set_learning_rate(0.1)
                .set_min_leaf_weight(5.0)
                .set_parallel(true);

            model.fit_unweighted(&matrix, &targets, None).unwrap();
            black_box(model)
        });
    });

    // gbdt-rs (single-threaded only for training)
    let gbdt_data = to_gbdt_data(&features, &targets, num_rows, num_features);
    group.bench_function("gbdt-rs", |b| {
        b.iter(|| {
            let mut data_clone = gbdt_data.clone();
            let mut cfg = Config::new();
            cfg.set_feature_size(num_features);
            cfg.set_max_depth(max_depth as u32);
            cfg.set_iterations(num_rounds);
            cfg.set_shrinkage(0.1);
            cfg.set_loss("SquaredError");
            cfg.set_min_leaf_size(5);

            let mut gbdt = GBDT::new(&cfg);
            gbdt.fit(&mut data_clone);
            black_box(gbdt)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_training,
    benchmark_prediction,
    benchmark_parallel_training
);
criterion_main!(benches);
