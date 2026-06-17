//! TreeBoost scaling benchmarks: CPU vs GPU (1K-1M rows)
//!
//! Measures training and inference performance across dataset sizes.
//! Fast execution (~3-5 minutes total) with consistent synthetic data.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::hint::black_box;
use std::time::Duration;

use treeboost::backend::BackendType;
use treeboost::booster::{GBDTConfig, GBDTModel};

/// Generate consistent synthetic dataset
fn generate_data(num_rows: usize, num_features: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::with_capacity(num_rows * num_features);
    let mut targets = Vec::with_capacity(num_rows);

    for _ in 0..num_rows {
        let mut row_sum = 0.0f32;
        for f in 0..num_features {
            let val: f32 = rng.random_range(0.0..10.0);
            features.push(val);
            row_sum += val * (f as f32 + 1.0) * 0.1;
        }
        targets.push(row_sum + rng.random_range(-1.0..1.0));
    }

    (features, targets)
}

fn benchmark_training_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("TrainingScaling");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let num_features = 20;
    let num_rounds = 50;
    let max_depth = 6;

    // Test dataset sizes: 1K, 10K, 100K
    for &num_rows in &[1_000, 10_000, 100_000] {
        let (features, targets) = generate_data(num_rows, num_features, 42);

        group.throughput(Throughput::Elements(num_rows as u64));
        group.bench_with_input(
            BenchmarkId::new("cpu", num_rows),
            &(&features, &targets),
            |b, (feats, targs)| {
                b.iter(|| {
                    let config = GBDTConfig::new()
                        .with_num_rounds(num_rounds)
                        .with_max_depth(max_depth)
                        .with_learning_rate(0.1)
                        .with_min_samples_leaf(5)
                        .with_backend(BackendType::Scalar);
                    black_box(GBDTModel::train(feats, num_features, targs, config, None).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_training_scaling_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("TrainingScalingGPU");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let num_features = 20;
    let num_rounds = 50;
    let max_depth = 6;

    // GPU test dataset sizes: 1K, 10K, 100K
    for &num_rows in &[1_000, 10_000, 100_000] {
        let (features, targets) = generate_data(num_rows, num_features, 42);

        group.throughput(Throughput::Elements(num_rows as u64));
        group.bench_with_input(
            BenchmarkId::new("gpu", num_rows),
            &(&features, &targets),
            |b, (feats, targs)| {
                b.iter(|| {
                    let config = GBDTConfig::new()
                        .with_num_rounds(num_rounds)
                        .with_max_depth(max_depth)
                        .with_learning_rate(0.1)
                        .with_min_samples_leaf(5)
                        .with_backend(BackendType::Auto);
                    black_box(GBDTModel::train(feats, num_features, targs, config, None).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_inference_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("InferenceScaling");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let num_features = 20;
    let num_rounds = 50;
    let max_depth = 6;

    // Train once on CPU
    let (train_features, train_targets) = generate_data(10_000, num_features, 42);
    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(max_depth)
        .with_learning_rate(0.1)
        .with_min_samples_leaf(5)
        .with_backend(BackendType::Scalar);
    let model =
        GBDTModel::train(&train_features, num_features, &train_targets, config, None).unwrap();

    // Benchmark inference on different sizes (always CPU - inference is CPU-only)
    for &num_rows in &[100, 1_000, 10_000, 100_000] {
        let (test_features, test_targets) = generate_data(num_rows, num_features, 123);

        group.throughput(Throughput::Elements(num_rows as u64));
        group.bench_with_input(
            BenchmarkId::new("inference", num_rows),
            &(&test_features, &test_targets),
            |b, (feats, _)| {
                b.iter(|| {
                    black_box(
                        model.predict_raw(&feats.iter().map(|&f| f as f64).collect::<Vec<_>>()),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_training_scaling,
    benchmark_training_scaling_gpu,
    benchmark_inference_scaling
);
criterion_main!(benches);
