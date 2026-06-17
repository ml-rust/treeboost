//! Profiling example for AutoML pipeline
//!
//! This example profiles the AutoML pipeline to identify bottlenecks,
//! particularly in the Analysis phase (40% → 50%) after Dataset Preparation.
//!
//! Run with:
//!   cargo run --release --example automl_profile -- [--sample]
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;
use treeboost::analysis::{AnalysisConfig, DatasetAnalysis};
use treeboost::dataset::{BinnedDataset, DatasetLoader};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Check for --sample flag (default: use full 3.5M rows like real data)
    let num_rows = if args.contains(&"--sample".to_string()) {
        20_000
    } else {
        3_500_000 // Simulate full dataset size
    };

    run_profile(num_rows)
}

fn run_profile(num_rows: usize) -> Result<()> {
    println!("=== AutoML Pipeline Profiling ===\n");
    println!(
        "Simulating {} rows (panel data: stocks × dates)\n",
        num_rows
    );

    // =========================================================================
    // PHASE 0: Generate Synthetic Panel Data
    // =========================================================================
    let phase0_start = Instant::now();
    println!("[  0%] Generating synthetic panel data...");

    let train_df = generate_synthetic_panel_data(num_rows)?;

    println!(
        "[  5%] Data generated: {} rows × {} cols [{:.2?}]",
        train_df.height(),
        train_df.width(),
        phase0_start.elapsed()
    );

    // =========================================================================
    // PHASE 1: Skip Feature Engineering (already have 205 features)
    // =========================================================================
    println!("[  5%] Skipping feature engineering (already have 205 features)");
    println!(
        "[  10%] Data ready: {} rows × {} cols",
        train_df.height(),
        train_df.width()
    );

    // =========================================================================
    // PHASE 2: Date-Based Split
    // =========================================================================
    let phase2_start = Instant::now();
    println!("[  20%] Splitting data by date...");

    let dates_col = train_df.column("date")?;
    let unique_dates: Vec<i64> = dates_col
        .as_materialized_series()
        .i64()?
        .iter()
        .flatten()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut sorted_dates = unique_dates;
    sorted_dates.sort();

    let num_train_dates = ((sorted_dates.len() as f32 * 0.8).floor() as usize).max(1);
    let train_cutoff_date = sorted_dates[num_train_dates - 1];

    let train_only_df = train_df
        .clone()
        .lazy()
        .filter(col("date").lt_eq(lit(train_cutoff_date)))
        .collect()?;

    let val_only_df = train_df
        .clone()
        .lazy()
        .filter(col("date").gt(lit(train_cutoff_date)))
        .collect()?;

    println!(
        "       Train: {} rows, Val: {} rows [{:.2?}]",
        train_only_df.height(),
        val_only_df.height(),
        phase2_start.elapsed()
    );

    // =========================================================================
    // PHASE 3: Dataset Preparation (Binning)
    // =========================================================================
    let phase3_start = Instant::now();
    println!("[  30%] Dataset Preparation (binning)...");

    let binner_start = Instant::now();
    let loader = DatasetLoader::new(255); // 255 bins
    let train_dataset = loader.from_dataframe(train_only_df, "y", None)?;
    println!("       Binning complete: {:.2?}", binner_start.elapsed());

    println!(
        "[  40%] Dataset Preparation complete: {} rows × {} cols [{:.2?}]",
        train_dataset.num_rows(),
        train_dataset.num_features(),
        phase3_start.elapsed()
    );

    // =========================================================================
    // PHASE 4: Analysis Phase
    // =========================================================================
    let phase4_start = Instant::now();
    println!("[  40%] Analysis Phase - PROFILING IN DETAIL...\n");

    // Create analysis config
    let analysis_config = AnalysisConfig {
        max_sample_rows: 20_000,
        linear_max_iter: 100,
        tree_max_depth: 4,
        top_features_to_analyze: 20,
        seed: 42,
    };

    // Profile each step of analysis
    println!("       Step 1: DETAILED PROFILING of analysis internals...");
    let step1_start = Instant::now();

    // DETAILED PROFILING: Let's manually profile what happens inside analyze()
    profile_analysis_internals(&train_dataset, &analysis_config)?;

    println!("       Step 1 complete: {:.2?}\n", step1_start.elapsed());

    // Now run the actual analysis
    println!("       Step 2: Running full DatasetAnalysis::analyze()...");
    let full_analysis_start = Instant::now();
    let analysis = DatasetAnalysis::analyze_with_config(&train_dataset, analysis_config)?;
    println!(
        "       Full analysis: {:.2?}",
        full_analysis_start.elapsed()
    );

    println!(
        "\n[  50%] Analysis Phase complete [{:.2?}]",
        phase4_start.elapsed()
    );

    // Show recommendations
    let mode = analysis.recommend_mode();
    let confidence = analysis.confidence();
    println!("\n=== Analysis Results ===");
    println!("Recommended mode: {:?}", mode);
    println!("Confidence: {:?}", confidence);
    println!("Report:\n{}", analysis.report());

    Ok(())
}

/// Profile the internals of DatasetAnalysis::analyze() to find the bottleneck
fn profile_analysis_internals(dataset: &BinnedDataset, config: &AnalysisConfig) -> Result<()> {
    let num_rows = dataset.num_rows();
    let num_features = dataset.num_features();

    println!(
        "          Dataset: {} rows × {} features",
        num_rows, num_features
    );

    // Step 2.1: Sampling
    println!("          Step 2.1: Sampling rows...");
    let sampling_start = Instant::now();

    let max_sample_rows = config.max_sample_rows.min(num_rows);
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Reservoir sampling
    let mut indices: Vec<usize> = Vec::with_capacity(max_sample_rows);
    for i in 0..max_sample_rows.min(num_rows) {
        indices.push(i);
    }
    for i in max_sample_rows..num_rows {
        let j = rng.random_range(0..=i);
        if j < max_sample_rows {
            indices[j] = i;
        }
    }
    indices.sort();

    println!(
        "          Step 2.1 complete: sampled {} rows [{:.2?}]",
        indices.len(),
        sampling_start.elapsed()
    );

    // Step 2.2: Feature Extraction (THE CRITICAL PART)
    println!("          Step 2.2: Extracting features from binned data...");
    let extract_start = Instant::now();
    let _ = extract_features_profiled(dataset, Some(&indices));
    println!(
        "          Step 2.2 complete: {:.2?}",
        extract_start.elapsed()
    );

    println!(
        "          Total profiling: {:.2?}",
        sampling_start.elapsed()
    );

    Ok(())
}

/// Profiled version of extract_features_for_probe to identify bottleneck
fn extract_features_profiled(
    dataset: &BinnedDataset,
    sample_indices: Option<&[usize]>,
) -> (Vec<f32>, Vec<f32>) {
    let num_features = dataset.num_features();
    let feature_info = dataset.all_feature_info();
    let all_targets = dataset.targets();

    let indices: Vec<usize> = if let Some(idx) = sample_indices {
        idx.to_vec()
    } else {
        (0..dataset.num_rows()).collect()
    };

    let num_samples = indices.len();
    let mut features = vec![0.0f32; num_samples * num_features];
    let mut targets = Vec::with_capacity(num_samples);

    println!(
        "              Extracting {} samples × {} features = {} cells",
        num_samples,
        num_features,
        num_samples * num_features
    );

    // Profile: Pre-computing lookup tables
    println!("              Building lookup tables...");
    let table_start = Instant::now();
    let bin_tables: Vec<Vec<f32>> = feature_info
        .iter()
        .map(|info| {
            let boundaries = &info.bin_boundaries;
            if boundaries.is_empty() {
                (0..256).map(|b| b as f32).collect()
            } else {
                let mut table = Vec::with_capacity(256);
                for bin in 0..256 {
                    let raw_value = if bin == 0 {
                        boundaries.first().copied().unwrap_or(0.0) as f32
                    } else if bin >= boundaries.len() {
                        boundaries.last().copied().unwrap_or(0.0) as f32
                    } else {
                        ((boundaries[bin - 1] + boundaries[bin.min(boundaries.len() - 1)]) / 2.0)
                            as f32
                    };
                    table.push(raw_value);
                }
                table
            }
        })
        .collect();
    println!(
        "              Lookup tables built: {:.2?}",
        table_start.elapsed()
    );

    // Profile: Fast lookup
    println!("              Extracting features using lookup...");
    let extract_start = Instant::now();
    for (out_idx, &row_idx) in indices.iter().enumerate() {
        targets.push(all_targets[row_idx]);
        for f in 0..num_features {
            let bin = dataset.get_bin(row_idx, f) as usize;
            features[out_idx * num_features + f] = bin_tables[f][bin];
        }
    }
    println!(
        "              Feature extraction: {:.2?}",
        extract_start.elapsed()
    );

    (features, targets)
}

/// Generate synthetic panel data (stocks × dates) for profiling
fn generate_synthetic_panel_data(num_rows: usize) -> Result<DataFrame> {
    println!(
        "       Generating {} rows with 205 features (matching real data after transforms)...",
        num_rows
    );

    let mut rng = StdRng::seed_from_u64(42);

    // Simulate: ~5000 stocks × ~700 dates = ~3.5M rows
    let num_stocks = 5000;
    let num_dates = (num_rows / num_stocks).max(1);
    let actual_rows = num_stocks * num_dates;

    println!(
        "       Panel structure: {} stocks × {} dates = {} rows",
        num_stocks, num_dates, actual_rows
    );

    // Generate stock codes
    let codes: Vec<String> = (0..actual_rows)
        .map(|i| format!("STOCK_{:04}", i % num_stocks))
        .collect();

    // Generate dates (sequential days)
    let dates: Vec<i64> = (0..actual_rows).map(|i| (i / num_stocks) as i64).collect();

    // Generate features to match real data after time-series transforms
    // Real data: 7 base + lags + rolling + EWMA + cross-sectional = ~205 features
    let num_features = 205;
    let mut all_columns = Vec::new();

    // Add code and date
    all_columns.push(Column::new("code".into(), codes));
    all_columns.push(Column::new("date".into(), dates));

    // Add features
    for f_idx in 0..num_features {
        let values: Vec<f64> = (0..actual_rows)
            .map(|_| rng.random::<f64>() * 100.0)
            .collect();
        all_columns.push(Column::new(format!("f_{}", f_idx).into(), values));
    }

    // Generate and add target
    let targets: Vec<f32> = (0..actual_rows)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0) // Returns in [-1, 1]
        .collect();
    all_columns.push(Column::new("y".into(), targets));

    let df = DataFrame::new(actual_rows, all_columns)?;

    Ok(df)
}
