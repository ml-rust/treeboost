//! Integration tests for preprocessing module

use treeboost::preprocessing::{
    FrequencyEncoder, ImputeStrategy, LabelEncoder, MinMaxScaler, OneHotEncoder, OutlierAction,
    OutlierDetector, OutlierMethod, RobustScaler, Scaler, SimpleImputer, StandardScaler,
    TransformResult, UnknownStrategy,
};

/// Test StandardScaler end-to-end workflow
#[test]
fn test_preprocessing_standard_scaler_workflow() {
    // Create row-major data: 100 rows × 3 features
    let num_rows = 100;
    let num_features = 3;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        data.push(i as f32 * 10.0); // f0: 0, 10, 20, ...
        data.push((i as f32).powf(2.0)); // f1: 0, 1, 4, 9, ...
        data.push(1000.0 + i as f32 * 0.1); // f2: 1000.0, 1000.1, ...
    }

    let mut scaler = StandardScaler::new();

    // Fit on data
    scaler.fit(&data, num_features).expect("Fit should succeed");
    assert!(scaler.is_fitted());

    // Transform
    let mut transformed = data.clone();
    scaler
        .transform(&mut transformed, num_features)
        .expect("Transform should succeed");

    // Verify: Each feature should have mean ≈ 0, std ≈ 1
    for f in 0..num_features {
        let feature_vals: Vec<f32> = (0..num_rows)
            .map(|r| transformed[r * num_features + f])
            .collect();

        let mean: f32 = feature_vals.iter().sum::<f32>() / num_rows as f32;
        let var: f32 =
            feature_vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / num_rows as f32;
        let std = var.sqrt();

        assert!(
            mean.abs() < 0.01,
            "Feature {} mean should be ~0, got {}",
            f,
            mean
        );
        assert!(
            (std - 1.0).abs() < 0.1,
            "Feature {} std should be ~1, got {}",
            f,
            std
        );
    }
}

/// Test MinMaxScaler range normalization
#[test]
fn test_preprocessing_minmax_scaler() {
    let num_rows = 50;
    let num_features = 2;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        data.push(i as f32); // f0: 0 to 49
        data.push(100.0 - i as f32); // f1: 100 to 51
    }

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&data, num_features).expect("Fit should succeed");

    let mut transformed = data.clone();
    scaler
        .transform(&mut transformed, num_features)
        .expect("Transform should succeed");

    // Verify all values in [0, 1]
    for &v in &transformed {
        assert!((0.0..=1.0).contains(&v), "Value {} not in [0, 1]", v);
    }

    // Verify min and max values
    for f in 0..num_features {
        let feature_vals: Vec<f32> = (0..num_rows)
            .map(|r| transformed[r * num_features + f])
            .collect();

        let min = feature_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = feature_vals
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            min.abs() < 0.001,
            "Feature {} min should be 0, got {}",
            f,
            min
        );
        assert!(
            (max - 1.0).abs() < 0.001,
            "Feature {} max should be 1, got {}",
            f,
            max
        );
    }
}

/// Test RobustScaler with outliers
#[test]
fn test_preprocessing_robust_scaler_outliers() {
    let num_rows = 100;
    let num_features = 1;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    // Normal data with outliers
    for i in 0..num_rows {
        if i < 95 {
            data.push(i as f32 % 10.0); // Values 0-9
        } else {
            data.push(10000.0); // Outliers
        }
    }

    let mut robust = RobustScaler::new();
    robust.fit(&data, num_features).expect("Fit should succeed");

    let mut transformed = data.clone();
    robust
        .transform(&mut transformed, num_features)
        .expect("Transform should succeed");

    // Non-outlier values should be in reasonable range
    let non_outlier_vals: Vec<f32> = transformed[0..95].to_vec();
    let max_non_outlier = non_outlier_vals
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_non_outlier = non_outlier_vals
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);

    // Robust scaling should keep non-outliers in reasonable range
    assert!(
        max_non_outlier < 5.0,
        "Non-outliers should be scaled reasonably: {}",
        max_non_outlier
    );
    assert!(
        min_non_outlier > -5.0,
        "Non-outliers should be scaled reasonably: {}",
        min_non_outlier
    );
}

/// Test FrequencyEncoder with transform workflow
#[test]
fn test_preprocessing_frequency_encoder_workflow() {
    let categories = vec!["apple", "banana", "apple", "cherry", "apple", "banana"];

    let mut encoder = FrequencyEncoder::new().with_normalize(true);
    encoder.fit(&categories);

    assert!(encoder.is_fitted());
    assert_eq!(encoder.num_categories(), 3);

    let transformed = encoder
        .transform(&["apple", "banana", "cherry"])
        .expect("Transform should succeed");

    // apple: 3/6 = 0.5, banana: 2/6 = 0.333, cherry: 1/6 = 0.167
    assert!((transformed[0] - 0.5).abs() < 0.01);
    assert!((transformed[1] - 0.333).abs() < 0.01);
    assert!((transformed[2] - 0.167).abs() < 0.01);

    // Unknown category with default value
    let unknown = encoder.transform_single("mango");
    assert_eq!(unknown, Some(0.0), "Unknown category should return default");
}

/// Test LabelEncoder with inverse transform
#[test]
fn test_preprocessing_label_encoder_roundtrip() {
    let categories = vec!["red", "green", "blue", "red", "green"];

    let mut encoder = LabelEncoder::new();
    encoder.fit(&categories);

    let labels = encoder
        .transform(&categories)
        .expect("Transform should succeed");
    let reversed = encoder
        .inverse_transform(&labels)
        .expect("Inverse should succeed");

    assert_eq!(reversed, categories, "Roundtrip should preserve categories");

    // Verify alphabetical ordering
    assert_eq!(encoder.get_label("blue"), Some(0));
    assert_eq!(encoder.get_label("green"), Some(1));
    assert_eq!(encoder.get_label("red"), Some(2));
}

/// Test OneHotEncoder with drop_first for linear models
#[test]
fn test_preprocessing_onehot_encoder_drop_first() {
    let categories = vec!["A", "B", "C"];

    let mut encoder = OneHotEncoder::new()
        .with_drop_first(true)
        .with_unknown_strategy(UnknownStrategy::AllZeros);
    let _ = encoder.fit(&categories);

    // 3 categories with drop_first → 2 columns
    assert_eq!(encoder.num_columns(), 2);

    let encoded = encoder
        .transform(&["A", "B", "C", "unknown"])
        .expect("Transform should succeed");

    // A (dropped): [0, 0]
    // B: [1, 0]
    // C: [0, 1]
    // unknown: [0, 0]
    assert_eq!(
        encoded,
        vec![
            0.0, 0.0, // A (reference)
            1.0, 0.0, // B
            0.0, 1.0, // C
            0.0, 0.0, // unknown
        ]
    );
}

/// Test SimpleImputer with different strategies
#[test]
fn test_preprocessing_simple_imputer_strategies() {
    let num_features = 2;

    // Data with NaN values (row-major: 10 rows × 2 features)
    let data = vec![
        1.0,
        10.0,
        2.0,
        f32::NAN,
        f32::NAN,
        30.0,
        4.0,
        40.0,
        5.0,
        50.0,
        6.0,
        60.0,
        7.0,
        70.0,
        8.0,
        80.0,
        9.0,
        90.0,
        10.0,
        100.0,
    ];

    // Mean imputation
    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
    imputer
        .fit(&data, num_features)
        .expect("Fit should succeed");

    let mut imputed = data.clone();
    imputer
        .transform(&mut imputed, num_features)
        .expect("Transform should succeed");

    // No NaN values should remain
    for &v in &imputed {
        assert!(!v.is_nan(), "NaN should be imputed");
    }

    // Feature 0 mean: (1+2+4+5+6+7+8+9+10)/9 ≈ 5.78
    // Feature 1 mean: (10+30+40+50+60+70+80+90+100)/9 ≈ 58.89
    assert!(
        (imputed[4] - 5.78).abs() < 0.1,
        "Row 2, f0 should be imputed to mean"
    );
    assert!(
        (imputed[3] - 58.89).abs() < 0.1,
        "Row 1, f1 should be imputed to mean"
    );
}

// ============================================================================
// Time-Series Feature Integration Tests
// ============================================================================

use treeboost::features::{
    EwmaGenerator, LagGenerator, NaNStrategy, RollingGenerator, RollingStat, SeasonalComponent,
    SeasonalGenerator,
};

/// Test LagGenerator end-to-end with realistic time-series data
#[test]
fn test_timeseries_lag_generator_workflow() {
    // Simulate daily stock prices for 30 days (row-major: 30 rows × 2 features)
    let num_rows = 30;
    let num_features = 2;
    let mut prices: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        prices.push(100.0 + (i as f32) * 0.5 + (i as f32 * 0.3).sin() * 5.0); // Price
        prices.push(1000.0 + (i as f32) * 100.0); // Volume
    }

    // Create lags for t-1, t-7 (daily, weekly)
    let gen = LagGenerator::new(vec![1, 7]);
    let lagged = gen
        .transform(&prices, num_features)
        .expect("Transform should succeed");

    // Output: 30 rows × 6 features (2 original + 2 lag1 + 2 lag7)
    assert_eq!(lagged.len(), 30 * 6);

    // Verify lag values at row 10
    let row10_start = 10 * 6;
    let original_price = prices[10 * 2]; // row 10, feature 0
    let lag1_price = prices[9 * 2]; // row 9, feature 0
    let lag7_price = prices[3 * 2]; // row 3, feature 0

    assert_eq!(lagged[row10_start], original_price);
    assert_eq!(lagged[row10_start + 2], lag1_price); // lag1 feat0
    assert_eq!(lagged[row10_start + 4], lag7_price); // lag7 feat0

    // First 7 rows should have NaN for lag7
    for row in 0..7 {
        let lag7_idx = row * 6 + 4;
        assert!(lagged[lag7_idx].is_nan(), "Row {} lag7 should be NaN", row);
    }
}

/// Test LagGenerator with NaN strategy options
#[test]
fn test_timeseries_lag_generator_nan_strategies() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    // Keep NaN (default)
    let gen_keep = LagGenerator::new(vec![2]);
    let result_keep = gen_keep.transform(&data, 1).unwrap();
    assert!(result_keep[1].is_nan()); // row 0, lag2
    assert!(result_keep[3].is_nan()); // row 1, lag2

    // Forward fill
    let gen_ff = LagGenerator::new(vec![2]).with_nan_strategy(NaNStrategy::ForwardFill);
    let result_ff = gen_ff.transform(&data, 1).unwrap();
    assert_eq!(result_ff[1], 10.0); // row 0, lag2 = first value
    assert_eq!(result_ff[3], 10.0); // row 1, lag2 = first value

    // Constant fill
    let gen_const = LagGenerator::new(vec![2]).with_nan_strategy(NaNStrategy::constant(0.0));
    let result_const = gen_const.transform(&data, 1).unwrap();
    assert_eq!(result_const[1], 0.0); // row 0, lag2 = 0
}

/// Test RollingGenerator with multiple statistics
#[test]
fn test_timeseries_rolling_generator_workflow() {
    // Create sample time-series data (100 rows × 1 feature)
    let num_rows = 100;
    let data: Vec<f32> = (0..num_rows).map(|i| 100.0 + (i as f32) * 2.0).collect();

    let gen = RollingGenerator::new(5)
        .with_stats(vec![
            RollingStat::Mean,
            RollingStat::Std,
            RollingStat::Min,
            RollingStat::Max,
        ])
        .with_min_periods(3);

    let rolled = gen.transform(&data, 1).expect("Transform should succeed");

    // Output: 100 rows × 5 features (1 original + 4 stats)
    assert_eq!(rolled.len(), 100 * 5);

    // Check values at row 50 (full window available)
    // Window: rows 46-50, values: 192, 194, 196, 198, 200
    let row50_start = 50 * 5;
    let expected_mean = (192.0 + 194.0 + 196.0 + 198.0 + 200.0) / 5.0; // 196.0
    let expected_min = 192.0;
    let expected_max = 200.0;

    assert_eq!(rolled[row50_start], 200.0); // original value
    assert!((rolled[row50_start + 1] - expected_mean).abs() < 0.01); // rolling mean
                                                                     // std has more tolerance due to sample vs population
    assert!((rolled[row50_start + 3] - expected_min).abs() < 0.01); // rolling min
    assert!((rolled[row50_start + 4] - expected_max).abs() < 0.01); // rolling max

    // First 2 rows should have NaN (min_periods=3)
    assert!(rolled[1].is_nan()); // row 0, stat 0
    assert!(rolled[6].is_nan()); // row 1, stat 0
}

/// Test EWMA generator for trend smoothing
#[test]
fn test_timeseries_ewma_workflow() {
    // Noisy data with trend
    let data: Vec<f32> = (0..50)
        .map(|i| 10.0 * i as f32 + ((i * 7) % 13) as f32)
        .collect();

    // Create EWMA with alpha=0.3
    let gen = EwmaGenerator::new(0.3);
    let smoothed = gen.transform(&data, 1).expect("Transform should succeed");

    assert_eq!(smoothed.len(), 50);

    // Smoothed values should have less variance than original
    let orig_var = variance(&data);
    let smooth_var = variance(&smoothed);

    assert!(
        smooth_var < orig_var,
        "EWMA should reduce variance: orig={}, smooth={}",
        orig_var,
        smooth_var
    );

    // All values should be finite
    for &v in &smoothed {
        let val: f32 = v;
        assert!(val.is_finite(), "EWMA value should be finite");
    }
}

/// Test SeasonalGenerator with realistic timestamps
#[test]
fn test_timeseries_seasonal_generator_workflow() {
    // Create timestamps for one week (168 hours), hourly
    let base_ts = 1705276800.0; // 2024-01-15 00:00:00 UTC (Monday)
    let timestamps: Vec<f64> = (0..168).map(|h| base_ts + (h * 3600) as f64).collect();

    let gen = SeasonalGenerator::new(vec![
        SeasonalComponent::Hour,
        SeasonalComponent::DayOfWeek,
        SeasonalComponent::IsWeekend,
    ]);

    let features = gen.transform_timestamps(&timestamps);

    // Output: 168 rows × 3 features
    assert_eq!(features.len(), 168 * 3);

    // Check hour cycles (should go 0-23 repeatedly)
    assert_eq!(features[0], 0.0); // Hour 0
    assert_eq!(features[12 * 3], 12.0); // Hour 12
    assert_eq!(features[23 * 3], 23.0); // Hour 23
    assert_eq!(features[24 * 3], 0.0); // Hour 0 (next day)

    // Check day of week (Monday=0, ..., Sunday=6)
    assert_eq!(features[1], 0.0); // Monday (first hour)
    assert_eq!(features[24 * 3 + 1], 1.0); // Tuesday (24 hours later)
    assert_eq!(features[5 * 24 * 3 + 1], 5.0); // Saturday (5 days later)

    // Check is_weekend
    assert_eq!(features[2], 0.0); // Monday - not weekend
    assert_eq!(features[5 * 24 * 3 + 2], 1.0); // Saturday - weekend
    assert_eq!(features[6 * 24 * 3 + 2], 1.0); // Sunday - weekend
}

/// Test cyclical encoding for seasonal features
#[test]
fn test_timeseries_seasonal_cyclical_encoding() {
    let gen = SeasonalGenerator::new(vec![SeasonalComponent::Hour]).with_cyclical(true);

    // Output should be 2 features (sin, cos) per component
    assert_eq!(gen.num_features(), 2);

    // Test specific hours
    // Hour 0: sin(0) = 0, cos(0) = 1
    // Hour 6: sin(π/2) = 1, cos(π/2) = 0
    // Hour 12: sin(π) = 0, cos(π) = -1
    // Hour 18: sin(3π/2) = -1, cos(3π/2) = 0

    let base_ts = 1705276800.0; // 2024-01-15 00:00:00 UTC
    let timestamps = vec![
        base_ts,           // Hour 0
        base_ts + 21600.0, // Hour 6
        base_ts + 43200.0, // Hour 12
        base_ts + 64800.0, // Hour 18
    ];

    let features = gen.transform_timestamps(&timestamps);
    assert_eq!(features.len(), 8); // 4 timestamps × 2 features

    // Hour 0
    assert!(features[0].abs() < 0.01); // sin ≈ 0
    assert!((features[1] - 1.0).abs() < 0.01); // cos ≈ 1

    // Hour 6
    assert!((features[2] - 1.0).abs() < 0.01); // sin ≈ 1
    assert!(features[3].abs() < 0.01); // cos ≈ 0

    // Hour 12
    assert!(features[4].abs() < 0.01); // sin ≈ 0
    assert!((features[5] + 1.0).abs() < 0.01); // cos ≈ -1

    // Hour 18
    assert!((features[6] + 1.0).abs() < 0.01); // sin ≈ -1
    assert!(features[7].abs() < 0.01); // cos ≈ 0
}

/// Test combining time-series features for forecasting workflow
#[test]
fn test_timeseries_combined_feature_engineering() {
    // Simulate 60 days of daily data
    let num_rows = 60;
    let data: Vec<f32> = (0..num_rows)
        .map(|i| {
            // Trend + weekly seasonality + noise
            let trend = 100.0 + (i as f32) * 0.5;
            let seasonal = 10.0 * ((i % 7) as f32 / 3.0).sin();
            let noise = ((i * 17) % 11) as f32 - 5.0;
            trend + seasonal + noise
        })
        .collect();

    // Apply lag features
    let lag_gen = LagGenerator::new(vec![1, 7]);
    let lagged = lag_gen
        .transform(&data, 1)
        .expect("Lag transform should succeed");

    // Apply rolling features to lagged data (on original feature only)
    // Note: in real workflow, you'd apply to specific columns
    let roll_gen = RollingGenerator::new(7)
        .with_stats(vec![RollingStat::Mean, RollingStat::Std])
        .with_min_periods(3);
    let rolled = roll_gen
        .transform(&data, 1)
        .expect("Roll transform should succeed");

    // Both should have correct number of rows
    assert_eq!(lagged.len() / 3, num_rows); // 1 orig + 2 lags
    assert_eq!(rolled.len() / 3, num_rows); // 1 orig + 2 stats

    // After row 7, all features should be available
    for row in 7..num_rows {
        // Lag features
        let lag_start = row * 3;
        assert!(
            !lagged[lag_start + 1].is_nan(),
            "Lag1 at row {} should be valid",
            row
        );
        assert!(
            !lagged[lag_start + 2].is_nan(),
            "Lag7 at row {} should be valid",
            row
        );

        // Rolling features
        let roll_start = row * 3;
        assert!(
            !rolled[roll_start + 1].is_nan(),
            "Rolling mean at row {} should be valid",
            row
        );
        assert!(
            !rolled[roll_start + 2].is_nan(),
            "Rolling std at row {} should be valid",
            row
        );
    }
}

/// Helper function to calculate variance
fn variance(data: &[f32]) -> f32 {
    let valid: Vec<f32> = data.iter().filter(|x| x.is_finite()).cloned().collect();
    if valid.is_empty() {
        return 0.0;
    }
    let mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
    valid.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / valid.len() as f32
}

// ============================================================================
// Outlier Detection Integration Tests
// ============================================================================

/// Test IQR-based outlier detection with capping
#[test]
fn test_outlier_iqr_cap_workflow() {
    // Create dataset with outliers
    let mut data: Vec<f32> = Vec::with_capacity(102 * 2);

    // 100 normal values
    for i in 0..100 {
        data.push(i as f32); // f0: 0-99
        data.push((i as f32 * 2.0) + 10.0); // f1: 10-208
    }

    // 2 outlier rows
    data.push(1000.0); // f0 outlier (high)
    data.push(100.0); // f1 normal
    data.push(-500.0); // f0 outlier (low)
    data.push(5000.0); // f1 outlier (high)

    let mut detector = OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Cap);

    detector.fit(&data, 2).expect("Fit should succeed");

    let names = vec!["f0".into(), "f1".into()];
    let result = detector
        .transform(&mut data, 2, &names)
        .expect("Transform should succeed");

    // Verify outliers were capped
    if let TransformResult::Capped { outlier_count } = result {
        assert!(outlier_count >= 2, "Should have capped at least 2 outliers");
    } else {
        panic!("Expected Capped result");
    }

    // Verify capped values are within bounds
    for feat in 0..2 {
        let bounds = &detector.bounds()[feat];
        for row in 0..102 {
            let val = data[row * 2 + feat];
            assert!(
                val >= bounds.lower && val <= bounds.upper,
                "Value {} should be within bounds [{}, {}]",
                val,
                bounds.lower,
                bounds.upper
            );
        }
    }
}

/// Test Z-score based outlier detection with flagging
#[test]
fn test_outlier_zscore_flag_workflow() {
    // Create dataset with clear outliers
    let num_rows = 50;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * 2);

    // Normal data centered around 50
    for i in 0..num_rows - 2 {
        data.push(50.0 + (i as f32 % 10.0) - 5.0); // f0: 45-54
        data.push(100.0 + (i as f32 % 5.0)); // f1: 100-104
    }

    // Add outliers (> 3σ from mean)
    data.push(200.0); // f0 extreme outlier
    data.push(102.0); // f1 normal
    data.push(47.0); // f0 normal
    data.push(500.0); // f1 extreme outlier

    let mut detector =
        OutlierDetector::new(OutlierMethod::zscore()).with_action(OutlierAction::Flag);

    detector.fit(&data, 2).expect("Fit should succeed");

    let names = vec!["feature_0".into(), "feature_1".into()];
    let result = detector
        .transform(&mut data, 2, &names)
        .expect("Transform should succeed");

    if let TransformResult::Flagged {
        indicators,
        indicator_names,
    } = result
    {
        // Verify indicator names
        assert_eq!(indicator_names.len(), 2);
        assert_eq!(indicator_names[0], "feature_0_outlier");
        assert_eq!(indicator_names[1], "feature_1_outlier");

        // Count flagged outliers
        let flagged: usize = indicators.iter().filter(|&&v| v > 0.0).count();
        assert!(
            flagged >= 2,
            "Should flag at least 2 outliers, got {}",
            flagged
        );
    } else {
        panic!("Expected Flagged result");
    }
}

/// Test outlier removal workflow
#[test]
fn test_outlier_remove_workflow() {
    // Create dataset: 10 normal rows + 2 outlier rows
    let mut data: Vec<f32> = Vec::new();

    // 10 normal rows
    for i in 0..10 {
        data.push(i as f32 + 1.0); // f0: 1-10
        data.push((i as f32 + 1.0) * 10.0); // f1: 10-100
    }

    // 2 outlier rows
    data.push(1000.0); // f0 outlier
    data.push(50.0); // f1 normal
    data.push(5.0); // f0 normal
    data.push(10000.0); // f1 outlier

    let mut detector =
        OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Remove);

    detector.fit(&data, 2).expect("Fit should succeed");

    let names = vec!["f0".into(), "f1".into()];
    let result = detector
        .transform(&mut data, 2, &names)
        .expect("Transform should succeed");

    if let TransformResult::Removed {
        cleaned_data,
        kept_indices,
        removed_count,
    } = result
    {
        // Should have removed the outlier rows
        assert!(removed_count >= 1, "Should remove at least 1 row");
        assert!(kept_indices.len() < 12, "Should have fewer than 12 rows");
        assert_eq!(
            cleaned_data.len(),
            kept_indices.len() * 2,
            "Cleaned data should match kept indices"
        );

        // Verify remaining data has no outliers
        for row in 0..kept_indices.len() {
            for feat in 0..2 {
                let val = cleaned_data[row * 2 + feat];
                assert!(
                    !detector.is_outlier(val, feat),
                    "Cleaned data should have no outliers"
                );
            }
        }
    } else {
        panic!("Expected Removed result");
    }
}

/// Test combining outlier detection with scaling
#[test]
fn test_outlier_then_scale_pipeline() {
    // Create dataset with outliers
    let mut data: Vec<f32> = Vec::new();

    for i in 0..50 {
        data.push(i as f32 * 2.0); // f0: 0-98
    }
    data.push(10000.0); // extreme outlier

    // Step 1: Cap outliers
    let mut detector = OutlierDetector::new(OutlierMethod::iqr()).with_action(OutlierAction::Cap);
    detector.fit(&data, 1).expect("Fit should succeed");
    detector
        .transform(&mut data, 1, &["f0".into()])
        .expect("Transform should succeed");

    // Step 2: Scale the capped data
    let mut scaler = StandardScaler::new();
    scaler.fit(&data, 1).expect("Fit should succeed");
    scaler
        .transform(&mut data, 1)
        .expect("Transform should succeed");

    // Verify: scaled data should have reasonable values
    for val in &data {
        assert!(
            val.abs() < 10.0,
            "Scaled values should be reasonable, got {}",
            val
        );
    }

    // Mean should be ~0
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.1, "Mean should be ~0, got {}", mean);
}

/// Test outlier detection with multifeature data
#[test]
fn test_outlier_multifeature() {
    // 20 rows × 4 features
    let num_rows = 20;
    let num_features = 4;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows - 1 {
        data.push(i as f32); // f0
        data.push(i as f32 * 2.0); // f1
        data.push(100.0 - i as f32); // f2
        data.push((i % 5) as f32); // f3
    }

    // One row with outliers in multiple features
    data.push(1000.0); // f0 outlier
    data.push(5000.0); // f1 outlier
    data.push(50.0); // f2 normal
    data.push(2.0); // f3 normal

    let mut detector = OutlierDetector::new(OutlierMethod::iqr());
    detector
        .fit(&data, num_features)
        .expect("Fit should succeed");

    let counts = detector
        .outlier_counts(&data, num_features)
        .expect("Count should succeed");

    assert_eq!(counts.len(), num_features);
    assert!(counts[0] >= 1, "f0 should have outliers");
    assert!(counts[1] >= 1, "f1 should have outliers");
}
