//! Test panel data detection and era-based IC in AutoBuilder

use polars::prelude::*;
use treeboost::dataset::DataPipeline;

#[test]
fn test_panel_data_detection_with_datapipeline() {
    // Create a DataFrame with panel structure:
    // - "code" column (groups): A, B, C, D
    // - "date" column (time): 20240101, 20240102, 20240103, 20240104, 20240105
    // - features and target
    // Total: 4 groups × 5 dates = 20 rows

    let dates = vec![
        20240101, 20240101, 20240101, 20240101, // Date 1: A, B, C, D
        20240102, 20240102, 20240102, 20240102, // Date 2: A, B, C, D
        20240103, 20240103, 20240103, 20240103, // Date 3: A, B, C, D
        20240104, 20240104, 20240104, 20240104, // Date 4: A, B, C, D
        20240105, 20240105, 20240105, 20240105, // Date 5: A, B, C, D
    ];

    let codes = vec![
        "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B",
        "C", "D",
    ];

    let feature1 = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
    ];

    let target = vec![
        0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45, 0.12, 0.22, 0.32, 0.42, 0.18, 0.28, 0.38, 0.48,
        0.14, 0.24, 0.34, 0.44,
    ];

    let df = df! {
        "code" => codes,
        "date" => dates,
        "feature1" => feature1,
        "target" => target
    }
    .expect("Failed to create DataFrame");

    println!("\n=== Testing Panel Data Detection ===");
    println!("DataFrame:");
    println!("{:?}", df);

    // Test DataPipeline should create dataset with era indices
    println!("\n--- DataPipeline with era_column ---");
    let pipeline = DataPipeline::with_defaults();
    let (dataset, _state, _filtered_df) = pipeline
        .process_for_training(df.clone(), "target", None, Some("date"))
        .expect("process_for_training failed");

    println!("Dataset has eras: {}", dataset.has_eras());
    println!("Num eras: {}", dataset.num_eras());

    assert!(
        dataset.has_eras(),
        "Dataset should have era indices when era_column is provided"
    );
    assert_eq!(dataset.num_eras(), 5, "Should have 5 unique dates");

    if let Some(eras) = dataset.era_indices() {
        println!("Era indices: {:?}", eras);
        // Should be [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]
        assert_eq!(eras[0], eras[1]);
        assert_eq!(eras[1], eras[2]);
        assert_eq!(eras[2], eras[3]);
        assert_ne!(eras[3], eras[4]);

        // Verify first 4 rows are era 0
        for &era in &eras[0..4] {
            assert_eq!(era, 0, "First 4 rows should be era 0");
        }
        // Verify next 4 rows are era 1
        for &era in &eras[4..8] {
            assert_eq!(era, 1, "Rows 4-7 should be era 1");
        }
    }

    println!("\n✅ Panel data detection test PASSED!");
}
