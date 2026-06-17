//! Target Encoding Example
//!
//! Demonstrates handling of high-cardinality categorical features using:
//! - Count-Min Sketch for rare category filtering
//! - Ordered Target Encoding for categorical features
//! - Memory-efficient data processing
//!
//! Run with: cargo run --release --example target_encoding

fn main() {
    println!("{}", "=".repeat(70));
    println!("TreeBoost: Target Encoding Example");
    println!("{}", "=".repeat(70));
    println!();

    // This example demonstrates the concepts behind TreeBoost's categorical handling
    // In practice, these are integrated into the DatasetLoader and training pipeline

    println!("1. Categorical Feature Handling Demonstration");
    println!();

    // Show Count-Min Sketch usage
    println!("   Count-Min Sketch:");
    println!("   - Efficiently estimates frequencies of categorical values");
    println!("   - Used to identify and filter rare categories");
    println!("   - Memory usage: O(1/eps * ln(1/delta))");
    println!("   - Example parameters:");
    println!("     * eps: 0.001 (1% error tolerance)");
    println!("     * confidence: 0.99 (99% confidence)");
    println!("     * min_count: 5 (categories with <5 samples -> 'unknown')");
    println!();

    // Simulate Count-Min Sketch behavior
    let categories = vec![
        "category_1",
        "category_2",
        "category_3",
        "category_1",
        "category_4",
        "category_1",
        "category_5",
        "category_2",
        "category_1",
        "category_6",
        "category_1",
        "category_7",
        "category_2",
        "category_8",
        "category_1",
    ];

    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for cat in &categories {
        *counts.entry(cat).or_insert(0) += 1;
    }

    println!("   Category Frequencies:");
    let mut sorted_counts: Vec<_> = counts.iter().collect();
    sorted_counts.sort_by_key(|&(_, &count)| std::cmp::Reverse(count));

    for (cat, count) in &sorted_counts {
        let freq = **count as f32 / categories.len() as f32;
        println!(
            "     {:15}: {:2} occurrences ({:.1}%)",
            cat,
            count,
            freq * 100.0
        );
    }
    println!();

    // Show Ordered Target Encoding concept
    println!("2. Ordered Target Encoding");
    println!();
    println!("   Approach:");
    println!("   - Process training data sequentially");
    println!("   - For each category, encode with mean of target values seen so far");
    println!("   - Prevents target leakage (uses only past information)");
    println!("   - Applies smoothing to reduce variance for rare categories");
    println!();

    // Simulate ordered target encoding
    let training_data = [
        ("cat_a", 1.2),
        ("cat_b", 2.5),
        ("cat_a", 1.8),
        ("cat_c", 0.9),
        ("cat_b", 2.1),
        ("cat_a", 1.5),
    ];

    println!("   Training Sequence:");
    let mut encodings: std::collections::HashMap<&str, Vec<f32>> = std::collections::HashMap::new();

    for (i, (cat, target)) in training_data.iter().enumerate() {
        encodings.entry(cat).or_default().push(*target);
        let mean_so_far = encodings[cat].iter().sum::<f32>() / encodings[cat].len() as f32;
        println!(
            "     Row {}: category='{}', target={:.1}, encoding (mean so far)={:.4}",
            i + 1,
            cat,
            target,
            mean_so_far
        );
    }
    println!();

    println!("   Final Encodings (without smoothing):");
    for cat in &["cat_a", "cat_b", "cat_c"] {
        if let Some(targets) = encodings.get(cat) {
            let mean = targets.iter().sum::<f32>() / targets.len() as f32;
            println!("     {}: {:.4}", cat, mean);
        }
    }
    println!();

    // Show smoothing effect
    println!("3. Smoothing Effect");
    println!();
    println!("   Smoothing reduces variance for rare categories:");
    println!(
        "   Smoothed encoding = (count * mean + smoothing * global_mean) / (count + smoothing)"
    );
    println!();

    let global_mean =
        training_data.iter().map(|(_, t)| t).sum::<f32>() / training_data.len() as f32;
    let smoothing = 10.0;

    println!("   Parameters:");
    println!("     Global mean (all data): {:.4}", global_mean);
    println!("     Smoothing parameter: {:.1}", smoothing);
    println!();

    println!("   Smoothed Encodings:");
    for cat in &["cat_a", "cat_b", "cat_c"] {
        if let Some(targets) = encodings.get(cat) {
            let count = targets.len() as f32;
            let mean = targets.iter().sum::<f32>() / count;
            let smoothed = (count * mean + smoothing * global_mean) / (count + smoothing);
            println!(
                "     {:5}: raw={:.4}, count={:.0}, smoothed={:.4}",
                cat, mean, count, smoothed
            );
        }
    }
    println!();

    // Integration with TreeBoost
    println!("4. Integration with TreeBoost");
    println!();
    println!("   In TreeBoost, categorical handling is automatic:");
    println!();
    println!("   Rust Example:");
    println!("   ```rust");
    println!("   let config = GBDTConfig::new()");
    println!("       .with_use_target_encoding(true)  // Enable encoding");
    println!("       .with_cms_params(eps: 0.001, confidence: 0.99)");
    println!("       .with_categorical_features(vec![0, 2, 5]);  // Which columns");
    println!();
    println!("   let model = GBDTModel::train(&features, num_features, &targets, config, None)?;");
    println!("   ```");
    println!();
    println!("   Python Example:");
    println!("   ```python");
    println!("   config = GBDTConfig()");
    println!("   config.use_target_encoding = True");
    println!("   config.cms_params = {{\"eps\": 0.001, \"confidence\": 0.99}}");
    println!("   config.categorical_features = [0, 2, 5]");
    println!();
    println!("   model = GBDTModel.train(X, y, config)");
    println!("   ```");
    println!();

    println!("5. Performance Characteristics");
    println!();
    println!("   Time Complexity:");
    println!("     - Count-Min Sketch: O(1) per update, O(k ln(1/delta)) for query");
    println!("     - Ordered Target Encoding: O(n) for n training samples");
    println!();
    println!("   Space Complexity:");
    println!("     - Count-Min Sketch: O(1/eps * ln(1/delta))");
    println!("     - Encoding map: O(k) where k = number of unique categories");
    println!();
    println!("   Handles:");
    println!("     - Arbitrary cardinality (100s to millions of categories)");
    println!("     - New categories at inference time (-> 'unknown' fallback)");
    println!("     - Imbalanced categories (rare categories smoothed)");
    println!();

    println!("6. Advantages over One-Hot Encoding");
    println!();
    println!("   One-Hot Encoding Issues:");
    println!("     - Explodes feature dimensionality (k categories -> k columns)");
    println!("     - Memory intensive for high-cardinality features");
    println!("     - Sparse representation (mostly zeros)");
    println!();
    println!("   Target Encoding Benefits:");
    println!("     - Single column per categorical feature");
    println!("     - Dense representation (no sparsity)");
    println!("     - Incorporates target information (supervised)");
    println!("     - Memory efficient even for millions of categories");
    println!();

    println!("{}", "=".repeat(70));
    println!("Example demonstration completed!");
    println!();
    println!("In practice, use DatasetLoader or DataPipeline for automatic");
    println!("categorical encoding in your TreeBoost pipeline.");
    println!("{}", "=".repeat(70));
}
