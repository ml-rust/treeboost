# TreeBoost Advanced Features

This page lists advanced capabilities and their entry points. See [API.md](API.md) for full signatures and examples.

## Feature Matrix

| Category           | Capability                        | Use case                              | Entry point                                    |
| ------------------ | --------------------------------- | ------------------------------------- | ---------------------------------------------- |
| **Classification** | Multi-label classification        | Multi-tag prediction, multi-diagnosis | `AutoModel::train_multilabel()`                |
|                    | Threshold tuning                  | Per-label threshold selection         | `AutoModel::tune_thresholds()`                 |
|                    | Multi-class classification        | Single category from several classes  | `GBDTConfig::with_multiclass_logloss()`        |
| **Model updates**  | Incremental learning (TRB format) | Updates on new data                   | `UniversalModel::update()`                     |
|                    | Update records                    | Append an update to a TRB file        | `save_trb_update()`                            |
|                    | Memory-mapped reading             | Large model files (`mmap` feature)    | `MmapTrbReader`                                |
| **Monitoring**     | Drift detection                   | PSI, Jensen-Shannon, and KS checks    | `IncrementalDriftDetector::check_update()`     |
|                    | Drift history                     | Long-term monitoring                  | `DriftHistory`                                 |
| **Ensembles**      | Multi-seed training               | Variance reduction                    | `UniversalConfig::with_ensemble_seeds()`       |
|                    | Stacked blending                  | Meta-learner combination              | `StackingStrategy::Ridge`                      |
| **Constraints**    | Monotonic constraints             | Domain rules on feature effects       | `TreeConfig::with_monotonic_constraints()`     |
|                    | Interaction constraints           | Limit which features interact         | `TreeConfig::with_interaction_constraints()`   |
| **Encoding**       | Ordered target encoding           | High-cardinality categoricals         | `OrderedTargetEncoder`                         |
|                    | Count-min sketch filtering        | Rare category handling                | `CategoryFilter`                               |
| **Features**       | Time-series (lag, rolling, EWMA)  | Panel data, forecasting               | `LagGenerator`, `RollingGenerator`, `EwmaGenerator` |
|                    | Cross-sectional (poly, ratio)     | Feature engineering                   | `PolynomialGenerator`, `RatioGenerator`        |
| **Preprocessing**  | Incremental scaler                | Adaptive scaling over a stream        | `StandardScaler::with_forget_factor()`         |
|                    | Outlier detection                 | IQR and z-score outlier handling      | `OutlierDetector`, `OutlierMethod::Iqr`, `OutlierMethod::ZScore` |
|                    | Outlier-resistant scaling         | Scaling for outlier-heavy data        | `RobustScaler`                                 |
| **Uncertainty**    | Split conformal prediction        | Prediction intervals                  | `GBDTConfig::with_conformal()`                 |

## Multi-Label Classification

Use multi-label training when one sample carries several independent labels, such as multi-tag classification.
`AutoModel::train_multilabel` uses `LinearThenTree` mode by default.

```rust
use treeboost::AutoModel;

let target_cols = vec!["tag_finance", "tag_tech", "tag_urgent"];
let mut model = AutoModel::train_multilabel(&df, &target_cols)?;

let probs = model.predict_proba_multilabel(&df)?; // Vec<Vec<f32>>
let labels = model.predict_labels(&df)?;          // Vec<Vec<bool>>, threshold 0.5

model.tune_thresholds(&val_df, &target_cols)?;    // per-label thresholds
let tuned = model.predict_labels_tuned(&df)?;     // Vec<Vec<bool>>
```

## Incremental Learning

This example trains a model, saves it as a TRB file, and updates it after a drift check.

```rust
use treeboost::loss::MseLoss;
use treeboost::monitoring::IncrementalDriftDetector;
use treeboost::{AutoModel, UniversalModel};

// 1. Initial training
let auto = AutoModel::train(&df, "target")?;
auto.inner().save_trb("model.trb", "Initial training")?;

// 2. Production: load the model and build a drift reference.
//    train_data is the BinnedDataset used for training.
let mut model = UniversalModel::load_trb("model.trb")?;
let detector = IncrementalDriftDetector::from_dataset(&train_data);

// 3. Check drift before each update.
//    new_data is a BinnedDataset with the same feature columns.
let result = detector.check_update(&new_data);
if !result.has_critical_drift() {
    let report = model.update(&new_data, &MseLoss, 10)?;
    model.save_trb_update("model.trb", new_data.num_rows(), "Weekly update")?;
} else {
    eprintln!("Critical drift detected: {}", result);
}
```

## Why These Features Matter

- **Incremental learning:** add boosting rounds on new rows instead of retraining from scratch.
- **Drift detection:** compare new data with the training distribution before an update.
- **Constraints:** enforce domain rules, such as a monotonic effect of one feature.
- **Rare categories:** `CategoryFilter` limits the category set with a count-min sketch.
- **Conformal prediction:** build prediction intervals from split conformal calibration.
