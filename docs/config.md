# Saved Config Reference

`AutoModel::save_config` writes the `UniversalConfig` of a trained model to a JSON file. The library does not read this file back. To rerun with the same values, copy them into `UniversalConfig` builders in code.

The file below is abridged. The saved file holds every field, including `colsample`, `calibration_ratio`, `conformal_quantile`, `seed`, and `linear_selection_mode`.

```json
{
  "mode": "LinearThenTree",
  "num_rounds": 120,
  "learning_rate": 0.1,
  "subsample": 0.9,
  "validation_ratio": 0.1,
  "early_stopping_rounds": 20,
  "linear_rounds": 10,
  "tree_config": {
    "max_depth": 6,
    "max_leaves": 31,
    "lambda": 1.0,
    "min_samples_leaf": 20,
    "colsample": 0.8
  },
  "linear_config": {
    "lambda": 1.0,
    "l1_ratio": 0.0,
    "shrinkage_factor": 0.3
  }
}
```

The values above are an example, not defaults.

See [API.md](API.md) for the meaning of each field.
