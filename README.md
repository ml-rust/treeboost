<div align="center">

<img src="images/treeboost.jpeg" alt="TreeBoost" width="640">

# treeboost

<h3>Gradient-boosted decision trees for tabular data, in Rust.</h3>

<p>
Trains a baseline model in one call and exposes full control over boosting modes, configs, and backends. Backends: Scalar, AVX-512, SVE2, WGPU, and CUDA.
</p>

<p>
  <a href="https://docs.rs/treeboost"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/treeboost"><strong>Crate</strong></a>
  ·
  <a href="#models"><strong>Models</strong></a>
  ·
  <a href="#backends"><strong>Backends</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://crates.io/crates/treeboost">
    <img src="https://img.shields.io/crates/v/treeboost" alt="crates.io">
  </a>
  <a href="https://docs.rs/treeboost">
    <img src="https://img.shields.io/docsrs/treeboost" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/treeboost" alt="License">
  </a>
  <a href="https://github.com/ml-rust/treeboost/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/treeboost?style=social" alt="GitHub stars">
  </a>
</p>

</div>

TreeBoost trains gradient-boosted tree models for tabular data. It handles time series, missing values, mixed feature types, and multi-label targets. It provides a Rust API, a CLI, and Python bindings. See [docs/API.md](docs/API.md) for the full API reference.

## Quick Start

Add the library to a Rust project:

```bash
cargo add treeboost
```

Train a baseline from a CSV file and save it:

```rust
use treeboost::auto_train_csv;

fn main() -> treeboost::Result<()> {
    let model = auto_train_csv("data.csv", "target")?;
    println!("{}", model.summary());

    model.save("model.rkyv")?;
    model.save_config("config.json")?;
    model.save_trb("model.trb", "initial training")?;
    Ok(())
}
```

Training writes three files:

| File          | Contents                                          |
| ------------- | ------------------------------------------------- |
| `model.rkyv`  | Model for inference and deployment                |
| `config.json` | `UniversalConfig` of the run. See [docs/config.md](docs/config.md). |
| `model.trb`   | Model with an update history for incremental learning |

## Models

TreeBoost has three API levels. You can move between them without changing your dataset format.

| Model            | Use it for                                                                |
| ---------------- | ------------------------------------------------------------------------- |
| `AutoModel`      | One call trains a baseline and reports the chosen mode.                   |
| `UniversalModel` | Choose a `BoostingMode` and tune `UniversalConfig` without low-level code. |
| `GBDTModel`      | Full control over `GBDTConfig`, backend selection, and benchmarks.        |

Use this workflow:

1. Train with `AutoModel` for a baseline and a training report.
2. Read the report and `config.json` to see the chosen mode.
3. Refine with `UniversalModel` or `GBDTModel` for accuracy, constraints, or incremental updates.

### Inference

Load a saved model and predict on a DataFrame with the training feature columns:

```rust
use treeboost::UniversalModel;

let model = UniversalModel::load("model.rkyv")?;
let predictions = model.predict_df(&df)?;
```

Load a TRB file with `UniversalModel::load_trb("model.trb")?` instead.

### UniversalModel

`UniversalModel::train` takes a `BinnedDataset`. `DatasetLoader` creates it from Parquet or CSV.

```rust
use treeboost::dataset::DatasetLoader;
use treeboost::loss::MseLoss;
use treeboost::{BoostingMode, UniversalConfig, UniversalModel};

let loader = DatasetLoader::new(255);
let dataset = loader.load_parquet("data.parquet", "target", None)?;

let config = UniversalConfig::new()
    .with_mode(BoostingMode::LinearThenTree)
    .with_num_rounds(100)
    .with_linear_rounds(10)
    .with_learning_rate(0.1)?;

let model = UniversalModel::train(&dataset, config, &MseLoss)?;
let predictions = model.predict(&dataset);
```

Choose the mode from the shape of your data:

| Your data                                  | Mode                           |
| ------------------------------------------ | ------------------------------ |
| General tabular data with categoricals     | `BoostingMode::PureTree`       |
| Time series, trends, and extrapolation     | `BoostingMode::LinearThenTree` |
| Noisy data with a variance-reduction goal  | `BoostingMode::RandomForest`   |

### GBDTModel

`GBDTModel::train` takes a flat `f32` slice. Each row holds `num_features` values.

```rust
use treeboost::{GBDTConfig, GBDTModel};

let config = GBDTConfig::new()
    .with_num_rounds(200)
    .with_max_depth(6)
    .with_learning_rate(0.05);

let model = GBDTModel::train(&features, num_features, &targets, config, None)?;
```

### Multi-label classification

For samples with several independent labels, use `AutoModel::train_multilabel`. See [docs/features.md](docs/features.md#multi-label-classification) for the example.

## Backends

TreeBoost selects a backend automatically. Set `BackendType` to override the choice.

```rust
use treeboost::backend::BackendType;
use treeboost::{GBDTConfig, GBDTModel};

let config = GBDTConfig::new().with_backend(BackendType::Scalar);
let model = GBDTModel::train(&features, num_features, &targets, config, None)?;
```

`BackendType::Auto` is the default. It selects the first available backend in this order:

| Priority | Backend  | Requirement                  |
| -------- | -------- | ---------------------------- |
| 1        | CUDA     | `cuda` feature               |
| 2        | WGPU     | `gpu` feature                |
| 3        | AVX-512  | x86-64 CPU with AVX-512      |
| 4        | SVE2     | ARM CPU with SVE2            |
| 5        | Scalar   | None. Used when no other backend is available. |

## CLI

Install the `treeboost` binary:

```bash
cargo install treeboost
```

Train, predict, and inspect a model:

```bash
treeboost train --data data.csv --target price --output model.rkyv
treeboost predict --model model.rkyv --data test.csv --output predictions.json
treeboost info --model model.rkyv --importances
```

The `train` and `predict` commands accept Parquet or CSV input. The `train` command writes `.trb` output when `--output` ends in `.trb`.

| Command   | Required flags                                    |
| --------- | ------------------------------------------------- |
| `train`   | `-d, --data <path>`, `-t, --target <column>`, `-o, --output <path>` |
| `predict` | `-m, --model <path>`, `-d, --data <path>`, `-o, --output <path>` |
| `info`    | `-m, --model <path>` (`--importances` shows feature importance) |

Run `treeboost <command> --help` for all flags.

## Python

TreeBoost ships Python bindings for `GBDTConfig` and `GBDTModel`. Build them from source with the `python` feature. See [docs/python.md](docs/python.md) for the build steps and an example.

## Features

- **AutoML mode selection:** `AutoModel` runs probes and reports the mode it chose.
- **Hybrid linear and tree mode:** `LinearThenTree` fits trends and trees together.
- **Preprocessing with the model:** encoders, scalers, and imputers save with the model.
- **Conformal prediction:** prediction intervals from split conformal calibration.
- **Incremental learning:** TRB files store updates, and drift checks run before each update.

## Installation and Feature Flags

```bash
cargo add treeboost --features gpu
```

| Feature  | Enables                                                 |
| -------- | ------------------------------------------------------- |
| `gpu`    | WGPU backend for Vulkan, Metal, and DX12 GPUs           |
| `cuda`   | CUDA backend. Implies `gpu`.                            |
| `mmap`   | Memory-mapped TRB reading with `MmapTrbReader`          |
| `python` | PyO3 Python bindings                                    |

## Documentation

- [API Reference](docs/API.md): types, methods, and examples
- [Advanced Features](docs/features.md): feature matrix and incremental learning
- [Saved Config](docs/config.md): fields of the saved `config.json`
- [Python Bindings](docs/python.md): build steps and example
- [Preset Guide](docs/PRESET_GUIDE.md): choosing a preset

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
