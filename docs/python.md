# Python Bindings

The Python bindings build from source. They expose `GBDTConfig` and `GBDTModel` through the `treeboost` module.

## Build

Install maturin and build the `python` feature from the repository root:

```bash
pip install maturin
maturin develop --release --features python
```

## Example

```python
import numpy as np
from treeboost import GBDTConfig, GBDTModel

X = np.random.randn(10000, 20).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(10000) * 0.1).astype(np.float32)

config = GBDTConfig()
config.num_rounds = 100
config.max_depth = 6
config.learning_rate = 0.1

model = GBDTModel.train(X, y, config)
```

`GBDTModel.train` also accepts `feature_names` and `output_dir`. Pass `output_dir` to write `model.rkyv` and `config.json` to that directory.

See [API.md](API.md) for the full `GBDTConfig` and `GBDTModel` reference.
