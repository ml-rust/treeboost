# Contributing to TreeBoost

Thank you for your interest in TreeBoost.

## Before You Start

Open an issue before you start anything non-trivial, such as a new backend, model type, or public API.
Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and Test

Run these commands from the repository root.

```bash
cargo build --release
cargo test
cargo fmt --all --check
cargo clippy
```

Build with optional features when you change them:

```bash
cargo build --release --features gpu
cargo build --release --features cuda
cargo build --release --features mmap
```

Build and test the Python bindings:

```bash
pip install maturin
maturin develop --release --features python
pytest python/tests
```

GPU benchmarks need CUDA. Run them with `cargo bench --bench competitors --features gpu,cuda`.

## Conventions

- Use Rust 2021 edition. Format with `cargo fmt`. Keep `cargo clippy` clean.
- Name modules and functions in `snake_case`. Name types and traits in `CamelCase`.
- Name config builders `with_*`, such as `with_num_rounds` and `with_backend`.
- Place Rust integration tests in `tests/`. Name each file after the feature it covers, such as `tests/encoding.rs`.
- Place Python tests in `python/tests/` with the `test_*.py` name.
- Add tests for new backends, feature flags, and serialization paths.
- Keep the `python`, `gpu`, and `cuda` features optional. The default build must not require them.

## Pull Requests

- Describe the change, the reason for it, and the commands you ran.
- Include tests for new behavior and for each bug fix.
- Update [docs/API.md](docs/API.md) when you change a public type or method.
- Include CLI output or a screenshot when you change CLI behavior.
- Link the related issue when one exists.
