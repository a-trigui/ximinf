# ximinf

**Simulation-Based Inference of Cosmological Parameters in JAX using Type Ia Supernovae.**

`ximinf` is a Python package for performing simulation-based inference (SBI) on cosmological parameters using type Ia supernova (SN Ia) data, built on top of JAX for fast, differentiable, and GPU/TPU-accelerated computation.

[![PyPI](https://img.shields.io/pypi/v/ximinf)](https://pypi.org/project/ximinf/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://ximinf.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Installation

```bash
pip install ximinf
```

Requires Python >= 3.10.

## Quickstart

```python
import ximinf
```

See the [documentation](https://ximinf.readthedocs.io) for full usage examples covering simulation, training, and inference.

## Features

`ximinf` is organized into five core modules:

- **`ximinf.generate_sim`** — Generates simulated SN Ia datasets. Includes Latin Hypercube Sampling (LHS) of cosmological/nuisance parameters under configurable priors (uniform, gaussian, half-gaussian, log-uniform, exponential, etc.), and per-dataset simulation via `skysurvey`.
- **`ximinf.selection_effects`** — Tools for injecting realistic observational selection effects (e.g. Malmquist bias) into simulated supernova samples via stochastic magnitude-limited detection.
- **`ximinf.nn_train`** — Training utilities for the neural network classifiers used in the inference step, including loss/accuracy functions, training/validation loops with early stopping, and JAX device setup helpers.
- **`ximinf.nn_test`** — Diagnostics for validating inference quality, notably TARP (Tests of Accuracy with Random Points) coverage statistics computed across parameter groups.
- **`ximinf.nn_inference`** — Neural posterior/likelihood-ratio inference. Builds and runs BlackJAX NUTS samplers over grouped parameter blocks, combining trained neural network outputs with analytic log-priors.

## Typical workflow

1. **Simulate** — Draw parameter sets with `generate_sim.scan_params` and generate mock SN Ia samples with `generate_sim.simulate_one`.
2. **(Optional) Apply selection effects** — Use `selection_effects.apply_malmquist_bias` to emulate realistic detection thresholds.
3. **Train** — Train per-group neural classifiers on simulated data with `nn_train.train_loop`.
4. **Infer** — Sample the posterior over cosmological parameters with `nn_inference.sample_posterior` / `nn_inference.inference_loop`, using a NUTS kernel built on the trained networks.
5. **Validate** — Check calibration of the resulting posteriors with `nn_test.compute_ecp_tarp_groups`.

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for details.

## Author

**Adam Trigui** — a.trigui@ip2i.in2p3.fr
IP2I Lyon

<!-- ## Citing

If you use `ximinf` in your research, please cite the repository and/or associated publication. -->