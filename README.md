# lightweight-mcnnm

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightweight-mcnnm.svg)](https://pypi.org/project/lightweight-mcnnm/)
![OS](https://img.shields.io/badge/OS-Linux%20|%20Windows%20|%20macOS-blue)
[![PyPI version](https://img.shields.io/pypi/v/lightweight-mcnnm.svg?color=brightgreen&cache-bust=2)](https://pypi.org/project/lightweight-mcnnm/)
[![Documentation Status](https://readthedocs.org/projects/mcnnm/badge/?version=latest)](https://mcnnm.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy checked](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/tobias-schnabel/mcnnm/actions/workflows/ci.yml)
[![GitHub last commit](https://img.shields.io/github/last-commit/tobias-schnabel/mcnnm)](https://github.com/tobias-schnabel/mcnnm/commits/)
[![codecov](https://codecov.io/gh/tobias-schnabel/mcnnm/graph/badge.svg?token=VYJ12XOQMP)](https://codecov.io/gh/tobias-schnabel/mcnnm)
[![Tests](https://github.com/tobias-schnabel/mcnnm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tobias-schnabel/mcnnm/actions/workflows/ci.yml)
![Issues](https://img.shields.io/github/issues/tobias-schnabel/mcnnm)
![Pull Requests](https://img.shields.io/github/issues-pr/tobias-schnabel/mcnnm)

lightweight-mcnnm is a Python package that provides a lightweight and performant implementation of the Matrix Completion with Nuclear Norm Minimization (MC-NNM) estimator for causal inference in panel data settings.

## Table of Contents
- [What is lightweight-mcnnm](#What-is-lightweight-mcnnm)
- [Features](#features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Using lightweight-mcnnm](#using-lightweight-mcnnm)
- [Development](#development)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)
 [Changelog, Contributing, and Templates](#changelog-contributing-and-templates)


## What is lightweight-mcnnm

lightweight-mcnnm implements the MC-NNM estimator exactly as described in "Matrix Completion Methods for Causal Panel Data Models" by Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi [(2021)](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924). This estimator provides a powerful tool for estimating causal effects in panel data settings, particularly when dealing with complex treatment patterns and potential confounders.

The implementation focuses on performance and minimal dependencies, making it suitable for use in various environments, including GPUs and cloud clusters.

## Features

- Lightweight implementation with minimal dependencies
- Utilizes JAX for improved performance and GPU compatibility
- Faithful to the original MC-NNM algorithm as described in [Athey et al. (2021)](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924)
- Suitable for large-scale panel data analysis
- Supports various treatment assignment mechanisms
- Includes unit-specific, time-specific, and unit-time specific covariates
- Offers flexible validation methods for parameter selection

## Installation

### Requirements

lightweight-mcnnm is compatible with Python 3.10 or later and depends on JAX and NumPy. CUDA-compatible versions of Jax are not currently supported directly by lightweight-mcnnm, but you can use JAX with CUDA support by installing it separately.

### Installing from PyPI

The simplest way to install lightweight-mcnnm and its dependencies is from PyPI using pip:

```bash
pip install lightweight-mcnnm
```

To upgrade lightweight-mcnnm to the latest version, use:
```bash
pip install --upgrade lightweight-mcnnm
```

#### JIT Compilation
By default, this package uses JAX's JIT compilation for better performance in typical use cases. If you want to disable JIT compilation, you can add the following line at the top of your script:

```python
jax.config.update('jax_disable_jit', True)
```

Note that disabling JIT may impact performance depending on your specific use case. I have found leaving JIT enabled to be the best option for most use cases. An example use case where disabling JIT may be sensible is calling estimate() multiple times on datasets of different sizes, which triggers recompilation any time the input data shape changes.

## Documentation
The full documentation for lightweight-mcnnm is available at:
https://mcnnm.readthedocs.io/en/latest/
## Using lightweight-mcnnm
1. A comprehensive example is available here: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tobias-schnabel/mcnnm/blob/main/Example.ipynb)
2. Simple example of how to use lightweight-mcnnm:
```python
import jax.numpy as jnp
from lightweight_mcnnm import estimate, generate_data

# Generate sample data
data, true_params = generate_data(nobs=500, nperiods=100, seed=42)

Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

# Estimate the MC-NNM model
results = estimate(Y, W)

print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
```
For more detailed usage instructions and examples, please refer to the documentation.

## Development

### Setting up the development environment

This project uses [Poetry](https://python-poetry.org/) for dependency management. To set up your development environment:

1. Ensure you have Poetry installed. If not, install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone the repository:
   ```bash
   git clone https://github.com/tobias-schnabel/mcnnm.git
   cd lightweight-mcnnm
   ```
3. Install the project dependencies:
    ```bash
    poetry install
    ```
    This command creates a virtual environment and installs all the necessary dependencies.
4. Activate the virtual environment:
    ```bash
   poetry shell
   ```
Now you're ready to start developing!
### Testing and building the package
5. Running tests: use the following command:
    ```bash
    poetry run pytest
   ```
   This will run all the tests in the tests/ directory with the exception of test_estimation_options.py, which is meant to
   exhaustively test possible combinations of estimation options and is disabled by default. To run all tests including
   this script, use the following command:
      ```bash
      poetry run pytest -m "not comprehensive or comprehensive"
      ```

6. Coverage: to generate a coverage report, run the following command:
    ```bash
    poetry run coverage report
    ```
    This will generate a coverage report showing the percentage of code covered by the tests.
6. Building the package: run the following command:
    ```bash
    poetry build
    ```
    This will create both wheel and source distributions in the dist/ directory.

## Development Workflow
### Pre-commit Hooks
This project uses pre-commit hooks to ensure code quality and consistency. Pre-commit hooks are scripts that run automatically every time you commit changes to your version control system. They help catch common issues before they get into the codebase. To set up:
1. Install pre-commit:
    ```bash
    poetry run pre-commit install
    ```
2. Install the hooks:
    ```bash
    poetry run pre-commit install
    ```
3. Run the hooks on all files (recommended for the first setup):
    ```bash
    poetry run pre-commit run --all-files
    ```
The configuration for the pre-commit hooks can be found in the .pre-commit-config.yaml file. The following hooks are configured:

    •	Trailing whitespace removal: Ensures no trailing whitespace is left in the code.
    •	End-of-file fixer: Ensures files end with a newline.
    •	YAML check: Validates YAML files.
    •	Flake8: Checks for Python style guide enforcement.
    •	Black: Ensures consistent code formatting.
    •	Bandit: Checks for common security issues in Python code.
    •	Mypy: Performs static type checking.

### Branch Protection
To maintain the integrity of the main branch, branch protection rules are enforced. These rules ensure that all changes to the main branch go through a review process and pass all required checks.
#### Protected Branch Rules

 1.	Require pull request reviews before merging: At least one approval from an administrator is required.
 2.	Require status checks to pass before merging: All CI checks must be successful before merging.

## References
This implementation is based on the method described in:
[Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix Completion Methods for Causal Panel Data Models. Journal of the American Statistical Association, 116(536), 1716-1730.](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924)

## Acknowledgements
This project was inspired by and draws upon ideas from
[CausalTensor](https://github.com/TianyiPeng/causaltensor) and
[fect](https://yiqingxu.org/packages/fect/fect.html). I am grateful for their contributions to the field of causal inference.
## Citing lightweight-mcnnm

If you use lightweight-mcnnm in your research, please cite both the software and the original paper describing the method:

For the software:
Schnabel, T. (2023). lightweight-mcnnm: A Python package for Matrix Completion with Nuclear Norm Minimization. https://github.com/tobias-schnabel/mcnnm

For the method:
Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix Completion Methods for Causal Panel Data Models. Journal of the American Statistical Association, 116(536), 1716-1730.

BibTeX entries:

@software{schnabel2023lightweightmcnnm,
  author = {Schnabel, Tobias},
  title = {lightweight-mcnnm: A Python package for Matrix Completion with Nuclear Norm Minimization},
  year = {2023},
  url = {https://github.com/tobias-schnabel/mcnnm}
}

@article{athey2021matrix,
  title={Matrix completion methods for causal panel data models},
  author={Athey, Susan and Bayati, Mohsen and Doudchenko, Nikolay and Imbens, Guido and Khosravi, Khashayar},
  journal={Journal of the American Statistical Association},
  volume={116},
  number={536},
  pages={1716--1730},
  year={2021},
  publisher={Taylor \& Francis}
}
## License
lightweight-mcnnm is released under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for more details.

## Changelog, Contributing, and Templates
1. For a detailed changelog of each release, please see the [GitHub Releases page](https://github.com/tobias-schnabel/mcnnm/releases)
2. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute to this project.
3. For reporting issues, please use the template provided in [ISSUE_TEMPLATE.md](./.github/ISSUE_TEMPLATE.md)
4. For submitting pull requests, please use the template provided in [PULL_REQUEST_TEMPLATE.md](./.github/PULL_REQUEST_TEMPLATE.md)
