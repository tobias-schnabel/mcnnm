# lightweight-mcnnm
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

lightweight-mcnnm is a Python package that provides a lightweight and performant implementation of the Matrix Completion with Nuclear Norm Minimization (MC-NNM) estimator for causal inference in panel data settings.

## What is lightweight-mcnnm

lightweight-mcnnm implements the MC-NNM estimator exactly as described in "Matrix Completion Methods for Causal Panel Data Models" by Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi [(2021)](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924). This estimator provides a powerful tool for estimating causal effects in panel data settings, particularly when dealing with complex treatment patterns and potential confounders.

The implementation focuses on performance and minimal dependencies, making it suitable for use in various environments, including GPUs and cloud clusters.

## Features

- Lightweight implementation with minimal dependencies
- Utilizes JAX for improved performance and GPU compatibility
- Faithful to the original MC-NNM algorithm as described in [Athey et al. (2021)](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924).
- Suitable for large-scale panel data analysis

## Installing lightweight-mcnnm

lightweight-mcnnm is compatible with Python 3.7 or later and depends on JAX and NumPy. The simplest way to install lightweight-mcnnm and its dependencies is from PyPI with pip:
```bash
$ pip install lightweight-mcnnm
```

To upgrade lightweight-mcnnm to the latest version, use:
```bash
$ pip install -U lightweight-mcnnm
```
## Using lightweight-mcnnm

Here's a simple example of how to use lightweight-mcnnm:

```python
import jax.numpy as jnp
from lightweight_mcnnm import fit
```
### Generate some sample data
```python
Y = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
W = jnp.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
```
### Fit the MC-NNM model
```python
results = fit(Y, W)
```
### Print the estimated treatment effect
```python
print(f"Estimated treatment effect: {results.tau}")
```
For more detailed usage instructions and examples, please refer to our documentation.

## Development

### Setting up the development environment

This project uses [Poetry](https://python-poetry.org/) for dependency management. To set up your development environment:

1. Ensure you have Poetry installed. If not, install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lightweight-mcnnm.git
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

## References
This implementation is based on the method described in:
[Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix Completion Methods for Causal Panel Data Models. Journal of the American Statistical Association, 116(536), 1716-1730.](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924)

## Acknowledgements
This project was inspired by and draws upon ideas from 
[CausalTensor](https://github.com/TianyiPeng/causaltensor) and 
[fect](https://yiqingxu.org/packages/fect/fect.html). I am grateful for their contributions to the field of causal inference.
## License
lightweight-mcnnm is released under the GNU General Public License v3.0. See the LICENSE file for more details.
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.