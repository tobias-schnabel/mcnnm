.. lightweight-mcnnm documentation master file, created by
   sphinx-quickstart on Sun Jul 21 11:22:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lightweight-mcnnm's documentation!
=============================================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/pypi/pyversions/lightweight-mcnnm.svg
   :target: https://pypi.org/project/lightweight-mcnnm/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/OS-Linux%20|%20Windows%20|%20macOS-blue
   :alt: OS

.. image:: https://img.shields.io/pypi/v/lightweight-mcnnm.svg?color=brightgreen&cache-bust=2
   :target: https://pypi.org/project/lightweight-mcnnm/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/mcnnm/badge/?version=latest
   :target: https://mcnnm.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://img.shields.io/badge/mypy-checked-blue
   :target: https://github.com/tobias-schnabel/mcnnm/actions/workflows/ci.yml
   :alt: mypy checked

.. image:: https://codecov.io/gh/tobias-schnabel/mcnnm/graph/badge.svg?token=VYJ12XOQMP
   :target: https://codecov.io/gh/tobias-schnabel/mcnnm
   :alt: codecov

.. image:: https://github.com/tobias-schnabel/mcnnm/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/tobias-schnabel/mcnnm/actions/workflows/ci.yml
   :alt: Tests

.. image:: https://img.shields.io/github/last-commit/tobias-schnabel/mcnnm
   :target: https://github.com/tobias-schnabel/mcnnm/commits/
   :alt: GitHub last commit

.. image:: https://img.shields.io/github/issues/tobias-schnabel/mcnnm
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/tobias-schnabel/mcnnm
   :alt: Pull Requests


lightweight-mcnnm is a Python package that provides a lightweight and performant implementation of the Matrix Completion with Nuclear Norm Minimization (MC-NNM) estimator for causal inference in panel data settings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   validation
   api
   examples
   development
   references

What is lightweight-mcnnm?
--------------------------

lightweight-mcnnm implements the MC-NNM estimator exactly as described in `Matrix Completion Methods for Causal Panel Data Models <https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924>`_ by Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi (2021). This estimator provides a powerful tool for estimating causal effects in panel data settings, particularly when dealing with complex treatment patterns and potential confounders.

The implementation focuses on performance and minimal dependencies, making it suitable for use in various environments, including GPUs and cloud clusters.

Features
--------

* Lightweight implementation with minimal dependencies
* Utilizes JAX for improved performance and GPU compatibility
* Faithful to the original MC-NNM algorithm as described in Athey et al. (2021)
* Suitable for large-scale panel data analysis
* Supports various treatment assignment mechanisms (staggered adoption, block assignment, single treated unit)
* Includes unit-specific, time-specific, and unit-time specific covariates
* Offers flexible validation methods for parameter selection (cross-validation and holdout)

Comparison to Other Implementations
-----------------------------------
lightweight-mcnnm is designed to be lightweight and easy to use, with a focus on performance and minimal dependencies.
The other two main implementations of the MC-NNM estimator are
`CausalTensor <https://github.com/TianyiPeng/causaltensor>`_
and
`fect <https://yiqingxu.org/packages/fect/fect.html>`_ .
Both packages implement MC-NNM as part of a broader set of causal inference methods. Both implement covariates and cross-validation differently from this package.
For a detailed comparison, see `this notebook <https://colab.research.google.com/github/tobias-schnabel/mcnnm/blob/main/Comparison.ipynb>`_.

Quick Start
-----------

Here's a simple example of how to use lightweight-mcnnm:

.. code-block:: python

   from mcnnm import estimate, generate_data

   Y, W, X, Z, V, true_params = generate_data(
        nobs=50,
        nperiods=10,
        unit_fe=True,
        time_fe=True,
        X_cov=True,
        Z_cov=True,
        V_cov=True,
        seed=2024,
        noise_scale=0.1,
        autocorrelation=0.0,
        assignment_mechanism="staggered",
        treatment_probability=0.1,
    )

   # Run estimation
   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='cv',
       K=3,
       n_lambda=30,
   )

   print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.3f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

For more detailed information and examples, check out the :doc:`usage` and :doc:`examples` pages.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
