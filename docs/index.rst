.. lightweight-mcnnm documentation master file, created by
   sphinx-quickstart on Sun Jul 21 11:22:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lightweight-mcnnm's documentation!
=============================================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

lightweight-mcnnm is a Python package that provides a lightweight and performant implementation of the Matrix Completion with Nuclear Norm Minimization (MC-NNM) estimator for causal inference in panel data settings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples
   development
   references

What is lightweight-mcnnm?
--------------------------

lightweight-mcnnm implements the MC-NNM estimator exactly as described in "Matrix Completion Methods for Causal Panel Data Models" by Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi (2021). This estimator provides a powerful tool for estimating causal effects in panel data settings, particularly when dealing with complex treatment patterns and potential confounders.

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

Quick Start
-----------

Here's a simple example of how to use lightweight-mcnnm:

.. code-block:: python

   import jax.numpy as jnp
   from lightweight_mcnnm import estimate

   # Generate some sample data
   Y = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   W = jnp.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]])

   # Fit the MC-NNM model
   results = estimate(Y, W)

   # Print the estimated treatment effect
   print(f"Estimated treatment effect: {results.tau}")

For more detailed information and examples, check out the :doc:`usage` and :doc:`examples` pages.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`