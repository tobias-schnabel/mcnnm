import pytest
from typing import Optional, Tuple
import jax.numpy as jnp
from mcnnm.util import *

def test_frobenius_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(frobenius_norm(A), jnp.sqrt(30))

def test_nuclear_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(nuclear_norm(A), jnp.sum(jnp.linalg.svd(A, compute_uv=False)))

def test_element_wise_l1_norm():
    A = jnp.array([[1, -2], [-3, 4]])
    assert jnp.allclose(element_wise_l1_norm(A), 10)

def test_propose_lambda():
    lambdas = propose_lambda(1.0, 5)
    assert len(lambdas) == 5
    assert jnp.allclose(lambdas[0], 0.01)
    assert jnp.allclose(lambdas[-1], 100.0)

def test_timer():
    @timer
    def dummy_function():
        return 42

    assert dummy_function() == 42

def test_time_fit():
    Y = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    W = jnp.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    result = time_fit(Y, W)
    assert isinstance(result, tuple)