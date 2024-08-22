from typing import Optional, Tuple

import jax.numpy as jnp


from .types import Array


def check_inputs(
    Y: Array,
    W: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array]:
    """
    Check and preprocess input arrays for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Optional[Array]): The unit-specific covariates matrix of shape (N, P). Default is None.
        Z (Optional[Array]): The time-specific covariates matrix of shape (T, Q). Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor of shape (N, T, J). Default is None.
        Omega (Optional[Array]): The autocorrelation matrix of shape (T, T). Default is None.

    Returns:
        Tuple: A tuple containing preprocessed X, Z, V, and Omega arrays.

    Raises:
        ValueError: If the input array dimensions are mismatched or invalid.
    """
    N, T = Y.shape

    if W.shape != (N, T):
        raise ValueError(f"The shape of W ({W.shape}) must match the shape of Y ({Y.shape}).")

    if X is not None:
        if X.shape[0] != N:
            raise ValueError(
                f"The first dimension of X ({X.shape[0]}) must match the first dimension of Y ({N})."
            )
    else:
        X = jnp.zeros((N, 0))

    if Z is not None:
        if Z.shape[0] != T:
            raise ValueError(
                f"The first dimension of Z ({Z.shape[0]}) must match the second dimension of Y ({T})."
            )
    else:
        Z = jnp.zeros((T, 0))

    if V is not None:
        if V.shape[:2] != (N, T):
            raise ValueError(
                f"The first two dimensions of V ({V.shape[:2]}) must match the shape of Y ({Y.shape})."
            )
    else:
        V = jnp.zeros((N, T, 1))  # Add a dummy dimension if V is None, otherwise causes issues

    if Omega is not None:
        if Omega.shape != (T, T):
            raise ValueError(f"The shape of Omega ({Omega.shape}) must be ({T}, {T}).")
    else:
        Omega = jnp.eye(T)

    return X, Z, V, Omega
