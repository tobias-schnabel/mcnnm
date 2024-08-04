from mcnnm.estimate import complete_matrix
from mcnnm.util import generate_data
import jax
import jax.numpy as jnp
import time

jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_disable_jit", True)

data, true_params = generate_data(
    nobs=12, nperiods=45_000, seed=42, treatment_effect=5, assignment_mechanism="last_periods"
)

Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)
X, Z, V = jnp.array(true_params["X"]), jnp.array(true_params["Z"]), jnp.array(true_params["V"])

start_time = time.time()
results = complete_matrix(
    Y,
    W,
    X,
    Z,
    V,
    validation_method="holdout",
    n_lambda_H=20,
    n_lambda_L=20,
    K=24,
    horizon=24,
    max_window_size=168,
)
print(f"Execution time: {time.time() - start_time:.4f}")
print(f"MSE {jnp.mean((Y - results.Y_completed) ** 2):.4f}")
print(f"MAE {jnp.mean(jnp.abs(Y - results.Y_completed)):.4f}")
