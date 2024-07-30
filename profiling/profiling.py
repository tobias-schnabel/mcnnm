from functools import wraps, partial
from typing import Callable, Any, Dict, List
import jax
import jax.numpy as jnp
from mcnnm.estimate import estimate
from mcnnm.types import Array


def time_jit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to measure compilation and execution times of a JIT-compiled function.

    This decorator wraps a function with jax.jit and measures:
    1. The time taken for the first call (compilation + execution)
    2. The time taken for subsequent calls (execution only)

    Args:
        func (Callable[..., Any]): The function to be JIT-compiled and timed.

    Returns:
        Callable[..., Any]: A wrapped version of the input function that prints timing information.

    Note:
        This decorator uses jax.block_until_ready() to ensure accurate timing
        by waiting for asynchronous operations to complete.
    """
    jitted_func = jax.jit(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # First call (compilation + execution)
        start_time = jax.config.jax_platform_name
        result = jitted_func(*args, **kwargs)
        jax.block_until_ready(result)
        first_call_time = jax.config.jax_platform_name - start_time

        # Second call (execution only)
        start_time = jax.config.jax_platform_name
        result = jitted_func(*args, **kwargs)
        jax.block_until_ready(result)
        second_call_time = jax.config.jax_platform_name - start_time

        print(f"Function: {func.__name__}")
        print(f"  First call (compilation + execution): {first_call_time:.4f} seconds")
        print(f"  Second call (execution only): {second_call_time:.4f} seconds")
        print(f"  Estimated compilation time: {first_call_time - second_call_time:.4f} seconds")

        return result

    return wrapper


def timed_estimate(*args, **kwargs) -> Any:
    """
    A wrapper function for the estimate() function that applies the time_jit decorator.

    This function creates a partial function from estimate() with the given arguments,
    applies the time_jit decorator, and then calls the resulting function.

    Args:
        *args: Positional arguments to pass to estimate().
        **kwargs: Keyword arguments to pass to estimate().

    Returns:
        Any: The result of the estimate() function.
    """
    partial_estimate = partial(estimate, *args, **kwargs)
    timed_estimate_func = time_jit(partial_estimate)
    return timed_estimate_func()


def benchmark_estimate(
    Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array, n_runs: int = 5
) -> Dict[str, List[float]]:
    """
    Benchmark the estimate() function with multiple runs.

    This function runs the estimate() function multiple times and records
    the compilation and execution times for each run.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Array): The unit-specific covariates matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        n_runs (int): Number of times to run the estimate() function. Default is 5.

    Returns:
        Dict[str, List[float]]: A dictionary containing lists of compilation and execution times.
    """
    compilation_times = []
    execution_times = []

    for i in range(n_runs):
        print(f"Run {i + 1}/{n_runs}")

        # First call (compilation + execution)
        start_time = jax.config.jax_platform_name
        result = estimate(Y, W, X, Z, V, Omega)
        jax.block_until_ready(result)
        first_call_time = jax.config.jax_platform_name - start_time

        # Second call (execution only)
        start_time = jax.config.jax_platform_name
        result = estimate(Y, W, X, Z, V, Omega)
        jax.block_until_ready(result)
        second_call_time = jax.config.jax_platform_name - start_time

        compilation_time = first_call_time - second_call_time
        execution_time = second_call_time

        compilation_times.append(compilation_time)
        execution_times.append(execution_time)

        print(f"  Compilation time: {compilation_time:.4f} seconds")
        print(f"  Execution time: {execution_time:.4f} seconds")
        print()

    return {"compilation_times": compilation_times, "execution_times": execution_times}


def print_benchmark_summary(benchmark_results: Dict[str, List[float]]) -> None:
    """
    Print a summary of benchmark results.

    Args:
        benchmark_results (Dict[str, List[float]]): The results from the benchmark_estimate function.

    Returns:
        None
    """
    compilation_times = jnp.array(benchmark_results["compilation_times"])
    execution_times = jnp.array(benchmark_results["execution_times"])

    print("Benchmark Summary:")
    print(f"  Number of runs: {len(compilation_times)}")
    print(f"  Average compilation time: {jnp.mean(compilation_times):.4f} seconds")
    print(f"  Average execution time: {jnp.mean(execution_times):.4f} seconds")
    print(
        f"  Total average time: {jnp.mean(compilation_times) + jnp.mean(execution_times):.4f} seconds"
    )
    print(f"  Compilation time std dev: {jnp.std(compilation_times):.4f} seconds")
    print(f"  Execution time std dev: {jnp.std(execution_times):.4f} seconds")


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    N, T = 100, 50
    Y = jax.random.normal(jax.random.PRNGKey(0), shape=(N, T))
    W = jax.random.choice(jax.random.PRNGKey(1), 2, shape=(N, T))
    X = jax.random.normal(jax.random.PRNGKey(2), shape=(N, 5))
    Z = jax.random.normal(jax.random.PRNGKey(3), shape=(T, 3))
    V = jax.random.normal(jax.random.PRNGKey(4), shape=(N, T, 2))
    Omega = jnp.eye(T)

    # Run the benchmark
    results = benchmark_estimate(Y, W, X, Z, V, Omega, n_runs=5)

    # Print the summary
    print_benchmark_summary(results)
