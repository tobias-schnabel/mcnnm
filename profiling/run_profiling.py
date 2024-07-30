from mcnnm.util import generate_data
from profiling import benchmark_estimate, print_benchmark_summary

# Generate sample data
data, true_params = generate_data(nobs=100, nperiods=50)
Y = data["y"].values.reshape(100, 50)
W = data["treat"].values.reshape(100, 50)
X = true_params["X"]
Z = true_params["Z"]
V = true_params["V"]
Omega = None  # Or generate an appropriate Omega matrix

# Run the benchmark
results = benchmark_estimate(Y, W, X, Z, V, Omega, n_runs=5)

# Print the summary
print_benchmark_summary(results)
