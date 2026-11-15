# Lab 5: Frameworks and Automatic Differentiation

## 1. Goal
Implement a scalar reverse-mode automatic differentiation (`Value`) engine from scratch, validate its gradients against PyTorch autograd and numerical central differences, benchmark scalar graph execution against vectorized PyTorch tensors on `make_moons`, and evaluate graph tracing (`torch.jit.trace`) and memory contiguity overhead.

## 2. Method
- **Scalar Autograd Engine (`src/autograd.py`):** Built a `Value` node storing scalar `data`, accumulated `grad`, parent `_prev` set, and local `_backward` closures for `+`, `*`, `**`, `ReLU`, `sigmoid`, and `log`, executing reverse topological backpropagation with gradient accumulation (`+=`). Verified `((a*b + c)**2).relu().sum()` against PyTorch `float64` autograd and central differences (`eps = 1e-6`).
- **Training Benchmark (`src/train_moons.py`):** Trained identical `2 -> 8 -> 1` sigmoid MLPs with BCE loss on `make_moons` (`n=100`, 20 epochs, `lr=0.2`) using both the scalar `Value` engine and PyTorch tensor autograd, reporting the median of 5 timed runs after warm-up.
- **Graph, Tracing & Contiguity (`src/graph_and_compile.py`):** Inspected PyTorch `grad_fn` and `torch.jit.trace` IR on `SmallNet` (`64 -> 128 -> 128 -> 10`, `B=256`), compared eager vs. traced latency (after `torch.compile` raised `BackendCompilerFailed` on Windows CPU), and measured elementwise reduction latency on contiguous vs. non-contiguous transposed `(1024, 1024)` views.

## 3. Results

### Gradient Verification (`((a*b + c)**2).relu().sum()`)
| Comparison | Max Absolute Difference | Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Forward Output vs. PyTorch** | `8.88e-16` | `< 1e-6` | Pass |
| **Gradients vs. PyTorch (`da`, `db`, `dc`)** | `0.00e+00` | `< 1e-6` | Pass |
| **Gradients vs. Numerical (`eps=1e-6`)** | `6.72e-10` | `< 1e-6` | Pass |

### `make_moons` Training Speed (`2 -> 8 -> 1`, 20 Epochs, Median of 5 Runs)
| Execution Engine | Median Train Time (ms) | Final Accuracy | Slowdown vs. PyTorch |
| :--- | :---: | :---: | :---: |
| **Custom Scalar `Value`** | 1008.799 | 0.67 | 157.5x |
| **PyTorch Tensor Autograd** | 6.405 | 0.67 | 1.0x |

### Graph Tracing & Tensor Contiguity Benchmarks
| Benchmark | Shape / Dtype / Contiguity | Median Latency (us) |
| :--- | :--- | :---: |
| **Eager `SmallNet` (`B=256`)** | `(256, 64)` / `float32` / `True` | 254.30 |
| **`torch.jit.trace` `SmallNet` (`B=256`)** | `(256, 64)` / `float32` / `True` | 233.70 |
| **Contiguous View Reduction** | `(1024, 1024)` / `float32` / `True` | 248.25 |
| **Non-Contiguous Transposed View** | `(1024, 1024)` / `float32` / `False` | 269.75 |

## 4. Conclusions
1. **Topological accumulation reproduces exact analytical gradients:** Reverse-order traversal with `+=` accumulation matched PyTorch `float64` gradients to `0.00e+00` and numerical central differences within `6.72e-10`.
2. **Python scalar graph dispatch is `157.5x` slower than vectorized C++ tensor ops:** Building and traversing thousands of Python `Value` objects per epoch incurs heavy interpreter and pointer-chasing overhead (`1008.8 ms` vs. `6.4 ms`), whereas PyTorch dispatches contiguous BLAS/SIMD tensor kernels at the operator level.
3. **Memory contiguity directly impacts SIMD/cache efficiency:** Transposing a `(1024, 1024)` tensor creates a zero-copy strided view (`contiguous=False`), causing strided cache misses and reducing reduction throughput by `8.7%` (`269.75 us` vs. `248.25 us`).
4. **When compilation / tracing helps:** Static graph capture (`torch.jit.trace` or `torch.compile`) removes Python framework dispatch overhead and enables operator fusion. On `SmallNet`, tracing cut latency by `8.1%` (`254.30 us -> 233.70 us`); compilation yields the largest speedups on models with static shapes, many small memory-bound elementwise operations (fusion opportunity), and repeated inference/training steps that amortize ahead-of-time compile cost, while offering little benefit on dynamic control-flow graphs or single-GEMM workloads already saturated by vendor BLAS libraries.
