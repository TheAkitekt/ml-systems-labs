# Lab 6: Training Pipelines and Project Proposal

## 1. Goal
Restructure Fashion-MNIST CNN training into a reproducible YAML-driven pipeline with MLflow tracking and checkpointing, profile epoch execution bottlenecks, evaluate worker-count, batch-size, optimizer/scheduler, gradient-accumulation, and checkpoint-resume trade-offs, and author the individual course project proposal (`project/proposal.md`).

## 2. Method
- **YAML + MLflow Pipeline (`config.yaml`, `src/train_mlflow.py`):** Configured seed (`42`), batch size (`128`), optimizer (`Adam`, `lr=1e-3`), and checkpoint frequency in `lab06/config.yaml`, logging parameters, per-epoch metrics, config YAML, and `.pt` checkpoints to MLflow.
- **Profiling & DataLoader Sweep (`src/profile_and_workers.py`):** Wrapped data loading, forward pass, backward pass, and optimizer step in `torch.profiler.record_function` for one epoch (`num_workers=0`), then benchmarked epoch wall-clock time across `num_workers` `[0, 1, 2, 4, 8]`.
- **Throughput, Schedulers, Accumulation & Resume (`src/batch_and_accum.py`, `src/schedulers_and_resume.py`):** Measured throughput, peak RSS memory, and 5-epoch accuracy across batch sizes `[32, 64, 128, 256, 512]`; compared `Adam` (`lr=1e-3`) and `SGD+Momentum` (`lr=1e-2`, `momentum=0.9`) paired with `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` over 10 epochs; verified gradient accumulation (`batch=64, accum=4`) against native `batch=256`; and tested full state/RNG checkpoint interruption and resumption at epoch 3.

## 3. Results

### `torch.profiler` Single-Epoch Time Breakdown (`num_workers=0`, `B=128`)
| Training Phase | CPU Time (ms) | Share of Epoch (%) |
| :--- | :---: | :---: |
| **Data Loading** | 7895.99 | **64.37%** |
| **Forward Pass** | 1820.24 | 14.84% |
| **Backward Pass** | 2196.42 | 17.91% |
| **Optimizer Step** | 353.38 | 2.88% |

### `DataLoader` `num_workers` Bottleneck Study (`B=128`, 1 Epoch)
| `num_workers` | Epoch Time (s) | Throughput (samples/s) |
| :---: | :---: | :---: |
| **0** | 20.183 | 2725.1 |
| **1** | **10.980** | **5009.0** |
| **2** | 11.729 | 4689.4 |
| **4** | 19.145 | 2872.8 |
| **8** | 33.227 | 1655.3 |

### Batch-Size Study (5 Epochs, `Adam`, `lr=1e-3`, `num_workers=1`)
| Batch Size | Throughput (samples/s) | Peak RSS Memory (MB) | Test Accuracy |
| :---: | :---: | :---: | :---: |
| **32** | 4646.8 | 361.66 | **0.9050** |
| **64** | 4975.4 | 371.31 | 0.8968 |
| **128** | **5177.6** | 398.46 | 0.8903 |
| **256** | 4831.0 | 424.93 | 0.8894 |
| **512** | 2803.5 | 473.27 | 0.8754 |

### Optimizer & LR Scheduler Comparison (10 Epochs, `B=128`)
| Configuration | Final LR | Final Train Loss | Val Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Adam + StepLR** | 0.00025 | 0.2125 | 0.9070 | 0.9034 |
| **Adam + CosineAnnealingLR** | 0.00000 | 0.2141 | 0.9074 | 0.9031 |
| **Adam + ReduceLROnPlateau** | 0.00100 | **0.2108** | **0.9098** | 0.9051 |
| **SGD+Momentum + StepLR** | 0.00250 | 0.2258 | 0.9042 | 0.9046 |
| **SGD+Momentum + CosineAnnealingLR** | 0.00000 | 0.2261 | 0.9018 | 0.9035 |
| **SGD+Momentum + ReduceLROnPlateau** | 0.01000 | 0.2262 | 0.9042 | **0.9083** |

### Gradient Accumulation & Epoch-3 Checkpoint Resume
- **Gradient Accumulation (`B=64 x 4` vs. Native `B=256`):** Training loss trajectories match closely across epochs (`0.6159 -> 0.3419 -> 0.2933` vs. `0.6160 -> 0.3418 -> 0.2931`), reaching nearly identical test accuracy (`0.8891` vs. `0.8894`).
- **Interrupt & Resume at Epoch 3:** Saving and restoring model weights, optimizer moments, and `torch.get_rng_state()` produced bit-identical training loss (`0.53462 -> 0.35023 -> 0.30394 -> 0.28022 -> 0.26319`) and validation accuracy (`0.8984` at epoch 5) relative to an uninterrupted 5-epoch run (`results/resume_curve.png`).

## 4. Conclusions
1. **Data loading is the primary bottleneck on small CPU CNNs:** With `num_workers=0`, synchronous Python image transforms consume `64.37%` of epoch time (`7.90 s`). Setting `num_workers=1` overlaps prefetching with compute and cuts epoch time nearly in half (`20.18 s -> 10.98 s`), whereas `num_workers >= 4` degrades performance (`33.23 s` at 8 workers) due to Windows `spawn` process creation and IPC serialization overhead.
2. **Moderate batch sizes maximize throughput and generalization:** `B=128` peaks at `5,177.6 samples/s`, while `B=32` achieves the highest 5-epoch test accuracy (`90.50%`) via more frequent gradient updates and higher stochastic gradient noise; `B=512` increases memory to `473.27 MB` and lowers accuracy (`87.54%`) under a fixed 5-epoch budget.
3. **Gradient accumulation and RNG-aware checkpoints ensure hardware flexibility and fault tolerance:** Emulating `B=256` via four `B=64` micro-batches reproduces large-batch optimization dynamics at lower activation memory, and serializing RNG alongside model/optimizer states enables exact deterministic recovery after interruptions.
