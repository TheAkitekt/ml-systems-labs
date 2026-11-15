# Lab 7: Compression — Pruning and Knowledge Distillation

## 1. Goal
Compress an oversized Fashion-MNIST CNN (~1M parameters) using global unstructured pruning, layer-wise unstructured pruning, structured L2 filter pruning, and teacher-student knowledge distillation (~100k-parameter student), evaluating accuracy, on-disk storage, CPU latency, compute, and energy trade-offs.

## 2. Method
- **Baseline Teacher (`src/train_teacher.py`):** Trained `TeacherCNN` (`32 -> 64 -> 128` conv channels + `Linear(6272, 144) -> Linear(144, 10)`, `997,434` parameters, `16,711,488` FLOPs) for 8 epochs on Fashion-MNIST (`seed=42`, `Adam`, `lr=1e-3`, `B=128`).
- **Magnitude & Structured Pruning (`src/prune_models.py`):** Applied L1 unstructured pruning globally and per layer across sparsity targets `[0%, 25%, 50%, 75%, 90%, 95%]` followed by 3 epochs of mask-fixed fine-tuning (`lr=2e-4`), and physically removed lowest L2-norm filters/units at `[10%, 25%, 50%]` before 3-epoch fine-tuning. Compared dense `.pt` vs. sparse COO `.pt` storage at 90% global sparsity.
- **Knowledge Distillation (`src/distill_and_summary.py`):** Trained `StudentCNN` (`99,550` parameters, `2,221,488` FLOPs) for 8 epochs with pure Cross-Entropy (`Plain`) vs. temperature-scaled distillation (`T=4.0`, `alpha=0.5`: `0.5 * CE + 0.5 * T^2 * KL(student/T || teacher/T)`).

## 3. Results

### Pruning Sparsity vs. Accuracy & CPU Latency
| Method | Target Sparsity | Params (Active/Total) | Latency B=1 (ms) | Latency B=128 (ms) | Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Teacher Baseline** | 0% | 997,434 | 0.6064 | 11.9594 | 0.9210 |
| **Global Unstructured (L1)** | 25% | 748,144 / 997,434 | 0.6372 | 12.6631 | 0.9303 |
| **Global Unstructured (L1)** | 50% | 498,854 / 997,434 | 0.6054 | 13.1950 | 0.9297 |
| **Global Unstructured (L1)** | 75% | 249,564 / 997,434 | 0.6866 | 12.6551 | 0.9298 |
| **Global Unstructured (L1)** | 90% | 99,990 / 997,434 | 0.5756 | 12.2618 | **0.9305** |
| **Global Unstructured (L1)** | 95% | 50,203 / 997,434 | 0.5907 | 11.4871 | 0.9193 |
| **Layer-wise Unstructured (L1)** | 75% | 249,564 / 997,434 | 0.5906 | 11.7491 | 0.9154 |
| **Layer-wise Unstructured (L1)** | 90% | 99,990 / 997,434 | 0.5684 | 11.9482 | 0.8753 |
| **Layer-wise Unstructured (L1)** | 95% | 50,203 / 997,434 | 0.6046 | 11.3147 | 0.7994 |
| **Structured Filter Pruning (L2)** | 10% | 809,621 | 0.6802 | 12.4107 | 0.9300 |
| **Structured Filter Pruning (L2)** | 25% | 561,454 | 0.5412 | 9.5494 | 0.9291 |
| **Structured Filter Pruning (L2)** | **50%** | **249,890** | **0.4268** | **5.5013** | **0.9254** |

### Dense vs. Sparse Storage & Knowledge Distillation Comparison
- **90% Global Pruned Storage:** Dense `.pt` = **3.8088 MB** vs. Sparse COO `.pt` = **2.3118 MB** (`-39.3%` on disk; COO stores `(row, col)` indices alongside non-zero values, so storage shrinks less than the 90% zero ratio).
- **Student Models (`99,550` params, `0.3830 MB`, `2.22M` FLOPs):**
  - **Plain `StudentCNN` (CE only):** Test Acc = `0.9053`, Latency B=1 = `0.3382 ms`, B=128 = `4.2143 ms`.
  - **Distilled `StudentCNN` (`T=4, alpha=0.5`):** Test Acc = **`0.9065`** (`+0.12%` over plain student at identical `0.383 MB` size and `2.22M` FLOPs), Latency B=1 = `0.4006 ms`, B=128 = `4.6747 ms`.

### Summary Across Chapter 9 Efficiency Pillars
| Efficiency Pillar | Teacher Baseline (`997k`) | Global Pruned 90% (`100k` active) | Structured Pruned 50% (`250k`) | Distilled Student (`100k`) |
| :--- | :--- | :--- | :--- | :--- |
| **Data** | Full 55k train / 10k test (`28x28` normalized) | Same input pipeline; mask-fixed fine-tuning | Same input pipeline; slimmed channel tensors | Dark knowledge via teacher soft logits (`T=4`) |
| **Model (Size & Acc)** | `3.8086 MB`, `92.10%` acc | `2.3118 MB` (COO) / `3.8088 MB` (dense), **`93.05%`** acc | `0.9570 MB` (`4.0x` smaller), `92.54%` acc | **`0.3830 MB`** (`9.9x` smaller), `90.65%` acc |
| **Compute (FLOPs & Latency)** | `16.71M` FLOPs; `11.96 ms` (B=128) | `16.71M` dense FLOPs; `12.26 ms` (no speedup) | `4.29M` FLOPs (`3.9x` less); **`5.50 ms`** (`2.2x` faster) | **`2.22M` FLOPs** (`7.5x` less); **`4.67 ms`** (`2.6x` faster) |
| **Energy (DRAM & Ops Proxy)** | Highest DRAM weight reads & MAC count | High DRAM traffic (dense zero-padded reads + mask overhead) | Low DRAM footprint & `74.3%` fewer MACs | Lowest DRAM footprint (`0.38 MB` fits in L2 cache) & `86.7%` fewer MACs |

## 4. Conclusions
1. **Unstructured pruning compresses storage without speeding up dense CPU kernels:** At 90% global sparsity, accuracy actually improves (`92.10% -> 93.05%`) due to regularization, and COO formatting trims disk size by `39.3%`, yet batch-128 CPU latency remains unchanged (`12.26 ms` vs. `11.96 ms`) because standard dense BLAS/AVX2 kernels execute dense matrix multiplications over zeroed entries and sparse gather/scatter indexing overhead outweighs skipped MACs below ~95% sparsity.
2. **Global pruning outperforms layer-wise pruning at extreme sparsity:** At 90%–95% sparsity, uniform layer-wise pruning starves small convolutional layers (`conv1` has only 288 weights) and collapses accuracy (`79.94%`), whereas global ranking preserves bottleneck early filters and prunes redundant `fc1` weights (`91.93%` at 95%).
3. **Structured pruning and distillation deliver real hardware speedups:** Removing 50% of filters physically shrinks tensor dimensions, cutting FLOPs `3.9x` and batch-128 latency `2.2x` (`5.50 ms`) at `92.54%` accuracy, while knowledge distillation packs `90.65%` accuracy into a `0.383 MB` student (`7.5x` fewer FLOPs than the teacher).
