# Lab 3: Architectures and Their Costs

## 1. Goal
Compare an MLP, a CNN, and a patch-based Transformer Attention network on Fashion-MNIST across parameter count, hand-calculated vs. profiled FLOPs, batch-1 and batch-128 inference latency, and test accuracy to evaluate architectural cost-effectiveness.

## 2. Method
- **Dataset:** Loaded Fashion-MNIST with `ToTensor()` and `Normalize((0.2860,), (0.3530,))`, splitting the 60,000 training images into 55,000 train and 5,000 validation samples (`seed=42`) alongside the 10,000-image test set.
- **Models:** Implemented `FashionMLP` (`784 -> 256 -> 128 -> 10`), `FashionCNN` (two `3x3` conv-pool blocks `1 -> 16 -> 32` followed by `Linear(1568, 10)`), and `FashionAttention` (16 `7x7` patches projected to `d_model=64`, 2 `TransformerEncoderLayer` blocks with `nhead=4` and `dim_feedforward=128`, mean pooling, `Linear(64, 10)`).
- **Profiling and Training:** Counted parameters with `torchinfo.summary`, derived FLOPs (`2 * MACs`) by hand and via `thop.profile`, measured median CPU inference latency across 100 runs (batch 1) and 20 runs (batch 128) after 5 warm-up runs, and trained all three models for 5 epochs with Adam (`lr=1e-3`, batch size 128).

## 3. Results

### Cost, FLOPs, Latency, and Accuracy Summary
| Model | Parameters | Hand FLOPs | `thop` FLOPs | Latency B=1 (ms) | Latency B=128 (ms) | Val Acc | Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | 235,146 | 469,504 | 469,504 | 0.1147 | 0.3805 | 0.8842 | 0.8784 |
| **CNN** | 20,490 | 2,063,488 | 2,063,488 | 0.2682 | 3.8219 | 0.8996 | 0.9002 |
| **Attention** | 70,794 | 2,329,856 | 101,632 | 0.9165 | 3.6485 | 0.8404 | 0.8392 |

### Hand-Calculated vs. `thop` FLOPs Breakdown
- **MLP (`Linear(784 -> 256)` representative layer):** `2 * 784 * 256 = 401,408` FLOPs. Full network total is `469,504` FLOPs, matching `thop` (`469,504`) exactly.
- **CNN (`Conv2d(16 -> 32, 3x3)` on `14x14` output):** `2 * 3 * 3 * 16 * 32 * 14 * 14 = 1,806,336` FLOPs. Full network total is `2,063,488` FLOPs, matching `thop` (`2,063,488`) exactly.
- **Attention (one encoder layer, `seq=16`, `d=64`, `ff=128`):** QKV projection (`393,216`) + attention scores `QK^T` (`32,768`) + value weighting `AV` (`32,768`) + output projection (`131,072`) + two-layer FFN (`524,288`) = `1,114,112` FLOPs per layer (`2,329,856` total with patch projection and classifier head).
- **Why `thop` undercounts Attention (`101,632` vs. `2,329,856`):** `thop` registers hooks on standard `nn.Linear` and `nn.Conv2d` modules, so it only counts `patch_proj` (`100,352` FLOPs) and `head` (`1,280` FLOPs) while skipping the internal `nn.MultiheadAttention` / `scaled_dot_product_attention` operations inside `nn.TransformerEncoderLayer`.

Trade-off plots are saved in `results/tradeoffs.png`.

## 4. Conclusions
1. **CNN is the most cost-effective architecture for Fashion-MNIST:** It reaches the highest test accuracy (`90.02%`) with `11.5x` fewer parameters than the MLP (`20,490` vs. `235,146`) because `3x3` convolutional filters exploit 2D spatial locality and reuse weights across all spatial positions.
2. **Access patterns and arithmetic intensity explain the latency differences:** The MLP is memory-bandwidth-bound at batch 1 (`235k` weights read once per image), so batching 128 samples amortizes weight reads and increases latency only `3.3x` (`0.1147 ms` to `0.3805 ms`). The CNN has high weight reuse and higher FLOPs (`2.06M`), making it compute-bound so batch-128 latency scales `14.3x` (`3.8219 ms`).
3. **Attention pays high dispatch overhead and lacks spatial inductive bias at small scale:** Even with `d_model=64` and `seq=16`, Attention has the slowest batch-1 latency (`0.9165 ms`) due to multi-head tensor reshapes, LayerNorm, and softmax memory passes, and achieves lower accuracy (`83.92%`) in 5 epochs without convolutional inductive bias.
