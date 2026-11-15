# Lab 2: A Neural Network from Scratch

## 1. Goal
Build a two-hidden-layer neural network in pure NumPy on the `load_digits` dataset, verify analytical backpropagation against numerical gradients, compare learning rates and hidden layer widths, and measure forward versus backward pass latency.

## 2. Method
- **Data Preparation:** Loaded `load_digits` (1,797 samples, 64 features, 10 classes), scaled inputs by dividing by 16 to `[0.0, 1.0]`, one-hot encoded labels, and split 70/30 stratified (`random_state=42`) into 1,257 train and 540 test samples.
- **Architecture and Initialization:** Built a `64 -> W -> ReLU -> W -> ReLU -> 10 -> Softmax` network with weights `W1 (64, W)`, `W2 (W, W)`, `W3 (W, 10)` drawn from `N(0, 0.01)` and zero biases `b1 (W,)`, `b2 (W,)`, `b3 (10,)` using seed `42`.
- **Verification and Training:** Checked softmax row sums and initial cross-entropy loss, verified analytical gradients on 10 random weights with central differences (`eps = 1e-5`), trained for 50 epochs (shuffled mini-batches of size 32) across widths `[16, 64]` and learning rates `[0.001, 0.01, 0.1]`, and timed 100 forward and backward passes after a warm-up run.

## 3. Results

### Forward Pass & Gradient Check Verification
- **Forward Sanity Check:** Both `W = 16` and `W = 64` produced `(32, 10)` probability outputs with maximum row-sum error `2.22e-16` (< `1e-6`) and initial cross-entropy loss `2.3026`, matching `ln(10) = 2.3026`.
- **Numerical Gradient Check (`W = 16`, `eps = 1e-5`):** **9 / 10** checked weights passed with relative error `< 1e-6` (ranging from `0.00e+00` to `4.39e-07`). The single miss (`W3[8,9]`, relative error `3.43e-06`) occurred where the gradient magnitude was near zero (`~2e-6`).

### Learning Rate & Width Sweep (50 Epochs, Batch Size 32)
| Width (W) | Learning Rate | Final Train Loss | Test Accuracy |
| :---: | :---: | :---: | :---: |
| **16** | 0.001 | 2.3026 | 0.1019 |
| **16** | 0.010 | 2.3026 | 0.1019 |
| **16** | 0.100 | 0.8609 | 0.4426 |
| **64** | 0.001 | 2.3025 | 0.1019 |
| **64** | 0.010 | 2.3024 | 0.1019 |
| **64** | 0.100 | 0.1384 | 0.9333 |

Loss curves are saved in `results/loss_curves.png`.

### Forward vs. Backward Pass Timing (Batch Size 32, Median of 100 Runs)
| Width (W) | Forward Median (us) | Backward Median (us) | Backward / Forward Ratio |
| :---: | :---: | :---: | :---: |
| **16** | 24.80 | 26.40 | 1.06x |
| **64** | 50.15 | 67.60 | 1.35x |

## 4. Conclusions
1. **Small `N(0, 0.01)` initialization causes vanishing gradients at low learning rates:** Multiplying inputs through three layers of `0.01`-scale weights shrinks early-layer gradients by roughly `10^-4`, leaving `lr = 0.001` and `lr = 0.01` stuck at random-guess accuracy (`0.1019`) after 50 epochs, whereas `lr = 0.1` escapes the flat region.
2. **Wider hidden layers improve optimization and capacity:** At `lr = 0.1`, increasing width from `16` to `64` provides stronger gradient flow and representational capacity, dropping final loss from `0.8609` to `0.1384` and raising test accuracy from `0.4426` to `0.9333`.
3. **Backward pass dominates runtime due to extra matrix multiplications:** Forward propagation executes 3 matrix multiplications (`X @ W1`, `a1 @ W2`, `a2 @ W3`), while backpropagation requires 5 matrix multiplications (3 weight-gradient products plus 2 transposed weight products to propagate error backward) along with ReLU derivative masking, making backward passes `1.06x` slower at `W = 16` and `1.35x` slower at `W = 64` as GEMM compute outweighs Python overhead.
