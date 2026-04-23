# Appendix: Mathematics Prerequisites for TurboQuant

This appendix builds every mathematical concept used in TurboQuant from the ground up. No prior linear algebra or probability is assumed. Each section introduces a concept, explains it intuitively, shows the formal definition, and provides runnable Python code to solidify understanding.

All code uses **PyTorch** (`torch`) exclusively — no NumPy or SciPy.

---

## Table of Contents

1. [Vectors and Vector Spaces](#1-vectors-and-vector-spaces)
2. [Dot Products and Inner Products](#2-dot-products-and-inner-products)
3. [Norms and Distance](#3-norms-and-distance)
4. [Matrices and Matrix Multiplication](#4-matrices-and-matrix-multiplication)
5. [Transpose and Inverse of a Matrix](#5-transpose-and-inverse-of-a-matrix)
6. [Orthogonal Matrices and Rotations](#6-orthogonal-matrices-and-rotations)
7. [QR Decomposition](#7-qr-decomposition)
8. [Random Gaussian Matrices](#8-random-gaussian-matrices)
9. [Probability Distributions: Gaussian and Beta](#9-probability-distributions-gaussian-and-beta)
10. [Statistical Independence](#10-statistical-independence)
11. [Expected Value, Bias, and Variance](#11-expected-value-bias-and-variance)
12. [Mean Squared Error (MSE)](#12-mean-squared-error-mse)
13. [Scalar Quantization and Codebooks](#13-scalar-quantization-and-codebooks)
14. [The Lloyd-Max Algorithm](#14-the-lloyd-max-algorithm)
15. [Information Theory and Shannon's Distortion Bound](#15-information-theory-and-shannons-distortion-bound)
16. [The Walsh-Hadamard Transform (WHT)](#16-the-walsh-hadamard-transform-wht)
17. [The Johnson-Lindenstrauss Transform (JL)](#17-the-johnson-lindenstrauss-transform-jl)
18. [Softmax and Why Quantization Errors Are Dangerous](#18-softmax-and-why-quantization-errors-are-dangerous)
19. [Bit Packing: From Numbers to Bytes](#19-bit-packing-from-numbers-to-bytes)
20. [Cosine Similarity](#20-cosine-similarity)
21. [Putting It All Together: TurboQuant Step by Step](#21-putting-it-all-together-turboquant-step-by-step)

---

## 1. Vectors and Vector Spaces

### What Is a Vector?

A **vector** is an ordered list of numbers. In machine learning, a KV (key or value) cache entry is a vector — each attention head produces one vector per token.

```
x = [1.2, -0.3, 4.7, 0.1]   ← a 4-dimensional vector
```

The number of elements is the **dimension**, written `d`. For a typical LLM attention head, `d = 128` or `d = 64`.

**Formal notation:** A `d`-dimensional real vector is written `x ∈ ℝ^d`.

### Vector Operations

**Addition (element-wise):**
```
a = [1, 2, 3]
b = [4, 5, 6]
a + b = [5, 7, 9]
```

**Scalar multiplication:**
```
3 · a = [3, 6, 9]
```

```python
import torch

# Create vectors
x = torch.tensor([1.2, -0.3, 4.7, 0.1])
y = torch.tensor([0.5, 2.1, -1.0, 3.3])

print("x:", x)
print("y:", y)
print("x + y:", x + y)
print("3 * x:", 3 * x)
print("dimension of x:", x.shape[0])  # → 4
```

### Why Vectors Matter in TurboQuant

A KV cache entry for one token at one attention layer is a vector of dimension `head_dim` (typically 64–128). TurboQuant compresses this vector from 16-bit floats to 3–4-bit integers. The mathematical problem is: how do you represent a vector using fewer bits while losing as little information as possible?

---

## 2. Dot Products and Inner Products

### Definition

The **dot product** (also called **inner product**) of two vectors `a` and `b` is the sum of the products of their corresponding elements:

```
⟨a, b⟩ = a · b = a[0]·b[0] + a[1]·b[1] + ... + a[d-1]·b[d-1]
```

Example:
```
a = [1, 2, 3]
b = [4, 5, 6]
⟨a, b⟩ = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Three equivalent ways to compute the dot product
print(torch.dot(a, b))       # → 32.0
print(a @ b)                 # → 32.0
print((a * b).sum())         # → 32.0
```

### Why Dot Products Matter in Attention

The core attention computation is:

```
scores = Q @ K^T   (matrix of dot products between query vectors and key vectors)
```

For a single query `q` and key `k`, the attention score is `⟨q, k⟩`. This score determines how much attention is paid to token `k` when generating output at position `q`. **If K is quantized (compressed), the dot product `⟨q, k_quantized⟩` will differ from `⟨q, k_original⟩` — and TurboQuant's entire design is about controlling this error.**

```python
import torch

# Simulated attention: 1 query, 4 keys
d = 8  # head_dim
torch.manual_seed(42)

q = torch.randn(d)
K = torch.randn(4, d)  # 4 tokens, each a d-dim key vector

# Attention scores (unnormalized)
scores = K @ q
print("Attention scores:", scores)
# These scores go through softmax → determine attention weights
```

---

## 3. Norms and Distance

### L2 Norm (Euclidean Length)

The **L2 norm** of a vector `x` is its length in Euclidean space:

```
||x||_2 = √(x[0]² + x[1]² + ... + x[d-1]²)
```

Example:
```
x = [3, 4]
||x||_2 = √(9 + 16) = √25 = 5
```

```python
import torch

x = torch.tensor([3.0, 4.0])
print(torch.linalg.norm(x))          # → 5.0
print(torch.sqrt((x**2).sum()))      # → 5.0 (same thing, explicit)

# For a higher-dimensional vector:
x = torch.randn(128)
print(f"||x||_2 = {torch.linalg.norm(x).item():.4f}")
```

### Distance Between Two Vectors

The **Euclidean distance** between `x` and `y` is `||x - y||_2`. This is the L2 norm of the difference:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 0.0, 1.0])
distance = torch.linalg.norm(x - y)
print(f"distance: {distance.item():.4f}")   # √(9 + 4 + 4) = √17 ≈ 4.123
```

### Norm Correction in TurboQuant

TurboQuant stores the ratio `||x|| / ||x_hat||` alongside the quantized indices. This is called **Norm Correction (NC)**. Here is why: after dequantization, the reconstructed vector `x_hat` may have a slightly different L2 norm from the original `x`. By storing and reapplying the norm ratio, the scale of the attention scores is preserved.

```python
import torch

# Simulate quantization that slightly distorts norm
x = torch.tensor([1.2, -0.3, 4.7, 0.1, 2.2, -1.5])

# Simulate a quantized/dequantized version (rounded to 1 decimal)
x_hat = torch.round(x * 10) / 10

original_norm     = torch.linalg.norm(x)
reconstructed_norm = torch.linalg.norm(x_hat)
norm_ratio        = original_norm / reconstructed_norm

print(f"||x||     = {original_norm.item():.6f}")
print(f"||x_hat|| = {reconstructed_norm.item():.6f}")
print(f"norm ratio = {norm_ratio.item():.6f}")

# Apply norm correction
x_hat_corrected = x_hat * norm_ratio
print(f"||x_hat_corrected|| = {torch.linalg.norm(x_hat_corrected).item():.6f}")
# Now matches ||x|| exactly
```

---

## 4. Matrices and Matrix Multiplication

### What Is a Matrix?

A **matrix** is a 2D array of numbers, with `m` rows and `n` columns. A matrix `A ∈ ℝ^{m×n}` has `m·n` elements:

```
A = [[1, 2, 3],
     [4, 5, 6]]   ← shape: 2×3
```

### Matrix-Vector Multiplication

Multiplying a matrix `A ∈ ℝ^{m×n}` by a vector `x ∈ ℝ^n` produces a new vector `y ∈ ℝ^m`:

```
y = A @ x
y[i] = Σ_j  A[i,j] · x[j]     (dot product of row i of A with x)
```

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)  # shape (2, 3)
x = torch.tensor([1.0, 0.0, 1.0])                  # shape (3,)

y = A @ x
print(y)   # → [4., 10.]
# y[0] = 1×1 + 2×0 + 3×1 = 4
# y[1] = 4×1 + 5×0 + 6×1 = 10
```

### Matrix-Matrix Multiplication

Multiplying `A ∈ ℝ^{m×k}` by `B ∈ ℝ^{k×n}` produces `C ∈ ℝ^{m×n}`:

```
C[i,j] = Σ_k  A[i,k] · B[k,j]
```

```python
A = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float32)  # 2×2
B = torch.tensor([[5, 6],
                  [7, 8]], dtype=torch.float32)  # 2×2

C = A @ B
print(C)
# tensor([[19., 22.],
#         [43., 50.]])
# [[1×5+2×7, 1×6+2×8],   = [[19, 22],
#  [3×5+4×7, 3×6+4×8]]     [43, 50]]

# In TurboQuant: y = Pi @ x  is a matrix-vector product
# where Pi is d×d and x is d-dimensional
```

### Computational Cost

Matrix-vector multiplication `A @ x` where `A` is `d×d` costs `O(d²)` operations. This is important for understanding why TurboQuant replaces QR rotation (O(d²)) with WHT (O(d log d)).

```python
import torch
import time

d = 1024
A = torch.randn(d, d)
x = torch.randn(d)

start = time.perf_counter()
for _ in range(1000):
    y = A @ x
elapsed_matmul = time.perf_counter() - start

print(f"d={d}: 1000x matrix-vector multiply took {elapsed_matmul*1000:.2f} ms")
# Compare this to WHT in Section 16 which will be much faster
```

---

## 5. Transpose and Inverse of a Matrix

### Transpose

The **transpose** of `A ∈ ℝ^{m×n}` is `A^T ∈ ℝ^{n×m}` — rows become columns:

```
A = [[1, 2, 3],      A^T = [[1, 4],
     [4, 5, 6]]             [2, 5],
                            [3, 6]]
```

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)
print(A.T)
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])
```

### Matrix Inverse

The **inverse** of a square matrix `A` is the matrix `A^{-1}` such that:
```
A @ A^{-1} = A^{-1} @ A = I
```
where `I` is the **identity matrix** (1s on diagonal, 0s elsewhere).

```python
A = torch.tensor([[2.0, 1.0],
                  [5.0, 3.0]])
A_inv = torch.linalg.inv(A)

print(A_inv)
print(A @ A_inv)   # Should be identity: [[1, 0], [0, 1]]
```

### Why This Matters for Decode

In TurboQuant, after quantizing the rotated vector `y = Π @ x`, we need to undo the rotation to recover `x_hat`. If `Π` is an orthogonal matrix (next section), then `Π^{-1} = Π^T`, so:

```
x_hat = Π^T @ y_hat
```

This is the decode step — it is just a matrix-vector product with the **transpose** of `Π`.

---

## 6. Orthogonal Matrices and Rotations

### Definition

A square matrix `Q ∈ ℝ^{d×d}` is **orthogonal** if:
```
Q^T @ Q = Q @ Q^T = I
```

This implies `Q^{-1} = Q^T`. Orthogonal matrices are rotation matrices — they rotate vectors without stretching or shrinking them.

### Key Properties

1. **Norm preservation:** `||Q @ x||_2 = ||x||_2` for all `x`
2. **Inner product preservation:** `⟨Q@a, Q@b⟩ = ⟨a, b⟩` for all `a, b`
3. **Inverse = transpose:** `Q^{-1} = Q^T`

```python
import torch

# Generate a random orthogonal matrix via QR decomposition (see Section 7)
torch.manual_seed(0)
G = torch.randn(4, 4)
Q, _ = torch.linalg.qr(G)

# Verify Q^T @ Q = I
print(torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-6))  # → True

# Verify norm preservation
x = torch.randn(4)
print(f"||x||     = {torch.linalg.norm(x).item():.6f}")
print(f"||Q @ x|| = {torch.linalg.norm(Q @ x).item():.6f}")  # Same!

# Verify inner product preservation
y = torch.randn(4)
print(f"<x, y>   = {torch.dot(x, y).item():.6f}")
print(f"<Qx, Qy> = {torch.dot(Q @ x, Q @ y).item():.6f}")  # Same!
```

### Intuition

An orthogonal matrix is a rigid rotation in d-dimensional space. It does not change distances — only directions. After rotation, the vector has the same length and the same angles between vectors are preserved.

**Why TurboQuant uses rotation:** KV vectors have structured outliers in specific channels (e.g., channel 5 is always large). A rotation "spreads" this energy uniformly — after rotation, every coordinate carries roughly equal energy, making each coordinate equally easy to quantize.

```python
import torch

torch.manual_seed(42)

# Simulate a KV vector with an outlier in dimension 0
x = torch.randn(64) * 0.1   # small noise in all dims
x[0] = 10.0                  # outlier in dim 0

# Generate random orthogonal rotation
G = torch.randn(64, 64)
Q, _ = torch.linalg.qr(G)

y = Q @ x   # rotated vector

print(f"Original x: max={x.abs().max().item():.2f}, std={x.std().item():.4f}")
print(f"Rotated  y: max={y.abs().max().item():.2f}, std={y.std().item():.4f}")
# Original has a huge max (outlier); rotated has a small max (spread out)
```

---

## 7. QR Decomposition

### What Is QR Decomposition?

Any matrix `A ∈ ℝ^{m×n}` can be factored as:
```
A = Q · R
```
where:
- `Q ∈ ℝ^{m×m}` is an **orthogonal** matrix (rotation/reflection)
- `R ∈ ℝ^{m×n}` is **upper triangular** (zeros below the diagonal)

```python
import torch

A = torch.randn(4, 4)
Q, R = torch.linalg.qr(A)

print("Q (orthogonal):")
print(Q.round(decimals=3))
print("\nR (upper triangular):")
print(R.round(decimals=3))

# Verify reconstruction
print("\nA ≈ Q @ R:", torch.allclose(A, Q @ R, atol=1e-5))

# Verify Q is orthogonal
print("Q^T @ Q ≈ I:", torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5))
```

### How TurboQuant Uses QR

The TurboQuant paper generates the random rotation matrix `Π` as the `Q` factor from QR decomposition of a **random Gaussian matrix**. Since QR always produces an orthogonal `Q`, this is a clean way to get a uniformly random rotation. The key point: this `Π` is computed **once** at model load time and then stored. All KV vectors for that layer use the same `Π`.

In practice, vLLM replaced QR rotation with WHT (Section 16) because WHT is O(d log d) rather than O(d²).

---

## 8. Random Gaussian Matrices

### What Is a Gaussian (Normal) Distribution?

The **Gaussian** (normal) distribution with mean `μ` and variance `σ²` is written `N(μ, σ²)`. The standard normal is `N(0, 1)`.

Its probability density function:
```
f(x) = (1/√(2π)) · exp(-x²/2)
```

This is the classic bell curve: most values are near 0, with probability falling off symmetrically and exponentially.

```python
import torch

# Sample from N(0, 1)
samples = torch.randn(10000)
print(f"Mean:   {samples.mean().item():.4f}")   # ≈ 0
print(f"Std:    {samples.std().item():.4f}")    # ≈ 1
print(f"Range:  [{samples.min().item():.2f}, {samples.max().item():.2f}]")
```

### A Random Gaussian Matrix

Fill a `d×d` matrix with independent samples from `N(0, 1)`:

```python
import torch

d = 4
G = torch.randn(d, d)   # G[i,j] ~ N(0,1), all independent
print(G)
```

### The Key Property: Rotational Symmetry

A random Gaussian matrix (after QR decomposition) is **rotationally symmetric** — the distribution of the resulting orthogonal matrix `Q` is uniform over all possible rotations. This is called the **Haar measure** on the orthogonal group. For TurboQuant, this means the rotation treats all directions in space equally — it cannot favor or disfavor any particular channel.

```python
import torch

# Verify: after rotation by a random Q, the distribution of Q@x
# is the same as x (for x ~ N(0, I))
torch.manual_seed(0)

G = torch.randn(128, 128)
Q, _ = torch.linalg.qr(G)

x = torch.randn(1000, 128)   # 1000 random vectors
y = (Q @ x.T).T              # rotate each vector

# x and y should have the same distribution
print(f"x: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
print(f"y: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
# They match — rotation preserves the Gaussian distribution
```

---

## 9. Probability Distributions: Gaussian and Beta

### The Standard Normal (Gaussian)

Already covered in Section 8. Mean = 0, std = 1. 68% of values fall within ±1, 95% within ±2.

### The Beta Distribution

The **Beta distribution** `Beta(α, β)` is a continuous distribution supported on `[0, 1]`. Its shape depends on `α` and `β`:
- `α = β = 1`: uniform on [0,1]
- `α = β > 1`: peaked at 0.5 (like a bell on [0,1])
- `α ≠ β`: skewed toward 0 or 1

```python
import torch

# Beta(2, 2) — symmetric, peaked at 0.5
alpha, beta_param = 2.0, 2.0
dist = torch.distributions.Beta(
    torch.tensor(alpha), torch.tensor(beta_param)
)
samples = dist.sample((10000,))

print(f"Beta(2,2): mean={samples.mean().item():.4f}, std={samples.std().item():.4f}")

# Beta(0.5, 0.5) — bimodal, peaks at 0 and 1
dist2 = torch.distributions.Beta(torch.tensor(0.5), torch.tensor(0.5))
samples2 = dist2.sample((10000,))
print(f"Beta(0.5,0.5): mean={samples2.mean().item():.4f}")
```

### Why TurboQuant Uses the Beta Distribution

After random rotation by an orthogonal matrix `Π`, the coordinates of the rotated vector `y = Π @ x` follow a **Beta distribution** (or converge to Gaussian in high dimensions). The specific shape of the distribution is known analytically. This is crucial:

- **Before rotation:** Each coordinate's distribution depends on the token content — no universal formula.
- **After rotation:** Every coordinate follows (approximately) the same Beta distribution — universal, independent of the input.

This allows TurboQuant to pre-compute the optimal codebook for the Beta distribution **once**, offline, and reuse it for all tokens.

```python
import torch

# Simulate: KV vectors from a real workload (outlier in dim 0)
torch.manual_seed(42)
N_vectors = 5000
d = 64

# Simulate KV vectors with outlier structure
X = torch.randn(N_vectors, d) * 0.2
X[:, 0] += 5.0   # systematic outlier in channel 0

# Generate random rotation
G = torch.randn(d, d)
Q, _ = torch.linalg.qr(G)

# Rotate all vectors
Y = (Q @ X.T).T  # shape (N_vectors, d)

# Check coordinate 0 distribution before and after rotation
print("Before rotation, dim 0:")
print(f"  mean={X[:,0].mean().item():.3f}, std={X[:,0].std().item():.3f}")
print("After rotation, dim 0:")
print(f"  mean={Y[:,0].mean().item():.3f}, std={Y[:,0].std().item():.3f}")
print("After rotation, dim 3:")
print(f"  mean={Y[:,3].mean().item():.3f}, std={Y[:,3].std().item():.3f}")
# After rotation, all coordinates have the same distribution
```

---

## 10. Statistical Independence

### What Is Independence?

Two random variables `X` and `Y` are **independent** if knowing the value of one tells you nothing about the other. Formally, their joint distribution factors:
```
P(X=x, Y=y) = P(X=x) · P(Y=y)
```

**Correlation vs independence:** If `X` and `Y` have correlation 0 (uncorrelated), they are not necessarily independent. But if they are jointly Gaussian, then uncorrelated implies independent.

```python
import torch

# Generate correlated variables using MultivariateNormal
mean = torch.zeros(2)
cov  = torch.tensor([[1.0, 0.9],   # high correlation
                     [0.9, 1.0]])
data = torch.distributions.MultivariateNormal(mean, cov).sample((10000,))
print(f"Correlation before rotation: {torch.corrcoef(data.T)[0,1].item():.4f}")  # ≈ 0.9

# Rotate to decorrelate (whitening / PCA via eigendecomposition)
cov_matrix = torch.cov(data.T)
eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
data_decorr = data @ eigenvectors
print(f"Correlation after  rotation: {torch.corrcoef(data_decorr.T)[0,1].item():.6f}")  # ≈ 0
```

### Why Independence Matters for Quantization

If coordinates are **independent**, you can quantize each coordinate separately (scalar quantization) without losing optimality. If they are **dependent**, you need to quantize multiple coordinates jointly (vector quantization), which is exponentially harder.

The random rotation in TurboQuant makes the rotated coordinates **nearly independent** for any input distribution. This transforms an intractable joint quantization problem into `d` independent 1D quantization problems.

```python
import torch

# Demonstrate near-independence after random rotation
torch.manual_seed(1)
d = 128

# Highly correlated KV vectors (simulate transformer activations)
# Use a low-rank structure: x = z @ A^T where A is low-rank
A = torch.randn(d, 4)   # 4 "concepts"
z = torch.randn(10000, 4)
X = z @ A.T + torch.randn(10000, d) * 0.1

# Measure average pairwise correlation before rotation
corr_before = torch.corrcoef(X.T)           # (d, d)
mask = torch.triu(torch.ones(d, d, dtype=torch.bool), diagonal=1)
off_diag = corr_before[mask]
print(f"Avg |correlation| before rotation: {off_diag.abs().mean().item():.4f}")

# Random rotation
G = torch.randn(d, d)
Q, _ = torch.linalg.qr(G)
Y = (Q @ X.T).T

corr_after = torch.corrcoef(Y.T)
off_diag2  = corr_after[mask]
print(f"Avg |correlation| after  rotation: {off_diag2.abs().mean().item():.4f}")
# After rotation, correlation approaches 0 (near-independent)
```

---

## 11. Expected Value, Bias, and Variance

### Expected Value

The **expected value** `E[X]` of a random variable is its average over many trials. For a discrete distribution:
```
E[X] = Σ_x  x · P(X = x)
```

```python
import torch

# Roll a fair 6-sided die 100000 times
rolls = torch.randint(1, 7, (100000,))
print(f"Empirical E[die]: {rolls.float().mean().item():.4f}")   # ≈ 3.5
print(f"Theoretical:      3.5")
```

### Expected Value of a Function

```
E[f(X)] = Σ_x  f(x) · P(X = x)
```

```python
import torch
import math

X = torch.randn(100000)   # standard normal
print(f"E[X]   = {X.mean().item():.4f}")         # ≈ 0
print(f"E[X²]  = {(X**2).mean().item():.4f}")    # ≈ 1 (variance of N(0,1))
print(f"E[|X|] = {X.abs().mean().item():.4f}")   # ≈ √(2/π) ≈ 0.7979
```

### Bias

An **estimator** `X_hat` of a quantity `θ` is **unbiased** if `E[X_hat] = θ`. Otherwise, the **bias** is `E[X_hat] - θ`.

**The bias problem in QJL:** At 1-bit MSE quantization, TurboQuant's inner product estimator has bias:
```
E[⟨Q, DeQuant(Quant(K))⟩] = (2/π) · ⟨Q, K⟩
```
The factor `2/π ≈ 0.637` means attention scores are systematically underestimated by 36.3%. QJL corrects this to make the estimator unbiased.

```python
import torch
import math

# Demonstrate 1-bit inner product bias
torch.manual_seed(0)
d = 256

q = torch.randn(d)
k = torch.randn(d)
true_inner = torch.dot(q, k)

# 1-bit quantization: sign(k)
k_1bit = torch.sign(k)  # each element becomes +1 or -1

# Inner product with 1-bit quantized k
approx_inner = torch.dot(q, k_1bit)

print(f"True inner product:         {true_inner.item():.4f}")
print(f"1-bit approx inner product: {approx_inner.item():.4f}")
print(f"Ratio (actual / true):      {(approx_inner / true_inner).item():.4f}")
print(f"Expected ratio (2/π):       {2/math.pi:.4f}")
# The ratio ≈ 2/π is the systematic bias from 1-bit sign quantization
```

### Variance

The **variance** of a random variable measures its spread:
```
Var[X] = E[(X - E[X])²] = E[X²] - (E[X])²
```

```python
import torch

X = torch.randn(100000)
print(f"Var[X] = {X.var().item():.4f}")   # ≈ 1 (for standard normal)
```

### Bias-Variance Trade-off in TurboQuant

QJL corrects bias (makes `E[estimate] = true value`) but increases variance (each estimate has higher spread). Softmax amplifies high-variance inputs — a small error in attention logits gets exponentially amplified. This is why QJL was dropped in production: **bias correction hurt quality because of variance amplification in softmax**.

---

## 12. Mean Squared Error (MSE)

### Definition

**Mean Squared Error** measures the average squared difference between an estimate and the true value:

```
MSE(x_hat, x) = E[||x - x_hat||²] = E[Σ_i (x[i] - x_hat[i])²]
```

For a single vector: `MSE = (1/d) · ||x - x_hat||²`

```python
import torch

x_true = torch.tensor([1.5, -0.3, 4.2, 0.8])
x_hat  = torch.tensor([1.0, -0.5, 4.0, 1.0])   # approximate

mse  = ((x_true - x_hat)**2).mean()
rmse = mse.sqrt()
print(f"MSE  = {mse.item():.6f}")
print(f"RMSE = {rmse.item():.6f}")
```

### MSE as the Quantization Objective

TurboQuant minimizes MSE:
```
minimize E[||x - DeQuant(Quant(x))||²]
```

Why MSE? It is:
1. **Mathematically tractable** — has a clean closed-form solution
2. **Information-theoretically principled** — directly related to Shannon's distortion theory
3. **Computationally fast** — no sorting, no calibration data

### TurboQuant's Distortion Guarantee

```
MSE(TurboQuant) ≤ (√3·π/2) · (1/4^b)   at bit-width b
Shannon lower bound:          ≥ 1/4^b
```

The factor `√3·π/2 ≈ 2.72` means TurboQuant wastes at most a factor of 2.72 in distortion vs the theoretical best possible quantizer.

```python
import math

# Illustrate the distortion bound at different bit-widths
for b in [1, 2, 3, 4]:
    lower_bound     = 1 / 4**b
    turboquant_bound = (math.sqrt(3) * math.pi / 2) * lower_bound
    factor          = math.sqrt(3) * math.pi / 2
    print(f"b={b}: lower_bound={lower_bound:.6f}, TQ_bound={turboquant_bound:.6f}, factor={factor:.4f}")
```

---

## 13. Scalar Quantization and Codebooks

### What Is Quantization?

**Quantization** maps continuous values to a finite set of discrete values (the **codebook**). The goal is to represent data using fewer bits.

**Uniform scalar quantization** divides the value range `[min, max]` into `2^b` equal-width buckets and maps each value to the nearest bucket center:

```python
import torch

def uniform_quantize(x: torch.Tensor, n_levels: int,
                     x_min=None, x_max=None):
    """
    Quantize a vector x to n_levels discrete values.
    Returns the quantized values (dequantized) and the indices.
    """
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    # Normalize to [0, 1]
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)

    # Quantize to integer indices
    indices = torch.clamp(
        torch.round(x_norm * (n_levels - 1)).long(),
        0, n_levels - 1
    )

    # Dequantize: map indices back to float values
    x_hat = x_min + (indices.float() / (n_levels - 1)) * (x_max - x_min)

    return x_hat, indices

# Example: quantize to 4 levels (2-bit)
x = torch.tensor([0.1, 0.5, 1.2, -0.3, 0.9, 1.8])
x_hat, indices = uniform_quantize(x, n_levels=4)
print("Original:  ", x.round(decimals=3))
print("Quantized: ", x_hat.round(decimals=3))
print("Indices:   ", indices)
print("MSE:       ", ((x - x_hat)**2).mean().item())
```

### Non-Uniform Quantization (Codebook)

Uniform buckets are optimal only for uniformly distributed data. For a non-uniform distribution (like the Beta distribution after rotation), you get better MSE by placing more buckets where the data is dense.

A **codebook** is a set of `k = 2^b` values `{c_1, c_2, ..., c_k}`. Each value of `x` is mapped to the nearest centroid:

```python
import torch

def codebook_quantize(x: torch.Tensor, codebook: torch.Tensor):
    """
    Quantize scalar array x using the given codebook.
    Returns dequantized values and centroid indices.
    """
    # For each element of x, find the nearest centroid
    # distances shape: (len(x), len(codebook))
    distances = (x.unsqueeze(1) - codebook.unsqueeze(0)).abs()
    indices = distances.argmin(dim=1)
    x_hat = codebook[indices]
    return x_hat, indices

# Example with non-uniform codebook optimized for Gaussian data
# (roughly: more centroids near zero, fewer at the tails)
codebook_4level = torch.tensor([-1.5, -0.5, 0.5, 1.5])

x = torch.randn(1000)   # Gaussian data
x_hat, indices = codebook_quantize(x, codebook_4level)
mse_nonuniform = ((x - x_hat)**2).mean()

# Compare to uniform quantization
x_hat_uniform, _ = uniform_quantize(x, 4)
mse_uniform = ((x - x_hat_uniform)**2).mean()

print(f"MSE non-uniform codebook: {mse_nonuniform.item():.6f}")
print(f"MSE uniform codebook:     {mse_uniform.item():.6f}")
# Non-uniform is better for Gaussian data
```

---

## 14. The Lloyd-Max Algorithm

### The Problem

Given a probability distribution `f(x)` and `k = 2^b` centroids, find the codebook `{c_1, ..., c_k}` that minimizes the expected MSE:

```
minimize Σ_i ∫_{bucket_i} (x - c_i)² · f(x) dx
```

This is a 1D version of k-means clustering, but with a known distribution instead of data samples.

### Optimality Conditions

The Lloyd-Max conditions for an optimal codebook are:

1. **Centroid condition:** Each centroid `c_i` is the mean of the distribution within its bucket:
   ```
   c_i = E[x | x ∈ bucket_i]  =  ∫_{bucket_i} x·f(x) dx / ∫_{bucket_i} f(x) dx
   ```

2. **Decision boundary condition:** The boundary `t_i` between bucket `i` and `i+1` is the midpoint of adjacent centroids:
   ```
   t_i = (c_i + c_{i+1}) / 2
   ```

These two conditions are interdependent — satisfying one changes the other. Lloyd-Max solves them iteratively.

### Implementation

```python
import torch
import math

def lloyd_max(distribution_pdf, n_levels, x_range=(-4.0, 4.0),
              n_iterations=100, n_grid=10000):
    """
    Lloyd-Max algorithm: find the optimal codebook for a given distribution.

    Args:
        distribution_pdf: callable f(x: Tensor) -> Tensor giving probability density
        n_levels:    number of codebook entries (2^b for b-bit quantization)
        x_range:     support of the distribution
        n_iterations: number of Lloyd-Max iterations
        n_grid:      number of grid points for numerical integration

    Returns:
        centroids:  optimal codebook (n_levels values), Tensor
        boundaries: decision boundaries (n_levels - 1 values), Tensor
    """
    x_grid   = torch.linspace(x_range[0], x_range[1], n_grid)
    pdf_vals = distribution_pdf(x_grid)
    pdf_vals = pdf_vals / torch.trapz(pdf_vals, x_grid)   # normalize

    # Initialize: equally-spaced centroids
    centroids = torch.linspace(x_range[0] * 0.8, x_range[1] * 0.8, n_levels)

    for iteration in range(n_iterations):
        old_centroids = centroids.clone()

        # Step 1: Update boundaries (midpoints of adjacent centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        boundaries = torch.cat([
            torch.tensor([x_range[0]]),
            boundaries,
            torch.tensor([x_range[1]])
        ])

        # Step 2: Update centroids (mean of distribution within each bucket)
        for i in range(n_levels):
            lo, hi = boundaries[i].item(), boundaries[i + 1].item()
            mask = (x_grid >= lo) & (x_grid < hi)
            if mask.sum() == 0:
                continue
            x_bucket = x_grid[mask]
            p_bucket = pdf_vals[mask]
            denom = torch.trapz(p_bucket, x_bucket)
            if denom < 1e-12:
                continue
            centroids[i] = torch.trapz(x_bucket * p_bucket, x_bucket) / denom

        # Check convergence
        if (centroids - old_centroids).abs().max().item() < 1e-10:
            break

    return centroids, boundaries[1:-1]


# Apply Lloyd-Max to Gaussian distribution (2-bit = 4 levels)
def gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

centroids_2bit, boundaries_2bit = lloyd_max(gaussian_pdf, n_levels=4)
print("2-bit Lloyd-Max centroids for Gaussian:")
print(centroids_2bit.round(decimals=4))
print("Decision boundaries:")
print(boundaries_2bit.round(decimals=4))

# Test: what is the MSE of this codebook?
x_test = torch.randn(100000)
distances = (x_test.unsqueeze(1) - centroids_2bit.unsqueeze(0)).abs()
indices   = distances.argmin(dim=1)
x_hat     = centroids_2bit[indices]
mse       = ((x_test - x_hat)**2).mean()
print(f"\n2-bit Lloyd-Max MSE on Gaussian data: {mse.item():.6f}")

# Compare to 2-bit uniform quantization
x_hat_uniform, _ = uniform_quantize(x_test, 4)
mse_uniform = ((x_test - x_hat_uniform)**2).mean()
print(f"2-bit Uniform     MSE on Gaussian data: {mse_uniform.item():.6f}")
print(f"Lloyd-Max improvement: {(mse_uniform/mse).item():.2f}×")
```

### What TurboQuant Does

1. **Offline:** Apply Lloyd-Max to the Beta distribution (the known distribution of rotated KV coordinates). Store the resulting codebook for each bit-width `b ∈ {1, 2, 3, 4}`.
2. **At inference:** Rotate the KV vector, then use `argmin` distance to find the nearest centroid — just a table lookup per coordinate.

This is why TurboQuant is "data-oblivious" — the codebook is precomputed from the theoretical distribution, not from calibration data.

---

## 15. Information Theory and Shannon's Distortion Bound

### Bits and Information

A **bit** encodes 1 binary choice. With `b` bits you can represent `2^b` distinct values:
- 1 bit → 2 values
- 2 bits → 4 values
- 4 bits → 16 values
- 8 bits (FP8) → 256 values

### Rate-Distortion Theory

Shannon's **rate-distortion theory** answers: "What is the minimum possible MSE when compressing a signal to `R` bits per sample?"

For a **Gaussian source** with variance `σ²`, the Shannon lower bound on distortion at rate `R` bits/sample is:
```
D*(R) = σ² · 2^{-2R}
```

At `R = b` bits per coordinate:
```
D* = σ² / 4^b
```

This is the best any quantizer can possibly achieve — it is a fundamental limit.

```python
import math

sigma_sq = 1.0  # unit variance

print("Rate-distortion bound for N(0,1):")
for b in [1, 2, 3, 4]:
    D_optimal = sigma_sq / 4**b
    print(f"  b={b}: minimum possible MSE = {D_optimal:.8f}")
```

### TurboQuant's Guarantee

TurboQuant achieves:
```
D_TQ ≤ (√3·π/2) · σ² / 4^b    for any input distribution
```

The factor `√3·π/2 ≈ 2.72` means TurboQuant uses at most 0.72 bits more than theoretically necessary. More precisely, it wastes at most half a bit:

```
Extra bits = log2(√3·π/2) / 2 ≈ 0.72 bits   (constant, independent of b)
```

```python
import math

factor     = math.sqrt(3) * math.pi / 2
extra_bits = math.log2(factor) / 2
print(f"TurboQuant overhead factor: {factor:.4f}")
print(f"Extra bits wasted: {extra_bits:.4f} bits")
# This is a hard theoretical guarantee: no input can do worse than this
```

### Why "Data-Oblivious" Matters

Standard FP8 quantization uses the empirical max to set the scale. This works well for the observed data but has no guarantee for out-of-distribution inputs. TurboQuant's guarantee holds for **any input** — the rotation makes the distribution predictable regardless of what tokens the model processes.

---

## 16. The Walsh-Hadamard Transform (WHT)

### What Is WHT?

The **Walsh-Hadamard Transform** is a specific linear transform defined by the Hadamard matrix `H`. For `d = 2^k`:

```
H_1 = [1]

H_2 = [[1,  1],
        [1, -1]]

H_4 = [[1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1]]

H_{2^k} = H_2 ⊗ H_{2^(k-1)}    (Kronecker product)
```

(where `⊗` is the Kronecker product — tensor product of matrices)

```python
import torch

def hadamard_matrix(d: int) -> torch.Tensor:
    """Build the d×d Hadamard matrix (d must be a power of 2)."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H

H4 = hadamard_matrix(4)
print("H_4:")
print(H4)

# Verify: H @ H^T = d * I
d = 4
print(f"\nH @ H^T = {d}·I?",
      torch.allclose(H4 @ H4.T, torch.tensor(float(d)) * torch.eye(d)))
```

### The Normalized WHT

The normalized WHT is `H/√d` — this makes it an orthogonal matrix:

```python
import torch
import math

d = 8
H = hadamard_matrix(d)
Q_wht = H / math.sqrt(d)   # normalized: now orthogonal

# Verify orthogonality
print("(H/√d) @ (H/√d)^T = I?",
      torch.allclose(Q_wht @ Q_wht.T, torch.eye(d), atol=1e-6))

# Apply WHT to a vector
x = torch.arange(1, 9, dtype=torch.float32)  # [1, 2, 3, 4, 5, 6, 7, 8]
y = Q_wht @ x
print("x:", x)
print("y = WHT(x):", y.round(decimals=4))

# Inverse WHT: same operation! (self-inverse)
x_reconstructed = Q_wht @ y
print("Reconstructed x:", x_reconstructed.round(decimals=4))
```

### WHT as a Butterfly Operation

The key insight: you do not need to store the `d×d` Hadamard matrix and multiply. The WHT can be computed by a recursive **butterfly** operation in `log2(d)` stages, each doing `d/2` additions and subtractions. Total cost: `O(d log d)` instead of `O(d²)`.

```python
import torch

def fast_wht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform using butterfly operations.
    x must have length = power of 2.
    Returns H @ x (unnormalized).
    """
    n = x.shape[0]
    assert (n & (n - 1)) == 0, "Length must be a power of 2"
    x = x.clone().float()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            # Vectorised butterfly over the slice pair
            u = x[i : i + h].clone()
            v = x[i + h : i + 2 * h].clone()
            x[i : i + h]     = u + v   # butterfly: +
            x[i + h : i + 2 * h] = u - v   # butterfly: -
        h *= 2
    return x

# Test
x = torch.arange(1, 9, dtype=torch.float32)
y_fast   = fast_wht(x) / math.sqrt(8)                  # fast WHT + normalize
y_matrix = (hadamard_matrix(8) / math.sqrt(8)) @ x     # matrix multiply

print("Fast WHT:", y_fast.round(decimals=6))
print("Matrix  :", y_matrix.round(decimals=6))
print("Match?  :", torch.allclose(y_fast, y_matrix, atol=1e-5))
```

### Speed Comparison

```python
import torch
import math
import time

d = 1024
x = torch.randn(d)
Q = hadamard_matrix(d) / math.sqrt(d)   # full matrix

# Method 1: matrix multiply O(d²)
start = time.perf_counter()
for _ in range(10000):
    y = Q @ x
t_matmul = time.perf_counter() - start

# Method 2: fast butterfly O(d log d)
start = time.perf_counter()
for _ in range(10000):
    y = fast_wht(x) / math.sqrt(d)
t_wht = time.perf_counter() - start

print(f"Matrix multiply O(d²):   {t_matmul*1000:.2f} ms")
print(f"Fast WHT O(d log d):     {t_wht*1000:.2f} ms")
print(f"Speedup: {t_matmul/t_wht:.1f}×")
```

### Random Sign Flips

Pure WHT has a fixed structure — without randomization, it is the same for every model load. vLLM adds **random sign flips** before the WHT:

```
y = WHT(signs * x)    where signs ∈ {-1, +1}^d are fixed random per-layer
```

```python
import torch
import math

torch.manual_seed(42)
d = 64

# Generate fixed random signs per layer (done once at model load)
signs = (torch.randint(0, 2, (d,)) * 2 - 1).float()   # ±1 each

def turboquant_rotate(x: torch.Tensor, signs: torch.Tensor, d: int) -> torch.Tensor:
    """WHT rotation with random sign flips."""
    y = signs * x                          # element-wise sign randomization
    y = fast_wht(y) / math.sqrt(d)        # WHT + normalize
    return y

def turboquant_derotate(y: torch.Tensor, signs: torch.Tensor, d: int) -> torch.Tensor:
    """Inverse: WHT is self-inverse, then undo sign flips."""
    x = fast_wht(y) / math.sqrt(d)        # WHT again (self-inverse up to scale)
    x = signs * x                          # undo sign flips
    return x

x = torch.randn(d)
y = turboquant_rotate(x, signs, d)
x_reconstructed = turboquant_derotate(y, signs, d)

print(f"Reconstruction error: {(x - x_reconstructed).abs().max().item():.2e}")
# Should be near machine epsilon
```

---

## 17. The Johnson-Lindenstrauss Transform (JL)

### The JL Lemma

The **Johnson-Lindenstrauss Lemma** states: any set of `n` points in high-dimensional space can be projected to `O(log n / ε²)` dimensions while preserving all pairwise distances up to a factor of `(1 ± ε)`.

The projection matrix `S ∈ ℝ^{k×d}` is a **random Gaussian matrix**, and the projected vector is `S @ x`.

```python
import torch
import math

def jl_project(x: torch.Tensor, k: int, seed: int = None) -> torch.Tensor:
    """
    Johnson-Lindenstrauss random projection from d to k dimensions.
    x: shape (d,)
    Returns: shape (k,)
    """
    d = x.shape[0]
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        S = torch.randn(k, d, generator=gen) / math.sqrt(k)
    else:
        S = torch.randn(k, d) / math.sqrt(k)
    return S @ x

# Demonstrate distance preservation
d = 512
k = 50   # project from 512 to 50 dims

x = torch.randn(d)
y = torch.randn(d)

dist_original  = torch.linalg.norm(x - y)
x_proj = jl_project(x, k, seed=0)
y_proj = jl_project(y, k, seed=0)   # same seed = same S
dist_projected = torch.linalg.norm(x_proj - y_proj)

print(f"Original distance:  {dist_original.item():.4f}")
print(f"Projected distance: {dist_projected.item():.4f}")
print(f"Ratio:              {(dist_projected/dist_original).item():.4f}")
# Should be close to 1.0
```

### Quantized JL (QJL)

**QJL** further compresses the JL projection to 1 bit by taking `sign(S @ x)`:

```
QJL(x) = sign(S @ x) ∈ {-1, +1}^d
```

The inner product between `x` and `y` can be estimated from 1-bit QJL projections:

```
E[⟨QJL(x), QJL(y)⟩] ∝ arccos(-⟨x/||x||, y/||y||⟩) · ||x|| · ||y||
```

More precisely, for vectors `x` and `y`:
```
E[sign(s_i·x) · sign(s_i·y)] = 1 - (2/π)·θ(x,y)
```
where `θ(x,y) = arccos(⟨x/||x||, y/||y||⟩)` is the angle between them.

```python
import torch
import math

def qjl_encode(x: torch.Tensor, S: torch.Tensor):
    """
    Quantized Johnson-Lindenstrauss encoding.
    S: shape (k, d) — fixed random Gaussian matrix
    Returns: sign(S @ x) ∈ {-1, +1}^k, and ||x||
    """
    projected = S @ x
    signs     = torch.sign(projected)
    norm      = torch.linalg.norm(x)
    return signs, norm

def qjl_inner_product_estimate(signs_x, norm_x, signs_y, norm_y):
    """
    Estimate ⟨x, y⟩ from QJL codes.
    Based on: E[sign(s·x)·sign(s·y)] = 1 - (2/π)·arccos(cos_sim)
    Inverted: cos_sim = cos(π·(1 - agreement))
    """
    # Fraction of matching signs
    agreement = (signs_x == signs_y).float().mean()
    cos_sim   = torch.cos(torch.tensor(math.pi) * (1.0 - agreement))
    estimate  = cos_sim * norm_x * norm_y
    return estimate

# Test QJL inner product estimation
torch.manual_seed(42)
d = 256
k = 256   # 1 bit per coordinate → same storage as d floats of 1 bit each

q   = torch.randn(d)   # query
key = torch.randn(d)   # key

S = torch.randn(k, d)   # fixed random matrix

true_ip       = torch.dot(q, key)
signs_q, nq   = qjl_encode(q, S)
signs_k, nk   = qjl_encode(key, S)
estimated_ip  = qjl_inner_product_estimate(signs_q, nq, signs_k, nk)

print(f"True inner product:      {true_ip.item():.4f}")
print(f"QJL estimated:           {estimated_ip.item():.4f}")
print(f"Relative error:          {(estimated_ip - true_ip).abs().item() / true_ip.abs().item():.4f}")
```

### How QJL Corrects TurboQuant's Bias

The residual `r = x - x_hat_mse` (the part of `x` not captured by Stage 1 MSE quantization) carries the inner product bias. QJL encodes this residual with 1-bit precision and provides an **unbiased estimator** of `⟨q, r⟩`, correcting the bias from Stage 1.

```python
import torch

# Simulate Stage 1 MSE quantization + QJL residual correction
torch.manual_seed(0)
d = 64
q = torch.randn(d)   # query vector
x = torch.randn(d)   # key vector

# Stage 1: quantize (simulate with rounding to 2-bit-like precision)
x_hat_mse = (x * 4).round() / 4

# Residual
r = x - x_hat_mse

# True inner products
true_ip    = torch.dot(q, x)
stage1_ip  = torch.dot(q, x_hat_mse)
residual_ip = torch.dot(q, r)   # this is what QJL should estimate

print(f"True ⟨q, x⟩:          {true_ip.item():.4f}")
print(f"Stage 1 ⟨q, x_hat⟩:   {stage1_ip.item():.4f}")
print(f"Residual ⟨q, r⟩:       {residual_ip.item():.4f}")
print(f"Stage 1 + residual:    {(stage1_ip + residual_ip).item():.4f}")
# stage1 + residual = true (by construction: x = x_hat_mse + r)
```

---

## 18. Softmax and Why Quantization Errors Are Dangerous

### The Softmax Function

**Softmax** converts a vector of raw scores into probabilities that sum to 1:

```
softmax(z)[i] = exp(z[i]) / Σ_j exp(z[j])
```

The exponential function amplifies differences: if `z[0]` is slightly larger than `z[1]`, softmax gives `z[0]` a disproportionately larger probability.

```python
import torch
import torch.nn.functional as F

# Example: how sensitive is softmax to small changes?
z_clean     = torch.tensor([2.0, 1.0, 0.5, 0.1])
z_perturbed = torch.tensor([2.1, 1.0, 0.5, 0.1])   # just +0.1 to first element

probs_clean     = F.softmax(z_clean, dim=0)
probs_perturbed = F.softmax(z_perturbed, dim=0)

print("Clean scores:    ", z_clean)
print("Perturbed scores:", z_perturbed)
print()
print("Clean probs:    ", probs_clean.round(decimals=4))
print("Perturbed probs:", probs_perturbed.round(decimals=4))
print()
print("Change in probs:", (probs_perturbed - probs_clean).round(decimals=4))
# A +0.1 change in one score causes a large probability shift
```

### Softmax Amplifies Quantization Errors

In attention, the scores are `Q @ K^T`. A quantization error `Δ` in the score `z[i]` multiplies the probability by `exp(Δ)`:

```
softmax(z + Δ_i)[i] ≈ softmax(z)[i] · exp(Δ_i)   (when Δ_i is small)
```

```python
import torch
import torch.nn.functional as F
import math

def attention_output(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    d = q.shape[-1]
    scores  = (K @ q) / math.sqrt(d)   # (n_tokens,)
    weights = F.softmax(scores, dim=0)  # (n_tokens,)
    return weights @ V                  # (d,)

torch.manual_seed(42)
d = 32
q = torch.randn(d)
K = torch.randn(6, d)   # 6 key vectors
V = torch.randn(6, d)   # 6 value vectors

# Clean attention output
output_clean = attention_output(q, K, V)

# Add Gaussian noise to K (simulate quantization error)
noise_level = 0.1
K_noisy     = K + torch.randn_like(K) * noise_level
output_noisy = attention_output(q, K_noisy, V)

error = torch.linalg.norm(output_clean - output_noisy)
print(f"Noise level:          {noise_level:.4f}")
print(f"Output error (||·||): {error.item():.6f}")
print(f"Relative error:       {(error / torch.linalg.norm(output_clean)).item():.4f}")
```

### Why QJL's Variance Is Dangerous

QJL corrects bias (mean error → 0) but increases variance (individual errors become larger). In softmax, variance is more damaging than bias:

- **Bias:** Every attention score is scaled by the same factor — softmax's relative probabilities are preserved.
- **Variance:** Different tokens have different error magnitudes — softmax redistributes probability mass incorrectly.

```python
import torch
import torch.nn.functional as F

# Demonstrate: biased-but-low-variance vs unbiased-but-high-variance
torch.manual_seed(0)

n_experiments = 5000
d = 32
q = torch.randn(d)
K = torch.randn(4, d)
V = torch.randn(4, d)

output_clean = attention_output(q, K, V)

errors_biased        = []
errors_unbiased_highvar = []

for _ in range(n_experiments):
    # Biased estimator: scales all scores by 0.8 (systematic but consistent)
    out_biased = attention_output(q, K * 0.8, V)
    errors_biased.append(torch.linalg.norm(out_biased - output_clean).item())

    # Unbiased but high-variance: each key gets independent noise
    K_noisy = K + torch.randn_like(K) * 0.5
    out_noisy = attention_output(q, K_noisy, V)
    errors_unbiased_highvar.append(torch.linalg.norm(out_noisy - output_clean).item())

print(f"Biased (0.8×), low variance:   mean error = {torch.tensor(errors_biased).mean().item():.6f}")
print(f"Unbiased, high variance:        mean error = {torch.tensor(errors_unbiased_highvar).mean().item():.6f}")
# High variance hurts more even though unbiased!
```

---

## 19. Bit Packing: From Numbers to Bytes

### Binary Representation

Every integer has a binary representation. A `b`-bit integer can represent `2^b` values (0 through `2^b - 1`):

```
4-bit unsigned integer: 0000 → 0, 0001 → 1, ..., 1111 → 15
```

```python
# Convert between decimal and binary
for i in range(8):
    print(f"  {i} = {i:04b}")   # 4-bit binary representation
```

### Packing Multiple Values into One Byte

A uint8 (1 byte = 8 bits) can hold:
- 2 values of 4 bits each (called "nibbles")
- 4 values of 2 bits each
- 8 values of 1 bit each

This is **bit packing** — the process of cramming multiple low-bit values into a standard byte.

```python
import torch

def pack_4bit(a: int, b: int) -> torch.Tensor:
    """Pack two 4-bit values (0-15 each) into one uint8 byte."""
    assert 0 <= a <= 15 and 0 <= b <= 15, "Values must be 0-15"
    # a occupies high 4 bits, b occupies low 4 bits
    return torch.tensor((a << 4) | b, dtype=torch.uint8)

def unpack_4bit(byte: torch.Tensor):
    """Unpack one uint8 byte into two 4-bit values."""
    val = byte.item()
    a = val >> 4          # high nibble
    b = val & 0x0F        # low nibble (mask out high bits)
    return a, b

# Example: pack indices 5 and 11 into one byte
byte = pack_4bit(5, 11)
val  = byte.item()
print(f"Packed:   byte = {val} = {val:08b}")   # → 01011011

a, b = unpack_4bit(byte)
print(f"Unpacked: a={a}, b={b}")    # → 5, 11

# Pack a whole array of 4-bit indices
def pack_array_4bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack an array of 4-bit indices (0-15) into a uint8 tensor half the size.
    Assumes even number of indices.
    """
    idx = indices.to(torch.uint8)
    return (idx[0::2] << 4) | idx[1::2]

def unpack_array_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor back to 4-bit indices."""
    hi = (packed >> 4).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.uint8)
    return torch.stack([hi, lo], dim=1).reshape(-1)

# Test with 8 indices
indices = torch.tensor([0, 5, 3, 12, 7, 1, 15, 8], dtype=torch.uint8)
packed   = pack_array_4bit(indices)
recovered = unpack_array_4bit(packed)

print(f"\nOriginal indices: {indices.tolist()}")
print(f"Packed ({len(packed)} bytes): {packed.tolist()}")
print(f"Recovered:       {recovered.tolist()}")
print(f"Match: {indices.tolist() == recovered.tolist()}")
```

### Full TurboQuant Pack/Unpack Cycle

```python
import torch
import math

def turboquant_encode(x: torch.Tensor, codebook: torch.Tensor,
                      signs: torch.Tensor, d: int):
    """
    Full TurboQuant encode: rotate → quantize → pack.
    x:        (d,) float vector
    codebook: (2^b,) float centroids
    signs:    (d,) tensor of ±1 for WHT randomization
    Returns:  packed uint8 tensor, scalar norm_ratio
    """
    # 1. WHT rotation with sign flips
    y = fast_wht(signs * x) / math.sqrt(d)

    # 2. Find nearest centroid index for each coordinate
    distances = (y.unsqueeze(1) - codebook.unsqueeze(0)).abs()   # (d, n_levels)
    indices   = distances.argmin(dim=1).to(torch.uint8)           # (d,)

    # 3. Dequantize to compute norm correction
    y_hat = codebook[indices.long()]
    x_hat = signs * (fast_wht(y_hat) / math.sqrt(d))   # inverse WHT

    norm_x     = torch.linalg.norm(x)
    norm_x_hat = torch.linalg.norm(x_hat)
    norm_ratio = (norm_x / (norm_x_hat + 1e-8)).item()

    # 4. Bit-pack indices (4-bit for this example)
    packed = pack_array_4bit(indices)   # (d//2,) uint8

    return packed, norm_ratio


def turboquant_decode(packed: torch.Tensor, norm_ratio: float,
                      codebook: torch.Tensor, signs: torch.Tensor, d: int) -> torch.Tensor:
    """
    Full TurboQuant decode: unpack → dequantize → derotate → norm correct.
    """
    # 1. Unpack
    indices = unpack_array_4bit(packed)[:d].long()   # (d,)

    # 2. Centroid lookup
    y_hat = codebook[indices]   # (d,)

    # 3. Inverse WHT + undo sign flip
    x_hat = signs * (fast_wht(y_hat) / math.sqrt(d))

    # 4. Apply norm correction
    return x_hat * norm_ratio


# Full round-trip test
d = 64
torch.manual_seed(99)

# Setup (done once at model load)
signs        = (torch.randint(0, 2, (d,)) * 2 - 1).float()
codebook_4bit, _ = lloyd_max(gaussian_pdf, n_levels=16)   # 4-bit = 16 levels

# Encode
x = torch.randn(d)   # simulate a KV vector
packed, norm_ratio = turboquant_encode(x, codebook_4bit, signs, d)

print(f"Original:   {d} floats × 32 bits = {d*32} bits")
print(f"Packed:     {len(packed)} bytes = {len(packed)*8} bits")
print(f"Norm ratio: {norm_ratio:.6f}")

# Decode
x_hat = turboquant_decode(packed, norm_ratio, codebook_4bit, signs, d)

mse        = ((x - x_hat)**2).mean()
cosine_sim = torch.dot(x, x_hat) / (torch.linalg.norm(x) * torch.linalg.norm(x_hat))
print(f"MSE after round-trip: {mse.item():.6f}")
print(f"Cosine similarity:    {cosine_sim.item():.6f}")
```

---

## 20. Cosine Similarity

### Definition

**Cosine similarity** measures the angle between two vectors, ignoring their magnitudes:

```
cos_sim(a, b) = ⟨a, b⟩ / (||a|| · ||b||) = cos(θ)
```

- `cos_sim = 1`: vectors point in the same direction (θ = 0°)
- `cos_sim = 0`: vectors are perpendicular (θ = 90°)
- `cos_sim = -1`: vectors point in opposite directions (θ = 180°)

```python
import torch

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))

a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.707, 0.707, 0.0])   # 45° from a
c = torch.tensor([0.0, 1.0, 0.0])       # 90° from a

print(f"cos_sim(a, a) = {cosine_similarity(a, a).item():.4f}")   # 1.0
print(f"cos_sim(a, b) = {cosine_similarity(a, b).item():.4f}")   # ≈ 0.707
print(f"cos_sim(a, c) = {cosine_similarity(a, c).item():.4f}")   # 0.0
```

### Cosine Similarity as a Quality Metric

TurboQuant's diagnostic tool uses cosine similarity per layer to verify that quantization does not distort the direction of KV vectors:

```bash
python -m turboquant_vllm.verify \
  --model meta-llama/Llama-3.1-8B \
  --k-bits 4 --v-bits 3 \
  --threshold 0.97   # minimum acceptable cosine similarity per layer
```

A threshold of 0.97 means: after quantization, the angle between the original and reconstructed KV vector must be less than `arccos(0.97) ≈ 14°`.

```python
import torch

def verify_quantization_quality(x: torch.Tensor, x_hat: torch.Tensor,
                                 threshold: float = 0.97) -> dict:
    """
    Check if quantization preserves direction well enough.
    Returns True if cosine similarity ≥ threshold.
    """
    cos_sim      = cosine_similarity(x, x_hat)
    angle_degrees = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
    passed       = cos_sim.item() >= threshold
    return {
        'cosine_similarity': cos_sim.item(),
        'angle_degrees':     angle_degrees.item(),
        'passed':            passed
    }

# Test with our TurboQuant round-trip from Section 19
result = verify_quantization_quality(x, x_hat, threshold=0.97)
print(f"Cosine similarity: {result['cosine_similarity']:.6f}")
print(f"Angle:             {result['angle_degrees']:.2f}°")
print(f"Passed threshold:  {result['passed']}")
```

---

## 21. Putting It All Together: TurboQuant Step by Step

### Full Mathematical Summary

Here is every step of TurboQuant with the math behind each decision:

| Step | Operation | Math | Why |
|------|-----------|------|-----|
| 1 | Random sign flip | `x' = signs ⊙ x` | Randomize direction for WHT |
| 2 | WHT | `y = H·x'/√d` | O(d log d) orthogonal rotation |
| 3 | Lloyd-Max quantize | `idx = argmin_i |y_j - c_i|` | Optimal for Beta distribution |
| 4 | Pack indices | `bytes = pack(idx, b bits)` | Memory compression |
| 5 | Store norm ratio | `ratio = ‖x‖/‖x̂‖` | Norm Correction |
| D1 | Unpack | `idx = unpack(bytes)` | Recover indices |
| D2 | Centroid lookup | `ŷ = codebook[idx]` | Dequantize |
| D3 | Inverse WHT + sign | `x̂ = signs ⊙ H·ŷ/√d` | Undo rotation |
| D4 | Norm correction | `x̂ ← x̂ · ratio` | Restore true scale |

### Complete, Self-Contained Demo

```python
import torch
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────

def fast_wht(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    x = x.clone().float()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            u = x[i : i + h].clone()
            v = x[i + h : i + 2 * h].clone()
            x[i : i + h]         = u + v
            x[i + h : i + 2 * h] = u - v
        h *= 2
    return x

def lloyd_max_gaussian(n_levels: int, x_range=(-4.0, 4.0),
                        n_iter: int = 200, n_grid: int = 10000) -> torch.Tensor:
    x   = torch.linspace(x_range[0], x_range[1], n_grid)
    pdf = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    pdf = pdf / torch.trapz(pdf, x)
    centroids = torch.linspace(x_range[0] * 0.8, x_range[1] * 0.8, n_levels)
    for _ in range(n_iter):
        old    = centroids.clone()
        bounds = torch.cat([torch.tensor([x_range[0]]),
                            (centroids[:-1] + centroids[1:]) / 2,
                            torch.tensor([x_range[1]])])
        for i in range(n_levels):
            lo, hi = bounds[i].item(), bounds[i + 1].item()
            m = (x >= lo) & (x < hi)
            if m.sum() == 0:
                continue
            denom = torch.trapz(pdf[m], x[m])
            if denom < 1e-12:
                continue
            centroids[i] = torch.trapz(x[m] * pdf[m], x[m]) / denom
        if (centroids - old).abs().max().item() < 1e-10:
            break
    return centroids

def pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    idx = indices.to(torch.uint8)
    return (idx[0::2] << 4) | idx[1::2]

def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    hi = (packed >> 4).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.uint8)
    return torch.stack([hi, lo], dim=1).reshape(-1)


# ─────────────────────────────────────────────────────────────────────
# Model-level initialization
# ─────────────────────────────────────────────────────────────────────

torch.manual_seed(7)
HEAD_DIM = 64             # typical transformer head dimension
BITS     = 4              # 4-bit quantization
N_LEVELS = 2**BITS        # 16 centroids

# Precompute once per layer:
signs    = (torch.randint(0, 2, (HEAD_DIM,)) * 2 - 1).float()  # random ±1 sign flip
codebook = lloyd_max_gaussian(N_LEVELS)                          # optimal Lloyd-Max centroids

print("=== TurboQuant Setup ===")
print(f"HEAD_DIM = {HEAD_DIM}, BITS = {BITS}, N_LEVELS = {N_LEVELS}")
print(f"Codebook: {codebook.round(decimals=3)}")


# ─────────────────────────────────────────────────────────────────────
# Encode  (called during prefill / decode, for each token×head)
# ─────────────────────────────────────────────────────────────────────

def tq_encode(k_vec: torch.Tensor, codebook: torch.Tensor,
              signs: torch.Tensor, head_dim: int):
    """Encode a single key vector to TurboQuant compressed format."""

    # Steps 1 & 2: sign flip + WHT rotation
    k_rotated = fast_wht(signs * k_vec) / math.sqrt(head_dim)

    # Step 3: Lloyd-Max quantize — nearest centroid per coordinate
    dist    = (k_rotated.unsqueeze(1) - codebook.unsqueeze(0)).abs()  # (d, n_levels)
    indices = dist.argmin(dim=1).to(torch.uint8)                        # (d,)

    # Compute norm correction: dequantize → inverse-rotate → compare norms
    k_hat_rot = codebook[indices.long()]
    k_hat     = signs * (fast_wht(k_hat_rot) / math.sqrt(head_dim))

    norm_k     = torch.linalg.norm(k_vec)
    norm_k_hat = torch.linalg.norm(k_hat)
    norm_ratio = (norm_k / (norm_k_hat + 1e-8)).item()

    # Step 4: Pack 4-bit indices
    packed = pack_nibbles(indices)   # (d//2,) uint8

    return packed, norm_ratio


# ─────────────────────────────────────────────────────────────────────
# Decode  (called during decode-phase attention)
# ─────────────────────────────────────────────────────────────────────

def tq_decode(packed: torch.Tensor, norm_ratio: float,
              codebook: torch.Tensor, signs: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Decode a TurboQuant compressed key vector."""

    # Step D1: Unpack
    indices = unpack_nibbles(packed)[:head_dim].long()   # (d,)

    # Step D2: Centroid lookup
    k_hat_rot = codebook[indices]   # (d,)

    # Step D3: Inverse WHT + undo sign flip
    k_hat = signs * (fast_wht(k_hat_rot) / math.sqrt(head_dim))

    # Step D4: Norm correction
    return k_hat * norm_ratio


# ─────────────────────────────────────────────────────────────────────
# Run the demo: compress and reconstruct KV vectors
# ─────────────────────────────────────────────────────────────────────

print("\n=== Round-trip Test ===")

# Simulate 100 KV vectors with outlier structure (like real transformers)
n_tokens = 100
K_true   = torch.randn(n_tokens, HEAD_DIM) * 0.5
K_true[:, 0] += 3.0   # outlier in channel 0

mse_list, cos_sim_list = [], []
orig_bits = n_tokens * HEAD_DIM * 16     # 16-bit BF16

for i in range(n_tokens):
    k     = K_true[i]
    packed, nr = tq_encode(k, codebook, signs, HEAD_DIM)
    k_hat = tq_decode(packed, nr, codebook, signs, HEAD_DIM)

    mse_list.append(((k - k_hat)**2).mean().item())
    cos_sim_list.append(
        (torch.dot(k, k_hat) / (torch.linalg.norm(k) * torch.linalg.norm(k_hat))).item()
    )

comp_bits = n_tokens * (HEAD_DIM // 2 * 8 + 32)  # 4-bit packed + 1 FP32 norm

print(f"Original size:     {orig_bits} bits")
print(f"Compressed size:   {comp_bits} bits")
print(f"Compression ratio: {orig_bits/comp_bits:.2f}×")

mse_t     = torch.tensor(mse_list)
cos_sim_t = torch.tensor(cos_sim_list)
print(f"\nMean MSE:          {mse_t.mean().item():.6f}")
print(f"Mean cosine sim:   {cos_sim_t.mean().item():.6f}")
print(f"Min  cosine sim:   {cos_sim_t.min().item():.6f}")
print(f"Frac above 0.97:   {(cos_sim_t >= 0.97).float().mean().item():.2f}")

print("\n=== Attention Score Accuracy Test ===")

q           = torch.randn(HEAD_DIM)     # query vector
true_scores = K_true @ q               # shape (n_tokens,)

# Compute attention scores using TurboQuant-decoded keys
K_hat = torch.zeros_like(K_true)
for i in range(n_tokens):
    packed, nr = tq_encode(K_true[i], codebook, signs, HEAD_DIM)
    K_hat[i]  = tq_decode(packed, nr, codebook, signs, HEAD_DIM)

tq_scores = K_hat @ q

w_true = F.softmax(true_scores / math.sqrt(HEAD_DIM), dim=0)
w_tq   = F.softmax(tq_scores   / math.sqrt(HEAD_DIM), dim=0)

print(f"Max attention weight error:  {(w_true - w_tq).abs().max().item():.6f}")
print(f"Mean attention weight error: {(w_true - w_tq).abs().mean().item():.6f}")
print(f"Top-1 match: {w_true.argmax().item() == w_tq.argmax().item()}")
# The top-attended token should be the same after quantization
```

---

## Quick Reference: Math Symbols Used in TurboQuant

| Symbol | Meaning | Section |
|--------|---------|---------|
| `d` | Head dimension (e.g., 64 or 128) | §1 |
| `x ∈ ℝ^d` | Original KV vector | §1 |
| `⟨a, b⟩` | Inner product (dot product) | §2 |
| `‖x‖` | L2 norm of x | §3 |
| `Π ∈ ℝ^{d×d}` | Random rotation matrix | §6 |
| `Π^T` | Transpose (= inverse for orthogonal Π) | §5 |
| `y = Π @ x` | Rotated vector | §6 |
| `N(0, 1)` | Standard Gaussian distribution | §8–9 |
| `Beta(α, β)` | Beta distribution (distribution of rotated coords) | §9 |
| `E[X]` | Expected value of X | §11 |
| `Var[X]` | Variance of X | §11 |
| `MSE` | Mean Squared Error | §12 |
| `c_1...c_{2^b}` | Codebook centroids | §13–14 |
| `f_Beta(x)` | PDF of Beta distribution | §14 |
| `D*(R)` | Shannon distortion lower bound at rate R | §15 |
| `H` | Hadamard matrix | §16 |
| `WHT(x) = H@x/√d` | Walsh-Hadamard Transform | §16 |
| `signs ∈ {±1}^d` | Random sign flip vector | §16 |
| `S ∈ ℝ^{d×d}` | Random Gaussian matrix for QJL | §17 |
| `sign(z)` | Sign function: +1 if z>0, -1 if z<0 | §17 |
| `‖r‖₂` | L2 norm of residual r | §17 |
| `softmax(z)` | Softmax function | §18 |
| `cos_sim(a,b)` | Cosine similarity | §20 |

---

## Further Reading

- **Information theory:** Cover & Thomas, *Elements of Information Theory* (Ch. 10 on Rate-Distortion)
- **Quantization theory:** Gersho & Gray, *Vector Quantization and Signal Compression* (Ch. 5 for Lloyd-Max)
- **Random projections:** Dasgupta & Gupta (1999), *An elementary proof of the Johnson-Lindenstrauss Lemma*
- **Walsh-Hadamard Transform:** Beauchamp, *Walsh Functions and Their Applications* (Ch. 2)
- **TurboQuant paper:** Google Research, ICLR 2026 (cited in `07_turboquant.md`)
- **vLLM PR #38479:** [github.com/vllm-project/vllm/pull/38479](https://github.com/vllm-project/vllm/pull/38479)
