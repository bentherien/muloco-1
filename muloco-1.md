# MuLoCo-1: Key Findings from the Paper

> **MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo**
> Benjamin Therien, Xiaolong Huang, Aaron Defazio, Irina Rish, Eugene Belilovsky
> arXiv: [2505.23725](https://arxiv.org/abs/2505.23725)

This document summarizes the key findings from the MuLoCo paper that are most relevant to the single-worker (K=1) MuLoCo optimizer implemented in this repository.

---

## What is MuLoCo-1?

MuLoCo-1 is the single-worker (K=1) variant of MuLoCo. It runs a Muon inner optimizer for H steps, then applies an outer Nesterov SGD update using the pseudogradient (parameter delta between the snapshot and current parameters). Despite having only a single worker and no distributed communication, MuLoCo-1 offers several advantages over standard data-parallel training.

### Algorithm (K=1)

```
For each sync round n = 1..N:
    1. Save snapshot:       theta_ref = theta
    2. Run H inner steps:   theta = Muon(theta, grad)  x H
    3. Pseudogradient:      delta = theta_ref - theta
    4. Momentum update:     u = mu * u + eta_out * delta
    5. Nesterov step:       theta = theta_ref - mu * u - eta_out * delta
```

---

## Key Findings Relevant to MuLoCo-1

### 1. MuLoCo K=1 Outperforms All Data-Parallel Baselines (150M-3.1B)

At every model scale from 150M to 3.1B parameters (with extensive hyperparameter tuning), K=1 MuLoCo achieves the **lowest final evaluation loss** among all methods tested, including DP Muon, DP AdamW, and K=1 DiLoCo.

| Scale | DP Muon | DP AdamW | MuLoCo K=1 | DiLoCo K=1 |
|-------|---------|----------|------------|------------|
| 150M  | 3.124   | 3.158    | **3.120**  | 3.142      |
| 416M  | 2.641   | 2.682    | **2.638**  | 2.650      |
| 914M  | 2.402   | 2.440    | **2.400**  | 2.411      |
| 1.76B | 2.246   | 2.266    | **2.238**  | 2.265      |
| 3.1B  | 2.128   | 2.145    | **2.122**  | 2.136      |

### 2. MuLoCo K=1 Has Much Larger Critical Batch Sizes

MuLoCo K=1 maintains strong performance at batch sizes where other optimizers degrade significantly. At 3.1B scale, MuLoCo K=1 matches the optimal performance of DP Muon (achieved at 1M token batch) while using an 8M token batch size. This translates to **8x more parallelism** with no performance loss.

The critical batch size of MuLoCo K=1 is projected via power law fits to grow faster with scale than DP Muon, DP AdamW, or DiLoCo K=1.

### 3. MuLoCo K=1 Has a Pareto-Optimal Performance-Time Tradeoff

Because MuLoCo K=1 can leverage much larger batch sizes without performance degradation, it achieves a **Pareto-optimal tradeoff** between training time and final loss. Under the same wall-clock training time, MuLoCo K=1 can reach up to **~10% lower loss** than DP AdamW.

### 4. 15B Scale Validation

At 15B parameters (extrapolated hyperparameters, no sweep):
- MuLoCo K=1 trains at **16M token batch size** (16x larger than DiLoCo K=1's 1M)
- Reaches nearly identical final loss (1.884) to DP Muon (1.864) and DiLoCo K=1 (1.891)
- Mean zero-shot accuracy: MuLoCo K=1 (66.96%) vs DP Muon (66.80%) vs DiLoCo K=1 (66.94%)
- In high-bandwidth settings (6400 Gbit/s), K=1 MuLoCo achieves the **fastest wall-clock training time** of any method (5.2 hours)

### 5. Optimal Hyperparameters for MuLoCo K=1

Key hyperparameter patterns found across scales:

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Inner LR (eta_in) | 0.016 - 0.063 | Muon LR; decreases slowly with scale |
| Outer LR (eta_out) | 0.7 | Constant across scales |
| Outer Momentum (mu) | 0.6 | Lower than DiLoCo K=1 (0.8) |
| Sync Interval (H) | 30 | Used in all experiments |
| Weight Decay | 5e-4 - 5e-3 | Rescaled with batch size per Defazio et al. |

**Important**: MuLoCo K=1 uses lower outer momentum (0.6) than DiLoCo K=1 (0.8). This is because Muon's optimizer steps are already well-aligned, reducing the benefit of additional momentum smoothing.

---

## Why Does MuLoCo Work Better?

### Pseudogradient Quality

The paper provides theoretical and empirical evidence that Muon's orthonormalized optimizer steps produce higher-quality pseudogradients:

1. **Better directional alignment**: Muon's pseudogradients are more aligned with the K=1 (data-parallel equivalent) pseudogradient at all worker counts (Fig. 4 in paper).

2. **Less spectral interference during averaging**: When averaging worker trajectories, DiLoCo's pseudogradient spectrum collapses significantly, while MuLoCo's spectrum is preserved (the "interference gap" decreases for MuLoCo but grows for DiLoCo as K increases).

3. **Stable Frobenius norms**: Muon's orthonormalized steps have constant Frobenius norm across workers and steps, meaning each step contributes equally to the pseudogradient. AdamW's step norms are highly variable.

### Theoretical Result (Proposition 4.2)

The nuclear norm of the pseudogradient equals:

**Muon:** ||Psi||_* = (r/K) * sum of rho * alpha (depends only on alignment and LR)

**AdamW:** ||Psi||_* = (sqrt(r)/K) * sum of rho * alpha * ||psi||_F (also depends on variable step norms)

Where rho is the cosine similarity between individual steps and the pseudogradient's orthonormal factor. Muon's constant step norms mean pseudogradient quality is driven purely by directional alignment.

---

## Compatibility with Communication Compression

While primarily relevant to multi-worker settings (K>1), MuLoCo's compression compatibility is noteworthy:

- **4-bit quantization** is effectively lossless for both MuLoCo and DiLoCo
- **2-bit statistical quantization** works well; 2-bit linear quantization degrades
- **Streaming (partitioned) communication** is fully compatible
- MuLoCo outperforms DiLoCo under all compression schemes tested

---

## Recommended Configuration for MuLoCo K=1

Based on the paper's extensive experiments:

```python
from muloco.pytorch import MuLoCo1

optimizer = MuLoCo1(
    params=param_groups,    # "muon" for 2D+ matrices, "adamw" for rest
    inner_lr=0.02,          # Scale down ~sqrt(2) per model size doubling
    outer_lr=0.7,           # Constant across scales
    outer_momentum=0.6,     # Lower than DiLoCo's 0.8
    sync_interval=30,       # H=30 used throughout
    weight_decay=0.01,      # Rescale with batch size
)
```

---

## Citation

```bibtex
@article{therien2025muloco,
    title={MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo},
    author={Therien, Benjamin and Huang, Xiaolong and Defazio, Aaron and Rish, Irina and Belilovsky, Eugene},
    journal={arXiv preprint arXiv:2505.23725},
    year={2025}
}
```
