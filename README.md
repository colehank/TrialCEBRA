# TrialCEBRA
*[中文](README_zh.md)*

[![PyPI](https://img.shields.io/pypi/v/TrialCEBRA?color=blue)](https://pypi.org/project/TrialCEBRA/)
[![Tests](https://github.com/colehank/TrialCEBRA/actions/workflows/tests.yml/badge.svg)](https://github.com/colehank/TrialCEBRA/actions)

**Trial-aware contrastive learning for CEBRA** — a wrapper that adds five trial-structured sampling conditionals to [CEBRA](https://cebra.ai) without modifying its source code.

Designed for neuroscience experiments where neural recordings are organized as repeated trials (stimuli, conditions, epochs). Positive-pair selection is lifted from the *timepoint* level to the *trial* level: first select a target trial by stimulus similarity or at random, then draw a positive timepoint within that trial.

---

## Background

CEBRA's native conditionals (`time`, `delta`, `time_delta`) operate over a flat sequence of timepoints. For trial-structured data they have two limitations:

1. **Temporal boundary artifacts** — a 1-D CNN convolves across trial boundaries, mixing pre- and post-stimulus activity.
2. **Flat sampling ignores trial structure** — `delta` finds the nearest-neighbor timepoint in stimulus space; when all timepoints within a trial share the same stimulus embedding, this collapses to intra-trial sampling with no cross-trial signal.

`trial_cebra` solves both by lifting positive-pair selection to the *trial* level.

---

## Installation

**Step 1 — Install PyTorch** for your hardware from [pytorch.org](https://pytorch.org/get-started/locally/) (select your CUDA version or CPU).

**Step 2 — Install TrialCEBRA:**

```bash
pip install TrialCEBRA
```

---

## Quick Start

```python
import numpy as np
from trial_cebra import TrialCEBRA

# Neural data: (N_timepoints, neural_dim)
X = np.random.randn(2000, 64).astype(np.float32)

# Continuous auxiliary variable (e.g. stimulus embedding): (N_timepoints, stim_dim)
y_cont = np.random.randn(2000, 16).astype(np.float32)

# Trial boundaries: 40 trials × 50 timepoints each
trial_starts = np.arange(0,   2000, 50)
trial_ends   = np.arange(50,  2001, 50)

model = TrialCEBRA(
    model_architecture = "offset10-model",
    conditional        = "trial_delta",
    time_offsets       = 5,
    delta              = 0.3,
    output_dimension   = 3,
    max_iterations     = 1000,
    batch_size         = 512,
)

model.fit(X, y_cont, trial_starts=trial_starts, trial_ends=trial_ends)
embeddings = model.transform(X)   # (N_timepoints, 3)
```

### Epoch format (ntrial × ntime × nneuro)

If your data are already organized as epochs, pass the 3-D array directly — trial boundaries are inferred automatically:

```python
X_ep = np.random.randn(40, 50, 64).astype(np.float32)

y_pertrial     = np.random.randn(40, 16).astype(np.float32)       # (ntrial, stim_dim)
y_pertimepoint = np.random.randn(40, 50, 16).astype(np.float32)   # (ntrial, ntime, stim_dim)

model.fit(X_ep, y_pertrial)          # auto-detects 3-D
emb = model.transform_epochs(X_ep)   # (ntrial, ntime, output_dimension)
```

**Label broadcasting rules for 3-D input:**

| Label shape | Interpretation | Output shape |
|---|---|---|
| `(ntrial,)` | per-trial discrete | `(ntrial*ntime,)` |
| `(ntrial, d)` where `d ≠ ntime` | per-trial continuous | `(ntrial*ntime, d)` |
| `(ntrial, ntime)` | per-timepoint | `(ntrial*ntime,)` |
| `(ntrial, ntime, d)` | per-timepoint | `(ntrial*ntime, d)` |

---

## Conditionals

Five trial-aware conditionals organized along three orthogonal axes:

| Axis | Options |
|---|---|
| **Trial selection** | Random (uniform) · Gaussian delta-style · Gaussian time_delta-style |
| **Time constraint** | `Time` — ±`time_offset` relative position within target trial · Free — uniform within trial |
| **Locking** | Locked — fixed mapping pre-computed at `__init__` · Re-sampled — independent per training step |

### Conditional reference

| `conditional` | Trial selection | Time constraint | Locking | Gap strategy |
|---|---|---|---|---|
| `"trialTime"` | Random | ±`time_offset` | — | global ±`time_offset` (class-uniform with discrete) |
| `"trialDelta"` | delta-style | Free | **Locked** | delta-style at timepoint level |
| `"trial_delta"` | delta-style | Free | Re-sampled | delta-style at timepoint level |
| `"trialTime_delta"` | delta-style | ±`time_offset` | Re-sampled | delta-style at timepoint level |
| `"trialTime_trialDelta"` | time_delta-style | ±`time_offset` | **Locked** | time_delta-style at timepoint level |

Native CEBRA conditionals (`"time"`, `"delta"`, `"time_delta"`, etc.) pass through unchanged.

### Naming convention

| Pattern | Meaning |
|---|---|
| `trialDelta` | capital D, no underscore → **Locked**, delta-style Gaussian |
| `trial_delta` | underscore + lowercase → **Re-sampled**, delta-style Gaussian |
| `trialTime` | Random trial + time constraint |
| `trialTime_delta` | Time constraint + Re-sampled delta-style |
| `trialTime_trialDelta` | Time constraint + Locked delta-style (time_delta mechanism) |

---

## How Sampling Works

### Trial selection: delta-style

Used by `trialDelta`, `trial_delta`, and `trialTime_delta`. Mirrors CEBRA's `DeltaNormalDistribution` at the trial level:

```
query        = trial_mean[anchor_trial] + N(0, δ²I)
target_trial = argmin_j  dist(query, trial_mean[j])
```

Each trial is represented by the **mean** of its timepoints' auxiliary variable. `δ` controls the exploration radius — small `δ` picks the most similar trial, large `δ` explores broadly. Noise is re-drawn every step, so the same anchor may pair with different trials across iterations.

### Trial selection: time_delta-style

Used only by `trialTime_trialDelta`. Mirrors CEBRA's `TimedeltaDistribution` at the trial level:

```
Δstim[k]     = continuous[k] - continuous[k − time_offset]   (pre-computed)
query        = trial_mean[anchor_trial] + Δstim[random_k]
target_trial = argmin_j  dist(query, trial_mean[j])
```

Uses empirical stimulus-velocity vectors as perturbations — data-driven rather than isotropic.

### Locked vs Re-sampled

| | Locked (`trialDelta`, `trialTime_trialDelta`) | Re-sampled (`trial_delta`, `trialTime_delta`) |
|---|---|---|
| Target trial | Pre-computed once at `__init__`, fixed | Independently drawn every training step |
| Gradient signal | Consistent — same trial pair repeated | Diverse — anchor sees different similar trials |
| Generalization | May learn pair-specific features | Learns features valid across all similar trials |
| Best for | Few trials, stable training | Many trials, rich stimulus content |

---

## Visualizing Sampling Behavior

The figures below are produced by `example/viz_trial_sampling.py` on real MEG data with ImageNet stimuli. Each panel shows **R** (reference anchor), **+** (positive samples), **−** (negative samples). Border color encodes in-trial time position (colorbar on the right; black = gap timepoints).

### Trial sampling: R / + / −

![Trial sampling](resources/fig_trial_sampling.png)

- **`trialTime`** — positives from a uniformly random other trial, centered near the anchor's relative time. Stimulus grid is diverse with no similarity bias.
- **`trialDelta`** — positives cluster on a single *locked* target trial (fixed by stimulus similarity at init). All positive frames show the same image, confirming the fixed mapping.
- **`trial_delta`** — target trial is re-sampled every step. Positive frames spread across several similar stimuli while maintaining content coherence.
- **`trialTime_delta`** — same trial diversity as `trial_delta`, but additionally constrained to ±`time_offset` of the anchor's relative position.
- **`trialTime_trialDelta`** — locked target trial + time window. Positives concentrate on a single stimulus image at a specific post-stimulus latency.

### Sampling timeline

![Sampling timeline](resources/fig_sampling.png)

Each sampled frame is placed on a timeline spanning the full trial duration. The green band marks the ±`time_offset` window around the anchor's relative position.

---

## Learned Embeddings

All eight conditionals (3 native CEBRA + 5 trial-aware) trained on the same MEG dataset for 10 000 iterations. Points colored by **in-trial time** (black = pre-stimulus / gap; yellow-green = late post-stimulus).

### 3D embeddings colored by time

![3D embeddings](resources/fig_3d_embeddings.png)

**Native CEBRA (top row):** `time` — uniform sphere, no temporal structure. `time_delta` — similar but with weak temporal gradients. `delta` — stimulus content dominates; gap frames collapse to a single dark patch.

**Trial-aware TrialCEBRA (bottom row):** `trialTime_delta` — clearest temporal ring with gap frames separated into a distinct cluster. `trialTime` — similar ring, smoother gradient. `trialDelta` — clean gap separation, more scattered trial frames. `trial_delta` — more uniform embedding of trial frames. `trialTime_trialDelta` — tightest per-latency clustering.

### Training loss

![Loss curves](resources/fig_loss.png)

All conditionals converge smoothly. Trial-aware conditionals start at higher loss (richer contrastive task) and converge to a similar level as native conditionals.

---

## Gap (Inter-trial) Timepoints

Timepoints between trials are **valid anchors**. Each conditional defines a fallback strategy:

| `conditional` | Gap strategy |
|---|---|
| `trialTime` | Global ±`time_offset` window; with discrete labels → global class-uniform (Gumbel-max) |
| `trialDelta` | delta-style at timepoint level |
| `trial_delta` | delta-style at timepoint level |
| `trialTime_delta` | delta-style at timepoint level |
| `trialTime_trialDelta` | time_delta-style at timepoint level |

> **Tip:** Pass a discrete label array marking trial vs. gap (e.g. `0 = gap`, `1 = trial`). With discrete labels, `trialTime`'s gap fallback switches to **global class-uniform sampling** (Gumbel-max trick), forcing gap timepoints to cluster together in embedding space.

---

## Discrete Label Support

All conditionals accept an optional discrete label array. When provided:

- `sample_prior` uses **class-balanced sampling** (matching CEBRA's `MixedDataLoader`).
- Trial selection is restricted to **same-class trials**.
- Gap anchor sampling switches to **global class-uniform** (Gumbel-max trick).

```python
# Discrete: 0 = gap, 1 = trial
y_disc = np.zeros(N, dtype=np.int64)
for s, e in zip(trial_starts, trial_ends):
    y_disc[s:e] = 1

model.fit(X, y_cont, y_disc, trial_starts=trial_starts, trial_ends=trial_ends)
```

**Discrete-only (no continuous labels)** is supported for `"trialTime"`:

```python
y_disc = np.zeros(ntrial, dtype=np.int64)
y_disc[ntrial // 2:] = 1

model.fit(X_ep, y_disc)   # X_ep: (ntrial, ntime, nneuro)
```

> Delta-style conditionals (`trialDelta`, `trial_delta`, `trialTime_delta`, `trialTime_trialDelta`) require continuous labels for trial similarity matching and will raise `ValueError` if none are provided.

---

## API Reference

### `TrialCEBRA`

Inherits all parameters from `cebra.CEBRA`. Key additions:

```python
TrialCEBRA(
    conditional: str,      # trial-aware or native CEBRA conditional
    time_offsets: int,     # half-width of time window; also used for Δstim lag
    delta: float,          # Gaussian kernel std for trial selection
    **cebra_kwargs,
)

# Flat format
model.fit(X, *y, trial_starts, trial_ends, adapt=False, callback=None, callback_frequency=None)

# Epoch format — trial boundaries inferred automatically
model.fit(X, *y)           # X: (ntrial, ntime, nneuro)

model.transform(X)         # → np.ndarray (N, output_dimension)
model.transform_epochs(X)  # → np.ndarray (ntrial, ntime, output_dimension)
model.distribution_        # TrialAwareDistribution instance (after fit)
```

### `TrialAwareDistribution`

The sampling distribution; can be used standalone for diagnostics.

```python
from trial_cebra import TrialAwareDistribution
import torch

dist = TrialAwareDistribution(
    continuous   = torch.randn(500, 16),
    trial_starts = torch.tensor([0, 100, 200, 300, 400]),
    trial_ends   = torch.tensor([100, 200, 300, 400, 500]),
    conditional  = "trial_delta",
    time_offset  = 10,
    delta        = 0.3,
    device       = "cpu",
    seed         = 42,
    discrete     = None,   # optional (N,) int tensor
)

ref, pos = dist.sample_joint(num_samples=64)
```

### `TrialTensorDataset`

Low-level PyTorch dataset with trial metadata, for use outside the sklearn interface.

```python
from trial_cebra import TrialTensorDataset

dataset = TrialTensorDataset(
    neural       = neural_tensor,
    continuous   = stim_tensor,
    discrete     = label_tensor,   # optional
    trial_starts = starts_tensor,
    trial_ends   = ends_tensor,
    device       = "cpu",
)
```

---

## Implementation Notes

**Post-replace distribution** — `TrialCEBRA` does not modify CEBRA's source. Instead it temporarily sets `conditional = "time_delta"` to pass CEBRA's internal validation, calls `super()._prepare_loader(...)` to obtain a standard loader, then replaces `loader.distribution` with a `TrialAwareDistribution` in-place. Both loader types call only `distribution.sample_prior` and `distribution.sample_conditional` inside `get_indices`, so the replacement is fully transparent to the training loop.

**Mixed-label routing** — When both discrete and continuous labels are provided, CEBRA always creates a `MixedDataLoader` regardless of `conditional`. `TrialCEBRA` inherits this routing and replaces the distribution afterwards; the `conditional` parameter only affects the `TrialAwareDistribution`.

---

## Project Structure

```
src/trial_cebra/
  __init__.py       public API: TrialCEBRA, TrialTensorDataset, TrialAwareDistribution, flatten_epochs
  cebra.py          TrialCEBRA sklearn estimator
  dataset.py        TrialTensorDataset (PyTorch dataset)
  distribution.py   TrialAwareDistribution (all five conditionals)
  epochs.py         flatten_epochs utility

tests/
  test_cebra.py
  test_dataset.py
  test_distribution.py
  test_epochs.py
```

---

## Contributing

**Setup** (run once after cloning):

```bash
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

**CI checks** run automatically on every push to `main`:

| Check | Command |
|---|---|
| Lint + format | `ruff check . && ruff format --check .` |
| Tests | `pytest tests/ -v` |

**Releasing a new version** — version is derived from the git tag, no files need editing:

```bash
git tag vx.x.x
git push origin vx.x.x   # triggers build + publish to PyPI
```
