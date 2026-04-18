# TrialCEBRA

**Trial-aware contrastive learning for CEBRA** — a wrapper library that adds five trial-structured sampling conditionals to [CEBRA](https://cebra.ai) without modifying its source code.

Designed for neuroscience experiments where neural recordings are organized as repeated trials (stimuli, conditions, epochs), this library enables hierarchical positive-pair sampling: first select a *target trial* based on stimulus similarity or at random, then draw a positive timepoint within that trial.

**[中文文档 README\_zh.md](README_zh.md)**

---

## Background

CEBRA's native conditionals (`time`, `delta`, `time_delta`) operate over a flat sequence of timepoints. For trial-structured data they have two limitations:

1. **Temporal boundary artifacts** — a 1-D CNN convolves across trial boundaries, mixing pre- and post-stimulus activity.
2. **Flat sampling ignores trial structure** — `delta` finds the nearest-neighbor timepoint in stimulus space; when all timepoints within a trial share the same stimulus embedding, this collapses to intra-trial sampling with no cross-trial signal.

`trial_cebra` solves both by lifting the positive-pair selection from the *timepoint* level to the *trial* level.

---

## Installation

```bash
pip install trial_cebra
```

Requires `cebra >= 0.4` and `torch`.

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
    conditional        = "trial_delta",   # re-sampled Gaussian trial selection
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
# Epoch data: (ntrial, ntime, nneuro)
X_ep = np.random.randn(40, 50, 64).astype(np.float32)

# Labels can be per-trial or per-timepoint:
y_pertrial  = np.random.randn(40, 16).astype(np.float32)   # (ntrial, stim_dim)
y_pertimepoint = np.random.randn(40, 50, 16).astype(np.float32)  # (ntrial, ntime, stim_dim)

model.fit(X_ep, y_pertrial)          # auto-detects 3-D, no trial_starts/ends needed
emb = model.transform_epochs(X_ep)   # (ntrial, ntime, output_dimension)
```

Label broadcasting rules when `X` is 3-D:

| Label shape | Interpretation | Output shape |
|---|---|---|
| `(ntrial,)` | per-trial discrete | `(ntrial*ntime,)` |
| `(ntrial, d)` where `d ≠ ntime` | per-trial continuous | `(ntrial*ntime, d)` |
| `(ntrial, ntime)` | per-timepoint | `(ntrial*ntime,)` |
| `(ntrial, ntime, d)` | per-timepoint | `(ntrial*ntime, d)` |

---

## Conditionals

Five trial-aware conditionals are available, organized along three orthogonal axes:

| Axis | Options |
|---|---|
| **Trial selection** | Random (uniform) · Gaussian delta-style · Gaussian time_delta-style |
| **Time constraint** | `Time` (±`time_offset` relative position within target trial) · Free (uniform within trial) |
| **Locking** | Locked (fixed mapping pre-computed at `__init__`) · Re-sampled (independent per training step) |

### Conditional Table

| `conditional` | Trial selection | Time constraint | Locking | Gap strategy |
|---|---|---|---|---|
| `"trialTime"` | Random | ±`time_offset` | — | global ±`time_offset` (or class-uniform with discrete) |
| `"trialDelta"` | delta-style | Free (uniform) | **Locked** | delta-style at timepoint level |
| `"trial_delta"` | delta-style | Free (uniform) | Re-sampled | delta-style at timepoint level |
| `"trialTime_delta"` | delta-style | ±`time_offset` | Re-sampled | delta-style at timepoint level |
| `"trialTime_trialDelta"` | time_delta-style | ±`time_offset` | **Locked** | time_delta-style at timepoint level |

Native CEBRA conditionals (`"time"`, `"delta"`, `"time_delta"`, etc.) pass through unchanged.

### Naming Convention

```
trialDelta          → capital D, no underscore → Locked, delta-style Gaussian
trial_delta         → underscore + lowercase d → Re-sampled, delta-style Gaussian
trialTime           → Random trial + time constraint
trialTime_delta     → Time constraint + Re-sampled delta-style
trialTime_trialDelta → Time constraint + Locked delta-style (time_delta mechanism)
```

The `_delta` suffix (underscore + lowercase) always signals re-sampled; `trialDelta` (capital D) always signals locked — mirroring CEBRA's own convention where `delta` is re-sampled.

---

## How Sampling Works

### Trial selection: delta-style

Used by `trialDelta`, `trial_delta`, and `trialTime_delta`. Mirrors CEBRA's [`DeltaNormalDistribution`](https://cebra.ai) at the trial level:

```
query = trial_mean[anchor_trial] + N(0, δ²I)
target_trial = argmin_j  dist(query, trial_mean[j])
```

Each trial is represented by the **mean** of its timepoints' continuous auxiliary variable. The Gaussian noise with std `δ` controls how far the query drifts — small `δ` picks the most similar trial, large `δ` explores broadly. Because noise is freshly sampled every step, the same anchor may be paired with different trials across training iterations.

### Trial selection: time_delta-style

Used only by `trialTime_trialDelta`. Mirrors CEBRA's [`TimedeltaDistribution`](https://cebra.ai) at the trial level:

```
Δstim[k] = continuous[k] - continuous[k − time_offset]   (pre-computed)
query    = trial_mean[anchor_trial] + Δstim[random_k]
target_trial = argmin_j  dist(query, trial_mean[j])
```

This uses empirical stimulus-velocity vectors as perturbations, data-driven rather than isotropic.

### Locked vs Re-sampled

| | Locked (`trialDelta`, `trialTime_trialDelta`) | Re-sampled (`trial_delta`, `trialTime_delta`) |
|---|---|---|
| Target trial | Pre-computed once at `__init__`, fixed for all steps | Independently drawn at every training step |
| Gradient signal | Consistent — same trial pair compared repeatedly | Diverse — anchor sees different similar trials each step |
| Generalization | May learn pair-specific features | Learns features valid across all similar trials |
| Best for | Few trials, need stable training | Many trials, rich stimulus content |

---

## Visualizing Sampling Behavior

The figures below are produced by `example/viz_trial_sampling.py` and `example/draft.py` on real MEG data with ImageNet stimuli. Each panel shows **R** (reference anchor), **+** (positive samples), and **−** (negative samples). Border color encodes in-trial time position (colorbar on the right, black = gap timepoints).

### Trial sampling: R / + / −

![Trial sampling](resources/fig_trial_sampling.png)

Each row shows a different conditional. Notice how positive samples differ in both stimulus content and time position depending on the mode:

- **`trialTime`** (top-left) — positives come from a uniformly random other trial, centred near the anchor's relative time position. The stimulus grid is diverse with no bias toward similar images.
- **`trialDelta`** (top-center) — positives cluster around a single *locked* target trial chosen by stimulus similarity. All positive frames show the same image (the bull terrier), confirming the fixed `ref_trial → target_trial` mapping.
- **`trial_delta`** (top-right) — the target trial is re-sampled every step. Positive frames spread across several similar stimuli while maintaining content coherence, achieving higher diversity than `trialDelta`.
- **`trialTime_delta`** (bottom-left) — same trial-selection diversity as `trial_delta`, but the time window is additionally constrained to ±`time_offset` ms of the anchor's relative position, visible as the tighter in-trial time spread in the colorbar.
- **`trialTime_trialDelta`** (bottom-center) — locked target trial (like `trialDelta`) combined with the ±`time_offset` window. Positives concentrate on a single stimulus image at a specific post-stimulus latency.

### Sampling timeline

![Sampling timeline](resources/fig_sampling.png)

This view places each sampled frame on a timeline that spans the full trial duration. The green highlighted band marks the ±`time_offset` window around the anchor's relative position. Key observations:

- `trialTime` and `trialTime_delta` / `trialTime_trialDelta` positives fall inside the green band, confirming the time constraint is respected.
- `trialDelta` and `trial_delta` positives scatter freely across the full trial length — no time constraint.
- The anchor (R) is always at the same absolute time within its trial; the target trial may differ but the relative position is aligned.

---

## Learned Embeddings

All eight conditionals (3 native CEBRA + 5 trial-aware) were trained on the same MEG dataset for 10 000 iterations. Points are colored by **in-trial time** (black = pre-stimulus / gap; yellow-green = late post-stimulus).

### 3D embeddings colored by time

![3D embeddings](resources/fig_3d_embeddings.png)

**Top row — native CEBRA:**

- **`time`** — timepoints are uniformly distributed on a sphere. The model learns no temporal structure because positive pairs are drawn from a flat time window with no stimulus information.
- **`time_delta`** — similar spherical layout but slightly more organized. The stimulus-velocity perturbation creates weak temporal gradients.
- **`delta`** — stimulus content dominates. The pre-stimulus gap frames (black) collapse to a single dark patch at the bottom, while post-stimulus trial frames scatter widely. No temporal ordering is preserved.

**Bottom row — trial-aware TrialCEBRA:**

- **`trialTime_delta`** — the clearest temporal ring: points rotate around the sphere ordered by in-trial time, with gap frames (black) separated into a distinct cluster. Both temporal structure and gap/trial discrimination are simultaneously learned.
- **`trialTime`** — similar temporal ring but with a wider time window, producing a smoother gradient at the cost of slightly looser trial–gap separation.
- **`trialDelta`** — gap frames separate cleanly, but the locked mapping and free time constraint produce a more scattered post-stimulus cloud.
- **`trial_delta`** — re-sampled trial diversity creates a more uniform embedding of trial frames while keeping gap frames distinct.
- **`trialTime_trialDelta`** — locked trial + time window yields a tight temporal ring similar to `trialTime_delta`, with the most compact per-latency clustering.

### Training loss

![Loss curves](resources/fig_loss.png)

All conditionals converge smoothly. Trial-aware conditionals generally start at higher loss (richer contrastive task) and converge to a similar level as their native counterparts, indicating the model successfully learns the hierarchical structure.

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

**Recommended practice**: pass a discrete label array marking trial vs. gap (e.g. `0 = gap`, `1 = trial`). When discrete labels are present, `trialTime`'s gap fallback switches from a local ±window to **global class-uniform sampling** (Gumbel-max trick), forcing all gap timepoints to cluster together in embedding space rather than preserving local temporal chain structure within the gap period.

---

## Discrete Label Support

All conditionals accept an optional discrete label array. When provided:

- `sample_prior` uses **class-balanced sampling** (matching CEBRA's `MixedDataLoader`).
- Trial selection is restricted to **same-class trials** only.
- Gap anchor sampling switches from local ±window to **global class-uniform** (Gumbel-max trick).

```python
# Discrete: 0 = gap, 1 = trial
y_disc = np.zeros(N, dtype=np.int64)
for s, e in zip(trial_starts, trial_ends):
    y_disc[s:e] = 1

model.fit(X, y_cont, y_disc, trial_starts=trial_starts, trial_ends=trial_ends)
```

### Discrete-only (no continuous labels)

Passing only a discrete label (no float array) is supported for `"trialTime"`. The model selects a random same-class trial and draws a positive within ±`time_offsets` of the anchor's relative position.

```python
y_disc = np.zeros(ntrial, dtype=np.int64)   # per-trial label (epoch format)
y_disc[ntrial // 2:] = 1

model.fit(X_ep, y_disc)   # X_ep: (ntrial, ntime, nneuro)
```

Delta-style conditionals (`trialDelta`, `trial_delta`, `trialTime_delta`, `trialTime_trialDelta`) require continuous labels for trial similarity matching and will raise a `ValueError` if none are provided.

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
model.fit(
    X,                              # (N, input_dim) neural data
    *y,                             # continuous and/or discrete labels
    trial_starts: array-like,       # (T,) start indices (inclusive)
    trial_ends:   array-like,       # (T,) end indices (exclusive)
    adapt: bool = False,
    callback: Callable = None,
    callback_frequency: int = None,
) -> TrialCEBRA

# Epoch format — trial boundaries inferred automatically
model.fit(
    X,                              # (ntrial, ntime, nneuro) epoch array
    *y,                             # per-trial or per-timepoint labels
) -> TrialCEBRA

model.transform(X)         -> np.ndarray   # (N, output_dimension)
model.transform_epochs(X)  -> np.ndarray   # (ntrial, ntime, output_dimension)
model.distribution_                        # TrialAwareDistribution instance (after fit)
```

### `TrialAwareDistribution`

The sampling distribution; can be used standalone for diagnostics.

```python
from trial_cebra import TrialAwareDistribution
import torch

dist = TrialAwareDistribution(
    continuous   = torch.randn(500, 16),    # (N, d) stimulus embeddings
    trial_starts = torch.tensor([0, 100, 200, 300, 400]),
    trial_ends   = torch.tensor([100, 200, 300, 400, 500]),
    conditional  = "trial_delta",
    time_offset  = 10,
    delta        = 0.3,
    device       = "cpu",
    seed         = 42,
    discrete     = None,                    # optional (N,) int tensor
)

ref, pos = dist.sample_joint(num_samples=64)
```

### `TrialTensorDataset`

Low-level PyTorch dataset with trial metadata, for use outside the sklearn interface.

```python
from trial_cebra import TrialTensorDataset

dataset = TrialTensorDataset(
    neural       = neural_tensor,     # (N, D)
    continuous   = stim_tensor,       # (N, d)
    discrete     = label_tensor,      # (N,) optional
    trial_starts = starts_tensor,     # (T,)
    trial_ends   = ends_tensor,       # (T,)
    device       = "cpu",
)
```

---

## Implementation Notes

### Post-replace distribution

`TrialCEBRA` does not modify CEBRA's source. Instead:

1. Temporarily sets `self.conditional = "time_delta"` to pass CEBRA's internal validation.
2. Calls `super()._prepare_loader(...)` to obtain a standard `ContinuousDataLoader` or `MixedDataLoader`.
3. Replaces `loader.distribution` with a `TrialAwareDistribution` in-place.

Both loader types call only `distribution.sample_prior` and `distribution.sample_conditional` inside `get_indices`, so the replacement is fully transparent to the training loop.

### CEBRA routing with mixed labels

When both discrete and continuous labels are provided, CEBRA's internal routing always creates a `MixedDataLoader` regardless of the `conditional` parameter. `TrialCEBRA` inherits this routing and replaces the distribution afterwards. The user-specified `conditional` only affects the `TrialAwareDistribution`, not the loader type.

---

## Project Structure

```
src/trial_cebra/
  __init__.py          # public API: TrialCEBRA, TrialTensorDataset, TrialAwareDistribution, flatten_epochs
  cebra.py             # TrialCEBRA sklearn estimator
  dataset.py           # TrialTensorDataset (PyTorch API)
  distribution.py      # TrialAwareDistribution (all five conditionals)
  epochs.py            # flatten_epochs utility for epoch-format data

tests/
  conftest.py
  test_cebra.py
  test_dataset.py
  test_distribution.py
  test_epochs.py

resources/             # figures generated by example scripts
  fig_trial_sampling.png
  fig_sampling.png
  fig_3d_embeddings.png
  fig_loss.png
```

---

## Contributing

**Setup** (run once after cloning):

```bash
uv sync --dev
```

**CI checks** (run automatically on push to main):

| Check | Command |
|---|---|
| Lint + format | `ruff check . && ruff format --check .` |
| Tests | `pytest tests/ -v` |

**Releasing a new version** — version is derived from the git tag, no files need editing:

```bash
git tag v0.1.x
git push origin v0.1.x   # triggers build + publish to PyPI
```
