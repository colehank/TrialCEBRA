# TrialCEBRA
[![PyPI](https://img.shields.io/pypi/v/TrialCEBRA?color=blue)](https://pypi.org/project/TrialCEBRA/)
[![Tests](https://github.com/colehank/TrialCEBRA/actions/workflows/tests.yml/badge.svg)](https://github.com/colehank/TrialCEBRA/actions)  
[English | [中文](README_zh.md)]

Trial-aware contrastive learning for [CEBRA](https://cebra.ai). Pass 3-D epoch-format data `(ntrial, ntime, nneuro)` — trial boundaries are respected automatically.

---

## Installation

**Step 1** — Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

**Step 2**

```bash
pip install TrialCEBRA
```

or install from github
```bash
pip install git+https://github.com/colehank/trialCEBRA.git
```

---

## Quick Start

```python
import numpy as np
from trial_cebra import TrialCEBRA

X = np.random.randn(40, 50, 64).astype(np.float32)   # (ntrial, ntime, nneuro)
y = np.random.randn(40, 16).astype(np.float32)        # (ntrial, stim_dim)

model = TrialCEBRA(
    model_architecture = "offset10-model",
    conditional        = "delta",
    time_offsets       = 5,
    delta              = 0.3,
    output_dimension   = 3,
    max_iterations     = 1000,
    batch_size         = 512,
)

model.fit(X, y)
emb = model.transform(X)        # (ntrial, ntime, 3)  — shape preserved
```

2-D flat input `(N, nneuro)` falls back to native CEBRA behavior unchanged.

---

## Conditionals

| `conditional` | Trial selection | Within-trial | y shape |
|---|---|---|---|
| `"time"` | Uniform random | ±`time_offsets` window | not required |
| `"delta"` | Gaussian similarity on y | Uniform (free) | `(ntrial, nd)` or `(ntrial, ntime, nd)` |
| `"time_delta"` | Joint argmin across trials | ±`time_offsets` window | `(ntrial, ntime, nd)` |

Pass a discrete integer label (e.g. `y_disc` of dtype `int64`) alongside a continuous label to enable class-conditional trial selection for `"delta"`.

---

## Key Parameters

```python
TrialCEBRA(
    conditional            = "delta",    # "time" | "delta" | "time_delta"
    time_offsets           = 10,         # half-width of within-trial time window
    delta                  = 0.1,        # Gaussian noise std for trial similarity
    sample_fix_trial       = False,      # True: fix trial pairing at init
    sample_exclude_intrial = True,       # True: positives always from a different trial
    sample_prior           = "balanced", # "balanced" (default) or "uniform"
    output_dimension       = 3,
    # ... all other cebra.CEBRA kwargs accepted
)
```

After `fit`, the distribution is accessible at `model.distribution_`.

---

## Transform

`transform()` preserves input dimensionality:

```python
emb = model.transform(X)          # (ntrial, ntime, 3) if X is (ntrial, ntime, nneuro)
emb = model.transform(X_flat)     # (N, 3)             if X_flat is (N, nneuro)
emb = model.transform_epochs(X)   # strict 3-D variant — raises if X.ndim != 3
```

---

## Metrics

All metric methods accept epoch-format `(ntrial, ntime, nneuro)` data directly:

```python
loss = model.infonce_loss(X, y)
gof  = model.goodness_of_fit_score(X, y)
hist = model.goodness_of_fit_history()       # training curve, no X needed

# Consistency score: accepts 3-D embedding lists
emb1 = model.transform(X1)   # (ntrial, ntime, 3)
emb2 = model.transform(X2)
scores, pairs, ids = TrialCEBRA.consistency_score(
    [emb1, emb2], between="runs"
)
# between-datasets: pass labels=(ntrial, ntime) or (ntrial*ntime,)
scores, pairs, ids = TrialCEBRA.consistency_score(
    [emb1, emb2],
    between="datasets",
    labels=[y1, y2],
    dataset_ids=["mouse1", "mouse2"],
)
```

---

## Decoders

CEBRA decoders (`KNNDecoder`, `L1LinearRegressor`) are standalone sklearn estimators that expect 2-D input. Flatten the embedding first:

```python
import cebra

emb      = model.transform(X)                  # (ntrial, ntime, 3)
emb_flat = emb.reshape(-1, emb.shape[-1])       # (ntrial*ntime, 3)
y_flat   = y.reshape(-1)                        # (ntrial*ntime,)

decoder = cebra.KNNDecoder()
decoder.fit(emb_flat, y_flat)
score = decoder.score(emb_flat, y_flat)
```

---

## Multi-session

Pass `X` as a list of epoch arrays (one per session):

```python
X = [
    np.random.randn(30, 100, 64).astype(np.float32),   # session 0
    np.random.randn(25,  80, 48).astype(np.float32),   # session 1
]
y_cont = [np.random.randn(30, 100, 16).astype(np.float32),
          np.random.randn(25,  80, 16).astype(np.float32)]
y_disc = [np.zeros((30, 100), dtype=np.int64),
          np.zeros((25,  80), dtype=np.int64)]

model = TrialCEBRA(conditional="delta", output_dimension=3, max_iterations=1000)
model.fit(X, y_disc, y_cont)
```

`"delta"` and `"time_delta"` are supported for multi-session; `"time"` raises `NotImplementedError`.

---

## Contributing

```bash
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pytest tests/ -v
```

Release: `git tag vX.X.X && git push origin vX.X.X`.
