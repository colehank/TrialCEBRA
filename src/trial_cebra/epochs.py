"""Utilities for epoch-format neural data (ntrial × ntime × nneuro)."""

from typing import Tuple

import numpy as np
import numpy.typing as npt


def flatten_epochs(
    X: npt.NDArray,
    *y: npt.NDArray,
) -> Tuple[npt.NDArray, Tuple[npt.NDArray, ...], npt.NDArray, npt.NDArray]:
    """Convert epoch-format data to flat format with trial boundaries.

    Args:
        X: Neural data, shape ``(ntrial, ntime, nneuro)``.
        y: Label arrays.  Each array is broadcast or reshaped according to
           its dimensionality:

           * ``(ntrial,)``          — per-trial discrete; tiled to
             ``(ntrial * ntime,)``
           * ``(ntrial, d)`` where ``d ≠ ntime`` — per-trial continuous;
             tiled to ``(ntrial * ntime, d)``
           * ``(ntrial, ntime)``    — per-timepoint; reshaped to
             ``(ntrial * ntime,)``
           * ``(ntrial, ntime, d)`` — per-timepoint; reshaped to
             ``(ntrial * ntime, d)``

           .. note::
               When a 2-D label has shape ``(ntrial, ntime)`` it is treated
               as **per-timepoint** (already expanded), *not* as a per-trial
               ``ntime``-dimensional vector.  If you have a per-trial
               continuous label whose feature dimension happens to equal
               ``ntime``, expand it manually to ``(ntrial, 1, ntime)`` first.

    Returns:
        X_flat:       Neural data, shape ``(ntrial * ntime, nneuro)``.
        y_flat:       Tuple of flattened label arrays.
        trial_starts: Start index of each trial, shape ``(ntrial,)``.
        trial_ends:   End index (exclusive) of each trial, shape ``(ntrial,)``.

    Raises:
        ValueError: If ``X`` is not 3-D, or a label array has incompatible
            shape.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (ntrial, ntime, nneuro), got shape {X.shape}")
    ntrial, ntime, nneuro = X.shape

    X_flat = X.reshape(ntrial * ntime, nneuro)
    trial_starts = np.arange(ntrial, dtype=np.int64) * ntime
    trial_ends = trial_starts + ntime

    y_flat = []
    for i, yi in enumerate(y):
        yi = np.asarray(yi)
        if yi.ndim == 1:
            # (ntrial,) → per-trial discrete
            if yi.shape[0] != ntrial:
                raise ValueError(
                    f"y[{i}] has shape {yi.shape}; expected ({ntrial},) for per-trial labels"
                )
            y_flat.append(np.repeat(yi, ntime))
        elif yi.ndim == 2:
            if yi.shape == (ntrial, ntime):
                # per-timepoint, flatten
                y_flat.append(yi.reshape(ntrial * ntime))
            elif yi.shape[0] == ntrial:
                # (ntrial, d) with d != ntime → per-trial continuous
                y_flat.append(np.repeat(yi, ntime, axis=0))
            else:
                raise ValueError(
                    f"y[{i}] has shape {yi.shape}; expected ({ntrial}, ...) or ({ntrial}, {ntime})"
                )
        elif yi.ndim == 3:
            if yi.shape[:2] != (ntrial, ntime):
                raise ValueError(f"y[{i}] has shape {yi.shape}; expected ({ntrial}, {ntime}, d)")
            y_flat.append(yi.reshape(ntrial * ntime, yi.shape[2]))
        else:
            raise ValueError(f"y[{i}] must be 1-D, 2-D, or 3-D, got {yi.ndim}-D array")

    return X_flat, tuple(y_flat), trial_starts, trial_ends
