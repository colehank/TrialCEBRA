"""TrialCEBRA: Trial-aware sklearn wrapper for CEBRA.

Subclasses the official ``cebra.CEBRA`` estimator to add trial-aware
contrastive learning without modifying CEBRA's source code.

The key technique is "post-replace distribution": the official CEBRA
pipeline creates a standard ``ContinuousDataLoader`` (or
``MixedDataLoader`` when a discrete index is present), then its
``distribution`` attribute is replaced in-place with a
:py:class:`~trial_cebra.distribution.TrialAwareDistribution`.
Both loader types call only ``distribution.sample_prior`` and
``distribution.sample_conditional`` inside ``get_indices``, so the
replacement is fully transparent to the training loop.

Five trial-aware conditionals are supported (three orthogonal axes:
trial selection × time constraint × locking):

  ===========================  ==========================================
  ``conditional``              Behaviour
  ===========================  ==========================================
  ``"trialTime"``              Random trial + ±time_offset window
  ``"trialDelta"``             Locked Gaussian trial + uniform in trial
  ``"trial_delta"``            Re-sampled Gaussian trial + uniform in trial
  ``"trialTime_delta"``        Re-sampled Gaussian trial + ±offset pool
  ``"trialTime_trialDelta"``   Locked Gaussian trial + ±time_offset window
  ===========================  ==========================================

All other CEBRA conditionals (``"time"``, ``"delta"``, ``"time_delta"``,
etc.) pass through unchanged to the base class.
"""

from collections.abc import Iterable
from typing import Callable, Optional

import cebra
import cebra.data
import cebra.solver
import numpy as np
import numpy.typing as npt
import torch

from trial_cebra.distribution import TRIAL_CONDITIONALS, TrialAwareDistribution


class TrialCEBRA(cebra.CEBRA):
    """Trial-aware CEBRA estimator.

    Extends :py:class:`cebra.CEBRA` with five trial-aware conditionals.
    All constructor parameters are inherited from :py:class:`cebra.CEBRA`;
    the ``conditional`` parameter accepts the five new values listed below
    in addition to all native CEBRA values.

    Trial-aware conditionals:

    * ``"trialTime"`` — pick a target trial uniformly at random (≠ own),
      draw a positive within ±``time_offsets`` of the reference's relative
      position in the target trial.
    * ``"trialDelta"`` — pre-lock one target trial per reference trial via
      Gaussian-kernel similarity (Locked); draw a positive uniformly from
      the locked target trial.
    * ``"trial_delta"`` — independently re-sample a target trial at each
      step via Gaussian-kernel similarity (Re-sampled); draw a positive
      uniformly from the selected target trial.  Provides more diverse
      positive pairs than ``"trialDelta"``.
    * ``"trialTime_delta"`` — at each step, select a target trial by
      trial-level Gaussian kernel, then draw a positive uniformly within
      ±``time_offsets`` of the reference's relative position (Re-sampled).
    * ``"trialTime_trialDelta"`` — pre-lock one target trial (as in
      ``"trialDelta"``), then draw a positive within ±``time_offsets`` of
      the reference's relative position in the locked target trial.

    Gap (inter-trial) timepoints are valid anchors.  ``"trialTime"`` gap
    anchors use a global ±``time_offsets`` window; all other conditionals
    fall back to content-based Gaussian-kernel sampling.

    When both a continuous **and** a discrete label are passed, positive
    samples are additionally restricted to the same discrete class.

    Example::

        >>> import numpy as np
        >>> from trial_cebra import TrialCEBRA
        >>> X = np.random.randn(500, 30).astype(np.float32)
        >>> y = np.random.randn(500, 10).astype(np.float32)
        >>> trial_starts = np.array([i * 50 for i in range(10)])
        >>> trial_ends   = trial_starts + 50
        >>> model = TrialCEBRA(
        ...     model_architecture="offset1-model",
        ...     conditional="trialTime_trialDelta",
        ...     time_offsets=5,
        ...     delta=0.1,
        ...     max_iterations=10,
        ...     batch_size=32,
        ...     output_dimension=3,
        ... )
        >>> model.fit(X, y, trial_starts=trial_starts, trial_ends=trial_ends)
        TrialCEBRA(...)
        >>> emb = model.transform(X)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        *y,
        adapt: bool = False,
        callback: Callable = None,
        callback_frequency: int = None,
        trial_starts: Optional[npt.NDArray] = None,
        trial_ends: Optional[npt.NDArray] = None,
    ) -> "TrialCEBRA":
        """Fit the estimator with optional trial metadata.

        Args:
            X: Neural data matrix, shape ``(N, input_dim)``.
            y: Continuous and/or discrete label arrays.
            adapt: If ``True``, adapt an existing model to new data.
            callback: Optional callback function.
            callback_frequency: Callback interval.
            trial_starts: Start indices of each trial, shape ``(T,)``.
                Required when ``conditional`` is one of the trial-aware
                modes.
            trial_ends: End indices (exclusive) of each trial, shape
                ``(T,)``.  Required together with ``trial_starts``.

        Returns:
            ``self``, to allow method chaining.
        """
        self._trial_starts = trial_starts
        self._trial_ends = trial_ends
        return super().fit(
            X,
            *y,
            adapt=adapt,
            callback=callback,
            callback_frequency=callback_frequency,
        )

    def _adapt_fit(
        self, X, *y, callback=None, callback_frequency=None, trial_starts=None, trial_ends=None
    ):
        self._trial_starts = trial_starts
        self._trial_ends = trial_ends
        return super()._adapt_fit(
            X,
            *y,
            callback=callback,
            callback_frequency=callback_frequency,
        )

    # ------------------------------------------------------------------
    # Overridden CEBRA internals
    # ------------------------------------------------------------------

    def _prepare_data(self, X, y):
        """Create ``SklearnDataset`` and attach trial boundary metadata."""
        dataset, is_multisession = super()._prepare_data(X, y)

        ts = getattr(self, "_trial_starts", None)
        te = getattr(self, "_trial_ends", None)

        if ts is not None and te is not None:
            dataset.trial_starts = torch.from_numpy(np.asarray(ts)).long().to(dataset.device)
            dataset.trial_ends = torch.from_numpy(np.asarray(te)).long().to(dataset.device)

        return dataset, is_multisession

    def _prepare_loader(self, dataset, max_iterations, is_multisession):
        """Create data loader; replace distribution for trial-aware modes.

        For trial-aware conditionals:

        1. Temporarily set ``self.conditional = "time_delta"`` so that
           CEBRA's ``_init_loader`` accepts the call and returns either a
           ``ContinuousDataLoader`` or ``MixedDataLoader``.
        2. Restore the original conditional name.
        3. Replace ``loader.distribution`` with a
           :py:class:`~trial_cebra.distribution.TrialAwareDistribution`.

        For native CEBRA conditionals, the call is forwarded unchanged.
        """
        if self.conditional not in TRIAL_CONDITIONALS:
            return super()._prepare_loader(dataset, max_iterations, is_multisession)

        orig_conditional = self.conditional

        # Validate trial metadata present on dataset
        if not (hasattr(dataset, "trial_starts") and hasattr(dataset, "trial_ends")):
            raise ValueError(
                f"conditional='{orig_conditional}' requires trial_starts and "
                "trial_ends.  Pass them as keyword arguments to fit()."
            )

        # Temporarily use a native conditional to pass CEBRA's validation
        self.conditional = "time_delta"
        try:
            loader, solver_name = super()._prepare_loader(dataset, max_iterations, is_multisession)
        finally:
            self.conditional = orig_conditional

        # Unpack time_offset (CEBRA stores it as a tuple internally)
        time_offset = self.time_offsets
        if isinstance(time_offset, Iterable):
            (time_offset,) = time_offset

        # Replace distribution with our trial-aware implementation
        dist = TrialAwareDistribution(
            continuous=dataset.continuous_index,
            trial_starts=dataset.trial_starts,
            trial_ends=dataset.trial_ends,
            conditional=orig_conditional,
            time_offset=time_offset,
            delta=self.delta,
            device=str(loader.device),
            discrete=getattr(dataset, "discrete_index", None),
        )
        loader.distribution = dist
        # Store reference for post-fit inspection
        self.distribution_ = dist
        return loader, solver_name
