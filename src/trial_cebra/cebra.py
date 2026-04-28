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

Three trial-aware conditionals:

  ================  ==========================================
  ``conditional``   Behaviour
  ================  ==========================================
  ``"time"``        Random trial + ±time_offsets window
  ``"delta"``       Gaussian-similarity trial + uniform in trial
  ``"time_delta"``  Velocity-similarity trial + ±time_offsets window
  ================  ==========================================

y shapes accepted for each conditional:

  * ``"time"``       — no y required
  * ``"delta"``      — y shape ``(ntrial, nd)`` **or** ``(ntrial, ntime, nd)``.
                       The 3-D form is required for class-conditional trial
                       selection when ``y_discrete`` is also provided with
                       per-timepoint class labels (see below).
  * ``"time_delta"`` — y shape ``(ntrial, ntime, nd)`` (timepoint-level labels)

``sample_fix_trial`` (default ``False``):
  When ``True``, the trial→trial mapping is pre-computed once at init.
  When ``False``, target trial is re-sampled at every training step.
  Has no effect for ``"time"``.

``sample_exclude_intrial`` (default ``True``):
  When ``True``, positive samples are always drawn from a different trial.
  When ``False``, any trial (including the anchor's own) may be selected.

Discrete-first class-conditional trial selection (``"delta"`` only):
  When a discrete label is supplied via ``fit(..., y_disc, ...)``, the delta
  trial-selection step follows CEBRA's ``ConditionalIndex`` principle —
  discrete first, continuous within.  Behaviour depends on the combination
  of ``y_discrete`` and ``y_continuous`` shapes (auto-detected at init):

  * **Mode A** — per-trial discrete (constant within each trial):
    candidates are restricted to trials sharing the anchor's class.
  * **Mode B** — per-timepoint discrete + 3-D ``y``: a class-conditional
    trial embedding ``trial_emb_per_class[c][trial] = mean(y[trial, t]
    for t where class(trial, t) == c)`` is used as the query basis.
  * **Mode C** — per-timepoint discrete + 2-D ``y``: the 2-D ``y`` is
    automatically broadcast to ``(ntrial, ntime, nd)`` and Mode B aggregation
    is applied. No warning is emitted.

  A tiny Gumbel perturbation is added before ``argmin`` to break ties
  stochastically (needed when all class-c trial embeddings are identical,
  e.g., pre-stim gray-screen labels).
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

    Extends :py:class:`cebra.CEBRA` with three trial-aware conditionals.
    All constructor parameters are inherited from :py:class:`cebra.CEBRA`;
    the ``conditional`` parameter accepts the three new values listed below
    in addition to all native CEBRA values.

    Trial-aware conditionals:

    * ``"time"`` — pick a target trial uniformly at random (≠ own),
      draw a positive within ±``time_offsets`` of the reference's relative
      position in the target trial.  No y required.
    * ``"delta"`` — select a target trial via Gaussian-kernel similarity
      on trial embeddings (``y`` shape ``(ntrial, nd)`` or
      ``(ntrial, ntime, nd)``); draw a positive uniformly from the selected
      trial.  When ``y_discrete`` is also provided the selection becomes
      class-conditional (see module docstring).
    * ``"time_delta"`` — select a target trial via empirical stimulus-
      velocity similarity on trial-onset embeddings (``y`` shape
      ``(ntrial, ntime, nd)``); draw a positive within ±``time_offsets``
      of the reference's relative position.

    Args:
        sample_fix_trial: If ``True``, pre-compute the trial→trial mapping once
            at init (deterministic pairing per trial).  If ``False``
            (default), re-sample the target trial at every training step
            for greater diversity.  Ignored for ``"time"``.
        sample_exclude_intrial: If ``True`` (default), positive samples are
            always from a different trial than the anchor.  If ``False``,
            any trial (including the anchor's own) may be selected.
        sample_prior: ``"balanced"`` (default) or ``"uniform"``.  When a
            discrete label is supplied, ``"balanced"`` draws anchors uniformly
            across classes (oversampling minority classes by ``1 / class_freq``),
            while ``"uniform"`` draws anchors uniformly over timepoints so the
            anchor class distribution matches the empirical frequencies.  Has
            no effect when no discrete label is provided.

    Example::

        >>> import numpy as np
        >>> from trial_cebra import TrialCEBRA
        >>> ntrial, ntime, nneuro, nd = 10, 50, 30, 8
        >>> X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
        >>> y = np.random.randn(ntrial, nd).astype(np.float32)
        >>> model = TrialCEBRA(
        ...     model_architecture="offset1-model",
        ...     conditional="delta",
        ...     time_offsets=5,
        ...     delta=0.1,
        ...     sample_fix_trial=False,
        ...     sample_exclude_intrial=True,
        ...     max_iterations=10,
        ...     batch_size=32,
        ...     output_dimension=3,
        ... )
        >>> model.fit(X, y)
        TrialCEBRA(...)
        >>> emb = model.transform(X.reshape(ntrial * ntime, nneuro))
    """

    def __init__(
        self,
        *args,
        sample_fix_trial: bool = False,
        sample_exclude_intrial: bool = True,
        sample_prior: str = "balanced",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_fix_trial = sample_fix_trial
        self.sample_exclude_intrial = sample_exclude_intrial
        self.sample_prior = sample_prior

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
        """Fit the estimator on epoch-format or flat data.

        If ``X`` is 3-D ``(ntrial, ntime, nneuro)``, trial structure is
        inferred from the shape and ``trial_starts`` / ``trial_ends`` are
        ignored.  Otherwise flat data with explicit trial metadata is used.

        Args:
            X: Neural data, shape ``(ntrial, ntime, nneuro)`` or ``(N, nneuro)``.
            y: Label arrays.  For trial-aware conditionals:
               * ``"delta"``      — ``(ntrial, nd)`` or ``(ntrial, ntime, nd)``
                 (3-D preferred when a per-timepoint discrete label is also
                 supplied, to enable class-conditional trial selection)
               * ``"time_delta"`` — ``(ntrial, ntime, nd)``
               * ``"time"``       — not required
            adapt: Adapt an existing model to new data.
            callback: Optional callback.
            callback_frequency: Callback interval.
            trial_starts: Ignored when X is 3-D.
            trial_ends: Ignored when X is 3-D.

        Returns:
            ``self``
        """
        # Multi-session detection: list of 3-D arrays (heterogeneous shapes allowed)
        if isinstance(X, list):
            return self._fit_multisession(
                X,
                *y,
                adapt=adapt,
                callback=callback,
                callback_frequency=callback_frequency,
            )
        X = np.asarray(X)
        if X.ndim == 3:
            from trial_cebra.epochs import flatten_epochs

            # Store epoch-format y before flattening (needed for distribution)
            self._y_epoch = tuple(np.asarray(yi) for yi in y) if y else ()
            self._ntrial, self._ntime = X.shape[0], X.shape[1]
            X, y, trial_starts, trial_ends = flatten_epochs(X, *y)

        self._trial_starts = trial_starts
        self._trial_ends = trial_ends
        return super().fit(
            X,
            *y,
            adapt=adapt,
            callback=callback,
            callback_frequency=callback_frequency,
        )

    def fit_epochs(
        self,
        X: npt.NDArray,
        *y: npt.NDArray,
        adapt: bool = False,
        callback: Callable = None,
        callback_frequency: int = None,
    ) -> "TrialCEBRA":
        """Fit the estimator on epoch-format data.

        A convenience wrapper around :py:meth:`fit` for epoch-format data.
        Trial boundaries are inferred automatically from the array shape.

        Args:
            X: Neural data, shape ``(ntrial, ntime, nneuro)``.
            y: Label arrays (see :py:meth:`fit` for shape requirements).
            adapt: Passed through to :py:meth:`fit`.
            callback: Passed through to :py:meth:`fit`.
            callback_frequency: Passed through to :py:meth:`fit`.

        Returns:
            ``self``

        Example::

            >>> import numpy as np
            >>> from trial_cebra import TrialCEBRA
            >>> X_ep = np.random.randn(20, 50, 64).astype(np.float32)
            >>> y_ep = np.random.randn(20, 16).astype(np.float32)
            >>> model = TrialCEBRA(
            ...     model_architecture="offset1-model",
            ...     conditional="delta",
            ...     time_offsets=5,
            ...     delta=0.3,
            ...     max_iterations=10,
            ...     batch_size=32,
            ...     output_dimension=3,
            ... )
            >>> model.fit_epochs(X_ep, y_ep)
            TrialCEBRA(...)
        """
        from trial_cebra.epochs import flatten_epochs

        X = np.asarray(X)
        # Store epoch-format metadata before flattening
        self._y_epoch = tuple(np.asarray(yi) for yi in y) if y else ()
        self._ntrial, self._ntime = X.shape[0], X.shape[1]
        X_flat, y_flat, trial_starts, trial_ends = flatten_epochs(X, *y)
        self._trial_starts = trial_starts
        self._trial_ends = trial_ends
        return super().fit(
            X_flat,
            *y_flat,
            adapt=adapt,
            callback=callback,
            callback_frequency=callback_frequency,
        )

    def transform_epochs(self, X: npt.NDArray) -> npt.NDArray:
        """Transform epoch-format neural data.

        Args:
            X: Neural data, shape ``(ntrial, ntime, nneuro)``.

        Returns:
            Embeddings, shape ``(ntrial, ntime, output_dimension)``.
        """
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (ntrial, ntime, nneuro), got shape {X.shape}")
        ntrial, ntime, _ = X.shape
        emb = self.transform(X.reshape(ntrial * ntime, -1))
        return emb.reshape(ntrial, ntime, -1)

    # ------------------------------------------------------------------
    # Multi-session fit
    # ------------------------------------------------------------------

    def _fit_multisession(
        self, X, *y, adapt: bool = False, callback=None, callback_frequency=None
    ) -> "TrialCEBRA":
        """Fit on multi-session epoch-format data.

        Triggered when ``X`` is a list (auto-detected in :py:meth:`fit`).

        Args:
            X: list of 3-D ``(ntrial_s, ntime_s, nneuro_s)`` arrays — one per
               session, may be heterogeneous.
            *y: variadic; each is a list of per-session label arrays (length
               equal to ``len(X)``).  Per-session labels follow the same
               broadcasting rules as :py:func:`flatten_epochs`.

        Returns: ``self``

        Raises:
            ValueError: invalid X / y structure
            NotImplementedError: ``conditional='time'`` (matches CEBRA native)
        """
        from trial_cebra.epochs import flatten_epochs_multisession

        if self.conditional not in TRIAL_CONDITIONALS:
            raise ValueError(
                f"multisession fit requires a trial-aware conditional "
                f"({sorted(TRIAL_CONDITIONALS)}); got {self.conditional!r}. "
                "For native CEBRA multisession, use cebra.CEBRA directly."
            )
        if self.conditional == "time":
            raise NotImplementedError(
                "conditional='time' is not supported in multisession (matches "
                "CEBRA native behaviour). Use 'delta' or 'time_delta'."
            )

        # Per-session flatten (validates X/y shapes, returns a list of dicts)
        sessions = flatten_epochs_multisession(X, *y)

        # Separate discrete and continuous y per session.
        # Trial-aware multisession requires continuous y (delta / time_delta);
        # discrete is optional but if present must be in every session.
        num_sessions = len(sessions)
        disc_list = [None] * num_sessions
        cont_list = [None] * num_sessions
        for s, sess in enumerate(sessions):
            for yf in sess["y_flat"]:
                yf = np.asarray(yf)
                if yf.dtype.kind in ("i", "u"):
                    if disc_list[s] is not None:
                        raise ValueError(f"session {s}: multiple discrete y; only one supported")
                    disc_list[s] = yf
                else:
                    if cont_list[s] is not None:
                        raise ValueError(f"session {s}: multiple continuous y; only one supported")
                    cont_list[s] = yf

        if any(c is None for c in cont_list):
            raise ValueError(
                "multisession trial-aware fit requires continuous y for every "
                "session (delta / time_delta)."
            )
        # Discrete consistency: present in all or none
        disc_present = [d is not None for d in disc_list]
        if any(disc_present) and not all(disc_present):
            raise ValueError(
                f"y_discrete presence must be consistent across sessions: got {disc_present}."
            )

        # Stash for _prepare_data / _prepare_loader to consume.
        self._ms_sessions = sessions
        self._ms_disc_list = disc_list
        self._ms_cont_list = cont_list
        self._ms_num_sessions = num_sessions
        # Also stash original epoch-format y for _resolve_y_epoch per session.
        # _resolve_y_epoch gets called per-session inside _prepare_loader_multisession.
        self._ms_y_epoch = [
            tuple(np.asarray(yi_list[s]) for yi_list in y) for s in range(num_sessions)
        ]

        # Hand off to CEBRA native fit with continuous-only y.  CEBRA's
        # _prepare_data will build a DatasetCollection of SklearnDataset; our
        # override below will attach trial metadata + discrete index per session.
        X_flat_list = [sess["X_flat"] for sess in sessions]
        return super().fit(
            X_flat_list,
            cont_list,
            adapt=adapt,
            callback=callback,
            callback_frequency=callback_frequency,
        )

    # ------------------------------------------------------------------
    # Overridden CEBRA internals
    # ------------------------------------------------------------------

    def _prepare_data(self, X, y):
        """Create dataset and attach trial boundary metadata."""
        dataset, is_multisession = super()._prepare_data(X, y)

        if is_multisession and getattr(self, "_ms_sessions", None) is not None:
            # Multi-session trial-aware path: attach per-session metadata
            sessions = self._ms_sessions
            for s, sub_ds in enumerate(dataset.iter_sessions()):
                sess = sessions[s]
                sub_ds.trial_starts = (
                    torch.from_numpy(np.asarray(sess["trial_starts"])).long().to(sub_ds.device)
                )
                sub_ds.trial_ends = (
                    torch.from_numpy(np.asarray(sess["trial_ends"])).long().to(sub_ds.device)
                )
                # Inject discrete index when present (used by TrialAwareDistribution
                # via class-balanced prior + same-class constraint).
                yd = self._ms_disc_list[s]
                if yd is not None:
                    yd_t = torch.from_numpy(np.asarray(yd)).long().to(sub_ds.device)
                    object.__setattr__(sub_ds, "_discrete_index", yd_t)
                    sub_ds._tensors.add("_discrete_index")
            return dataset, is_multisession

        # Single-session path
        ts = getattr(self, "_trial_starts", None)
        te = getattr(self, "_trial_ends", None)

        if ts is not None and te is not None:
            dataset.trial_starts = torch.from_numpy(np.asarray(ts)).long().to(dataset.device)
            dataset.trial_ends = torch.from_numpy(np.asarray(te)).long().to(dataset.device)

        return dataset, is_multisession

    def _prepare_loader(self, dataset, max_iterations, is_multisession):
        """Create data loader; replace distribution when trial metadata is present."""
        # Multi-session trial-aware path
        if (
            is_multisession
            and self.conditional in TRIAL_CONDITIONALS
            and getattr(self, "_ms_sessions", None) is not None
        ):
            return self._prepare_loader_multisession(dataset, max_iterations)

        has_trial_meta = (
            self.conditional in TRIAL_CONDITIONALS
            and hasattr(dataset, "trial_starts")
            and hasattr(dataset, "trial_ends")
        )
        if not has_trial_meta:
            return super()._prepare_loader(dataset, max_iterations, is_multisession)

        orig_conditional = self.conditional

        if not (hasattr(dataset, "trial_starts") and hasattr(dataset, "trial_ends")):
            raise ValueError(
                f"conditional='{orig_conditional}' requires trial metadata. "
                "Pass epoch-format X (3-D) to fit()."
            )

        # Temporarily use a native conditional to pass CEBRA's validation.
        # Use "time_delta" only when continuous y is present (CEBRA requires it for time_delta).
        # Fall back to "time" when only discrete y is provided.
        has_continuous = any(
            np.asarray(yi).dtype.kind == "f" for yi in getattr(self, "_y_epoch", ())
        )
        placeholder = "time_delta" if has_continuous else "time"
        self.conditional = placeholder
        # Hide discrete_index so CEBRA creates ContinuousDataLoader (not MixedDataLoader).
        # TrialAwareDistribution replaces the distribution entirely, so class-balanced
        # prior from MixedDataLoader is not needed and would break trial-aware sampling.
        # Use object.__setattr__ + manual _tensors management to avoid CEBRA's internal
        # "Remove {property}" print that fires when setting a tracked tensor to None.
        saved_discrete = dataset.__dict__.get("_discrete_index")
        had_discrete_tracked = saved_discrete is not None
        if had_discrete_tracked:
            dataset._tensors.discard("_discrete_index")
            object.__setattr__(dataset, "_discrete_index", None)
        try:
            loader, solver_name = super()._prepare_loader(dataset, max_iterations, is_multisession)
        finally:
            self.conditional = orig_conditional
            if had_discrete_tracked:
                object.__setattr__(dataset, "_discrete_index", saved_discrete)
                dataset._tensors.add("_discrete_index")

        # Resolve ntrial and ntime
        ntrial = getattr(self, "_ntrial", None)
        ntime = getattr(self, "_ntime", None)
        if ntrial is None or ntime is None:
            ts = dataset.trial_starts
            ntrial = len(ts)
            ntime = int((dataset.trial_ends[0] - ts[0]).item())

        # Resolve epoch-format y for the distribution
        y_epoch = self._resolve_y_epoch(orig_conditional, ntrial, ntime)

        # Unpack time_offset
        time_offset = self.time_offsets
        if isinstance(time_offset, Iterable):
            (time_offset,) = time_offset

        dist = TrialAwareDistribution(
            ntrial=ntrial,
            ntime=ntime,
            conditional=orig_conditional,
            y=y_epoch,
            y_discrete=saved_discrete,
            sample_fix_trial=self.sample_fix_trial,
            sample_exclude_intrial=self.sample_exclude_intrial,
            sample_prior=self.sample_prior,
            time_offsets=time_offset,
            delta=self.delta,
            device=str(loader.device),
        )
        loader.distribution = dist
        self.distribution_ = dist
        return loader, solver_name

    def _prepare_loader_multisession(self, dataset, max_iterations):
        """Build multi-session trial-aware loader + sampler.

        Assumes ``_prepare_data`` has already attached ``trial_starts`` /
        ``trial_ends`` (and optionally ``_discrete_index``) to each of the
        session datasets inside the :py:class:`cebra.data.DatasetCollection`.

        Returns: ``(loader, solver_name='multi-session')`` where ``loader``
        is a :py:class:`TrialAwareMultisessionLoader` whose ``sampler`` has
        been set to a :py:class:`TrialAwareMultisessionSampler`.
        """
        from trial_cebra.multisession import (
            TrialAwareMultisessionLoader,
            TrialAwareMultisessionSampler,
        )

        # Build one TrialAwareDistribution per session.
        per_session_dists = []
        sub_datasets = list(dataset.iter_sessions())
        for s, sub_ds in enumerate(sub_datasets):
            ts = sub_ds.trial_starts
            te = sub_ds.trial_ends
            ntrial_s = len(ts)
            # Assume equal trial length within a session (single-session contract).
            ntime_s = int((te[0] - ts[0]).item())

            # Continuous y: stored as sub_ds.continuous_index, flat
            # (ntrial_s * ntime_s, nd). Reshape to 3-D for delta/time_delta.
            y_cont_flat = sub_ds.continuous_index
            y_cont_3d = y_cont_flat.reshape(ntrial_s, ntime_s, -1)

            # Discrete y (if any), already attached to sub_ds by _prepare_data
            y_disc = sub_ds.__dict__.get("_discrete_index")

            # Unpack time_offset scalar
            time_offset = self.time_offsets
            if isinstance(time_offset, Iterable):
                (time_offset,) = time_offset

            dist_s = TrialAwareDistribution(
                ntrial=ntrial_s,
                ntime=ntime_s,
                conditional=self.conditional,
                y=y_cont_3d,
                y_discrete=y_disc,
                sample_fix_trial=self.sample_fix_trial,
                # Cross-session strictness is enforced at the sampler layer.
                # Per-session exclude_intrial is disabled to avoid double masking
                # (the query post-shuffle is foreign to this session anyway).
                sample_exclude_intrial=False,
                sample_prior=self.sample_prior,
                time_offsets=time_offset,
                delta=self.delta,
                device=str(sub_ds.device),
            )
            per_session_dists.append(dist_s)

        # Hide _discrete_index on each session to force CEBRA's multi-session
        # code paths to use ContinuousMultiSessionDataLoader pathways (we
        # subclass it). We restore afterwards so the dataset remains intact.
        saved_disc = []
        for sub_ds in sub_datasets:
            d = sub_ds.__dict__.get("_discrete_index")
            saved_disc.append(d)
            if d is not None:
                sub_ds._tensors.discard("_discrete_index")
                object.__setattr__(sub_ds, "_discrete_index", None)

        try:
            time_offset = self.time_offsets
            if isinstance(time_offset, Iterable):
                (time_offset,) = time_offset
            loader = TrialAwareMultisessionLoader(
                dataset=dataset,
                num_steps=max_iterations,
                batch_size=self.batch_size,
                time_offset=time_offset,
            )
        finally:
            for sub_ds, d in zip(sub_datasets, saved_disc):
                if d is not None:
                    object.__setattr__(sub_ds, "_discrete_index", d)
                    sub_ds._tensors.add("_discrete_index")

        sampler = TrialAwareMultisessionSampler(
            per_session_dists,
            conditional=self.conditional,
            seed=None,
        )
        loader.sampler = sampler
        self.distribution_ = sampler  # expose for tests / inspection
        return loader, "multi-session"

    def _resolve_y_epoch(self, conditional: str, ntrial: int, ntime: int) -> Optional[torch.Tensor]:
        """Extract the epoch-format y tensor appropriate for the conditional.

        Matches by *both* dtype (float only; integer arrays are discrete labels
        and are skipped) and shape (must satisfy the conditional's requirement):

        * ``"delta"``      → 3-D float ``(ntrial, ntime, nd)`` preferred
                             (enables class-conditional trial embeddings when a
                             discrete label is also provided); 2-D float
                             ``(ntrial, nd)`` is also accepted as a fallback.
        * ``"time_delta"`` → 3-D float ``(ntrial, ntime, nd)``

        Label order in :py:meth:`fit` does not matter.
        """
        if conditional == "time":
            return None

        y_epoch = getattr(self, "_y_epoch", ())
        if not y_epoch:
            return None

        # For delta, prefer 3-D y over 2-D when both are provided.
        if conditional == "delta":
            three_d = None
            two_d = None
            for yi in y_epoch:
                yi = np.asarray(yi)
                if yi.dtype.kind != "f":
                    continue
                if yi.ndim == 3 and yi.shape[:2] == (ntrial, ntime):
                    three_d = yi
                elif yi.ndim == 2 and yi.shape[0] == ntrial:
                    two_d = yi
            chosen = three_d if three_d is not None else two_d
            return None if chosen is None else torch.from_numpy(chosen.copy())

        for yi in y_epoch:
            yi = np.asarray(yi)
            if yi.dtype.kind != "f":  # skip discrete
                continue
            if conditional == "time_delta" and yi.ndim == 3 and yi.shape[:2] == (ntrial, ntime):
                return torch.from_numpy(yi.copy())

        return None
