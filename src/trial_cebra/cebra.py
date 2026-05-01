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

Transform shape contract:
  :py:meth:`transform` preserves the number of input dimensions:

  * 2-D input ``(N, nneuro)``                → 2-D output ``(N, output_dim)``
  * 3-D input ``(ntrial, ntime, nneuro)``    → 3-D output ``(ntrial, ntime, output_dim)``

  :py:meth:`transform_epochs` is a strict 3-D-only variant that raises if
  ``X.ndim != 3``.

Metrics wrappers:
  All CEBRA metric functions that take raw neural data accept epoch-format
  ``(ntrial, ntime, nneuro)`` input via the following instance / static
  methods:

  * :py:meth:`infonce_loss` — wraps :py:func:`cebra.sklearn.metrics.infonce_loss`
  * :py:meth:`goodness_of_fit_score` — wraps
    :py:func:`cebra.sklearn.metrics.goodness_of_fit_score`
  * :py:meth:`goodness_of_fit_history` — passthrough (no X/y, reads training log)
  * :py:meth:`consistency_score` — static method; flattens 3-D embeddings and
    2-D labels before calling :py:func:`cebra.sklearn.metrics.consistency_score`

Decoder workflow:
  CEBRA decoders (:py:class:`cebra.KNNDecoder`,
  :py:class:`cebra.L1LinearRegressor`) are standalone sklearn estimators that
  expect 2-D embeddings.  When using epoch data, flatten the output of
  :py:meth:`transform` before fitting a decoder::

      emb      = model.transform(X)                   # (ntrial, ntime, dim)
      emb_flat = emb.reshape(-1, emb.shape[-1])        # (ntrial*ntime, dim)
      y_flat   = y.reshape(-1)                         # (ntrial*ntime,)
      decoder  = cebra.KNNDecoder().fit(emb_flat, y_flat)
"""

from collections.abc import Iterable
from typing import Callable, List, Literal, Optional, Tuple, Union

import cebra
import cebra.data
import cebra.solver
import numpy as np
import numpy.typing as npt
import torch

from trial_cebra.distribution import TRIAL_CONDITIONALS, TrialAwareDistribution


class TrialCEBRA(cebra.CEBRA):
    """Trial-aware CEBRA estimator.

    Extends :py:class:`cebra.CEBRA` with three trial-aware conditionals and
    epoch-format (3-D) data support throughout the fit / transform / metrics
    pipeline.

    **Trial-aware conditionals** (pass via ``conditional=``):

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

    **Shape contract for** :py:meth:`transform`:

    Input dimensionality is preserved in the output:

    * ``(N, nneuro)``             → ``(N, output_dim)``
    * ``(ntrial, ntime, nneuro)`` → ``(ntrial, ntime, output_dim)``

    **Metrics** (all accept epoch-format X directly):

    * :py:meth:`infonce_loss`
    * :py:meth:`goodness_of_fit_score`
    * :py:meth:`goodness_of_fit_history`
    * :py:meth:`consistency_score` (static, accepts 3-D embedding lists)

    **Decoder workflow** (external sklearn decoders expect 2-D embeddings)::

        emb      = model.transform(X)                   # (ntrial, ntime, dim)
        emb_flat = emb.reshape(-1, emb.shape[-1])        # (ntrial*ntime, dim)
        y_flat   = y.reshape(-1)
        decoder  = cebra.KNNDecoder().fit(emb_flat, y_flat)

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

    All other constructor parameters are inherited from :py:class:`cebra.CEBRA`
    and forwarded unchanged.

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
        ...     max_iterations=10,
        ...     batch_size=32,
        ...     output_dimension=3,
        ... )
        >>> model.fit(X, y)
        TrialCEBRA(...)
        >>> emb = model.transform(X)        # (ntrial, ntime, 3)
        >>> gof = model.goodness_of_fit_score(X, y)
    """

    def __init__(
        self,
        model_architecture: str = "offset1-model",
        device: str = "cuda_if_available",
        criterion: str = "infonce",
        distance: str = "cosine",
        conditional: str = None,
        temperature: float = 1.0,
        temperature_mode: Literal["constant", "auto"] = "constant",
        min_temperature: Optional[float] = 0.1,
        time_offsets: int = 1,
        delta: float = None,
        max_iterations: int = 10000,
        max_adapt_iterations: int = 500,
        batch_size: int = None,
        learning_rate: float = 0.0003,
        optimizer: str = "adam",
        output_dimension: int = 8,
        verbose: bool = False,
        num_hidden_units: int = 32,
        pad_before_transform: bool = True,
        hybrid: bool = False,
        optimizer_kwargs: Tuple[Tuple[str, object], ...] = (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-08),
            ("weight_decay", 0),
            ("amsgrad", False),
        ),
        masking_kwargs: Optional[
            Tuple[Tuple[str, Union[float, List[float], Tuple[float, ...]]], ...]
        ] = None,
        sample_fix_trial: bool = False,
        sample_exclude_intrial: bool = True,
        sample_prior: str = "balanced",
    ):
        super().__init__(
            model_architecture=model_architecture,
            device=device,
            criterion=criterion,
            distance=distance,
            conditional=conditional,
            temperature=temperature,
            temperature_mode=temperature_mode,
            min_temperature=min_temperature,
            time_offsets=time_offsets,
            delta=delta,
            max_iterations=max_iterations,
            max_adapt_iterations=max_adapt_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            output_dimension=output_dimension,
            verbose=verbose,
            num_hidden_units=num_hidden_units,
            pad_before_transform=pad_before_transform,
            hybrid=hybrid,
            optimizer_kwargs=optimizer_kwargs,
            masking_kwargs=masking_kwargs,
        )
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

    def transform(self, X, batch_size=None, session_id=None):
        """Transform neural data into embeddings.

        Extends :py:meth:`cebra.CEBRA.transform` to accept 3-D epoch-format
        input ``(ntrial, ntime, nneuro)`` in addition to the native 2-D
        ``(N, nneuro)`` format.  Output dimensionality matches input:
        3-D in → ``(ntrial, ntime, output_dimension)`` out;
        2-D in → ``(N, output_dimension)`` out.
        """
        if not torch.is_tensor(X):
            X = np.asarray(X)
        is_3d = X.ndim == 3
        if is_3d:
            ntrial, ntime = X.shape[0], X.shape[1]
            X = X.reshape(ntrial * ntime, -1)
        emb = super().transform(X, batch_size=batch_size, session_id=session_id)
        if is_3d:
            emb = emb.reshape(ntrial, ntime, -1)
        return emb

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
    # Metrics — epoch-aware wrappers
    # ------------------------------------------------------------------

    def _set_epoch_state(self, X: np.ndarray, y: tuple) -> dict:
        """Temporarily overwrite instance trial metadata with values from new epoch data.

        Returns a *saved* dict that must be passed to :py:meth:`_restore_epoch_state`
        after the call that needed the updated state.

        Only call this when ``X.ndim == 3``; callers are responsible for the check.
        """
        from trial_cebra.epochs import flatten_epochs

        ntrial, ntime = X.shape[0], X.shape[1]
        X_flat, y_flat, trial_starts, trial_ends = flatten_epochs(X, *y)

        saved = {
            "_ntrial": getattr(self, "_ntrial", None),
            "_ntime": getattr(self, "_ntime", None),
            "_trial_starts": getattr(self, "_trial_starts", None),
            "_trial_ends": getattr(self, "_trial_ends", None),
            "_y_epoch": getattr(self, "_y_epoch", ()),
        }
        self._ntrial = ntrial
        self._ntime = ntime
        self._trial_starts = trial_starts
        self._trial_ends = trial_ends
        self._y_epoch = tuple(np.asarray(yi) for yi in y)
        return X_flat, y_flat, saved

    def _restore_epoch_state(self, saved: Optional[dict]) -> None:
        if saved is None:
            return
        for key, val in saved.items():
            setattr(self, key, val)

    def infonce_loss(
        self,
        X,
        *y,
        session_id: Optional[int] = None,
        num_batches: int = 500,
        correct_by_batchsize: bool = False,
    ) -> float:
        """Compute InfoNCE loss on epoch or flat data.

        Wraps :py:func:`cebra.sklearn.metrics.infonce_loss` with support for
        3-D epoch-format input ``(ntrial, ntime, nneuro)``.  Labels ``y``
        follow the same broadcasting rules as :py:meth:`fit`.

        Args:
            X: Neural data, shape ``(ntrial, ntime, nneuro)`` or ``(N, nneuro)``.
            y: Label arrays matching ``X``.
            session_id: Session ID for multisession models.
            num_batches: Number of batches to average over.
            correct_by_batchsize: Subtract ``log(batch_size)`` from the loss.

        Returns:
            Average InfoNCE loss (float).
        """
        import cebra.integrations.sklearn.metrics as _m

        X = np.asarray(X)
        saved = None
        if X.ndim == 3:
            X, y, saved = self._set_epoch_state(X, y)
        try:
            return _m.infonce_loss(
                self,
                X,
                *y,
                session_id=session_id,
                num_batches=num_batches,
                correct_by_batchsize=correct_by_batchsize,
            )
        finally:
            self._restore_epoch_state(saved)

    def goodness_of_fit_score(
        self,
        X,
        *y,
        session_id: Optional[int] = None,
        num_batches: int = 500,
    ) -> float:
        """Compute goodness-of-fit score on epoch or flat data.

        Wraps :py:func:`cebra.sklearn.metrics.goodness_of_fit_score` with
        support for 3-D epoch-format input ``(ntrial, ntime, nneuro)``.

        Args:
            X: Neural data, shape ``(ntrial, ntime, nneuro)`` or ``(N, nneuro)``.
            y: Label arrays matching ``X``.
            session_id: Session ID for multisession models.
            num_batches: Number of batches to average over.

        Returns:
            Goodness-of-fit score in bits (float).
        """
        import cebra.integrations.sklearn.metrics as _m

        X = np.asarray(X)
        saved = None
        if X.ndim == 3:
            X, y, saved = self._set_epoch_state(X, y)
        try:
            return _m.goodness_of_fit_score(
                self,
                X,
                *y,
                session_id=session_id,
                num_batches=num_batches,
            )
        finally:
            self._restore_epoch_state(saved)

    def goodness_of_fit_history(self) -> np.ndarray:
        """Return the goodness-of-fit history from training.

        Passthrough to :py:func:`cebra.sklearn.metrics.goodness_of_fit_history`.

        Returns:
            Array of goodness-of-fit values (bits) over training iterations.
        """
        import cebra.integrations.sklearn.metrics as _m

        return _m.goodness_of_fit_history(self)

    @staticmethod
    def consistency_score(
        embeddings,
        between=None,
        labels=None,
        dataset_ids=None,
        num_discretization_bins: int = 100,
    ):
        """Compute consistency score, with support for 3-D epoch embeddings.

        Wraps :py:func:`cebra.sklearn.metrics.consistency_score`.  Any 3-D
        embedding ``(ntrial, ntime, output_dim)`` is automatically reshaped to
        ``(ntrial*ntime, output_dim)`` before comparison.  Any 2-D label array
        ``(ntrial, ntime)`` is flattened to ``(ntrial*ntime,)``; 1-D labels
        are passed through unchanged.

        Args:
            embeddings: List of embedding arrays, each ``(N, dim)`` or
                ``(ntrial, ntime, dim)``.
            between: ``"runs"`` or ``"datasets"`` — type of comparison.
            labels: List of label arrays for ``between="datasets"`` alignment.
                Each may be ``(N,)`` or ``(ntrial, ntime)``.
            dataset_ids: List of dataset IDs (one per embedding).
            num_discretization_bins: Bins for label digitization.

        Returns:
            ``(scores, pairs, ids)`` as returned by
            :py:func:`cebra.sklearn.metrics.consistency_score`.
        """
        import cebra.integrations.sklearn.metrics as _m

        flat_embs = []
        for emb in embeddings:
            emb = np.asarray(emb) if not isinstance(emb, np.ndarray) else emb
            if emb.ndim == 3:
                emb = emb.reshape(-1, emb.shape[-1])
            flat_embs.append(emb)

        flat_labels = None
        if labels is not None:
            flat_labels = []
            for lbl in labels:
                lbl = np.asarray(lbl) if not isinstance(lbl, np.ndarray) else lbl
                if lbl.ndim == 2:
                    lbl = lbl.reshape(-1)
                flat_labels.append(lbl)

        return _m.consistency_score(
            flat_embs,
            between=between,
            labels=flat_labels,
            dataset_ids=dataset_ids,
            num_discretization_bins=num_discretization_bins,
        )

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

        # Per-session flatten (validates X/y shapes, returns a list of dicts)
        sessions = flatten_epochs_multisession(X, *y)

        # Separate discrete and continuous y per session.
        # Rules:
        #   delta / time_delta: continuous y required; discrete optional.
        #   time:               continuous y NOT required; discrete required.
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

        disc_only = all(c is None for c in cont_list)
        if self.conditional in ("delta", "time_delta"):
            if any(c is None for c in cont_list):
                raise ValueError(
                    "multisession trial-aware fit requires continuous y for every "
                    f"session when conditional='{self.conditional}'."
                )
        elif self.conditional == "time":
            if disc_only and any(d is None for d in disc_list):
                raise ValueError(
                    "conditional='time' in multisession requires y_discrete for every "
                    "session (needed as the cross-session alignment signal)."
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

        X_flat_list = [sess["X_flat"] for sess in sessions]
        # For disc-only + "time": pass disc_list as y so CEBRA builds DatasetCollection
        # with discrete_index.  Our _prepare_loader override intercepts before the native
        # loader (DiscreteMultiSessionDataLoader) is actually used.
        y_for_super = disc_list if disc_only else cont_list
        return super().fit(
            X_flat_list,
            y_for_super,
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
            delta=self.delta if self.delta is not None else 0.1,
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
            # May be None for disc-only + conditional="time".
            y_cont_flat = sub_ds.continuous_index
            y_cont_3d = (
                y_cont_flat.reshape(ntrial_s, ntime_s, -1) if y_cont_flat is not None else None
            )

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
                delta=self.delta if self.delta is not None else 0.1,
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
