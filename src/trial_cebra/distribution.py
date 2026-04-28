"""Trial-aware distributions for epoch-format contrastive sampling.

Three conditionals mirroring CEBRA's originals, lifted to trial level:

  ===============  ===============  ================  ==============================
  ``conditional``  Trial selection  Within-trial      y required
  ===============  ===============  ================  ==============================
  ``"time"``       Random           ±time_offsets     None
  ``"delta"``      delta-style      Uniform (free)    (ntrial, nd) or
                                                      (ntrial, ntime, nd)
  ``"time_delta"`` joint argmin     ±time_offsets     (ntrial, ntime, nd)
  ===============  ===============  ================  ==============================

``sample_fix_trial`` (bool, default False):
  True  — pre-compute a fixed trial→trial mapping at init (one draw per trial).
  False — re-sample target trial independently at each training step.
  Ignored for ``"time"`` (always random).

``sample_exclude_intrial`` (bool, default True):
  True  — positive samples are always drawn from a different trial (cross-trial).
  False — positive samples may come from any trial, including the anchor's own.
          For ``"delta"`` without ``y_discrete``, the trial selection switches
          from deterministic ``argmin`` to stochastic Gumbel-max sampling
          (softmax(-dists/T) with T=delta).  Pure ``argmin`` would otherwise
          collapse to "always self trial" in high dimensions because Gaussian
          noise of std ``delta`` is mostly orthogonal to inter-trial offsets
          and cannot escape self-similarity.  Stochastic selection biases
          toward similar trials but does not pin to self.  Note that even
          with this fix the resulting embedding may have limited structure
          when no ``y_discrete`` is supplied — InfoNCE benefits substantially
          from the same-class constraint that ``y_discrete`` provides.

``y_discrete`` (optional LongTensor, shape ``(ntrial*ntime,)``):
  Per-timepoint discrete class labels in flat epoch format.  When provided:

  * **Prior** (controlled by ``sample_prior``):
      - ``"balanced"`` (default): uniform over classes, then uniform within
        the selected class.  Equalises class representation in anchors but
        oversamples minority classes by ``1 / class_freq``.
      - ``"uniform"``: uniform over all timepoints; class frequency in
        anchors matches the empirical distribution.  Use this for severely
        imbalanced labels where ``"balanced"`` would distort the prior.
  * **Conditional**: same-class constraint — positives must carry the same
    discrete label as the anchor.  Falls back to unconstrained sampling for
    the rare case where no same-class candidate exists in the target pool.

``sample_prior`` only affects the anchor distribution; the same-class
conditional constraint is applied independently.

Discrete-first principle (only ``"delta"`` path):
  Following CEBRA's ``ConditionalIndex`` design, when ``y_discrete`` is
  supplied the trial-selection step uses **class-conditional** trial
  embeddings whenever possible:

    * Mode A — discrete is per-trial (constant within each trial):
      candidates restricted to trials sharing the anchor's class.
    * Mode B — discrete is per-timepoint AND ``y`` is 3-D:
      ``trial_emb_per_class[c][trial] = mean(y[trial, t]
      for t where class(trial,t) == c)``; the anchor uses its own class's
      basis to query trials.
    * Mode C — discrete is per-timepoint but ``y`` is only 2-D: the 2-D y is
      automatically broadcast to 3-D ``(ntrial, ntime, nd)`` by repeating each
      trial's embedding across all timepoints, then Mode B aggregation is applied.
      Per-class mean of a constant-within-trial tensor equals the original value,
      so trial selection is identical to a class-agnostic query but with the
      class-conditional code path active.  No warning is emitted.

  In all modes a tiny Gumbel perturbation is added to ``dists`` before
  ``argmin`` to break ties stochastically (e.g., when all trials share the
  same class-c embedding, as happens for pre-stim gray-screen labels).

y semantics:
  ``"delta"``      — 2-D ``(ntrial, nd)`` (per-trial) or 3-D
                     ``(ntrial, ntime, nd)`` (per-timepoint).  When
                     ``y_discrete`` is per-timepoint and ``y`` is 2-D, ``y``
                     is silently broadcast to 3-D for Mode B aggregation.
  ``"time_delta"`` — 3-D ``(ntrial, ntime, nd)``;  ``trial_emb = y[:, 0, :]``
                     (trial-onset embedding, no mean aggregation).

``"time_delta"`` sampling (fix_trial=False):
  For anchor (trial_i, rel_i), the candidate pool is all cross-trial timepoints
  within ±time_offsets of rel_i::

      {(trial_j, t) : trial_j ≠ trial_i, |t − rel_i| ≤ time_offsets}

  The positive is chosen as the argmin in y-space after adding Gaussian noise to
  the anchor's y query.  On static stimuli (constant y within trial) this
  degrades gracefully to delta-style trial selection + time-window sampling.

``"time_delta"`` sampling (fix_trial=True):
  Target trial is locked at init (same Gaussian-similarity query as ``"delta"``
  on trial-onset embeddings).  At each step the within-trial timepoint is the
  argmin of y-distance inside the ±time_offsets window of the locked trial.
"""

from __future__ import annotations

import warnings
from typing import Optional

import cebra.distributions.base as abc_
import torch

TRIAL_CONDITIONALS = frozenset({"time", "delta", "time_delta"})
PRIOR_MODES = frozenset({"uniform", "balanced"})

# Class-conditional mode tags used internally when y_discrete is provided
# with conditional == "delta".  See _detect_disc_mode() for selection logic.
_DISC_MODE_NONE = "none"  # no y_discrete
_DISC_MODE_PER_TRIAL = "per_trial"  # each trial has a single class
_DISC_MODE_PER_TP_3D = "per_tp_3d"  # per-timepoint class + 3-D y (full class-cond)
_DISC_MODE_PER_TP_2D = "per_tp_2d_fallback"  # per-timepoint class + 2-D y (cannot class-cond)

# Chunk size for the joint argmin loop (controls peak VRAM).
# Peak tensor: (chunk, ntrial*W, nd) — e.g. 16×4200×768×4 ≈ 206 MB.
_ARGMIN_CHUNK = 4


class TrialAwareDistribution(abc_.JointDistribution, abc_.HasGenerator):
    """Trial-aware hierarchical sampling distribution for epoch-format data.

    Args:
        ntrial:               Number of trials.
        ntime:                Timepoints per trial (equal-length trials assumed).
        conditional:          Sampling mode — ``"time"``, ``"delta"``, or ``"time_delta"``.
        y:                    Label tensor.  Shape ``(ntrial, nd)`` for ``"delta"``;
                              ``(ntrial, ntime, nd)`` for ``"time_delta"``; ``None`` for
                              ``"time"``.
        y_discrete:           Optional per-timepoint discrete labels, flat shape
                              ``(ntrial*ntime,)``.  Enables class-balanced prior and
                              same-class conditional constraint.
        sample_prior:         ``"balanced"`` (default) for class-balanced anchor
                              sampling when ``y_discrete`` is provided;
                              ``"uniform"`` for frequency-weighted sampling (uniform
                              over all timepoints).  Ignored when ``y_discrete`` is
                              ``None``.
        sample_fix_trial:     If ``True``, pre-compute a fixed trial-to-trial mapping at
                              init.  Ignored for ``"time"``.
        sample_exclude_intrial: If ``True`` (default), positive samples are always from a
                              different trial than the anchor.  If ``False``, any trial
                              (including the anchor's own) may be selected.
        time_offsets:         Half-width of the within-trial time window (used by ``"time"``
                              and ``"time_delta"``).
        delta:                Gaussian noise std for delta-style trial selection.
        device:               Compute device.
        seed:                 RNG seed.
    """

    def __init__(
        self,
        ntrial: int,
        ntime: int,
        conditional: str,
        y: Optional[torch.Tensor] = None,
        y_discrete: Optional[torch.Tensor] = None,
        sample_fix_trial: bool = False,
        sample_exclude_intrial: bool = True,
        sample_prior: str = "balanced",
        time_offsets: int = 10,
        delta: float = 0.1,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        abc_.HasGenerator.__init__(self, device=device, seed=seed)

        if conditional not in TRIAL_CONDITIONALS:
            raise ValueError(
                f"Unknown conditional: {conditional!r}. "
                f"Must be one of {sorted(TRIAL_CONDITIONALS)}."
            )

        if sample_prior not in PRIOR_MODES:
            raise ValueError(
                f"Unknown sample_prior: {sample_prior!r}. Must be one of {sorted(PRIOR_MODES)}."
            )

        if ntrial < 2:
            raise ValueError("At least 2 trials are required for cross-trial sampling.")

        self.ntrial = ntrial
        self.ntime = ntime
        self.conditional = conditional
        self.sample_fix_trial = sample_fix_trial
        self.sample_exclude_intrial = sample_exclude_intrial
        self.sample_prior_mode = sample_prior
        self.time_offsets = time_offsets
        self.delta = delta

        # Validate y shape against conditional
        if conditional == "delta":
            # Accept 2-D (per-trial) or 3-D (per-timepoint) y.
            # The 3-D form enables class-conditional trial embeddings when
            # y_discrete is also provided (see _detect_disc_mode).
            if y is None or y.ndim not in (2, 3) or y.shape[0] != ntrial:
                raise ValueError(
                    f"conditional='delta' requires y of shape (ntrial={ntrial}, nd) "
                    f"or (ntrial={ntrial}, ntime={ntime}, nd); "
                    f"got {None if y is None else tuple(y.shape)}."
                )
            if y.ndim == 3 and y.shape[1] != ntime:
                raise ValueError(
                    f"3-D y for conditional='delta' must have shape "
                    f"(ntrial={ntrial}, ntime={ntime}, nd); got {tuple(y.shape)}."
                )
            y = y.to(device)
            if y.ndim == 3:
                self._y_delta_3d = y
                self.trial_emb = y.mean(dim=1)  # (ntrial, nd) — class-agnostic mean
            else:
                # 2-D y: repeat along time axis so _build_class_conditional_state
                # can compute per-class trial embeddings (Mode B) instead of
                # falling back to class-agnostic Mode C.  Per-class mean of a
                # constant tensor equals the tensor itself, so trial selection
                # is identical to Mode C but class-conditional.
                self._y_delta_3d = y.unsqueeze(1).expand(ntrial, ntime, -1)
                self.trial_emb = y  # (ntrial, nd)

        elif conditional == "time_delta":
            if y is None or y.ndim != 3 or y.shape[:2] != (ntrial, ntime):
                raise ValueError(
                    f"conditional='time_delta' requires y of shape "
                    f"(ntrial={ntrial}, ntime={ntime}, nd); "
                    f"got {None if y is None else tuple(y.shape)}. "
                    "y must be 3-D timepoint-level labels."
                )
            y = y.to(device)
            self.trial_emb = y[:, 0, :]  # (ntrial, nd) — onset, no mean
            self._y_flat = y.reshape(ntrial * ntime, -1)  # (ntrial*ntime, nd)
            self._y_norm2 = (self._y_flat**2).sum(-1)  # (ntrial*ntime,) — precomputed for fast dist

        # Discrete labels: class-balanced prior + same-class conditional constraint
        if y_discrete is not None:
            yd = y_discrete.reshape(-1).to(device).long()
            if yd.numel() != ntrial * ntime:
                raise ValueError(
                    f"y_discrete must have ntrial*ntime={ntrial * ntime} elements; "
                    f"got {yd.numel()}."
                )
            self._y_discrete = yd  # (ntrial*ntime,)
            classes = yd.unique(sorted=True)
            self._n_classes = classes.numel()
            # Per-class indices (concatenated) + offsets and sizes for vectorized sampling
            idx_list, offsets, sizes = [], [], []
            offset = 0
            for c in classes:
                idx = (yd == c).nonzero(as_tuple=True)[0]
                idx_list.append(idx)
                offsets.append(offset)
                sizes.append(idx.numel())
                offset += idx.numel()
            self._class_indices_flat = torch.cat(idx_list)  # (ntrial*ntime,) by class
            self._class_offsets = torch.tensor(
                offsets, dtype=torch.long, device=device
            )  # (n_classes,)
            self._class_sizes = torch.tensor(sizes, dtype=torch.long, device=device)  # (n_classes,)
            self._classes_sorted = classes  # (n_classes,) for value→idx

            # Class-conditional structures for delta path.
            # Mode A (per_trial):     each trial constant class    -> filter trials by class
            # Mode B (per_tp_3d):     class varies + 3-D y         -> trial_emb_per_class
            # Mode C (per_tp_2d):     class varies + 2-D y         -> auto-broadcast to 3-D, Mode B
            self._disc_mode = self._build_class_conditional_state(yd, conditional, device)
        else:
            self._disc_mode = _DISC_MODE_NONE

        # _y_delta_3d is only used inside _build_class_conditional_state for
        # Mode B aggregation. Drop it (and its HasDevice registration) now.
        if hasattr(self, "_y_delta_3d"):
            self._tensors.discard("_y_delta_3d")
            object.__setattr__(self, "_y_delta_3d", None)

        # Pre-compute locked trial mapping if requested (not applicable for "time")
        if sample_fix_trial and conditional != "time":
            all_trials = torch.arange(ntrial, device=device)
            if conditional == "delta" and self._disc_mode in (
                _DISC_MODE_PER_TRIAL,
                _DISC_MODE_PER_TP_3D,
            ):
                # Per-class locked targets: shape (n_classes, ntrial).
                # For each (class, trial) pair, what's the class-conditional target?
                locked = torch.zeros(self._n_classes, ntrial, dtype=torch.long, device=device)
                for ci in range(self._n_classes):
                    c_val = self._classes_sorted[ci]
                    anchor_cls = torch.full_like(all_trials, int(c_val.item()))
                    locked[ci] = self._class_conditional_trial_select(all_trials, anchor_cls)
                self._locked_target_trials_per_class = locked
            elif self._disc_mode != _DISC_MODE_NONE:
                # time_delta + y_discrete (any mode): per-class locked targets via
                # class-filtered y-similarity.  For each class c, restrict candidates
                # to trials that contain ≥1 timepoint of class c, then pick the
                # most y-similar trial via _delta_trial_select.
                locked = torch.zeros(self._n_classes, ntrial, dtype=torch.long, device=device)
                yd_2d = self._y_discrete.view(ntrial, ntime)
                for ci in range(self._n_classes):
                    c_val = self._classes_sorted[ci]
                    has_c = (yd_2d == c_val).any(dim=1)  # (ntrial,) valid target trials
                    locked[ci] = self._delta_trial_select(all_trials, valid_trials_mask=has_c)
                self._locked_target_trials_per_class = locked
            else:
                # No y_discrete: class-agnostic trial_emb-based selection
                self._locked_target_trials = self._delta_trial_select(all_trials)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_prior(self, num_samples: int, offset=None) -> torch.Tensor:
        """Sample anchor indices.

        Without ``y_discrete``: uniform over ``[0, ntrial * ntime)``.
        With ``y_discrete``:
          * ``sample_prior="balanced"`` — uniform over classes, then uniform
            within the selected class (oversamples minority classes).
          * ``sample_prior="uniform"`` — uniform over all timepoints; class
            frequency in anchors matches the empirical distribution.
        """
        if not hasattr(self, "_y_discrete") or self.sample_prior_mode == "uniform":
            return self.randint(0, self.ntrial * self.ntime, (num_samples,))

        # Class-balanced: pick class uniformly, then pick random sample within class
        class_idx = self.randint(0, self._n_classes, (num_samples,))  # (B,)
        within_size = self._class_sizes[class_idx]  # (B,)
        within_off = (
            torch.rand(num_samples, device=self.device, generator=self.generator)
            * within_size.float()
        ).long()  # (B,)
        flat_ptr = self._class_offsets[class_idx] + within_off
        return self._class_indices_flat[flat_ptr]

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Sample positive timepoints for each reference anchor."""
        dispatch = {
            "time": self._sample_time,
            "delta": self._sample_delta,
            "time_delta": self._sample_time_delta,
        }
        return dispatch[self.conditional](reference_idx)

    # ------------------------------------------------------------------
    # Multisession primitives: compute_query / search_given_query
    #
    # These decouple the "build a y-space query from the anchor" step from the
    # "find a positive given a query" step, so the query can be shuffled across
    # sessions before searching in a target session (CEBRA multisession
    # philosophy).  Single-session ``sample_conditional`` is untouched.
    #
    # Supported for ``conditional in {"delta", "time_delta"}`` only; ``"time"``
    # has no continuous query to shuffle (CEBRA natively rejects time in
    # multisession via ``_init_loader`` incompatible combinations).
    # ------------------------------------------------------------------

    def compute_query(self, ref: torch.Tensor):
        """Compute a y-space query with Gaussian noise for multisession shuffle.

        Args:
            ref: (B,) flat reference indices in this session.

        Returns:
            ``(query, anchor_class)``:
                query: ``(B, nd)`` y-space query (per-session).
                anchor_class: ``(B,)`` long, or ``None`` when ``y_discrete`` is absent.
        """
        if self.conditional == "time":
            raise NotImplementedError(
                "compute_query is not supported for conditional='time'. "
                "CEBRA natively rejects time in multisession."
            )
        ref.size(0)
        ref_trial = ref // self.ntime
        anchor_class = self._y_discrete[ref] if hasattr(self, "_y_discrete") else None

        if self.conditional == "delta":
            if anchor_class is not None and self._disc_mode == _DISC_MODE_PER_TP_3D:
                class_idx = self._class_value_to_idx(anchor_class)
                # avoid (B, ntrial, nd) — gather per-anchor trial embedding directly
                # _trial_emb_per_class is a stacked (n_classes, ntrial, nd) tensor;
                # use advanced indexing: [class_idx, ref_trial] → (B, nd)
                mean = self._trial_emb_per_class[class_idx, ref_trial]  # (B, nd)
            else:
                mean = self.trial_emb[ref_trial]  # (B, nd)
        else:  # time_delta
            mean = self._y_flat[ref]  # (B, nd)

        noise = torch.empty_like(mean).normal_(generator=self.generator)
        noise = noise * self.delta / (mean.size(-1) ** 0.5)
        return mean + noise, anchor_class

    def search_given_query(
        self,
        query: torch.Tensor,
        anchor_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Find positive flat-idx in THIS session given a y-space query.

        Designed for multisession cross-session use where the query may have
        been produced by a different session and shuffled in.  No time window
        and no self-trial exclusion are applied — cross-session property is
        ensured at the sampler layer.  Same-class constraint is enforced if
        ``anchor_class`` is provided.

        Args:
            query: ``(B, nd)``.
            anchor_class: optional ``(B,)`` long tensor.  Values must exist
                in this session's ``_classes_sorted`` (guaranteed by the
                multisession sampler via class-set validation).

        Returns:
            ``(B,)`` positive flat-idx in ``[0, ntrial * ntime)``.
        """
        if self.conditional == "time":
            raise NotImplementedError("search_given_query is not supported for conditional='time'.")
        B = query.size(0)

        if self.conditional == "delta":
            return self._search_given_query_delta(query, anchor_class, B)
        # time_delta
        return self._search_given_query_time_delta(query, anchor_class, B)

    def _search_given_query_delta(
        self,
        query: torch.Tensor,
        anchor_class: Optional[torch.Tensor],
        B: int,
    ) -> torch.Tensor:
        """Trial select on given query, then class-aware uniform within trial.

        Class-conditional trial selection replicates the logic of
        :py:meth:`_class_conditional_trial_select` but without self exclusion
        (multisession context).  A small Gumbel perturbation on dists handles
        ties (e.g., all trials sharing the same class-c embedding).
        """
        if anchor_class is not None and self._disc_mode == _DISC_MODE_PER_TP_3D:
            class_idx = self._class_value_to_idx(anchor_class)
            # avoid (B, ntrial, nd) — loop over unique classes, cdist each slice
            dists = torch.full((B, self.ntrial), float("inf"), device=query.device)
            for c_idx in class_idx.unique():
                mask_b = class_idx == c_idx
                emb_c = self._trial_emb_per_class[c_idx]  # (ntrial, nd)
                dists[mask_b] = torch.cdist(query[mask_b], emb_c)  # (B_c, ntrial)
        elif anchor_class is not None and self._disc_mode == _DISC_MODE_PER_TRIAL:
            dists = torch.cdist(query, self.trial_emb)  # (B, ntrial)
            class_row = self._trial_class.unsqueeze(0)  # (1, ntrial)
            anchor_col = anchor_class.unsqueeze(1)  # (B, 1)
            dists = dists.masked_fill(class_row != anchor_col, float("inf"))
        else:
            dists = torch.cdist(query, self.trial_emb)  # (B, ntrial)

        # Robust stochastic tiebreak (per-row reference magnitude + abs floor)
        finite = torch.isfinite(dists)
        finite_safe = dists.masked_fill(~finite, 0.0)
        finite_count = finite.sum(dim=1).clamp(min=1).float()
        row_ref = (finite_safe.abs().sum(dim=1) / finite_count).unsqueeze(1)  # (B, 1)
        scale = torch.clamp(row_ref * 1e-6, min=1e-6)
        gumbel = -torch.empty_like(dists).exponential_(generator=self.generator).log()
        dists = dists + scale * gumbel
        target_trial = dists.argmin(dim=1)  # (B,)

        if anchor_class is not None:
            return self._trial_sample_classaware(target_trial, anchor_class)
        rand_off = (torch.rand(B, device=self.device, generator=self.generator) * self.ntime).long()
        return target_trial * self.ntime + rand_off

    def _search_given_query_time_delta(
        self,
        query: torch.Tensor,
        anchor_class: Optional[torch.Tensor],
        B: int,
    ) -> torch.Tensor:
        """Joint argmin over all (trial, t) in y space — time window dropped.

        Chunked over the batch axis to bound peak VRAM when
        ``ntrial * ntime`` is large.
        """
        self.ntrial * self.ntime
        results: list[torch.Tensor] = []
        for start in range(0, B, _ARGMIN_CHUNK):
            end = min(start + _ARGMIN_CHUNK, B)
            q = query[start:end]  # (C, nd)
            # ||q - y||² = ||q||² + ||y||² - 2 q·y, batched via bmm
            q_norm2 = (q**2).sum(-1, keepdim=True)  # (C, 1)
            y_norm2 = self._y_norm2.unsqueeze(0)  # (1, N)
            cross = q @ self._y_flat.T  # (C, N)
            dists = q_norm2 + y_norm2 - 2 * cross  # (C, N)
            if anchor_class is not None:
                cand_disc = self._y_discrete.unsqueeze(0)  # (1, N)
                ac = anchor_class[start:end].unsqueeze(1)  # (C, 1)
                dists = dists.masked_fill(cand_disc != ac, float("inf"))
            # Stochastic tiebreak — per-row scale with abs floor
            finite = torch.isfinite(dists)
            finite_safe = dists.masked_fill(~finite, 0.0)
            finite_count = finite.sum(dim=1).clamp(min=1).float()
            row_ref = (finite_safe.abs().sum(dim=1) / finite_count).unsqueeze(1)
            scale = torch.clamp(row_ref * 1e-6, min=1e-6)
            gumbel = -torch.empty_like(dists).exponential_(generator=self.generator).log()
            dists = dists + scale * gumbel
            results.append(dists.argmin(dim=1))
        return torch.cat(results)

    # ------------------------------------------------------------------
    # Per-conditional sampling (single-session)
    # ------------------------------------------------------------------

    def _sample_time(self, ref: torch.Tensor) -> torch.Tensor:
        ref_trial = ref // self.ntime
        ref_rel = ref % self.ntime
        if hasattr(self, "_y_discrete"):
            anchor_class = self._y_discrete[ref]
            target_trial = self._select_trial_uniform_classaware(ref_trial, ref_rel, anchor_class)
            return self._window_sample_classaware(target_trial, ref_rel, anchor_class)
        target_trial = self._select_trial_uniform(ref_trial)
        return self._window_sample(target_trial, ref_rel)

    def _sample_delta(self, ref: torch.Tensor) -> torch.Tensor:
        ref_trial = ref // self.ntime

        if hasattr(self, "_y_discrete"):
            anchor_class = self._y_discrete[ref]
            if self._disc_mode == _DISC_MODE_PER_TP_2D:
                # Fallback: class-agnostic trial selection (warned at init)
                if self.sample_fix_trial:
                    target_trial = self._locked_target_trials[ref_trial]
                else:
                    target_trial = self._delta_trial_select(ref_trial)
            else:
                # Class-conditional path (Mode A or Mode B)
                if self.sample_fix_trial:
                    class_idx = self._class_value_to_idx(anchor_class)
                    target_trial = self._locked_target_trials_per_class[class_idx, ref_trial]
                else:
                    target_trial = self._class_conditional_trial_select(ref_trial, anchor_class)
            return self._trial_sample_classaware(target_trial, anchor_class)

        # No discrete: original class-agnostic path
        if self.sample_fix_trial:
            target_trial = self._locked_target_trials[ref_trial]
        else:
            target_trial = self._delta_trial_select(ref_trial)
        rand_off = (
            torch.rand(ref.size(0), device=self.device, generator=self.generator) * self.ntime
        ).long()
        return target_trial * self.ntime + rand_off

    def _sample_time_delta(self, ref: torch.Tensor) -> torch.Tensor:
        ref_trial = ref // self.ntime
        ref_rel = ref % self.ntime
        if self.sample_fix_trial:
            anchor_class = self._y_discrete[ref] if hasattr(self, "_y_discrete") else None
            if hasattr(self, "_locked_target_trials_per_class"):
                class_idx = self._class_value_to_idx(anchor_class)
                target_trial = self._locked_target_trials_per_class[class_idx, ref_trial]
            else:
                target_trial = self._locked_target_trials[ref_trial]
            query = self._y_flat[ref]  # (B, nd)
            return self._window_argmin(target_trial, ref_rel, query, anchor_class=anchor_class)
        else:
            return self._joint_argmin(ref, ref_trial, ref_rel)

    # ------------------------------------------------------------------
    # Trial selection helpers
    # ------------------------------------------------------------------

    def _delta_trial_select(
        self,
        ref_trial_ids: torch.Tensor,
        valid_trials_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select target trial via Gaussian-noise delta query on trial_emb.

        For ``sample_exclude_intrial=True`` (default), uses argmin after self
        masking — closest cross-trial neighbour wins. For ``sample_exclude_intrial=False``,
        uses Gumbel-max stochastic sampling (softmax(-dists/T) with T=delta)
        instead of argmin.

        Args:
            valid_trials_mask: Optional ``(ntrial,)`` bool mask. When provided,
                only trials where the mask is True are eligible targets. Falls
                back to unconstrained selection (minus self) if no valid + non-self
                candidate exists for a given anchor.
        """
        B = ref_trial_ids.size(0)
        mean = self.trial_emb[ref_trial_ids]  # (B, nd)
        noise = torch.empty_like(mean).normal_(generator=self.generator)
        noise = noise * self.delta / (mean.size(-1) ** 0.5)
        query = mean + noise  # (B, nd)
        dists = torch.cdist(query, self.trial_emb)  # (B, ntrial)
        if self.sample_exclude_intrial:
            mask = torch.zeros(B, self.ntrial, dtype=torch.bool, device=self.device)
            mask.scatter_(1, ref_trial_ids.unsqueeze(1), True)
            dists = dists.masked_fill(mask, float("inf"))
        if valid_trials_mask is not None:
            dists = dists.masked_fill(~valid_trials_mask.unsqueeze(0), float("inf"))
            # Fallback: if no valid candidate after class filtering, lift the class
            # mask but keep self-exclusion so we always return a finite result.
            no_valid = ~torch.isfinite(dists).any(dim=1)
            if no_valid.any():
                dists_fb = torch.cdist(query[no_valid], self.trial_emb)
                if self.sample_exclude_intrial:
                    mask_fb = torch.zeros(
                        int(no_valid.sum()), self.ntrial, dtype=torch.bool, device=self.device
                    )
                    mask_fb.scatter_(1, ref_trial_ids[no_valid].unsqueeze(1), True)
                    dists_fb = dists_fb.masked_fill(mask_fb, float("inf"))
                dists[no_valid] = dists_fb
        if self.sample_exclude_intrial or valid_trials_mask is not None:
            return dists.argmin(dim=1)
        # excl=False + no filter: stochastic Gumbel-max sampling biased toward closer trials
        T = max(float(self.delta), 1e-6)
        log_w = -dists / T  # (B, ntrial)
        gumbel = -torch.empty_like(log_w).exponential_(generator=self.generator).log()
        return (log_w + gumbel).argmax(dim=1)

    # ------------------------------------------------------------------
    # Class-conditional trial selection (used by _sample_delta when y_discrete)
    # ------------------------------------------------------------------

    def _build_class_conditional_state(
        self,
        yd: torch.Tensor,
        conditional: str,
        device: str,
    ) -> str:
        """Detect discrete mode and build supporting tensors for delta path.

        Returns one of ``_DISC_MODE_*``. Only delta uses the resulting state;
        time enforces same-class at trial level via ``_select_trial_uniform_classaware``;
        time_delta (fix_trial=False) enforces it via ``_joint_argmin``;
        time_delta (fix_trial=True) gets per-class locked trials built in ``__init__``.
        """
        # Detect per-trial vs per-timepoint
        yd_2d = yd.view(self.ntrial, self.ntime)
        is_per_trial = bool((yd_2d == yd_2d[:, :1]).all().item())

        if conditional != "delta":
            # Other conditionals: discrete is enforced inside positive-sampling
            # helpers (_window_sample_classaware, _trial_sample_classaware,
            # _joint_argmin), no extra trial-level state required.
            return _DISC_MODE_PER_TRIAL if is_per_trial else _DISC_MODE_PER_TP_3D

        if is_per_trial:
            # Mode A: each trial belongs to a single class
            self._trial_class = yd_2d[:, 0].clone()  # (ntrial,) class value per trial
            self._trial_class_idx = torch.searchsorted(
                self._classes_sorted, self._trial_class
            )  # (ntrial,) idx into classes
            return _DISC_MODE_PER_TRIAL

        # Per-timepoint discrete
        if self._y_delta_3d is not None:
            # Mode B: per-class trial embedding aggregated over class-c timepoints
            y_3d = self._y_delta_3d  # (ntrial, ntime, nd)
            nd = y_3d.shape[-1]
            trial_emb_pc = torch.zeros(self._n_classes, self.ntrial, nd, device=device)
            for ci in range(self._n_classes):
                c_val = self._classes_sorted[ci]
                mask = (yd_2d == c_val).float()  # (ntrial, ntime)
                counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (ntrial, 1)
                weighted = y_3d * mask.unsqueeze(-1)  # (ntrial, ntime, nd)
                trial_emb_pc[ci] = weighted.sum(dim=1) / counts  # (ntrial, nd)
                # Trials with zero class-c timepoints fall back to overall mean
                missing = mask.sum(dim=1) == 0  # (ntrial,)
                if missing.any():
                    trial_emb_pc[ci, missing] = y_3d[missing].mean(dim=1)
            self._trial_emb_per_class = trial_emb_pc  # (n_classes, ntrial, nd)
            return _DISC_MODE_PER_TP_3D

        # Mode C: per-timepoint discrete with only 2-D y.
        # Should not be reached: __init__ now broadcasts 2-D y to 3-D before
        # calling _build_class_conditional_state, so _y_delta_3d is always set.
        # Kept as a defensive fallback.
        warnings.warn(
            "delta with per-timepoint y_discrete and 2-D y_continuous: "
            "unexpected code path (should have been broadcast in __init__). "
            "Trial selection will be class-agnostic.",
            UserWarning,
            stacklevel=3,
        )
        return _DISC_MODE_PER_TP_2D

    def _class_value_to_idx(self, class_values: torch.Tensor) -> torch.Tensor:
        """Map class-value tensor to indices into ``_classes_sorted``."""
        return torch.searchsorted(self._classes_sorted, class_values)

    def _class_conditional_trial_select(
        self,
        ref_trial_ids: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> torch.Tensor:
        """Select target trial via class-conditional Gaussian-similarity query.

        Behavior depends on ``self._disc_mode`` (set in ``__init__``):

        * ``per_trial`` — restrict candidates to trials whose (single) class
          equals the anchor's class; argmin cdist over ``trial_emb``.
        * ``per_tp_3d`` — use ``_trial_emb_per_class[anchor_class]`` as the
          query basis; argmin cdist among all trials.
        * ``per_tp_2d_fallback`` — delegate to class-agnostic
          ``_delta_trial_select`` (defensive fallback; not reachable via normal
          init since 2-D y is broadcast to 3-D in ``__init__``).

        A small Gumbel noise is added to dists before argmin to stochastically
        break ties (e.g., when all trials share the same class-c embedding,
        as happens for pre-stim gray-screen labels).
        """
        if self._disc_mode == _DISC_MODE_PER_TP_2D:
            return self._delta_trial_select(ref_trial_ids)

        B = ref_trial_ids.size(0)

        if self._disc_mode == _DISC_MODE_PER_TRIAL:
            emb = self.trial_emb  # (ntrial, nd)
            mean = emb[ref_trial_ids]  # (B, nd)
            noise = torch.empty_like(mean).normal_(generator=self.generator)
            noise = noise * self.delta / (mean.size(-1) ** 0.5)
            query = mean + noise  # (B, nd)
            dists = torch.cdist(query, emb)  # (B, ntrial)
            # Restrict candidates to trials whose class matches anchor's
            anchor_col = anchor_class.unsqueeze(1)  # (B, 1)
            class_row = self._trial_class.unsqueeze(0)  # (1, ntrial)
            class_mask = class_row == anchor_col  # (B, ntrial)
            dists = dists.masked_fill(~class_mask, float("inf"))

        else:  # _DISC_MODE_PER_TP_3D
            # Avoid materialising (B, ntrial, nd) by looping over the small
            # number of unique classes present in the batch.
            class_idx = self._class_value_to_idx(anchor_class)  # (B,)
            dists = torch.full((B, self.ntrial), float("inf"), device=self.device)
            for c_idx in class_idx.unique():
                mask_b = class_idx == c_idx  # (B,) bool
                emb_c = self._trial_emb_per_class[c_idx]  # (ntrial, nd)
                mean_c = emb_c[ref_trial_ids[mask_b]]  # (B_c, nd)
                noise = torch.empty_like(mean_c).normal_(generator=self.generator)
                noise = noise * self.delta / (mean_c.size(-1) ** 0.5)
                query_c = mean_c + noise  # (B_c, nd)
                dists[mask_b] = torch.cdist(query_c, emb_c)  # (B_c, ntrial)

        if self.sample_exclude_intrial:
            self_mask = torch.zeros(B, self.ntrial, dtype=torch.bool, device=self.device)
            self_mask.scatter_(1, ref_trial_ids.unsqueeze(1), True)
            dists_excl = dists.masked_fill(self_mask, float("inf"))
            # Fallback: if all candidates masked (e.g., anchor's class has only 1
            # trial), drop self-exclusion so we still return a same-class target
            # rather than degenerating to argmin-of-inf (which silently returns
            # index 0, possibly the wrong class).
            no_valid = ~torch.isfinite(dists_excl).any(dim=1)
            if no_valid.any():
                dists = torch.where(no_valid.unsqueeze(1), dists, dists_excl)
            else:
                dists = dists_excl

        # Stochastic tiebreak: tiny gumbel perturbation, scaled to be robust
        # across batch sizes. We use a *per-anchor* scale derived from the
        # finite dists in each row, with an absolute floor so that even a
        # single-anchor batch with fully tied dists still randomizes.
        finite = torch.isfinite(dists)  # (B, ntrial)
        # Per-row reference magnitude (mean of finite dists in that row).
        # Rows with no finite dists fall back to absolute floor.
        finite_safe = dists.masked_fill(~finite, 0.0)
        finite_count = finite.sum(dim=1).clamp(min=1).float()
        row_ref = (finite_safe.abs().sum(dim=1) / finite_count).unsqueeze(1)  # (B, 1)
        scale = torch.clamp(row_ref * 1e-6, min=1e-6)  # (B, 1)
        gumbel = -torch.empty_like(dists).exponential_(generator=self.generator).log()
        # Gumbel is positive; finite + small_positive stays finite, inf stays inf
        dists = dists + scale * gumbel
        return dists.argmin(dim=1)

    def _select_trial_uniform(self, ref_trial_ids: torch.Tensor) -> torch.Tensor:
        """Sample a target trial uniformly at random, optionally excluding own trial."""
        B = ref_trial_ids.size(0)
        if self.sample_exclude_intrial:
            log_w = torch.zeros(B, self.ntrial, device=self.device)
            log_w.scatter_(
                1,
                ref_trial_ids.unsqueeze(1),
                torch.full((B, 1), float("-inf"), device=self.device),
            )
            gumbel = -torch.empty_like(log_w).exponential_(generator=self.generator).log()
            return (log_w + gumbel).argmax(dim=1)
        else:
            return self.randint(0, self.ntrial, (B,))

    def _select_trial_uniform_classaware(
        self,
        ref_trial_ids: torch.Tensor,
        ref_rel: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> torch.Tensor:
        """Sample target trial uniformly, restricted to trials with ≥1 same-class
        timepoint in the ±time_offsets window around ref_rel.

        Falls back to unrestricted uniform (respecting sample_exclude_intrial) when
        no valid trial exists for an anchor (e.g., rare class with very few trials).
        """
        B = ref_trial_ids.size(0)
        dt_vec = torch.arange(-self.time_offsets, self.time_offsets + 1, device=self.device)
        # (B, W) — clamped window positions for each anchor
        t_pos = (ref_rel.unsqueeze(1) + dt_vec.unsqueeze(0)).clamp(0, self.ntime - 1)

        # Build trial_valid (B, ntrial): True if ≥1 same-class timepoint in window.
        # Iterate over W window offsets to avoid (B, ntrial, W) tensor.
        all_trials = torch.arange(self.ntrial, device=self.device)
        trial_valid = torch.zeros(B, self.ntrial, dtype=torch.bool, device=self.device)
        for w in range(t_pos.shape[1]):
            t_w = t_pos[:, w]  # (B,)
            flat = all_trials.unsqueeze(0) * self.ntime + t_w.unsqueeze(1)  # (B, ntrial)
            trial_valid |= self._y_discrete[flat] == anchor_class.unsqueeze(1)

        log_w = torch.zeros(B, self.ntrial, device=self.device)
        log_w = log_w.masked_fill(~trial_valid, float("-inf"))
        if self.sample_exclude_intrial:
            log_w.scatter_(1, ref_trial_ids.unsqueeze(1), float("-inf"))

        # Fallback rows: no valid candidate after masking → allow all (minus self)
        no_valid = ~torch.isfinite(log_w).any(dim=1)
        if no_valid.any():
            log_w_fb = torch.zeros(B, self.ntrial, device=self.device)
            if self.sample_exclude_intrial:
                log_w_fb.scatter_(1, ref_trial_ids.unsqueeze(1), float("-inf"))
            log_w = torch.where(no_valid.unsqueeze(1), log_w_fb, log_w)

        gumbel = -torch.empty_like(log_w).exponential_(generator=self.generator).log()
        return (log_w + gumbel).argmax(dim=1)

    # ------------------------------------------------------------------
    # Within-trial window sampling
    # ------------------------------------------------------------------

    def _window_sample(self, target_trial: torch.Tensor, ref_rel: torch.Tensor) -> torch.Tensor:
        """Sample uniformly within ±time_offsets of ref_rel inside target_trial."""
        t_rel = torch.clamp(ref_rel, max=self.ntime - 1)
        low = torch.clamp(t_rel - self.time_offsets, min=0)
        high = torch.clamp(t_rel + self.time_offsets + 1, max=self.ntime)
        offset = self._randint_range(low, high)
        return target_trial * self.ntime + low + offset

    def _window_sample_classaware(
        self,
        target_trial: torch.Tensor,
        ref_rel: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> torch.Tensor:
        """Gumbel-max over ±time_offsets window positions matching anchor_class.

        Falls back to unconstrained :py:meth:`_window_sample` for anchors where
        no same-class candidate exists in the target trial's window.
        """
        B = target_trial.size(0)
        W = 2 * self.time_offsets + 1
        dt_vec = torch.arange(-self.time_offsets, self.time_offsets + 1, device=self.device)

        # t positions: (B, W)
        t_pos = (ref_rel.unsqueeze(1) + dt_vec.unsqueeze(0)).clamp(0, self.ntime - 1)
        flat_idx = target_trial.unsqueeze(1) * self.ntime + t_pos  # (B, W)
        disc_match = self._y_discrete[flat_idx] == anchor_class.unsqueeze(1)  # (B, W)

        gumbel = -torch.empty(B, W, device=self.device).exponential_(generator=self.generator).log()
        gumbel = gumbel.masked_fill(~disc_match, float("-inf"))
        best_w = gumbel.argmax(dim=1)  # (B,)
        best_t = t_pos[torch.arange(B, device=self.device), best_w]  # (B,)

        result = target_trial * self.ntime + best_t
        no_match = ~disc_match.any(dim=1)
        if no_match.any():
            result = result.clone()
            result[no_match] = self._window_sample(target_trial[no_match], ref_rel[no_match])
        return result

    def _trial_sample_classaware(
        self,
        target_trial: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> torch.Tensor:
        """Gumbel-max over all timepoints in target_trial matching anchor_class.

        Falls back to uniform random within trial for anchors where no same-class
        timepoint exists in the target trial.
        """
        B = target_trial.size(0)
        all_t = (
            torch.arange(self.ntime, device=self.device).unsqueeze(0).expand(B, -1)
        )  # (B, ntime)
        flat_idx = target_trial.unsqueeze(1) * self.ntime + all_t  # (B, ntime)
        disc_match = self._y_discrete[flat_idx] == anchor_class.unsqueeze(1)  # (B, ntime)

        gumbel = (
            -torch.empty(B, self.ntime, device=self.device)
            .exponential_(generator=self.generator)
            .log()
        )
        gumbel = gumbel.masked_fill(~disc_match, float("-inf"))
        best_t = gumbel.argmax(dim=1)  # (B,)

        result = target_trial * self.ntime + best_t
        no_match = ~disc_match.any(dim=1)
        if no_match.any():
            rand_off = (
                torch.rand(no_match.sum(), device=self.device, generator=self.generator)
                * self.ntime
            ).long()
            result = result.clone()
            result[no_match] = target_trial[no_match] * self.ntime + rand_off
        return result

    def _window_argmin(
        self,
        target_trial: torch.Tensor,
        ref_rel: torch.Tensor,
        query: torch.Tensor,
        anchor_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Within a fixed target trial, return argmin-y timepoint in ±time_offsets window.

        Args:
            target_trial: (B,) trial indices for positive targets.
            ref_rel:      (B,) anchor relative positions (within-trial).
            query:        (B, nd) query embeddings (no noise added here).
            anchor_class: (B,) optional discrete class labels for same-class constraint.

        Returns:
            (B,) flat positive indices.
        """
        B = target_trial.size(0)
        best_dist = torch.full((B,), float("inf"), device=self.device)
        best_t = ref_rel.clamp(0, self.ntime - 1)

        for dt in range(-self.time_offsets, self.time_offsets + 1):
            t_pos = (ref_rel + dt).clamp(0, self.ntime - 1)
            flat_idx = target_trial * self.ntime + t_pos
            y_at_t = self._y_flat[flat_idx]  # (B, nd)
            dist = (y_at_t - query).pow(2).sum(-1)  # (B,)
            if anchor_class is not None:
                disc_mismatch = self._y_discrete[flat_idx] != anchor_class
                dist = dist.masked_fill(disc_mismatch, float("inf"))
            update = dist < best_dist
            best_dist = torch.where(update, dist, best_dist)
            best_t = torch.where(update, t_pos, best_t)

        return target_trial * self.ntime + best_t

    # ------------------------------------------------------------------
    # Joint argmin for time_delta (fix_trial=False)
    # ------------------------------------------------------------------

    def _joint_argmin(
        self,
        ref: torch.Tensor,
        ref_trial: torch.Tensor,
        ref_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Argmin over all cross-trial candidates within the ±time_offsets window.

        Candidate pool for anchor (trial_i, rel_i)::

            {(trial_j, t) : trial_j ≠ trial_i, |t − rel_i| ≤ time_offsets}

        All W window positions are gathered at once and distances computed via
        a single batched matrix multiply, avoiding the Python for-loop over W.
        Processed in chunks of ``_ARGMIN_CHUNK`` anchors to bound peak VRAM
        (peak tensor: ``(chunk, ntrial*W, nd)``).

        Returns:
            (B,) flat positive indices.
        """
        B = ref.size(0)
        nd = self._y_flat.shape[1]
        W = 2 * self.time_offsets + 1
        all_trials = torch.arange(self.ntrial, device=self.device)  # (ntrial,)
        dt_vec = torch.arange(
            -self.time_offsets,
            self.time_offsets + 1,
            device=self.device,  # (W,)
        )

        results: list[torch.Tensor] = []
        for start in range(0, B, _ARGMIN_CHUNK):
            end = min(start + _ARGMIN_CHUNK, B)
            c_ref = ref[start:end]
            c_trial = ref_trial[start:end]
            c_rel = ref_rel[start:end]
            C = c_ref.size(0)

            # Query: y at anchor + Gaussian noise — (C, nd)
            noise = torch.empty(C, nd, device=self.device).normal_(generator=self.generator)
            noise *= self.delta / (nd**0.5)
            query = self._y_flat[c_ref] + noise

            # Window t positions for every (anchor, w): (C, W)
            t_cands = (c_rel.unsqueeze(1) + dt_vec.unsqueeze(0)).clamp(0, self.ntime - 1)

            # Flat indices for all (anchor, trial, w): (C, ntrial*W)
            flat_idx = (all_trials.view(1, -1, 1) * self.ntime + t_cands.view(C, 1, W)).view(C, -1)

            # Gather y and compute distances via ||q-y||² = ||q||² + ||y||² - 2q·y
            y_cands = self._y_flat[flat_idx]  # (C, ntrial*W, nd)
            q_norm2 = (query**2).sum(-1, keepdim=True)  # (C, 1)
            y_norm2 = self._y_norm2[flat_idx]  # (C, ntrial*W)
            cross = torch.bmm(y_cands, query.unsqueeze(-1)).squeeze(-1)  # (C, ntrial*W)
            dist_flat = q_norm2 + y_norm2 - 2 * cross  # (C, ntrial*W)

            # Same-class constraint: mask candidates whose discrete label ≠ anchor's
            if hasattr(self, "_y_discrete"):
                anchor_disc = self._y_discrete[c_ref]  # (C,)
                cand_disc = self._y_discrete[flat_idx]  # (C, ntrial*W)
                disc_mismatch = cand_disc != anchor_disc.unsqueeze(1)
                dist_flat = dist_flat.masked_fill(disc_mismatch, float("inf"))

            # Min over W for each (anchor, trial): (C, ntrial)
            best_dist, best_w = dist_flat.view(C, self.ntrial, W).min(dim=2)

            # Recover best t: best_w indexes into t_cands's W axis
            c_idx = torch.arange(C, device=self.device).unsqueeze(1).expand_as(best_w)
            best_t = t_cands[c_idx, best_w]  # (C, ntrial)

            if self.sample_exclude_intrial:
                best_dist.scatter_(1, c_trial.unsqueeze(1), float("inf"))

            target_trial = best_dist.argmin(dim=1)  # (C,)
            target_t = best_t[torch.arange(C, device=self.device), target_trial]
            results.append(target_trial * self.ntime + target_t)

        return torch.cat(results)

    def _randint_range(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Per-element uniform integer in ``[0, high - low)``."""
        span = (high - low).float()
        return (torch.rand(low.size(0), device=self.device, generator=self.generator) * span).long()
