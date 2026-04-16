"""Trial-aware distributions for hierarchical contrastive sampling.

Five trial-aware conditionals based on three orthogonal axes
(trial selection × time constraint × locking):

  ===========================  ================  ==============  =========
  ``conditional``              Trial selection   Time constraint  Locking
  ===========================  ================  ==============  =========
  ``"trialTime"``              Random            ±time_offset    —
  ``"trialDelta"``             delta-style       Free            Locked
  ``"trial_delta"``            delta-style       Free            Re-sampled
  ``"trialTime_delta"``        delta-style       ±time_offset    Re-sampled
  ``"trialTime_trialDelta"``   time_delta-style  ±time_offset    Locked
  ===========================  ================  ==============  =========

Trial selection principles mirror CEBRA's own conditionals:

  **delta-style** (``trialDelta``, ``trial_delta``):
    Follows CEBRA's ``DeltaNormalDistribution``.  A query is formed by adding
    isotropic Gaussian noise to the anchor's trial-mean embedding::

        query = trial_mean[anchor] + N(0, δ²I)
        target_trial = argmin_j  dist(query, trial_mean[j])

    δ controls the exploration radius in trial-embedding space.  Because the
    noise is freshly sampled each training step, different steps map the same
    anchor to different target trials, providing structured diversity.

  **time_delta-style** (``trialTime_delta``, ``trialTime_trialDelta``):
    Follows CEBRA's ``TimedeltaDistribution``.  A query is formed by adding a
    randomly-drawn empirical stim-velocity vector to the anchor's trial-mean::

        Δstim[k] = continuous[k] - continuous[k - time_offset]  (pre-computed)
        query    = trial_mean[anchor] + Δstim[random_k]
        target_trial = argmin_j  dist(query, trial_mean[j])

    This mirrors the data-driven perturbation of CEBRA's ``time_delta``,
    transposed from the timepoint level to the trial level.

Naming convention:
  - ``trialDelta`` (capital D, no underscore) = Locked delta-style trial.
  - ``trial_delta`` (underscore, lowercase d) = Re-sampled delta-style trial.
  - ``_delta`` and ``_trialDelta`` suffixes denote the trial-selection mechanism.

Trial-level locking (``trialDelta``, ``trialTime_trialDelta``):
  A fixed ``ref_trial → target_trial`` mapping is pre-computed once at
  ``__init__`` time using the respective query mechanism (delta-style or
  time_delta-style, one noise/diff sample per trial).

Re-sampled (``trial_delta``, ``trialTime_delta``):
  Target trial is independently re-sampled at each training step, providing
  more diverse positive pairs.

Gap (inter-trial) timepoint handling:

  ===========================  ========================
  ``conditional``              Gap strategy
  ===========================  ========================
  ``"trialTime"``              Global ±time_offset
  ``"trialDelta"``             delta-style (timepoint)
  ``"trial_delta"``            delta-style (timepoint)
  ``"trialTime_delta"``        delta-style (timepoint)
  ``"trialTime_trialDelta"``   time_delta-style (tp)
  ===========================  ========================

When a discrete index is provided, positive samples are restricted to the
same discrete class as the reference (trial selection and gap sampling both
respect this constraint).
"""

from __future__ import annotations

from typing import Optional

import cebra.distributions.base as abc_
import torch
from cebra.data.datatypes import Offset
from cebra.distributions.discrete import DiscreteUniform

TRIAL_CONDITIONALS = frozenset(
    {"trialTime", "trialDelta", "trial_delta", "trialTime_delta", "trialTime_trialDelta"}
)

# Conditionals that require trial-level embeddings and query-based selection
_NEEDS_SIMILARITY = frozenset(
    {"trialDelta", "trial_delta", "trialTime_delta", "trialTime_trialDelta"}
)

# trial selection follows CEBRA's delta principle (isotropic Gaussian query)
_NEEDS_DELTA_TRIAL = frozenset({"trialDelta", "trial_delta"})

# trial selection follows CEBRA's time_delta principle (empirical stim-diff query)
# Note: trialTime_delta uses delta-style selection + ±time_offset window (NOT pool-based).
_NEEDS_TIMEDELT_TRIAL = frozenset({"trialTime_trialDelta"})

# Conditionals that use trial-level locking (_locked_target_trials)
_NEEDS_LOCKING = frozenset({"trialDelta", "trialTime_trialDelta"})


class TrialAwareDistribution(abc_.JointDistribution, abc_.HasGenerator):
    """Trial-aware hierarchical sampling distribution.

    Args:
        continuous: Continuous auxiliary variable per timepoint, shape ``(N, d)``.
        trial_starts: Start index of each trial (inclusive), shape ``(T,)``.
        trial_ends: End index of each trial (exclusive), shape ``(T,)``.
        conditional: Sampling mode.  One of ``"trialTime"``, ``"trialDelta"``,
            ``"trial_delta"``, ``"trialTime_delta"``, ``"trialTime_trialDelta"``.
        time_offset: Half-width of the within-trial (or pool) time window.
            Also used as the lag for pre-computing ``Δstim`` in
            time_delta-style conditionals.
        delta: Standard deviation of the isotropic Gaussian noise used for
            delta-style trial selection (``trialDelta``, ``trial_delta``) and
            delta-style gap sampling.  ``None`` falls back to deterministic
            nearest-neighbour.  Required (non-None) for delta-style conditionals.
        device: Compute device.
        seed: Random seed for the generator.
        discrete: Integer class label per timepoint, shape ``(N,)``.
            Positive samples are restricted to the same class.

    Example — ``trialTime``::

        >>> import torch
        >>> continuous   = torch.randn(500, 10)
        >>> trial_starts = torch.tensor([0, 100, 200, 300, 400])
        >>> trial_ends   = torch.tensor([100, 200, 300, 400, 500])
        >>> dist = TrialAwareDistribution(
        ...     continuous=continuous,
        ...     trial_starts=trial_starts,
        ...     trial_ends=trial_ends,
        ...     conditional="trialTime",
        ...     time_offset=10,
        ... )
        >>> ref, pos = dist.sample_joint(num_samples=32)

    Example — ``trialDelta`` (locked delta-style + uniform in trial)::

        >>> dist = TrialAwareDistribution(
        ...     continuous=continuous,
        ...     trial_starts=trial_starts,
        ...     trial_ends=trial_ends,
        ...     conditional="trialDelta",
        ...     time_offset=10,
        ...     delta=0.1,
        ... )
        >>> ref, pos = dist.sample_joint(num_samples=32)

    Example — ``trial_delta`` (re-sampled delta-style + free)::

        >>> dist = TrialAwareDistribution(
        ...     continuous=continuous,
        ...     trial_starts=trial_starts,
        ...     trial_ends=trial_ends,
        ...     conditional="trial_delta",
        ...     time_offset=10,
        ...     delta=0.1,
        ... )
        >>> ref, pos = dist.sample_joint(num_samples=32)

    Example — ``trialTime_delta`` (re-sampled delta-style + ±offset window)::

        >>> dist = TrialAwareDistribution(
        ...     continuous=continuous,
        ...     trial_starts=trial_starts,
        ...     trial_ends=trial_ends,
        ...     conditional="trialTime_delta",
        ...     time_offset=10,
        ...     delta=0.1,
        ... )
        >>> ref, pos = dist.sample_joint(num_samples=32)

    Example — ``trialTime_trialDelta`` (locked time_delta-style + ±offset)::

        >>> dist = TrialAwareDistribution(
        ...     continuous=continuous,
        ...     trial_starts=trial_starts,
        ...     trial_ends=trial_ends,
        ...     conditional="trialTime_trialDelta",
        ...     time_offset=10,
        ...     delta=0.1,
        ... )
        >>> ref, pos = dist.sample_joint(num_samples=32)
    """

    def __init__(
        self,
        continuous: torch.Tensor,
        trial_starts: torch.Tensor,
        trial_ends: torch.Tensor,
        conditional: str,
        time_offset: int = 10,
        delta: Optional[float] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        discrete: Optional[torch.Tensor] = None,
    ):
        abc_.HasGenerator.__init__(self, device=device, seed=seed)

        # --- Validate conditional ---
        if conditional not in TRIAL_CONDITIONALS:
            raise ValueError(
                f"Unknown conditional: {conditional!r}. "
                f"Must be one of {sorted(TRIAL_CONDITIONALS)}."
            )

        # --- Core attributes ---
        self.continuous = continuous.to(device)
        self.trial_starts = trial_starts.long().to(device)
        self.trial_ends = trial_ends.long().to(device)
        self.conditional = conditional
        self.time_offset = time_offset
        self.delta = delta
        self.num_trials = len(trial_starts)

        # --- Validate trial metadata ---
        if self.num_trials == 0:
            raise ValueError("At least one trial must be provided.")
        if len(trial_starts) != len(trial_ends):
            raise ValueError(
                f"trial_starts ({len(trial_starts)}) and trial_ends "
                f"({len(trial_ends)}) must have the same length."
            )
        if not torch.all(self.trial_starts < self.trial_ends):
            raise ValueError("All trial_starts must be < trial_ends.")
        if self.num_trials < 2:
            raise ValueError("At least 2 trials are required for cross-trial sampling.")

        self.trial_lengths = self.trial_ends - self.trial_starts

        # --- Discrete index ---
        if discrete is not None:
            self.discrete = discrete.long().to(device)
            self.trial_discrete = self.discrete[self.trial_starts]  # (T,)
            self._trial_same_class = torch.eq(
                self.trial_discrete.unsqueeze(1),  # (T, 1)
                self.trial_discrete.unsqueeze(0),  # (1, T)
            )  # (T, T)
            # Class-balanced prior (same design as native CEBRA's MixedDataLoader)
            self._discrete_prior = DiscreteUniform(self.discrete.cpu())
        else:
            self.discrete = None

        # --- Timepoint → trial mapping (-1 for gap timepoints) ---
        N = len(self.continuous)
        self.timepoint_to_trial = torch.full((N,), -1, dtype=torch.long, device=device)
        for t in range(self.num_trials):
            s, e = int(self.trial_starts[t]), int(self.trial_ends[t])
            self.timepoint_to_trial[s:e] = t

        # --- Timepoint relative position within its trial (-1 for gap) ---
        self.timepoint_rel_pos = torch.full((N,), -1, dtype=torch.long, device=device)
        for t in range(self.num_trials):
            s, e = int(self.trial_starts[t]), int(self.trial_ends[t])
            self.timepoint_rel_pos[s:e] = torch.arange(e - s, device=device)

        # Convenience: all trial timepoint indices concatenated
        self._all_trial_indices = torch.cat(
            [
                torch.arange(int(self.trial_starts[t]), int(self.trial_ends[t]), device=device)
                for t in range(self.num_trials)
            ]
        )

        # --- Trial-level embeddings (mean of each trial's continuous index) ---
        if conditional in _NEEDS_SIMILARITY:
            trial_embs = [
                self.continuous[int(self.trial_starts[t]) : int(self.trial_ends[t])].mean(dim=0)
                for t in range(self.num_trials)
            ]
            self.trial_embeddings = torch.stack(trial_embs).to(device)  # (T, d)

            # time_delta-style: pre-compute timepoint-level stim velocity vectors.
            # Identical construction to CEBRA's TimedeltaDistribution:
            #   _time_difference[k] = continuous[k] - continuous[k - time_offset]
            if conditional in _NEEDS_TIMEDELT_TRIAL:
                td = self.time_offset
                _diff = torch.zeros_like(self.continuous)  # (N, d)
                _diff[td:] = self.continuous[td:] - self.continuous[:-td]
                self._time_difference = _diff  # (N, d)

        # --- Pre-computed locked target trials ---
        if conditional in _NEEDS_LOCKING:
            self._locked_target_trials = self._compute_locked_trials()  # (T,)

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------

    def sample_prior(self, num_samples: int, offset: Optional[Offset] = None) -> torch.Tensor:
        """Sample reference timepoints for use as anchors and negatives.

        When no discrete index is provided, samples uniformly over all ``N``
        timepoints (original behaviour).

        When a discrete index is present, delegates to
        :py:class:`~cebra.distributions.discrete.DiscreteUniform` so that
        every discrete class is equally likely regardless of class size —
        matching the design of native CEBRA's ``MixedDataLoader``.

        Args:
            num_samples: Number of samples to draw.
            offset: Unused; kept for API compatibility with CEBRA base class.

        Returns:
            Integer tensor of shape ``(num_samples,)``.
        """
        if self.discrete is None:
            N = len(self.continuous)
            return self.randint(0, N, (num_samples,))
        return self._discrete_prior.sample_prior(num_samples).to(self.device)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Sample positive pairs according to the configured conditional.

        Args:
            reference_idx: Reference timepoint indices, shape ``(B,)``.

        Returns:
            Positive timepoint indices, shape ``(B,)``.
        """
        if reference_idx.dim() != 1:
            raise ValueError(f"Reference indices must be 1D, got shape {reference_idx.shape}.")
        _dispatch = {
            "trialTime": self._sample_trial_time,
            "trialDelta": self._sample_trial_delta,
            "trial_delta": self._sample_trial_delta_resampled,
            "trialTime_delta": self._sample_trial_time_delta,
            "trialTime_trialDelta": self._sample_trial_time_trial_delta,
        }
        return _dispatch[self.conditional](reference_idx)

    # ------------------------------------------------------------------
    # Per-conditional sampling implementations
    # ------------------------------------------------------------------

    def _sample_trial_time(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """``trialTime``: uniform random target trial + ±time_offset window.

        Trial anchors:  pick a target trial uniformly at random (≠ own trial),
                        sample within ±time_offset of the reference's relative
                        position inside the target trial.
        Gap anchors:    global ±time_offset window sampling.
        """
        ref_trial_ids = self.timepoint_to_trial[reference_idx]
        is_trial = ref_trial_ids >= 0
        positive_idx = torch.empty_like(reference_idx)

        if is_trial.any():
            ref_t = reference_idx[is_trial]
            ref_trial = ref_trial_ids[is_trial]
            ref_rel = ref_t - self.trial_starts[ref_trial]

            target_trial = self._uniform_other_trial(ref_trial)

            t_start = self.trial_starts[target_trial]
            t_len = self.trial_lengths[target_trial]
            t_rel = torch.clamp(ref_rel, max=t_len - 1)
            low = torch.clamp(t_rel - self.time_offset, min=0)
            high = torch.clamp(t_rel + self.time_offset + 1, max=t_len)
            positive_idx[is_trial] = t_start + low + self._randint_range(low, high)

        if (~is_trial).any():
            positive_idx[~is_trial] = self._gap_by_time(reference_idx[~is_trial])

        return positive_idx

    def _sample_trial_delta(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """``trialDelta``: locked delta-style target trial + uniform in trial.

        Trial anchors:  use pre-computed ``_locked_target_trials`` mapping
                        (delta-style, fixed at init); sample uniformly from
                        the entire target trial.
        Gap anchors:    delta-style content sampling (timepoint level).
        """
        ref_trial_ids = self.timepoint_to_trial[reference_idx]
        is_trial = ref_trial_ids >= 0
        positive_idx = torch.empty_like(reference_idx)

        if is_trial.any():
            ref_trial = ref_trial_ids[is_trial]
            target_trial = self._locked_target_trials[ref_trial]

            t_start = self.trial_starts[target_trial]
            t_len = self.trial_lengths[target_trial]
            rand_off = (
                torch.rand(ref_trial.size(0), device=self.device, generator=self.generator)
                * t_len.float()
            ).long()
            positive_idx[is_trial] = t_start + rand_off

        if (~is_trial).any():
            positive_idx[~is_trial] = self._gap_by_delta(reference_idx[~is_trial])

        return positive_idx

    def _sample_trial_delta_resampled(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """``trial_delta``: re-sampled delta-style trial + uniform in trial.

        Each call independently samples a target trial via delta-style query
        (CEBRA ``DeltaNormalDistribution`` principle at trial level), then
        draws a timepoint uniformly from the entire target trial.

        Trial anchors:  per-step delta-style trial selection +
                        uniform sampling within the selected trial.
        Gap anchors:    delta-style content sampling (timepoint level).
        """
        ref_trial_ids = self.timepoint_to_trial[reference_idx]
        is_trial = ref_trial_ids >= 0
        positive_idx = torch.empty_like(reference_idx)

        if is_trial.any():
            ref_trial = ref_trial_ids[is_trial]
            target_trial = self._delta_trial_select(ref_trial)

            t_start = self.trial_starts[target_trial]
            t_len = self.trial_lengths[target_trial]
            rand_off = (
                torch.rand(ref_trial.size(0), device=self.device, generator=self.generator)
                * t_len.float()
            ).long()
            positive_idx[is_trial] = t_start + rand_off

        if (~is_trial).any():
            positive_idx[~is_trial] = self._gap_by_delta(reference_idx[~is_trial])

        return positive_idx

    def _sample_trial_time_delta(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """``trialTime_delta``: re-sampled delta-style trial + ±offset window.

        Uses the identical trial-selection mechanism as ``trial_delta``
        (per-step isotropic Gaussian query on trial-mean embeddings), but
        constrains the drawn timepoint to ±time_offset of the anchor's
        relative position inside the target trial.

        This gives the same trial diversity as ``trial_delta`` while
        additionally enforcing temporal proximity within the target trial.
        Pool-based argmin was discarded: for stimulus data where embeddings
        are constant within a trial (Δstim ≈ 0), argmin over timepoints
        always selects the same trial, yielding near-zero diversity.

        Trial anchors:  per-step delta-style trial selection
                        (``_delta_trial_select``); draw uniformly within
                        ``[ref_rel ± time_offset]`` of the target trial.
        Gap anchors:    delta-style content sampling (timepoint level).
        """
        ref_trial_ids = self.timepoint_to_trial[reference_idx]
        is_trial = ref_trial_ids >= 0
        positive_idx = torch.empty_like(reference_idx)

        if is_trial.any():
            ref_t = reference_idx[is_trial]
            ref_trial = ref_trial_ids[is_trial]
            ref_rel = ref_t - self.trial_starts[ref_trial]

            target_trial = self._delta_trial_select(ref_trial)

            t_start = self.trial_starts[target_trial]
            t_len = self.trial_lengths[target_trial]
            t_rel = torch.clamp(ref_rel, max=t_len - 1)
            low = torch.clamp(t_rel - self.time_offset, min=0)
            high = torch.clamp(t_rel + self.time_offset + 1, max=t_len)
            positive_idx[is_trial] = t_start + low + self._randint_range(low, high)

        if (~is_trial).any():
            positive_idx[~is_trial] = self._gap_by_delta(reference_idx[~is_trial])

        return positive_idx

    def _sample_trial_time_trial_delta(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """``trialTime_trialDelta``: locked time_delta-style trial + ±offset window.

        Trial anchors:  use pre-computed ``_locked_target_trials`` mapping
                        (time_delta-style, fixed at init); sample within
                        ±time_offset of the reference's relative position in
                        the locked target trial.
        Gap anchors:    time_delta-style content sampling (timepoint level).
        """
        ref_trial_ids = self.timepoint_to_trial[reference_idx]
        is_trial = ref_trial_ids >= 0
        positive_idx = torch.empty_like(reference_idx)

        if is_trial.any():
            ref_t = reference_idx[is_trial]
            ref_trial = ref_trial_ids[is_trial]
            ref_rel = ref_t - self.trial_starts[ref_trial]

            target_trial = self._locked_target_trials[ref_trial]

            t_start = self.trial_starts[target_trial]
            t_len = self.trial_lengths[target_trial]
            t_rel = torch.clamp(ref_rel, max=t_len - 1)
            low = torch.clamp(t_rel - self.time_offset, min=0)
            high = torch.clamp(t_rel + self.time_offset + 1, max=t_len)
            positive_idx[is_trial] = t_start + low + self._randint_range(low, high)

        if (~is_trial).any():
            positive_idx[~is_trial] = self._gap_by_timedelt(reference_idx[~is_trial])

        return positive_idx

    # ------------------------------------------------------------------
    # Private helpers: trial selection
    # ------------------------------------------------------------------

    def _compute_locked_trials(self) -> torch.Tensor:
        """Pre-compute a fixed ``ref_trial → target_trial`` mapping at init.

        Uses the appropriate query mechanism for the conditional:

        * ``trialDelta``: delta-style (one Gaussian noise sample per trial).
        * ``trialTime_trialDelta``: time_delta-style (one stim-diff sample per trial).
        """
        trial_ids = torch.arange(self.num_trials, device=self.device)
        if self.conditional in _NEEDS_DELTA_TRIAL:
            return self._delta_trial_select(trial_ids)
        else:
            return self._timedelt_trial_select(trial_ids)

    def _delta_trial_select(self, ref_trial_ids: torch.Tensor) -> torch.Tensor:
        """Select target trial via CEBRA delta principle.

        Follows :py:class:`cebra.distributions.continuous.DeltaNormalDistribution`
        transposed to trial level::

            query = trial_mean[ref_trial] + N(0, δ²I)
            target = argmin_j  dist(query, trial_mean[j])

        Both own-trial exclusion and discrete-class filtering are applied.

        Args:
            ref_trial_ids: Trial IDs of reference timepoints, shape ``(B,)``.

        Returns:
            Target trial IDs, shape ``(B,)``.
        """
        B, T = ref_trial_ids.size(0), self.num_trials
        mean = self.trial_embeddings[ref_trial_ids]  # (B, d)
        noise = torch.empty_like(mean).normal_(generator=self.generator)
        if self.delta is not None:
            noise = noise * self.delta / (mean.size(-1) ** 0.5)
        query = mean + noise  # (B, d)

        dists = torch.cdist(query, self.trial_embeddings)  # (B, T)

        # Exclude own trial
        self_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        self_mask.scatter_(1, ref_trial_ids.unsqueeze(1), True)
        dists = dists.masked_fill(self_mask, float("inf"))

        # Discrete class filter
        if self.discrete is not None:
            same_class = self._trial_same_class[ref_trial_ids]
            dists = dists.masked_fill(~same_class, float("inf"))

        return dists.argmin(dim=1)  # (B,)

    def _timedelt_trial_select(self, ref_trial_ids: torch.Tensor) -> torch.Tensor:
        """Select target trial via CEBRA time_delta principle.

        Follows :py:class:`cebra.distributions.continuous.TimedeltaDistribution`
        transposed to trial level::

            Δstim[k] = continuous[k] - continuous[k - time_offset]
            query    = trial_mean[ref_trial] + Δstim[random_k]
            target   = argmin_j  dist(query, trial_mean[j])

        Both own-trial exclusion and discrete-class filtering are applied.

        Args:
            ref_trial_ids: Trial IDs of reference timepoints, shape ``(B,)``.

        Returns:
            Target trial IDs, shape ``(B,)``.
        """
        B, T = ref_trial_ids.size(0), self.num_trials
        anchor_stim = self.trial_embeddings[ref_trial_ids]  # (B, d)
        diff_idx = self.randint(len(self._time_difference), (B,))
        query = anchor_stim + self._time_difference[diff_idx]  # (B, d)

        dists = torch.cdist(query, self.trial_embeddings)  # (B, T)

        # Exclude own trial
        self_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        self_mask.scatter_(1, ref_trial_ids.unsqueeze(1), True)
        dists = dists.masked_fill(self_mask, float("inf"))

        # Discrete class filter
        if self.discrete is not None:
            same_class = self._trial_same_class[ref_trial_ids]
            dists = dists.masked_fill(~same_class, float("inf"))

        return dists.argmin(dim=1)  # (B,)

    def _uniform_other_trial(self, ref_trial_ids: torch.Tensor) -> torch.Tensor:
        """Sample a target trial uniformly at random (≠ own, same class).

        Uses the Gumbel-max trick: log-weight ``0`` for valid trials,
        ``-inf`` for own trial and (optionally) different-class trials.

        Args:
            ref_trial_ids: Trial IDs of reference timepoints, shape ``(B,)``.

        Returns:
            Target trial IDs, shape ``(B,)``.
        """
        B, T = ref_trial_ids.size(0), self.num_trials
        log_w = torch.zeros(B, T, device=self.device)

        log_w.scatter_(
            1, ref_trial_ids.unsqueeze(1), torch.full((B, 1), float("-inf"), device=self.device)
        )

        if self.discrete is not None:
            same_class = self._trial_same_class[ref_trial_ids]
            log_w = log_w.masked_fill(~same_class, float("-inf"))

        gumbel = -torch.empty_like(log_w).exponential_(generator=self.generator).log()
        return (log_w + gumbel).argmax(dim=1)

    # ------------------------------------------------------------------
    # Private helpers: gap anchor strategies
    # ------------------------------------------------------------------

    def _gap_by_time(self, ref_idx: torch.Tensor) -> torch.Tensor:
        """Global ±time_offset window sampling for gap anchors (``trialTime``).

        Without a discrete index: uniform over the full ±time_offset window.
        With a discrete index: sample uniformly from **all** same-class
        timepoints globally (the ±time_offset window is dropped).  This
        forces every gap timepoint to be treated as an exchangeable member
        of its discrete class, so the encoder clusters the entire gap period
        into one region rather than preserving intra-gap temporal structure.
        """
        N = len(self.continuous)

        if self.discrete is None:
            low = torch.clamp(ref_idx - self.time_offset, min=0)
            high = torch.clamp(ref_idx + self.time_offset + 1, max=N)
            return low + self._randint_range(low, high)

        # Discrete present: global class-uniform sampling (Gumbel-max trick).
        B = ref_idx.size(0)
        all_idx = torch.arange(N, device=self.device)  # (N,)

        ref_class = self.discrete[ref_idx]  # (B,)
        same_class = self.discrete.unsqueeze(0) == ref_class.unsqueeze(1)  # (B, N)
        not_self = all_idx.unsqueeze(0) != ref_idx.unsqueeze(1)  # (B, N)
        valid = same_class & not_self  # (B, N)

        log_w = torch.where(
            valid,
            torch.zeros(B, N, device=self.device),
            torch.full((B, N), float("-inf"), device=self.device),
        )
        gumbel = -torch.empty(B, N, device=self.device).exponential_(generator=self.generator).log()
        return (log_w + gumbel).argmax(dim=1)  # (B,)

    def _gap_by_delta(self, ref_idx: torch.Tensor) -> torch.Tensor:
        """Gap anchor sampling: CEBRA delta principle at timepoint level.

        Constructs ``query = stim[anchor] + N(0, δ²I)`` (following
        :py:class:`cebra.distributions.continuous.DeltaNormalDistribution`)
        and returns the nearest same-class timepoint via argmin.
        """
        mean = self.continuous[ref_idx]  # (n, d)
        noise = torch.empty_like(mean).normal_(generator=self.generator)
        if self.delta is not None:
            noise = noise * self.delta / (mean.size(-1) ** 0.5)
        query = mean + noise
        return self._gap_query_argmin(ref_idx, query)

    def _gap_by_timedelt(self, ref_idx: torch.Tensor) -> torch.Tensor:
        """Gap anchor sampling: CEBRA time_delta principle at timepoint level.

        Constructs ``query = stim[anchor] + Δstim[random_k]`` (following
        :py:class:`cebra.distributions.continuous.TimedeltaDistribution`)
        and returns the nearest same-class timepoint via argmin.
        """
        mean = self.continuous[ref_idx]  # (n, d)
        diff_idx = self.randint(len(self._time_difference), (ref_idx.size(0),))
        query = mean + self._time_difference[diff_idx]
        return self._gap_query_argmin(ref_idx, query)

    def _gap_query_argmin(self, ref_idx: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Shared argmin for gap anchor queries with masking.

        Computes ``argmin_i dist(query, continuous[i])`` over all same-class
        timepoints, excluding self.

        Args:
            ref_idx: Gap anchor indices, shape ``(n,)``.
            query:   Query embeddings, shape ``(n, d)``.

        Returns:
            Positive timepoint indices, shape ``(n,)``.
        """
        dist = torch.cdist(query, self.continuous)  # (n, N)

        # Exclude self
        dist.scatter_(
            1,
            ref_idx.unsqueeze(1),
            torch.full((ref_idx.size(0), 1), float("inf"), device=self.device),
        )

        # Discrete class filter
        if self.discrete is not None:
            ref_class = self.discrete[ref_idx]  # (n,)
            same = torch.eq(
                self.discrete.unsqueeze(0),  # (1, N)
                ref_class.unsqueeze(1),  # (n, 1)
            )
            dist = dist.masked_fill(~same, float("inf"))

        return dist.argmin(dim=1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _randint_range(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Per-element uniform integer in ``[0, high - low)``.

        Args:
            low:  Lower bound tensor, shape ``(B,)``.
            high: Upper bound tensor, shape ``(B,)``.

        Returns:
            Integer offset tensor of shape ``(B,)``.
        """
        range_size = (high - low).float()
        return (
            torch.rand(low.size(0), device=self.device, generator=self.generator) * range_size
        ).long()
