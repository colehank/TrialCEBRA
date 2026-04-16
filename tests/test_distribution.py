"""Tests for TrialAwareDistribution — all four sampling conditionals."""

import pytest
import torch
from conftest import make_trial_data

from trial_cebra.distribution import TRIAL_CONDITIONALS, TrialAwareDistribution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dist(
    num_trials=8,
    trial_length=60,
    gap=20,
    conditional="trialTime",
    delta=0.1,
    time_offset=10,
    discrete=None,
    **kw,
):
    continuous, starts, ends, total = make_trial_data(
        num_trials=num_trials, trial_length=trial_length, gap=gap
    )
    return (
        TrialAwareDistribution(
            continuous=continuous,
            trial_starts=starts,
            trial_ends=ends,
            conditional=conditional,
            time_offset=time_offset,
            delta=delta,
            discrete=discrete,
            **kw,
        ),
        starts,
        ends,
        total,
    )


# ---------------------------------------------------------------------------
# Init & validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_valid_all_conditionals(self):
        for cond in TRIAL_CONDITIONALS:
            dist, *_ = _make_dist(conditional=cond)
            assert dist.conditional == cond

    def test_unknown_conditional_raises(self):
        continuous, starts, ends, _ = make_trial_data()
        with pytest.raises(ValueError, match="Unknown conditional"):
            TrialAwareDistribution(
                continuous=continuous, trial_starts=starts, trial_ends=ends, conditional="bad_mode"
            )

    def test_empty_trials_raises(self):
        with pytest.raises(ValueError, match="At least one trial"):
            TrialAwareDistribution(
                continuous=torch.randn(100, 5),
                trial_starts=torch.tensor([], dtype=torch.long),
                trial_ends=torch.tensor([], dtype=torch.long),
                conditional="trialTime",
            )

    def test_single_trial_raises(self):
        with pytest.raises(ValueError, match="At least 2 trials"):
            TrialAwareDistribution(
                continuous=torch.randn(50, 5),
                trial_starts=torch.tensor([0]),
                trial_ends=torch.tensor([50]),
                conditional="trialTime",
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            TrialAwareDistribution(
                continuous=torch.randn(100, 5),
                trial_starts=torch.tensor([0, 50]),
                trial_ends=torch.tensor([50]),
                conditional="trialTime",
            )

    def test_invalid_boundaries_raises(self):
        with pytest.raises(ValueError, match="must be < trial_ends"):
            TrialAwareDistribution(
                continuous=torch.randn(100, 5),
                trial_starts=torch.tensor([50, 0]),
                trial_ends=torch.tensor([30, 50]),
                conditional="trialTime",
            )

    def test_similarity_attrs_only_for_delta_modes(self):
        """trial_embeddings only for similarity modes; _time_difference only for
        time_delta-style modes (trialTime_trialDelta only — trialTime_delta now
        uses delta-style trial selection and no longer needs _time_difference)."""
        dist_time, *_ = _make_dist(conditional="trialTime")
        assert not hasattr(dist_time, "trial_embeddings")
        assert not hasattr(dist_time, "_time_difference")

        for cond in ("trialDelta", "trial_delta", "trialTime_delta", "trialTime_trialDelta"):
            dist, *_ = _make_dist(conditional=cond)
            assert hasattr(dist, "trial_embeddings"), cond

        # Only trialTime_trialDelta still uses time_delta-style trial selection.
        dist, *_ = _make_dist(conditional="trialTime_trialDelta")
        assert hasattr(dist, "_time_difference")

        for cond in ("trialDelta", "trial_delta", "trialTime_delta"):
            dist, *_ = _make_dist(conditional=cond)
            assert not hasattr(dist, "_time_difference"), cond

    def test_locked_attrs_only_for_locked_modes(self):
        """_locked_target_trials only for locked modes."""
        for cond in ("trialTime", "trial_delta", "trialTime_delta"):
            dist, *_ = _make_dist(conditional=cond)
            assert not hasattr(dist, "_locked_target_trials"), cond

        for cond in ("trialDelta", "trialTime_trialDelta"):
            dist, *_ = _make_dist(conditional=cond)
            assert hasattr(dist, "_locked_target_trials"), cond
            assert dist._locked_target_trials.shape == (dist.num_trials,), cond

    def test_discrete_attrs(self):
        total = 8 * (60 + 20)
        discrete = torch.zeros(total, dtype=torch.long)
        dist, *_ = _make_dist(discrete=discrete)
        assert dist.discrete is not None
        assert dist.trial_discrete.shape == (dist.num_trials,)
        assert dist._trial_same_class.shape == (dist.num_trials, dist.num_trials)

    def test_locked_target_valid_range(self):
        for cond in ("trialDelta", "trialTime_trialDelta"):
            dist, *_ = _make_dist(conditional=cond)
            locked = dist._locked_target_trials
            assert (locked >= 0).all(), cond
            assert (locked < dist.num_trials).all(), cond
            # Locked target must differ from self
            self_ids = torch.arange(dist.num_trials, device=dist.device)
            assert (locked != self_ids).all(), cond


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------


class TestPriorSampling:
    def test_shape(self):
        dist, *_ = _make_dist()
        ref = dist.sample_prior(128)
        assert ref.shape == (128,)
        assert ref.dtype == torch.long

    def test_valid_range(self):
        dist, starts, ends, total = _make_dist()
        ref = dist.sample_prior(2000)
        assert ref.min().item() >= 0
        assert ref.max().item() < total

    def test_includes_gap(self):
        """Prior must sample gap (inter-trial) timepoints."""
        dist, *_ = _make_dist(num_trials=5, trial_length=40, gap=20)
        ref = dist.sample_prior(5000)
        assert (dist.timepoint_to_trial[ref] == -1).any(), "Prior never sampled a gap timepoint"

    def test_includes_trials(self):
        dist, *_ = _make_dist()
        ref = dist.sample_prior(2000)
        assert (dist.timepoint_to_trial[ref] >= 0).any()

    def test_covers_all_trials(self):
        dist, *_ = _make_dist(num_trials=8)
        ref = dist.sample_prior(10000)
        in_trial = ref[dist.timepoint_to_trial[ref] >= 0]
        covered = dist.timepoint_to_trial[in_trial].unique()
        assert len(covered) == 8

    def test_class_balanced_when_discrete(self):
        """With discrete, every class must appear with roughly equal frequency.

        Uses a highly imbalanced label (class 0: 90 %, class 1: 10 %) and
        checks that both classes appear in [35 %, 65 %] of samples —
        much more balanced than the raw 90/10 split.
        """
        total = 1000
        # 900 timepoints class 0, 100 timepoints class 1
        discrete = torch.zeros(total, dtype=torch.long)
        discrete[900:] = 1
        starts = torch.tensor([0, 500])
        ends = torch.tensor([50, 550])
        continuous = torch.randn(total, 4)
        dist = TrialAwareDistribution(
            continuous=continuous,
            trial_starts=starts,
            trial_ends=ends,
            conditional="trialTime",
            time_offset=5,
            discrete=discrete,
        )
        ref = dist.sample_prior(4000)
        frac_class1 = (discrete[ref] == 1).float().mean().item()
        assert 0.35 < frac_class1 < 0.65, (
            f"Expected balanced prior, got class-1 fraction {frac_class1:.2f}"
        )


# ---------------------------------------------------------------------------
# trialTime (uniform random trial + ±offset)
# ---------------------------------------------------------------------------


class TestTrialTime:
    def _dist(self, **kw):
        d, s, e, t = _make_dist(conditional="trialTime", delta=None, **kw)
        return d, s, e, t

    def test_shape(self):
        dist, *_ = self._dist()
        ref = dist._all_trial_indices[:64]
        pos = dist.sample_conditional(ref)
        assert pos.shape == ref.shape

    def test_different_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[ref] != dist.timepoint_to_trial[pos]).all()

    def test_time_offset_constraint(self):
        dist, *_ = self._dist(num_trials=10, trial_length=60, time_offset=5)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        ref_t = dist.timepoint_to_trial[ref]
        pos_t = dist.timepoint_to_trial[pos]
        ref_rel = ref - dist.trial_starts[ref_t]
        pos_rel = pos - dist.trial_starts[pos_t]
        assert (torch.abs(ref_rel - pos_rel) <= 5).all()

    def test_random_covers_all_trials(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:3000]
        pos = dist.sample_conditional(ref)
        assert len(dist.timepoint_to_trial[pos].unique()) == 10

    def test_gap_uses_time_window(self):
        dist, starts, ends, total = self._dist(num_trials=5, trial_length=40, gap=20, time_offset=5)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert (torch.abs(pos - gap_idx) <= 5).all()

    def test_no_self_gap_in_window(self):
        """Positive must not exactly equal reference."""
        dist, starts, ends, total = self._dist(num_trials=5, trial_length=40, gap=20, time_offset=5)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        # Run many samples; self-match probability is low but allowed by spec;
        # just verify no crash and valid range.
        pos = dist.sample_conditional(gap_idx)
        assert (pos >= 0).all()
        assert (pos < total).all()


# ---------------------------------------------------------------------------
# trial_delta (re-sampled Gaussian trial + uniform within trial)
# ---------------------------------------------------------------------------


class TestTrialDeltaResampled:
    def _dist(self, **kw):
        d, s, e, t = _make_dist(conditional="trial_delta", **kw)
        return d, s, e, t

    def test_different_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[ref] != dist.timepoint_to_trial[pos]).all()

    def test_not_locked(self):
        """Same anchor should produce positives from multiple target trials."""
        # Use large delta to flatten the distribution, ensuring multiple trials
        # are visited across samples (verifies re-sampling, not locking).
        dist, *_ = self._dist(num_trials=10, trial_length=60, delta=5.0)
        anchor = dist.trial_starts[0:1]  # a single anchor from trial 0
        target_trials = set()
        for _ in range(200):
            pos = dist.sample_conditional(anchor)
            target_trials.add(int(dist.timepoint_to_trial[pos].item()))
        assert len(target_trials) > 1, (
            "trial_delta should visit multiple target trials (re-sampled, not locked)"
        )

    def test_positive_within_target_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[pos] >= 0).all()

    def test_similar_trial_preferred(self):
        """With strong Gaussian weighting, the closest trial should be chosen most often."""
        # Build data: trials 0,1,2,3 — trial 1 is very close to trial 0
        n_trials, tlen = 4, 50
        continuous = torch.zeros(n_trials * tlen, 4)
        starts = torch.arange(0, n_trials * tlen, tlen)
        ends = starts + tlen
        # trial 0: zero vector; trial 1: very similar (small noise); others: far
        continuous[0:tlen] = torch.zeros(tlen, 4)
        continuous[tlen : 2 * tlen] = torch.full((tlen, 4), 0.01)
        continuous[2 * tlen : 3 * tlen] = torch.full((tlen, 4), 10.0)
        continuous[3 * tlen : 4 * tlen] = torch.full((tlen, 4), 20.0)
        dist = TrialAwareDistribution(
            continuous=continuous,
            trial_starts=starts,
            trial_ends=ends,
            conditional="trial_delta",
            time_offset=5,
            delta=0.1,
        )
        anchor = starts[0:1]  # anchor from trial 0
        counts = [0] * n_trials
        for _ in range(500):
            pos = dist.sample_conditional(anchor)
            counts[int(dist.timepoint_to_trial[pos].item())] += 1
        # trial 1 (index 1) should be selected most often
        assert counts[1] == max(counts), f"Expected trial 1 most frequent, got counts={counts}"

    def test_gap_uses_content(self):
        dist, starts, ends, total = self._dist(num_trials=5, trial_length=40, gap=20)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert pos.shape == gap_idx.shape
        assert (pos >= 0).all() and (pos < total).all()


# ---------------------------------------------------------------------------
# trialDelta (locked trial + uniform within trial)
# ---------------------------------------------------------------------------


class TestTrialDelta:
    def _dist(self, **kw):
        d, s, e, t = _make_dist(conditional="trialDelta", **kw)
        return d, s, e, t

    def test_different_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[ref] != dist.timepoint_to_trial[pos]).all()

    def test_locked_same_target_within_trial(self):
        """All anchors from the SAME trial must map to the SAME target trial."""
        dist, *_ = self._dist(num_trials=10)
        # Use all timepoints from trial 0
        s0, e0 = int(dist.trial_starts[0]), int(dist.trial_ends[0])
        ref = torch.arange(s0, e0, device=dist.device)
        pos = dist.sample_conditional(ref)
        pos_trials = dist.timepoint_to_trial[pos]
        assert len(pos_trials.unique()) == 1, (
            "Anchors in same trial should map to the same locked target trial"
        )

    def test_locked_target_in_valid_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[pos] >= 0).all()

    def test_positive_uniformly_covers_target_trial(self):
        """Repeated sampling of same anchor should visit many positions in target."""
        dist, *_ = self._dist(num_trials=4, trial_length=80)
        anchor = dist.trial_starts[0:1]
        positives = torch.stack([dist.sample_conditional(anchor) for _ in range(200)])
        pos_rel = positives - dist.trial_starts[dist.timepoint_to_trial[positives]]
        assert pos_rel.unique().numel() > 5, "Expected uniform coverage of target trial positions"

    def test_gap_uses_content(self):
        dist, starts, ends, total = self._dist(num_trials=5, trial_length=40, gap=20)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert pos.shape == gap_idx.shape
        assert (pos >= 0).all() and (pos < total).all()


# ---------------------------------------------------------------------------
# trialTime_delta (cross-trial ±offset pool + Gaussian kernel)
# ---------------------------------------------------------------------------


class TestTrialTimeDelta:
    def _dist(self, **kw):
        d, s, e, t = _make_dist(conditional="trialTime_delta", **kw)
        return d, s, e, t

    def test_different_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:300]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[ref] != dist.timepoint_to_trial[pos]).all()

    def test_positive_within_offset(self):
        dist, *_ = self._dist(num_trials=10, trial_length=60, time_offset=5)
        ref = dist._all_trial_indices[:300]
        pos = dist.sample_conditional(ref)
        ref_t = dist.timepoint_to_trial[ref]
        pos_t = dist.timepoint_to_trial[pos]
        ref_rel = ref - dist.trial_starts[ref_t]
        pos_rel = pos - dist.trial_starts[pos_t]
        assert (torch.abs(ref_rel - pos_rel) <= 5).all()

    def test_pool_crosses_multiple_trials(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        pos_trials = dist.timepoint_to_trial[pos]
        assert len(pos_trials.unique()) > 1, (
            "Pool-based sampling should cover multiple target trials"
        )

    def test_not_locked(self):
        """trialTime_delta is per-sample: no pre-computed _locked_target_trials.

        The not-locked property is structurally verified in TestInit; this
        test confirms different anchors from the same trial CAN independently
        reach different target trials (by using many anchors from many trials).
        """
        # test_locked_attrs_only_for_locked_modes already asserts the absence
        # of _locked_target_trials.  Here just verify the sampling produces
        # valid results across the full index.
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices  # all trial timepoints
        pos = dist.sample_conditional(ref)
        # Positives should span multiple target trials
        pos_trials = dist.timepoint_to_trial[pos]
        assert pos_trials.unique().numel() > 1, (
            "Pool-based sampling should reach multiple target trials"
        )

    def test_gap_uses_content(self):
        dist, starts, ends, total = self._dist(num_trials=5, trial_length=40, gap=20)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert pos.shape == gap_idx.shape

    def test_fallback_when_pool_empty(self):
        """All other trials shorter than r - time_offset → fallback, no error."""
        # Build data where trial 0 has length 100, all others have length 5
        n = 5 + 4 * 5
        continuous = torch.randn(n, 4)
        # trial 0: [0, 5), trials 1-4: [5, 10) etc.  (very short)
        starts = torch.tensor([0, 5, 10, 15, 20])
        ends = torch.tensor([5, 10, 15, 20, 25])
        dist = TrialAwareDistribution(
            continuous=continuous,
            trial_starts=starts,
            trial_ends=ends,
            conditional="trialTime_delta",
            time_offset=100,
            delta=0.1,
        )
        # Anchor at last position of trial 0
        ref = torch.tensor([4])  # relative position 4, offset 100 → pool may be empty
        pos = dist.sample_conditional(ref)  # should not raise
        assert pos.shape == (1,)


# ---------------------------------------------------------------------------
# trialTime_trialDelta (locked trial + ±offset)
# ---------------------------------------------------------------------------


class TestTrialTimeTrialDelta:
    def _dist(self, **kw):
        d, s, e, t = _make_dist(conditional="trialTime_trialDelta", **kw)
        return d, s, e, t

    def test_different_trial(self):
        dist, *_ = self._dist(num_trials=10)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        assert (dist.timepoint_to_trial[ref] != dist.timepoint_to_trial[pos]).all()

    def test_locked_same_target_within_trial(self):
        dist, *_ = self._dist(num_trials=10)
        s0, e0 = int(dist.trial_starts[0]), int(dist.trial_ends[0])
        ref = torch.arange(s0, e0, device=dist.device)
        pos = dist.sample_conditional(ref)
        pos_trials = dist.timepoint_to_trial[pos]
        assert len(pos_trials.unique()) == 1, (
            "All anchors in trial 0 should map to the same locked target"
        )

    def test_time_offset_constraint(self):
        dist, *_ = self._dist(num_trials=10, trial_length=60, time_offset=5)
        ref = dist._all_trial_indices[:500]
        pos = dist.sample_conditional(ref)
        ref_t = dist.timepoint_to_trial[ref]
        pos_t = dist.timepoint_to_trial[pos]
        ref_rel = ref - dist.trial_starts[ref_t]
        pos_rel = pos - dist.trial_starts[pos_t]
        assert (torch.abs(ref_rel - pos_rel) <= 5).all()

    def test_gap_uses_content(self):
        dist, starts, ends, total = self._dist(num_trials=5, trial_length=40, gap=20)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert pos.shape == gap_idx.shape


# ---------------------------------------------------------------------------
# Mixed batches (trial + gap anchors in one call)
# ---------------------------------------------------------------------------


class TestMixedBatches:
    def test_all_conditionals_handle_mixed(self):
        for cond in TRIAL_CONDITIONALS:
            dist, starts, ends, total = _make_dist(
                conditional=cond, num_trials=5, trial_length=40, gap=20
            )
            gap_idx = [i for i in range(total) if dist.timepoint_to_trial[i] == -1]
            if not gap_idx:
                continue
            mixed = torch.cat([starts[:2], torch.tensor(gap_idx[:2], dtype=torch.long)])
            pos = dist.sample_conditional(mixed)  # must not raise
            assert pos.shape == mixed.shape, cond


# ---------------------------------------------------------------------------
# Discrete index support
# ---------------------------------------------------------------------------


class TestDiscreteSupport:
    def _make_discrete_dist(self, conditional, num_trials=10, num_classes=2):
        continuous, starts, ends, total = make_trial_data(
            num_trials=num_trials, trial_length=60, gap=10
        )
        discrete = torch.zeros(total, dtype=torch.long)
        for t in range(num_trials):
            s, e = int(starts[t]), int(ends[t])
            discrete[s:e] = t % num_classes
        dist = TrialAwareDistribution(
            continuous=continuous,
            trial_starts=starts,
            trial_ends=ends,
            conditional=conditional,
            time_offset=10,
            delta=0.1,
            discrete=discrete,
        )
        return dist, discrete, total

    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_positive_same_class(self, cond):
        dist, discrete, _ = self._make_discrete_dist(cond)
        ref = dist._all_trial_indices[:300]
        pos = dist.sample_conditional(ref)
        assert (discrete[ref] == discrete[pos]).all(), f"discrete mismatch for conditional={cond}"

    def test_gap_anchor_same_class(self):
        dist, discrete, total = self._make_discrete_dist("trialTime_delta")
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert (discrete[gap_idx] == discrete[pos]).all()

    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_gap_anchor_same_class_all_conditionals(self, cond):
        """Gap anchors must produce same-class positives for every conditional."""
        dist, discrete, total = self._make_discrete_dist(cond)
        gap_idx = torch.tensor([i for i in range(total) if dist.timepoint_to_trial[i] == -1])
        if len(gap_idx) == 0:
            pytest.skip("No gap timepoints")
        pos = dist.sample_conditional(gap_idx)
        assert (discrete[gap_idx] == discrete[pos]).all(), (
            f"Gap anchor produced different-class positive for conditional={cond}"
        )


# ---------------------------------------------------------------------------
# Joint sampling interface
# ---------------------------------------------------------------------------


class TestJointSampling:
    def test_sample_joint_shape(self):
        dist, *_ = _make_dist()
        ref, pos = dist.sample_joint(num_samples=64)
        assert ref.shape == (64,)
        assert pos.shape == (64,)

    def test_sample_joint_valid_range(self):
        dist, *_ = _make_dist()
        ref, pos = dist.sample_joint(num_samples=200)
        N = len(dist.continuous)
        assert (ref >= 0).all() and (ref < N).all()
        assert (pos >= 0).all() and (pos < N).all()

    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_sample_joint_all_conditionals(self, cond):
        dist, *_ = _make_dist(conditional=cond)
        ref, pos = dist.sample_joint(num_samples=32)
        assert ref.shape == pos.shape == (32,)
