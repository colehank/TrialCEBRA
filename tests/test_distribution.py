"""Tests for TrialAwareDistribution — new 3-conditional design.

Conditionals:      "time" | "delta" | "time_delta"
Sampling params:   sample_fix_trial (bool, only for delta/time_delta)
                   sample_exclude_intrial (bool, all conditionals)
y shapes:          (ntrial, nd) for delta; (ntrial, ntime, nd) for time_delta; None for time
"""

import pytest
import torch
from conftest import make_epoch_data

from trial_cebra.distribution import TRIAL_CONDITIONALS, TrialAwareDistribution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NTRIAL, NTIME, NNEURO, ND = 8, 50, 32, 10


def _dist(
    conditional="time",
    sample_fix_trial=False,
    sample_exclude_intrial=True,
    time_offsets=10,
    delta=0.1,
    seed=0,
    **kw,
):
    _, y2d, y3d = make_epoch_data(ntrial=NTRIAL, ntime=NTIME, nneuro=NNEURO, nd=ND)
    y = None
    if conditional == "delta":
        y = y2d
    elif conditional == "time_delta":
        y = y3d
    return TrialAwareDistribution(
        ntrial=NTRIAL,
        ntime=NTIME,
        conditional=conditional,
        y=y,
        sample_fix_trial=sample_fix_trial,
        sample_exclude_intrial=sample_exclude_intrial,
        time_offsets=time_offsets,
        delta=delta,
        seed=seed,
        **kw,
    )


def _all_trial_idx():
    return torch.arange(NTRIAL * NTIME)


# ---------------------------------------------------------------------------
# Init & validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_time_no_y_ok(self):
        d = _dist("time")
        assert d.conditional == "time"

    def test_delta_2d_y_ok(self):
        d = _dist("delta")
        assert d.conditional == "delta"
        assert d.trial_emb.shape == (NTRIAL, ND)

    def test_time_delta_3d_y_ok(self):
        d = _dist("time_delta")
        assert d.conditional == "time_delta"
        assert d.trial_emb.shape == (NTRIAL, ND)

    def test_delta_3d_y_ok(self):
        """3-D y is accepted by delta (enables class-conditional trial_emb)."""
        _, _, y3d = make_epoch_data()
        d = TrialAwareDistribution(ntrial=NTRIAL, ntime=NTIME, conditional="delta", y=y3d)
        assert d.conditional == "delta"
        assert d.trial_emb.shape == (NTRIAL, ND)  # mean-aggregated to per-trial

    def test_delta_3d_y_wrong_ntime_raises(self):
        _, _, y3d = make_epoch_data()
        bad = y3d[:, : NTIME - 1, :]  # wrong ntime
        with pytest.raises(ValueError, match="ntime"):
            TrialAwareDistribution(ntrial=NTRIAL, ntime=NTIME, conditional="delta", y=bad)

    def test_delta_no_y_raises(self):
        with pytest.raises(ValueError, match="ntrial"):
            TrialAwareDistribution(ntrial=NTRIAL, ntime=NTIME, conditional="delta", y=None)

    def test_time_delta_2d_y_raises(self):
        _, y2d, _ = make_epoch_data()
        with pytest.raises(ValueError, match="3-D"):
            TrialAwareDistribution(ntrial=NTRIAL, ntime=NTIME, conditional="time_delta", y=y2d)

    def test_time_delta_no_y_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            TrialAwareDistribution(ntrial=NTRIAL, ntime=NTIME, conditional="time_delta", y=None)

    def test_unknown_conditional_raises(self):
        with pytest.raises(ValueError, match="Unknown conditional"):
            TrialAwareDistribution(ntrial=NTRIAL, ntime=NTIME, conditional="bad", y=None)

    def test_single_trial_raises(self):
        _, y2d, _ = make_epoch_data(ntrial=1)
        with pytest.raises(ValueError, match="At least 2"):
            TrialAwareDistribution(ntrial=1, ntime=NTIME, conditional="delta", y=y2d)

    def test_known_conditionals_set(self):
        assert TRIAL_CONDITIONALS == {"time", "delta", "time_delta"}

    def test_sample_fix_trial_creates_locked_map_delta(self):
        d = _dist("delta", sample_fix_trial=True)
        assert hasattr(d, "_locked_target_trials")
        assert d._locked_target_trials.shape == (NTRIAL,)

    def test_sample_fix_trial_creates_locked_map_time_delta(self):
        d = _dist("time_delta", sample_fix_trial=True)
        assert hasattr(d, "_locked_target_trials")
        assert d._locked_target_trials.shape == (NTRIAL,)

    def test_no_sample_fix_trial_no_locked_map(self):
        for cond in ("time", "delta", "time_delta"):
            d = _dist(cond, sample_fix_trial=False)
            assert not hasattr(d, "_locked_target_trials"), cond

    def test_sample_fix_trial_ignored_for_time(self):
        d = _dist("time", sample_fix_trial=True)
        assert not hasattr(d, "_locked_target_trials")

    def test_locked_target_valid_range(self):
        for cond in ("delta", "time_delta"):
            d = _dist(cond, sample_fix_trial=True)
            locked = d._locked_target_trials
            assert (locked >= 0).all(), cond
            assert (locked < NTRIAL).all(), cond

    def test_locked_target_differs_from_self(self):
        for cond in ("delta", "time_delta"):
            d = _dist(cond, sample_fix_trial=True)
            self_ids = torch.arange(NTRIAL)
            assert (d._locked_target_trials != self_ids).all(), cond

    def test_time_delta_stores_y_flat(self):
        d = _dist("time_delta")
        assert hasattr(d, "_y_flat")
        assert d._y_flat.shape == (NTRIAL * NTIME, ND)


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------


class TestSamplePrior:
    def test_shape(self):
        d = _dist()
        ref = d.sample_prior(128)
        assert ref.shape == (128,)
        assert ref.dtype == torch.long

    def test_range(self):
        d = _dist()
        ref = d.sample_prior(2000)
        assert ref.min().item() >= 0
        assert ref.max().item() < NTRIAL * NTIME

    def test_covers_all_trials(self):
        d = _dist()
        ref = d.sample_prior(5000)
        trial_ids = ref // NTIME
        assert trial_ids.unique().numel() == NTRIAL


# ---------------------------------------------------------------------------
# "time" conditional
# ---------------------------------------------------------------------------


class TestTime:
    def test_shape(self):
        d = _dist("time")
        ref = torch.arange(64)
        pos = d.sample_conditional(ref)
        assert pos.shape == (64,)
        assert pos.dtype == torch.long

    def test_positive_in_different_trial(self):
        d = _dist("time", seed=42)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        ref_trial = ref // NTIME
        pos_trial = pos // NTIME
        assert (ref_trial != pos_trial).all()

    def test_positive_within_time_offset(self):
        time_offsets = 8
        d = _dist("time", time_offsets=time_offsets)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        ref_rel = ref % NTIME
        pos_rel = pos % NTIME
        assert (torch.abs(pos_rel.long() - ref_rel.long()) <= time_offsets).all()

    def test_positive_in_bounds(self):
        d = _dist("time")
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (pos >= 0).all()
        assert (pos < NTRIAL * NTIME).all()


# ---------------------------------------------------------------------------
# "delta" conditional
# ---------------------------------------------------------------------------


class TestDelta:
    def test_shape(self):
        d = _dist("delta")
        ref = torch.arange(64)
        pos = d.sample_conditional(ref)
        assert pos.shape == (64,)

    def test_positive_in_different_trial(self):
        d = _dist("delta", seed=7)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        ref_trial = ref // NTIME
        pos_trial = pos // NTIME
        assert (ref_trial != pos_trial).all()

    def test_positive_in_bounds(self):
        d = _dist("delta")
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (pos >= 0).all()
        assert (pos < NTRIAL * NTIME).all()

    def test_similar_trial_selected(self):
        """With small delta, nearest trial (in y-space) is almost always selected."""
        _, y2d, _ = make_epoch_data(ntrial=NTRIAL, ntime=NTIME, nd=ND)
        # Make trial 2 and trial 5 very similar
        y2d[5] = y2d[2] + 1e-4 * torch.randn(ND)
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y2d,
            sample_fix_trial=False,
            delta=1e-6,
            seed=0,
        )
        # Anchors from trial 2; expect positive mostly in trial 5
        ref = torch.arange(2 * NTIME, 3 * NTIME)  # trial 2
        pos = d.sample_conditional(ref)
        pos_trial = pos // NTIME
        frac_trial5 = (pos_trial == 5).float().mean().item()
        assert frac_trial5 > 0.8, f"Expected target ≈ trial 5, got fraction {frac_trial5:.2f}"

    def test_sample_fix_trial_deterministic(self):
        d = _dist("delta", sample_fix_trial=True, seed=0)
        ref = torch.arange(NTRIAL * NTIME)
        pos1 = d.sample_conditional(ref)
        pos2 = d.sample_conditional(ref)
        # With sample_fix_trial, target TRIAL is always the same; within-trial is still random
        assert (pos1 // NTIME == pos2 // NTIME).all()

    def test_no_sample_fix_trial_can_vary(self):
        """Without sample_fix_trial, different seeds produce different trial pairings."""
        ref = torch.arange(NTRIAL * NTIME)
        d1 = _dist("delta", sample_fix_trial=False, seed=0)
        d2 = _dist("delta", sample_fix_trial=False, seed=99)
        pos1 = d1.sample_conditional(ref) // NTIME
        pos2 = d2.sample_conditional(ref) // NTIME
        assert not (pos1 == pos2).all(), "Different seeds should occasionally differ"

    def test_positive_free_in_trial(self):
        """Positive is not constrained to ±time_offsets of anchor's relative pos."""
        d = _dist("delta", sample_fix_trial=True, time_offsets=2, seed=0)
        ref = torch.zeros(500, dtype=torch.long)  # anchor at rel_pos=0 in trial 0
        pos = d.sample_conditional(ref)
        pos_trial = pos // NTIME
        pos_rel = pos % NTIME
        # Consider only positives from a single target trial
        target_t = pos_trial[0].item()
        rel_pos = pos_rel[pos_trial == target_t]
        # Should cover positions well beyond time_offsets=2
        assert rel_pos.max().item() > 2, "delta positive should be free within trial"


# ---------------------------------------------------------------------------
# "time_delta" conditional
# ---------------------------------------------------------------------------


class TestTimeDelta:
    def test_shape(self):
        d = _dist("time_delta")
        ref = torch.arange(64)
        pos = d.sample_conditional(ref)
        assert pos.shape == (64,)

    def test_positive_in_different_trial(self):
        d = _dist("time_delta", seed=3)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        ref_trial = ref // NTIME
        pos_trial = pos // NTIME
        assert (ref_trial != pos_trial).all()

    def test_positive_in_bounds(self):
        d = _dist("time_delta")
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (pos >= 0).all()
        assert (pos < NTRIAL * NTIME).all()

    def test_positive_within_time_offset(self):
        time_offsets = 8
        d = _dist("time_delta", time_offsets=time_offsets)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        ref_rel = ref % NTIME
        pos_rel = pos % NTIME
        assert (torch.abs(pos_rel.long() - ref_rel.long()) <= time_offsets).all()

    def test_sample_fix_trial_deterministic(self):
        d = _dist("time_delta", sample_fix_trial=True, seed=0)
        ref = torch.arange(NTRIAL * NTIME)
        pos1 = d.sample_conditional(ref)
        pos2 = d.sample_conditional(ref)
        assert (pos1 // NTIME == pos2 // NTIME).all()

    def test_no_sample_fix_trial_can_vary(self):
        ref = torch.arange(NTRIAL * NTIME)
        d1 = _dist("time_delta", sample_fix_trial=False, seed=0)
        d2 = _dist("time_delta", sample_fix_trial=False, seed=99)
        pos1 = d1.sample_conditional(ref) // NTIME
        pos2 = d2.sample_conditional(ref) // NTIME
        assert not (pos1 == pos2).all()

    def test_similar_trial_and_time_selected(self):
        """With small delta, joint argmin selects the (trial, t) with closest y."""
        _, _, y3d = make_epoch_data(ntrial=NTRIAL, ntime=NTIME, nd=ND)
        # Make trial 5 at all timepoints very close to trial 2
        y3d[5] = y3d[2] + 1e-4 * torch.randn(NTIME, ND)
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="time_delta",
            y=y3d,
            sample_fix_trial=False,
            delta=1e-6,
            seed=0,
        )
        # Anchors from trial 2; expect positives mostly in trial 5
        ref = torch.arange(2 * NTIME, 3 * NTIME)
        pos = d.sample_conditional(ref)
        pos_trial = pos // NTIME
        frac = (pos_trial == 5).float().mean().item()
        assert frac > 0.8, f"Expected target ≈ trial 5, got {frac:.2f}"

    def test_fix_trial_within_trial_argmin(self):
        """fix_trial=True: locked trial, within-trial position tracks y-argmin."""
        _, _, y3d = make_epoch_data(ntrial=NTRIAL, ntime=NTIME, nd=ND)
        # Make trial 5 very close to trial 2 so it gets locked
        y3d[5] = y3d[2] + 1e-4 * torch.randn(NTIME, ND)
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="time_delta",
            y=y3d,
            sample_fix_trial=True,
            delta=1e-6,
            seed=0,
        )
        ref = torch.arange(2 * NTIME, 3 * NTIME)
        pos1 = d.sample_conditional(ref)
        pos2 = d.sample_conditional(ref)
        # Locked trial: same target trial across calls
        assert (pos1 // NTIME == pos2 // NTIME).all()
        # Target trial is 5 (most similar to trial 2)
        assert (pos1 // NTIME == 5).all()


# ---------------------------------------------------------------------------
# sample_exclude_intrial
# ---------------------------------------------------------------------------


class TestSampleExcludeIntrial:
    def test_default_excludes_own_trial_time(self):
        d = _dist("time", sample_exclude_intrial=True, seed=0)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (ref // NTIME != pos // NTIME).all()

    def test_default_excludes_own_trial_delta(self):
        d = _dist("delta", sample_exclude_intrial=True, seed=0)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (ref // NTIME != pos // NTIME).all()

    def test_default_excludes_own_trial_time_delta(self):
        d = _dist("time_delta", sample_exclude_intrial=True, seed=0)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (ref // NTIME != pos // NTIME).all()

    def test_allow_intrial_can_sample_own_trial_time(self):
        """With sample_exclude_intrial=False, own-trial positives eventually appear."""
        d = _dist("time", sample_exclude_intrial=False, seed=0)
        ref = torch.zeros(2000, dtype=torch.long)  # all anchors in trial 0
        pos = d.sample_conditional(ref)
        pos_trial = pos // NTIME
        assert (pos_trial == 0).any(), "Expected some same-trial positives"

    def test_allow_intrial_can_sample_own_trial_delta(self):
        d = _dist("delta", sample_exclude_intrial=False, seed=0)
        ref = torch.zeros(2000, dtype=torch.long)  # all anchors in trial 0
        pos = d.sample_conditional(ref)
        pos_trial = pos // NTIME
        assert (pos_trial == 0).any(), "Expected some same-trial positives"

    def test_allow_intrial_can_sample_own_trial_time_delta(self):
        d = _dist("time_delta", sample_exclude_intrial=False, seed=0)
        ref = torch.zeros(2000, dtype=torch.long)  # all anchors in trial 0
        pos = d.sample_conditional(ref)
        pos_trial = pos // NTIME
        assert (pos_trial == 0).any(), "Expected some same-trial positives"

    def test_allow_intrial_still_in_bounds(self):
        for cond in ("time", "delta", "time_delta"):
            d = _dist(cond, sample_exclude_intrial=False, seed=0)
            ref = torch.arange(NTRIAL * NTIME)
            pos = d.sample_conditional(ref)
            assert (pos >= 0).all(), cond
            assert (pos < NTRIAL * NTIME).all(), cond

    def test_delta_excl_false_no_self_dominance_high_dim(self):
        """Without y_discrete, delta+excl=False used to deterministically pick
        self trial in high-dim because argmin can't escape self-similarity when
        Gaussian noise is mostly orthogonal to inter-trial offsets. Stochastic
        Gumbel-max sampling (T=delta) must diversify target trials when delta
        is comparable to the inter-trial scale.
        """
        # High-dim trial embeddings on unit hypersphere (mimics user's L2-normalized
        # stim embeddings where argmin always picks self). delta=1.0 matches the
        # natural inter-trial distance scale (~sqrt(2)) so stochastic selection
        # has meaningful spread.
        ntrial, ntime, nd = 20, 10, 256
        torch.manual_seed(0)
        emb = torch.randn(ntrial, nd)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # unit-norm
        d = TrialAwareDistribution(
            ntrial=ntrial,
            ntime=ntime,
            conditional="delta",
            y=emb,
            sample_exclude_intrial=False,
            delta=1.0,
            seed=0,
        )
        ref = torch.arange(ntrial * ntime)
        pos = d.sample_conditional(ref)
        ref_trial = ref // ntime
        pos_trial = pos // ntime
        same_trial_rate = (ref_trial == pos_trial).float().mean().item()
        # Old (argmin) behavior: ≈1.0 (self always wins).
        # New (Gumbel-max with T=delta): self preferred but does not dominate.
        assert same_trial_rate < 0.5, (
            f"delta+excl=False in high-dim should diversify target trials; "
            f"got self-trial rate {same_trial_rate:.3f} (≈1.0 means "
            f"argmin self-dominance bug is back)"
        )


# ---------------------------------------------------------------------------
# y_discrete: class-balanced prior + same-class conditional
# ---------------------------------------------------------------------------


def _make_disc_y(ntrial=NTRIAL, ntime=NTIME):
    """Binary labels: first half of each trial = class 0, second half = class 1."""
    y = torch.zeros(ntrial * ntime, dtype=torch.long)
    half = ntime // 2
    for t in range(ntrial):
        y[t * ntime + half : t * ntime + ntime] = 1
    return y


class TestDiscrete:
    def _dist_disc(self, conditional="time", **kw):
        y_disc = _make_disc_y()
        return _dist(conditional, y_discrete=y_disc, **kw)

    def test_prior_class_balanced(self):
        """Class-balanced prior should assign ~50% to each binary class."""
        d = self._dist_disc("time", seed=42)
        y_disc = _make_disc_y()
        ref = d.sample_prior(10_000)
        frac_0 = (y_disc[ref] == 0).float().mean().item()
        assert 0.45 < frac_0 < 0.55, f"Class 0 fraction: {frac_0:.3f}"

    def test_prior_uniform_mode_matches_frequency(self):
        """sample_prior='uniform' should reproduce the empirical class frequency.

        Build an imbalanced label (20% class 0 / 80% class 1).  In 'uniform'
        mode the anchor distribution must reflect this 20/80 split, not the
        50/50 split that 'balanced' would impose.
        """
        ntrial, ntime = NTRIAL, NTIME
        y_disc = torch.ones(ntrial * ntime, dtype=torch.long)
        cutoff = int(0.2 * ntime)  # first 20% of each trial = class 0
        for t in range(ntrial):
            y_disc[t * ntime : t * ntime + cutoff] = 0
        d = _dist("time", y_discrete=y_disc, sample_prior="uniform", seed=7)
        ref = d.sample_prior(20_000)
        frac_0 = (y_disc[ref] == 0).float().mean().item()
        assert 0.18 < frac_0 < 0.22, f"Class 0 fraction (uniform): {frac_0:.3f}"

    def test_prior_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="sample_prior"):
            TrialAwareDistribution(
                ntrial=NTRIAL, ntime=NTIME, conditional="time", sample_prior="weighted"
            )

    def test_prior_shape_and_range(self):
        d = self._dist_disc("time")
        ref = d.sample_prior(256)
        assert ref.shape == (256,)
        assert ref.min().item() >= 0
        assert ref.max().item() < NTRIAL * NTIME

    def test_same_class_conditional_time(self):
        d = self._dist_disc("time", seed=1)
        y_disc = _make_disc_y()
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (y_disc[pos] == y_disc[ref]).all()

    def test_same_class_conditional_delta(self):
        d = self._dist_disc("delta", seed=2)
        y_disc = _make_disc_y()
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (y_disc[pos] == y_disc[ref]).all()

    def test_same_class_conditional_time_delta(self):
        d = self._dist_disc("time_delta", seed=3)
        y_disc = _make_disc_y()
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (y_disc[pos] == y_disc[ref]).all()

    def test_same_class_fix_trial_time_delta(self):
        d = self._dist_disc("time_delta", sample_fix_trial=True, seed=4)
        y_disc = _make_disc_y()
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (y_disc[pos] == y_disc[ref]).all()

    def test_time_delta_fix_trial_trial_selection_is_class_constrained(self):
        """time_delta fix_trial=True + y_discrete: the LOCKED TARGET TRIAL must
        belong to the same-class partition as the anchor trial.

        Use per-trial constant class labels so trial membership is unambiguous.
        """
        ntrial, ntime, nd = 10, 20, 4
        # First 5 trials = class 0, last 5 = class 1 (constant within each trial)
        y_disc = torch.zeros(ntrial * ntime, dtype=torch.long)
        for t in range(5, ntrial):
            y_disc[t * ntime : (t + 1) * ntime] = 1
        y3d = torch.randn(ntrial, ntime, nd)

        d = TrialAwareDistribution(
            ntrial=ntrial,
            ntime=ntime,
            conditional="time_delta",
            y=y3d,
            y_discrete=y_disc,
            sample_fix_trial=True,
            sample_exclude_intrial=True,
            time_offsets=3,
            seed=42,
        )
        assert hasattr(d, "_locked_target_trials_per_class"), (
            "expected per-class locked trials for time_delta + y_discrete"
        )

        ref = torch.arange(ntrial * ntime).repeat(10)
        pos = d.sample_conditional(ref)

        pos_trial = pos // ntime
        ref_class = y_disc[ref]
        pos_class = y_disc[pos]

        assert (pos_class == ref_class).all(), "same-class timepoint constraint violated"
        pos_trial_class = y_disc[pos_trial * ntime]  # constant-within-trial
        assert (pos_trial_class == ref_class).all(), (
            "time_delta fix_trial chose a wrong-class locked trial"
        )

    def test_cross_trial_still_enforced_with_discrete(self):
        """Class constraint must not weaken cross-trial exclusion."""
        d = self._dist_disc("time", sample_exclude_intrial=True, seed=5)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        assert (ref // NTIME != pos // NTIME).all()

    def test_time_trial_selection_is_class_constrained(self):
        """time + y_discrete: the selected TARGET TRIAL must have >=1 same-class
        timepoint in the ±time_offsets window, not just the final timepoint.

        Use per-trial constant class labels so it is easy to check: anchors
        from class-0 trials must land on class-0 trials, and vice versa.
        """
        ntrial, ntime = 10, 20
        # First 5 trials = class 0, last 5 = class 1 (constant within each trial)
        y_disc = torch.zeros(ntrial * ntime, dtype=torch.long)
        for t in range(5, ntrial):
            y_disc[t * ntime : (t + 1) * ntime] = 1

        d = TrialAwareDistribution(
            ntrial=ntrial,
            ntime=ntime,
            conditional="time",
            y_discrete=y_disc,
            sample_exclude_intrial=True,
            time_offsets=3,
            seed=99,
        )
        # Sample many times to reduce flakiness
        ref = torch.arange(ntrial * ntime).repeat(20)
        pos = d.sample_conditional(ref)

        pos_trial = pos // ntime
        ref_class = y_disc[ref]
        pos_class = y_disc[pos]

        # Final timepoint must be same class
        assert (pos_class == ref_class).all(), "same-class timepoint constraint violated"
        # Target trial must also be from the same-class partition
        pos_trial_class = y_disc[pos_trial * ntime]  # class of that trial (constant)
        assert (pos_trial_class == ref_class).all(), (
            "time trial selection chose a wrong-class trial"
        )

    def test_discrete_wrong_length_raises(self):
        y_disc = torch.zeros(NTRIAL * NTIME + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="ntrial\\*ntime"):
            TrialAwareDistribution(
                ntrial=NTRIAL, ntime=NTIME, conditional="time", y_discrete=y_disc
            )

    def test_no_discrete_uniform_prior(self):
        """Without y_discrete, prior is uniform (all trials covered)."""
        d = _dist("time")
        ref = d.sample_prior(5000)
        trial_ids = ref // NTIME
        assert trial_ids.unique().numel() == NTRIAL


# ---------------------------------------------------------------------------
# Class-conditional delta trial selection (discrete-first principle)
# ---------------------------------------------------------------------------


def _binary_per_tp_disc(ntrial=NTRIAL, ntime=NTIME, pre_len=None):
    """Per-timepoint discrete: class 0 first half, class 1 second half (per trial)."""
    if pre_len is None:
        pre_len = ntime // 2
    yd = torch.zeros(ntrial * ntime, dtype=torch.long)
    for t in range(ntrial):
        yd[t * ntime + pre_len : (t + 1) * ntime] = 1
    return yd


def _per_trial_disc(ntrial=NTRIAL, ntime=NTIME, n_classes=2, seed=0):
    """Per-trial discrete: each trial constant class, classes vary across trials."""
    g = torch.Generator().manual_seed(seed)
    classes = torch.randint(0, n_classes, (ntrial,), generator=g)
    yd = classes.repeat_interleave(ntime)
    return yd, classes


def _make_3d_y_with_class_structure(ntrial=NTRIAL, ntime=NTIME, nd=ND, pre_len=None):
    """3-D y where pre-stim half is constant gray-like emb (same across trials)
    and post-stim half is per-trial-distinct stim emb. Mimics user's data."""
    if pre_len is None:
        pre_len = ntime // 2
    g = torch.Generator().manual_seed(0)
    gray = torch.randn(nd, generator=g)
    stims = torch.randn(ntrial, nd, generator=g)
    y = torch.zeros(ntrial, ntime, nd)
    y[:, :pre_len, :] = gray.view(1, 1, nd)
    y[:, pre_len:, :] = stims.view(ntrial, 1, nd)
    return y, gray, stims


class TestClassConditionalDelta:
    """Verify discrete-first principle in delta path under all 4 dim combos."""

    def test_mode_per_trial_disc(self):
        """1-D-style discrete (per-trial) + 2-D y → trial_class set, _disc_mode=per_trial."""
        from trial_cebra.distribution import _DISC_MODE_PER_TRIAL

        yd, classes = _per_trial_disc()
        d = _dist("delta", y_discrete=yd, delta=0.5)
        assert d._disc_mode == _DISC_MODE_PER_TRIAL
        assert d._trial_class.shape == (NTRIAL,)
        assert torch.equal(d._trial_class, classes)

    def test_mode_per_tp_3d(self):
        """2-D-style discrete (per-timepoint) + 3-D y → trial_emb_per_class built."""
        from trial_cebra.distribution import _DISC_MODE_PER_TP_3D

        yd = _binary_per_tp_disc()
        y3d, gray, stims = _make_3d_y_with_class_structure()
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.5,
            seed=0,
        )
        assert d._disc_mode == _DISC_MODE_PER_TP_3D
        assert d._trial_emb_per_class.shape == (2, NTRIAL, ND)
        # Class 0 emb per trial == gray (constant across trials)
        assert torch.allclose(
            d._trial_emb_per_class[0], gray.view(1, ND).expand(NTRIAL, ND), atol=1e-5
        )
        # Class 1 emb per trial == stims (distinct per trial)
        assert torch.allclose(d._trial_emb_per_class[1], stims, atol=1e-5)

    def test_mode_per_tp_2d_auto_broadcasts_to_mode_b(self):
        """Per-timepoint discrete + 2-D y → auto-broadcast to 3-D, no warning, Mode B."""
        from trial_cebra.distribution import _DISC_MODE_PER_TP_3D

        yd = _binary_per_tp_disc()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            d = _dist("delta", y_discrete=yd, delta=0.5)
        assert d._disc_mode == _DISC_MODE_PER_TP_3D

    def test_per_trial_disc_target_is_same_class(self):
        """Mode A: target trial must always have same class as anchor's trial."""
        # Balanced 2-class per-trial assignment (each class has ≥2 trials)
        classes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        yd = classes.repeat_interleave(NTIME)
        d = _dist("delta", y_discrete=yd, delta=0.5, sample_exclude_intrial=True, seed=1)
        ref = torch.arange(NTRIAL * NTIME)
        pos = d.sample_conditional(ref)
        ref_trial = ref // NTIME
        pos_trial = pos // NTIME
        assert (classes[ref_trial] == classes[pos_trial]).all()
        # cross-trial enforced
        assert (ref_trial != pos_trial).all()

    def test_per_trial_disc_singleton_class_fallback(self):
        """Mode A: if anchor's class has only 1 trial, excl=True should fall back
        to allowing self (rather than returning a wrong-class target)."""
        classes = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])  # class 1 singleton
        yd = classes.repeat_interleave(NTIME)
        d = _dist("delta", y_discrete=yd, delta=0.5, sample_exclude_intrial=True, seed=0)
        # anchor in trial 7 (only class-1 trial) → must pick trial 7 itself (fallback)
        ref = torch.tensor([7 * NTIME])
        pos = d.sample_conditional(ref)
        assert (pos // NTIME) == 7
        assert classes[7] == classes[pos // NTIME]

    def test_per_tp_3d_pre_anchor_diverse_target_trials(self):
        """Mode B with tied class-c emb (pre-stim gray): pre anchor's target trial
        should be diverse cross-trial (stochastic tiebreak), not deterministic.
        """
        yd = _binary_per_tp_disc()
        y3d, _, _ = _make_3d_y_with_class_structure()
        # Use delta-disc mode B and check pre-anchor target diversity
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.5,
            sample_exclude_intrial=True,
            seed=42,
        )
        # Sample many pre-stim anchors (rel < ntime//2)
        pre_anchors = torch.tensor(
            [t * NTIME + r for t in range(NTRIAL) for r in range(NTIME // 2)]
        )
        # Repeat sampling to amortize randomness
        all_pos_trials = []
        for _ in range(5):
            pos = d.sample_conditional(pre_anchors)
            all_pos_trials.append((pos // NTIME).tolist())
        unique_pos_trials = set()
        for lst in all_pos_trials:
            unique_pos_trials.update(lst)
        # Must use more than 1 target trial (i.e., not deterministically pinned to first)
        assert len(unique_pos_trials) > 1, (
            f"pre-anchor target trials should be diverse via stochastic tiebreak, "
            f"got only {unique_pos_trials}"
        )
        # And same-class constraint still holds
        ref0 = pre_anchors
        pos0 = d.sample_conditional(ref0)
        yd_cpu = d._y_discrete.cpu()
        assert (yd_cpu[pos0] == 0).all()

    def test_per_tp_3d_post_anchor_picks_similar_stim(self):
        """Mode B post anchor (stim-distinct trials): target trial should be the
        one whose stim is most similar to anchor's stim. With small noise and
        excl=True, target should be one of the closer trials."""
        yd = _binary_per_tp_disc()
        y3d, _, stims = _make_3d_y_with_class_structure()
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.01,  # tiny noise → near-deterministic
            sample_exclude_intrial=True,
            seed=0,
        )
        # post anchor in trial 0 → target should be trial whose stim is closest to stims[0]
        # (excluding trial 0 itself)
        from torch import cdist

        d_stim = cdist(stims[:1], stims).squeeze(0)  # (NTRIAL,)
        d_stim[0] = float("inf")
        expected_top3 = torch.topk(-d_stim, k=3).indices.tolist()
        post_anchor_trial0 = torch.tensor([NTIME - 1])  # last frame of trial 0 (post-stim)
        # repeat sampling and check majority lands in top-3 nearest
        targets = []
        for _ in range(20):
            pos = d.sample_conditional(post_anchor_trial0)
            targets.append((pos // NTIME).item())
        in_top3 = sum(t in expected_top3 for t in targets)
        assert in_top3 >= 15, f"Expected most picks in top-3 nearest stims, got {in_top3}/20"

    def test_locked_targets_per_class_shape(self):
        """fix_trial=True + class-cond delta → locked_target_trials_per_class (n_classes, ntrial)."""
        yd = _binary_per_tp_disc()
        y3d, _, _ = _make_3d_y_with_class_structure()
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.5,
            sample_fix_trial=True,
            sample_exclude_intrial=True,
            seed=0,
        )
        assert hasattr(d, "_locked_target_trials_per_class")
        assert d._locked_target_trials_per_class.shape == (2, NTRIAL)
        # No fallback class-agnostic locked targets
        assert not hasattr(d, "_locked_target_trials")

    def test_locked_targets_class_dependent(self):
        """fix_trial=True: the same ref_trial under different anchor_class should
        be allowed to map to different targets (proves per-class precompute)."""
        yd = _binary_per_tp_disc()
        y3d, _, _ = _make_3d_y_with_class_structure()
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.5,
            sample_fix_trial=True,
            sample_exclude_intrial=True,
            seed=0,
        )
        # Same ref_trial, different class → potentially different locked target
        targets_per_class = d._locked_target_trials_per_class
        # Class 0 targets: arbitrary (gray emb tied → any) ; Class 1 targets: stim-similar
        # Verify they're not identical (would indicate class index ignored)
        assert not torch.equal(targets_per_class[0], targets_per_class[1])

    def test_no_discrete_no_class_state(self):
        """Without y_discrete, none of the class-conditional state should exist."""
        from trial_cebra.distribution import _DISC_MODE_NONE

        d = _dist("delta")
        assert d._disc_mode == _DISC_MODE_NONE
        assert not hasattr(d, "_trial_emb_per_class")
        assert not hasattr(d, "_trial_class")
        assert not hasattr(d, "_locked_target_trials_per_class")

    def test_per_tp_3d_excl_false_pre_anchor_can_escape_self(self):
        """Mode B + excl=False: pre anchors should be able to find positives in
        OTHER trials (not pinned to own trial), thanks to stochastic tiebreak."""
        yd = _binary_per_tp_disc()
        y3d, _, _ = _make_3d_y_with_class_structure()
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.5,
            sample_exclude_intrial=False,
            seed=0,
        )
        pre_anchors = torch.tensor(
            [t * NTIME + r for t in range(NTRIAL) for r in range(NTIME // 2)]
        )
        # Sample many times
        cross_trial_count = 0
        total = 0
        for _ in range(5):
            pos = d.sample_conditional(pre_anchors)
            ref_trial = pre_anchors // NTIME
            pos_trial = pos // NTIME
            cross_trial_count += (ref_trial != pos_trial).sum().item()
            total += pre_anchors.numel()
        cross_trial_rate = cross_trial_count / total
        # Without stochastic tiebreak, this would be near 0 (all self because tied dists);
        # with tiebreak, should be roughly (ntrial-1)/ntrial = 7/8 ≈ 0.875
        assert cross_trial_rate > 0.5, (
            f"Mode B + excl=False: pre anchors should mostly cross-trial via "
            f"tiebreak. got cross_trial_rate={cross_trial_rate:.3f}"
        )

    def test_tiebreak_works_for_single_anchor_fully_tied(self):
        """Edge case: batch_size=1 with perfectly tied dists (e.g., constant
        pre-stim emb across trials). Tiebreak must still randomize."""
        yd = _binary_per_tp_disc()
        # Construct y where pre-stim is identical across all trials (worst case)
        y3d = torch.randn(NTRIAL, NTIME, ND)
        y3d[:, : NTIME // 2, :] = y3d[0, 0, :].view(1, 1, ND).expand(NTRIAL, NTIME // 2, ND)
        d = TrialAwareDistribution(
            ntrial=NTRIAL,
            ntime=NTIME,
            conditional="delta",
            y=y3d,
            y_discrete=yd,
            delta=0.3,
            sample_exclude_intrial=False,
            seed=123,
        )
        ref = torch.tensor([5])  # single pre-stim anchor
        targets = set()
        for _ in range(100):
            pos = d.sample_conditional(ref)
            targets.add((pos // NTIME).item())
        # Must visit more than 1 trial across 100 draws (was deterministic before
        # the robust tiebreak scale fix)
        assert len(targets) > 1, (
            f"Tiebreak must work for batch_size=1 fully tied case; got {len(targets)} unique"
        )
