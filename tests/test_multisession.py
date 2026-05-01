"""Tests for TrialAwareMultisessionSampler — isolated sampler behavior.

Covers per-session prior shapes/bounds, cross-session strict invariant,
class-set / device validations, heterogeneous session lengths,
and mix round-trip.
"""

import numpy as np
import pytest
import torch

from trial_cebra.distribution import TrialAwareDistribution
from trial_cebra.multisession import (
    TrialAwareMultisessionSampler,
    _invert_index,
    _random_derangement,
    _strict_cross_session_permutation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_3d(ntrial=8, ntime=20, nd=10, pre_len=None, seed=0):
    """Build a single session: gray-screen pre + per-trial stim post.

    Returns y_3d (ntrial, ntime, nd), y_disc_flat (ntrial*ntime,).
    """
    if pre_len is None:
        pre_len = ntime // 2
    g = torch.Generator().manual_seed(seed)
    gray = torch.randn(nd, generator=g)
    stims = torch.randn(ntrial, nd, generator=g)
    y = torch.zeros(ntrial, ntime, nd)
    y[:, :pre_len, :] = gray.view(1, 1, nd)
    y[:, pre_len:, :] = stims.view(ntrial, 1, nd)
    yd = torch.zeros(ntrial * ntime, dtype=torch.long)
    for t in range(ntrial):
        yd[t * ntime + pre_len : (t + 1) * ntime] = 1
    return y, yd


def _build_per_session_dists(
    sessions_specs,  # list of dict with keys ntrial, ntime, nd, seed, with_disc
    conditional="delta",
    delta=0.5,
):
    dists = []
    for spec in sessions_specs:
        y3d, yd = _make_session_3d(
            ntrial=spec["ntrial"], ntime=spec["ntime"], nd=spec["nd"], seed=spec["seed"]
        )
        kwargs = dict(
            ntrial=spec["ntrial"],
            ntime=spec["ntime"],
            conditional=conditional,
            y=y3d,
            sample_exclude_intrial=False,  # multisession ensures cross-session at sampler layer
            delta=delta,
            seed=spec["seed"],
        )
        if spec.get("with_disc", True):
            kwargs["y_discrete"] = yd
        dists.append(TrialAwareDistribution(**kwargs))
    return dists


# ---------------------------------------------------------------------------
# Sampler basic behaviour
# ---------------------------------------------------------------------------


class TestSamplerInit:
    def test_num_sessions_one_raises(self):
        d = _build_per_session_dists([{"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0}])
        with pytest.raises(ValueError, match=">= 2 sessions"):
            TrialAwareMultisessionSampler(d, conditional="delta", seed=0)

    def test_time_conditional_no_disc_raises(self):
        # "time" without y_discrete has no alignment signal → NotImplementedError
        d = _build_per_session_dists(
            [
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0, "with_disc": False},
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 1, "with_disc": False},
            ],
            conditional="time",
        )
        with pytest.raises(NotImplementedError, match="y_discrete"):
            TrialAwareMultisessionSampler(d, conditional="time", seed=0)

    def test_unknown_conditional_raises(self):
        d = _build_per_session_dists(
            [
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0},
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 1},
            ]
        )
        with pytest.raises(ValueError, match="Unknown conditional"):
            TrialAwareMultisessionSampler(d, conditional="bogus", seed=0)

    def test_dim_mismatch_raises(self):
        d = _build_per_session_dists(
            [
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0},
                {"ntrial": 8, "ntime": 20, "nd": 12, "seed": 1},  # nd mismatch
            ]
        )
        with pytest.raises(ValueError, match="feature dim"):
            TrialAwareMultisessionSampler(d, conditional="delta", seed=0)

    def test_class_set_mismatch_raises(self):
        # Session 0 has classes {0, 1}; session 1 has classes {0, 2}
        y3d_0, yd_0 = _make_session_3d(ntrial=8, ntime=20, nd=10, seed=0)
        y3d_1, _ = _make_session_3d(ntrial=8, ntime=20, nd=10, seed=1)
        yd_1 = torch.zeros(8 * 20, dtype=torch.long)
        for t in range(8):
            yd_1[t * 20 + 10 : (t + 1) * 20] = 2  # class 2 instead of 1
        d0 = TrialAwareDistribution(
            ntrial=8,
            ntime=20,
            conditional="delta",
            y=y3d_0,
            y_discrete=yd_0,
            sample_exclude_intrial=False,
            delta=0.5,
            seed=0,
        )
        d1 = TrialAwareDistribution(
            ntrial=8,
            ntime=20,
            conditional="delta",
            y=y3d_1,
            y_discrete=yd_1,
            sample_exclude_intrial=False,
            delta=0.5,
            seed=1,
        )
        with pytest.raises(ValueError, match="classes"):
            TrialAwareMultisessionSampler([d0, d1], conditional="delta", seed=0)

    def test_inconsistent_discrete_presence_raises(self):
        d_with = _build_per_session_dists(
            [{"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0, "with_disc": True}]
        )[0]
        d_without = _build_per_session_dists(
            [{"ntrial": 8, "ntime": 20, "nd": 10, "seed": 1, "with_disc": False}]
        )[0]
        with pytest.raises(ValueError, match="inconsistent"):
            TrialAwareMultisessionSampler([d_with, d_without], conditional="delta", seed=0)

    def test_mode_c_in_multisession_accepted(self):
        # 2-D y + per-timepoint y_discrete: auto-broadcasts to Mode B, accepted in multisession.
        from trial_cebra.distribution import _DISC_MODE_PER_TP_3D

        y2d_0 = torch.randn(8, 10)
        y2d_1 = torch.randn(8, 10)
        _, yd = _make_session_3d(ntrial=8, ntime=20, nd=10, seed=0)
        d0 = TrialAwareDistribution(
            ntrial=8,
            ntime=20,
            conditional="delta",
            y=y2d_0,
            y_discrete=yd,
            sample_exclude_intrial=False,
            delta=0.5,
            seed=0,
        )
        d1 = TrialAwareDistribution(
            ntrial=8,
            ntime=20,
            conditional="delta",
            y=y2d_1,
            y_discrete=yd,
            sample_exclude_intrial=False,
            delta=0.5,
            seed=1,
        )
        # Both distributions should be Mode B (auto-broadcast from 2-D y)
        assert d0._disc_mode == _DISC_MODE_PER_TP_3D
        assert d1._disc_mode == _DISC_MODE_PER_TP_3D
        # Sampler should accept them without raising
        sampler = TrialAwareMultisessionSampler([d0, d1], conditional="delta", seed=0)
        assert sampler is not None


# ---------------------------------------------------------------------------
# Prior + conditional shapes
# ---------------------------------------------------------------------------


class TestSamplerSampling:
    def _two_session_sampler(self, conditional="delta"):
        dists = _build_per_session_dists(
            [
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0},
                {"ntrial": 6, "ntime": 30, "nd": 10, "seed": 1},  # heterogeneous lengths
            ],
            conditional=conditional,
        )
        return TrialAwareMultisessionSampler(dists, conditional=conditional, seed=42), dists

    def test_sample_prior_shape_and_bounds(self):
        s, dists = self._two_session_sampler()
        ref = s.sample_prior(64)
        assert ref.shape == (2, 64)
        # Bounds per session
        assert ref[0].min() >= 0 and ref[0].max() < dists[0].ntrial * dists[0].ntime
        assert ref[1].min() >= 0 and ref[1].max() < dists[1].ntrial * dists[1].ntime

    def test_sample_conditional_shapes_delta(self):
        s, _ = self._two_session_sampler("delta")
        ref = s.sample_prior(32)
        pos, idx, idx_rev = s.sample_conditional(ref)
        assert pos.shape == (2, 32)
        assert idx.shape == (2 * 32,)
        assert idx_rev.shape == (2 * 32,)
        # idx is a valid permutation
        assert sorted(idx.tolist()) == list(range(64))
        # idx and idx_rev are mutually inverse
        assert (idx[idx_rev] == np.arange(64)).all()

    def test_sample_conditional_shapes_time_delta(self):
        s, _ = self._two_session_sampler("time_delta")
        ref = s.sample_prior(16)
        pos, idx, idx_rev = s.sample_conditional(ref)
        assert pos.shape == (2, 16)
        assert idx.shape == (2 * 16,)


# ---------------------------------------------------------------------------
# Strict cross-session invariant
# ---------------------------------------------------------------------------


class TestStrictCrossSession:
    def _build(self, num_sessions, conditional="delta"):
        specs = [
            {"ntrial": 6 + s, "ntime": 20 + s * 5, "nd": 10, "seed": s} for s in range(num_sessions)
        ]
        dists = _build_per_session_dists(specs, conditional=conditional)
        s = TrialAwareMultisessionSampler(dists, conditional=conditional, seed=99)
        return s, dists

    def test_strict_cross_session_2_sessions_delta(self):
        s, _ = self._build(2, "delta")
        for _ in range(5):  # multiple draws to catch flake
            ref = s.sample_prior(32)
            _, idx, _ = s.sample_conditional(ref)
            B = 32
            for sp in range(2):
                for b in range(B):
                    flat_target = sp * B + b
                    flat_origin = idx[flat_target]
                    origin_session = flat_origin // B
                    assert origin_session != sp, (
                        f"strict cross-session violated: target session {sp} got "
                        f"query from session {origin_session}"
                    )

    def test_strict_cross_session_3_sessions_time_delta(self):
        s, _ = self._build(3, "time_delta")
        for _ in range(5):
            ref = s.sample_prior(20)
            _, idx, _ = s.sample_conditional(ref)
            B = 20
            for sp in range(3):
                for b in range(B):
                    flat_target = sp * B + b
                    flat_origin = idx[flat_target]
                    origin_session = flat_origin // B
                    assert origin_session != sp


# ---------------------------------------------------------------------------
# Same-class enforcement across sessions
# ---------------------------------------------------------------------------


class TestSameClassMultisession:
    def test_same_class_delta(self):
        # Build 2 sessions with shared class set {0, 1}
        dists = _build_per_session_dists(
            [
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 0},
                {"ntrial": 8, "ntime": 20, "nd": 10, "seed": 1},
            ],
            conditional="delta",
        )
        s = TrialAwareMultisessionSampler(dists, conditional="delta", seed=7)
        ref = s.sample_prior(32)
        pos, idx, _ = s.sample_conditional(ref)
        # For each (sp, b), the query came from origin session via idx; the
        # anchor's class must equal pos's class in target session.
        B = 32
        ref_t = torch.from_numpy(ref)
        pos_t = torch.from_numpy(pos)
        # Recover anchor class for each query at its target slot
        for sp in range(2):
            for b in range(B):
                flat_origin = idx[sp * B + b]
                so = flat_origin // B
                bo = flat_origin % B
                origin_anchor = ref_t[so, bo].item()
                anchor_cls = dists[so]._y_discrete[origin_anchor].item()
                pos_cls = dists[sp]._y_discrete[pos_t[sp, b]].item()
                assert anchor_cls == pos_cls, (
                    f"same-class violated at (target s={sp}, b={b}): "
                    f"anchor cls {anchor_cls} from session {so}, pos cls {pos_cls}"
                )


# ---------------------------------------------------------------------------
# Heterogeneous session lengths
# ---------------------------------------------------------------------------


class TestHeterogeneousLengths:
    def test_three_diff_lengths(self):
        specs = [
            {"ntrial": 5, "ntime": 30, "nd": 10, "seed": 0},
            {"ntrial": 8, "ntime": 15, "nd": 10, "seed": 1},
            {"ntrial": 10, "ntime": 20, "nd": 10, "seed": 2},
        ]
        dists = _build_per_session_dists(specs, conditional="delta")
        s = TrialAwareMultisessionSampler(dists, conditional="delta", seed=0)
        ref = s.sample_prior(16)
        pos, _, _ = s.sample_conditional(ref)
        # Each session's positives must be in its own range
        for sp in range(3):
            assert pos[sp].min() >= 0
            assert pos[sp].max() < specs[sp]["ntrial"] * specs[sp]["ntime"]


# ---------------------------------------------------------------------------
# mix round-trip
# ---------------------------------------------------------------------------


class TestMixRoundTrip:
    def test_mix_with_idx_rev_undoes_idx(self):
        s, _ = TestSamplerSampling()._two_session_sampler()
        ref = s.sample_prior(8)
        pos, idx, idx_rev = s.sample_conditional(ref)
        # Build a fake (2, 8, D) embedding that contains source position info
        D = 5
        emb = np.arange(2 * 8 * D, dtype=np.float32).reshape(2, 8, D)
        # Apply idx (the 'forward' shuffle that brought queries to their search slots)
        shuffled = s.mix(emb, idx)
        # Apply idx_rev to undo
        recovered = s.mix(shuffled, idx_rev)
        np.testing.assert_array_equal(recovered, emb)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_invert_index_round_trip(self):
        rng = np.random.default_rng(0)
        idx = rng.permutation(20)
        rev = _invert_index(idx)
        assert (idx[rev] == np.arange(20)).all()

    def test_random_derangement_no_fixed_points(self):
        rng = np.random.default_rng(0)
        for n in (2, 3, 5, 10):
            for _ in range(20):
                perm = _random_derangement(n, rng)
                assert not np.any(perm == np.arange(n))

    def test_strict_perm_each_target_from_other_session(self):
        rng = np.random.default_rng(0)
        S, B = 4, 7
        idx = _strict_cross_session_permutation(S, B, rng)
        for sp in range(S):
            for b in range(B):
                flat_origin = idx[sp * B + b]
                origin_session = flat_origin // B
                # Cross-session strict
                assert origin_session != sp
                # Batch position preserved
                assert flat_origin % B == b


# ---------------------------------------------------------------------------
# End-to-end TrialCEBRA.fit with multisession
# ---------------------------------------------------------------------------


class TestTrialCEBRAFitMultisession:
    """Integration tests: TrialCEBRA.fit with list-of-X (multisession)."""

    def _make_ms_data(self, conditional="delta"):
        np.random.seed(0)
        X = [
            np.random.randn(10, 30, 32).astype(np.float32),
            np.random.randn(8, 25, 48).astype(np.float32),
        ]
        y_cont = [
            np.random.randn(10, 30, 8).astype(np.float32),
            np.random.randn(8, 25, 8).astype(np.float32),
        ]
        y_disc = [
            np.stack([[0] * 15 + [1] * 15] * 10).astype(np.int64),
            np.stack([[0] * 12 + [1] * 13] * 8).astype(np.int64),
        ]
        return X, y_disc, y_cont

    def _base_model(self, conditional="delta", **kw):
        from trial_cebra import TrialCEBRA

        return TrialCEBRA(
            model_architecture="offset1-model",
            conditional=conditional,
            sample_fix_trial=False,
            sample_exclude_intrial=False,
            time_offsets=5,
            delta=0.3,
            max_iterations=5,
            batch_size=16,
            output_dimension=3,
            device="cpu",
            verbose=False,
            **kw,
        )

    def test_fit_delta_multisession_runs(self):
        from trial_cebra.multisession import TrialAwareMultisessionSampler

        X, yd, yc = self._make_ms_data()
        model = self._base_model("delta")
        model.fit(X, yd, yc)
        assert isinstance(model.distribution_, TrialAwareMultisessionSampler)
        assert model.distribution_.num_sessions == 2
        # session_lengths = ntrial_s * ntime_s
        assert list(model.distribution_.session_lengths) == [10 * 30, 8 * 25]

    def test_fit_time_delta_multisession_runs(self):
        from trial_cebra.multisession import TrialAwareMultisessionSampler

        X, yd, yc = self._make_ms_data()
        model = self._base_model("time_delta")
        model.fit(X, yd, yc)
        assert isinstance(model.distribution_, TrialAwareMultisessionSampler)

    def test_fit_time_disc_only_runs(self):
        """conditional='time' + y_disc-only multisession should now succeed."""
        from trial_cebra.multisession import TrialAwareMultisessionSampler

        X, yd, _ = self._make_ms_data()
        model = self._base_model("time")
        model.fit(X, yd)
        assert isinstance(model.distribution_, TrialAwareMultisessionSampler)
        assert model.distribution_.conditional == "time"

    def test_fit_time_no_disc_raises(self):
        """conditional='time' without y_disc has no alignment signal → error."""
        X, _, _ = self._make_ms_data()
        model = self._base_model("time")
        with pytest.raises((NotImplementedError, ValueError)):
            model.fit(X)  # no labels at all

    def test_transform_per_session(self):
        X, yd, yc = self._make_ms_data()
        model = self._base_model("delta")
        model.fit(X, yd, yc)
        for s in range(2):
            X_flat = X[s].reshape(-1, X[s].shape[-1])
            emb = model.transform(X_flat, session_id=s)
            assert emb.shape == (X[s].shape[0] * X[s].shape[1], 3)

    def test_fit_no_discrete_multisession(self):
        """Multisession without y_discrete should still work (delta, class-agnostic)."""
        X, _, yc = self._make_ms_data()
        model = self._base_model("delta")
        model.fit(X, yc)  # no discrete
        assert model.distribution_.num_sessions == 2

    def test_fit_raises_on_mismatched_session_counts(self):
        X, yd, yc = self._make_ms_data()
        yc_bad = yc[:1]  # only 1 session in continuous, 2 in X
        model = self._base_model("delta")
        with pytest.raises(ValueError, match="length"):
            model.fit(X, yd, yc_bad)

    def test_fit_raises_without_continuous_y(self):
        """Trial-aware multisession requires continuous y for delta/time_delta."""
        X, yd, _ = self._make_ms_data()
        model = self._base_model("delta")
        with pytest.raises(ValueError, match="continuous y"):
            model.fit(X, yd)  # only discrete, no continuous


# ---------------------------------------------------------------------------
# Disc-only multisession: conditional="time"
# ---------------------------------------------------------------------------


def _make_disc_only_sessions(num_sessions=2, ntrial=6, ntime=15, nneuro=12, n_classes=3, seed=0):
    """Build num_sessions sessions of (X_3d, y_disc_flat) with shared class set."""
    rng = np.random.default_rng(seed)
    sessions = []
    for s in range(num_sessions):
        X = torch.from_numpy(rng.standard_normal((ntrial, ntime, nneuro)).astype("float32"))
        classes = np.tile(np.arange(n_classes), ntrial // n_classes + 1)[:ntrial]
        y_disc_per_trial = classes
        y_disc_flat = np.repeat(y_disc_per_trial, ntime).astype(np.int64)
        sessions.append((X, y_disc_flat, ntrial, ntime))
    return sessions


def _build_disc_only_dists(sessions, time_offsets=3):
    """Build TrialAwareDistribution(conditional='time') per session."""
    dists = []
    for X, y_disc_flat, ntrial, ntime in sessions:
        yd = torch.from_numpy(y_disc_flat).long()
        d = TrialAwareDistribution(
            ntrial=ntrial,
            ntime=ntime,
            conditional="time",
            y=None,
            y_discrete=yd,
            sample_exclude_intrial=False,
            time_offsets=time_offsets,
        )
        dists.append(d)
    return dists


class TestDiscOnlyMultisession:
    """y_disc-only + conditional='time' multisession sampler."""

    def test_sampler_init_succeeds(self):
        sessions = _make_disc_only_sessions()
        dists = _build_disc_only_dists(sessions)
        sampler = TrialAwareMultisessionSampler(dists, conditional="time")
        assert sampler.conditional == "time"
        assert sampler.num_sessions == 2

    def test_sampler_init_no_disc_raises(self):
        """conditional='time' without y_discrete has no alignment signal → error."""
        sessions = _make_disc_only_sessions()
        dists_no_disc = []
        for _, _, ntrial, ntime in sessions:
            d = TrialAwareDistribution(ntrial=ntrial, ntime=ntime, conditional="time", y=None)
            dists_no_disc.append(d)
        with pytest.raises(NotImplementedError, match="y_discrete"):
            TrialAwareMultisessionSampler(dists_no_disc, conditional="time")

    def test_prior_shape(self):
        sessions = _make_disc_only_sessions()
        dists = _build_disc_only_dists(sessions)
        sampler = TrialAwareMultisessionSampler(dists, conditional="time")
        B = 32
        prior = sampler.sample_prior(B)
        assert prior.shape == (2, B)
        for s, (_, _, ntrial, ntime) in enumerate(sessions):
            assert prior[s].min() >= 0
            assert prior[s].max() < ntrial * ntime

    def test_positive_same_class(self):
        sessions = _make_disc_only_sessions(ntrial=8, ntime=15, n_classes=3)
        dists = _build_disc_only_dists(sessions, time_offsets=20)  # wide window
        sampler = TrialAwareMultisessionSampler(dists, conditional="time")
        B = 64
        ref = sampler.sample_prior(B)
        pos, idx_flat, _ = sampler.sample_conditional(ref)
        for s in range(2):
            _, y_disc_flat, ntrial, ntime = sessions[s]
            for b in range(B):
                # Which session does pos[s, b] come from?  It came from target session s.
                # The anchor that produced this query may have come from a different session.
                # Check the anchor that was mapped here via idx_flat.
                pass
        # Instead verify that for the anchor session, classes match after unshuffle.
        # Easier: verify each pos[sp] has same class as the shuffled anchor.
        pos_classes = np.empty((2, B), dtype=np.int64)
        ref_classes = np.empty((2, B), dtype=np.int64)
        for s in range(2):
            _, y_disc_flat, ntrial, ntime = sessions[s]
            pos_classes[s] = y_disc_flat[pos[s]]
            ref_classes[s] = y_disc_flat[ref[s]]
        # After shuffling, each anchor in session sp got a query from session s != sp.
        # The shuffle is encoded in idx_flat. We just check that all positives
        # have the same class as their post-shuffle anchor (class is what's shuffled).
        ref_classes_flat = ref_classes.ravel()  # (S*B,)
        pos_classes_flat = pos_classes.ravel()  # (S*B,) but pos is already in target session order
        # idx_flat maps ref→pos: pos[idx_flat[i]] should have same class as ref[i].
        # Equivalently: ref_classes_flat and pos_classes_flat[inverse] agree.
        # Simple check: after unshuffling, anchor class == pos class.
        idx_rev = _invert_index(idx_flat)
        assert np.all(pos_classes_flat[idx_rev] == ref_classes_flat)

    @pytest.mark.parametrize("time_offsets", [1, 5])
    def test_positive_within_time_window(self, time_offsets):
        ntrial, ntime = 8, 20
        sessions = _make_disc_only_sessions(ntrial=ntrial, ntime=ntime, n_classes=2)
        dists = _build_disc_only_dists(sessions, time_offsets=time_offsets)
        sampler = TrialAwareMultisessionSampler(dists, conditional="time")
        B = 128
        ref = sampler.sample_prior(B)
        pos, idx_flat, idx_rev_flat = sampler.sample_conditional(ref)
        # For each anchor, check that pos rel_time is within ±time_offsets of anchor rel_time.
        # anchor rel_time comes from the SOURCE session (before shuffle).
        # pos rel_time is in the TARGET session.
        # After unshuffle: anchor[s, b] was sent to session sp = (s + perm) % 2.
        # Simpler: check that pos_rel and anchor_rel (post-shuffle) satisfy the window.
        ref_rel = ref % ntime  # (2, B) — anchor's relative time in its session
        pos_rel = pos % ntime  # (2, B) — positive's relative time in target session
        # Flatten and apply derangement to get "shuffled anchor rel_time" per target session.
        ref_rel_flat = ref_rel.ravel()
        shuffled_ref_rel = ref_rel_flat[idx_flat].reshape(2, B)
        diff = np.abs(pos_rel - shuffled_ref_rel)
        # Allow fallback cases (class with no candidates in window) to exceed by checking
        # that the vast majority satisfy the constraint.
        assert (diff <= time_offsets).mean() >= 0.95

    def test_cross_session_guarantee(self):
        sessions = _make_disc_only_sessions(num_sessions=3, ntrial=6, ntime=10, n_classes=2)
        dists = _build_disc_only_dists(sessions, time_offsets=5)
        sampler = TrialAwareMultisessionSampler(dists, conditional="time")
        B = 64
        ref = sampler.sample_prior(B)
        pos, idx_flat, _ = sampler.sample_conditional(ref)
        # Verify that every positive in session sp did NOT come from session sp.
        # (cross-session is enforced by the derangement: idx_flat ensures each element
        # changes session.)  The sampler's _sample_conditional_time respects this.
        # We verify via idx_flat: for each flat position i in [sp*B, (sp+1)*B),
        # the source session is idx_flat[i] // B != sp.
        S = 3
        for sp in range(S):
            for b in range(B):
                flat_pos = sp * B + b
                src_session = idx_flat[flat_pos] // B
                assert src_session != sp

    def test_trial_cebra_fit_disc_only_multisession(self):
        """End-to-end: TrialCEBRA.fit with disc-only + conditional='time'."""
        import trial_cebra

        ntrial, ntime, nneuro, n_classes = 6, 12, 10, 3
        sessions_data = _make_disc_only_sessions(
            num_sessions=2, ntrial=ntrial, ntime=ntime, nneuro=nneuro, n_classes=n_classes
        )
        X_list = [s[0].numpy() for s in sessions_data]
        # Pass per-trial disc labels (ntrial,) — flatten_epochs handles broadcasting
        y_disc_list = [
            np.tile(np.arange(n_classes), ntrial // n_classes + 1)[:ntrial].astype(np.int64)
            for _ in range(2)
        ]

        model = trial_cebra.TrialCEBRA(
            model_architecture="offset1-model",
            batch_size=16,
            max_iterations=3,
            output_dimension=4,
            conditional="time",
            time_offsets=3,
            device="cpu",
            verbose=False,
        )
        model.fit(X_list, y_disc_list)
        assert model.distribution_.num_sessions == 2
        assert model.distribution_.conditional == "time"
        for s in range(2):
            X_flat = X_list[s].reshape(-1, nneuro)
            emb = model.transform(X_flat, session_id=s)
            assert emb.shape[0] == ntrial * ntime
            assert emb.shape[1] == 4
