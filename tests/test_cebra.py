"""Integration tests for TrialCEBRA sklearn wrapper."""

import numpy as np
import pytest
import torch

from trial_cebra import TrialCEBRA
from trial_cebra.distribution import TRIAL_CONDITIONALS, TrialAwareDistribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_data(n_trials=6, trial_len=40, gap=10, embed_dim=8, neural_dim=20):
    """Synthetic sklearn-compatible trial data."""
    total = n_trials * (trial_len + gap)
    neural = np.random.randn(total, neural_dim).astype(np.float32)
    continuous = np.random.randn(total, embed_dim).astype(np.float32)
    for i in range(n_trials):
        s = i * (trial_len + gap) + gap
        e = s + trial_len
        continuous[s:e] = np.random.randn(1, embed_dim)

    trial_starts = np.array([i * (trial_len + gap) + gap for i in range(n_trials)])
    trial_ends = trial_starts + trial_len
    return neural, continuous, trial_starts, trial_ends, total


def _make_discrete_labels(trial_starts, trial_ends, total, n_classes=2):
    discrete = np.zeros(total, dtype=np.int64)
    for i, (s, e) in enumerate(zip(trial_starts, trial_ends)):
        discrete[s:e] = i % n_classes
    return discrete


def _base_kwargs(conditional, delta=0.1, time_offsets=5):
    kw = dict(
        model_architecture="offset1-model",
        conditional=conditional,
        time_offsets=time_offsets,
        max_iterations=5,
        batch_size=32,
        output_dimension=3,
    )
    if conditional != "trialTime":
        kw["delta"] = delta
    return kw


# ---------------------------------------------------------------------------
# trialTime
# ---------------------------------------------------------------------------


class TestTrialTimeFit:
    def test_fit_and_transform(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        emb = model.transform(neural)
        assert emb.shape == (total, 3)

    def test_loss_recorded(self):
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert len(model.state_dict_["loss"]) == 5

    def test_no_delta_needed(self):
        """trialTime does not require delta — should not raise without it."""
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="trialTime",
            time_offsets=5,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)

    def test_missing_trial_metadata_raises(self):
        neural, continuous, _, _, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime"))
        with pytest.raises((ValueError, AttributeError)):
            model.fit(neural, continuous)  # no trial_starts/ends


# ---------------------------------------------------------------------------
# trialDelta
# ---------------------------------------------------------------------------


class TestTrialDeltaFit:
    def test_fit_and_transform(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialDelta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)

    def test_loss_recorded(self):
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialDelta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert len(model.state_dict_["loss"]) == 5

    def test_missing_trial_metadata_raises(self):
        neural, continuous, _, _, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialDelta"))
        with pytest.raises((ValueError, AttributeError)):
            model.fit(neural, continuous)


# ---------------------------------------------------------------------------
# trial_delta
# ---------------------------------------------------------------------------


class TestTrialDeltaResampledFit:
    def test_fit_and_transform(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs("trial_delta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)

    def test_loss_recorded(self):
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trial_delta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert len(model.state_dict_["loss"]) == 5

    def test_not_locked(self):
        """trial_delta must not pre-compute _locked_target_trials."""
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trial_delta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert not hasattr(model.distribution_, "_locked_target_trials")

    def test_missing_trial_metadata_raises(self):
        neural, continuous, _, _, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trial_delta"))
        with pytest.raises((ValueError, AttributeError)):
            model.fit(neural, continuous)


# ---------------------------------------------------------------------------
# trialTime_delta
# ---------------------------------------------------------------------------


class TestTrialTimeDeltaFit:
    def test_fit_and_transform(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_delta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)

    def test_loss_recorded(self):
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_delta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert len(model.state_dict_["loss"]) == 5

    def test_missing_trial_metadata_raises(self):
        neural, continuous, _, _, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_delta"))
        with pytest.raises((ValueError, AttributeError)):
            model.fit(neural, continuous)


# ---------------------------------------------------------------------------
# trialTime_trialDelta
# ---------------------------------------------------------------------------


class TestTrialTimeTrialDeltaFit:
    def test_fit_and_transform(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_trialDelta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)

    def test_loss_recorded(self):
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_trialDelta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert len(model.state_dict_["loss"]) == 5

    def test_missing_trial_metadata_raises(self):
        neural, continuous, _, _, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_trialDelta"))
        with pytest.raises((ValueError, AttributeError)):
            model.fit(neural, continuous)


# ---------------------------------------------------------------------------
# Native CEBRA conditionals pass through unchanged
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_time_delta_passthrough(self):
        neural, continuous, _, _, total = _make_data()
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="time_delta",
            time_offsets=5,
            delta=0.1,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural, continuous)
        assert model.transform(neural).shape == (total, 3)

    def test_time_contrastive_passthrough(self):
        neural = np.random.randn(200, 20).astype(np.float32)
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="time",
            time_offsets=5,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural)
        assert model.transform(neural).shape == (200, 3)

    def test_distribution_not_replaced_for_native(self):
        """Native conditional must NOT set distribution_ on the model."""
        neural, continuous, _, _, _ = _make_data()
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="time_delta",
            time_offsets=5,
            delta=0.1,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural, continuous)
        # TrialAwareDistribution should NOT be set for native conditionals
        assert not hasattr(model, "distribution_") or not isinstance(
            model.distribution_, TrialAwareDistribution
        )


# ---------------------------------------------------------------------------
# Discrete + continuous (MixedDataLoader path)
# ---------------------------------------------------------------------------


class TestMixedLabels:
    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_fit_discrete_continuous(self, cond):
        """discrete + continuous y must work for all trial conditionals."""
        neural, continuous, starts, ends, total = _make_data()
        discrete = _make_discrete_labels(starts, ends, total)

        kw = _base_kwargs(cond)
        model = TrialCEBRA(**kw)
        model.fit(neural, continuous, discrete, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)

    def test_distribution_is_trial_aware_with_discrete(self):
        """After fit, model.distribution_ must be TrialAwareDistribution."""
        neural, continuous, starts, ends, total = _make_data()
        discrete = _make_discrete_labels(starts, ends, total)
        model = TrialCEBRA(**_base_kwargs("trialTime_trialDelta"))
        model.fit(neural, continuous, discrete, trial_starts=starts, trial_ends=ends)
        assert isinstance(model.distribution_, TrialAwareDistribution)

    def test_distribution_has_discrete(self):
        """TrialAwareDistribution must carry the discrete index."""
        neural, continuous, starts, ends, total = _make_data()
        discrete = _make_discrete_labels(starts, ends, total)
        model = TrialCEBRA(**_base_kwargs("trialDelta"))
        model.fit(neural, continuous, discrete, trial_starts=starts, trial_ends=ends)
        assert model.distribution_.discrete is not None


# ---------------------------------------------------------------------------
# Loader distribution replacement
# ---------------------------------------------------------------------------


class TestDistributionReplacement:
    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_distribution_is_trial_aware(self, cond):
        neural, continuous, starts, ends, _ = _make_data()
        kw = _base_kwargs(cond)
        model = TrialCEBRA(**kw)
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert isinstance(model.distribution_, TrialAwareDistribution)

    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_distribution_conditional_matches(self, cond):
        neural, continuous, starts, ends, _ = _make_data()
        model = TrialCEBRA(**_base_kwargs(cond))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.distribution_.conditional == cond


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class TestTransform:
    def test_torch_input(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs("trialTime_trialDelta"))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        emb = model.transform(torch.from_numpy(neural))
        assert emb.shape == (total, 3)

    def test_output_dimension(self):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="trialTime_trialDelta",
            time_offsets=5,
            delta=0.1,
            max_iterations=5,
            batch_size=32,
            output_dimension=8,
        )
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 8)

    @pytest.mark.parametrize("cond", list(TRIAL_CONDITIONALS))
    def test_all_conditionals_transform(self, cond):
        neural, continuous, starts, ends, total = _make_data()
        model = TrialCEBRA(**_base_kwargs(cond))
        model.fit(neural, continuous, trial_starts=starts, trial_ends=ends)
        assert model.transform(neural).shape == (total, 3)
