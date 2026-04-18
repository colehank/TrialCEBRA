"""Tests for epoch-format utilities (flatten_epochs, fit_epochs, transform_epochs)."""

import numpy as np
import pytest

from trial_cebra import TrialCEBRA, flatten_epochs


# ---------------------------------------------------------------------------
# flatten_epochs — shape contracts
# ---------------------------------------------------------------------------


def test_flatten_basic():
    ntrial, ntime, nneuro = 8, 50, 32
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    X_flat, y_flat, starts, ends = flatten_epochs(X)

    assert X_flat.shape == (ntrial * ntime, nneuro)
    assert starts.shape == (ntrial,)
    assert ends.shape == (ntrial,)
    assert (starts == np.arange(ntrial) * ntime).all()
    assert (ends == starts + ntime).all()
    assert y_flat == ()


def test_flatten_label_per_trial_1d():
    ntrial, ntime, nneuro = 10, 40, 16
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    y_disc = np.arange(ntrial, dtype=np.int64)

    _, y_flat, _, _ = flatten_epochs(X, y_disc)
    assert len(y_flat) == 1
    assert y_flat[0].shape == (ntrial * ntime,)
    # first ntime elements all equal trial-0 label
    assert (y_flat[0][:ntime] == 0).all()
    assert (y_flat[0][ntime : 2 * ntime] == 1).all()


def test_flatten_label_per_trial_2d():
    ntrial, ntime, nneuro = 6, 30, 8
    stim_dim = 12  # intentionally != ntime
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    y_cont = np.random.randn(ntrial, stim_dim).astype(np.float32)

    _, y_flat, _, _ = flatten_epochs(X, y_cont)
    assert y_flat[0].shape == (ntrial * ntime, stim_dim)
    # each group of ntime rows should be identical (broadcast)
    for i in range(ntrial):
        block = y_flat[0][i * ntime : (i + 1) * ntime]
        assert np.allclose(block, block[0])


def test_flatten_label_per_timepoint_2d():
    ntrial, ntime, nneuro = 5, 20, 4
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    y_tp = np.random.randn(ntrial, ntime).astype(np.float32)

    _, y_flat, _, _ = flatten_epochs(X, y_tp)
    assert y_flat[0].shape == (ntrial * ntime,)
    assert np.allclose(y_flat[0], y_tp.reshape(-1))


def test_flatten_label_per_timepoint_3d():
    ntrial, ntime, nneuro = 4, 25, 6
    stim_dim = 10
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    y_3d = np.random.randn(ntrial, ntime, stim_dim).astype(np.float32)

    _, y_flat, _, _ = flatten_epochs(X, y_3d)
    assert y_flat[0].shape == (ntrial * ntime, stim_dim)
    assert np.allclose(y_flat[0], y_3d.reshape(ntrial * ntime, stim_dim))


def test_flatten_multiple_labels():
    ntrial, ntime, nneuro = 6, 20, 8
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    y_cont = np.random.randn(ntrial, 10).astype(np.float32)   # per-trial
    y_disc = np.zeros(ntrial, dtype=np.int64)                  # per-trial discrete

    _, y_flat, _, _ = flatten_epochs(X, y_cont, y_disc)
    assert len(y_flat) == 2
    assert y_flat[0].shape == (ntrial * ntime, 10)
    assert y_flat[1].shape == (ntrial * ntime,)


def test_flatten_invalid_x_2d():
    with pytest.raises(ValueError, match="3-D"):
        flatten_epochs(np.random.randn(100, 32))


def test_flatten_invalid_x_4d():
    with pytest.raises(ValueError, match="3-D"):
        flatten_epochs(np.random.randn(4, 10, 32, 2))


def test_flatten_invalid_label_wrong_ntrial():
    ntrial, ntime, nneuro = 6, 20, 8
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    with pytest.raises(ValueError):
        flatten_epochs(X, np.zeros(ntrial + 1))


def test_flatten_invalid_label_4d():
    ntrial, ntime, nneuro = 4, 20, 8
    X = np.random.randn(ntrial, ntime, nneuro).astype(np.float32)
    with pytest.raises(ValueError, match="1-D, 2-D, or 3-D"):
        flatten_epochs(X, np.zeros((ntrial, ntime, 3, 2)))


# ---------------------------------------------------------------------------
# TrialCEBRA.fit_epochs / transform_epochs — end-to-end
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_epoch_data(ntrial=12, ntime=50, nneuro=16, stim_dim=8):
    X = RNG.standard_normal((ntrial, ntime, nneuro)).astype(np.float32)
    y_cont = RNG.standard_normal((ntrial, stim_dim)).astype(np.float32)
    return X, y_cont


def _small_model(conditional="trial_delta"):
    return TrialCEBRA(
        model_architecture="offset1-model",
        conditional=conditional,
        time_offsets=5,
        delta=0.3,
        max_iterations=5,
        batch_size=32,
        output_dimension=3,
    )


def test_fit_epochs_returns_self():
    X, y = _make_epoch_data()
    model = _small_model()
    result = model.fit_epochs(X, y)
    assert result is model


def test_fit_epochs_transform_shape():
    ntrial, ntime, nneuro = 10, 40, 16
    X, y = _make_epoch_data(ntrial, ntime, nneuro)
    model = _small_model()
    model.fit_epochs(X, y)

    emb = model.transform(X.reshape(ntrial * ntime, nneuro))
    assert emb.shape == (ntrial * ntime, 3)


def test_transform_epochs_shape():
    ntrial, ntime, nneuro = 10, 40, 16
    X, y = _make_epoch_data(ntrial, ntime, nneuro)
    model = _small_model()
    model.fit_epochs(X, y)

    emb = model.transform_epochs(X)
    assert emb.shape == (ntrial, ntime, 3)


def test_transform_epochs_consistency():
    ntrial, ntime, nneuro = 8, 30, 12
    X, y = _make_epoch_data(ntrial, ntime, nneuro)
    model = _small_model()
    model.fit_epochs(X, y)

    emb_flat = model.transform(X.reshape(ntrial * ntime, nneuro))
    emb_ep = model.transform_epochs(X)
    assert np.allclose(emb_flat, emb_ep.reshape(ntrial * ntime, -1))


def test_fit_epochs_all_conditionals():
    conditionals = ["trialTime", "trialDelta", "trial_delta", "trialTime_delta", "trialTime_trialDelta"]
    X, y = _make_epoch_data()
    for cond in conditionals:
        model = _small_model(conditional=cond)
        model.fit_epochs(X, y)
        emb = model.transform_epochs(X)
        assert emb.ndim == 3, f"conditional={cond!r} returned {emb.ndim}-D array"


def test_fit_epochs_with_discrete_label():
    ntrial, ntime, nneuro = 12, 40, 16
    X, y_cont = _make_epoch_data(ntrial, ntime, nneuro)
    y_disc = RNG.integers(0, 2, size=ntrial).astype(np.int64)  # per-trial class

    model = _small_model()
    model.fit_epochs(X, y_cont, y_disc)
    emb = model.transform_epochs(X)
    assert emb.shape == (ntrial, ntime, 3)


def test_transform_epochs_invalid_input():
    X, y = _make_epoch_data()
    model = _small_model()
    model.fit_epochs(X, y)
    with pytest.raises(ValueError, match="3-D"):
        model.transform_epochs(np.random.randn(100, 16).astype(np.float32))
