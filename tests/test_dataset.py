"""Tests for TrialTensorDataset."""

import numpy as np
import pytest
import torch

from trial_cebra.dataset import TrialTensorDataset


class TestTrialTensorDatasetInit:
    def test_basic_init(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        trial_starts = torch.tensor([0, 250, 500, 750])
        trial_ends = torch.tensor([250, 500, 750, 1000])

        dataset = TrialTensorDataset(
            neural=neural, continuous=continuous, trial_starts=trial_starts, trial_ends=trial_ends
        )

        assert hasattr(dataset, "trial_starts")
        assert hasattr(dataset, "trial_ends")
        assert len(dataset.trial_starts) == 4

    def test_init_with_numpy(self):
        neural = np.random.randn(1000, 50).astype(np.float32)
        continuous = np.random.randn(1000, 10).astype(np.float32)
        trial_starts = np.array([0, 250, 500, 750])
        trial_ends = np.array([250, 500, 750, 1000])

        dataset = TrialTensorDataset(
            neural=neural, continuous=continuous, trial_starts=trial_starts, trial_ends=trial_ends
        )

        assert isinstance(dataset.neural, torch.Tensor)
        assert isinstance(dataset.trial_starts, torch.Tensor)

    def test_init_without_trials(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        dataset = TrialTensorDataset(neural=neural, continuous=continuous)
        assert not hasattr(dataset, "trial_starts") or dataset.trial_starts is None

    def test_init_mismatched_trial_lengths(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        with pytest.raises(ValueError, match="same shape"):
            TrialTensorDataset(
                neural=neural,
                continuous=continuous,
                trial_starts=torch.tensor([0, 250, 500]),
                trial_ends=torch.tensor([250, 500]),
            )

    def test_init_invalid_trial_boundaries(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        with pytest.raises(ValueError, match="less than trial_ends"):
            TrialTensorDataset(
                neural=neural,
                continuous=continuous,
                trial_starts=torch.tensor([0, 500, 250]),
                trial_ends=torch.tensor([250, 400, 500]),
            )

    def test_init_empty_trials(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        with pytest.raises(ValueError, match="cannot be empty"):
            TrialTensorDataset(
                neural=neural,
                continuous=continuous,
                trial_starts=torch.tensor([], dtype=torch.long),
                trial_ends=torch.tensor([], dtype=torch.long),
            )

    def test_init_only_one_trial_param(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        with pytest.raises(ValueError, match="Both trial_starts and trial_ends"):
            TrialTensorDataset(
                neural=neural,
                continuous=continuous,
                trial_starts=torch.tensor([0, 250]),
                trial_ends=None,
            )

    def test_device_placement(self):
        neural = torch.randn(1000, 50)
        continuous = torch.randn(1000, 10)
        trial_starts = torch.tensor([0, 250, 500, 750])
        trial_ends = torch.tensor([250, 500, 750, 1000])

        dataset = TrialTensorDataset(
            neural=neural,
            continuous=continuous,
            trial_starts=trial_starts,
            trial_ends=trial_ends,
            device="cpu",
        )

        assert dataset.neural.device.type == "cpu"
        assert dataset.trial_starts.device.type == "cpu"
