"""Trial-structured dataset for CEBRA.

Extends CEBRA's TensorDataset with trial boundary metadata to enable
hierarchical trial-aware sampling.
"""

from typing import Union

import cebra.data
import numpy as np
import numpy.typing as npt
import torch
from cebra.data.datatypes import Offset


class TrialTensorDataset(cebra.data.TensorDataset):
    """TensorDataset with trial metadata support for trial-aware sampling.

    Extends :py:class:`cebra.data.TensorDataset` with trial boundary
    information. This enables hierarchical trial-aware sampling via
    :py:class:`trial_cebra.distribution.TrialAwareDistribution`.

    Args:
        neural: Array of shape ``(N, D)`` containing neural activity.
        continuous: Array of shape ``(N, d)`` containing continuous variables.
            For trial-aware sampling, this should contain stimulus embeddings
            that are constant within each trial.
        discrete: Array of shape ``(N,)`` containing discrete variables.
        trial_starts: Array of shape ``(num_trials,)`` with start index of each trial.
        trial_ends: Array of shape ``(num_trials,)`` with end index (exclusive) of each trial.
        offset: Time offset for the model architecture.
        device: Compute device (``cpu`` or ``cuda``).

    Example:

        >>> import torch
        >>> neural = torch.randn(1000, 50)
        >>> continuous = torch.randn(1000, 10)
        >>> trial_starts = torch.tensor([0, 250, 500, 750])
        >>> trial_ends = torch.tensor([250, 500, 750, 1000])
        >>> dataset = TrialTensorDataset(
        ...     neural=neural,
        ...     continuous=continuous,
        ...     trial_starts=trial_starts,
        ...     trial_ends=trial_ends,
        ... )
    """

    def __init__(
        self,
        neural: Union[torch.Tensor, npt.NDArray],
        continuous: Union[torch.Tensor, npt.NDArray] = None,
        discrete: Union[torch.Tensor, npt.NDArray] = None,
        trial_starts: Union[torch.Tensor, npt.NDArray] = None,
        trial_ends: Union[torch.Tensor, npt.NDArray] = None,
        offset: Offset = Offset(0, 1),
        device: str = "cpu",
    ):
        super().__init__(neural, continuous, discrete, offset, device)
        self._parse_trial_metadata(trial_starts, trial_ends)

    def _parse_trial_metadata(self, trial_starts, trial_ends):
        """Parse and validate trial metadata.

        Raises:
            ValueError: If trial metadata is invalid or inconsistent.
        """
        if trial_starts is None and trial_ends is None:
            return

        if trial_starts is None or trial_ends is None:
            raise ValueError("Both trial_starts and trial_ends must be provided together.")

        # Convert to tensors
        if isinstance(trial_starts, np.ndarray):
            trial_starts = torch.from_numpy(trial_starts)
        elif not isinstance(trial_starts, torch.Tensor):
            trial_starts = torch.tensor(trial_starts)

        if isinstance(trial_ends, np.ndarray):
            trial_ends = torch.from_numpy(trial_ends)
        elif not isinstance(trial_ends, torch.Tensor):
            trial_ends = torch.tensor(trial_ends)

        if len(trial_starts) == 0:
            raise ValueError("trial_starts and trial_ends cannot be empty")

        if trial_starts.shape != trial_ends.shape:
            raise ValueError(
                f"trial_starts and trial_ends must have same shape, "
                f"got {trial_starts.shape} and {trial_ends.shape}"
            )

        if not torch.all(trial_starts < trial_ends):
            raise ValueError("All trial_starts must be less than trial_ends")

        self.trial_starts = trial_starts.long().to(self.device)
        self.trial_ends = trial_ends.long().to(self.device)
