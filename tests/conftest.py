"""Shared test fixtures for trial_cebra tests."""

import torch


def make_trial_data(
    num_trials=10, trial_length=80, embedding_dim=16, gap=20, num_similar_pairs=2, device="cpu"
):
    """Create synthetic trial data with controlled stimulus similarity.

    Returns:
        Tuple of (continuous, trial_starts, trial_ends, total_samples).
    """
    total_samples = num_trials * (trial_length + gap)
    continuous = torch.randn(total_samples, embedding_dim, device=device)

    trial_starts = []
    trial_ends = []
    for i in range(num_trials):
        start = i * (trial_length + gap) + gap
        end = start + trial_length
        trial_starts.append(start)
        trial_ends.append(end)
        continuous[start:end] = torch.randn(1, embedding_dim, device=device)

    trial_starts = torch.tensor(trial_starts, dtype=torch.long, device=device)
    trial_ends = torch.tensor(trial_ends, dtype=torch.long, device=device)

    # Make similar pairs: trial i and trial num_trials//2 + i share embeddings
    half = num_trials // 2
    for i in range(min(num_similar_pairs, half)):
        s_src = int(trial_starts[i])
        s_tgt = int(trial_starts[half + i])
        continuous[s_tgt : s_tgt + trial_length] = continuous[
            s_src : s_src + trial_length
        ] + 0.001 * torch.randn(trial_length, embedding_dim, device=device)

    return continuous, trial_starts, trial_ends, total_samples
