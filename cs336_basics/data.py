from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-token-prediction examples from a 1D token array.

    For a token sequence [10, 20, 30, 40] and context length 3, an example
    input is [10, 20, 30] and its target is [20, 30, 40]. The model therefore
    learns to predict the token one position ahead at every position.
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array of token IDs")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    # A window needs context_length input tokens plus one target token. Thus,
    # the final valid starting index is len(dataset) - context_length - 1.
    num_start_positions = len(dataset) - context_length
    if num_start_positions <= 0:
        raise ValueError("dataset must contain more tokens than context_length")

    # Sample each training example independently and uniformly. np.random.randint
    # excludes its upper bound, so every sampled window stays inside the dataset.
    starts = np.random.randint(0, num_start_positions, size=batch_size)
    offsets = np.arange(context_length)

    # Broadcasting starts[:, None] with offsets[None, :] creates a 2D table
    # of indices with shape (batch_size, context_length).
    x = np.asarray(dataset[starts[:, None] + offsets[None, :]], dtype=np.int64)
    y = np.asarray(dataset[starts[:, None] + offsets[None, :] + 1], dtype=np.int64)

    # Language-model token IDs must be torch.long. Moving to `device` here lets
    # the same loader serve CPU and GPU training.
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
