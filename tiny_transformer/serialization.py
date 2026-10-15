from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


CheckpointDestination = str | os.PathLike | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointDestination,
) -> None:
    """Serialize all state needed to resume training."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")

    # state_dict stores tensors and optimizer metadata without pickling the
    # Python class instances themselves. This keeps restoration explicit.
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointDestination,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model/optimizer state and return the saved iteration."""
    # Loading onto the model's device makes CPU/GPU checkpoint restoration
    # portable. Models without parameters fall back to CPU.
    first_parameter = next(model.parameters(), None)
    device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    checkpoint = torch.load(src, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
