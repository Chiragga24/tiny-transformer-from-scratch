from __future__ import annotations

from collections.abc import Iterable

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean cross-entropy from unnormalized class logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, num_classes)")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape (batch_size,)")

    # log(sum(exp(logits))) can overflow for large logits. Subtracting the row
    # maximum before exponentiation keeps the calculation in a safe range.
    row_max = logits.max(dim=-1, keepdim=True).values
    log_normalizer = row_max.squeeze(-1) + torch.log(
        torch.exp(logits - row_max).sum(dim=-1)
    )

    # Gather the single correct-class logit from every row. Cross-entropy for
    # one example is logsumexp(all logits) minus its correct-class logit.
    correct_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    per_example_loss = log_normalizer - correct_logits
    return per_example_loss.mean()


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip the combined L2 norm of parameter gradients in place."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")

    # Some parameters can be frozen or unused in the current computation, in
    # which case autograd leaves grad as None. Materialize the list because the
    # incoming parameter iterable may be a one-use generator.
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    # Treat every parameter gradient as one segment of a single large vector.
    # The norm of that concatenated vector is sqrt(sum of every squared entry).
    total_squared_norm = torch.zeros(
        (),
        device=gradients[0].device,
        dtype=gradients[0].dtype,
    )
    for gradient in gradients:
        total_squared_norm += gradient.detach().square().sum()
    total_norm = torch.sqrt(total_squared_norm)

    # A shared factor preserves the global gradient direction. Clamp at 1 so
    # gradients already below the threshold are never enlarged.
    clip_coefficient = max_l2_norm / (total_norm + 1e-6)
    clip_coefficient = torch.clamp(clip_coefficient, max=1.0)

    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
