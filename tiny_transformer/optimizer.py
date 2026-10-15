from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """Adam with bias correction and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must lie in [0, 1)")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        # Optimizer's base class manages parameter groups, zero_grad(), and
        # serialization. We provide the defaults stored in each parameter group.
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            # A closure recomputes the loss and therefore must run with gradient
            # tracking even though parameter updates below do not.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                # State is created lazily because some parameters may never receive
                # gradients. Each parameter has its own step, m, and v values.
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # m_t tracks the gradient's exponentially weighted mean.
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)

                # v_t tracks its exponentially weighted mean square, giving Adam
                # a separate adaptive step scale for every parameter element.
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                # Both moment estimates start at zero, so early estimates are
                # biased low. Divide by these factors to correct that bias.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denominator.add_(eps)

                # AdamW decouples weight decay from the gradient/moment update.
                # It directly shrinks the parameter before applying Adam's step.
                parameter.mul_(1.0 - lr * weight_decay)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate for linear warmup followed by cosine decay."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be at least warmup_iters")
    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate cannot exceed max_learning_rate")

    # Warm up along the straight line from (0, 0) to
    # (warmup_iters, max_learning_rate).
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters

    # Once the decay interval is over, hold the learning rate at its floor.
    if iteration > cosine_cycle_iters:
        return min_learning_rate

    # Map the decay interval to progress in [0, 1]. cos(pi * progress)
    # moves from 1 to -1; shifting and halving changes that range to 1 to 0.
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate
    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    # Scale the 1-to-0 cosine factor into [min_learning_rate, max_learning_rate].
    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
