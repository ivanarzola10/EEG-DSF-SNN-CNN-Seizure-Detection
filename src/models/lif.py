"""Discrete leaky integrate-and-fire (LIF) neuron with a surrogate gradient.

Update for one time step (proposal Eq. 1; Zhang et al. 2024 Eq. 2):

    u_half = leak * u + I               leak and integrate
    s      = H(u_half - v_th)           spike if the threshold is reached
    u_new  = u_half - v_th * s          soft reset: subtract the threshold

H is the Heaviside step, which has zero gradient almost everywhere. For training, its
derivative is replaced by a rectangular surrogate (Zhang et al. 2024, h1):

    dS/du = 1/width   if |u - v_th| < width / 2,   else 0

The EESNN reference code uses width = 1, where 1/width = 1 matches its gradient exactly.
As in that code, the spike in the reset term is not detached, so gradients also flow
through the reset.
"""
from __future__ import annotations

import torch


class RectangularSpike(torch.autograd.Function):
    """Heaviside spike ``u >= v_th`` with a rectangular surrogate gradient."""

    @staticmethod
    def forward(ctx, u: torch.Tensor, v_th: float, width: float) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.v_th = v_th
        ctx.width = width
        return (u >= v_th).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (u,) = ctx.saved_tensors
        window = (torch.abs(u - ctx.v_th) < ctx.width / 2).to(u.dtype)
        return grad_out * window / ctx.width, None, None


def spike(u: torch.Tensor, v_th: float, width: float = 1.0) -> torch.Tensor:
    return RectangularSpike.apply(u, v_th, width)


def lif_step(
    u: torch.Tensor,
    current: torch.Tensor,
    v_th: float,
    leak: float,
    width: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance LIF neurons one step; returns ``(u_new, spikes)`` with the shape of ``current``."""
    u_half = leak * u + current
    s = spike(u_half, v_th, width)
    return u_half - v_th * s, s
