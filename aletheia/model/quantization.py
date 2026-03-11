"""
Aletheia Quantization Primitives (§1.1)

Implements the BitNet b1.58 quantization functions:
  - absmean_quantize: Weights → ternary {-1, 0, +1}
  - absmax_quantize:  Activations → int8
  - STERound:         Straight-Through Estimator for non-differentiable rounding
"""

from __future__ import annotations

import torch
import torch.nn as nn


class STERound(torch.autograd.Function):
    """Straight-Through Estimator for rounding.

    Forward: applies torch.round (non-differentiable).
    Backward: passes gradients through unchanged (identity).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # pass-through


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper around STERound.apply."""
    return STERound.apply(x)


def absmean_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to ternary {-1, 0, +1} using absmean scaling.

    Algorithm:
        γ = mean(|W|)
        Ŵ = clamp(round(W / γ), -1, 1)

    Args:
        weight: Float weight tensor of shape (out_features, in_features).

    Returns:
        Tuple of (quantized_weight, scale γ).
    """
    gamma = weight.abs().mean()
    # Avoid division by zero for zero-initialized weights
    gamma = gamma.clamp(min=1e-8)
    scaled = weight / gamma
    quantized = ste_round(scaled).clamp(-1, 1)
    return quantized, gamma


def absmax_quantize(
    x: torch.Tensor,
    bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to int-N using per-token absmax scaling.

    Algorithm (for 8-bit):
        α = max(|x|)  per-token
        x̂ = clamp(round(x × 127 / α), -128, 127)

    Args:
        x: Float activation tensor of shape (..., dim).
           The last dimension is treated as the feature dimension;
           scaling is per-token (all dims except last).
        bits: Quantization bit-width (default 8).

    Returns:
        Tuple of (quantized_activation, scale α).
    """
    qmax = 2 ** (bits - 1) - 1  # 127 for 8-bit
    qmin = -(2 ** (bits - 1))    # -128 for 8-bit

    # Per-token scale: max absolute value across the feature dimension
    alpha = x.abs().amax(dim=-1, keepdim=True)
    alpha = alpha.clamp(min=1e-8)

    scaled = x * qmax / alpha
    quantized = ste_round(scaled).clamp(qmin, qmax)
    return quantized, alpha
