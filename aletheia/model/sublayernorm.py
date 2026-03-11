"""
Sub-Layer Normalization (SubLN) (§1.1)

Applies LayerNorm both BEFORE each sub-layer (attention / FFN) and AFTER
the residual addition, stabilizing gradient flow in the low-precision
ternary weight regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubLayerNorm(nn.Module):
    """Pre- and post-residual LayerNorm wrapper for ternary training stability.

    Usage:
        sub_ln = SubLayerNorm(dim)
        out = sub_ln(x, sublayer_fn)
        # Equivalent to: LayerNorm(x + sublayer_fn(LayerNorm(x)))

    Args:
        dim: Feature dimension for LayerNorm.
        eps: Epsilon for numerical stability (default: 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim, eps=eps)
        self.post_norm = nn.LayerNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: nn.Module | callable,
    ) -> torch.Tensor:
        """Apply pre-norm → sublayer → residual → post-norm.

        Args:
            x:        Input tensor of shape (batch, seq_len, dim).
            sublayer: Callable or nn.Module that processes the normalized input.

        Returns:
            Post-normalized residual output.
        """
        # Pre-norm
        normed = self.pre_norm(x)
        # Sublayer computation (attention or FFN)
        sublayer_out = sublayer(normed)
        # Residual connection + post-norm
        return self.post_norm(x + sublayer_out)
