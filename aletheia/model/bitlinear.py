"""
BitLinear Layer (§1.1)

Drop-in replacement for nn.Linear that quantizes weights to ternary {-1, 0, +1}
and activations to 8-bit integers during the forward pass, using Straight-Through
Estimators for gradient flow during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aletheia.model.quantization import absmean_quantize, absmax_quantize


class BitLinear(nn.Module):
    """Ternary-quantized linear layer (BitNet b1.58).

    Weights are stored as full-precision during training but quantized to
    {-1, 0, +1} in the forward pass. Activations are quantized to int8
    per-token. Gradients flow through via STE.

    Args:
        in_features:  Size of input dimension.
        out_features: Size of output dimension.
        bias:         Whether to include a bias term (default: False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weight for training; quantized on-the-fly in forward
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform initialization, same as nn.Linear."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary weight and int8 activation quantization.

        Steps:
            1. Quantize weights to ternary via absmean → W_q, γ
            2. Quantize activations to int8 via absmax  → x_q, α
            3. Compute F.linear(x_q, W_q) in float
            4. Rescale output by (α * γ / 127) to recover correct magnitude
        """
        # (1) Quantize weights: W → {-1, 0, +1}
        w_q, gamma = absmean_quantize(self.weight)

        # (2) Quantize activations: x → int8
        x_q, alpha = absmax_quantize(x, bits=8)

        # (3) Linear projection in float (replaced by CUDA kernel on GPU)
        out = F.linear(x_q.float(), w_q.float(), bias=None)

        # (4) Rescale to recover real-valued output
        scale = (alpha * gamma) / 127.0
        out = out * scale

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
