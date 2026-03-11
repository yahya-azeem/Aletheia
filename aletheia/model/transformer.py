"""
Ternary BitLinear Transformer (§1.1)

Full pre-norm Transformer encoder/decoder stack where every nn.Linear
projection is replaced with BitLinear (ternary weights + int8 activations).
Uses SubLayerNorm for training stability in the quantized regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aletheia.model.bitlinear import BitLinear
from aletheia.model.sublayernorm import SubLayerNorm


@dataclass
class TransformerConfig:
    """Configuration for the BitLinear Transformer."""
    num_layers: int = 24
    hidden_dim: int = 1024
    num_heads: int = 16
    ffn_inner_dim: int = 4096
    max_seq_len: int = 2048
    vocab_size: int = 64000
    dropout: float = 0.0


class BitLinearAttention(nn.Module):
    """Multi-head self-attention with BitLinear projections."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        assert config.hidden_dim % config.num_heads == 0

        self.q_proj = BitLinear(config.hidden_dim, config.hidden_dim)
        self.k_proj = BitLinear(config.hidden_dim, config.hidden_dim)
        self.v_proj = BitLinear(config.hidden_dim, config.hidden_dim)
        self.out_proj = BitLinear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: optional attention mask (batch, 1, seq_len, seq_len)
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class BitLinearFFN(nn.Module):
    """Feed-forward network with BitLinear layers and SiLU activation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.up = BitLinear(config.hidden_dim, config.ffn_inner_dim)
        self.gate = BitLinear(config.hidden_dim, config.ffn_inner_dim)
        self.down = BitLinear(config.ffn_inner_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-style: silu(gate(x)) * up(x)
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class BitLinearTransformerBlock(nn.Module):
    """Single Transformer block with SubLN, BitLinear attention and FFN."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_sublayer = SubLayerNorm(config.hidden_dim)
        self.ffn_sublayer = SubLayerNorm(config.hidden_dim)
        self.attn = BitLinearAttention(config)
        self.ffn = BitLinearFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn_sublayer(x, lambda normed: self.attn(normed, mask))
        x = self.ffn_sublayer(x, self.ffn)
        return x


class BitLinearTransformer(nn.Module):
    """Full BitLinear Transformer encoder stack.

    Used as the backbone for both the VQ-VAE encoder/decoder
    and the diffusion denoiser.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        self.blocks = nn.ModuleList([
            BitLinearTransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize embeddings
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        embeds: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Provide EITHER input_ids (token indices) OR embeds (pre-computed).

        Args:
            input_ids: (batch, seq_len) token indices.
            embeds:    (batch, seq_len, hidden_dim) pre-computed embeddings.
            mask:      optional attention mask.

        Returns:
            Hidden states of shape (batch, seq_len, hidden_dim).
        """
        if embeds is None:
            assert input_ids is not None, "Provide input_ids or embeds"
            B, S = input_ids.shape
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            x = self.token_emb(input_ids) + self.pos_emb(positions)
        else:
            x = embeds

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return self.final_norm(x)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
