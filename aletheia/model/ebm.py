"""
Energy-Based Diffusion Language Model (EDLM) (§3.2)

Scores entire latent sequences with a global "energy" value.
Low energy = high probability = coherent, well-structured text.
Integrated into the diffusion reverse process to guide denoising
toward globally coherent outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from aletheia.model.bitlinear import BitLinear
from aletheia.model.transformer import TransformerConfig, BitLinearTransformerBlock


@dataclass
class EBMConfig:
    """Configuration for the Energy-Based Model head."""
    num_layers: int = 4
    hidden_dim: int = 1024
    code_dim: int = 256
    num_heads: int = 16
    ffn_inner_dim: int = 4096
    max_seq_len: int = 2048
    dropout: float = 0.0


class EnergyHead(nn.Module):
    """Global energy scoring head for latent sequences.

    Architecture: Input projection → Transformer → Mean pooling → MLP → Scalar

    The energy E_φ(z) represents -log p_φ(z):
      - Lower energy = higher probability = more coherent sequence
      - Higher energy = lower probability = incoherent / noisy sequence
    """

    def __init__(self, config: EBMConfig) -> None:
        super().__init__()
        self.input_proj = BitLinear(config.code_dim, config.hidden_dim)

        tx_config = TransformerConfig(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_inner_dim=config.ffn_inner_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([
            BitLinearTransformerBlock(tx_config)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # MLP to scalar energy
        self.energy_mlp = nn.Sequential(
            BitLinear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            BitLinear(config.hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute scalar energy for a latent sequence.

        Args:
            z: (batch, seq_len, code_dim) latent sequence.

        Returns:
            energy: (batch,) scalar energy per sequence.
        """
        h = self.input_proj(z)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)

        # Mean pooling across sequence
        pooled = h.mean(dim=1)  # (B, H)

        # Scalar energy
        energy = self.energy_mlp(pooled).squeeze(-1)  # (B,)
        return energy

    def compute_loss(
        self,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Contrastive divergence loss for EBM training.

        Positive samples = real latent sequences from VQ-VAE.
        Negative samples = corrupted / shuffled / noised sequences.

        Loss: E(positive) - E(negative) + regularization
        Lower energy for positives, higher for negatives.

        Args:
            z_positive: (batch, seq_len, code_dim) real sequences.
            z_negative: (batch, seq_len, code_dim) corrupted sequences.

        Returns:
            loss:    Scalar contrastive loss.
            metrics: Dict with energy values for logging.
        """
        e_pos = self.forward(z_positive)
        e_neg = self.forward(z_negative)

        # Contrastive divergence: push positive energies down, negative up
        loss = e_pos.mean() - e_neg.mean()

        # Regularization to prevent energy from diverging
        reg = 0.01 * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
        loss = loss + reg

        metrics = {
            "energy_pos": e_pos.mean().item(),
            "energy_neg": e_neg.mean().item(),
            "energy_gap": (e_neg.mean() - e_pos.mean()).item(),
            "ebm_loss": loss.item(),
        }
        return loss, metrics

    @torch.no_grad()
    def energy_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Compute ∇_z E(z) for guiding diffusion denoising.

        Used during inference to steer the denoising process toward
        lower-energy (more coherent) latent configurations.

        Args:
            z: (batch, seq_len, code_dim) latent to evaluate.

        Returns:
            grad: (batch, seq_len, code_dim) energy gradient.
        """
        z_in = z.detach().requires_grad_(True)
        energy = self.forward(z_in)
        energy.sum().backward()
        return z_in.grad.detach()
