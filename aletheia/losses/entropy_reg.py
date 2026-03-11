"""
Representation Entropy Minimization (§3.3)

Regularizes codebook usage distribution: prevents both codebook collapse
(all tokens mapping to a few codes) and uniform spread (no decisive selection).
Encourages sparse, meaningful code selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EntropyRegularization(nn.Module):
    """Codebook usage entropy regularizer.

    H(Z) = -Σ p(z_k) log p(z_k)

    We want entropy to stay in a target range:
      - Too low → codebook collapse (bad)
      - Too high → uniform / no structure (bad)

    Args:
        codebook_size: K — number of codebook entries.
        target_entropy_ratio: Target entropy as fraction of max entropy (default 0.7).
        weight: Loss weight (default 0.01).
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        target_entropy_ratio: float = 0.7,
        weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.K = codebook_size
        self.max_entropy = torch.log(torch.tensor(float(codebook_size)))
        self.target_entropy = target_entropy_ratio * self.max_entropy.item()
        self.weight = weight

    def forward(
        self, codebook_indices: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute entropy regularization loss.

        Args:
            codebook_indices: (batch, seq_len) codebook indices from VQ-VAE.

        Returns:
            loss:    Scalar regularization loss.
            metrics: Dict with entropy stats.
        """
        flat = codebook_indices.reshape(-1)

        # Compute usage histogram
        counts = torch.bincount(flat, minlength=self.K).float()
        probs = counts / counts.sum().clamp(min=1)

        # Shannon entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum()

        # Penalize deviation from target entropy
        loss = self.weight * (entropy - self.target_entropy) ** 2

        metrics = {
            "codebook_entropy": entropy.item(),
            "max_entropy": self.max_entropy.item(),
            "entropy_ratio": (entropy / self.max_entropy).item(),
            "entropy_reg_loss": loss.item(),
        }
        return loss, metrics
