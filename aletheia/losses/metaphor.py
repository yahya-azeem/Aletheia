"""
Metaphorical Distance Reward Loss (§3.3)

Rewards moderate semantic distance between source and target conceptual
domains — too close is banal synonymy, too far is incoherent.
Uses a Gaussian window centered on the optimal "creative distance."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaphorDistanceLoss(nn.Module):
    """Gaussian-window loss rewarding moderate cosine distance.

    Optimal metaphor creativity lies at moderate semantic distance
    (0.3–0.7 cosine distance), per cognitive metaphor theory.

    Args:
        center:  Center of the optimal distance window (default 0.5).
        sigma:   Standard deviation of the Gaussian window (default 0.15).
        weight:  Loss weight multiplier (default 0.1).
    """

    def __init__(
        self,
        center: float = 0.5,
        sigma: float = 0.15,
        weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.center = center
        self.sigma = sigma
        self.weight = weight

    def forward(
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute metaphorical distance loss.

        Args:
            source_embeds: (batch, dim) source domain embeddings.
            target_embeds: (batch, dim) target domain embeddings.

        Returns:
            loss:    Scalar loss (higher when distance is outside optimal window).
            metrics: Dict with distance stats.
        """
        # Cosine distance (1 - cosine_similarity)
        cos_sim = F.cosine_similarity(source_embeds, target_embeds, dim=-1)
        cos_dist = 1.0 - cos_sim

        # Gaussian window: maximum reward at center, decaying outward
        # Loss = -log(Gaussian(distance | center, sigma))
        log_prob = -0.5 * ((cos_dist - self.center) / self.sigma) ** 2
        loss = -log_prob.mean() * self.weight

        metrics = {
            "metaphor_loss": loss.item(),
            "mean_cos_dist": cos_dist.mean().item(),
            "std_cos_dist": cos_dist.std().item(),
        }
        return loss, metrics
