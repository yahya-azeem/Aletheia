"""
Multivariate Information Bottleneck Loss (§3.3)

Compresses surface syntactic noise (minimize I(X; Z)) while preserving
the core semantic intent (maximize I(Z; Y)). Estimated via variational bounds.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InformationBottleneckLoss(nn.Module):
    """Variational Information Bottleneck.

    L_IB = I(X; Z) - β · I(Z; Y)

    Uses a learned critic network to estimate mutual information.

    Args:
        input_dim:  Dimension of input representations.
        latent_dim: Dimension of latent (codebook) representations.
        target_dim: Dimension of target representations.
        beta:       Trade-off between compression and preservation (default 1.0).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 256,
        target_dim: int = 1024,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta

        # Critic for I(X; Z) — input-latent mutual information
        self.critic_xz = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Critic for I(Z; Y) — latent-target mutual information
        self.critic_zy = nn.Sequential(
            nn.Linear(latent_dim + target_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _estimate_mi(
        self,
        critic: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate mutual information via MINE (Mutual Information Neural Estimation).

        Uses the Donsker-Varadhan representation:
            I(X;Y) ≥ E[T(x,y)] - log E[exp(T(x,y'))]
        where y' is sampled from the marginal.
        """
        # Joint samples
        joint = torch.cat([x, y], dim=-1)
        joint_score = critic(joint)

        # Marginal samples (shuffle y)
        perm = torch.randperm(y.size(0))
        y_shuffled = y[perm]
        marginal = torch.cat([x, y_shuffled], dim=-1)
        marginal_score = critic(marginal)

        # DV bound
        mi = joint_score.mean() - torch.logsumexp(marginal_score, dim=0) + torch.log(
            torch.tensor(float(y.size(0)))
        )
        return mi

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Information Bottleneck loss.

        Args:
            x: Input representations (batch, dim).
            z: Latent representations (batch, latent_dim).
            y: Target representations (batch, dim).

        Returns:
            loss:    I(X;Z) - β · I(Z;Y)
            metrics: Dict with MI estimates.
        """
        mi_xz = self._estimate_mi(self.critic_xz, x, z)
        mi_zy = self._estimate_mi(self.critic_zy, z, y)

        loss = mi_xz - self.beta * mi_zy

        metrics = {
            "mi_xz": mi_xz.item(),
            "mi_zy": mi_zy.item(),
            "ib_loss": loss.item(),
        }
        return loss, metrics
