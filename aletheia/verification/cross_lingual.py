"""
Cross-Lingual Alignment Evaluator (§4.2)

Verifies that the VQ-VAE Latent Interlingua preserves semantic topology
across languages using contrastive learning and distance preservation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLingualAlignmentEvaluator(nn.Module):
    """Evaluates cross-lingual alignment quality of the Latent Interlingua.

    Two metrics:
    1. Inter-sentence contrastive similarity (parallel pairs should be close)
    2. Topological preservation (pairwise distances preserved after decoding)
    """

    def __init__(self, temperature: float = 0.07, cosine_threshold: float = 0.85) -> None:
        super().__init__()
        self.temperature = temperature
        self.cosine_threshold = cosine_threshold

    def contrastive_alignment_loss(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Inter-sentence contrastive loss for parallel sentence pairs.

        Positive pairs: (z_src[i], z_tgt[i]) should have high cosine similarity.
        Negative pairs: all cross-pairs should have lower similarity.

        Args:
            z_src: (batch, dim) source language latent means (pooled over seq).
            z_tgt: (batch, dim) target language latent means.

        Returns:
            loss:    InfoNCE contrastive loss.
            metrics: Dict with alignment stats.
        """
        z_src = F.normalize(z_src, dim=-1)
        z_tgt = F.normalize(z_tgt, dim=-1)

        # Similarity matrix
        logits = z_src @ z_tgt.T / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2

        # Metrics: diagonal cosine similarity
        cos_sim = (z_src * z_tgt).sum(dim=-1)
        alignment_rate = (cos_sim > self.cosine_threshold).float().mean()

        metrics = {
            "contrastive_loss": loss.item(),
            "mean_cosine_sim": cos_sim.mean().item(),
            "alignment_rate": alignment_rate.item(),
        }
        return loss, metrics

    def topological_preservation_loss(
        self,
        z_latent: torch.Tensor,
        z_decoded: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Verify pairwise distance preservation between latent and output space.

        Minimizes Frobenius norm of (D_latent - D_decoded) where D is the
        pairwise distance matrix.

        Args:
            z_latent:  (batch, dim) latent representations.
            z_decoded: (batch, dim) decoded output representations.

        Returns:
            loss:    Frobenius norm of distance matrix difference.
            metrics: Dict with preservation stats.
        """
        # Pairwise cosine distance matrices
        z_lat_norm = F.normalize(z_latent, dim=-1)
        z_dec_norm = F.normalize(z_decoded, dim=-1)

        d_latent = 1 - z_lat_norm @ z_lat_norm.T   # (B, B)
        d_decoded = 1 - z_dec_norm @ z_dec_norm.T   # (B, B)

        # Frobenius norm of difference
        loss = (d_latent - d_decoded).pow(2).mean().sqrt()

        metrics = {
            "topo_loss": loss.item(),
            "mean_latent_dist": d_latent.mean().item(),
            "mean_decoded_dist": d_decoded.mean().item(),
        }
        return loss, metrics
