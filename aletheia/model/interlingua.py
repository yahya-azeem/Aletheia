"""
Morphological-Syntactic Engine (§1.3)

Constrains the VQ-VAE codebook structure via:
  1. Arabic Jadhr (Root) Mapper — trilateral root anchors in the codebook
  2. Sanskrit Aṣṭādhyāyī FST — deterministic syntactic constraint layer

These produce auxiliary losses that guide the codebook topology toward
linguistically meaningful partitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArabicRootMapper(nn.Module):
    """Maps Arabic trilateral roots to codebook partitions.

    The first `num_root_anchors` entries in the codebook are designated
    as "root anchors."  During training, tokens whose surface form
    decomposes to the same trilateral root are encouraged to map to
    codes within the same root partition (via a hinge loss).

    Args:
        codebook_dim:      Dimension of each codebook vector.
        num_root_anchors:  Number of codebook entries reserved for roots
                           (default 2048 ↔ ~2000 known roots).
        margin:            Hinge-loss margin for root consistency.
        roots_path:        Path to JSON file mapping root → id.
    """

    def __init__(
        self,
        codebook_dim: int = 256,
        num_root_anchors: int = 2048,
        margin: float = 0.5,
        roots_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_root_anchors = num_root_anchors
        self.margin = margin

        # Root-to-ID mapping (loaded from JSON or empty for stub)
        self.root_to_id: dict[str, int] = {}
        if roots_path and Path(roots_path).exists():
            with open(roots_path, "r", encoding="utf-8") as f:
                self.root_to_id = json.load(f)

        # Learnable root centroids for each partition
        self.centroids = nn.Parameter(
            torch.randn(num_root_anchors, codebook_dim) * 0.02
        )

    def root_consistency_loss(
        self,
        z_q: torch.Tensor,
        root_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute root consistency hinge loss.

        Encourages tokens with the same root to cluster near the
        corresponding root centroid in the codebook space.

        Args:
            z_q:      Quantized latents (batch, seq_len, d_code).
            root_ids: Root partition IDs per token (batch, seq_len).
                      -1 indicates no root (non-Arabic or unknown).

        Returns:
            Scalar loss.
        """
        mask = root_ids >= 0  # valid root assignments
        if not mask.any():
            return torch.tensor(0.0, device=z_q.device)

        valid_z = z_q[mask]          # (N_valid, d_code)
        valid_ids = root_ids[mask]   # (N_valid,)

        centroids = self.centroids[valid_ids]  # (N_valid, d_code)
        distances = (valid_z - centroids).pow(2).sum(dim=-1).sqrt()

        # Hinge: penalize only when distance exceeds margin
        loss = F.relu(distances - self.margin).mean()
        return loss


class PaniniConstraintLayer(nn.Module):
    """Sanskrit Aṣṭādhyāyī syntactic constraint layer.

    Encodes Pāṇinian grammar rules as a differentiable finite-state
    transducer (FST) over codebook indices. Violations produce a
    penalty loss that guides the VQ-VAE toward syntactically valid
    latent sequences.

    Args:
        codebook_size:  Total number of VQ-VAE codebook entries.
        num_rules:      Number of encoded Pāṇinian rules (default 200).
        rules_path:     Path to JSON defining the rule transitions.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        num_rules: int = 200,
        rules_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.num_rules = num_rules

        # Learnable penalty matrix: (codebook_size, codebook_size)
        # penalty[i, j] = soft penalty for transitioning from code i to code j
        # Initialized near zero; shaped by rules during training
        self.penalty_matrix = nn.Parameter(
            torch.zeros(codebook_size, codebook_size) * 0.01
        )

        # Load rule definitions (if available)
        self.rules: list[dict[str, Any]] = []
        if rules_path and Path(rules_path).exists():
            with open(rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)

        # Initialize penalty matrix from rules
        self._init_from_rules()

    def _init_from_rules(self) -> None:
        """Seed penalty matrix from Pāṇinian rule definitions.

        Each rule specifies forbidden or costly transitions between
        syntactic categories mapped to codebook index ranges.
        """
        for rule in self.rules:
            src_range = rule.get("src_range", [0, 0])
            tgt_range = rule.get("tgt_range", [0, 0])
            penalty_val = rule.get("penalty", 1.0)

            src_start, src_end = src_range
            tgt_start, tgt_end = tgt_range

            if src_end <= self.codebook_size and tgt_end <= self.codebook_size:
                self.penalty_matrix.data[src_start:src_end, tgt_start:tgt_end] = penalty_val

    def syntactic_penalty_loss(
        self,
        codebook_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute syntactic violation penalty over a sequence of codes.

        For each consecutive pair (i, j) in the codebook index sequence,
        look up penalty_matrix[i, j] and sum violations.

        Args:
            codebook_indices: (batch, seq_len) codebook index sequence.

        Returns:
            Scalar penalty loss.
        """
        if codebook_indices.size(1) < 2:
            return torch.tensor(0.0, device=codebook_indices.device)

        # Consecutive pairs
        src = codebook_indices[:, :-1]  # (B, S-1)
        tgt = codebook_indices[:, 1:]   # (B, S-1)

        # Clamp indices to valid range
        src = src.clamp(0, self.codebook_size - 1)
        tgt = tgt.clamp(0, self.codebook_size - 1)

        # Soft penalties - use sigmoid for differentiability
        penalties = torch.sigmoid(self.penalty_matrix[src, tgt])

        return penalties.mean()


class MorphoSyntacticEngine(nn.Module):
    """Combined Arabic root + Sanskrit syntax constraint engine.

    Produces auxiliary losses that shape the VQ-VAE codebook topology.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_dim: int = 256,
        num_root_anchors: int = 2048,
        roots_path: str | Path | None = None,
        rules_path: str | Path | None = None,
        root_loss_weight: float = 0.1,
        syntax_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.root_mapper = ArabicRootMapper(
            codebook_dim=codebook_dim,
            num_root_anchors=num_root_anchors,
            roots_path=roots_path,
        )
        self.panini = PaniniConstraintLayer(
            codebook_size=codebook_size,
            rules_path=rules_path,
        )
        self.root_weight = root_loss_weight
        self.syntax_weight = syntax_loss_weight

    def forward(
        self,
        z_q: torch.Tensor,
        codebook_indices: torch.Tensor,
        root_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute morphological + syntactic constraint losses.

        Args:
            z_q:               Quantized latents (batch, seq_len, d_code).
            codebook_indices:  VQ-VAE indices (batch, seq_len).
            root_ids:          Optional Arabic root IDs (batch, seq_len).

        Returns:
            loss:    Weighted combination of root + syntax losses.
            metrics: Dict of individual losses for logging.
        """
        # Arabic root consistency
        if root_ids is not None:
            root_loss = self.root_mapper.root_consistency_loss(z_q, root_ids)
        else:
            root_loss = torch.tensor(0.0, device=z_q.device)

        # Sanskrit syntactic constraints
        syntax_loss = self.panini.syntactic_penalty_loss(codebook_indices)

        total = self.root_weight * root_loss + self.syntax_weight * syntax_loss

        metrics = {
            "root_loss": root_loss.item(),
            "syntax_loss": syntax_loss.item(),
            "morphosyntax_loss": total.item(),
        }
        return total, metrics
