"""
VQ-VAE Latent Interlingua (§1.2)

Vector Quantized Variational Autoencoder that forces all representations
through a discrete codebook bottleneck. The codebook acts as the Latent
Interlingua — a language-agnostic symbolic space.

Components:
    - InterlinguaEncoder:  Transformer encoder → continuous latent Z_e
    - VectorQuantizer:     Discrete codebook with EMA updates
    - InterlinguaDecoder:  Transformer decoder → reconstructed tokens
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aletheia.model.bitlinear import BitLinear
from aletheia.model.transformer import BitLinearTransformer, TransformerConfig


@dataclass
class VQVAEConfig:
    """Configuration for the VQ-VAE Latent Interlingua."""
    codebook_size: int = 8192       # K — number of discrete codes
    code_dim: int = 256             # d_code — dimension of each code vector
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    hidden_dim: int = 1024          # must match Transformer hidden_dim
    num_heads: int = 16
    ffn_inner_dim: int = 4096
    max_seq_len: int = 2048
    vocab_size: int = 64000
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    dead_code_threshold: int = 2    # min assignments before reset
    dropout: float = 0.0


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Maps continuous encoder outputs to the nearest discrete codebook entry.
    Uses exponential moving average for stable codebook learning and
    resets dead codes to prevent index collapse.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.K = config.codebook_size
        self.d = config.code_dim
        self.beta = config.commitment_beta
        self.ema_decay = config.ema_decay
        self.dead_threshold = config.dead_code_threshold

        # Codebook embeddings
        self.codebook = nn.Embedding(self.K, self.d)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.K, 1.0 / self.K)

        # EMA tracking
        self.register_buffer("ema_count", torch.zeros(self.K))
        self.register_buffer("ema_weight", self.codebook.weight.clone())

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous encoder output to nearest codebook entry.

        Args:
            z_e: Continuous latent from encoder, shape (batch, seq_len, d_code).

        Returns:
            z_q:     Quantized latent (batch, seq_len, d_code) — with
                     straight-through gradient.
            indices: Codebook indices (batch, seq_len).
            vq_loss: Combined codebook + commitment loss (scalar).
        """
        B, S, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (B*S, D)

        # Nearest-neighbor lookup: ||z_e - e_k||^2
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).T
        )
        indices = distances.argmin(dim=1)  # (B*S,)
        z_q = self.codebook(indices).view(B, S, D)

        # EMA codebook update (training only)
        if self.training:
            self._ema_update(flat, indices)

        # Losses
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through: gradient flows through z_q as if it were z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices.view(B, S), vq_loss

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via exponential moving average."""
        one_hot = F.one_hot(indices, self.K).float()  # (N, K)
        counts = one_hot.sum(dim=0)                    # (K,)
        sums = one_hot.T @ flat                        # (K, D)

        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)

        # Laplace smoothing to avoid division by zero
        n = self.ema_count.sum()
        count_smoothed = (
            (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n
        )
        self.codebook.weight.data.copy_(self.ema_weight / count_smoothed.unsqueeze(1))

        # Reset dead codes
        dead_mask = self.ema_count < self.dead_threshold
        if dead_mask.any():
            # Replace dead codes with random encoder outputs
            num_dead = dead_mask.sum().item()
            random_idx = torch.randint(0, flat.size(0), (num_dead,))
            self.codebook.weight.data[dead_mask] = flat[random_idx]
            self.ema_count[dead_mask] = 1.0
            self.ema_weight[dead_mask] = flat[random_idx]

    def codebook_utilization(self) -> float:
        """Return fraction of codebook entries actively used."""
        return (self.ema_count > self.dead_threshold).float().mean().item()


class InterlinguaEncoder(nn.Module):
    """Encoder: tokens → continuous latent Z_e."""

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        tx_config = TransformerConfig(
            num_layers=config.num_encoder_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_inner_dim=config.ffn_inner_dim,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
        )
        self.transformer = BitLinearTransformer(tx_config)
        # Project from hidden_dim → code_dim
        self.proj = BitLinear(config.hidden_dim, config.code_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs to continuous latent vectors.

        Args:
            input_ids: (batch, seq_len) token indices.

        Returns:
            z_e: (batch, seq_len, code_dim) continuous latent.
        """
        hidden = self.transformer(input_ids=input_ids)
        return self.proj(hidden)


class InterlinguaDecoder(nn.Module):
    """Decoder: quantized latent Z_q → reconstructed token logits."""

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        # Project code_dim → hidden_dim
        self.proj = BitLinear(config.code_dim, config.hidden_dim)

        tx_config = TransformerConfig(
            num_layers=config.num_decoder_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_inner_dim=config.ffn_inner_dim,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
        )
        self.transformer = BitLinearTransformer(tx_config)
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to token logits.

        Args:
            z_q: (batch, seq_len, code_dim) quantized latent.

        Returns:
            logits: (batch, seq_len, vocab_size) token probabilities.
        """
        embeds = self.proj(z_q)
        hidden = self.transformer(embeds=embeds)
        return self.lm_head(hidden)


class VQVAE(nn.Module):
    """Full VQ-VAE Latent Interlingua model.

    Encoder → VectorQuantizer (discrete bottleneck) → Decoder.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = InterlinguaEncoder(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder = InterlinguaDecoder(config)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → quantize → decode.

        Args:
            input_ids: (batch, seq_len) token indices.

        Returns:
            logits:  (batch, seq_len, vocab_size) reconstructed token logits.
            z_q:     (batch, seq_len, code_dim) quantized latents.
            indices: (batch, seq_len) codebook indices.
            vq_loss: VQ-VAE quantization loss (scalar).
        """
        z_e = self.encoder(input_ids)
        z_q, indices, vq_loss = self.quantizer(z_e)
        logits = self.decoder(z_q)
        return logits, z_q, indices, vq_loss

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to discrete codebook indices.

        Returns:
            z_q:     (batch, seq_len, code_dim)
            indices: (batch, seq_len)
        """
        z_e = self.encoder(input_ids)
        z_q, indices, _ = self.quantizer(z_e)
        return z_q, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to token logits."""
        return self.decoder(z_q)

    def compute_loss(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute full VQ-VAE loss (reconstruction + VQ).

        Returns:
            loss: Total scalar loss.
            metrics: Dict of individual loss components for logging.
        """
        logits, z_q, indices, vq_loss = self.forward(input_ids)

        # Reconstruction loss: cross-entropy
        recon_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            input_ids.view(-1),
        )

        total_loss = recon_loss + vq_loss

        metrics = {
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
            "total_loss": total_loss.item(),
            "codebook_util": self.quantizer.codebook_utilization(),
        }
        return total_loss, metrics
