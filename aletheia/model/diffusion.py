"""
Continuous-Time Text Diffusion (§3.1)

Non-autoregressive generation engine that "sculpts" text by iteratively
denoising a random Gaussian sample in the continuous VQ-VAE latent space.

Forward process:  z_0 → z_t  (add noise)
Reverse process:  z_T → z_0  (denoise via learned network)
Rounding step:    z_0_pred → nearest VQ-VAE codebook entry
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aletheia.model.bitlinear import BitLinear
from aletheia.model.sublayernorm import SubLayerNorm
from aletheia.model.transformer import TransformerConfig, BitLinearTransformerBlock


@dataclass
class DiffusionConfig:
    """Configuration for the continuous text diffusion model."""
    num_timesteps: int = 1000
    num_denoiser_layers: int = 12
    code_dim: int = 256            # must match VQ-VAE code_dim
    hidden_dim: int = 1024
    num_heads: int = 16
    ffn_inner_dim: int = 4096
    max_seq_len: int = 2048
    schedule: str = "cosine"       # "cosine" or "linear"
    sampling_steps: int = 50       # DDIM steps at inference
    sampling_method: str = "ddim"  # "ddpm" or "ddim"
    dropout: float = 0.0


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → adaptive layer norm conditioning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps to continuous vectors.

        Args:
            t: (batch,) integer timesteps.

        Returns:
            (batch, dim) timestep embeddings.
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(emb)


class DenoisingTransformer(nn.Module):
    """Transformer-based denoiser for the continuous diffusion process.

    Predicts the clean latent z_0 from a noisy input z_t, conditioned
    on the diffusion timestep t via adaptive layer norms.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.input_proj = BitLinear(config.code_dim, config.hidden_dim)
        self.output_proj = BitLinear(config.hidden_dim, config.code_dim)

        self.time_emb = TimestepEmbedding(config.hidden_dim)

        tx_config = TransformerConfig(
            num_layers=config.num_denoiser_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_inner_dim=config.ffn_inner_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([
            BitLinearTransformerBlock(tx_config)
            for _ in range(config.num_denoiser_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict clean z_0 from noisy z_t at timestep t.

        Args:
            z_t: (batch, seq_len, code_dim) noisy latent.
            t:   (batch,) integer timesteps.

        Returns:
            z_0_pred: (batch, seq_len, code_dim) predicted clean latent.
        """
        # Project to hidden dim + add timestep conditioning
        h = self.input_proj(z_t)
        t_emb = self.time_emb(t).unsqueeze(1)  # (B, 1, H)
        h = h + t_emb  # broadcast along sequence

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        return self.output_proj(h)


class ContinuousTextDiffusion(nn.Module):
    """Continuous-time text diffusion model operating in VQ-VAE latent space.

    Manages the noise schedule, forward noising, reverse denoising,
    and codebook rounding.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = DenoisingTransformer(config)

        # Precompute noise schedule
        self.register_buffer(
            "alphas_cumprod", self._build_schedule(config)
        )

    @staticmethod
    def _build_schedule(config: DiffusionConfig) -> torch.Tensor:
        """Build cumulative alpha schedule."""
        T = config.num_timesteps
        if config.schedule == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float32) / T
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        elif config.schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod = torch.cat([torch.ones(1), alphas_cumprod])
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
        return alphas_cumprod

    def q_sample(
        self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward process: add noise to z_0 at timestep t.

        z_t = √(ᾱ_t) · z_0 + √(1 - ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        alpha_bar = self.alphas_cumprod[t]  # (B,)
        # Reshape for broadcasting: (B, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1)

        return torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1 - alpha_bar) * noise

    def compute_loss(
        self, z_0: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Training loss: MSE between z_0 and predicted z_0.

        Args:
            z_0: (batch, seq_len, code_dim) clean latent from VQ-VAE.

        Returns:
            loss:    Scalar MSE loss.
            metrics: Dict with loss value for logging.
        """
        B = z_0.size(0)
        # Sample random timesteps
        t = torch.randint(1, self.config.num_timesteps + 1, (B,), device=z_0.device)

        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)

        z_0_pred = self.denoiser(z_t, t)
        loss = F.mse_loss(z_0_pred, z_0)

        return loss, {"diffusion_loss": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        codebook: nn.Embedding,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples via reverse diffusion (DDIM).

        Args:
            shape:    (batch, seq_len, code_dim) shape to generate.
            codebook: VQ-VAE codebook for rounding.
            device:   Target device.

        Returns:
            z_0:     Final denoised latent (batch, seq_len, code_dim).
            indices: Rounded codebook indices (batch, seq_len).
        """
        # Start from pure noise
        z = torch.randn(shape, device=device)

        # DDIM timestep subsequence
        total_steps = self.config.num_timesteps
        step_size = total_steps // self.config.sampling_steps
        timesteps = list(range(total_steps, 0, -step_size))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict clean z_0
            z_0_pred = self.denoiser(z, t_batch)

            if i < len(timesteps) - 1:
                # DDIM update: jump to next timestep
                t_prev = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t_prev]

                # Reshape for broadcasting
                alpha_t = alpha_t.view(1, 1, 1)
                alpha_prev = alpha_prev.view(1, 1, 1)

                # Predicted noise
                eps_pred = (z - torch.sqrt(alpha_t) * z_0_pred) / torch.sqrt(1 - alpha_t)

                # DDIM deterministic step
                z = (
                    torch.sqrt(alpha_prev) * z_0_pred
                    + torch.sqrt(1 - alpha_prev) * eps_pred
                )
            else:
                z = z_0_pred

        # Round to nearest codebook entry
        B, S, D = z.shape
        flat = z.reshape(-1, D)
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.weight.T
            + codebook.weight.pow(2).sum(1, keepdim=True).T
        )
        indices = distances.argmin(dim=1).view(B, S)
        z_0 = codebook(indices)

        return z_0, indices
