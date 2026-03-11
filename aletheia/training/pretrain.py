"""
Pretraining Loop for Project Aletheia

Orchestrates the multi-stage training pipeline:
  Stage 1: VQ-VAE pretraining (reconstruction + quantization)
  Stage 2: Full model (Transformer + VQ-VAE + morpho-syntactic constraints)
  Stage 3: Diffusion head training
  Stage 4: EBM fine-tuning + metaphorical losses
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from aletheia.model.transformer import TransformerConfig
from aletheia.model.vqvae import VQVAE, VQVAEConfig
from aletheia.model.diffusion import ContinuousTextDiffusion, DiffusionConfig
from aletheia.model.ebm import EnergyHead, EBMConfig
from aletheia.model.interlingua import MorphoSyntacticEngine
from aletheia.losses.entropy_reg import EntropyRegularization

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_vqvae(cfg: dict) -> VQVAE:
    """Build VQ-VAE model from config dict."""
    model_cfg = cfg["model"]
    vq_cfg = cfg["vqvae"]
    return VQVAE(VQVAEConfig(
        codebook_size=vq_cfg["codebook_size"],
        code_dim=vq_cfg["code_dim"],
        num_encoder_layers=vq_cfg["num_encoder_layers"],
        num_decoder_layers=vq_cfg["num_decoder_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        ffn_inner_dim=model_cfg["ffn_inner_dim"],
        max_seq_len=model_cfg["max_seq_len"],
        vocab_size=model_cfg["vocab_size"],
        commitment_beta=vq_cfg["commitment_beta"],
        dropout=model_cfg.get("dropout", 0.0),
    ))


def build_diffusion(cfg: dict) -> ContinuousTextDiffusion:
    """Build diffusion model from config dict."""
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    return ContinuousTextDiffusion(DiffusionConfig(
        num_timesteps=diff_cfg["num_timesteps"],
        num_denoiser_layers=diff_cfg["num_denoiser_layers"],
        code_dim=cfg["vqvae"]["code_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        ffn_inner_dim=model_cfg["ffn_inner_dim"],
        max_seq_len=model_cfg["max_seq_len"],
        schedule=diff_cfg["schedule"],
        sampling_steps=diff_cfg["sampling_steps"],
        sampling_method=diff_cfg["sampling_method"],
        dropout=model_cfg.get("dropout", 0.0),
    ))


def build_ebm(cfg: dict) -> EnergyHead:
    """Build EBM head from config dict."""
    model_cfg = cfg["model"]
    ebm_cfg = cfg["ebm"]
    return EnergyHead(EBMConfig(
        num_layers=ebm_cfg["num_layers"],
        hidden_dim=ebm_cfg["hidden_dim"],
        code_dim=cfg["vqvae"]["code_dim"],
        num_heads=model_cfg["num_heads"],
        ffn_inner_dim=model_cfg["ffn_inner_dim"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg.get("dropout", 0.0),
    ))


def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    """Build AdamW optimizer from training config."""
    train_cfg = cfg["training"]
    return AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(optimizer: AdamW, cfg: dict):
    """Build LR scheduler: linear warmup + cosine decay."""
    train_cfg = cfg["training"]
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=train_cfg["warmup_steps"],
    )
    decay = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["max_steps"] - train_cfg["warmup_steps"],
    )
    return SequentialLR(optimizer, [warmup, decay], milestones=[train_cfg["warmup_steps"]])


def train_vqvae(
    vqvae: VQVAE,
    cfg: dict,
    device: torch.device,
    max_steps: int | None = None,
) -> None:
    """Stage 1: Pretrain VQ-VAE on reconstruction."""
    train_cfg = cfg["training"]
    max_steps = max_steps or train_cfg["max_steps"]

    optimizer = build_optimizer(vqvae, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    entropy_reg = EntropyRegularization(
        codebook_size=cfg["vqvae"]["codebook_size"],
    )

    logger.info(f"Stage 1: VQ-VAE pretraining on {device} for {max_steps} steps")
    vqvae.train()

    for step in range(1, max_steps + 1):
        # Synthetic data for smoke testing (replaced by real DataLoader in production)
        batch_size = train_cfg["batch_size"]
        seq_len = min(train_cfg["max_seq_len"], 256)
        input_ids = torch.randint(0, cfg["model"]["vocab_size"], (batch_size, seq_len), device=device)

        loss, metrics = vqvae.compute_loss(input_ids)

        # Entropy regularization
        _, _, indices, _ = vqvae(input_ids)
        ent_loss, ent_metrics = entropy_reg(indices)
        loss = loss + ent_loss
        metrics.update(ent_metrics)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == 1:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Step {step}/{max_steps}] loss={metrics['total_loss']:.4f} "
                f"recon={metrics['recon_loss']:.4f} vq={metrics['vq_loss']:.4f} "
                f"util={metrics['codebook_util']:.2%} lr={lr:.2e}"
            )

    logger.info("Stage 1 complete.")


def main():
    parser = argparse.ArgumentParser(description="Aletheia Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Training stage")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    device = torch.device(train_cfg.get("device", "cpu"))

    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")

    if args.stage == 1:
        vqvae = build_vqvae(cfg).to(device)
        param_count = sum(p.numel() for p in vqvae.parameters())
        logger.info(f"VQ-VAE parameters: {param_count:,}")
        train_vqvae(vqvae, cfg, device, max_steps=args.max_steps)
    else:
        logger.info(f"Stage {args.stage} not yet implemented. Run --stage 1 first.")


if __name__ == "__main__":
    main()
