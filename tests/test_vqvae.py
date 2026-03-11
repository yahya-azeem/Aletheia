"""Tests for VQ-VAE Latent Interlingua."""
import torch
from aletheia.model.vqvae import VQVAE, VQVAEConfig, VectorQuantizer


def _tiny_config() -> VQVAEConfig:
    """Minimal config for CPU testing."""
    return VQVAEConfig(
        codebook_size=64,
        code_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        hidden_dim=64,
        num_heads=4,
        ffn_inner_dim=128,
        max_seq_len=32,
        vocab_size=100,
        dropout=0.0,
    )


def test_vqvae_forward_shape():
    """VQ-VAE produces correct output shapes."""
    cfg = _tiny_config()
    model = VQVAE(cfg)
    ids = torch.randint(0, 100, (2, 16))
    logits, z_q, indices, vq_loss = model(ids)
    assert logits.shape == (2, 16, 100)
    assert z_q.shape == (2, 16, 32)
    assert indices.shape == (2, 16)
    assert vq_loss.dim() == 0  # scalar


def test_vqvae_encode_decode():
    """Encode-decode round trip produces valid outputs."""
    cfg = _tiny_config()
    model = VQVAE(cfg)
    ids = torch.randint(0, 100, (2, 8))
    z_q, indices = model.encode(ids)
    logits = model.decode(z_q)
    assert logits.shape == (2, 8, 100)


def test_vqvae_loss():
    """compute_loss returns scalar loss and metrics dict."""
    cfg = _tiny_config()
    model = VQVAE(cfg)
    ids = torch.randint(0, 100, (2, 8))
    loss, metrics = model.compute_loss(ids)
    assert loss.dim() == 0
    assert "recon_loss" in metrics
    assert "vq_loss" in metrics
    assert "codebook_util" in metrics


def test_codebook_indices_range():
    """Codebook indices are within valid range [0, K)."""
    cfg = _tiny_config()
    model = VQVAE(cfg)
    ids = torch.randint(0, 100, (4, 16))
    _, indices = model.encode(ids)
    assert indices.min() >= 0
    assert indices.max() < cfg.codebook_size


def test_codebook_usage():
    """Codebook utilization is reported correctly."""
    cfg = _tiny_config()
    vq = VectorQuantizer(cfg)
    z = torch.randn(8, 16, 32)
    vq.train()
    _, _, _ = vq(z)
    util = vq.codebook_utilization()
    assert 0.0 <= util <= 1.0


def test_vqvae_gradient_flow():
    """Gradients flow through VQ-VAE via straight-through estimator."""
    cfg = _tiny_config()
    model = VQVAE(cfg)
    ids = torch.randint(0, 100, (2, 8))
    loss, _ = model.compute_loss(ids)
    loss.backward()
    # Check encoder has gradients
    for p in model.encoder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break
