"""Tests for the continuous text diffusion model."""
import torch
from aletheia.model.diffusion import ContinuousTextDiffusion, DiffusionConfig


def _tiny_config() -> DiffusionConfig:
    return DiffusionConfig(
        num_timesteps=50,
        num_denoiser_layers=2,
        code_dim=32,
        hidden_dim=64,
        num_heads=4,
        ffn_inner_dim=128,
        max_seq_len=32,
        schedule="cosine",
        sampling_steps=5,
        sampling_method="ddim",
        dropout=0.0,
    )


def test_diffusion_noise_schedule():
    """Noise schedule has correct shape and monotonicity."""
    cfg = _tiny_config()
    model = ContinuousTextDiffusion(cfg)
    alphas = model.alphas_cumprod
    assert alphas.shape[0] == cfg.num_timesteps + 1
    # Cosine schedule is monotonically decreasing
    assert (alphas[:-1] >= alphas[1:]).all()


def test_q_sample_shape():
    """Forward noising preserves shape."""
    cfg = _tiny_config()
    model = ContinuousTextDiffusion(cfg)
    z_0 = torch.randn(2, 8, 32)
    t = torch.tensor([10, 25])
    z_t = model.q_sample(z_0, t)
    assert z_t.shape == z_0.shape


def test_diffusion_loss():
    """Training loss is a valid scalar."""
    cfg = _tiny_config()
    model = ContinuousTextDiffusion(cfg)
    z_0 = torch.randn(2, 8, 32)
    loss, metrics = model.compute_loss(z_0)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert "diffusion_loss" in metrics


def test_diffusion_gradient_flow():
    """Gradients flow through diffusion training."""
    cfg = _tiny_config()
    model = ContinuousTextDiffusion(cfg)
    z_0 = torch.randn(2, 8, 32)
    loss, _ = model.compute_loss(z_0)
    loss.backward()
    for p in model.denoiser.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_diffusion_sample():
    """Sampling produces correct shapes."""
    cfg = _tiny_config()
    model = ContinuousTextDiffusion(cfg)
    codebook = torch.nn.Embedding(64, 32)
    z_0, indices = model.sample((2, 8, 32), codebook)
    assert z_0.shape == (2, 8, 32)
    assert indices.shape == (2, 8)
    assert indices.min() >= 0
    assert indices.max() < 64
