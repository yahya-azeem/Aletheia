"""Tests for quantization primitives."""
import torch
from aletheia.model.quantization import absmean_quantize, absmax_quantize, ste_round


def test_ste_round_forward():
    """STE rounding produces correct integer values."""
    x = torch.tensor([0.3, 0.7, -0.3, -0.7, 1.5, -1.5])
    result = ste_round(x)
    expected = torch.tensor([0.0, 1.0, 0.0, -1.0, 2.0, -2.0])
    assert torch.equal(result, expected)


def test_ste_round_gradient():
    """STE passes gradients through unchanged."""
    x = torch.tensor([0.3, 0.7, -0.5], requires_grad=True)
    y = ste_round(x)
    loss = y.sum()
    loss.backward()
    # Gradient should be all ones (identity pass-through)
    assert torch.equal(x.grad, torch.ones_like(x))


def test_absmean_ternary():
    """absmean_quantize produces only {-1, 0, +1} values."""
    w = torch.randn(64, 128)
    w_q, gamma = absmean_quantize(w)
    unique = w_q.unique()
    assert all(v in {-1, 0, 1} for v in unique.tolist()), f"Non-ternary values: {unique}"


def test_absmean_scale_positive():
    """absmean scale gamma is positive."""
    w = torch.randn(32, 64)
    _, gamma = absmean_quantize(w)
    assert gamma > 0


def test_absmax_int8_range():
    """absmax_quantize produces values in [-128, 127]."""
    x = torch.randn(4, 16, 128)
    x_q, alpha = absmax_quantize(x, bits=8)
    assert x_q.min() >= -128
    assert x_q.max() <= 127


def test_absmax_scale_shape():
    """absmax scale has correct shape for per-token scaling."""
    x = torch.randn(4, 16, 128)
    _, alpha = absmax_quantize(x, bits=8)
    assert alpha.shape == (4, 16, 1)


def test_zero_weights():
    """absmean_quantize handles zero weights gracefully."""
    w = torch.zeros(8, 8)
    w_q, gamma = absmean_quantize(w)
    assert not torch.isnan(w_q).any()
    assert not torch.isinf(w_q).any()
