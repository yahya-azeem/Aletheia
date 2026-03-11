"""Tests for BitLinear layer."""
import torch
from aletheia.model.bitlinear import BitLinear


def test_bitlinear_output_shape():
    """BitLinear produces correct output shape."""
    layer = BitLinear(128, 64)
    x = torch.randn(2, 8, 128)
    out = layer(x)
    assert out.shape == (2, 8, 64)


def test_bitlinear_no_nan():
    """BitLinear output has no NaN values."""
    layer = BitLinear(64, 32)
    x = torch.randn(4, 16, 64)
    out = layer(x)
    assert not torch.isnan(out).any()


def test_bitlinear_gradient_flow():
    """Gradients flow through BitLinear despite quantization (STE)."""
    layer = BitLinear(32, 16)
    x = torch.randn(2, 4, 32, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_bitlinear_weight_ternary_in_forward():
    """During forward, effective weights are ternary."""
    layer = BitLinear(16, 8)
    from aletheia.model.quantization import absmean_quantize
    w_q, _ = absmean_quantize(layer.weight)
    unique = w_q.unique()
    assert all(v in {-1, 0, 1} for v in unique.tolist())


def test_bitlinear_with_bias():
    """BitLinear works with bias enabled."""
    layer = BitLinear(64, 32, bias=True)
    x = torch.randn(2, 4, 64)
    out = layer(x)
    assert out.shape == (2, 4, 32)
    assert layer.bias is not None
