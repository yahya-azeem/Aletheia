"""Tests for the Morphological-Syntactic Engine."""
import torch
from aletheia.model.interlingua import (
    ArabicRootMapper,
    PaniniConstraintLayer,
    MorphoSyntacticEngine,
)


def test_arabic_root_consistency_loss():
    """Root consistency loss is computed for valid root IDs."""
    mapper = ArabicRootMapper(codebook_dim=32, num_root_anchors=64)
    z_q = torch.randn(2, 8, 32)
    root_ids = torch.randint(0, 64, (2, 8))
    loss = mapper.root_consistency_loss(z_q, root_ids)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_arabic_root_no_roots():
    """Root loss is zero when all root IDs are -1 (no roots)."""
    mapper = ArabicRootMapper(codebook_dim=32, num_root_anchors=64)
    z_q = torch.randn(2, 8, 32)
    root_ids = torch.full((2, 8), -1, dtype=torch.long)
    loss = mapper.root_consistency_loss(z_q, root_ids)
    assert loss.item() == 0.0


def test_panini_penalty_shape():
    """Syntactic penalty loss is a valid scalar."""
    panini = PaniniConstraintLayer(codebook_size=64)
    indices = torch.randint(0, 64, (2, 16))
    loss = panini.syntactic_penalty_loss(indices)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_panini_short_sequence():
    """Penalty is zero for sequences too short to have transitions."""
    panini = PaniniConstraintLayer(codebook_size=64)
    indices = torch.randint(0, 64, (2, 1))
    loss = panini.syntactic_penalty_loss(indices)
    assert loss.item() == 0.0


def test_morpho_engine_combined():
    """Combined engine produces total loss and metrics dict."""
    engine = MorphoSyntacticEngine(codebook_size=64, codebook_dim=32, num_root_anchors=32)
    z_q = torch.randn(2, 8, 32)
    indices = torch.randint(0, 64, (2, 8))
    root_ids = torch.randint(-1, 32, (2, 8))
    loss, metrics = engine(z_q, indices, root_ids)
    assert loss.dim() == 0
    assert "root_loss" in metrics
    assert "syntax_loss" in metrics
    assert "morphosyntax_loss" in metrics


def test_panini_with_rules_file():
    """PaniniConstraintLayer loads rules from JSON."""
    import json
    import tempfile
    from pathlib import Path

    rules = [
        {"src_range": [0, 8], "tgt_range": [8, 16], "penalty": 2.0},
        {"src_range": [16, 24], "tgt_range": [0, 8], "penalty": 1.5},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rules, f)
        rules_path = f.name

    panini = PaniniConstraintLayer(codebook_size=64, rules_path=rules_path)
    # Check that penalty matrix was seeded
    assert panini.penalty_matrix.data[0:8, 8:16].abs().sum() > 0
    Path(rules_path).unlink()
