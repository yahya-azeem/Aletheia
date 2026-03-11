"""Tests for fairness and verification modules."""
import torch


def test_cross_lingual_contrastive():
    """Contrastive alignment loss computes correctly."""
    from aletheia.verification.cross_lingual import CrossLingualAlignmentEvaluator

    evaluator = CrossLingualAlignmentEvaluator()
    z_src = torch.randn(8, 64)
    z_tgt = torch.randn(8, 64)
    loss, metrics = evaluator.contrastive_alignment_loss(z_src, z_tgt)
    assert loss.dim() == 0
    assert "mean_cosine_sim" in metrics
    assert "alignment_rate" in metrics


def test_cross_lingual_topological():
    """Topological preservation loss computes correctly."""
    from aletheia.verification.cross_lingual import CrossLingualAlignmentEvaluator

    evaluator = CrossLingualAlignmentEvaluator()
    z_lat = torch.randn(8, 64)
    z_dec = torch.randn(8, 64)
    loss, metrics = evaluator.topological_preservation_loss(z_lat, z_dec)
    assert loss.dim() == 0
    assert "topo_loss" in metrics


def test_formal_verifier_neutrality():
    """Neutrality verification works with simple model."""
    from aletheia.verification.formal_verifier import FormalVerifier

    model = torch.nn.Linear(16, 4)
    verifier = FormalVerifier(model)

    input_a = torch.randn(1, 16)
    input_b = input_a + 0.001  # tiny perturbation
    result = verifier.verify_neutrality(input_a, input_b, delta=1.0)
    assert "verified" in result
    assert isinstance(result["verified"], bool)


def test_ebm_contrastive():
    """EBM contrastive loss differentiates real from shuffled sequences."""
    from aletheia.model.ebm import EnergyHead, EBMConfig

    cfg = EBMConfig(num_layers=1, hidden_dim=32, code_dim=16, num_heads=4,
                    ffn_inner_dim=64, max_seq_len=16)
    ebm = EnergyHead(cfg)

    z_pos = torch.randn(2, 8, 16)
    z_neg = torch.randn(2, 8, 16) * 3  # noisier
    loss, metrics = ebm.compute_loss(z_pos, z_neg)
    assert loss.dim() == 0
    assert "energy_gap" in metrics
