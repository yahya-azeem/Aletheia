"""
Formal Verification Engine (§4.1)

Integrates the α-CROWN neural network verifier (auto_LiRPA) to
mathematically prove safety constraints on sub-networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class VerificationSpec:
    """Polytopic safety specification for formal verification.

    Defined as: C · output ≤ d
    where C is a constraint matrix and d is a bound vector.
    """
    constraint_matrix: torch.Tensor  # (num_constraints, output_dim)
    bound_vector: torch.Tensor       # (num_constraints,)
    epsilon: float = 0.01            # input perturbation bound
    description: str = ""


class FormalVerifier:
    """Formal neural network verifier using α-CROWN / auto_LiRPA.

    Verifies that a sub-network satisfies linear constraints for all
    inputs within an ε-ball of a reference input.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._lirpa_available = self._check_lirpa()

    @staticmethod
    def _check_lirpa() -> bool:
        """Check if auto_LiRPA is installed."""
        try:
            import auto_LiRPA  # noqa: F401
            return True
        except ImportError:
            return False

    def verify(
        self,
        reference_input: torch.Tensor,
        spec: VerificationSpec,
    ) -> dict[str, Any]:
        """Verify that the model satisfies spec for all inputs in ε-ball.

        Args:
            reference_input: Reference input tensor.
            spec:            Safety specification to verify.

        Returns:
            Dict with keys: "verified" (bool), "bounds" (tensor), "details" (str).
        """
        if not self._lirpa_available:
            return {
                "verified": False,
                "bounds": None,
                "details": "auto_LiRPA not installed. Install via: pip install auto_LiRPA",
            }

        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

        # Wrap model in bounded module
        bounded_model = BoundedModule(
            self.model,
            reference_input,
            bound_opts={"optimize_bound_args": {"ob_iteration": 20}},
        )

        # Define perturbation
        ptb = PerturbationLpNorm(norm=float("inf"), eps=spec.epsilon)
        bounded_input = BoundedTensor(reference_input, ptb)

        # Compute bounds
        lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="CROWN")

        # Check constraints: C · output ≤ d
        # Worst case: use upper bounds for positive coefficients, lower for negative
        C = spec.constraint_matrix.to(lb.device)
        d = spec.bound_vector.to(lb.device)

        worst_case = torch.where(C > 0, C @ ub.T, C @ lb.T).sum(dim=1)
        verified = (worst_case <= d).all().item()

        return {
            "verified": verified,
            "bounds": {"lower": lb, "upper": ub},
            "details": f"Spec '{spec.description}': {'VERIFIED' if verified else 'FAILED'}",
        }

    def verify_neutrality(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        delta: float = 0.01,
    ) -> dict[str, Any]:
        """Verify demographic neutrality: outputs for swapped inputs are bounded.

        Checks that |f(input_a) - f(input_b)| < delta for all outputs.

        Args:
            input_a: First input (e.g., original prompt).
            input_b: Second input (e.g., demographic-swapped prompt).
            delta:   Maximum allowed output difference.
        """
        with torch.no_grad():
            out_a = self.model(input_a)
            out_b = self.model(input_b)

        diff = (out_a - out_b).abs()
        max_diff = diff.max().item()
        verified = max_diff < delta

        return {
            "verified": verified,
            "max_difference": max_diff,
            "delta_threshold": delta,
            "details": f"Max diff {max_diff:.6f} {'<' if verified else '>='} δ={delta}",
        }
