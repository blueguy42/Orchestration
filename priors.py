"""
Prior distributions for the Beta-Binomial CapabilityEstimator.

Each prior encodes initial beliefs about agent correctness probabilities θ_{k,m}
before any observations are collected.  They are used by CapabilityEstimator
(orchestration_framework.py) and all teacher classes (machine_teaching.py) to
compute MAP estimates and posterior means via Beta-Binomial conjugacy.

Three prior families:
1. BetaPrior         — conjugate Beta(α₁, α₀); default Beta(1,1) = uniform
2. JeffreysPrior     — non-informative Beta(0.5, 0.5); invariant under
                       re-parameterisation, less biased at the boundaries
3. SkewedExpertPrior — informative Beta(3, 1); encodes the belief that
                       agents are generally competent (prior mean = 0.75)

All priors expose:
    - alpha0, alpha1, name
    - prior_mean(), posterior_mean(n_correct, n_incorrect)
    - posterior_variance(n_correct, n_incorrect)
    - map_estimate(n_correct, n_incorrect)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class BetaPrior:
    """
    Conjugate Beta prior.
    alpha0 = pseudo-count for 'incorrect'; alpha1 = pseudo-count for 'correct'.
    Default: Beta(1,1) = uniform.
    """
    alpha0: float = 1.0
    alpha1: float = 1.0
    name: str = "Beta(1,1) — Uniform (Dirichlet)"

    def prior_mean(self) -> float:
        return self.alpha1 / (self.alpha0 + self.alpha1)

    def prior_variance(self) -> float:
        a, b = self.alpha1, self.alpha0
        s = a + b
        return (a * b) / (s * s * (s + 1))

    def posterior_params(self, n_correct: int, n_incorrect: int) -> Tuple[float, float]:
        return self.alpha1 + n_correct, self.alpha0 + n_incorrect

    def posterior_mean(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        return a1 / (a1 + a0)

    def map_estimate(self, n_correct: int, n_incorrect: int) -> float:
        """
        MAP estimate (mode of Beta posterior).
        Mode = (α₁ + n_cor − 1) / (α₀ + α₁ + n_total − 2)
        Falls back to posterior mean when denominator ≤ 0.
        """
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        denom = a1 + a0 - 2
        if denom > 0:
            return float(np.clip((a1 - 1) / denom, 0.0, 1.0))
        # Fallback to posterior mean
        return a1 / (a1 + a0)

    def posterior_variance(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))


@dataclass
class JeffreysPrior:
    """
    Jeffreys prior: Beta(0.5, 0.5).
    Invariant under re-parameterisation.  More diffuse at both extremes
    than the uniform prior; tends to produce less biased estimates.
    """
    alpha0: float = 0.5
    alpha1: float = 0.5
    name: str = "Jeffreys — Beta(0.5, 0.5)"

    def prior_mean(self) -> float:
        return 0.5

    def prior_variance(self) -> float:
        s = self.alpha0 + self.alpha1
        return (self.alpha1 * self.alpha0) / (s * s * (s + 1))

    def posterior_params(self, n_correct: int, n_incorrect: int) -> Tuple[float, float]:
        return self.alpha1 + n_correct, self.alpha0 + n_incorrect

    def posterior_mean(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        return a1 / (a1 + a0)

    def map_estimate(self, n_correct: int, n_incorrect: int) -> float:
        """MAP with Jeffreys: denominator can be ≤ 0 with few obs; use posterior mean."""
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        denom = a1 + a0 - 2
        if denom > 0:
            return float(np.clip((a1 - 1) / denom, 0.0, 1.0))
        return a1 / (a1 + a0)

    def posterior_variance(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))


@dataclass
class SkewedExpertPrior:
    """
    Informative 'expert-optimistic' prior: Beta(3, 1).
    Prior mean = 0.75.  Encodes the belief that agents are generally competent.
    """
    alpha0: float = 1.0
    alpha1: float = 3.0
    name: str = "Skewed Expert — Beta(3, 1)"

    def prior_mean(self) -> float:
        return self.alpha1 / (self.alpha0 + self.alpha1)

    def prior_variance(self) -> float:
        a, b = self.alpha1, self.alpha0
        s = a + b
        return (a * b) / (s * s * (s + 1))

    def posterior_params(self, n_correct: int, n_incorrect: int) -> Tuple[float, float]:
        return self.alpha1 + n_correct, self.alpha0 + n_incorrect

    def posterior_mean(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        return a1 / (a1 + a0)

    def map_estimate(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        denom = a1 + a0 - 2
        if denom > 0:
            return float(np.clip((a1 - 1) / denom, 0.0, 1.0))
        return a1 / (a1 + a0)

    def posterior_variance(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

ALL_PRIORS = [
    BetaPrior(),
    JeffreysPrior(),
    SkewedExpertPrior(),
]

PRIOR_LABELS = {
    "Beta(1,1) — Uniform (Dirichlet)":  "Uniform β(1,1)",
    "Jeffreys — Beta(0.5, 0.5)":        "Jeffreys β(½,½)",
    "Skewed Expert — Beta(3, 1)":        "Expert β(3,1)",
}

PRIOR_COLORS = {
    "Beta(1,1) — Uniform (Dirichlet)":  "#2196F3",
    "Jeffreys — Beta(0.5, 0.5)":        "#FF9800",
    "Skewed Expert — Beta(3, 1)":        "#4CAF50",
}
