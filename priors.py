"""
priors.py
=========
Prior distributions for the Beta-Binomial CapabilityEstimator.

Three prior families are provided:

1. BetaPrior (Dirichlet / conjugate) — the default used everywhere else.
   Parameterised by (alpha0, alpha1); uniform Beta(1,1) is the standard
   non-informative choice.

2. JeffreysPrior — the objective, invariant prior Beta(0.5, 0.5).
   It is the unique prior that is invariant under re-parameterisation and
   is equivalent to Jeffreys' rule for a Bernoulli likelihood.

3. SkewedExpertPrior — an *informative* prior that encodes the belief that
   agents tend to be good (successes more likely); Beta(a, b) with a > b.

All priors expose:
    - alpha0   : pseudo-count for 'incorrect'
    - alpha1   : pseudo-count for 'correct'
    - name     : human-readable label
    - prior_mean() : E[θ] under the prior
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class BetaPrior:
    """
    Conjugate Beta prior (Dirichlet in the region-marginal sense).

    Parameters
    ----------
    alpha0 : float
        Pseudo-count for 'incorrect' outcomes.  1.0 → uniform prior.
    alpha1 : float
        Pseudo-count for 'correct' outcomes.   1.0 → uniform prior.
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
        """Return (alpha1_post, alpha0_post) after observing data."""
        return self.alpha1 + n_correct, self.alpha0 + n_incorrect

    def posterior_mean(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        return a1 / (a1 + a0)

    def posterior_variance(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))


@dataclass
class JeffreysPrior:
    """
    Jeffreys prior: Beta(0.5, 0.5).

    This is the unique prior that is invariant under bijective
    re-parameterisations of the Bernoulli/Binomial model.  It is more
    diffuse at both extremes than the uniform prior and tends to produce
    less biased posterior estimates when data are scarce.
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

    def posterior_variance(self, n_correct: int, n_incorrect: int) -> float:
        a1, a0 = self.posterior_params(n_correct, n_incorrect)
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))


@dataclass
class SkewedExpertPrior:
    """
    Informative 'expert-optimistic' prior: Beta(3, 1).

    Encodes the belief that deployed agents are generally competent
    (P(correct) biased toward 1).  Prior mean = 0.75.

    This is analogous to inserting 3 virtual 'correct' observations and
    1 virtual 'incorrect' observation before seeing any real data.
    Using a stronger prior (larger alpha0+alpha1) slows adaptation but
    regularises estimation when data are very scarce.
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
