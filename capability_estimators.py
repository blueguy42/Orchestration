"""
Alternative Capability Estimators for Multi-Agent Orchestration

Implements two alternative distributional models for agent capability
θ_{k,m} = P(A_k | R_m) alongside the existing Beta-Binomial conjugate model,
providing a richer comparison for MPhil dissertation Part 7.

Models
------
BetaBinomialEstimator (reference)
    The default model from Bhatt et al. (2025): θ ~ Beta(α₁, α₀) prior,
    conjugate to Bernoulli likelihood. Closed-form MAP and posterior mean.

TruncatedNormalEstimator
    Prior: θ ~ TruncatedNormal(μ₀, σ₀², [0,1])
    Likelihood: outcome | θ ~ Bernoulli(θ)
    Non-conjugate — posterior computed via 1D Gaussian quadrature (fast and
    exact to machine precision). MAP via bounded scalar minimisation.

    Interpretation: assumes capabilities are roughly normally distributed on
    [0,1] with known prior mean μ₀ and spread σ₀. Natural when you expect
    capabilities to cluster around a central value (e.g. ~0.5) with bounded
    support.

LogisticNormalEstimator
    Prior: φ ~ Normal(μ₀, σ₀²) where θ = σ(φ) = 1/(1+e^{-φ})
    Likelihood: outcome | φ ~ Bernoulli(σ(φ))
    Non-conjugate — MAP via bounded scalar optimisation on φ ∈ [-10,10];
    posterior mean via Laplace approximation:
        E[θ] ≈ σ(φ_MAP / √(1 + π·Var_post/8))   (probit approximation)
    Posterior variance:  Var[θ] ≈ (∂σ/∂φ|_MAP)² · Var_post

    Interpretation: logit-linear parameterisation allows unconstrained normal
    prior on the log-odds scale. More flexible tails than Beta at the boundary
    and the natural conjugate of logistic regression.

All estimators share the same interface as CapabilityEstimator:
    update(agent_idx, region, is_correct)
    get_capability(agent_idx, region)      → MAP estimate
    get_posterior_mean(agent_idx, region)  → posterior mean
    get_posterior_variance(agent_idx, region)
    get_all_capabilities()                 → (K, M) MAP array
    get_all_posterior_means()              → (K, M) posterior mean array
    inject_estimates(estimates, n_virtual)

References
----------
* Beta-Binomial: Murphy (2022) Probabilistic Machine Learning, Ch. 3.
* Truncated Normal posterior: exact via 1D quadrature (Gaussian-Legendre).
* Logistic-Normal / Laplace approx: Murphy (2022) Ch. 10; also known as
  the probit approximation to the sigmoid–Gaussian integral (MacKay, 2003).
"""

import numpy as np
from scipy.special import expit          # σ(x) = 1/(1+e^{-x})
from scipy.optimize import minimize_scalar
from scipy import stats
from scipy.integrate import quad
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared interface mixin (duck-typed; no ABC to keep things simple)
# ─────────────────────────────────────────────────────────────────────────────

class _BaseEstimator:
    """Shared bookkeeping: observation counts and inject_estimates."""

    def __init__(self, K: int, M: int):
        self.K = K
        self.M = M
        # counts[k, m, 0] = n_incorrect,  counts[k, m, 1] = n_correct
        self.counts = np.zeros((K, M, 2))

    def update(self, agent_idx: int, region: int, is_correct: bool):
        self.counts[agent_idx, region, 1 if is_correct else 0] += 1

    def inject_estimates(self, estimates: np.ndarray, n_virtual: int = 50):
        """
        Pre-seed with virtual observations so posterior mean ≈ estimates[k,m].
        Identical to CapabilityEstimator.inject_estimates for comparability.
        """
        for k in range(self.K):
            for m in range(self.M):
                p = np.clip(estimates[k, m], 0.01, 0.99)
                n_cor = int(round(p * n_virtual))
                n_inc = n_virtual - n_cor
                self.counts[k, m, 1] = float(n_cor)
                self.counts[k, m, 0] = float(n_inc)

    def get_all_capabilities(self) -> np.ndarray:
        caps = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                caps[k, m] = self.get_capability(k, m)
        return caps

    def get_all_posterior_means(self) -> np.ndarray:
        means = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                means[k, m] = self.get_posterior_mean(k, m)
        return means

    def _counts(self, agent_idx: int, region: int):
        n_inc = int(self.counts[agent_idx, region, 0])
        n_cor = int(self.counts[agent_idx, region, 1])
        return n_cor, n_inc


# ─────────────────────────────────────────────────────────────────────────────
# 1. Beta-Binomial (reference — mirrors CapabilityEstimator from
#    orchestration_framework.py, self-contained here for clean comparison)
# ─────────────────────────────────────────────────────────────────────────────

class BetaBinomialEstimator(_BaseEstimator):
    """
    Conjugate Beta-Binomial model (reference implementation).

    Prior:   θ ~ Beta(α₁, α₀)        [default: α₁=α₀=1 → uniform]
    Update:  θ | data ~ Beta(α₁+n_cor, α₀+n_inc)   (conjugate)
    MAP:     (α₁+n_cor−1) / (α₀+α₁+n_total−2)     [falls back to mean if ≤0]
    Mean:    (α₁+n_cor) / (α₀+α₁+n_total)
    Var:     a₁·a₀ / (s²·(s+1))  where a₁=α₁+n_cor, a₀=α₀+n_inc, s=a₁+a₀
    """

    name = "Beta-Binomial"

    def __init__(self, K: int, M: int, alpha0: float = 1.0, alpha1: float = 1.0):
        super().__init__(K, M)
        self.alpha0 = alpha0
        self.alpha1 = alpha1

    def get_capability(self, agent_idx: int, region: int) -> float:
        n_cor, n_inc = self._counts(agent_idx, region)
        n_tot = n_cor + n_inc
        denom = self.alpha0 + self.alpha1 + n_tot - 2
        if denom > 0:
            return float(np.clip((self.alpha1 + n_cor - 1) / denom, 0.0, 1.0))
        return (self.alpha1 + n_cor) / (self.alpha0 + self.alpha1 + n_tot)

    def get_posterior_mean(self, agent_idx: int, region: int) -> float:
        n_cor, n_inc = self._counts(agent_idx, region)
        return (self.alpha1 + n_cor) / (self.alpha0 + self.alpha1 + n_cor + n_inc)

    def get_posterior_variance(self, agent_idx: int, region: int) -> float:
        n_cor, n_inc = self._counts(agent_idx, region)
        a1 = self.alpha1 + n_cor
        a0 = self.alpha0 + n_inc
        s  = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Truncated-Normal model
# ─────────────────────────────────────────────────────────────────────────────

class TruncatedNormalEstimator(_BaseEstimator):
    """
    Truncated-Normal prior on θ ∈ [0, 1].

    Prior:   θ ~ TruncatedNormal(μ₀, σ₀², a=0, b=1)
    Likelihood: outcome | θ ~ Bernoulli(θ)

    Non-conjugate: the posterior is
        p(θ | data) ∝ θ^n_cor · (1−θ)^n_inc · TN(θ; μ₀, σ₀²)

    Posterior mean and variance are computed via adaptive 1D Gauss-Legendre
    quadrature (fast and numerically exact).  MAP is found via bounded scalar
    minimisation on [ε, 1−ε].

    Parameters
    ----------
    mu0    : prior mean (default 0.5 — diffuse centre)
    sigma0 : prior standard deviation (default 0.25 — covers [0,1] well)
    """

    name = "Truncated-Normal"

    def __init__(self, K: int, M: int, mu0: float = 0.5, sigma0: float = 0.25):
        super().__init__(K, M)
        self.mu0    = mu0
        self.sigma0 = sigma0

    def _log_unnorm_post(self, theta: float, n_cor: int, n_inc: int) -> float:
        if theta <= 0.0 or theta >= 1.0:
            return -np.inf
        ll = n_cor * np.log(theta) + n_inc * np.log(1.0 - theta)
        a  = (0.0 - self.mu0) / self.sigma0
        b  = (1.0 - self.mu0) / self.sigma0
        lp = stats.truncnorm.logpdf(theta, a, b, loc=self.mu0, scale=self.sigma0)
        return ll + lp

    def _unnorm_post(self, theta: float, n_cor: int, n_inc: int) -> float:
        v = self._log_unnorm_post(theta, n_cor, n_inc)
        return np.exp(v) if np.isfinite(v) else 0.0

    def _normalise(self, n_cor: int, n_inc: int):
        """Return (Z, E[θ], E[θ²]) via quadrature."""
        fn   = lambda t: self._unnorm_post(t, n_cor, n_inc)
        eps  = 1e-8
        Z,   _ = quad(fn,            eps, 1 - eps, limit=100)
        Em,  _ = quad(lambda t: t      * fn(t), eps, 1 - eps, limit=100)
        Em2, _ = quad(lambda t: t**2   * fn(t), eps, 1 - eps, limit=100)
        if Z < 1e-15:
            return 1.0, self.mu0, self.mu0**2 + self.sigma0**2
        return Z, Em / Z, Em2 / Z

    def get_capability(self, agent_idx: int, region: int) -> float:
        """MAP estimate (bounded scalar optimisation)."""
        n_cor, n_inc = self._counts(agent_idx, region)
        result = minimize_scalar(
            lambda t: -self._log_unnorm_post(t, n_cor, n_inc),
            bounds=(1e-6, 1.0 - 1e-6), method='bounded'
        )
        return float(np.clip(result.x, 0.0, 1.0))

    def get_posterior_mean(self, agent_idx: int, region: int) -> float:
        n_cor, n_inc = self._counts(agent_idx, region)
        _, mean, _   = self._normalise(n_cor, n_inc)
        return float(np.clip(mean, 0.0, 1.0))

    def get_posterior_variance(self, agent_idx: int, region: int) -> float:
        n_cor, n_inc = self._counts(agent_idx, region)
        _, mean, mean2 = self._normalise(n_cor, n_inc)
        return float(max(mean2 - mean**2, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logistic-Normal model (Laplace approximation)
# ─────────────────────────────────────────────────────────────────────────────

class LogisticNormalEstimator(_BaseEstimator):
    """
    Logistic-Normal (normal prior on log-odds) with Laplace approximation.

    Model:   φ ~ Normal(μ₀, σ₀²),   θ = σ(φ) = 1/(1+e^{-φ})
    Likelihood: outcome | φ ~ Bernoulli(σ(φ))

    Log-posterior of φ:
        log p(φ | data) = n_cor·log σ(φ) + n_inc·log(1−σ(φ)) − (φ−μ₀)²/(2σ₀²) + const

    MAP φ̂: bounded scalar minimisation on φ ∈ [−10, 10].
    Laplace approximation: posterior of φ ≈ N(φ̂, Σ_post) where
        Σ_post = −(d²/dφ² log p(φ|data))|_{φ̂}^{−1}
               = [n_cor·(1−θ̂)² + n_inc·θ̂²  +  1/σ₀²]^{−1}

    Posterior mean (probit approximation, MacKay 2003):
        E[θ] ≈ σ(φ̂ / √(1 + π·Σ_post/8))

    Posterior variance (delta method):
        Var[θ] ≈ (∂σ/∂φ|_{φ̂})² · Σ_post  =  [θ̂(1−θ̂)]² · Σ_post

    Parameters
    ----------
    mu0      : prior mean in log-odds space (default 0.0 → prior mean θ=0.5)
    sigma0   : prior std in log-odds space  (default 1.0 → prior θ ∈ [0.27,0.73] at ±1σ)
    """

    name = "Logistic-Normal"

    def __init__(self, K: int, M: int, mu0: float = 0.0, sigma0: float = 1.0):
        super().__init__(K, M)
        self.mu0    = mu0
        self.sigma0 = sigma0

    def _log_post(self, phi: float, n_cor: int, n_inc: int) -> float:
        theta = float(expit(phi))
        ll = n_cor * np.log(theta + 1e-15) + n_inc * np.log(1.0 - theta + 1e-15)
        lp = -0.5 * (phi - self.mu0)**2 / (self.sigma0**2)
        return ll + lp

    def _map_and_laplace(self, n_cor: int, n_inc: int):
        """Return (phi_MAP, theta_MAP, var_post)."""
        result = minimize_scalar(
            lambda phi: -self._log_post(phi, n_cor, n_inc),
            bounds=(-10.0, 10.0), method='bounded'
        )
        phi_map  = result.x
        theta_map = float(expit(phi_map))
        # Hessian of log-posterior at MAP (negative definite):
        # d²log-lik/dφ² = −[n_cor·(1−θ)² + n_inc·θ²]   (exact for Bernoulli–logistic)
        hess_ll    = -(n_cor * (1.0 - theta_map)**2 + n_inc * theta_map**2)
        hess_prior = -1.0 / (self.sigma0**2)
        hess_total  = hess_ll + hess_prior
        var_post = -1.0 / hess_total if hess_total < -1e-12 else self.sigma0**2
        return phi_map, theta_map, var_post

    def get_capability(self, agent_idx: int, region: int) -> float:
        """MAP estimate: θ̂ = σ(φ̂)."""
        n_cor, n_inc = self._counts(agent_idx, region)
        _, theta_map, _ = self._map_and_laplace(n_cor, n_inc)
        return float(np.clip(theta_map, 0.0, 1.0))

    def get_posterior_mean(self, agent_idx: int, region: int) -> float:
        """
        Posterior mean via probit approximation:
            E[σ(φ)] ≈ σ(φ̂ / √(1 + π·Var_post/8))
        """
        n_cor, n_inc = self._counts(agent_idx, region)
        phi_map, _, var_post = self._map_and_laplace(n_cor, n_inc)
        mean = float(expit(phi_map / np.sqrt(1.0 + np.pi * var_post / 8.0)))
        return float(np.clip(mean, 0.0, 1.0))

    def get_posterior_variance(self, agent_idx: int, region: int) -> float:
        """
        Posterior variance via delta method:
            Var[θ] ≈ [θ̂(1−θ̂)]² · Var_post
        """
        n_cor, n_inc = self._counts(agent_idx, region)
        _, theta_map, var_post = self._map_and_laplace(n_cor, n_inc)
        dtheta_dphi = theta_map * (1.0 - theta_map)
        return float(dtheta_dphi**2 * var_post)


# ─────────────────────────────────────────────────────────────────────────────
# Registry for Part 7 experiments
# ─────────────────────────────────────────────────────────────────────────────

ESTIMATOR_CLASSES = [
    (BetaBinomialEstimator,    {"alpha0": 1.0, "alpha1": 1.0}, "Beta-Binomial"),
    (TruncatedNormalEstimator, {"mu0": 0.5, "sigma0": 0.25},   "Truncated-Normal"),
    (LogisticNormalEstimator,  {"mu0": 0.0, "sigma0": 1.0},    "Logistic-Normal"),
]

ESTIMATOR_COLORS = {
    "Beta-Binomial":    "#2196F3",
    "Truncated-Normal": "#FF9800",
    "Logistic-Normal":  "#4CAF50",
}
