"""
Machine Teaching for Efficient Capability Identification in Multi-Agent Orchestration

Implements a machine teaching approach for efficiently identifying agent capabilities
across task regions, extending Bhatt et al. (2025) as described in the MPhil
dissertation proposal.

Instead of passively observing a random task stream, a teacher strategically selects
which (agent, region) slot to probe next, minimising the evaluation budget required
to reach accurate capability estimates.

The "student" is a Beta-Binomial CapabilityEstimator (orchestration_framework.py)
that maintains a posterior over each agent's correctness probability θ_{k,m} in each
region.  The teacher drives all K × M estimates close to their true values as quickly
as possible (minimising MSE), then hands them to a downstream orchestrator.

Teachers (inspired by Liu et al. 2017 "Iterative Machine Teaching", ICML)
--------------------------------------------------------------------------
OmniscientTeacher  — Knows true capabilities θ*; selects the slot whose next
                     observation gives the greatest expected reduction in MSE
                     (EMSR criterion, closed-form for Beta-Binomial posterior mean).
                     Theoretical upper bound on teaching efficiency.

SurrogateTeacher   — Does not know θ*; selects the slot with the highest
                     posterior variance (uncertainty sampling / A-optimal design).
                     Note: this is NOT identical to max query entropy
                     (H[Bernoulli(θ̂)]) — variance peaks at α₁=α₀ while query
                     entropy peaks at θ̂=0.5.  In practice the two are highly
                     correlated but the criterion used here is posterior variance.
                     Also note: under the uniform Beta(1,1) prior all slots start
                     with identical variance, so the surrogate teacher behaves like
                     round-robin for the first K×M steps; a random tiebreak is
                     applied to improve initial diversity.

ImitationTeacher   — Bridges omniscient and surrogate.  Maintains a
                     Robbins-Monro running estimate v_{k,m} of θ*_{k,m} (by
                     averaging observed binary outcomes with a decaying rate
                     η = c / (c + n_{k,m})) and uses v in the EMSR formula in
                     place of θ*.
                     Note: This adapts Liu et al.'s imitation teacher to the
                     Bernoulli / Beta-Binomial setting.  Liu et al.'s original
                     teacher imitates the student's weight vector via stochastic
                     mirror descent on a regression problem; here v tracks scalar
                     outcome probabilities, which is the natural analogue for
                     binary correctness data.

RoundRobinTeacher  — Deterministic baseline: cycles through all (agent, region)
                     pairs in fixed row-major order.

RandomTeacher      — Stochastic baseline: selects (agent, region) uniformly.
                     Uses an isolated RNG (no global numpy state pollution).
                     Lower bound on teaching efficiency.

Randomness
----------
All teachers use isolated np.random.Generator instances seeded from the
constructor seed argument.  No teacher calls np.random.seed() or uses the
global numpy random state.  BaseTeacher._pred_rng drives agent predictions;
SurrogateTeacher._select_rng handles tiebreaking; RandomTeacher._select_rng
drives pair selection.  This ensures fully reproducible experiments.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from orchestration_framework import Agent, Task, CapabilityEstimator


# ---------------------------------------------------------------------------
# Teaching result record
# ---------------------------------------------------------------------------

@dataclass
class TeachingStep:
    """Records one evaluation step chosen by the teacher."""
    t: int
    agent_idx: int
    region: int
    is_correct: bool
    estimated_caps: np.ndarray
    true_caps: np.ndarray
    mse: float


# ---------------------------------------------------------------------------
# Task pool
# ---------------------------------------------------------------------------

class TaskPool:
    """Pool of tasks grouped by region."""

    def __init__(self, M: int, tasks_per_region: int = 200, seed: int = 0):
        self.M = M
        self.rng = np.random.default_rng(seed)
        self._pools: Dict[int, List[Task]] = {m: [] for m in range(M)}
        for m in range(M):
            for _ in range(tasks_per_region):
                x = self.rng.standard_normal(10)
                y = int(self.rng.integers(0, 2))
                self._pools[m].append(Task(x=x, y=y, region=m))
        self._indices = {m: 0 for m in range(M)}

    def get_task(self, region: int) -> Task:
        idx = self._indices[region]
        task = self._pools[region][idx % len(self._pools[region])]
        self._indices[region] += 1
        return Task(x=task.x.copy(), y=task.y, region=task.region)


# ---------------------------------------------------------------------------
# Beta-Binomial helpers: EMSR and posterior variance
# ---------------------------------------------------------------------------

def _expected_mse_reduction(alpha1: float, alpha0: float,
                             n_cor: int, n_inc: int,
                             theta_star: float) -> float:
    """
    Expected reduction in squared error for slot (k,m) from one observation.

    E[Δ SE_{k,m}] = (θ* − θ̂_current)² − E[(θ* − θ̂_new)² | θ*]

    where θ̂ is the Beta-Binomial posterior mean and the expectation is over
    the Bernoulli(θ*) outcome.  Closed-form derivation:

        θ̂_current = (α₁ + n_cor) / (α₀ + α₁ + n_total)
        new_denom  = α₀ + α₁ + n_total + 1
        θ̂_correct  = (α₁ + n_cor + 1) / new_denom
        θ̂_incorrect= (α₁ + n_cor)     / new_denom

        E[Δ SE] = (θ* − θ̂)²
                  − [θ* (θ* − θ̂_correct)²  + (1−θ*)(θ* − θ̂_incorrect)²]
    """
    n = n_cor + n_inc
    s = alpha0 + alpha1 + n
    theta_hat = (alpha1 + n_cor) / s  # current posterior mean

    new_s = s + 1
    mean_if_correct   = (alpha1 + n_cor + 1) / new_s
    mean_if_incorrect = (alpha1 + n_cor)     / new_s

    current_se = (theta_star - theta_hat) ** 2

    se_if_correct   = (theta_star - mean_if_correct) ** 2
    se_if_incorrect = (theta_star - mean_if_incorrect) ** 2
    expected_new_se = theta_star * se_if_correct + (1 - theta_star) * se_if_incorrect

    return current_se - expected_new_se


def _get_alpha(ce) -> tuple:
    """Return (alpha1, alpha0) from any estimator, defaulting to 1.0/1.0."""
    a1 = getattr(ce, "alpha1", 1.0)
    a0 = getattr(ce, "alpha0", 1.0)
    return float(a1), float(a0)


def _posterior_variance(alpha1: float, alpha0: float,
                        n_cor: int, n_inc: int) -> float:
    """
    Variance of the Beta(α₁ + n_cor, α₀ + n_inc) posterior.

        Var[θ] = a₁ · a₀ / (s² · (s + 1))

    where a₁ = α₁ + n_cor, a₀ = α₀ + n_inc, s = a₁ + a₀.
    """
    a1 = alpha1 + n_cor
    a0 = alpha0 + n_inc
    s  = a1 + a0
    return (a1 * a0) / (s * s * (s + 1))


# ---------------------------------------------------------------------------
# Base teacher
# ---------------------------------------------------------------------------

class BaseTeacher:
    """Abstract base class for machine teachers."""

    def __init__(
        self,
        agents: List[Agent],
        M: int,
        task_pool: TaskPool,
        alpha0: float = 1.0,
        alpha1: float = 1.0,
        prior=None,
        seed: int = 42,
    ):
        self.agents = agents
        self.K = len(agents)
        self.M = M
        self.task_pool = task_pool
        self.capability_estimator = CapabilityEstimator(
            self.K, self.M, alpha0, alpha1, prior=prior
        )
        # Isolated RNG for agent predictions — reproducible, no global state.
        self._pred_rng = np.random.default_rng(seed + 7919)
        self.true_caps = np.array(
            [[agents[k].get_capability(m) for m in range(M)] for k in range(self.K)]
        )
        self.history: List[TeachingStep] = []

    def select_pair(self, t: int) -> Tuple[int, int]:
        raise NotImplementedError

    def step(self, t: int) -> TeachingStep:
        agent_idx, region = self.select_pair(t)
        task = self.task_pool.get_task(region)
        prediction = self.agents[agent_idx].predict(task, rng=self._pred_rng)
        is_correct = prediction == task.y
        self.capability_estimator.update(agent_idx, region, is_correct)
        estimated = self.capability_estimator.get_all_posterior_means()
        mse = float(np.mean((estimated - self.true_caps) ** 2))
        record = TeachingStep(
            t=t, agent_idx=agent_idx, region=region,
            is_correct=is_correct,
            estimated_caps=estimated.copy(),
            true_caps=self.true_caps.copy(),
            mse=mse,
        )
        self.history.append(record)
        return record

    def run(self, budget: int) -> List[TeachingStep]:
        return [self.step(t) for t in range(budget)]

    def current_estimates(self) -> np.ndarray:
        return self.capability_estimator.get_all_posterior_means()

    def get_posterior_variance(self, agent_idx: int, region: int) -> float:
        ce = self.capability_estimator
        # If the estimator exposes its own get_posterior_variance, use it directly.
        if hasattr(ce, "get_posterior_variance") and callable(
            getattr(ce, "get_posterior_variance")
        ):
            return ce.get_posterior_variance(agent_idx, region)
        # Fallback: compute from Beta parameters (original CapabilityEstimator path).
        n_inc = int(ce.counts[agent_idx, region, 0])
        n_cor = int(ce.counts[agent_idx, region, 1])
        if hasattr(ce, "prior") and ce.prior is not None and hasattr(
            ce.prior, "posterior_variance"
        ):
            return ce.prior.posterior_variance(n_cor, n_inc)
        return _posterior_variance(*_get_alpha(ce), n_cor, n_inc)

    def total_variance(self) -> float:
        return sum(
            self.get_posterior_variance(k, m)
            for k in range(self.K)
            for m in range(self.M)
        )

    def get_summary(self) -> Dict:
        mses = [s.mse for s in self.history]
        return {
            "teacher": self.__class__.__name__,
            "budget": len(self.history),
            "final_mse": mses[-1] if mses else None,
            "mse_curve": mses,
            "final_estimates": self.current_estimates(),
            "true_caps": self.true_caps,
        }


# ---------------------------------------------------------------------------
# 1. Omniscient Teacher — Expected MSE Reduction (EMSR)
# ---------------------------------------------------------------------------

class OmniscientTeacher(BaseTeacher):
    """
    Has full knowledge of θ*_{k,m}.

    Selection criterion: Expected MSE Reduction (EMSR).

    For each candidate slot (k, m), compute the exact expected reduction in
    squared estimation error from one additional Bernoulli(θ*_{k,m}) observation,
    in closed form for the Beta-Binomial posterior mean:

        EMSR(k,m) = (θ* − θ̂)²  −  E_y[(θ* − θ̂_new)² | θ*]

    Select the (k, m) with maximum EMSR.  This is the greedy one-step-ahead
    optimal criterion for MSE minimisation.

    Bootstrap: every (k, m) slot is visited once in row-major order before
    EMSR-guided selection begins (total_obs < K*M phase).
    """

    def select_pair(self, t: int) -> Tuple[int, int]:
        # Bootstrap: visit every (k, m) at least once
        total_obs = int(np.sum(self.capability_estimator.counts))
        if total_obs < self.K * self.M:
            slot = total_obs % (self.K * self.M)
            return slot // self.M, slot % self.M

        ce = self.capability_estimator
        best_score = -np.inf
        best_k, best_m = 0, 0

        for k in range(self.K):
            for m in range(self.M):
                n_cor = int(ce.counts[k, m, 1])
                n_inc = int(ce.counts[k, m, 0])
                theta_star = self.true_caps[k, m]
                score = _expected_mse_reduction(
                    _get_alpha(ce)[0], _get_alpha(ce)[1], n_cor, n_inc, theta_star
                )
                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# 2. Surrogate Teacher — Maximum Posterior Variance (uncertainty sampling)
# ---------------------------------------------------------------------------

class SurrogateTeacher(BaseTeacher):
    """
    Does NOT know θ*.

    Selection criterion: maximum posterior variance (A-optimal / uncertainty
    sampling):
        score(k, m) = Var[θ_{k,m} | data]  =  a₁ a₀ / (s² (s+1))

    This selects the slot where our Beta posterior is most diffuse, i.e. where
    we are most uncertain about the agent's capability.

    Note on relationship to Liu et al. (2017):
    Liu et al.'s surrogate teacher replaces the T2 term in the omniscient
    objective with a convexity lower bound that only requires querying the
    learner's function output.  Here, without access to a gradient-based
    learner model, we adapt the surrogate concept to Bayesian experimental
    design: we select the slot that maximises posterior uncertainty, which is
    the natural analogue when θ* is unknown.

    Note on tiebreaking:
    Under a uniform Beta(1,1) prior all K×M slots start with identical
    variance.  A small random perturbation is added to break ties and avoid
    deterministically always starting at (k=0, m=0).
    """

    def __init__(self, *args, seed: int = 42, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        # Dedicated RNG for tiebreaking — isolated from prediction RNG.
        self._select_rng = np.random.default_rng(seed + 1337)

    def select_pair(self, t: int) -> Tuple[int, int]:
        ce = self.capability_estimator
        best_score = -np.inf
        best_k, best_m = 0, 0

        for k in range(self.K):
            for m in range(self.M):
                n_cor = int(ce.counts[k, m, 1])
                n_inc = int(ce.counts[k, m, 0])
                # Add a tiny random tiebreak so equal-variance slots are not
                # always resolved to (k=0, m=0).
                score = (self.get_posterior_variance(k, m)
                         + self._select_rng.random() * 1e-9)
                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# 3. Imitation Teacher — Robbins-Monro approximation of θ*
# ---------------------------------------------------------------------------

class ImitationTeacher(BaseTeacher):
    """
    Bridges omniscient and surrogate teachers.

    Maintains a Robbins-Monro running estimate v_{k,m} of θ*_{k,m} by
    averaging observed binary outcomes with a per-slot decaying learning rate:

        v_{k,m}^{t+1} = v_{k,m}^t + η_t · (outcome − v_{k,m}^t)

    where η_t = c / (c + n_{k,m}) and outcome ∈ {0, 1}.

    Selection uses v in place of θ* in the EMSR formula:
        score(k,m) = EMSR(v_{k,m})  (same closed form as OmniscientTeacher)

    Relationship to Liu et al. (2017):
    Liu et al.'s imitation teacher learns to imitate the student's weight
    vector w^t (a continuous regression parameter) via stochastic mirror
    descent, so that it can compute T2 = ⟨w^t − w*, ∂ℓ/∂w⟩ without direct
    access to w^t.  In our Bernoulli / Beta-Binomial setting the analogous
    "parameter" is the scalar θ*_{k,m} for each slot.  The Robbins-Monro
    update is the natural adaptation: it converges to θ* by the strong law of
    large numbers and provides the same conceptual role as the mirror-descent
    imitation in the original paper.

    Parameters
    ----------
    eta_v : float
        Constant c in the decay schedule η = c / (c + n).  Default 5.0.
    """

    def __init__(self, *args, eta_v: float = 5.0, seed: int = 42, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.eta_base = eta_v
        prior_mean = (
            self.capability_estimator.alpha1
            / (self.capability_estimator.alpha0 + self.capability_estimator.alpha1)
        )
        # Initialise v at the prior mean for each slot.
        self.v = np.full((self.K, self.M), prior_mean)
        self._slot_obs_count = np.zeros((self.K, self.M))

    def step(self, t: int) -> TeachingStep:
        agent_idx, region = self.select_pair(t)
        task = self.task_pool.get_task(region)
        prediction = self.agents[agent_idx].predict(task, rng=self._pred_rng)
        is_correct = prediction == task.y

        self.capability_estimator.update(agent_idx, region, is_correct)

        # Robbins-Monro update of v_{k,m} (after observation, before next select).
        self._slot_obs_count[agent_idx, region] += 1
        n_slot = self._slot_obs_count[agent_idx, region]
        eta_t = self.eta_base / (self.eta_base + n_slot)
        outcome = 1.0 if is_correct else 0.0
        self.v[agent_idx, region] += eta_t * (outcome - self.v[agent_idx, region])

        estimated = self.capability_estimator.get_all_posterior_means()
        mse = float(np.mean((estimated - self.true_caps) ** 2))

        record = TeachingStep(
            t=t, agent_idx=agent_idx, region=region,
            is_correct=is_correct,
            estimated_caps=estimated.copy(),
            true_caps=self.true_caps.copy(),
            mse=mse,
        )
        self.history.append(record)
        return record

    def select_pair(self, t: int) -> Tuple[int, int]:
        # Bootstrap: visit every (k, m) at least once
        total_obs = int(np.sum(self.capability_estimator.counts))
        if total_obs < self.K * self.M:
            slot = total_obs % (self.K * self.M)
            return slot // self.M, slot % self.M

        ce = self.capability_estimator
        best_score = -np.inf
        best_k, best_m = 0, 0

        for k in range(self.K):
            for m in range(self.M):
                n_cor = int(ce.counts[k, m, 1])
                n_inc = int(ce.counts[k, m, 0])
                # Use v (Robbins-Monro estimate of θ*) in the EMSR formula.
                score = _expected_mse_reduction(
                    _get_alpha(ce)[0], _get_alpha(ce)[1], n_cor, n_inc, self.v[k, m]
                )
                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class RandomTeacher(BaseTeacher):
    """
    Selects (agent, region) uniformly at random.

    Uses an isolated np.random.Generator seeded from the constructor — no
    global numpy state is modified, ensuring reproducibility.
    """

    def __init__(self, *args, seed: int = 42, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._select_rng = np.random.default_rng(seed + 2718)

    def select_pair(self, t: int) -> Tuple[int, int]:
        k = int(self._select_rng.integers(0, self.K))
        m = int(self._select_rng.integers(0, self.M))
        return k, m


class RoundRobinTeacher(BaseTeacher):
    """
    Cycles deterministically through all (agent, region) pairs in row-major
    order: (0,0), (0,1), ..., (0,M-1), (1,0), ..., (K-1,M-1), then repeats.
    """

    def select_pair(self, t: int) -> Tuple[int, int]:
        slot = t % (self.K * self.M)
        return slot // self.M, slot % self.M


# ---------------------------------------------------------------------------
# Teaching experiment runner
# ---------------------------------------------------------------------------

def run_teaching_experiment(
    agents: List[Agent],
    M: int,
    budget: int,
    teacher_classes,
    seed: int = 42,
    tasks_per_region: int = 500,
    prior=None,
) -> Dict[str, Dict]:
    """
    Run all teacher classes on the same agent set with isolated RNGs.

    Each teacher receives its own TaskPool (same seed → same pool) and its
    own prediction RNG, so results are reproducible and independent.
    """
    results = {}
    for teacher_cls, kwargs in teacher_classes:
        pool = TaskPool(M=M, tasks_per_region=tasks_per_region, seed=seed)
        kw = dict(kwargs)
        if prior is not None:
            kw["prior"] = prior
        kw.setdefault("seed", seed)
        teacher = teacher_cls(agents=agents, M=M, task_pool=pool, **kw)
        teacher.run(budget)
        results[teacher_cls.__name__] = teacher.get_summary()
    return results


def compute_teaching_efficiency(
    results: Dict[str, Dict], target_mse: float
) -> Dict[str, Optional[int]]:
    efficiency = {}
    for name, summary in results.items():
        curve = summary["mse_curve"]
        reached = next((t for t, mse in enumerate(curve) if mse <= target_mse), None)
        efficiency[name] = reached
    return efficiency


def print_teaching_results(results: Dict[str, Dict], target_mse: float = 0.01):
    print("\n" + "=" * 70)
    print("MACHINE TEACHING RESULTS")
    print("=" * 70)
    efficiency = compute_teaching_efficiency(results, target_mse)
    header = f"{'Teacher':<25} {'Final MSE':>12} {'Steps to MSE<'+str(target_mse):>22}"
    print(header)
    print("-" * 65)
    for name, summary in results.items():
        mse = summary["final_mse"]
        steps = efficiency[name]
        steps_str = str(steps) if steps is not None else "not reached"
        print(f"{name:<25} {mse:>12.6f} {steps_str:>22}")
    print()

    ref_name = next(
        (n for n in ["OmniscientTeacher", "ImitationTeacher", "SurrogateTeacher"]
         if n in results),
        next(iter(results)),
    )
    ref = results[ref_name]
    K, M = ref["true_caps"].shape
    print(f"Capability estimates (after budget) — {ref_name}")
    print(f"  {'':12} " + "  ".join(f"Region {m}" for m in range(M)))
    for k in range(K):
        est_row  = "  ".join(f"{ref['final_estimates'][k,m]:.3f}" for m in range(M))
        true_row = "  ".join(f"{ref['true_caps'][k,m]:.3f}" for m in range(M))
        print(f"  Agent_{k+1} est  {est_row}")
        print(f"  Agent_{k+1} true {true_row}")
    print()
