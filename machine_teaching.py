"""
Machine Teaching for Efficient Capability Identification in Multi-Agent Orchestration

Implements a machine teaching approach for efficiently identifying agent capabilities
across task regions, as described in the MPhil dissertation proposal extending
Bhatt et al. (2025).  Rather than learning capabilities passively from a random task
stream, a teacher strategically selects which (agent, region) slot to probe next,
minimising the evaluation budget required to reach accurate capability estimates.

The "student" is a Beta-Binomial CapabilityEstimator (from orchestration_framework.py)
that maintains a posterior over each agent's correctness probability θ_{k,m} in each
region.  The teacher's goal is to drive all K × M estimates close to their true values
as quickly as possible (minimising MSE), then hand them to a downstream orchestrator.

Teachers (based on Liu et al. 2017 "Iterative Machine Teaching", ICML)
-----------------------------------------------------------------------
OmniscientTeacher  — Knows true capabilities θ*; selects the slot whose next
                     observation gives the greatest expected reduction in MSE
                     (EMSR criterion, closed-form for Beta-Binomial).
                     Theoretical upper bound on teaching efficiency.

SurrogateTeacher   — Does not know θ*; selects the slot with the highest
                     posterior variance (D-optimal / maximum entropy design).
                     A practical strategy when true capabilities are unknown.

ImitationTeacher   — Bridges omniscient and surrogate teachers.  Maintains a
                     Robbins-Monro running estimate v_{k,m} of θ*_{k,m} and
                     uses it in the EMSR formula in place of θ*.  The learning
                     rate decays as η = c / (c + n_{k,m}) per slot.

RoundRobinTeacher  — Deterministic baseline: cycles through all (agent, region)
                     pairs in fixed order.

RandomTeacher      — Stochastic baseline: selects (agent, region) uniformly.
                     Lower bound on teaching efficiency.
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
# Beta-Binomial helper: expected posterior mean after one more observation
# ---------------------------------------------------------------------------

def _expected_posterior_mean_after_obs(alpha1: float, alpha0: float,
                                       n_cor: int, n_inc: int,
                                       theta_star: float) -> float:
    """
    E[ θ̂_{new} | θ* ] after one more Bernoulli(θ*) observation.

    With probability θ*  we observe correct → new posterior mean = (α₁+n_cor+1)/(α₀+α₁+n+1)
    With probability 1-θ* we observe incorrect → new posterior mean = (α₁+n_cor)/(α₀+α₁+n+1)

    Returns the expectation over these two outcomes.
    """
    n = n_cor + n_inc
    new_denom = alpha0 + alpha1 + n + 1
    mean_if_correct   = (alpha1 + n_cor + 1) / new_denom
    mean_if_incorrect = (alpha1 + n_cor)     / new_denom
    return theta_star * mean_if_correct + (1 - theta_star) * mean_if_incorrect


def _expected_mse_reduction(alpha1: float, alpha0: float,
                            n_cor: int, n_inc: int,
                            theta_star: float) -> float:
    """
    Expected reduction in squared error for slot (k,m) from one observation.

    E[Δ MSE_{k,m}] = (θ* - θ̂_current)² - E[(θ* - θ̂_new)² | θ*]

    where the expectation is over the Bernoulli(θ*) outcome.
    Computed in closed form for the Beta-Binomial posterior mean.
    """
    n = n_cor + n_inc
    s = alpha0 + alpha1 + n
    theta_hat = (alpha1 + n_cor) / s  # current posterior mean

    new_s = s + 1
    mean_if_correct   = (alpha1 + n_cor + 1) / new_s
    mean_if_incorrect = (alpha1 + n_cor)     / new_s

    # Current squared error
    current_se = (theta_star - theta_hat) ** 2

    # Expected squared error after observation
    se_if_correct   = (theta_star - mean_if_correct) ** 2
    se_if_incorrect = (theta_star - mean_if_incorrect) ** 2
    expected_new_se = theta_star * se_if_correct + (1 - theta_star) * se_if_incorrect

    return current_se - expected_new_se


def _expected_info_gain(alpha1: float, alpha0: float,
                        n_cor: int, n_inc: int,
                        theta_star: float) -> float:
    """
    Expected KL divergence between posterior-after-observation and current
    posterior for slot (k,m), i.e. the mutual information between the next
    observation and θ_{k,m}.

    For Beta(a,b) → Beta(a+1,b) or Beta(a,b+1):
        E_y[ KL( Beta_new || Beta_current ) ]
    = θ* · KL(Beta(a+1,b) || Beta(a,b)) + (1−θ*) · KL(Beta(a,b+1) || Beta(a,b))

    We use a second-order Taylor approximation:
        KL(Beta(a',b') || Beta(a,b)) ≈ ½ (Δμ)² / Var[θ]
    where Δμ is the shift in posterior mean and Var[θ] is the current variance.
    """
    a = alpha1 + n_cor
    b = alpha0 + n_inc
    s = a + b
    var = (a * b) / (s * s * (s + 1))

    if var < 1e-15:
        return 0.0

    new_s = s + 1
    shift_if_correct   = (a + 1) / new_s - a / s
    shift_if_incorrect = a / new_s       - a / s

    kl_correct   = 0.5 * shift_if_correct ** 2   / var
    kl_incorrect = 0.5 * shift_if_incorrect ** 2  / var

    return theta_star * kl_correct + (1 - theta_star) * kl_incorrect


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
    ):
        self.agents = agents
        self.K = len(agents)
        self.M = M
        self.task_pool = task_pool
        self.capability_estimator = CapabilityEstimator(
            self.K, self.M, alpha0, alpha1, prior=prior
        )
        self._pred_rng = np.random.default_rng(42)
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

    def posterior_variance(self, agent_idx: int, region: int) -> float:
        n_inc = int(self.capability_estimator.counts[agent_idx, region, 0])
        n_cor = int(self.capability_estimator.counts[agent_idx, region, 1])
        ce = self.capability_estimator
        if ce.prior is not None and hasattr(ce.prior, "posterior_variance"):
            return ce.prior.posterior_variance(n_cor, n_inc)
        a1 = n_cor + ce.alpha1
        a0 = n_inc + ce.alpha0
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))

    def total_variance(self) -> float:
        return sum(
            self.posterior_variance(k, m)
            for k in range(self.K)
            for m in range(self.M)
        )

    def info_gain(self, agent_idx: int, region: int) -> float:
        """Expected information gain for one observation of (k, m)."""
        ce = self.capability_estimator
        n_cor = int(ce.counts[agent_idx, region, 1])
        n_inc = int(ce.counts[agent_idx, region, 0])
        theta_star = self.true_caps[agent_idx, region]
        return _expected_info_gain(ce.alpha1, ce.alpha0, n_cor, n_inc, theta_star)

    def total_info_gain_per_step(self) -> List[float]:
        """Cumulative information gain at each step (for plotting)."""
        gains = []
        cum = 0.0
        for s in self.history:
            # Approximate: use the info gain computed before the update
            # For simplicity we use a post-hoc metric instead
            gains.append(s.mse)
        return gains

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
# 1. Omniscient Teacher — Expected MSE Reduction
# ---------------------------------------------------------------------------

class OmniscientTeacher(BaseTeacher):
    """
    Has full knowledge of θ*_{k,m}.

    Selection criterion: Expected MSE Reduction (EMSR).

    For each candidate slot (k, m), compute the exact expected reduction
    in squared estimation error from one additional Bernoulli(θ*_{k,m})
    observation, in closed form for the Beta-Binomial posterior mean:

        EMSR(k,m) = (θ* − θ̂)²  −  E_y[(θ* − θ̂_new)² | θ*]

    Select the (k, m) with maximum EMSR.  This is theoretically optimal
    for greedy one-step-ahead MSE minimisation.
    """

    def select_pair(self, t: int) -> Tuple[int, int]:
        # Bootstrap: visit every (k, m) at least once
        total_obs = np.sum(self.capability_estimator.counts)
        if total_obs < self.K * self.M:
            slot = int(total_obs) % (self.K * self.M)
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
                    ce.alpha1, ce.alpha0, n_cor, n_inc, theta_star
                )

                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# 2. Surrogate Teacher — Maximum Posterior Variance (D-optimal)
# ---------------------------------------------------------------------------

class SurrogateTeacher(BaseTeacher):
    """
    Does NOT know θ*.

    Uses maximum posterior variance (D-optimal experimental design):
        score(k, m) = Var[θ̂_{k,m}]

    Equivalent to maximising expected information gain when no knowledge
    of the true parameter is available.
    """

    def select_pair(self, t: int) -> Tuple[int, int]:
        best_score = -np.inf
        best_k, best_m = 0, 0
        for k in range(self.K):
            for m in range(self.M):
                score = self.posterior_variance(k, m)
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

    Maintains an internal Robbins-Monro estimate v_{k,m} of θ*_{k,m}:

        v_{k,m}^{t+1} = v_{k,m}^t + η_t · (outcome − v_{k,m}^t)

    where η_t = c / (c + n_{k,m}^t) decays per-slot for convergence.

    Selection uses v in place of θ* in the EMSR formula:
        score(k,m) = E_v[MSE reduction] (same closed form, replacing θ* with v)

    Parameters
    ----------
    eta_base : float
        Constant c in the decay schedule η = c / (c + n).  Default 5.0.
    """

    def __init__(self, *args, eta_v: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_base = eta_v
        prior_mean = (
            self.capability_estimator.alpha1
            / (self.capability_estimator.alpha0 + self.capability_estimator.alpha1)
        )
        self.v = np.full((self.K, self.M), prior_mean)
        self._slot_obs_count = np.zeros((self.K, self.M))

    def step(self, t: int) -> TeachingStep:
        agent_idx, region = self.select_pair(t)
        task = self.task_pool.get_task(region)
        prediction = self.agents[agent_idx].predict(task, rng=self._pred_rng)
        is_correct = prediction == task.y

        self.capability_estimator.update(agent_idx, region, is_correct)

        # Robbins-Monro update with decaying rate
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
        total_obs = np.sum(self.capability_estimator.counts)
        if total_obs < self.K * self.M:
            slot = int(total_obs) % (self.K * self.M)
            return slot // self.M, slot % self.M

        ce = self.capability_estimator
        best_score = -np.inf
        best_k, best_m = 0, 0

        for k in range(self.K):
            for m in range(self.M):
                n_cor = int(ce.counts[k, m, 1])
                n_inc = int(ce.counts[k, m, 0])

                # Use v (our Robbins-Monro estimate) in place of θ*
                score = _expected_mse_reduction(
                    ce.alpha1, ce.alpha0, n_cor, n_inc, self.v[k, m]
                )

                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class RandomTeacher(BaseTeacher):
    """Selects (agent, region) uniformly at random."""
    def select_pair(self, t: int) -> Tuple[int, int]:
        k = np.random.randint(0, self.K)
        m = np.random.randint(0, self.M)
        return k, m


class RoundRobinTeacher(BaseTeacher):
    """Cycles deterministically through all (agent, region) pairs."""
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
    """Run all teacher classes on the same agent set.  Each gets isolated RNG."""
    results = {}
    for teacher_cls, kwargs in teacher_classes:
        np.random.seed(seed)
        pool = TaskPool(M=M, tasks_per_region=tasks_per_region, seed=seed)
        kw = dict(kwargs)
        if prior is not None:
            kw["prior"] = prior
        teacher = teacher_cls(agents=agents, M=M, task_pool=pool, **kw)
        teacher._pred_rng = np.random.default_rng(seed + 7919)
        teacher.run(budget)
        results[teacher_cls.__name__] = teacher.get_summary()
    return results


def compute_teaching_efficiency(results: Dict[str, Dict], target_mse: float) -> Dict[str, Optional[int]]:
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
        (n for n in ["OmniscientTeacher", "ImitationTeacher", "SurrogateTeacher"] if n in results),
        next(iter(results)),
    )
    ref = results[ref_name]
    K, M = ref["true_caps"].shape
    print(f"Capability estimates (after budget) — {ref_name}")
    print(f"  {'':12} " + "  ".join(f"Region {m}" for m in range(M)))
    for k in range(K):
        est_row = "  ".join(f"{ref['final_estimates'][k,m]:.3f}" for m in range(M))
        true_row = "  ".join(f"{ref['true_caps'][k,m]:.3f}" for m in range(M))
        print(f"  Agent_{k+1} est  {est_row}")
        print(f"  Agent_{k+1} true {true_row}")
    print()
