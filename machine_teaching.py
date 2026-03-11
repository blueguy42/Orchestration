"""
Machine Teaching for Efficient Capability Identification in Multi-Agent Orchestration
======================================================================================
Based on:
  - Liu et al. (2017) "Iterative Machine Teaching" (ICML)
  - Bhatt et al. (2025) "When Should We Orchestrate Multiple Agents?"

The teacher strategically selects (agent, region) pairs to evaluate — rather than
feeding training examples to a gradient learner — in order to minimise the number
of evaluations required to estimate each agent's capability across all regions.

Key design decisions
--------------------
* The "student" in IMT terms is the CapabilityEstimator (a Beta-Binomial model).
  Its "parameter" for agent k in region m is the scalar θ_{k,m} ∈ (0,1).
* The "teacher" chooses *which* (agent, region) slot to probe next, i.e. which
  evaluation to run, rather than synthesising gradient-descent examples.
* We adapt the IMT difficulty / usefulness decomposition (Eq. 3 of Liu et al.):
    - Difficulty  T1 ∝ posterior variance of θ_{k,m}   (how uncertain we are)
    - Usefulness  T2 ∝ distance of current estimate from the true value
  The teacher maximises T2 - λ·T1, i.e. it prefers slots that are both
  far from the truth *and* informative to observe.

Three teacher variants are implemented, mirroring Liu et al.'s hierarchy:

1. OmniscientTeacher   — knows the true capabilities; maximises expected
                         reduction in total squared estimation error.
2. SurrogateTeacher    — does not know the true capabilities; uses the
                         current posterior uncertainty (variance) as a proxy
                         for usefulness (pure uncertainty-based selection).
3. ImitationTeacher    — approximates the omniscient teacher by maintaining
                         an internal running estimate of the true capabilities
                         and selecting based on that estimate; bridges the gap
                         between the two above.

All teachers are evaluated against a RandomTeacher baseline and a
RoundRobinTeacher that cycles uniformly through all (agent, region) pairs.
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
    t: int                    # Step index
    agent_idx: int            # Which agent was evaluated
    region: int               # Which region task was drawn from
    is_correct: bool          # Observed outcome
    estimated_caps: np.ndarray  # Full K×M capability matrix after update
    true_caps: np.ndarray       # True K×M capability matrix (for logging)
    mse: float                  # Mean squared estimation error after update


# ---------------------------------------------------------------------------
# Task pool: region-labelled tasks available for the teacher to draw from
# ---------------------------------------------------------------------------

class TaskPool:
    """
    A pool of tasks grouped by region.

    The teacher selects a region; the pool returns a task from that region.
    In a real deployment these would be genuine evaluation tasks; here we
    generate them on-the-fly using the same synthetic model as the
    existing SyntheticDataGenerator.
    """

    def __init__(self, M: int, tasks_per_region: int = 200, seed: int = 0):
        self.M = M
        self.rng = np.random.default_rng(seed)
        self._pools: Dict[int, List[Task]] = {m: [] for m in range(M)}

        for m in range(M):
            for _ in range(tasks_per_region):
                x = self.rng.standard_normal(10)
                y = int(self.rng.integers(0, 2))
                self._pools[m].append(Task(x=x, y=y, region=m))

        # Circular indices so we never exhaust the pool
        self._indices = {m: 0 for m in range(M)}

    def get_task(self, region: int) -> Task:
        """Return the next task from the requested region (cycles)."""
        idx = self._indices[region]
        task = self._pools[region][idx % len(self._pools[region])]
        self._indices[region] += 1
        return Task(x=task.x.copy(), y=task.y, region=task.region)


# ---------------------------------------------------------------------------
# Base teacher
# ---------------------------------------------------------------------------

class BaseTeacher:
    """
    Abstract base class for machine teachers.

    The teacher's job is to decide, at each step t, which (agent, region)
    pair to evaluate.  It then:
      1. Draws a task from that region via the TaskPool.
      2. Asks the chosen agent to answer the task.
      3. Observes correctness and updates the shared CapabilityEstimator.
    """

    def __init__(
        self,
        agents: List[Agent],
        M: int,
        task_pool: TaskPool,
        alpha0: float = 1.0,
        alpha1: float = 1.0,
    ):
        self.agents = agents
        self.K = len(agents)
        self.M = M
        self.task_pool = task_pool
        self.capability_estimator = CapabilityEstimator(self.K, self.M, alpha0, alpha1)

        # Ground-truth capability matrix (K × M) for computing MSE
        self.true_caps = np.array(
            [[agents[k].get_capability(m) for m in range(M)] for k in range(self.K)]
        )

        self.history: List[TeachingStep] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_pair(self, t: int) -> Tuple[int, int]:
        """
        Choose the next (agent_idx, region) pair to evaluate.

        Returns
        -------
        agent_idx : int
        region    : int
        """
        raise NotImplementedError

    def step(self, t: int) -> TeachingStep:
        """Execute one teaching step: select → observe → update."""
        agent_idx, region = self.select_pair(t)

        # Draw a task from the chosen region
        task = self.task_pool.get_task(region)

        # Ask the agent — outcome is stochastic, governed by true capability
        prediction = self.agents[agent_idx].predict(task)
        is_correct = prediction == task.y

        # Update the shared capability estimator
        self.capability_estimator.update(agent_idx, region, is_correct)

        # Snapshot current estimates and compute MSE
        estimated = self.capability_estimator.get_all_capabilities()
        mse = float(np.mean((estimated - self.true_caps) ** 2))

        record = TeachingStep(
            t=t,
            agent_idx=agent_idx,
            region=region,
            is_correct=is_correct,
            estimated_caps=estimated.copy(),
            true_caps=self.true_caps.copy(),
            mse=mse,
        )
        self.history.append(record)
        return record

    def run(self, budget: int) -> List[TeachingStep]:
        """Run the teacher for *budget* evaluation steps."""
        return [self.step(t) for t in range(budget)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def current_estimates(self) -> np.ndarray:
        """Return the current K×M capability estimate matrix."""
        return self.capability_estimator.get_all_capabilities()

    def posterior_variance(self, agent_idx: int, region: int) -> float:
        """
        Posterior variance of θ_{k,m} under Beta(α₁, α₀) updated by observations.

        Beta variance: (α₁ · α₀) / ((α₁+α₀)² · (α₁+α₀+1))
        where α₁ = prior_α₁ + n_correct, α₀ = prior_α₀ + n_incorrect.
        """
        n_inc = self.capability_estimator.counts[agent_idx, region, 0]
        n_cor = self.capability_estimator.counts[agent_idx, region, 1]
        a1 = n_cor + self.capability_estimator.alpha1
        a0 = n_inc + self.capability_estimator.alpha0
        s = a1 + a0
        return (a1 * a0) / (s * s * (s + 1))

    def total_variance(self) -> float:
        """Sum of posterior variances over all (agent, region) pairs."""
        return sum(
            self.posterior_variance(k, m)
            for k in range(self.K)
            for m in range(self.M)
        )

    def get_summary(self) -> Dict:
        """Summary statistics over the teaching run."""
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
# 1. Omniscient Teacher (Liu et al. §4)
# ---------------------------------------------------------------------------

class OmniscientTeacher(BaseTeacher):
    """
    Has full knowledge of the true capabilities θ*_{k,m}.

    Selection criterion (adapted from IMT Eq. 3–4):
    For each (k, m) pair compute the expected reduction in squared error
    after one additional observation:

        Δ(k, m) = (θ*_{k,m} - θ̂_{k,m})²  ·  var(θ̂_{k,m})

    The first factor is the *usefulness* T2: how far off our current
    estimate is (larger gap → more useful to probe).
    The second factor is *difficulty* T1: current posterior uncertainty
    (less certain → more to gain).  The product encodes the IMT trade-off
    between choosing informative and useful examples.

    We maximise this score, i.e. prefer the slot where:
      - we are currently farthest from the truth, AND
      - we are still uncertain (haven't already over-sampled it).
    """

    def select_pair(self, t: int) -> Tuple[int, int]:
        # Bootstrap phase: visit every (k, m) at least once before scoring,
        # so posterior estimates are meaningful before the oracle can guide.
        total_obs = np.sum(self.capability_estimator.counts)
        if total_obs < self.K * self.M:
            slot = int(total_obs) % (self.K * self.M)
            return slot // self.M, slot % self.M

        best_score = -np.inf
        best_k, best_m = 0, 0

        for k in range(self.K):
            for m in range(self.M):
                theta_hat = self.capability_estimator.get_capability(k, m)
                theta_star = self.true_caps[k, m]

                # T2 — usefulness: how far off is our current estimate?
                usefulness = (theta_star - theta_hat) ** 2

                # T1 — difficulty / variance: are we still uncertain here?
                difficulty = self.posterior_variance(k, m)

                # IMT-style score (Eq. 3 discretised):
                # minimise  η²·T1 − 2η·T2  ↔  maximise  T2 − 0.5·η·T1
                # We use η=1 for simplicity; the relative weight controls
                # how aggressively we explore vs exploit.
                score = usefulness - 0.5 * difficulty

                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# 2. Surrogate Teacher (Liu et al. §5.1)
# ---------------------------------------------------------------------------

class SurrogateTeacher(BaseTeacher):
    """
    Does NOT know the true capabilities.

    Replaces the usefulness term T2 with its tractable surrogate: the
    posterior variance (our uncertainty about θ̂_{k,m}).  This is equivalent
    to maximum uncertainty / active-learning-style selection.

        score(k, m) = posterior_variance(k, m)

    Intuition: when we are most uncertain about an (agent, region) pair, an
    additional observation reduces our posterior uncertainty the most (on
    average), mirroring the surrogate teacher's lower-bound substitution.
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
# 3. Imitation Teacher (Liu et al. §5.2)
# ---------------------------------------------------------------------------

class ImitationTeacher(BaseTeacher):
    """
    Bridges the omniscient and surrogate teachers.

    Maintains a running *internal estimate* v_{k,m} of θ*_{k,m} by
    imitating the stochastic mirror-descent update of Liu et al. Alg. 2:

        v_{k,m}^{t+1} = v_{k,m}^t − η_v · (v_{k,m}^t − θ̂_{k,m}^t)

    This lets v track the true capability even without direct access.
    Selection then mirrors the omniscient teacher but uses v instead of θ*:

        score(k, m) = (v_{k,m} − θ̂_{k,m})² − 0.5 · var(θ̂_{k,m})

    Parameters
    ----------
    eta_v : float
        Learning rate for the internal imitation update (default 0.3).
    """

    def __init__(self, *args, eta_v: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_v = eta_v
        # Initialise internal estimate at the prior mean (0.5 for uniform Beta)
        prior_mean = (
            self.capability_estimator.alpha1
            / (self.capability_estimator.alpha0 + self.capability_estimator.alpha1)
        )
        self.v = np.full((self.K, self.M), prior_mean)

    # Override step() to also update the internal imitation model
    def step(self, t: int) -> TeachingStep:
        record = super().step(t)
        # After the observation, update v toward the current posterior mean
        # (Alg. 2 line 4 adapted: v ← v − η_v · (v − θ̂))
        theta_hat = self.capability_estimator.get_all_capabilities()
        self.v -= self.eta_v * (self.v - theta_hat)
        return record

    def select_pair(self, t: int) -> Tuple[int, int]:
        # Bootstrap: ensure every slot is visited once before scoring
        total_obs = np.sum(self.capability_estimator.counts)
        if total_obs < self.K * self.M:
            slot = int(total_obs) % (self.K * self.M)
            return slot // self.M, slot % self.M

        best_score = -np.inf
        best_k, best_m = 0, 0

        for k in range(self.K):
            for m in range(self.M):
                theta_hat = self.capability_estimator.get_capability(k, m)

                # Usefulness proxy: squared distance from internal estimate v
                usefulness = (self.v[k, m] - theta_hat) ** 2

                # Difficulty proxy: posterior variance
                difficulty = self.posterior_variance(k, m)

                score = usefulness - 0.5 * difficulty

                if score > best_score:
                    best_score = score
                    best_k, best_m = k, m

        return best_k, best_m


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class RandomTeacher(BaseTeacher):
    """Selects (agent, region) uniformly at random — lower bound baseline."""

    def select_pair(self, t: int) -> Tuple[int, int]:
        k = np.random.randint(0, self.K)
        m = np.random.randint(0, self.M)
        return k, m


class RoundRobinTeacher(BaseTeacher):
    """
    Cycles deterministically through all (agent, region) pairs.

    Ensures every slot is covered uniformly — a strong baseline when
    the capability landscape is unknown.
    """

    def select_pair(self, t: int) -> Tuple[int, int]:
        slot = t % (self.K * self.M)
        k = slot // self.M
        m = slot % self.M
        return k, m


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
) -> Dict[str, Dict]:
    """
    Run all provided teacher classes on the same agent set and compare.

    Parameters
    ----------
    agents         : List of agents with known capabilities
    M              : Number of regions
    budget         : Total number of evaluation steps (budget)
    teacher_classes: List of (class, kwargs) pairs, e.g.
                     [(OmniscientTeacher, {}), (SurrogateTeacher, {})]
    seed           : Random seed
    tasks_per_region: Pool size per region

    Returns
    -------
    dict mapping teacher name → summary dict (with mse_curve, final_mse, …)
    """
    results = {}

    for teacher_cls, kwargs in teacher_classes:
        np.random.seed(seed)
        pool = TaskPool(M=M, tasks_per_region=tasks_per_region, seed=seed)
        teacher = teacher_cls(agents=agents, M=M, task_pool=pool, **kwargs)
        teacher.run(budget)
        results[teacher_cls.__name__] = teacher.get_summary()

    return results


def compute_teaching_efficiency(results: Dict[str, Dict], target_mse: float) -> Dict[str, Optional[int]]:
    """
    Compute teaching efficiency: steps to reach *target_mse*.

    Returns
    -------
    dict mapping teacher name → number of steps needed (None if not reached)
    """
    efficiency = {}
    for name, summary in results.items():
        curve = summary["mse_curve"]
        reached = next((t for t, mse in enumerate(curve) if mse <= target_mse), None)
        efficiency[name] = reached
    return efficiency


def print_teaching_results(results: Dict[str, Dict], target_mse: float = 0.01):
    """Pretty-print a teaching experiment summary."""
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

    # Per-pair capability estimates vs truth for one teacher (omniscient)
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
