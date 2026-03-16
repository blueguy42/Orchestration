"""
Multi-Agent Orchestration Framework

Implementation of the online Bayesian orchestration framework from
Bhatt et al. (2025) "When Should We Orchestrate Multiple Agents?"

The framework models a sequential task stream where an orchestrator selects
one agent per task to maximise expected correctness weighted by cost.  Agent
capability P(A_k | R_m) and region probability P(R_m) are unknown and
estimated online via conjugate Bayesian updates:

  - Region probabilities: Dirichlet-Multinomial MAP (Eq. 2)
  - Agent capabilities:   Beta-Binomial MAP (Eq. 3)
  - Empirical utility:    Ĉ_{≥t}(A_k) / γ_{k,r_t} (Eq. 4)

Orchestrators
-------------
PaperOrchestrator   — maximises empirical utility (Bhatt et al. Eq. 4)
GreedyOrchestrator  — maximises estimated capability in current region only
UCB1Orchestrator    — Upper Confidence Bound; per-region bandit formulation
RandomOrchestrator  — uniform random baseline (lower bound)
OracleOrchestrator  — uses true capabilities; theoretical upper bound

Bayesian Estimators
-------------------
RegionEstimator     — Dirichlet-Multinomial posterior over region weights
CapabilityEstimator — Beta-Binomial posterior per (agent, region) slot;
                      supports custom prior objects from priors.py and
                      exposes both MAP estimates (for orchestration) and
                      posterior means (for teaching modules)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Task:
    """Represents a task with input, label, and region"""
    x: np.ndarray  # Input features
    y: int  # True label
    region: int  # Region label (0 to M-1)
    

class Agent:
    """Represents an agent with capabilities across regions"""
    
    def __init__(self, name: str, capabilities: np.ndarray, costs: np.ndarray):
        self.name = name
        self.capabilities = capabilities  # P(Ak | Rm)
        self.costs = costs  # gamma_km
        self.M = len(capabilities)
        
    def predict(self, task: Task, rng: np.random.Generator = None) -> int:
        """
        Simulate agent prediction based on capability.
        
        Args:
            task: The task to predict
            rng: Optional numpy Generator for reproducible randomness.
                 Falls back to global np.random if None.
        """
        prob_correct = self.capabilities[task.region]
        if rng is not None:
            is_correct = rng.random() < prob_correct
        else:
            is_correct = np.random.random() < prob_correct
        
        if is_correct:
            return task.y
        else:
            return 1 - task.y
    
    def get_capability(self, region: int) -> float:
        return self.capabilities[region]
    
    def get_cost(self, region: int) -> float:
        return self.costs[region]


class RegionEstimator:
    """
    Estimates region probabilities using Dirichlet-Multinomial model.

    Bhatt et al. (2025) Eq. 2 — MAP estimate:
        ŵ_{t,m} = (n_{<t,m} + α_m − 1) / Σ_j (n_{<t,j} + α_j − 1)

    Falls back to posterior mean when α_m < 1 to avoid negative numerators.
    """
    
    def __init__(self, M: int, alpha: Optional[np.ndarray] = None):
        self.M = M
        self.alpha = alpha if alpha is not None else np.ones(M)
        self.counts = np.zeros(M)
        
    def update(self, region: int):
        self.counts[region] += 1
        
    def get_probabilities(self) -> np.ndarray:
        """MAP estimate of region probabilities (Eq. 2)."""
        if np.all(self.alpha >= 1.0):
            numerator = self.counts + self.alpha - 1
            denominator = np.sum(numerator)
            if denominator > 0:
                return numerator / denominator
        # Posterior mean fallback
        numerator = self.counts + self.alpha
        return numerator / np.sum(numerator)


class CapabilityEstimator:
    """
    Estimates agent capabilities per region using Beta-Binomial model.

    Bhatt et al. (2025) Eq. 3 — MAP estimate:
        ĉ_{t,km} = (n_{<t,1} + α₁ − 1) / (n_{<t,0} + n_{<t,1} + α₀ + α₁ − 2)

    Falls back to posterior mean when the MAP denominator ≤ 0.

    Accepts either a prior object (from priors.py) or raw (alpha0, alpha1).
    """

    def __init__(self, K: int, M: int, alpha0: float = 1.0, alpha1: float = 1.0,
                 prior=None):
        self.K = K
        self.M = M

        if prior is not None:
            self.alpha0 = prior.alpha0
            self.alpha1 = prior.alpha1
            self.prior  = prior
        else:
            self.alpha0 = alpha0
            self.alpha1 = alpha1
            self.prior  = None

        self.counts = np.zeros((K, M, 2))  # [agent, region, 0=inc / 1=cor]

    def update(self, agent_idx: int, region: int, is_correct: bool):
        self.counts[agent_idx, region, 1 if is_correct else 0] += 1

    def get_capability(self, agent_idx: int, region: int) -> float:
        """
        MAP estimate of capability (Eq. 3).

        MAP: (α₁ + n_cor − 1) / (α₀ + α₁ + n_total − 2)
        Falls back to posterior mean when MAP denominator ≤ 0.
        """
        n_inc = self.counts[agent_idx, region, 0]
        n_cor = self.counts[agent_idx, region, 1]
        n_tot = n_inc + n_cor

        # If prior provides MAP, use it
        if self.prior is not None and hasattr(self.prior, "map_estimate"):
            return self.prior.map_estimate(int(n_cor), int(n_inc))

        # MAP formula
        map_denom = self.alpha0 + self.alpha1 + n_tot - 2
        if map_denom > 0:
            map_num = self.alpha1 + n_cor - 1
            return float(np.clip(map_num / map_denom, 0.0, 1.0))

        # Posterior mean fallback (always valid)
        return (self.alpha1 + n_cor) / (self.alpha0 + self.alpha1 + n_tot)

    def get_posterior_mean(self, agent_idx: int, region: int) -> float:
        """
        Posterior mean E[θ | data] = (α₁ + n_cor) / (α₀ + α₁ + n_total).
        Always well-defined.  Used by teaching modules.
        """
        n_inc = self.counts[agent_idx, region, 0]
        n_cor = self.counts[agent_idx, region, 1]
        if self.prior is not None and hasattr(self.prior, "posterior_mean"):
            return self.prior.posterior_mean(int(n_cor), int(n_inc))
        return (self.alpha1 + n_cor) / (self.alpha0 + self.alpha1 + n_inc + n_cor)

    def get_all_capabilities(self) -> np.ndarray:
        """Return MAP estimates for all agents and regions (K × M)."""
        caps = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                caps[k, m] = self.get_capability(k, m)
        return caps

    def get_all_posterior_means(self) -> np.ndarray:
        """Return posterior mean estimates for all agents and regions (K × M)."""
        caps = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                caps[k, m] = self.get_posterior_mean(k, m)
        return caps

    def inject_estimates(self, estimates: np.ndarray, n_virtual: int = 50):
        """
        Pre-seed the estimator with virtual observations matching `estimates`.

        For each (k, m), injects n_virtual pseudo-observations split according
        to estimates[k, m], so the posterior mean ≈ estimates[k, m].

        Parameters
        ----------
        estimates : (K, M) array of capability estimates in [0, 1]
        n_virtual : number of virtual observations per slot
        """
        for k in range(self.K):
            for m in range(self.M):
                p = np.clip(estimates[k, m], 0.01, 0.99)
                n_cor = int(round(p * n_virtual))
                n_inc = n_virtual - n_cor
                self.counts[k, m, 1] = float(n_cor)
                self.counts[k, m, 0] = float(n_inc)


class BaseOrchestrator(ABC):
    """Base class for orchestration algorithms"""
    
    def __init__(self, agents: List[Agent], M: int):
        self.agents = agents
        self.K = len(agents)
        self.M = M
        self.region_estimator = RegionEstimator(M)
        self.capability_estimator = CapabilityEstimator(self.K, M)
        self.history = []
        self.selected_agents = []
        self.correctness = []
        
    @abstractmethod
    def select_agent(self, task: Task, t: int) -> int:
        pass
    
    def compute_empirical_utility(self, agent_idx: int, task: Task) -> float:
        """
        Empirical utility (Bhatt et al. Eq. 4):
            Û_{≥t}(A_k) = ĉ_{t,kr_t} · Σ_m ŵ_{t,m} · ĉ_{t,km}  /  γ_{kr_t}
        """
        rt = task.region
        c_t_krt = self.capability_estimator.get_capability(agent_idx, rt)
        w_t = self.region_estimator.get_probabilities()
        c_t_k = np.array([self.capability_estimator.get_capability(agent_idx, m) 
                         for m in range(self.M)])
        
        future_correctness = np.sum(w_t * c_t_k)
        empirical_correctness = c_t_krt * future_correctness
        
        cost = self.agents[agent_idx].get_cost(rt)
        return empirical_correctness / cost if cost > 0 else empirical_correctness
    
    def update(self, agent_idx: int, task: Task, prediction: int):
        is_correct = (prediction == task.y)
        self.region_estimator.update(task.region)
        self.capability_estimator.update(agent_idx, task.region, is_correct)
        self.history.append({
            'task': task,
            'agent_idx': agent_idx,
            'prediction': prediction,
            'is_correct': is_correct
        })
        self.selected_agents.append(agent_idx)
        self.correctness.append(is_correct)
    
    def get_performance_stats(self) -> Dict:
        return {
            'overall_accuracy': np.mean(self.correctness) if self.correctness else 0,
            'num_tasks': len(self.correctness),
            'agent_usage': {self.agents[i].name: self.selected_agents.count(i) 
                          for i in range(self.K)}
        }


class PaperOrchestrator(BaseOrchestrator):
    """Bhatt et al. (2025): select agent with highest empirical utility."""
    
    def select_agent(self, task: Task, t: int) -> int:
        utilities = np.array([self.compute_empirical_utility(k, task) 
                            for k in range(self.K)])
        return int(np.argmax(utilities))


class GreedyOrchestrator(BaseOrchestrator):
    """Myopic: highest estimated capability in current region."""
    
    def select_agent(self, task: Task, t: int) -> int:
        rt = task.region
        capabilities = np.array([self.capability_estimator.get_capability(k, rt) 
                                for k in range(self.K)])
        return int(np.argmax(capabilities))


class UCB1Orchestrator(BaseOrchestrator):
    """
    UCB1 orchestrator.

    UCB(k, r) = Q(k, r) + c · √(ln(N_r) / N(k, r))

    where N_r = total selections in region r, N(k,r) = selections of
    agent k in region r.  This is the standard per-region UCB1 formulation
    (each region is treated as an independent bandit).
    """
    
    def __init__(self, agents: List[Agent], M: int, c: float = 1.0):
        super().__init__(agents, M)
        self.c = c
        self.agent_region_selections = np.zeros((self.K, M))
        
    def select_agent(self, task: Task, t: int) -> int:
        rt = task.region
        
        # Ensure each agent is tried at least once in this region
        for k in range(self.K):
            if self.agent_region_selections[k, rt] == 0:
                return k
        
        total_in_region = np.sum(self.agent_region_selections[:, rt])
        ucb_scores = np.zeros(self.K)
        
        for k in range(self.K):
            q_value = self.capability_estimator.get_capability(k, rt)
            n_sel = self.agent_region_selections[k, rt]
            exploration = self.c * np.sqrt(np.log(total_in_region + 1) / n_sel)
            ucb_scores[k] = q_value + exploration
        
        return int(np.argmax(ucb_scores))
    
    def update(self, agent_idx: int, task: Task, prediction: int):
        super().update(agent_idx, task, prediction)
        self.agent_region_selections[agent_idx, task.region] += 1


class RandomOrchestrator(BaseOrchestrator):
    """Random baseline: selects agents uniformly at random"""
    
    def select_agent(self, task: Task, t: int) -> int:
        return np.random.randint(0, self.K)


class OracleOrchestrator(BaseOrchestrator):
    """Oracle: perfect knowledge of true capabilities."""
    
    def select_agent(self, task: Task, t: int) -> int:
        rt = task.region
        true_capabilities = np.array([agent.get_capability(rt) for agent in self.agents])
        return int(np.argmax(true_capabilities))
