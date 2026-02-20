"""
Multi-Agent Orchestration Framework
Based on Bhatt et al. (2025) "When Should We Orchestrate Multiple Agents?"
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
        """
        Initialize an agent.
        
        Args:
            name: Agent identifier
            capabilities: Array of shape (M,) with P(correct | region) for each region
            costs: Array of shape (M,) with cost for each region
        """
        self.name = name
        self.capabilities = capabilities  # P(Ak | Rm)
        self.costs = costs  # gamma_km
        self.M = len(capabilities)  # Number of regions
        
    def predict(self, task: Task) -> int:
        """
        Make a prediction for a task based on agent's capability in that region.
        
        Args:
            task: The task to predict
            
        Returns:
            Predicted label (0 or 1 for binary, or any int for multi-class)
        """
        # Simulate agent prediction based on capability
        prob_correct = self.capabilities[task.region]
        is_correct = np.random.random() < prob_correct
        
        if is_correct:
            return task.y
        else:
            # Return a random incorrect answer
            # For simplicity, assume binary classification
            return 1 - task.y
    
    def get_capability(self, region: int) -> float:
        """Get capability for a specific region"""
        return self.capabilities[region]
    
    def get_cost(self, region: int) -> float:
        """Get cost for a specific region"""
        return self.costs[region]


class RegionEstimator:
    """Estimates region probabilities using Beta-Binomial model"""
    
    def __init__(self, M: int, alpha: Optional[np.ndarray] = None):
        """
        Initialize region estimator.
        
        Args:
            M: Number of regions
            alpha: Dirichlet prior parameters (default: uniform)
        """
        self.M = M
        self.alpha = alpha if alpha is not None else np.ones(M)
        self.counts = np.zeros(M)  # n_<t,m
        
    def update(self, region: int):
        """Update counts after observing a task from a region"""
        self.counts[region] += 1
        
    def get_probabilities(self) -> np.ndarray:
        """
        Get current estimate of region probabilities.
        
        Returns:
            Array of shape (M,) with estimated P(Rm)
        """
        numerator = self.counts + self.alpha - 1
        denominator = np.sum(numerator)
        return numerator / denominator if denominator > 0 else self.alpha / np.sum(self.alpha)


class CapabilityEstimator:
    """Estimates agent capabilities per region using Beta-Binomial model"""
    
    def __init__(self, K: int, M: int, alpha0: float = 1.0, alpha1: float = 1.0):
        """
        Initialize capability estimator.
        
        Args:
            K: Number of agents
            M: Number of regions
            alpha0: Prior pseudo-count for incorrect
            alpha1: Prior pseudo-count for correct
        """
        self.K = K
        self.M = M
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
        # Counts: [agent, region, outcome] where outcome is 0 (incorrect) or 1 (correct)
        self.counts = np.zeros((K, M, 2))
        
    def update(self, agent_idx: int, region: int, is_correct: bool):
        """Update counts after agent makes a prediction"""
        outcome = 1 if is_correct else 0
        self.counts[agent_idx, region, outcome] += 1
        
    def get_capability(self, agent_idx: int, region: int) -> float:
        """
        Get estimated capability for an agent in a region.
        
        Returns:
            Estimated P(Ak | Rm)
        """
        n_incorrect = self.counts[agent_idx, region, 0]
        n_correct = self.counts[agent_idx, region, 1]
        
        numerator = n_correct + self.alpha1 - 1
        denominator = (n_incorrect + n_correct) + (self.alpha0 + self.alpha1) - 2
        
        if denominator > 0:
            return numerator / denominator
        else:
            return self.alpha1 / (self.alpha0 + self.alpha1)
    
    def get_all_capabilities(self) -> np.ndarray:
        """Get estimated capabilities for all agents and regions"""
        capabilities = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                capabilities[k, m] = self.get_capability(k, m)
        return capabilities


class BaseOrchestrator(ABC):
    """Base class for orchestration algorithms"""
    
    def __init__(self, agents: List[Agent], M: int):
        """
        Initialize orchestrator.
        
        Args:
            agents: List of available agents
            M: Number of regions
        """
        self.agents = agents
        self.K = len(agents)
        self.M = M
        
        # Initialize estimators
        self.region_estimator = RegionEstimator(M)
        self.capability_estimator = CapabilityEstimator(self.K, M)
        
        # Track history
        self.history = []
        self.selected_agents = []
        self.correctness = []
        
    @abstractmethod
    def select_agent(self, task: Task, t: int) -> int:
        """
        Select an agent for the given task.
        
        Args:
            task: Current task
            t: Time step
            
        Returns:
            Index of selected agent
        """
        pass
    
    def compute_empirical_utility(self, agent_idx: int, task: Task) -> float:
        """
        Compute empirical utility for an agent (Equation 4 from paper).
        
        Args:
            agent_idx: Index of the agent
            task: Current task
            
        Returns:
            Empirical utility estimate
        """
        rt = task.region
        
        # Get current capability estimate for this agent in current region
        c_t_krt = self.capability_estimator.get_capability(agent_idx, rt)
        
        # Get region probabilities
        w_t = self.region_estimator.get_probabilities()
        
        # Get capabilities across all regions
        c_t_k = np.array([self.capability_estimator.get_capability(agent_idx, m) 
                         for m in range(self.M)])
        
        # Compute onwards correctness (Equation 4)
        future_correctness = np.sum(w_t * c_t_k)
        empirical_correctness = c_t_krt * future_correctness
        
        # Account for cost
        cost = self.agents[agent_idx].get_cost(rt)
        empirical_utility = empirical_correctness / cost if cost > 0 else empirical_correctness
        
        return empirical_utility
    
    def update(self, agent_idx: int, task: Task, prediction: int):
        """Update estimators after agent makes prediction"""
        is_correct = (prediction == task.y)
        
        # Update region estimator
        self.region_estimator.update(task.region)
        
        # Update capability estimator
        self.capability_estimator.update(agent_idx, task.region, is_correct)
        
        # Track history
        self.history.append({
            'task': task,
            'agent_idx': agent_idx,
            'prediction': prediction,
            'is_correct': is_correct
        })
        self.selected_agents.append(agent_idx)
        self.correctness.append(is_correct)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'overall_accuracy': np.mean(self.correctness) if self.correctness else 0,
            'num_tasks': len(self.correctness),
            'agent_usage': {self.agents[i].name: self.selected_agents.count(i) 
                          for i in range(self.K)}
        }


class PaperOrchestrator(BaseOrchestrator):
    """
    Orchestrator as defined in Bhatt et al. (2025).
    Selects agent with highest empirical utility.
    """
    
    def select_agent(self, task: Task, t: int) -> int:
        """Select agent with highest empirical utility"""
        utilities = np.array([self.compute_empirical_utility(k, task) 
                            for k in range(self.K)])
        return np.argmax(utilities)


class GreedyOrchestrator(BaseOrchestrator):
    """
    Greedy orchestrator: always selects agent with highest estimated capability
    in the current region, without considering future performance or cost.
    """
    
    def select_agent(self, task: Task, t: int) -> int:
        """Select agent with highest estimated capability in current region"""
        rt = task.region
        capabilities = np.array([self.capability_estimator.get_capability(k, rt) 
                                for k in range(self.K)])
        return np.argmax(capabilities)


class UCB1Orchestrator(BaseOrchestrator):
    """
    Upper Confidence Bound (UCB1) orchestrator.
    Balances exploitation and exploration using confidence bounds.
    """
    
    def __init__(self, agents: List[Agent], M: int, c: float = 1.0):
        """
        Initialize UCB1 orchestrator.
        
        Args:
            agents: List of available agents
            M: Number of regions
            c: Exploration parameter (default: 1.0)
        """
        super().__init__(agents, M)
        self.c = c
        self.agent_region_selections = np.zeros((self.K, M))  # Track selections per agent-region
        
    def select_agent(self, task: Task, t: int) -> int:
        """
        Select agent using UCB1 algorithm.
        
        UCB1 formula: Q(a) + c * sqrt(ln(t) / N(a,r))
        where Q(a) is estimated capability, t is total time steps,
        N(a,r) is number of times agent a was selected for region r
        """
        rt = task.region
        
        # For first K*M steps, ensure each agent-region pair is tried once
        if t < self.K * self.M:
            agent_idx = t % self.K
            return agent_idx
        
        ucb_scores = np.zeros(self.K)
        
        for k in range(self.K):
            # Estimated capability (exploitation term)
            q_value = self.capability_estimator.get_capability(k, rt)
            
            # Number of times this agent-region pair was selected
            n_selections = self.agent_region_selections[k, rt]
            
            # Exploration bonus
            if n_selections > 0:
                exploration_bonus = self.c * np.sqrt(np.log(t + 1) / n_selections)
            else:
                exploration_bonus = float('inf')  # Force exploration if never tried
            
            ucb_scores[k] = q_value + exploration_bonus
        
        return np.argmax(ucb_scores)
    
    def update(self, agent_idx: int, task: Task, prediction: int):
        """Update with UCB1-specific tracking"""
        super().update(agent_idx, task, prediction)
        self.agent_region_selections[agent_idx, task.region] += 1


class RandomOrchestrator(BaseOrchestrator):
    """Random baseline: selects agents uniformly at random"""
    
    def select_agent(self, task: Task, t: int) -> int:
        """Select random agent"""
        return np.random.randint(0, self.K)


class OracleOrchestrator(BaseOrchestrator):
    """
    Oracle orchestrator: has perfect knowledge of agent capabilities.
    Selects the agent with highest true capability in each region.
    This represents the theoretical maximum performance.
    """
    
    def select_agent(self, task: Task, t: int) -> int:
        """Select agent with highest true capability in current region"""
        rt = task.region
        true_capabilities = np.array([agent.get_capability(rt) for agent in self.agents])
        return np.argmax(true_capabilities)
