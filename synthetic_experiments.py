"""
Synthetic Experiments for Multi-Agent Orchestration
Implements the three expertise scenarios from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from orchestration_framework import (
    Agent, Task, PaperOrchestrator, GreedyOrchestrator, 
    UCB1Orchestrator, RandomOrchestrator, OracleOrchestrator
)


class SyntheticDataGenerator:
    """Generate synthetic data streams for orchestration experiments"""
    
    def __init__(self, M: int = 3, seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            M: Number of regions
            seed: Random seed for reproducibility
        """
        self.M = M
        self.seed = seed
        np.random.seed(seed)
        
    def generate_task_stream(self, N: int, region_probs: np.ndarray = None) -> List[Task]:
        """
        Generate a stream of N tasks.
        
        Args:
            N: Number of tasks
            region_probs: Probability distribution over regions (default: uniform)
            
        Returns:
            List of tasks
        """
        if region_probs is None:
            region_probs = np.ones(self.M) / self.M
            
        tasks = []
        for _ in range(N):
            # Sample region
            region = np.random.choice(self.M, p=region_probs)
            
            # For simplicity, use dummy features and binary labels
            x = np.random.randn(10)  # 10-dimensional features
            y = np.random.randint(0, 2)  # Binary classification
            
            tasks.append(Task(x=x, y=y, region=region))
            
        return tasks


def create_agents_approximately_invariant(M: int = 3) -> List[Agent]:
    """
    Create agents with approximately invariant expertise profile.
    All agents have similar capabilities across all regions.
    """
    capabilities = np.array([
        [0.350, 0.336, 0.314],
        [0.339, 0.338, 0.323],
        [0.349, 0.322, 0.329],
        [0.331, 0.311, 0.357]
    ])
    
    # Uniform costs
    costs = np.ones((4, M)) * 5.0
    
    agents = []
    for i in range(4):
        agents.append(Agent(
            name=f"Agent_{i+1}",
            capabilities=capabilities[i],
            costs=costs[i]
        ))
    
    return agents


def create_agents_dominant(M: int = 3) -> List[Agent]:
    """
    Create agents with dominant expertise profile.
    One agent (Agent_1) is strictly better than all others across all regions.
    """
    capabilities = np.array([
        [0.650, 0.852, 0.877],
        [0.399, 0.298, 0.303],
        [0.079, 0.076, 0.069],
        [0.031, 0.091, 0.274]
    ])
    
    # Uniform costs
    costs = np.ones((4, M)) * 5.0
    
    agents = []
    for i in range(4):
        agents.append(Agent(
            name=f"Agent_{i+1}",
            capabilities=capabilities[i],
            costs=costs[i]
        ))
    
    return agents


def create_agents_dominant_misaligned_cost(M: int = 3) -> List[Agent]:
    """
    Create agents with dominant expertise but misaligned costs.
    Best agent has highest cost.
    """
    capabilities = np.array([
        [0.650, 0.852, 0.877],
        [0.399, 0.298, 0.303],
        [0.079, 0.076, 0.069],
        [0.031, 0.091, 0.274]
    ])
    
    # Misaligned costs - dominant agent is expensive
    costs = np.array([
        [50.915, 120.683, 110.287],
        [51.582, 111.053, 1.412],
        [45.006, 1.568, 123.644],
        [1.971, 100.274, 121.872]
    ])
    
    agents = []
    for i in range(4):
        agents.append(Agent(
            name=f"Agent_{i+1}",
            capabilities=capabilities[i],
            costs=costs[i]
        ))
    
    return agents


def create_agents_varying(M: int = 3) -> List[Agent]:
    """
    Create agents with varying expertise profile.
    Each agent excels in different regions.
    """
    capabilities = np.array([
        [0.650, 0.076, 0.274],
        [0.399, 0.298, 0.303],
        [0.079, 0.852, 0.069],
        [0.031, 0.091, 0.877]
    ])
    
    # Uniform costs
    costs = np.ones((4, M)) * 5.0
    
    agents = []
    for i in range(4):
        agents.append(Agent(
            name=f"Agent_{i+1}",
            capabilities=capabilities[i],
            costs=costs[i]
        ))
    
    return agents


def compute_appropriateness(agents: List[Agent], M: int) -> float:
    """
    Compute appropriateness of orchestration (Equation 5).
    App = C_max / C_rand
    
    Args:
        agents: List of agents
        M: Number of regions
        
    Returns:
        Appropriateness score
    """
    K = len(agents)
    
    # Assume uniform region distribution
    region_probs = np.ones(M) / M
    
    # Compute C_max: expected correctness of optimal agent selection per region
    C_max = 0
    for m in range(M):
        max_capability = max(agent.get_capability(m) for agent in agents)
        C_max += region_probs[m] * max_capability
    
    # Compute C_rand: expected correctness of random agent selection
    C_rand = 0
    for k in range(K):
        agent_correctness = sum(region_probs[m] * agents[k].get_capability(m) 
                              for m in range(M))
        C_rand += agent_correctness / K
    
    appropriateness = C_max / C_rand if C_rand > 0 else 1.0
    return appropriateness


def run_experiment(agents: List[Agent], 
                   orchestrator_class,
                   tasks: List[Task],
                   orchestrator_kwargs: Dict = None) -> Dict:
    """
    Run orchestration experiment.
    
    Args:
        agents: List of agents
        orchestrator_class: Class of orchestrator to use
        tasks: List of tasks
        orchestrator_kwargs: Additional kwargs for orchestrator
        
    Returns:
        Dictionary with results
    """
    M = len(agents[0].capabilities)
    
    if orchestrator_kwargs is None:
        orchestrator_kwargs = {}
    
    orchestrator = orchestrator_class(agents, M, **orchestrator_kwargs)
    
    # Run through task stream
    for t, task in enumerate(tasks):
        # Select agent
        agent_idx = orchestrator.select_agent(task, t)
        
        # Get prediction
        prediction = agents[agent_idx].predict(task)
        
        # Update orchestrator
        orchestrator.update(agent_idx, task, prediction)
    
    # Get performance stats
    stats = orchestrator.get_performance_stats()
    
    # Add additional metrics
    stats['orchestrator_name'] = orchestrator_class.__name__
    stats['appropriateness'] = compute_appropriateness(agents, M)
    
    return stats


def run_all_baselines(agents: List[Agent], 
                     tasks: List[Task],
                     verbose: bool = True) -> Dict[str, Dict]:
    """
    Run all baseline orchestrators on the same task stream.
    
    Args:
        agents: List of agents
        tasks: List of tasks
        verbose: Whether to print results
        
    Returns:
        Dictionary mapping orchestrator name to results
    """
    results = {}
    
    # Define orchestrators to test
    orchestrators = [
        (RandomOrchestrator, {}),
        (GreedyOrchestrator, {}),
        (UCB1Orchestrator, {'c': 1.0}),
        (PaperOrchestrator, {}),
        (OracleOrchestrator, {})
    ]
    
    for orchestrator_class, kwargs in orchestrators:
        # Create fresh copy of tasks for each orchestrator
        task_copy = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
        
        # Run experiment
        stats = run_experiment(agents, orchestrator_class, task_copy, kwargs)
        results[orchestrator_class.__name__] = stats
        
        if verbose:
            print(f"\n{orchestrator_class.__name__}:")
            print(f"  Overall Accuracy: {stats['overall_accuracy']:.3f}")
            print(f"  Agent Usage: {stats['agent_usage']}")
    
    return results


def plot_results(results_by_scenario: Dict[str, Dict[str, Dict]],
                output_path: str = None):
    """
    Plot comparison of orchestrators across scenarios.
    
    Args:
        results_by_scenario: Nested dict {scenario_name: {orchestrator_name: stats}}
        output_path: Path to save figure
    """
    scenarios = list(results_by_scenario.keys())
    orchestrators = list(next(iter(results_by_scenario.values())).keys())
    
    # Extract accuracies
    accuracies = {orch: [] for orch in orchestrators}
    for scenario in scenarios:
        for orch in orchestrators:
            acc = results_by_scenario[scenario][orch]['overall_accuracy']
            accuracies[orch].append(acc)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.15
    
    for i, orch in enumerate(orchestrators):
        offset = (i - len(orchestrators)/2) * width
        ax.bar(x + offset, accuracies[orch], width, label=orch)
    
    ax.set_xlabel('Expertise Scenario')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Orchestrator Performance Across Expertise Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curves(agents: List[Agent],
                        tasks: List[Task],
                        window_size: int = 50,
                        output_path: str = None):
    """
    Plot learning curves showing accuracy over time for each orchestrator.
    
    Args:
        agents: List of agents
        tasks: List of tasks
        window_size: Window size for moving average
        output_path: Path to save figure
    """
    M = len(agents[0].capabilities)
    
    orchestrators = [
        (RandomOrchestrator, {}, 'Random'),
        (GreedyOrchestrator, {}, 'Greedy'),
        (UCB1Orchestrator, {'c': 1.0}, 'UCB1'),
        (PaperOrchestrator, {}, 'Paper (Bhatt et al.)'),
        (OracleOrchestrator, {}, 'Oracle')
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for orchestrator_class, kwargs, label in orchestrators:
        # Create fresh copy of tasks
        task_copy = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
        
        # Initialize orchestrator
        orch = orchestrator_class(agents, M, **kwargs)
        
        # Run and track accuracy
        accuracies = []
        for t, task in enumerate(task_copy):
            agent_idx = orch.select_agent(task, t)
            prediction = agents[agent_idx].predict(task)
            orch.update(agent_idx, task, prediction)
            
            # Compute moving average accuracy
            if t >= window_size:
                recent_correctness = orch.correctness[-window_size:]
                accuracies.append(np.mean(recent_correctness))
        
        # Plot
        ax.plot(range(window_size, len(task_copy)), accuracies, label=label, linewidth=2)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Accuracy (Moving Avg, window={window_size})')
    ax.set_title('Learning Curves: Orchestrator Performance Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
