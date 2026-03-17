"""
Synthetic experiment utilities for multi-agent orchestration.

Provides agent configurations and task generation for the four expertise
scenarios described in Bhatt et al. (2025), plus helpers for running
baseline orchestrators and computing the appropriateness metric.

Agent Scenarios
---------------
Approximately Invariant — all agents have similar P(A_k | R_m) across regions;
                          orchestration provides minimal benefit (App ≈ 1).
Dominant                — one agent strictly outperforms all others in every
                          region; orchestration benefits from identifying it.
Dominant + Mis. Cost    — dominant expertise but costs are misaligned, so the
                          best agent is the most expensive; cost-aware methods
                          may select a cheaper, slightly weaker agent.
Varying                 — each agent excels in a different subset of regions;
                          orchestration must learn which agent to route to which
                          region (the most challenging and rewarding case).

Capability matrices match Appendix B of Bhatt et al. (2025).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from orchestration_framework import (
    Agent, Task, PaperOrchestrator, GreedyOrchestrator,
    UCB1Orchestrator, RandomOrchestrator, OracleOrchestrator,
)


class SyntheticDataGenerator:
    """Generate synthetic data streams for orchestration experiments."""

    def __init__(self, M: int = 3, seed: int = 42):
        """
        Args:
            M:    Number of regions.
            seed: Seed for the isolated Generator — does NOT touch global state.
        """
        self.M = M
        self.seed = seed
        # Isolated Generator: no global np.random.seed() call.
        self._rng = np.random.default_rng(seed)

    def generate_task_stream(self, N: int,
                              region_probs: np.ndarray = None) -> List[Task]:
        """
        Generate a stream of N tasks.

        Args:
            N:            Number of tasks.
            region_probs: Probability distribution over regions (default: uniform).
        """
        if region_probs is None:
            region_probs = np.ones(self.M) / self.M

        tasks = []
        for _ in range(N):
            region = int(self._rng.choice(self.M, p=region_probs))
            x = self._rng.standard_normal(10)
            y = int(self._rng.integers(0, 2))
            tasks.append(Task(x=x, y=y, region=region))

        return tasks


def create_agents_approximately_invariant(M: int = 3) -> List[Agent]:
    """
    Approximately Invariant: all agents have similar capabilities across all
    regions.  Capability matrix from Bhatt et al. (2025) Appendix B.
    """
    capabilities = np.array([
        [0.350, 0.336, 0.314],
        [0.339, 0.338, 0.323],
        [0.349, 0.322, 0.329],
        [0.331, 0.311, 0.357],
    ])
    costs = np.ones((4, M)) * 5.0
    return [Agent(name=f"Agent_{i+1}", capabilities=capabilities[i], costs=costs[i])
            for i in range(4)]


def create_agents_dominant(M: int = 3) -> List[Agent]:
    """
    Dominant: Agent_1 strictly dominates all others in every region.
    Capability matrix from Bhatt et al. (2025) Appendix B.
    """
    capabilities = np.array([
        [0.650, 0.852, 0.877],
        [0.399, 0.298, 0.303],
        [0.079, 0.076, 0.069],
        [0.031, 0.091, 0.274],
    ])
    costs = np.ones((4, M)) * 5.0
    return [Agent(name=f"Agent_{i+1}", capabilities=capabilities[i], costs=costs[i])
            for i in range(4)]


def create_agents_dominant_misaligned_cost(M: int = 3) -> List[Agent]:
    """
    Dominant + Misaligned Cost: same capability matrix as Dominant, but the
    dominant agent (Agent_1) has the highest cost in each region.
    Cost matrix from Bhatt et al. (2025) Appendix B.
    """
    capabilities = np.array([
        [0.650, 0.852, 0.877],
        [0.399, 0.298, 0.303],
        [0.079, 0.076, 0.069],
        [0.031, 0.091, 0.274],
    ])
    costs = np.array([
        [50.915, 120.683, 110.287],
        [51.582, 111.053,   1.412],
        [45.006,   1.568, 123.644],
        [ 1.971, 100.274, 121.872],
    ])
    return [Agent(name=f"Agent_{i+1}", capabilities=capabilities[i], costs=costs[i])
            for i in range(4)]


def create_agents_varying(M: int = 3) -> List[Agent]:
    """
    Varying: each agent excels in a different region.
    Capability matrix from Bhatt et al. (2025) Appendix B.
    """
    capabilities = np.array([
        [0.650, 0.076, 0.274],
        [0.399, 0.298, 0.303],
        [0.079, 0.852, 0.069],
        [0.031, 0.091, 0.877],
    ])
    costs = np.ones((4, M)) * 5.0
    return [Agent(name=f"Agent_{i+1}", capabilities=capabilities[i], costs=costs[i])
            for i in range(4)]


def compute_appropriateness(agents: List[Agent], M: int) -> float:
    """
    Compute appropriateness of orchestration (Bhatt et al. Eq. 5):

        App = C_max / C_rand

    Assumes uniform region distribution.
    """
    K = len(agents)
    region_probs = np.ones(M) / M

    C_max = sum(region_probs[m] * max(a.get_capability(m) for a in agents)
                for m in range(M))

    C_rand = sum(
        sum(region_probs[m] * agents[k].get_capability(m) for m in range(M)) / K
        for k in range(K)
    )

    return C_max / C_rand if C_rand > 0 else 1.0


def run_experiment(
    agents: List[Agent],
    orchestrator_class,
    tasks: List[Task],
    orchestrator_kwargs: Dict = None,
    seed: int = 0,
) -> Dict:
    """
    Run a single orchestration experiment.

    Args:
        agents:             List of agents.
        orchestrator_class: Orchestrator class to instantiate.
        tasks:              Task stream (consumed in order).
        orchestrator_kwargs:Additional keyword arguments for the orchestrator.
        seed:               Seed for the orchestrator's internal RNG.
    """
    M = len(agents[0].capabilities)
    if orchestrator_kwargs is None:
        orchestrator_kwargs = {}

    # Give every orchestrator its own isolated Generator.
    rng = np.random.default_rng(seed)
    pred_rng = np.random.default_rng(seed + 31337)

    orchestrator = orchestrator_class(agents, M, rng=rng, **orchestrator_kwargs)

    for t, task in enumerate(tasks):
        agent_idx = orchestrator.select_agent(task, t)
        prediction = agents[agent_idx].predict(task, rng=pred_rng)
        orchestrator.update(agent_idx, task, prediction)

    stats = orchestrator.get_performance_stats()
    stats['orchestrator_name'] = orchestrator_class.__name__
    stats['appropriateness'] = compute_appropriateness(agents, M)
    return stats


def run_all_baselines(
    agents: List[Agent],
    tasks: List[Task],
    verbose: bool = True,
    seed: int = 0,
) -> Dict[str, Dict]:
    """
    Run all five baseline orchestrators on the same task stream.

    Args:
        agents:  List of agents.
        tasks:   Task stream (a fresh copy is made for each orchestrator).
        verbose: Print results.
        seed:    Base seed for the orchestrators' internal RNGs.
    """
    orchestrators = [
        (RandomOrchestrator, {}),
        (GreedyOrchestrator, {}),
        (UCB1Orchestrator,   {'c': 1.0}),
        (PaperOrchestrator,  {}),
        (OracleOrchestrator, {}),
    ]

    results = {}
    for i, (orchestrator_class, kwargs) in enumerate(orchestrators):
        task_copy = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
        stats = run_experiment(agents, orchestrator_class, task_copy, kwargs,
                               seed=seed + i * 100)
        results[orchestrator_class.__name__] = stats

        if verbose:
            print(f"\n{orchestrator_class.__name__}:")
            print(f"  Overall Accuracy: {stats['overall_accuracy']:.3f}")
            print(f"  Agent Usage: {stats['agent_usage']}")

    return results


def plot_results(results_by_scenario: Dict[str, Dict[str, Dict]],
                 output_path: str = None):
    """Grouped bar chart comparing orchestrators across scenarios."""
    scenarios    = list(results_by_scenario.keys())
    orchestrators = list(next(iter(results_by_scenario.values())).keys())

    accuracies = {orch: [] for orch in orchestrators}
    for scenario in scenarios:
        for orch in orchestrators:
            accuracies[orch].append(
                results_by_scenario[scenario][orch]['overall_accuracy']
            )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios))
    width = 0.15
    for i, orch in enumerate(orchestrators):
        offset = (i - len(orchestrators) / 2) * width
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
                         output_path: str = None,
                         seed: int = 0):
    """Rolling accuracy curves for all orchestrators."""
    M = len(agents[0].capabilities)

    orchestrators = [
        (RandomOrchestrator, {},           'Random'),
        (GreedyOrchestrator, {},           'Greedy'),
        (UCB1Orchestrator,   {'c': 1.0},   'UCB1'),
        (PaperOrchestrator,  {},           'Paper (Bhatt et al.)'),
        (OracleOrchestrator, {},           'Oracle'),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (orchestrator_class, kwargs, label) in enumerate(orchestrators):
        task_copy = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
        pred_rng  = np.random.default_rng(seed + i * 100 + 31337)
        orch_rng  = np.random.default_rng(seed + i * 100)
        orch = orchestrator_class(agents, M, rng=orch_rng, **kwargs)

        accuracies = []
        for t, task in enumerate(task_copy):
            agent_idx = orch.select_agent(task, t)
            prediction = agents[agent_idx].predict(task, rng=pred_rng)
            orch.update(agent_idx, task, prediction)
            if t >= window_size:
                accuracies.append(np.mean(orch.correctness[-window_size:]))

        ax.plot(range(window_size, len(task_copy)), accuracies,
                label=label, linewidth=2)

    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Accuracy (Moving Avg, window={window_size})')
    ax.set_title('Learning Curves: Orchestrator Performance Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
