"""
Run machine teaching experiments and compare all teacher variants.
"""

import sys
import os
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from synthetic_experiments import (
    create_agents_approximately_invariant,
    create_agents_dominant,
    create_agents_dominant_misaligned_cost,
    create_agents_varying,
    compute_appropriateness,
)
from machine_teaching import (
    TaskPool,
    OmniscientTeacher,
    SurrogateTeacher,
    ImitationTeacher,
    RandomTeacher,
    RoundRobinTeacher,
    run_teaching_experiment,
    compute_teaching_efficiency,
    print_teaching_results,
)

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "OmniscientTeacher": "#2196F3",
    "ImitationTeacher":  "#4CAF50",
    "SurrogateTeacher":  "#FF9800",
    "RoundRobinTeacher": "#9C27B0",
    "RandomTeacher":     "#F44336",
}
LABELS = {
    "OmniscientTeacher": "Omniscient",
    "ImitationTeacher":  "Imitation",
    "SurrogateTeacher":  "Surrogate",
    "RoundRobinTeacher": "Round-Robin",
    "RandomTeacher":     "Random",
}

TEACHER_CLASSES = [
    (OmniscientTeacher, {}),
    (ImitationTeacher,  {"eta_v": 0.3}),
    (SurrogateTeacher,  {}),
    (RoundRobinTeacher, {}),
    (RandomTeacher,     {}),
]

SCENARIOS = {
    "Approximately\nInvariant":    create_agents_approximately_invariant,
    "Dominant":                    create_agents_dominant,
    "Dominant +\nMisaligned Cost": create_agents_dominant_misaligned_cost,
    "Varying":                     create_agents_varying,
}

M       = 3
BUDGET  = 200          # evaluation budget
SEED    = 42
TARGET_MSE = 0.005     # threshold for efficiency metric


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Run all experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_all(budget: int = BUDGET, seed: int = SEED) -> dict:
    all_results = {}
    print("=" * 70)
    print("Machine Teaching — Capability Identification Experiments")
    print(f"Budget: {budget} evaluation steps  |  M={M} regions  |  seed={seed}")
    print("=" * 70)

    for scenario_name, agent_factory in SCENARIOS.items():
        clean = scenario_name.replace("\n", " ")
        print(f"\n{'─'*60}")
        print(f"Scenario: {clean}")
        agents = agent_factory(M)
        app = compute_appropriateness(agents, M)
        print(f"  Appropriateness: {app:.3f}")

        results = run_teaching_experiment(
            agents=agents,
            M=M,
            budget=budget,
            teacher_classes=TEACHER_CLASSES,
            seed=seed,
        )
        all_results[clean] = results
        print_teaching_results(results, target_mse=TARGET_MSE)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_mse_curves(all_results: dict, output_path: str = None):
    """4-panel figure: MSE-over-steps for every scenario."""
    scenarios = list(all_results.keys())
    n = len(scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios):
        results = all_results[scenario]
        for name, summary in results.items():
            curve = summary["mse_curve"]
            ax.plot(
                range(len(curve)), curve,
                color=COLORS.get(name, "black"),
                label=LABELS.get(name, name),
                linewidth=2,
            )
        ax.axhline(TARGET_MSE, color="grey", linestyle="--", linewidth=1,
                   label=f"Target MSE={TARGET_MSE}")
        ax.set_title(scenario, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Steps (Budget)")
        ax.set_ylabel("MSE (estimated vs true)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Machine Teaching: Capability Estimation MSE over Evaluation Budget",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_efficiency_bars(all_results: dict, output_path: str = None):
    """Bar chart: steps-to-target for each teacher × scenario."""
    scenarios = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)

    x = np.arange(len(scenarios))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, name in enumerate(teacher_names):
        steps = []
        for scenario in scenarios:
            eff = compute_teaching_efficiency(all_results[scenario], TARGET_MSE)
            v = eff.get(name)
            steps.append(v if v is not None else BUDGET)   # cap at budget

        offset = (i - n_t / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, steps, width,
            label=LABELS.get(name, name),
            color=COLORS.get(name, "grey"),
            alpha=0.85,
            edgecolor="white",
        )

    ax.axhline(BUDGET, color="grey", linestyle="--", linewidth=1,
               label="Max budget (not reached)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel(f"Steps to reach MSE < {TARGET_MSE}")
    ax.set_title("Teaching Efficiency: Evaluation Steps to Target MSE",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, BUDGET * 1.12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_final_mse_heatmap(all_results: dict, output_path: str = None):
    """Heatmap of final MSE: rows = scenarios, cols = teachers."""
    scenarios = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    data = np.zeros((len(scenarios), len(teacher_names)))
    for i, sc in enumerate(scenarios):
        for j, tn in enumerate(teacher_names):
            data[i, j] = all_results[sc][tn]["final_mse"]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Final MSE")

    ax.set_xticks(range(len(teacher_names)))
    ax.set_xticklabels([LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=10)
    ax.set_title(f"Final MSE after {BUDGET} Evaluation Steps",
                 fontsize=12, fontweight="bold")

    for i in range(len(scenarios)):
        for j in range(len(teacher_names)):
            ax.text(j, i, f"{data[i,j]:.4f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i,j] < data.max() * 0.6 else "white")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_capability_estimates(all_results: dict, scenario: str = "Varying",
                               output_path: str = None):
    """
    For the Varying scenario, compare true vs estimated capabilities for
    each teacher after the full budget.
    """
    results = all_results.get(scenario)
    if results is None:
        return

    K = results["OmniscientTeacher"]["true_caps"].shape[0]
    M_local = results["OmniscientTeacher"]["true_caps"].shape[1]
    true_caps = results["OmniscientTeacher"]["true_caps"]

    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)

    fig, axes = plt.subplots(K, M_local, figsize=(4 * M_local, 3.5 * K))

    for k in range(K):
        for m in range(M_local):
            ax = axes[k, m]
            truth = true_caps[k, m]

            # Plot estimates as scatter for each teacher
            estimates = [results[tn]["final_estimates"][k, m] for tn in teacher_names]
            labels_plot = [LABELS.get(n, n) for n in teacher_names]
            colors_plot = [COLORS.get(n, "grey") for n in teacher_names]

            ax.scatter(range(n_t), estimates, color=colors_plot, s=80, zorder=3)
            ax.axhline(truth, color="black", linestyle="--", linewidth=1.5,
                       label=f"True={truth:.3f}")
            ax.set_xticks(range(n_t))
            ax.set_xticklabels(labels_plot, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("P(correct)")
            ax.set_title(f"Agent {k+1} | Region {m}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.25)

    fig.suptitle(
        f'Capability Estimates vs Truth — "{scenario}" Scenario\n'
        f'after {BUDGET} evaluation steps',
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Downstream orchestration utility evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_downstream_orchestration(all_results: dict, n_tasks: int = 500):
    """
    Use the estimated capabilities from each teacher to run the PaperOrchestrator
    downstream.  Compares final orchestration accuracy across teaching methods.
    """
    # Import here to avoid circular dependencies at module level
    from orchestration_framework import (
        Agent, Task, PaperOrchestrator, OracleOrchestrator
    )
    from synthetic_experiments import (
        SyntheticDataGenerator,
        create_agents_varying,
    )

    print("\n" + "=" * 70)
    print("DOWNSTREAM ORCHESTRATION UTILITY (Varying scenario)")
    print(f"  {n_tasks} orchestration tasks using capabilities estimated by each teacher")
    print("=" * 70)

    agents = create_agents_varying(M)
    np.random.seed(SEED)
    gen = SyntheticDataGenerator(M=M, seed=SEED + 1)
    tasks = gen.generate_task_stream(n_tasks)

    scenario = "Varying"
    results = all_results[scenario]
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    downstream = {}

    for tname in teacher_names:
        estimated = results[tname]["final_estimates"]   # K × M

        # Build synthetic "estimated" agents whose capabilities are what the
        # teacher inferred (costs remain uniform = 5.0)
        estimated_agents = [
            Agent(
                name=agents[k].name,
                capabilities=estimated[k],
                costs=agents[k].costs,
            )
            for k in range(len(agents))
        ]

        orch = PaperOrchestrator(estimated_agents, M)
        for t, task in enumerate(tasks):
            agent_idx = orch.select_agent(task, t)
            # Prediction uses TRUE agent, not estimated
            prediction = agents[agent_idx].predict(task)
            orch.update(agent_idx, task, prediction)

        stats = orch.get_performance_stats()
        downstream[tname] = stats["overall_accuracy"]

    # Oracle (true capabilities)
    oracle_orch = OracleOrchestrator(agents, M)
    for t, task in enumerate(tasks):
        agent_idx = oracle_orch.select_agent(task, t)
        prediction = agents[agent_idx].predict(task)
        oracle_orch.update(agent_idx, task, prediction)
    oracle_acc = oracle_orch.get_performance_stats()["overall_accuracy"]

    print(f"\n{'Teacher':<25} {'Downstream Accuracy':>22}")
    print("-" * 50)
    for tname, acc in downstream.items():
        label = LABELS.get(tname, tname)
        print(f"{label:<25} {acc:>22.3f}")
    print(f"{'Oracle (true caps)':<25} {oracle_acc:>22.3f}")

    return downstream, oracle_acc


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Run teaching experiments
    all_results = run_all(budget=BUDGET, seed=SEED)

    # Figures
    print("\nGenerating figures...")

    fig1 = plot_mse_curves(all_results, "output/teaching_mse_curves.png")
    print("  Saved: output/teaching_mse_curves.png")

    fig2 = plot_efficiency_bars(all_results, "output/teaching_efficiency.png")
    print("  Saved: output/teaching_efficiency.png")

    fig3 = plot_final_mse_heatmap(all_results, "output/teaching_mse_heatmap.png")
    print("  Saved: output/teaching_mse_heatmap.png")

    fig4 = plot_capability_estimates(
        all_results, scenario="Varying",
        output_path="output/capability_estimates_varying.png"
    )
    print("  Saved: output/capability_estimates_varying.png")

    # Downstream orchestration evaluation
    downstream, oracle_acc = evaluate_downstream_orchestration(all_results)

    print("\nDone.")
