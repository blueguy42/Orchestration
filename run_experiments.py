"""
run_experiments.py
==================
Unified experiment runner for:
  1. Baseline orchestrators  (Random, Greedy, UCB1, Paper/Umang, Oracle)
  2. Machine-teaching orchestrators (Omniscient, Imitation, Surrogate,
                                      Round-Robin, Random teacher)
  3. Prior sensitivity analysis  (Dirichlet/Beta-uniform, Jeffreys, Expert)

Produces all figures to output/ and prints console tables.

Usage:
    python run_experiments.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────────────────────────
from orchestration_framework import (
    Agent, Task,
    PaperOrchestrator, GreedyOrchestrator,
    UCB1Orchestrator, RandomOrchestrator, OracleOrchestrator,
)
from machine_teaching import (
    TaskPool,
    OmniscientTeacher, SurrogateTeacher, ImitationTeacher,
    RandomTeacher, RoundRobinTeacher,
    run_teaching_experiment,
    compute_teaching_efficiency,
    print_teaching_results,
)
from synthetic_experiments import (
    SyntheticDataGenerator,
    create_agents_approximately_invariant,
    create_agents_dominant,
    create_agents_dominant_misaligned_cost,
    create_agents_varying,
    compute_appropriateness,
    run_experiment,
    run_all_baselines,
)
from priors import (
    BetaPrior, JeffreysPrior, SkewedExpertPrior,
    ALL_PRIORS, PRIOR_LABELS, PRIOR_COLORS,
)

os.makedirs("output", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Global palette / label maps
# ──────────────────────────────────────────────────────────────────────────────

# ── Baseline orchestrators
BASELINE_COLORS = {
    "RandomOrchestrator":   "#F44336",
    "GreedyOrchestrator":   "#FF9800",
    "UCB1Orchestrator":     "#2196F3",
    "PaperOrchestrator":    "#9C27B0",
    "OracleOrchestrator":   "#4CAF50",
}
BASELINE_LABELS = {
    "RandomOrchestrator":   "Random",
    "GreedyOrchestrator":   "Greedy",
    "UCB1Orchestrator":     "UCB1",
    "PaperOrchestrator":    "Paper (Bhatt et al.)",
    "OracleOrchestrator":   "Oracle",
}

# ── Machine-teaching teachers
TEACHER_COLORS = {
    "OmniscientTeacher":  "#2196F3",
    "ImitationTeacher":   "#4CAF50",
    "SurrogateTeacher":   "#FF9800",
    "RoundRobinTeacher":  "#9C27B0",
    "RandomTeacher":      "#F44336",
}
TEACHER_LABELS = {
    "OmniscientTeacher":  "Omniscient",
    "ImitationTeacher":   "Imitation",
    "SurrogateTeacher":   "Surrogate",
    "RoundRobinTeacher":  "Round-Robin",
    "RandomTeacher":      "Random",
}

TEACHER_CLASSES = [
    (OmniscientTeacher, {}),
    (ImitationTeacher,  {"eta_v": 0.3}),
    (SurrogateTeacher,  {}),
    (RoundRobinTeacher, {}),
    (RandomTeacher,     {}),
]

SCENARIOS = {
    "Approx. Invariant":     create_agents_approximately_invariant,
    "Dominant":              create_agents_dominant,
    "Dominant + Mis. Cost":  create_agents_dominant_misaligned_cost,
    "Varying":               create_agents_varying,
}

M          = 3
N_TASKS    = 1000
BUDGET     = 200
SEED       = 42
TARGET_MSE = 0.005


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.28,
    "grid.linestyle":   "--",
})

FIGSIZE_DEFAULT = (13, 6)


def _bar_labels(ax, fmt="{:.3f}", fontsize=8, color="black", padding=1):
    """Annotate every bar in *ax* with its height at the top."""
    for rect in ax.patches:
        h = rect.get_height()
        if h > 0:
            ax.annotate(
                fmt.format(h),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, padding),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=fontsize, color=color,
            )


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Baseline orchestrator experiments
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_experiments() -> dict:
    """Run all baseline orchestrators on all scenarios."""
    print("\n" + "=" * 70)
    print("PART 1 — Baseline Orchestrators")
    print("=" * 70)

    data_gen = SyntheticDataGenerator(M=M, seed=SEED)
    tasks = data_gen.generate_task_stream(N_TASKS)

    all_results = {}
    for scenario_name, agent_factory in SCENARIOS.items():
        agents = agent_factory(M)
        app = compute_appropriateness(agents, M)
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario_name}  (appropriateness={app:.3f})")
        results = run_all_baselines(agents, tasks, verbose=True)
        results["__app__"] = app
        all_results[scenario_name] = results

    return all_results


def plot_baseline_accuracy(all_results: dict, output_path: str = None):
    """Grouped bar chart: accuracy per scenario × baseline orchestrator."""
    scenarios = list(all_results.keys())
    orch_names = [k for k in list(next(iter(all_results.values())).keys())
                  if k != "__app__"]

    x = np.arange(len(scenarios))
    n = len(orch_names)
    width = 0.14

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, name in enumerate(orch_names):
        accs = [all_results[sc][name]["overall_accuracy"] for sc in scenarios]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, accs, width,
            label=BASELINE_LABELS.get(name, name),
            color=BASELINE_COLORS.get(name, "grey"),
            alpha=0.88,
            edgecolor="white",
        )

    # Numbers at top of every bar
    _bar_labels(ax, fmt="{:.3f}", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10.5)
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Baseline Orchestrators — Accuracy by Scenario",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.12)
    plt.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


def plot_baseline_learning_curves(all_results: dict, output_path: str = None,
                                   window: int = 50):
    """
    4-panel rolling-accuracy curves (one per scenario).
    Each panel shows all 5 baselines on the same task stream.
    """
    scenarios = list(all_results.keys())
    orch_defs = [
        (RandomOrchestrator,  {}, "RandomOrchestrator"),
        (GreedyOrchestrator,  {}, "GreedyOrchestrator"),
        (UCB1Orchestrator,    {"c": 1.0}, "UCB1Orchestrator"),
        (PaperOrchestrator,   {}, "PaperOrchestrator"),
        (OracleOrchestrator,  {}, "OracleOrchestrator"),
    ]

    data_gen = SyntheticDataGenerator(M=M, seed=SEED)
    tasks = data_gen.generate_task_stream(N_TASKS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, scenario_name in zip(axes, scenarios):
        agents = SCENARIOS[scenario_name](M)
        for cls, kwargs, key in orch_defs:
            task_copy = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
            orch = cls(agents, M, **kwargs)
            accs = []
            for t, task in enumerate(task_copy):
                aidx = orch.select_agent(task, t)
                pred = agents[aidx].predict(task)
                orch.update(aidx, task, pred)
                if t >= window:
                    accs.append(np.mean(orch.correctness[-window:]))
            ax.plot(
                range(window, len(task_copy)), accs,
                color=BASELINE_COLORS[key],
                label=BASELINE_LABELS[key],
                linewidth=2,
            )
        ax.set_title(scenario_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Task Index")
        ax.set_ylabel(f"Rolling Accuracy (w={window})")
        ax.legend(fontsize=7.5)

    fig.suptitle("Baseline Orchestrators — Learning Curves",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Machine-teaching experiments
# ──────────────────────────────────────────────────────────────────────────────

def run_teaching_experiments() -> dict:
    """Run all teacher variants on all scenarios."""
    print("\n" + "=" * 70)
    print("PART 2 — Machine Teaching")
    print(f"Budget: {BUDGET}  |  M={M}  |  Target MSE < {TARGET_MSE}")
    print("=" * 70)

    all_results = {}
    for scenario_name, agent_factory in SCENARIOS.items():
        agents = agent_factory(M)
        app = compute_appropriateness(agents, M)
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario_name}  (appropriateness={app:.3f})")

        results = run_teaching_experiment(
            agents=agents, M=M, budget=BUDGET,
            teacher_classes=TEACHER_CLASSES, seed=SEED,
        )
        all_results[scenario_name] = results
        print_teaching_results(results, target_mse=TARGET_MSE)

    return all_results


def plot_mse_curves(all_results: dict, output_path: str = None):
    """4-panel MSE-over-steps for every scenario."""
    scenarios = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios):
        for name, summary in all_results[scenario].items():
            curve = summary["mse_curve"]
            ax.plot(
                range(len(curve)), curve,
                color=TEACHER_COLORS.get(name, "black"),
                label=TEACHER_LABELS.get(name, name),
                linewidth=2,
            )
        ax.axhline(TARGET_MSE, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Target MSE={TARGET_MSE}")
        ax.set_title(scenario, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Steps (Budget)")
        ax.set_ylabel("MSE (estimated vs true)")
        ax.legend(fontsize=7.5)

    fig.suptitle("Machine Teaching — Capability Estimation MSE",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_efficiency_bars(all_results: dict, output_path: str = None):
    """Bar chart: steps-to-target-MSE per teacher × scenario."""
    scenarios = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)
    x = np.arange(len(scenarios))
    width = 0.15

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    for i, name in enumerate(teacher_names):
        steps = []
        for sc in scenarios:
            eff = compute_teaching_efficiency(all_results[sc], TARGET_MSE)
            v = eff.get(name)
            steps.append(v if v is not None else BUDGET)
        offset = (i - n_t / 2 + 0.5) * width
        ax.bar(
            x + offset, steps, width,
            label=TEACHER_LABELS.get(name, name),
            color=TEACHER_COLORS.get(name, "grey"),
            alpha=0.87,
            edgecolor="white",
        )

    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    ax.axhline(BUDGET, color="grey", linestyle="--", linewidth=1,
               label="Max budget (not reached)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel(f"Steps to MSE < {TARGET_MSE}")
    ax.set_title("Teaching Efficiency — Steps to Target MSE",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, BUDGET * 1.18)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_final_mse_heatmap(all_results: dict, output_path: str = None):
    """Heatmap of final MSE: rows=scenarios, cols=teachers."""
    scenarios = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    data = np.array([
        [all_results[sc][tn]["final_mse"] for tn in teacher_names]
        for sc in scenarios
    ])

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Final MSE")

    ax.set_xticks(range(len(teacher_names)))
    ax.set_xticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
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
        _save(fig, output_path)
    return fig


def plot_capability_estimates(all_results: dict, scenario: str = "Varying",
                              output_path: str = None):
    """
    For one scenario: scatter of true vs estimated capability per (agent, region)
    for every teacher, plus posterior uncertainty bands.
    """
    results = all_results.get(scenario)
    if results is None:
        return

    true_caps = results["OmniscientTeacher"]["true_caps"]
    K, M_local = true_caps.shape
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)

    fig, axes = plt.subplots(K, M_local, figsize=(4.5 * M_local, 3.5 * K))

    for k in range(K):
        for m in range(M_local):
            ax = axes[k, m]
            truth = true_caps[k, m]
            estimates = [results[tn]["final_estimates"][k, m] for tn in teacher_names]
            colors_p  = [TEACHER_COLORS.get(n, "grey") for n in teacher_names]
            labels_p  = [TEACHER_LABELS.get(n, n) for n in teacher_names]

            ax.scatter(range(n_t), estimates, color=colors_p, s=80, zorder=3)
            ax.axhline(truth, color="black", linestyle="--", linewidth=1.5,
                       label=f"True={truth:.3f}")
            ax.set_xticks(range(n_t))
            ax.set_xticklabels(labels_p, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("P(correct)")
            ax.set_title(f"Agent {k+1} | Region {m}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f'Capability Estimates vs Truth — "{scenario}" — after {BUDGET} steps',
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_convergence_speed(all_results: dict, output_path: str = None):
    """
    Convergence speed metric: for each teacher, steps until MSE first
    drops below 2×, 1×, and 0.5× TARGET_MSE — shown as a grouped bar
    chart for the 'Varying' scenario and as a table for all scenarios.
    """
    scenario = "Varying"
    results = all_results.get(scenario, {})
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    thresholds = [TARGET_MSE * 2, TARGET_MSE, TARGET_MSE * 0.5]
    thresh_labels = ["2× target", "1× target", "½× target"]
    colors_thresh = ["#90CAF9", "#2196F3", "#0D47A1"]

    x = np.arange(len(teacher_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    for i, (thr, lbl, col) in enumerate(zip(thresholds, thresh_labels, colors_thresh)):
        steps = []
        for name in teacher_names:
            curve = results[name]["mse_curve"] if name in results else []
            reached = next((t for t, v in enumerate(curve) if v <= thr), None)
            steps.append(reached if reached is not None else BUDGET)
        offset = (i - 1) * width
        ax.bar(x + offset, steps, width, label=lbl, color=col, alpha=0.88, edgecolor="white")

    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_ylabel("Steps until threshold reached")
    ax.set_title(f"Convergence Speed — \"{scenario}\" Scenario",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, BUDGET * 1.18)
    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_mse_auc(all_results: dict, output_path: str = None):
    """
    Area-under-MSE-curve (lower = faster convergence overall).
    One bar per teacher × scenario.
    """
    scenarios = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)
    x = np.arange(len(scenarios))
    width = 0.15

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    for i, name in enumerate(teacher_names):
        aucs = []
        for sc in scenarios:
            curve = all_results[sc][name]["mse_curve"]
            aucs.append(float(np.trapezoid(curve)))
        offset = (i - n_t / 2 + 0.5) * width
        ax.bar(
            x + offset, aucs, width,
            label=TEACHER_LABELS.get(name, name),
            color=TEACHER_COLORS.get(name, "grey"),
            alpha=0.87, edgecolor="white",
        )

    _bar_labels(ax, fmt="{:.1f}", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel("Area Under MSE Curve (lower = faster)")
    ax.set_title("Convergence Area Under MSE Curve (AUC)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Unified comparison: baselines + downstream teaching accuracy
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_downstream_orchestration(teaching_results: dict, n_tasks: int = 500):
    """
    Use estimated capabilities from each teacher to run PaperOrchestrator
    downstream on the Varying scenario.
    Compares ALL baselines + all teachers in one figure.
    """
    print("\n" + "=" * 70)
    print("PART 3 — Downstream Orchestration Accuracy (Varying scenario)")
    print("=" * 70)

    agents = create_agents_varying(M)
    np.random.seed(SEED)
    gen = SyntheticDataGenerator(M=M, seed=SEED + 1)
    tasks = gen.generate_task_stream(n_tasks)

    scenario = "Varying"
    results   = teaching_results[scenario]
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    downstream_teach = {}
    for tname in teacher_names:
        estimated = results[tname]["final_estimates"]
        est_agents = [
            Agent(name=agents[k].name,
                  capabilities=estimated[k],
                  costs=agents[k].costs)
            for k in range(len(agents))
        ]
        orch = PaperOrchestrator(est_agents, M)
        for t, task in enumerate(tasks):
            aidx = orch.select_agent(task, t)
            pred = agents[aidx].predict(task)
            orch.update(aidx, task, pred)
        downstream_teach[tname] = orch.get_performance_stats()["overall_accuracy"]

    # Baselines on same task stream
    baseline_accs = {}
    for cls, kwargs, key in [
        (RandomOrchestrator,  {}, "RandomOrchestrator"),
        (GreedyOrchestrator,  {}, "GreedyOrchestrator"),
        (UCB1Orchestrator,    {"c": 1.0}, "UCB1Orchestrator"),
        (PaperOrchestrator,   {}, "PaperOrchestrator"),
        (OracleOrchestrator,  {}, "OracleOrchestrator"),
    ]:
        task_copy = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
        orch = cls(agents, M, **kwargs)
        for t, task in enumerate(task_copy):
            aidx = orch.select_agent(task, t)
            pred = agents[aidx].predict(task)
            orch.update(aidx, task, pred)
        baseline_accs[key] = orch.get_performance_stats()["overall_accuracy"]

    # Oracle from true capabilities
    print(f"\n{'Method':<32} {'Accuracy':>10}")
    print("-" * 45)
    for k, v in baseline_accs.items():
        print(f"  [Baseline] {BASELINE_LABELS[k]:<22} {v:.3f}")
    for k, v in downstream_teach.items():
        print(f"  [Teaching] {TEACHER_LABELS[k]:<22} {v:.3f}")

    return baseline_accs, downstream_teach


def plot_unified_accuracy(baseline_accs: dict, downstream_teach: dict,
                          output_path: str = None):
    """
    Single grouped bar chart: baselines (left group) vs
    teacher-guided orchestrators (right group), Varying scenario.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    all_keys    = (list(baseline_accs.keys()) +
                   ["__sep__"] +
                   list(downstream_teach.keys()))
    all_vals    = (list(baseline_accs.values()) +
                   [None] +
                   list(downstream_teach.values()))
    all_labels  = ([BASELINE_LABELS[k] for k in baseline_accs] +
                   [""] +
                   [TEACHER_LABELS[k] for k in downstream_teach])
    all_colors  = ([BASELINE_COLORS[k] for k in baseline_accs] +
                   ["none"] +
                   [TEACHER_COLORS[k] for k in downstream_teach])

    xs = np.arange(len(all_keys))
    for i, (val, col) in enumerate(zip(all_vals, all_colors)):
        if val is None:
            continue
        bar = ax.bar(xs[i], val, color=col, alpha=0.88, edgecolor="white", width=0.7)

    # numbers at top
    for rect in ax.patches:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    # Divider between baselines and teaching
    sep = len(baseline_accs)
    ax.axvline(sep - 0.2, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.text(sep - 0.2, ax.get_ylim()[1] * 0.96, "  ←Baselines  |  Teaching→",
            ha="center", fontsize=9, color="grey")

    ax.set_xticks(xs)
    ax.set_xticklabels(all_labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Downstream Orchestration Accuracy")
    ax.set_title("Unified Accuracy Comparison — Varying Scenario\n"
                 "(Teacher-guided uses Paper Orchestrator with estimated capabilities)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)

    # Legend: baselines vs teaching
    legend_elems = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=BASELINE_COLORS["OracleOrchestrator"],
               markersize=10, label="Baseline orchestrators"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=TEACHER_COLORS["OmniscientTeacher"],
               markersize=10, label="Teaching-guided"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Prior sensitivity analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_prior_experiments(scenario_name: str = "Varying") -> dict:
    """
    For one scenario (default: Varying), compare all 3 priors across all
    teacher types and return MSE curves + final MSEs.
    """
    print("\n" + "=" * 70)
    print(f"PART 4 — Prior Sensitivity Analysis  (scenario: {scenario_name})")
    print("=" * 70)

    agents = SCENARIOS[scenario_name](M)
    prior_results = {}

    for prior in ALL_PRIORS:
        print(f"\n  Prior: {prior.name}")
        results = run_teaching_experiment(
            agents=agents, M=M, budget=BUDGET,
            teacher_classes=TEACHER_CLASSES, seed=SEED,
            prior=prior,
        )
        prior_results[prior.name] = results
        print_teaching_results(results, target_mse=TARGET_MSE)

    return prior_results


def plot_prior_mse_curves(prior_results: dict, scenario_name: str = "Varying",
                          output_path: str = None):
    """
    3 rows (one per prior) × 1 column, each showing MSE curves for all teachers.
    """
    priors_list = list(prior_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_p = len(priors_list)

    fig, axes = plt.subplots(n_p, 1, figsize=(12, 5 * n_p), sharex=True)
    if n_p == 1:
        axes = [axes]

    for ax, prior_name in zip(axes, priors_list):
        results = prior_results[prior_name]
        for tname in teacher_names:
            if tname not in results:
                continue
            curve = results[tname]["mse_curve"]
            ax.plot(
                range(len(curve)), curve,
                color=TEACHER_COLORS.get(tname, "black"),
                label=TEACHER_LABELS.get(tname, tname),
                linewidth=2,
            )
        ax.axhline(TARGET_MSE, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Target MSE")
        short = PRIOR_LABELS.get(prior_name, prior_name)
        ax.set_title(f"Prior: {short}", fontsize=11, fontweight="bold",
                     color=PRIOR_COLORS.get(prior_name, "black"))
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Evaluation Steps (Budget)")
    fig.suptitle(
        f"Prior Sensitivity — MSE Curves  |  Scenario: {scenario_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_prior_efficiency_bars(prior_results: dict, scenario_name: str = "Varying",
                               output_path: str = None):
    """
    Grouped bar chart: steps-to-target-MSE for each teacher × prior.
    """
    priors_list   = list(prior_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_p = len(priors_list)
    x   = np.arange(len(teacher_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    for i, prior_name in enumerate(priors_list):
        results = prior_results[prior_name]
        steps = []
        for tname in teacher_names:
            curve = results.get(tname, {}).get("mse_curve", [])
            reached = next((t for t, v in enumerate(curve) if v <= TARGET_MSE), None)
            steps.append(reached if reached is not None else BUDGET)
        offset = (i - n_p / 2 + 0.5) * width
        short  = PRIOR_LABELS.get(prior_name, prior_name)
        color  = PRIOR_COLORS.get(prior_name, "grey")
        ax.bar(x + offset, steps, width, label=short, color=color,
               alpha=0.85, edgecolor="white")

    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    ax.axhline(BUDGET, color="grey", linestyle="--", linewidth=1,
               label="Max budget (not reached)")
    ax.set_xticks(x)
    ax.set_xticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_ylabel(f"Steps to MSE < {TARGET_MSE}")
    ax.set_title(
        f"Prior Sensitivity — Efficiency  |  Scenario: {scenario_name}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, BUDGET * 1.18)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_prior_final_mse_comparison(prior_results: dict, scenario_name: str = "Varying",
                                    output_path: str = None):
    """
    Heatmap: rows = teacher, cols = prior.  Cells = final MSE.
    """
    priors_list   = list(prior_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    data = np.array([
        [prior_results[p].get(tn, {}).get("final_mse", np.nan)
         for p in priors_list]
        for tn in teacher_names
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Final MSE")

    short_priors = [PRIOR_LABELS.get(p, p) for p in priors_list]
    ax.set_xticks(range(len(priors_list)))
    ax.set_xticklabels(short_priors, fontsize=10)
    ax.set_yticks(range(len(teacher_names)))
    ax.set_yticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_title(
        f"Prior Sensitivity — Final MSE  |  Scenario: {scenario_name}",
        fontsize=12, fontweight="bold"
    )

    for i in range(len(teacher_names)):
        for j in range(len(priors_list)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=9,
                        color="black" if v < np.nanmax(data) * 0.6 else "white")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    print("\n" + "=" * 70)
    print(" Multi-Agent Orchestration — Unified Experiment Suite")
    print(f" M={M} regions | N={N_TASKS} tasks | Budget={BUDGET} | seed={SEED}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Part 1 — Baselines
    # ------------------------------------------------------------------
    baseline_results = run_baseline_experiments()

    print("\nGenerating baseline figures...")
    plot_baseline_accuracy(
        baseline_results,
        "output/baseline_accuracy.png"
    )
    plot_baseline_learning_curves(
        baseline_results,
        "output/baseline_learning_curves.png"
    )

    # ------------------------------------------------------------------
    # Part 2 — Machine teaching
    # ------------------------------------------------------------------
    teaching_results = run_teaching_experiments()

    print("\nGenerating teaching figures...")
    plot_mse_curves(teaching_results,      "output/teaching_mse_curves.png")
    plot_efficiency_bars(teaching_results, "output/teaching_efficiency.png")
    plot_final_mse_heatmap(teaching_results, "output/teaching_mse_heatmap.png")
    plot_capability_estimates(teaching_results, scenario="Varying",
                              output_path="output/capability_estimates_varying.png")
    plot_convergence_speed(teaching_results, "output/teaching_convergence_speed.png")
    plot_mse_auc(teaching_results, "output/teaching_mse_auc.png")

    # ------------------------------------------------------------------
    # Part 3 — Unified accuracy comparison
    # ------------------------------------------------------------------
    baseline_accs, downstream_teach = evaluate_downstream_orchestration(teaching_results)
    plot_unified_accuracy(
        baseline_accs, downstream_teach,
        "output/unified_accuracy_comparison.png"
    )

    # ------------------------------------------------------------------
    # Part 4 — Prior sensitivity (Varying scenario only)
    # ------------------------------------------------------------------
    prior_results = run_prior_experiments(scenario_name="Varying")

    print("\nGenerating prior-sensitivity figures...")
    plot_prior_mse_curves(prior_results, "Varying",
                          "output/prior_mse_curves.png")
    plot_prior_efficiency_bars(prior_results, "Varying",
                               "output/prior_efficiency_bars.png")
    plot_prior_final_mse_comparison(prior_results, "Varying",
                                    "output/prior_final_mse_heatmap.png")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All experiments complete.  Figures saved to output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
