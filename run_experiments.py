"""
Unified experiment runner for multi-agent orchestration and machine teaching.

Runs five experimental parts and saves all figures to output/:

  Part 1 — Baseline Orchestrators
      Evaluates Random, Greedy, UCB1, Paper (Bhatt et al.), and Oracle
      orchestrators across four expertise scenarios.  Results are averaged
      over N_RUNS independent seeds and reported with ±1σ error bars.

  Part 2 — Machine Teaching
      Compares five teachers (Omniscient, Imitation, Surrogate, RoundRobin,
      Random) on capability estimation: MSE convergence curves, efficiency
      (steps to target MSE), AUC under MSE curve, and convergence speed.

  Part 3 — Unified Comparison
      Uses capability estimates produced by each teacher to initialise a
      PaperOrchestrator and measures downstream task accuracy on the Varying
      scenario.  Also overlays baseline and teaching MSE curves on a shared
      absolute x-axis.

  Part 4 — Prior Sensitivity
      Re-runs machine teaching with three Beta priors (Uniform, Jeffreys,
      Expert) on the Varying scenario.

  Part 5 — Budget vs Downstream Accuracy
      Sweeps teaching budget from 10 to 300 evaluations and plots how
      downstream orchestration accuracy grows with budget for each teacher.

  Part 6 — With vs Without Teacher Variance Comparison  [NEW]
      Direct side-by-side variance comparison for the Varying scenario:
        6a) MSE curves on the same axes — best passive baseline (Paper
            orchestrator) vs. each teacher, ±1σ bands, shared x-axis.
        6b) Violin plots of the final accuracy distribution across 100 runs
            at key budget checkpoints (50, 100, 200 steps), showing how
            teaching narrows the distribution compared to the passive baseline.

  Part 7 — Distribution Comparison
      Compares three capability estimator models on the Varying scenario:
        – Beta-Binomial (reference, conjugate)
        – Truncated-Normal (non-conjugate; quadrature posterior)
        – Logistic-Normal  (non-conjugate; Laplace approximation)
        7a) MSE convergence curves for all teachers × all estimators (±1σ).
        7b) Final-MSE heatmap (estimator × teacher).
        7c) Downstream accuracy bar chart (estimator × teacher).

Implementation notes
--------------------
* All runs use isolated np.random.Generator instances; no global seeding.
* OracleOrchestrator is excluded from the capability MSE panels (it uses
  true capabilities directly and never maintains an estimator).
* np.trapezoid is aliased to np.trapz for numpy <2.0 / >=2.0 compatibility.
* Results across all parts are averaged over N_RUNS=100 independent seeds.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# numpy 2.0 renamed trapz → trapezoid; support both.
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from capability_estimators import (
    BetaBinomialEstimator, TruncatedNormalEstimator, LogisticNormalEstimator,
    ESTIMATOR_CLASSES, ESTIMATOR_COLORS,
)

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
    run_all_baselines,
)
from priors import (
    BetaPrior, JeffreysPrior, SkewedExpertPrior,
    ALL_PRIORS, PRIOR_LABELS, PRIOR_COLORS,
)

os.makedirs("output", exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────

BASELINE_COLORS = {
    "RandomOrchestrator": "#F44336", "GreedyOrchestrator": "#FF9800",
    "UCB1Orchestrator": "#2196F3",   "PaperOrchestrator": "#9C27B0",
    "OracleOrchestrator": "#4CAF50",
}
BASELINE_LABELS = {
    "RandomOrchestrator": "Random",   "GreedyOrchestrator": "Greedy",
    "UCB1Orchestrator":   "UCB1",     "PaperOrchestrator":  "Paper (Bhatt et al.)",
    "OracleOrchestrator": "Oracle",
}
TEACHER_COLORS = {
    "OmniscientTeacher": "#2196F3", "ImitationTeacher": "#4CAF50",
    "SurrogateTeacher":  "#FF9800", "RoundRobinTeacher": "#9C27B0",
    "RandomTeacher":     "#F44336",
}
TEACHER_LABELS = {
    "OmniscientTeacher": "Omniscient (EMSR)", "ImitationTeacher": "Imitation",
    "SurrogateTeacher":  "Surrogate",         "RoundRobinTeacher": "Round-Robin",
    "RandomTeacher":     "Random",
}

TEACHER_CLASSES = [
    (OmniscientTeacher, {}),
    (ImitationTeacher,  {"eta_v": 5.0}),
    (SurrogateTeacher,  {}),
    (RoundRobinTeacher, {}),
    (RandomTeacher,     {}),
]

SCENARIOS = {
    "Approx. Invariant":    create_agents_approximately_invariant,
    "Dominant":             create_agents_dominant,
    "Dominant + Mis. Cost": create_agents_dominant_misaligned_cost,
    "Varying":              create_agents_varying,
}

M          = 3
N_TASKS    = 1000
BUDGET     = 200
SEED       = 42
TARGET_MSE = 0.005
N_RUNS     = 100

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.28, "grid.linestyle": "--",
})
FIGSIZE    = (13, 6)
BAND_ALPHA = 0.15


def _bar_labels(ax, fmt="{:.3f}", fontsize=8, color="black", padding=1):
    for rect in ax.patches:
        h = rect.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, padding), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize, color=color)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE ORCHESTRATORS
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_experiments() -> dict:
    """
    Returns {scenario: {orch_name: {"acc_runs": list[float], "overall_accuracy": float,
                                    "std_accuracy": float}, "__app__": float}}
    Stores per-run accuracy arrays so bar charts can show ±1σ.
    """
    print("\n" + "=" * 70)
    print(f"PART 1 — Baseline Orchestrators ({N_RUNS} runs)")
    print("=" * 70)
    all_results = {}
    for sc_name, agent_factory in SCENARIOS.items():
        agents = agent_factory(M)
        app = compute_appropriateness(agents, M)
        print(f"\n  Scenario: {sc_name}  (App={app:.3f})")
        acc_store: dict = {}  # orch_name -> list of per-run accuracies
        for run in range(N_RUNS):
            if run and run % 25 == 0:
                print(f"    run {run}")
            gen   = SyntheticDataGenerator(M=M, seed=SEED + run)
            tasks = gen.generate_task_stream(N_TASKS)
            res   = run_all_baselines(agents, tasks, verbose=False, seed=SEED + run)
            for k, v in res.items():
                acc_store.setdefault(k, []).append(v["overall_accuracy"])

        avg = {}
        for k, runs in acc_store.items():
            arr = np.array(runs)
            avg[k] = {
                "acc_runs":       arr,
                "overall_accuracy": float(arr.mean()),
                "std_accuracy":     float(arr.std()),
            }
        avg["__app__"] = app
        all_results[sc_name] = avg
        # Print table
        for k, v in avg.items():
            if k == "__app__":
                continue
            print(f"    {k:<28} {v['overall_accuracy']:.3f} ± {v['std_accuracy']:.3f}")
    return all_results


def plot_baseline_accuracy(all_results: dict, output_path: str = None):
    """Grouped bar chart with ±1σ error bars (Fig 1a)."""
    scenarios  = list(all_results.keys())
    orch_names = [k for k in next(iter(all_results.values())) if k != "__app__"]
    x = np.arange(len(scenarios))
    n = len(orch_names)
    w = 0.14
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(orch_names):
        accs = [all_results[sc][name]["overall_accuracy"] for sc in scenarios]
        errs = [all_results[sc][name]["std_accuracy"]     for sc in scenarios]
        ax.bar(x + (i - n / 2 + 0.5) * w, accs, w,
               label=BASELINE_LABELS.get(name, name),
               color=BASELINE_COLORS.get(name, "grey"),
               alpha=0.88, edgecolor="white",
               yerr=errs, capsize=3, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10.5)
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Baseline Orchestrators — Accuracy by Scenario (±1σ, 100 runs)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.12)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def compute_baseline_curves(window: int = 50) -> dict:
    """
    Returns {scenario: {orch_name: {"acc_runs": (N_RUNS, T-window),
                                    "mse_runs": (N_RUNS, T)}}}

    OracleOrchestrator is excluded from mse_runs because it does not maintain
    a capability estimator (its MSE would be trivially 0 throughout).
    """
    orch_defs = [
        (RandomOrchestrator,  {},           "RandomOrchestrator"),
        (GreedyOrchestrator,  {},           "GreedyOrchestrator"),
        (UCB1Orchestrator,    {"c": 1.0},   "UCB1Orchestrator"),
        (PaperOrchestrator,   {},           "PaperOrchestrator"),
        (OracleOrchestrator,  {},           "OracleOrchestrator"),
    ]
    # Orchestrators included in capability MSE tracking (Oracle excluded).
    mse_keys = {"RandomOrchestrator", "GreedyOrchestrator",
                 "UCB1Orchestrator",  "PaperOrchestrator"}

    all_curves = {}
    for sc_name in SCENARIOS:
        agents    = SCENARIOS[sc_name](M)
        true_caps = np.array([[agents[k].get_capability(m) for m in range(M)]
                               for k in range(len(agents))])
        print(f"  Baseline curves: {sc_name}")
        sc = {}
        for cls, kwargs, key in orch_defs:
            acc_runs, mse_runs = [], []
            for run in range(N_RUNS):
                if run and run % 25 == 0:
                    print(f"    {key} run {run}")
                gen   = SyntheticDataGenerator(M=M, seed=SEED + run)
                tasks = gen.generate_task_stream(N_TASKS)
                tc    = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]

                # Per-run isolated RNGs — fully reproducible.
                orch_rng = np.random.default_rng(SEED + run)
                pred_rng = np.random.default_rng(SEED + run + 31337)
                orch = cls(agents, M, rng=orch_rng, **kwargs)

                acc_r, mse_r = [], []
                for t, task in enumerate(tc):
                    aidx = orch.select_agent(task, t)
                    pred = agents[aidx].predict(task, rng=pred_rng)
                    orch.update(aidx, task, pred)
                    if t >= window:
                        acc_r.append(np.mean(orch.correctness[-window:]))
                    if key in mse_keys:
                        est = orch.capability_estimator.get_all_posterior_means()
                        mse_r.append(float(np.mean((est - true_caps) ** 2)))

                acc_runs.append(acc_r)
                if key in mse_keys:
                    mse_runs.append(mse_r)

            sc[key] = {
                "acc_runs": np.array(acc_runs),                     # (N_RUNS, T-window)
                "mse_runs": np.array(mse_runs) if mse_runs else None,  # None for Oracle
            }
        all_curves[sc_name] = sc
    return all_curves


def plot_baseline_learning_curves(curve_data: dict, output_path: str = None,
                                   window: int = 50):
    """4-panel rolling accuracy with ±1σ bands (Fig 1b)."""
    scenarios = list(SCENARIOS.keys())
    orch_keys = ["RandomOrchestrator", "GreedyOrchestrator",
                 "UCB1Orchestrator",   "PaperOrchestrator", "OracleOrchestrator"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, sc in zip(axes, scenarios):
        for key in orch_keys:
            d    = curve_data[sc][key]
            arr  = d["acc_runs"]           # (N_RUNS, L)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(window, window + len(mean))
            ax.plot(xs, mean, color=BASELINE_COLORS[key],
                    label=BASELINE_LABELS[key], linewidth=2)
            ax.fill_between(xs, mean - std, mean + std,
                            color=BASELINE_COLORS[key], alpha=BAND_ALPHA)
        ax.set_title(sc, fontsize=11, fontweight="bold")
        ax.set_xlabel("Task Index")
        ax.set_ylabel(f"Rolling Accuracy (w={window})")
        ax.legend(fontsize=7)
    fig.suptitle("Baseline Orchestrators — Learning Curves (±1σ, 100 runs)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_baseline_capability_mse(baseline_curves: dict, output_path: str = None):
    """
    4-panel capability MSE with ±1σ bands (Fig 1c).
    OracleOrchestrator is excluded — it does not maintain a capability estimator.
    """
    scenarios = list(SCENARIOS.keys())
    # Only include orchestrators that have MSE data (Oracle excluded).
    orch_keys = ["RandomOrchestrator", "GreedyOrchestrator",
                 "UCB1Orchestrator",   "PaperOrchestrator"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, sc in zip(axes, scenarios):
        for key in orch_keys:
            d = baseline_curves[sc][key]
            if d["mse_runs"] is None:
                continue
            arr  = d["mse_runs"]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(len(mean))
            ax.plot(xs, mean, color=BASELINE_COLORS[key],
                    label=BASELINE_LABELS[key], linewidth=2)
            ax.fill_between(xs, mean - std, mean + std,
                            color=BASELINE_COLORS[key], alpha=BAND_ALPHA)
        ax.axhline(TARGET_MSE, color="grey", ls="--", lw=1.2,
                   label=f"Target={TARGET_MSE}")
        ax.set_title(sc, fontsize=11, fontweight="bold")
        ax.set_xlabel("Task Index")
        ax.set_ylabel("Capability MSE")
        ax.legend(fontsize=7)
    fig.suptitle("Baseline Orchestrators — Capability MSE (±1σ, 100 runs)\n"
                 "(Oracle excluded — it uses true capabilities directly)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. MACHINE TEACHING
# ══════════════════════════════════════════════════════════════════════════════

def run_teaching_experiments() -> dict:
    """Run all teachers × all scenarios, storing per-run MSE curves."""
    print("\n" + "=" * 70)
    print(f"PART 2 — Machine Teaching ({N_RUNS} runs)")
    print("=" * 70)
    all_results = {}
    for sc_name, agent_factory in SCENARIOS.items():
        agents = agent_factory(M)
        print(f"\n  Scenario: {sc_name}")
        sc_data = {}
        for run in range(N_RUNS):
            if run and run % 25 == 0:
                print(f"    run {run}")
            res = run_teaching_experiment(
                agents=agents, M=M, budget=BUDGET,
                teacher_classes=TEACHER_CLASSES, seed=SEED + run,
            )
            for name, summary in res.items():
                if name not in sc_data:
                    sc_data[name] = {
                        "mse_runs":       [],
                        "final_mse_runs": [],
                        "final_est_runs": [],
                        "true_caps":      summary["true_caps"],
                    }
                sc_data[name]["mse_runs"].append(summary["mse_curve"])
                sc_data[name]["final_mse_runs"].append(summary["final_mse"])
                sc_data[name]["final_est_runs"].append(
                    summary["final_estimates"].copy()
                )

        for name in sc_data:
            d = sc_data[name]
            d["mse_runs"]       = np.array(d["mse_runs"])          # (N_RUNS, BUDGET)
            d["mse_curve"]      = d["mse_runs"].mean(axis=0).tolist()
            d["final_mse"]      = float(np.mean(d["final_mse_runs"]))
            d["final_estimates"] = np.mean(d["final_est_runs"], axis=0)

        all_results[sc_name] = sc_data
        avg_dict = {
            n: {"final_mse":       sc_data[n]["final_mse"],
                "mse_curve":       sc_data[n]["mse_curve"],
                "final_estimates": sc_data[n]["final_estimates"],
                "true_caps":       sc_data[n]["true_caps"]}
            for n in sc_data
        }
        print_teaching_results(avg_dict, target_mse=TARGET_MSE)

    return all_results


def plot_mse_curves(all_results: dict, output_path: str = None):
    """4-panel MSE curves with ±1σ bands (Fig 2a)."""
    scenarios = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, sc in zip(axes, scenarios):
        for name in all_results[sc]:
            d    = all_results[sc][name]
            arr  = d["mse_runs"]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(len(mean))
            ax.plot(xs, mean, color=TEACHER_COLORS.get(name, "black"),
                    label=TEACHER_LABELS.get(name, name), linewidth=2)
            ax.fill_between(xs, mean - std, mean + std,
                            color=TEACHER_COLORS.get(name, "black"),
                            alpha=BAND_ALPHA)
        ax.axhline(TARGET_MSE, color="grey", ls="--", lw=1.2,
                   label=f"Target={TARGET_MSE}")
        ax.set_title(sc, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Steps")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=7)
    fig.suptitle("Machine Teaching — MSE Curves (±1σ, 100 runs)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_efficiency_bars(all_results: dict, output_path: str = None):
    scenarios     = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)
    x   = np.arange(len(scenarios))
    w   = 0.15
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, name in enumerate(teacher_names):
        steps = []
        for sc in scenarios:
            curve = all_results[sc][name]["mse_curve"]
            eff   = next((t for t, v in enumerate(curve) if v <= TARGET_MSE), None)
            steps.append(eff if eff is not None else BUDGET)
        ax.bar(x + (i - n_t / 2 + 0.5) * w, steps, w,
               label=TEACHER_LABELS.get(name, name),
               color=TEACHER_COLORS.get(name, "grey"), alpha=0.87,
               edgecolor="white")
    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    ax.axhline(BUDGET, color="grey", ls="--", lw=1, label="Budget limit")
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
    scenarios     = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    data = np.array([[all_results[sc][tn]["final_mse"] for tn in teacher_names]
                      for sc in scenarios])
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Final MSE")
    ax.set_xticks(range(len(teacher_names)))
    ax.set_xticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=10)
    ax.set_title(f"Final MSE after {BUDGET} Steps", fontsize=12, fontweight="bold")
    for i in range(len(scenarios)):
        for j in range(len(teacher_names)):
            v = data[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9,
                    color="black" if v < data.max() * 0.6 else "white")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_capability_estimates(all_results: dict, scenario: str = "Varying",
                               output_path: str = None):
    results = all_results.get(scenario)
    if not results:
        return
    true_caps     = results["OmniscientTeacher"]["true_caps"]
    K, Ml         = true_caps.shape
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t           = len(teacher_names)
    fig, axes     = plt.subplots(K, Ml, figsize=(4.5 * Ml, 3.5 * K))
    for k in range(K):
        for m in range(Ml):
            ax    = axes[k, m]
            truth = true_caps[k, m]
            ests  = [results[tn]["final_estimates"][k, m] for tn in teacher_names]
            cols  = [TEACHER_COLORS.get(n, "grey") for n in teacher_names]
            lbls  = [TEACHER_LABELS.get(n, n) for n in teacher_names]
            ax.scatter(range(n_t), ests, color=cols, s=80, zorder=3)
            ax.axhline(truth, color="black", ls="--", lw=1.5,
                       label=f"True={truth:.3f}")
            ax.set_xticks(range(n_t))
            ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("P(correct)")
            ax.set_title(f"Agent {k+1} | Region {m}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f'Capability Estimates — "{scenario}" — {BUDGET} steps',
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_convergence_speed(all_results: dict, output_path: str = None):
    sc            = "Varying"
    results       = all_results.get(sc, {})
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    thresholds    = [TARGET_MSE * 2, TARGET_MSE, TARGET_MSE * 0.5]
    labels        = ["2× target", "1× target", "½× target"]
    colors        = ["#90CAF9", "#2196F3", "#0D47A1"]
    x = np.arange(len(teacher_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (thr, lbl, col) in enumerate(zip(thresholds, labels, colors)):
        steps = []
        for name in teacher_names:
            curve   = results.get(name, {}).get("mse_curve", [])
            reached = next((t for t, v in enumerate(curve) if v <= thr), None)
            steps.append(reached if reached is not None else BUDGET)
        ax.bar(x + (i - 1) * w, steps, w, label=lbl, color=col,
               alpha=0.88, edgecolor="white")
    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_ylabel("Steps until threshold")
    ax.set_ylim(0, BUDGET * 1.18)
    ax.set_title(f'Convergence Speed — "{sc}"', fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_mse_auc(all_results: dict, output_path: str = None):
    scenarios     = list(all_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_t = len(teacher_names)
    x   = np.arange(len(scenarios))
    w   = 0.15
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, name in enumerate(teacher_names):
        aucs = [float(_trapz(all_results[sc][name]["mse_curve"]))
                for sc in scenarios]
        ax.bar(x + (i - n_t / 2 + 0.5) * w, aucs, w,
               label=TEACHER_LABELS.get(name, name),
               color=TEACHER_COLORS.get(name, "grey"), alpha=0.87,
               edgecolor="white")
    _bar_labels(ax, fmt="{:.1f}", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel("AUC (lower = faster)")
    ax.legend(fontsize=9)
    ax.set_title("Area Under MSE Curve", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNIFIED COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_downstream_orchestration(n_tasks: int = 500):
    """Per-run downstream eval for the Varying scenario."""
    print("\n" + "=" * 70)
    print(f"PART 3 — Downstream Orchestration ({N_RUNS} runs, Varying)")
    print("=" * 70)
    agents        = create_agents_varying(M)
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    dt_avg        = {t: 0.0 for t in teacher_names}
    baseline_defs = [
        (RandomOrchestrator, {},         "RandomOrchestrator"),
        (GreedyOrchestrator, {},         "GreedyOrchestrator"),
        (UCB1Orchestrator,   {"c": 1.0}, "UCB1Orchestrator"),
        (PaperOrchestrator,  {},         "PaperOrchestrator"),
        (OracleOrchestrator, {},         "OracleOrchestrator"),
    ]
    bl_avg = {k: 0.0 for _, _, k in baseline_defs}

    for run in range(N_RUNS):
        if run and run % 25 == 0:
            print(f"  run {run}")
        tr = run_teaching_experiment(
            agents=agents, M=M, budget=BUDGET,
            teacher_classes=TEACHER_CLASSES, seed=SEED + run,
        )
        gen   = SyntheticDataGenerator(M=M, seed=SEED + run + 1000)
        tasks = gen.generate_task_stream(n_tasks)

        for tname in teacher_names:
            est  = tr[tname]["final_estimates"]
            tc   = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
            pred_rng = np.random.default_rng(SEED + run + 77777)
            orch = PaperOrchestrator(agents, M,
                                     rng=np.random.default_rng(SEED + run + 88888))
            orch.capability_estimator.inject_estimates(est, n_virtual=50)
            for t, task in enumerate(tc):
                aidx = orch.select_agent(task, t)
                pred = agents[aidx].predict(task, rng=pred_rng)
                orch.update(aidx, task, pred)
            dt_avg[tname] += orch.get_performance_stats()["overall_accuracy"]

        for i, (cls, kwargs, key) in enumerate(baseline_defs):
            tc   = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
            pred_rng = np.random.default_rng(SEED + run + i * 1000 + 55555)
            orch = cls(agents, M,
                       rng=np.random.default_rng(SEED + run + i * 1000), **kwargs)
            for t, task in enumerate(tc):
                aidx = orch.select_agent(task, t)
                pred = agents[aidx].predict(task, rng=pred_rng)
                orch.update(aidx, task, pred)
            bl_avg[key] += orch.get_performance_stats()["overall_accuracy"]

    for k in dt_avg:
        dt_avg[k] /= N_RUNS
    for k in bl_avg:
        bl_avg[k] /= N_RUNS

    print(f"\n{'Method':<35} {'Accuracy':>8}")
    print("-" * 45)
    for k, v in bl_avg.items():
        print(f"  [Baseline] {BASELINE_LABELS[k]:<22} {v:.3f}")
    for k, v in dt_avg.items():
        print(f"  [Teaching] {TEACHER_LABELS[k]:<22} {v:.3f}")
    return bl_avg, dt_avg


def plot_unified_accuracy(bl: dict, dt: dict, output_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 6))
    keys = list(bl.keys()) + ["__sep__"] + list(dt.keys())
    vals = list(bl.values()) + [None] + list(dt.values())
    lbls = ([BASELINE_LABELS[k] for k in bl]
            + [""]
            + [TEACHER_LABELS[k] for k in dt])
    cols = ([BASELINE_COLORS[k] for k in bl]
            + ["none"]
            + [TEACHER_COLORS[k] for k in dt])
    xs = np.arange(len(keys))
    for i, (v, c) in enumerate(zip(vals, cols)):
        if v is not None:
            ax.bar(xs[i], v, color=c, alpha=0.88, edgecolor="white", width=0.7)
    for r in ax.patches:
        h = r.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}",
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)
    sep = len(bl)
    ax.axvline(sep - 0.2, color="black", ls=":", lw=1.5, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Downstream Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Unified Accuracy — Varying Scenario", fontsize=12, fontweight="bold")
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
               markersize=10, label="Baselines"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2196F3",
               markersize=10, label="Teaching-guided"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_convergence_mse_absolute(baseline_curves: dict, teaching_results: dict,
                                   output_path: str = None):
    """
    4-panel: capability MSE on an ABSOLUTE x-axis (evaluation steps).

    Baselines run 0→N_TASKS; teaching runs 0→BUDGET.  The efficiency gap is
    visually apparent because teaching curves stop at step BUDGET.

    OracleOrchestrator is excluded (no capability MSE data).
    """
    scenarios  = list(SCENARIOS.keys())
    orch_keys  = ["RandomOrchestrator", "GreedyOrchestrator",
                  "UCB1Orchestrator",   "PaperOrchestrator"]
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, sc in zip(axes, scenarios):
        # Baselines (solid lines, 0 → N_TASKS)
        for key in orch_keys:
            d    = baseline_curves[sc][key]
            if d["mse_runs"] is None:
                continue
            arr  = d["mse_runs"]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(len(mean))
            ax.plot(xs, mean, color=BASELINE_COLORS[key],
                    label=BASELINE_LABELS[key], lw=2, ls="-")
            ax.fill_between(xs, mean - std, mean + std,
                            color=BASELINE_COLORS[key], alpha=BAND_ALPHA)
        # Teaching (dashed lines, 0 → BUDGET)
        for tname in teacher_names:
            td = teaching_results.get(sc, {}).get(tname)
            if td is None:
                continue
            arr  = td["mse_runs"]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(len(mean))
            ax.plot(xs, mean, color=TEACHER_COLORS[tname],
                    label=TEACHER_LABELS[tname], lw=2, ls="--")
            ax.fill_between(xs, mean - std, mean + std,
                            color=TEACHER_COLORS[tname], alpha=BAND_ALPHA)
        ax.axhline(TARGET_MSE, color="grey", ls=":", lw=1.2,
                   label=f"Target={TARGET_MSE}")
        ax.axvline(BUDGET, color="grey", ls="-.", lw=1, alpha=0.5,
                   label=f"Budget={BUDGET}")
        ax.set_title(sc, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Steps (absolute)")
        ax.set_ylabel("Capability MSE")
        ax.legend(fontsize=5.5, ncol=2)
    style_legend = [
        Line2D([0], [0], color="black", lw=2, ls="-",
               label="Baselines (over full task stream)"),
        Line2D([0], [0], color="black", lw=2, ls="--",
               label="Teaching (over evaluation budget)"),
    ]
    fig.legend(handles=style_legend, fontsize=9, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Convergence MSE — Absolute Axis (±1σ, 100 runs)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRIOR SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def run_prior_experiments(scenario_name: str = "Varying") -> dict:
    print("\n" + "=" * 70)
    print(f"PART 4 — Prior Sensitivity ({scenario_name}, {N_RUNS} runs)")
    print("=" * 70)
    agents = SCENARIOS[scenario_name](M)
    prior_results = {}
    for prior in ALL_PRIORS:
        print(f"\n  Prior: {prior.name}")
        pr_data = {}
        for run in range(N_RUNS):
            if run and run % 25 == 0:
                print(f"    run {run}")
            res = run_teaching_experiment(
                agents=agents, M=M, budget=BUDGET,
                teacher_classes=TEACHER_CLASSES, seed=SEED + run, prior=prior,
            )
            for name, summary in res.items():
                if name not in pr_data:
                    pr_data[name] = {
                        "mse_runs": [], "final_mse_runs": [],
                        "final_est_runs": [], "true_caps": summary["true_caps"],
                    }
                pr_data[name]["mse_runs"].append(summary["mse_curve"])
                pr_data[name]["final_mse_runs"].append(summary["final_mse"])
                pr_data[name]["final_est_runs"].append(
                    summary["final_estimates"].copy()
                )
        for name in pr_data:
            d = pr_data[name]
            d["mse_runs"]       = np.array(d["mse_runs"])
            d["mse_curve"]      = d["mse_runs"].mean(axis=0).tolist()
            d["final_mse"]      = float(np.mean(d["final_mse_runs"]))
            d["final_estimates"] = np.mean(d["final_est_runs"], axis=0)
        prior_results[prior.name] = pr_data
    return prior_results


def plot_prior_mse_curves(prior_results: dict, scenario_name: str = "Varying",
                           output_path: str = None):
    """3 rows × 1 col, one panel per prior (Fig 4a)."""
    priors_list   = list(prior_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_p = len(priors_list)
    fig, axes = plt.subplots(n_p, 1, figsize=(12, 5 * n_p), sharex=True)
    if n_p == 1:
        axes = [axes]
    for ax, pn in zip(axes, priors_list):
        for tname in teacher_names:
            d = prior_results[pn].get(tname)
            if d is None:
                continue
            arr  = d["mse_runs"]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(len(mean))
            ax.plot(xs, mean, color=TEACHER_COLORS.get(tname, "black"),
                    label=TEACHER_LABELS.get(tname, tname), lw=2)
            ax.fill_between(xs, mean - std, mean + std,
                            color=TEACHER_COLORS.get(tname, "black"),
                            alpha=BAND_ALPHA)
        ax.axhline(TARGET_MSE, color="grey", ls="--", lw=1.2, label="Target MSE")
        short = PRIOR_LABELS.get(pn, pn)
        ax.set_title(f"Prior: {short}", fontsize=11, fontweight="bold",
                     color=PRIOR_COLORS.get(pn, "black"))
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Evaluation Steps")
    fig.suptitle(f"Prior Sensitivity — MSE Curves (±1σ)  |  {scenario_name}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_prior_efficiency_bars(prior_results: dict, scenario_name: str = "Varying",
                                output_path: str = None):
    priors_list   = list(prior_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    n_p = len(priors_list)
    x   = np.arange(len(teacher_names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, pn in enumerate(priors_list):
        steps = []
        for tn in teacher_names:
            curve = prior_results[pn].get(tn, {}).get("mse_curve", [])
            r = next((t for t, v in enumerate(curve) if v <= TARGET_MSE), None)
            steps.append(r if r is not None else BUDGET)
        ax.bar(x + (i - n_p / 2 + 0.5) * w, steps, w,
               label=PRIOR_LABELS.get(pn, pn),
               color=PRIOR_COLORS.get(pn, "grey"), alpha=0.85, edgecolor="white")
    _bar_labels(ax, fmt="{:.0f}", fontsize=7.5)
    ax.axhline(BUDGET, color="grey", ls="--", lw=1, label="Budget limit")
    ax.set_xticks(x)
    ax.set_xticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_ylabel(f"Steps to MSE < {TARGET_MSE}")
    ax.set_ylim(0, BUDGET * 1.18)
    ax.set_title(f"Prior Sensitivity — Efficiency  |  {scenario_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_prior_final_mse_heatmap(prior_results: dict, scenario_name: str = "Varying",
                                  output_path: str = None):
    priors_list   = list(prior_results.keys())
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    data = np.array([[prior_results[p].get(tn, {}).get("final_mse", np.nan)
                       for p in priors_list]
                      for tn in teacher_names])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Final MSE")
    ax.set_xticks(range(len(priors_list)))
    ax.set_xticklabels([PRIOR_LABELS.get(p, p) for p in priors_list], fontsize=10)
    ax.set_yticks(range(len(teacher_names)))
    ax.set_yticklabels([TEACHER_LABELS.get(n, n) for n in teacher_names], fontsize=10)
    ax.set_title(f"Prior Sensitivity — Final MSE  |  {scenario_name}",
                 fontsize=12, fontweight="bold")
    for i in range(len(teacher_names)):
        for j in range(len(priors_list)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9,
                        color="black" if v < np.nanmax(data) * 0.6 else "white")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUDGET vs DOWNSTREAM ACCURACY TRADE-OFF
# ══════════════════════════════════════════════════════════════════════════════

def run_budget_tradeoff(budget_values=None, n_downstream: int = 500) -> dict:
    """
    For each teaching budget, run each teacher on Varying then evaluate
    downstream accuracy with PaperOrchestrator.

    Returns {teacher_name: {"budgets": [...], "acc_mean": [...], "acc_std": [...]}}
    """
    if budget_values is None:
        budget_values = [10, 25, 50, 75, 100, 150, 200, 300]

    print("\n" + "=" * 70)
    print(f"PART 5 — Budget vs Downstream Accuracy ({N_RUNS} runs, Varying)")
    print("=" * 70)

    agents        = create_agents_varying(M)
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    results = {tn: {"budgets": budget_values,
                    "acc_runs": {b: [] for b in budget_values}}
               for tn in teacher_names}

    for b in budget_values:
        print(f"\n  Budget = {b}")
        for run in range(N_RUNS):
            if run and run % 25 == 0:
                print(f"    run {run}")
            tr  = run_teaching_experiment(
                agents=agents, M=M, budget=b,
                teacher_classes=TEACHER_CLASSES, seed=SEED + run,
            )
            gen   = SyntheticDataGenerator(M=M, seed=SEED + run + 5000)
            tasks = gen.generate_task_stream(n_downstream)

            for tname in teacher_names:
                est      = tr[tname]["final_estimates"]
                tc       = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
                pred_rng = np.random.default_rng(SEED + run + 66666)
                orch     = PaperOrchestrator(
                    agents, M, rng=np.random.default_rng(SEED + run + 44444))
                orch.capability_estimator.inject_estimates(est, n_virtual=50)
                for t, task in enumerate(tc):
                    aidx = orch.select_agent(task, t)
                    pred = agents[aidx].predict(task, rng=pred_rng)
                    orch.update(aidx, task, pred)
                results[tname]["acc_runs"][b].append(
                    orch.get_performance_stats()["overall_accuracy"]
                )

    # Compute mean ± std
    for tname in teacher_names:
        means, stds = [], []
        for b in budget_values:
            arr = np.array(results[tname]["acc_runs"][b])
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
        results[tname]["acc_mean"] = means
        results[tname]["acc_std"]  = stds

    # Print table
    print(f"\n{'Teacher':<25}", end="")
    for b in budget_values:
        print(f"  B={b:>3}", end="")
    print()
    print("-" * (25 + 7 * len(budget_values)))
    for tn in teacher_names:
        print(f"{TEACHER_LABELS[tn]:<25}", end="")
        for m in results[tn]["acc_mean"]:
            print(f"  {m:.3f}", end="")
        print()

    return results


def plot_budget_tradeoff(tradeoff_results: dict, output_path: str = None):
    """
    Line plot: x = teaching budget, y = downstream accuracy, ±1σ bands.
    Oracle and Random horizontal lines for reference.
    """
    fig, ax       = plt.subplots(figsize=FIGSIZE)
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    for tname in teacher_names:
        d       = tradeoff_results[tname]
        budgets = d["budgets"]
        means   = np.array(d["acc_mean"])
        stds    = np.array(d["acc_std"])
        ax.plot(budgets, means, color=TEACHER_COLORS[tname],
                label=TEACHER_LABELS[tname], linewidth=2.5,
                marker="o", markersize=5)
        ax.fill_between(budgets, means - stds, means + stds,
                        color=TEACHER_COLORS[tname], alpha=BAND_ALPHA)

    # Oracle & Random baselines (budget-independent)
    agents       = create_agents_varying(M)
    oracle_accs, random_accs = [], []
    for run in range(N_RUNS):
        gen   = SyntheticDataGenerator(M=M, seed=SEED + run + 5000)
        tasks = gen.generate_task_stream(500)
        for cls, acc_list in [(OracleOrchestrator, oracle_accs),
                               (RandomOrchestrator, random_accs)]:
            tc       = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
            pred_rng = np.random.default_rng(SEED + run + 99999)
            orch     = cls(agents, M, rng=np.random.default_rng(SEED + run))
            for t, task in enumerate(tc):
                aidx = orch.select_agent(task, t)
                pred = agents[aidx].predict(task, rng=pred_rng)
                orch.update(aidx, task, pred)
            acc_list.append(orch.get_performance_stats()["overall_accuracy"])

    ax.axhline(np.mean(oracle_accs), color="#4CAF50", ls="--", lw=2,
               label=f"Oracle ({np.mean(oracle_accs):.3f})")
    ax.axhline(np.mean(random_accs), color="#F44336", ls="--", lw=2,
               label=f"Random ({np.mean(random_accs):.3f})")

    ax.set_xlabel("Teaching Budget (number of evaluations)", fontsize=12)
    ax.set_ylabel("Downstream Orchestration Accuracy", fontsize=12)
    ax.set_title("Budget vs Downstream Accuracy — Varying Scenario (±1σ)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_ylim(0.2, 0.95)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. WITH vs WITHOUT TEACHER — VARIANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def run_with_without_teacher_comparison(
    baseline_curves: dict,
    teaching_results: dict,
    scenario: str = "Varying",
) -> dict:
    """
    Collect per-run data needed for the with/without teacher variance plots.

    Returns a dict with:
      "baseline_mse"  : (N_RUNS, N_TASKS) — PaperOrchestrator MSE per task
      "teacher_mse"   : {teacher_name: (N_RUNS, BUDGET)} — teacher MSE per step
      "baseline_acc"  : (N_RUNS,) — PaperOrchestrator final accuracy per run
      "teacher_acc"   : {teacher_name: {budget: [acc per run]}}  — downstream acc
    All already collected; we just extract and re-package from existing data.
    """
    print(f"\n  Packaging with/without comparison data for '{scenario}'...")

    out = {
        "baseline_mse":   baseline_curves[scenario]["PaperOrchestrator"]["mse_runs"],
        "teacher_mse":    {name: teaching_results[scenario][name]["mse_runs"]
                           for name in teaching_results[scenario]},
        "scenario":       scenario,
    }
    return out


def plot_variance_comparison_curves(cmp_data: dict, output_path: str = None):
    """
    Fig 6a — MSE curves on the SAME axes: passive baseline vs each teacher.

    One panel per teacher.  Both share the same absolute x-axis so the
    efficiency gap is immediately visible.  ±1σ bands across 100 runs.
    """
    teacher_names  = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    baseline_arr   = cmp_data["baseline_mse"]          # (N_RUNS, N_TASKS)
    bl_mean        = baseline_arr.mean(axis=0)
    bl_std         = baseline_arr.std(axis=0)
    bl_xs          = np.arange(len(bl_mean))

    n_teachers = len(teacher_names)
    ncols = 3
    nrows = (n_teachers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True)
    axes = np.array(axes).flatten()

    for ax, tname in zip(axes, teacher_names):
        # Passive baseline (solid grey)
        ax.plot(bl_xs, bl_mean, color="#78909C", lw=2, ls="-",
                label="Passive (Paper orch.)", zorder=2)
        ax.fill_between(bl_xs, bl_mean - bl_std, bl_mean + bl_std,
                        color="#78909C", alpha=0.18, zorder=1)

        # Teacher (coloured dashed, shorter x-range)
        t_arr  = cmp_data["teacher_mse"][tname]        # (N_RUNS, BUDGET)
        t_mean = t_arr.mean(axis=0)
        t_std  = t_arr.std(axis=0)
        t_xs   = np.arange(len(t_mean))
        ax.plot(t_xs, t_mean, color=TEACHER_COLORS[tname], lw=2.5, ls="--",
                label=TEACHER_LABELS[tname], zorder=3)
        ax.fill_between(t_xs, t_mean - t_std, t_mean + t_std,
                        color=TEACHER_COLORS[tname], alpha=BAND_ALPHA, zorder=2)

        ax.axhline(TARGET_MSE, color="grey", ls=":", lw=1.2,
                   label=f"Target={TARGET_MSE}")
        ax.axvline(BUDGET, color="grey", ls="-.", lw=1, alpha=0.45)
        ax.set_title(TEACHER_LABELS[tname], fontsize=11, fontweight="bold",
                     color=TEACHER_COLORS[tname])
        ax.set_xlabel("Evaluation Steps")
        ax.set_ylabel("Capability MSE")
        ax.legend(fontsize=7.5)

    # Hide unused axes
    for ax in axes[len(teacher_names):]:
        ax.set_visible(False)

    style_legend = [
        Line2D([0], [0], color="#78909C", lw=2, ls="-",
               label="Without teacher (passive orchestration)"),
        Line2D([0], [0], color="black",   lw=2, ls="--",
               label="With teacher (active teaching)"),
    ]
    fig.legend(handles=style_legend, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "With vs Without Teacher — MSE Convergence (±1σ, 100 runs)\n"
        f"Scenario: {cmp_data['scenario']}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def run_violin_data(
    scenario: str = "Varying",
    checkpoint_budgets: tuple = (50, 100, 200),
    n_downstream: int = 500,
) -> dict:
    """
    Collect final-accuracy distributions at each checkpoint budget for violin
    plots.  Runs N_RUNS independent teaching + evaluation cycles.

    Returns {
      "checkpoints": [...],
      "baseline_accs": [acc_run_0, ..., acc_run_{N_RUNS-1}]  (budget-independent),
      "teacher_accs":  {teacher_name: {budget: [acc_run_0, ...]}}
    }
    """
    print(f"\n  Collecting violin data (checkpoints={checkpoint_budgets}) ...")
    agents        = SCENARIOS[scenario](M)
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]

    baseline_accs = []
    teacher_accs  = {tn: {b: [] for b in checkpoint_budgets}
                     for tn in teacher_names}

    for run in range(N_RUNS):
        if run and run % 25 == 0:
            print(f"    run {run}")

        gen   = SyntheticDataGenerator(M=M, seed=SEED + run + 9000)
        tasks = gen.generate_task_stream(n_downstream)

        # Passive baseline — PaperOrchestrator with no pre-seeding
        tc       = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
        pred_rng = np.random.default_rng(SEED + run + 10000)
        orch     = PaperOrchestrator(agents, M,
                                     rng=np.random.default_rng(SEED + run + 11000))
        for t, task in enumerate(tc):
            aidx = orch.select_agent(task, t)
            pred = agents[aidx].predict(task, rng=pred_rng)
            orch.update(aidx, task, pred)
        baseline_accs.append(orch.get_performance_stats()["overall_accuracy"])

        # Teachers at each checkpoint budget
        for budget in checkpoint_budgets:
            tr = run_teaching_experiment(
                agents=agents, M=M, budget=budget,
                teacher_classes=TEACHER_CLASSES, seed=SEED + run,
            )
            for tname in teacher_names:
                est      = tr[tname]["final_estimates"]
                tc2      = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
                pred_rng2 = np.random.default_rng(SEED + run + budget * 100 + 20000)
                orch2     = PaperOrchestrator(
                    agents, M, rng=np.random.default_rng(SEED + run + budget * 100 + 30000))
                orch2.capability_estimator.inject_estimates(est, n_virtual=50)
                for t, task in enumerate(tc2):
                    aidx = orch2.select_agent(task, t)
                    pred = agents[aidx].predict(task, rng=pred_rng2)
                    orch2.update(aidx, task, pred)
                teacher_accs[tname][budget].append(
                    orch2.get_performance_stats()["overall_accuracy"]
                )

    return {
        "checkpoints":    list(checkpoint_budgets),
        "baseline_accs":  baseline_accs,
        "teacher_accs":   teacher_accs,
        "scenario":       scenario,
    }


def plot_variance_violin(violin_data: dict, output_path: str = None):
    """
    Fig 6b — Violin plots of downstream accuracy across 100 runs.

    One panel per teacher.  For each teacher panel, violins show the
    distribution at each budget checkpoint; the passive baseline distribution
    is overlaid as a horizontal band (mean ± std) for comparison.
    """
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    checkpoints   = violin_data["checkpoints"]
    bl_arr        = np.array(violin_data["baseline_accs"])
    bl_mean, bl_std = bl_arr.mean(), bl_arr.std()

    n_teachers = len(teacher_names)
    ncols = 3
    nrows = (n_teachers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True)
    axes = np.array(axes).flatten()

    for ax, tname in zip(axes, teacher_names):
        data_per_budget = [
            np.array(violin_data["teacher_accs"][tname][b])
            for b in checkpoints
        ]
        col = TEACHER_COLORS[tname]

        parts = ax.violinplot(data_per_budget, positions=range(len(checkpoints)),
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_alpha(0.55)
        parts["cmedians"].set_color(col)
        parts["cmedians"].set_linewidth(2)

        # Overlay means as scatter
        means = [d.mean() for d in data_per_budget]
        ax.scatter(range(len(checkpoints)), means,
                   color=col, zorder=5, s=60, label="Mean acc.")

        # Passive baseline band
        ax.axhline(bl_mean, color="#78909C", ls="-", lw=2,
                   label=f"Passive mean ({bl_mean:.3f})")
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std,
                   color="#78909C", alpha=0.12,
                   label=f"Passive ±1σ")

        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels([f"B={b}" for b in checkpoints], fontsize=10)
        ax.set_xlabel("Teaching Budget")
        ax.set_ylabel("Downstream Accuracy")
        ax.set_title(TEACHER_LABELS[tname], fontsize=11, fontweight="bold",
                     color=col)
        ax.legend(fontsize=7, loc="lower right")

    for ax in axes[len(teacher_names):]:
        ax.set_visible(False)

    fig.suptitle(
        "With vs Without Teacher — Accuracy Distribution (100 runs)\n"
        f"Scenario: {violin_data['scenario']}  |  Downstream N={500}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. DISTRIBUTION COMPARISON (Beta-Binomial vs Truncated-Normal vs Logistic-Normal)
# ══════════════════════════════════════════════════════════════════════════════

def _run_teaching_with_estimator(
    agents, M: int, budget: int, teacher_classes,
    estimator_class, estimator_kwargs: dict,
    seed: int = 42, tasks_per_region: int = 500,
) -> dict:
    """
    Like run_teaching_experiment but uses a custom estimator instead of
    the default Beta-Binomial CapabilityEstimator inside each teacher.

    Returns {teacher_name: {"mse_curve": [...], "final_mse": float,
                             "final_estimates": (K,M), "true_caps": (K,M)}}
    """
    from machine_teaching import TaskPool, BaseTeacher, TeachingStep

    true_caps = np.array([[agents[k].get_capability(m) for m in range(M)]
                           for k in range(len(agents))])
    K = len(agents)

    results = {}
    for teacher_cls, kwargs in teacher_classes:
        pool = TaskPool(M=M, tasks_per_region=tasks_per_region, seed=seed)
        kw   = dict(kwargs)
        kw.setdefault("seed", seed)

        # Build teacher with standard init, then swap its estimator
        teacher = teacher_cls(agents=agents, M=M, task_pool=pool, **kw)

        # Replace the internal CapabilityEstimator with the chosen alternative.
        # We need to also update alpha0/alpha1 on the teacher (used in EMSR).
        new_est = estimator_class(K=K, M=M, **estimator_kwargs)
        teacher.capability_estimator = new_est
        # For EMSR helpers that read alpha0/alpha1 off the estimator:
        if hasattr(new_est, "alpha0"):
            teacher.capability_estimator.alpha0 = new_est.alpha0
            teacher.capability_estimator.alpha1 = new_est.alpha1

        # Re-run: override step() to use get_all_posterior_means from new est
        mse_curve = []
        for t in range(budget):
            agent_idx, region = teacher.select_pair(t)
            task       = pool.get_task(region)
            prediction = agents[agent_idx].predict(task, rng=teacher._pred_rng)
            is_correct = prediction == task.y
            new_est.update(agent_idx, region, is_correct)

            # For ImitationTeacher: also update its internal v and slot count
            if hasattr(teacher, "_slot_obs_count"):
                teacher._slot_obs_count[agent_idx, region] += 1
                n_slot = teacher._slot_obs_count[agent_idx, region]
                eta_t  = teacher.eta_base / (teacher.eta_base + n_slot)
                teacher.v[agent_idx, region] += eta_t * (
                    (1.0 if is_correct else 0.0) - teacher.v[agent_idx, region]
                )

            estimated = new_est.get_all_posterior_means()
            mse = float(np.mean((estimated - true_caps) ** 2))
            mse_curve.append(mse)

        results[teacher_cls.__name__] = {
            "mse_curve":       mse_curve,
            "final_mse":       mse_curve[-1],
            "final_estimates": new_est.get_all_posterior_means(),
            "true_caps":       true_caps,
        }
    return results


def run_distribution_comparison(scenario_name: str = "Varying") -> dict:
    """
    Run all teacher × estimator combinations across N_RUNS.

    Returns {estimator_name: {teacher_name: {"mse_runs": (N_RUNS, BUDGET),
                                              "mse_curve": [...],
                                              "final_mse": float,
                                              "downstream_accs": [...]}}}
    """
    print("\n" + "=" * 70)
    print(f"PART 7 — Distribution Comparison ({scenario_name}, {N_RUNS} runs)")
    print("=" * 70)

    agents        = SCENARIOS[scenario_name](M)
    teacher_names = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    all_results   = {}

    for est_cls, est_kwargs, est_name in ESTIMATOR_CLASSES:
        print(f"\n  Estimator: {est_name}")
        est_data = {tn: {"mse_runs": [], "downstream_accs": []}
                    for tn in teacher_names}

        for run in range(N_RUNS):
            if run and run % 25 == 0:
                print(f"    run {run}")

            run_res = _run_teaching_with_estimator(
                agents, M, BUDGET, TEACHER_CLASSES,
                est_cls, est_kwargs, seed=SEED + run,
            )

            # Downstream eval
            gen   = SyntheticDataGenerator(M=M, seed=SEED + run + 7000)
            tasks = gen.generate_task_stream(500)

            for tname in teacher_names:
                mse_c = run_res[tname]["mse_curve"]
                est_data[tname]["mse_runs"].append(mse_c)

                # Downstream accuracy
                est_vals = run_res[tname]["final_estimates"]
                tc       = [Task(x=t.x.copy(), y=t.y, region=t.region) for t in tasks]
                pred_rng = np.random.default_rng(SEED + run + 40000)
                orch     = PaperOrchestrator(
                    agents, M, rng=np.random.default_rng(SEED + run + 50000))
                orch.capability_estimator.inject_estimates(est_vals, n_virtual=50)
                for t, task in enumerate(tc):
                    aidx = orch.select_agent(task, t)
                    pred = agents[aidx].predict(task, rng=pred_rng)
                    orch.update(aidx, task, pred)
                est_data[tname]["downstream_accs"].append(
                    orch.get_performance_stats()["overall_accuracy"]
                )

        for tname in teacher_names:
            arr = np.array(est_data[tname]["mse_runs"])
            est_data[tname]["mse_runs"]  = arr
            est_data[tname]["mse_curve"] = arr.mean(axis=0).tolist()
            est_data[tname]["final_mse"] = float(arr[:, -1].mean())

        all_results[est_name] = est_data

    # Print summary table
    print(f"\n{'Estimator':<22} {'Teacher':<25} {'Final MSE':>10} {'Downstream Acc':>16}")
    print("-" * 76)
    for est_name, est_data in all_results.items():
        for tname in teacher_names:
            d   = est_data[tname]
            acc = float(np.mean(d["downstream_accs"]))
            print(f"  {est_name:<20} {TEACHER_LABELS[tname]:<25}"
                  f" {d['final_mse']:>10.5f} {acc:>16.3f}")

    return all_results


def plot_distribution_mse_curves(dist_results: dict,
                                  scenario_name: str = "Varying",
                                  output_path: str = None):
    """
    Fig 7a — MSE curves: one panel per teacher, three lines per estimator (±1σ).
    """
    teacher_names  = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    estimator_names = [n for _, _, n in ESTIMATOR_CLASSES]

    n_teachers = len(teacher_names)
    ncols = 3
    nrows = (n_teachers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True)
    axes = np.array(axes).flatten()

    for ax, tname in zip(axes, teacher_names):
        for est_name in estimator_names:
            arr  = dist_results[est_name][tname]["mse_runs"]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = np.arange(len(mean))
            col  = ESTIMATOR_COLORS[est_name]
            ax.plot(xs, mean, color=col, lw=2, label=est_name)
            ax.fill_between(xs, mean - std, mean + std, color=col, alpha=BAND_ALPHA)

        ax.axhline(TARGET_MSE, color="grey", ls="--", lw=1.2,
                   label=f"Target={TARGET_MSE}")
        ax.set_title(TEACHER_LABELS[tname], fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Steps")
        ax.set_ylabel("Capability MSE")
        ax.legend(fontsize=7.5)

    for ax in axes[len(teacher_names):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Distribution Comparison — MSE Curves (±1σ, 100 runs)\nScenario: {scenario_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_distribution_mse_heatmap(dist_results: dict,
                                   scenario_name: str = "Varying",
                                   output_path: str = None):
    """Fig 7b — Final-MSE heatmap: rows=teachers, cols=estimators."""
    teacher_names   = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    estimator_names = [n for _, _, n in ESTIMATOR_CLASSES]

    data = np.array([
        [dist_results[en][tn]["final_mse"] for en in estimator_names]
        for tn in teacher_names
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Final MSE")
    ax.set_xticks(range(len(estimator_names)))
    ax.set_xticklabels(estimator_names, fontsize=11)
    ax.set_yticks(range(len(teacher_names)))
    ax.set_yticklabels([TEACHER_LABELS[n] for n in teacher_names], fontsize=10)
    ax.set_title(f"Distribution Comparison — Final MSE  |  {scenario_name}",
                 fontsize=12, fontweight="bold")
    for i in range(len(teacher_names)):
        for j in range(len(estimator_names)):
            v = data[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9,
                    color="black" if v < data.max() * 0.6 else "white")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_distribution_downstream_accuracy(dist_results: dict,
                                           scenario_name: str = "Varying",
                                           output_path: str = None):
    """Fig 7c — Downstream accuracy bar chart: grouped by estimator, one bar per teacher."""
    teacher_names   = [cls.__name__ for cls, _ in TEACHER_CLASSES]
    estimator_names = [n for _, _, n in ESTIMATOR_CLASSES]
    n_t = len(teacher_names)
    x   = np.arange(len(estimator_names))
    w   = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, tname in enumerate(teacher_names):
        accs = [float(np.mean(dist_results[en][tname]["downstream_accs"]))
                for en in estimator_names]
        errs = [float(np.std(dist_results[en][tname]["downstream_accs"]))
                for en in estimator_names]
        ax.bar(x + (i - n_t / 2 + 0.5) * w, accs, w,
               label=TEACHER_LABELS[tname],
               color=TEACHER_COLORS[tname], alpha=0.87, edgecolor="white",
               yerr=errs, capsize=3, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(estimator_names, fontsize=11)
    ax.set_ylabel("Downstream Accuracy (±1σ)")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Distribution Comparison — Downstream Accuracy  |  {scenario_name}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7d. ESTIMATOR COMPARISON INSIDE ORCHESTRATORS
# ══════════════════════════════════════════════════════════════════════════════

def run_orchestrator_estimator_comparison(
    scenario_name: str = "Varying",
) -> dict:
    """
    Run PaperOrchestrator with each of the three capability estimators
    (Beta-Binomial, Truncated-Normal, Logistic-Normal) across N_RUNS seeds
    on scenario_name.

    This directly answers: does the choice of θ_{k,m} estimator inside the
    orchestrator change online accuracy and capability-learning speed?

    Returns:
        {estimator_name: {
            "acc_runs":  (N_RUNS,)          — final overall accuracy per run
            "mse_runs":  (N_RUNS, N_TASKS)  — per-step capability MSE per run
            "acc_mean":  float
            "acc_std":   float
            "mse_curve": list[float]        — mean MSE over steps
            "mse_std":   list[float]        — std MSE over steps
        }}
    """
    print("\n" + "=" * 70)
    print(f"PART 7d — Estimator Comparison in Orchestrators "
          f"({scenario_name}, {N_RUNS} runs)")
    print("=" * 70)

    agents    = SCENARIOS[scenario_name](M)
    true_caps = np.array([[agents[k].get_capability(m) for m in range(M)]
                           for k in range(len(agents))])

    results = {}
    for est_cls, est_kwargs, est_name in ESTIMATOR_CLASSES:
        print(f"\n  Estimator: {est_name}")
        acc_runs, mse_runs = [], []

        for run in range(N_RUNS):
            if run and run % 25 == 0:
                print(f"    run {run}")

            gen   = SyntheticDataGenerator(M=M, seed=SEED + run)
            tasks = gen.generate_task_stream(N_TASKS)

            orch_rng = np.random.default_rng(SEED + run)
            pred_rng = np.random.default_rng(SEED + run + 31337)

            # Build a fresh estimator instance for this run
            est = est_cls(K=len(agents), M=M, **est_kwargs)

            orch = PaperOrchestrator(
                agents, M, rng=orch_rng, capability_estimator=est
            )

            mse_r = []
            for t, task in enumerate(tasks):
                aidx = orch.select_agent(task, t)
                pred = agents[aidx].predict(task, rng=pred_rng)
                orch.update(aidx, task, pred)
                est_means = orch.capability_estimator.get_all_posterior_means()
                mse_r.append(float(np.mean((est_means - true_caps) ** 2)))

            acc_runs.append(orch.get_performance_stats()["overall_accuracy"])
            mse_runs.append(mse_r)

        acc_arr = np.array(acc_runs)
        mse_arr = np.array(mse_runs)
        results[est_name] = {
            "acc_runs":  acc_arr,
            "mse_runs":  mse_arr,
            "acc_mean":  float(acc_arr.mean()),
            "acc_std":   float(acc_arr.std()),
            "mse_curve": mse_arr.mean(axis=0).tolist(),
            "mse_std":   mse_arr.std(axis=0).tolist(),
        }
        print(f"    Accuracy: {results[est_name]['acc_mean']:.3f} "
              f"± {results[est_name]['acc_std']:.3f}")

    return results


def plot_orchestrator_estimator_accuracy(
    orch_est_results: dict,
    scenario_name: str = "Varying",
    output_path: str = None,
):
    """
    Fig 7d-i — Bar chart: PaperOrchestrator accuracy per estimator with ±1σ.

    Also overlays individual run values as a strip plot so the full
    distribution shape is visible alongside the summary bar.
    """
    estimator_names = list(orch_est_results.keys())
    accs  = [orch_est_results[n]["acc_mean"] for n in estimator_names]
    errs  = [orch_est_results[n]["acc_std"]  for n in estimator_names]
    cols  = [ESTIMATOR_COLORS[n] for n in estimator_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(estimator_names))
    bars = ax.bar(x, accs, color=cols, alpha=0.80, edgecolor="white",
                  yerr=errs, capsize=5, error_kw={"elinewidth": 1.5},
                  zorder=2)

    # Strip plot of individual run values
    rng_jitter = np.random.default_rng(0)
    for i, name in enumerate(estimator_names):
        runs = orch_est_results[name]["acc_runs"]
        jitter = rng_jitter.uniform(-0.18, 0.18, size=len(runs))
        ax.scatter(i + jitter, runs, color=cols[i], alpha=0.25,
                   s=12, zorder=3)

    # Annotate bar tops
    for bar, acc, err in zip(bars, accs, errs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                acc + err + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(estimator_names, fontsize=11)
    ax.set_ylabel("Overall Accuracy (±1σ, 100 runs)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Estimator Comparison in PaperOrchestrator — Accuracy\n"
        f"Scenario: {scenario_name}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_orchestrator_estimator_mse(
    orch_est_results: dict,
    scenario_name: str = "Varying",
    output_path: str = None,
):
    """
    Fig 7d-ii — MSE convergence curves: one line per estimator, ±1σ bands.

    Shows how fast each estimator drives capability estimates toward the true
    values when embedded inside a live PaperOrchestrator (passive stream).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for name, d in orch_est_results.items():
        mean = np.array(d["mse_curve"])
        std  = np.array(d["mse_std"])
        xs   = np.arange(len(mean))
        col  = ESTIMATOR_COLORS[name]
        ax.plot(xs, mean, color=col, lw=2.5, label=name)
        ax.fill_between(xs, mean - std, mean + std, color=col, alpha=BAND_ALPHA)

    ax.axhline(TARGET_MSE, color="grey", ls="--", lw=1.2,
               label=f"Target MSE={TARGET_MSE}")
    ax.set_xlabel("Task Index (online stream)")
    ax.set_ylabel("Capability MSE")
    ax.set_title(
        f"Estimator Comparison in PaperOrchestrator — MSE Convergence (±1σ)\n"
        f"Scenario: {scenario_name}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print(" Multi-Agent Orchestration — Unified Experiment Suite")
    print(f" M={M} | N={N_TASKS} | Budget={BUDGET} | seed={SEED} | runs={N_RUNS}")
    print("=" * 70)

    # Part 1 — Baselines
    baseline_results = run_baseline_experiments()
    print("\nComputing baseline curves...")
    baseline_curves = compute_baseline_curves()
    print("\nGenerating baseline figures...")
    plot_baseline_accuracy(baseline_results, "output/1a_baseline_accuracy.png")
    plot_baseline_learning_curves(baseline_curves,
                                   "output/1b_baseline_learning_curves.png")
    plot_baseline_capability_mse(baseline_curves,
                                  "output/1c_baseline_capability_mse.png")

    # Part 2 — Machine teaching
    teaching_results = run_teaching_experiments()
    print("\nGenerating teaching figures...")
    plot_mse_curves(teaching_results, "output/2a_teaching_mse_curves.png")
    plot_efficiency_bars(teaching_results, "output/2b_teaching_efficiency.png")
    plot_final_mse_heatmap(teaching_results, "output/2c_teaching_mse_heatmap.png")
    plot_capability_estimates(teaching_results, scenario="Varying",
                               output_path="output/2d_capability_estimates_varying.png")
    plot_convergence_speed(teaching_results, "output/2e_teaching_convergence_speed.png")
    plot_mse_auc(teaching_results, "output/2f_teaching_mse_auc.png")

    # Part 3 — Unified comparison
    bl_accs, dt_accs = evaluate_downstream_orchestration()
    plot_unified_accuracy(bl_accs, dt_accs,
                           "output/3a_unified_accuracy_comparison.png")
    plot_convergence_mse_absolute(baseline_curves, teaching_results,
                                   "output/3b_convergence_mse_absolute.png")

    # Part 4 — Prior sensitivity
    prior_results = run_prior_experiments(scenario_name="Varying")
    print("\nGenerating prior figures...")
    plot_prior_mse_curves(prior_results, "Varying",
                           "output/4a_prior_mse_curves.png")
    plot_prior_efficiency_bars(prior_results, "Varying",
                                "output/4b_prior_efficiency_bars.png")
    plot_prior_final_mse_heatmap(prior_results, "Varying",
                                  "output/4c_prior_final_mse_heatmap.png")

    # Part 5 — Budget vs Downstream Accuracy
    tradeoff_results = run_budget_tradeoff()
    plot_budget_tradeoff(tradeoff_results,
                          "output/5a_budget_vs_downstream_accuracy.png")

    # Part 6 — With vs Without Teacher Variance Comparison
    print("\n" + "=" * 70)
    print("PART 6 — With vs Without Teacher Variance Comparison")
    print("=" * 70)
    cmp_data = run_with_without_teacher_comparison(baseline_curves, teaching_results,
                                                    scenario="Varying")
    plot_variance_comparison_curves(cmp_data,
                                    "output/6a_variance_comparison_curves.png")
    print("\nCollecting violin data (this runs N_RUNS × checkpoints teaching jobs)...")
    violin_data = run_violin_data(scenario="Varying",
                                   checkpoint_budgets=(50, 100, 200))
    plot_variance_violin(violin_data, "output/6b_variance_violin.png")

    # Part 7 — Distribution Comparison
    dist_results = run_distribution_comparison(scenario_name="Varying")
    print("\nGenerating distribution comparison figures...")
    plot_distribution_mse_curves(dist_results, "Varying",
                                  "output/7a_distribution_mse_curves.png")
    plot_distribution_mse_heatmap(dist_results, "Varying",
                                   "output/7b_distribution_mse_heatmap.png")
    plot_distribution_downstream_accuracy(dist_results, "Varying",
                                           "output/7c_distribution_downstream_accuracy.png")

    # Part 7d — Estimator comparison inside orchestrators
    orch_est_results = run_orchestrator_estimator_comparison(scenario_name="Varying")
    plot_orchestrator_estimator_accuracy(
        orch_est_results, "Varying",
        "output/7d_orchestrator_estimator_accuracy.png")
    plot_orchestrator_estimator_mse(
        orch_est_results, "Varying",
        "output/7e_orchestrator_estimator_mse.png")

    print("\n" + "=" * 70)
    print("All experiments complete.  Figures in output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
