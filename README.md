# Multi-Agent Orchestration Framework

Implementation of baseline orchestration algorithms based on:
**Bhatt et al. (2025) "When Should We Orchestrate Multiple Agents?"**

## Overview

This framework implements a multi-agent orchestration system for efficiently routing tasks between heterogeneous agents (humans and AI) whose capabilities vary across different task regions. The system learns agent capabilities online and makes orchestration decisions that balance performance and cost.

**Two Approaches:**
1. **Passive Orchestration**: Observe random task stream and learn which agent is best for each region
2. **Active Machine Teaching**: Strategically select agent-region pairs to evaluate for faster learning

## Key Concepts

### Agents
- Heterogeneous agents with varying capabilities across task regions
- Each agent has region-specific performance probabilities P(A_k | R_m)
- Agents incur costs γ_km for handling tasks in region R_m

### Regions
- Task space partitioned into M regions (e.g., different subjects, difficulty levels)
- Within a region, agent performance varies slowly
- Across regions, agents may have very different capabilities (jagged frontier)

### Orchestration
- Sequential decision-making: select optimal agent for each incoming task
- Learn agent capabilities online using Beta-Binomial estimation
- Account for both performance and cost in selection

## Implementation Structure

```
orchestration_framework.py    # Core orchestrator classes (Agent, Task, Orchestrators)
machine_teaching.py          # Machine teaching approach (Teachers and task pool)
priors.py                    # Prior distributions for Bayesian estimation
synthetic_experiments.py      # Synthetic task generation and evaluation utilities
run_experiments.py           # Main script for comprehensive experiments
```

## Orchestration Algorithms Implemented

### 1. Paper Orchestrator (Bhatt et al. 2025)
**Algorithm**: Selects agent with highest empirical utility U≥t(Ak)

```
U≥t(Ak) = C≥t(Ak) / γ_krt

where C≥t(Ak) = c_t,krt × Σ_m w_t,m × c_t,km
```

- `c_t,krt`: Estimated capability in current region
- `w_t,m`: Estimated region probability  
- `γ_krt`: Cost in current region

**Key features**:
- Accounts for both immediate and future performance
- Balances correctness with cost
- Uses Bayesian estimation with Dirichlet-Multinomial for regions and Beta-Binomial for capabilities

### 2. UCB1 Orchestrator
**Algorithm**: Upper Confidence Bound for exploration-exploitation

```
UCB(k, r) = Q(k, r) + c × sqrt(ln(t) / N(k, r))
```

- `Q(k, r)`: Estimated capability of agent k in region r
- `N(k, r)`: Number of times agent k used in region r
- `c`: Exploration parameter (default: 1.0)

**Key features**:
- Explicitly balances exploration and exploitation
- Confidence bounds ensure sufficient exploration early on
- Converges to optimal policy under stationarity assumptions

### 3. Greedy Orchestrator
**Algorithm**: Myopic selection based on current region only

```
k* = argmax_k c_t,krt
```

**Key features**:
- Simple and fast
- No lookahead to future regions
- No cost consideration
- Can be effective if regions are isolated

### 4. Random Orchestrator (Baseline)
**Algorithm**: Uniform random selection

```
k* ~ Uniform(1, ..., K)
```

**Key features**:
- No learning
- Provides lower bound on performance
- Useful for computing appropriateness metric

### 5. Oracle Orchestrator (Upper Bound)
**Algorithm**: Has perfect knowledge of true capabilities

```
k* = argmax_k P(Ak | Rrt)
```

**Key features**:
- Theoretical upper bound
- Shows maximum achievable performance
- Useful for measuring learning efficiency

## Appropriateness of Orchestration

The **appropriateness metric** (Equation 5 in paper) measures whether orchestration is worthwhile:

```
App = C_max / C_rand
```

- `C_max`: Expected correctness with optimal agent selection
- `C_rand`: Expected correctness with random selection

**Interpretation**:
- `App ≈ 1.0`: Orchestration provides minimal benefit (agents are similar)
- `App > 1.0`: Orchestration is beneficial (agents have differentiated expertise)
- `App >> 1.0`: Orchestration is highly beneficial (strong specialization)

## Expertise Scenarios

### 1. Approximately Invariant
All agents have similar capabilities across all regions.

**Expected behavior**:
- Low appropriateness (≈1.05)
- Minimal benefit from orchestration
- All methods perform similarly

### 2. Dominant
One agent strictly better than others across all regions.

**Expected behavior**:
- High appropriateness (≈2.38)
- Strong benefit from orchestration
- Paper method should converge to always using dominant agent
- Cost misalignment can reduce benefits

### 3. Varying (Complementary)
Each agent excels in different regions.

**Expected behavior**:
- High appropriateness (≈2.38)
- Greatest benefit from orchestration
- Need to learn which agent for which region
- Paper method should match oracle performance

## Experimental Results

**Note:** All results are averaged over **100 independent runs** with different random seeds for robust statistical analysis.

### Performance Summary

| Scenario | App | Random | Greedy | UCB1 | Paper | Oracle |
|----------|-----|--------|--------|------|-------|--------|
| Approx. Invariant | 1.045 | 0.301 | 0.320 | 0.298 | **0.329** | 0.333 |
| Dominant | 2.380 | 0.306 | 0.572 | 0.685 | **0.794** | 0.783 |
| Dominant + Mis. Cost | 2.380 | 0.302 | **0.785** | 0.769 | 0.627 | 0.789 |
| Varying | 2.380 | 0.313 | 0.790 | 0.733 | **0.804** | 0.797 |

### Key Findings

1. **Paper orchestrator excels in high-appropriateness scenarios**
   - Dominant: +159.5% vs random, matches oracle
   - Varying: +156.9% vs random, slightly beats oracle

2. **Greedy can outperform in some cases**
   - Dominant + Misaligned Cost: Greedy ignores cost and focuses on capability
   - Paper method properly accounts for cost, reducing performance

3. **UCB1 provides robust exploration**
   - Better than greedy in dominant scenario (0.685 vs 0.572)
   - Explicit exploration prevents premature convergence

4. **Appropriateness predicts orchestration value**
   - Low (1.045): Only 9.3% improvement vs random
   - High (2.380): 100-160% improvement vs random

## Usage Examples

### Running Comprehensive Experiments

```bash
python run_experiments.py
```

### Baseline Orchestration

```python
from orchestration_framework import Agent, Task, PaperOrchestrator
from synthetic_experiments import SyntheticDataGenerator, create_agents_varying

# Create agents with varying expertise
agents = create_agents_varying(M=3)

# Generate task stream
data_gen = SyntheticDataGenerator(M=3, seed=42)
tasks = data_gen.generate_task_stream(N=1000)

# Initialize orchestrator (uses uniform Beta(1,1) prior by default)
orchestrator = PaperOrchestrator(agents, M=3)

# Run orchestration
for t, task in enumerate(tasks):
    agent_idx = orchestrator.select_agent(task, t)
    prediction = agents[agent_idx].predict(task)
    orchestrator.update(agent_idx, task, prediction)

# Get results
stats = orchestrator.get_performance_stats()
print(f"Overall Accuracy: {stats['overall_accuracy']:.3f}")
```

### Machine Teaching

```python
from machine_teaching import TaskPool, SurrogateTeacher, run_teaching_experiment
from synthetic_experiments import create_agents_varying

agents = create_agents_varying(M=3)
M = 3

# Run a single teacher
pool = TaskPool(M=M, tasks_per_region=500, seed=42)
teacher = SurrogateTeacher(agents=agents, M=M, task_pool=pool)
teacher.run(budget=200)

summary = teacher.get_summary()
print(f"Final MSE: {summary['final_mse']:.4f}")

# Or run all teachers at once for comparison
from machine_teaching import OmniscientTeacher, ImitationTeacher, RandomTeacher, RoundRobinTeacher

teacher_classes = [
    (OmniscientTeacher, {}),
    (ImitationTeacher,  {"eta_v": 5.0}),
    (SurrogateTeacher,  {}),
    (RoundRobinTeacher, {}),
    (RandomTeacher,     {}),
]
results = run_teaching_experiment(agents, M=M, budget=200, teacher_classes=teacher_classes)
```

## Mathematical Framework

### Region Probability Estimation
Using Dirichlet-Multinomial conjugacy:

```
P(w | D<t) ∝ ∏_m w_m^(n<t,m + α_m - 1)

MAP estimate: w_t,m = (n<t,m + α_m - 1) / Σ_j(n<t,j + α_j - 1)
```

### Capability Estimation
Using Beta-Binomial conjugacy:

```
P(c_km | D<t) ∝ (1-c_km)^(n<t,0 + α_0 - 1) × c_km^(n<t,1 + α_1 - 1)

MAP estimate: c_t,km = (n<t,1 + α_1 - 1) / (n<t,0 + n<t,1 + α_0 + α_1 - 2)
```

### Empirical Utility (Equation 4)
```
Û≥t(Ak) = ĉ_t,krt × Σ_m ŵ_t,m × ĉ_t,km / γ_krt
```

## File Descriptions

### `orchestration_framework.py`
Core orchestration implementation:
- `Task`: Data structure for tasks (input, label, region)
- `Agent`: Agent with region-specific capabilities and costs
- `RegionEstimator`: Dirichlet-Multinomial Bayesian estimation of region probabilities
- `CapabilityEstimator`: Beta-Binomial Bayesian estimation of agent capabilities per region; supports custom prior objects from `priors.py`; exposes `get_capability()` (MAP), `get_posterior_mean()`, `get_all_capabilities()`, `get_all_posterior_means()`, and `inject_estimates()` (pre-seeds with virtual observations for teaching-guided warm-starts)
- `BaseOrchestrator`: Abstract base class
- `PaperOrchestrator`: Bhatt et al. empirical utility implementation
- `UCB1Orchestrator`: Upper Confidence Bound (per-region bandit)
- `GreedyOrchestrator`: Myopic selection by current-region capability
- `RandomOrchestrator`: Random baseline
- `OracleOrchestrator`: Perfect knowledge upper bound

### `priors.py`
Prior distributions for Bayesian capability estimation:
- **3 Prior Types**: `BetaPrior` (uniform Beta(1,1)), `JeffreysPrior` (Beta(0.5,0.5)), `SkewedExpertPrior` (Beta(3,1))
- All priors expose `alpha0`, `alpha1`, `name`, `prior_mean()`, `prior_variance()`, `posterior_params()`, `posterior_mean()`, `posterior_variance()`, and `map_estimate()`
- `ALL_PRIORS`: registry list used in prior sensitivity experiments
- `PRIOR_LABELS`, `PRIOR_COLORS`: display metadata for plots

### `machine_teaching.py`
Machine teaching orchestrators (inspired by Liu et al. 2017 "Iterative Machine Teaching"):
- `TeachingStep`: Dataclass recording one evaluation step (agent, region, outcome, MSE)
- `TaskPool`: Region-grouped task pool for controlled teacher queries
- `BaseTeacher`: Abstract teacher base class with `run()`, `step()`, `current_estimates()`, `get_posterior_variance()`, `total_variance()`, `get_summary()`
- `OmniscientTeacher`: Knows true capabilities; selects by Expected MSE Reduction (EMSR) criterion (upper bound)
- `ImitationTeacher`: Maintains a Robbins-Monro running estimate of θ* and uses it in the EMSR formula (bridges omniscient and surrogate)
- `SurrogateTeacher`: Uncertainty-based sampling via maximum posterior variance (A-optimal / uncertainty sampling); does not require knowledge of true capabilities
- `RoundRobinTeacher`: Deterministic cycling through agent-region pairs
- `RandomTeacher`: Uniform random selection (lower bound)
- `run_teaching_experiment()`: Runs all teacher classes on a shared agent set with isolated RNGs
- `compute_teaching_efficiency()`: Steps-to-target-MSE metric
- `print_teaching_results()`: Console summary table

### `synthetic_experiments.py`
Experiment utilities:
- `SyntheticDataGenerator`: Generate task streams
- `create_agents_*`: Create agents for each expertise scenario
- `compute_appropriateness`: Calculate App metric
- `run_experiment`: Run single orchestrator
- `run_all_baselines`: Run all orchestrators
- `plot_results`: Visualize comparisons
- `plot_learning_curves`: Visualize learning over time

### `run_experiments.py`
Main comprehensive experiment script with 5 parts:
- **Part 1** — Baseline orchestrators (Random, Greedy, UCB1, Paper, Oracle) across all scenarios
- **Part 2** — Machine teaching approaches (Random, RoundRobin, Surrogate, Imitation, Omniscient): MSE convergence and efficiency
- **Part 3** — Unified comparison: downstream orchestration accuracy and convergence curves for all baselines and teaching-guided methods on a shared absolute x-axis
- **Part 4** — Prior sensitivity analysis (Uniform, Jeffreys, Expert) on the Varying scenario
- **Part 5** — Budget vs downstream accuracy trade-off sweep (budget 10–300) for the Varying scenario
- **100 independent runs** per configuration for robust statistical analysis
- Generates all visualizations to `output/` and prints summary tables

## Outputs Generated

### Experiment Results (`run_experiments.py`)

All figures are saved to `output/`:

**Part 1 — Baselines**
- `1a_baseline_accuracy.png` — grouped bar chart of overall accuracy per scenario
- `1b_baseline_learning_curves.png` — rolling accuracy over the task stream (4 panels)
- `1c_baseline_capability_mse.png` — capability estimation MSE over the task stream (4 panels)

**Part 2 — Machine Teaching**
- `2a_teaching_mse_curves.png` — MSE convergence per teacher × scenario (4 panels)
- `2b_teaching_efficiency.png` — steps to target MSE bar chart
- `2c_teaching_mse_heatmap.png` — final MSE heatmap (scenarios × teachers)
- `2d_capability_estimates_varying.png` — true vs estimated capability scatter (Varying scenario)
- `2e_teaching_convergence_speed.png` — steps to 2×, 1×, ½× target MSE (Varying)
- `2f_teaching_mse_auc.png` — area under MSE curve (lower = faster convergence)

**Part 3 — Unified Comparison**
- `3a_unified_accuracy_comparison.png` — baselines vs teaching-guided downstream accuracy (Varying scenario)
- `3b_convergence_mse_absolute.png` — capability MSE for baselines and teachers on a shared absolute evaluation-step axis (4 panels)

**Part 4 — Prior Sensitivity (Varying scenario)**
- `4a_prior_mse_curves.png` — MSE curves per prior type
- `4b_prior_efficiency_bars.png` — steps to target MSE per teacher × prior
- `4c_prior_final_mse_heatmap.png` — final MSE heatmap (teachers × priors)

**Part 5 — Budget vs Downstream Accuracy (Varying scenario)**
- `5a_budget_vs_downstream_accuracy.png` — downstream accuracy as a function of teaching budget for each teacher (±1σ), with Oracle and Random reference lines

Console output includes performance tables, appropriateness metrics, teaching efficiency, and prior-specific analysis.

## Machine Teaching Framework

An active learning approach for efficient capability estimation. Instead of passively observing a task stream, the teacher strategically selects which agent-region pairs to evaluate.

### Teaching Approaches

**Available Teachers:**
1. **OmniscientTeacher** - Knows true capabilities; selects by Expected MSE Reduction (EMSR) criterion (upper bound)
2. **ImitationTeacher** - Maintains a Robbins-Monro running estimate of true capabilities and uses it in the EMSR formula (bridges omniscient and surrogate)
3. **SurrogateTeacher** - Selects the slot with maximum posterior variance (A-optimal uncertainty sampling); does not require true capabilities
4. **RoundRobinTeacher** - Deterministic cycling through agent-region pairs
5. **RandomTeacher** - Uniform random selection (lower bound)

### Key Features

- **Prior Support**: All teachers support custom prior distributions (Beta priors for capabilities, Dirichlet for regions)
- **Efficiency Metric**: Measures MSE convergence of capability estimates over teaching budget
- **Comparison Mode**: Unified experiments compare baseline orchestrators with machine teaching approaches
- **Flexible Sampling**: Different strategies to probe agent capabilities for faster learning

## Running Experiments

### Main Experiment Runner
```bash
python run_experiments.py
```

Runs comprehensive analysis:
- **4 expertise scenarios**: Approximately Invariant, Dominant, Dominant+Misaligned Cost, Varying
- **5 baseline orchestrators**: Random, Greedy, UCB1, Paper, Oracle  
- **5 machine teaching approaches**: Random, RoundRobin, Surrogate, Imitation, Omniscient
- **3 prior distributions**: Uniform Beta(1,1), Jeffreys Beta(0.5,0.5), Expert Beta(3,1)
- **100 independent runs** per configuration for robust statistical analysis
- Generates visualizations in `output/` directory

All results are stored in the `output/` folder for easy access and further analysis.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

```
numpy>=1.21.0
matplotlib>=3.4.0
```

## References

Bhatt, U., Kapoor, S., Upadhyay, M., Sucholutsky, I., Quinzan, F., Collins, K. M.,
Weller, A., Wilson, A. G., and Zafar, M. B. (2025). When should we orchestrate
multiple agents? arXiv preprint arXiv:2503.13577.

Liu, W., Dai, B., Humayun, A., Tay, C., Yu, C., Smith, L. B., Rehg, J. M., and
Song, L. (2017). Iterative machine teaching. Proceedings of the 34th International
Conference on Machine Learning (ICML), PMLR 70:2149–2158.
