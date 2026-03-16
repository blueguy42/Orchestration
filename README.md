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
from priors import BetaPrior

# Create agents with varying expertise
agents = create_agents_varying(M=3)

# Generate task stream  
data_gen = SyntheticDataGenerator(M=3, seed=42)
tasks = data_gen.generate_task_stream(N=1000)

# Initialize orchestrator with custom prior
prior = BetaPrior("optimistic")  # or use "uniform", "weak", "strong", etc.
orchestrator = PaperOrchestrator(agents, M=3, prior=prior)

# Run orchestration
for t, task in enumerate(tasks):
    # Select agent
    agent_idx = orchestrator.select_agent(task, t)
    
    # Get prediction
    prediction = agents[agent_idx].predict(task)
    
    # Update with feedback
    orchestrator.update(agent_idx, task, prediction)

# Get results
stats = orchestrator.get_performance_stats()
print(f"Overall Accuracy: {stats['overall_accuracy']:.3f}")
```

### Machine Teaching

```python
from machine_teaching import SurrogateTeacher
from orchestration_framework import PaperOrchestrator
from synthetic_experiments import create_agents_varying

agents = create_agents_varying(M=3)

# Initialize teacher (actively selects agent-region pairs)
teacher = SurrogateTeacher(agents, M=3, prior="weak", budget=200)

# Run teaching: returns orchestrator trained on strategically selected tasks
trained_orchestrator = teacher.run_teaching()

# Access learning curves
print(f"Final MSE: {teacher.mse_curve[-1]:.4f}")
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
- `Task`: Data structure for tasks
- `Agent`: Agent with capabilities and costs
- `Prior`: Base class for prior distributions
- `RegionEstimator`: Bayesian region probability estimation (supports custom priors)
- `CapabilityEstimator`: Bayesian capability estimation (supports custom priors)
- `BaseOrchestrator`: Abstract base class
- `PaperOrchestrator`: Bhatt et al. implementation
- `UCB1Orchestrator`: Upper Confidence Bound
- `GreedyOrchestrator`: Myopic selection
- `RandomOrchestrator`: Random baseline
- `OracleOrchestrator`: Perfect knowledge upper bound

### `priors.py`
Prior distributions for Bayesian capability estimation:
- **3 Prior Types**: `BetaPrior` (uniform Beta(1,1)), `JeffreysPrior` (Beta(0.5,0.5)), `SkewedExpertPrior` (Beta(3,1))
- All priors expose `alpha0`, `alpha1`, `name`, `prior_mean()`, and `posterior_mean()`
- `ALL_PRIORS`: registry list used in prior sensitivity experiments
- `PRIOR_LABELS`, `PRIOR_COLORS`: display metadata for plots

### `machine_teaching.py`
Machine teaching orchestrators:
- `BaseTeacher`: Abstract teacher base class
- `OmniscientTeacher`: Knows true capabilities (upper bound)
- `ImitationTeacher`: Maintains internal capability estimate
- `SurrogateTeacher`: Uncertainty-based sampling (max query entropy)
- `RoundRobinTeacher`: Deterministic region cycling
- `RandomTeacher`: Uniform random sampling (lower bound)
- Teaching experiment infrastructure

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
Main comprehensive experiment script with 4 parts:
- **Part 1** — Baseline orchestrators (Random, Greedy, UCB1, Paper, Oracle) across all scenarios
- **Part 2** — Machine teaching approaches (Random, RoundRobin, Surrogate, Imitation, Omniscient): MSE convergence and efficiency
- **Part 3** — Unified comparison: downstream orchestration accuracy and convergence curves for all baselines and teaching-guided methods
- **Part 4** — Prior sensitivity analysis (Uniform, Jeffreys, Expert) on the Varying scenario
- **100 independent runs** per configuration for robust statistical analysis
- Generates all visualizations to `output/` and prints summary tables

## Outputs Generated

### Experiment Results (`run_experiments.py`)

All figures are saved to `output/`:

**Part 1 — Baselines**
- `baseline_accuracy.png` — grouped bar chart of overall accuracy per scenario
- `baseline_learning_curves.png` — rolling accuracy over the task stream (4 panels)
- `baseline_capability_mse.png` — capability estimation MSE over the task stream (4 panels)

**Part 2 — Machine Teaching**
- `teaching_mse_curves.png` — MSE convergence per teacher × scenario (4 panels)
- `teaching_efficiency.png` — steps to target MSE bar chart
- `teaching_mse_heatmap.png` — final MSE heatmap (scenarios × teachers)
- `capability_estimates_varying.png` — true vs estimated capability scatter (Varying scenario)
- `teaching_convergence_speed.png` — steps to 2×, 1×, ½× target MSE (Varying)
- `teaching_mse_auc.png` — area under MSE curve (lower = faster convergence)

**Part 3 — Unified Comparison**
- `unified_accuracy_comparison.png` — baselines vs teaching-guided accuracy (Varying scenario)
- `convergence_rolling_accuracy.png` — rolling accuracy for all methods, normalized x-axis (4 panels)
- `convergence_mse.png` — capability MSE for all methods, normalized x-axis (4 panels)

**Part 4 — Prior Sensitivity (Varying scenario)**
- `prior_mse_curves.png` — MSE curves per prior type
- `prior_efficiency_bars.png` — steps to target MSE per teacher × prior
- `prior_final_mse_heatmap.png` — final MSE heatmap (teachers × priors)

Console output includes performance tables, appropriateness metrics, teaching efficiency, and prior-specific analysis.

## Machine Teaching Framework

An active learning approach for efficient capability estimation. Instead of passively observing a task stream, the teacher strategically selects which agent-region pairs to evaluate.

### Teaching Approaches

**Available Teachers:**
1. **OmniscientTeacher** - Knows true capabilities (upper bound)
2. **ImitationTeacher** - Maintains internal capability estimate and learns from orchestration feedback
3. **SurrogateTeacher** - Selects tasks that maximize uncertainty (highest query entropy)
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

```
numpy>=1.21.0
matplotlib>=3.4.0
```

## References

Bhatt, U., Kapoor, S., Upadhyay, M., Sucholutsky, I., Quinzan, F., Collins, K. M., 
Weller, A., Wilson, A. G., and Zafar, M. B. (2025). When should we orchestrate 
multiple agents? arXiv preprint arXiv:2503.13577.
