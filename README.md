# Multi-Agent Orchestration Framework

Implementation of baseline orchestration algorithms based on:
**Bhatt et al. (2025) "When Should We Orchestrate Multiple Agents?"**

## Overview

This framework implements a multi-agent orchestration system for efficiently routing tasks between heterogeneous agents (humans and AI) whose capabilities vary across different task regions. The system learns agent capabilities online and makes orchestration decisions that balance performance and cost.

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
orchestration_framework.py    # Core classes and orchestrators
synthetic_experiments.py      # Experiment setup and evaluation
run_experiments.py           # Main script to run all experiments
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

## Usage Example

```python
from orchestration_framework import Agent, Task, PaperOrchestrator
from synthetic_experiments import SyntheticDataGenerator, create_agents_varying

# Create agents with varying expertise
agents = create_agents_varying(M=3)

# Generate task stream  
data_gen = SyntheticDataGenerator(M=3, seed=42)
tasks = data_gen.generate_task_stream(N=1000)

# Initialize orchestrator
orchestrator = PaperOrchestrator(agents, M=3)

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
Core implementation:
- `Task`: Data structure for tasks
- `Agent`: Agent with capabilities and costs
- `RegionEstimator`: Bayesian region probability estimation
- `CapabilityEstimator`: Bayesian capability estimation
- `BaseOrchestrator`: Abstract base class
- `PaperOrchestrator`: Bhatt et al. implementation
- `UCB1Orchestrator`: Upper Confidence Bound
- `GreedyOrchestrator`: Myopic selection
- `RandomOrchestrator`: Random baseline
- `OracleOrchestrator`: Perfect knowledge upper bound

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
Main script:
- Runs all scenarios
- Compares all orchestrators
- Generates visualizations
- Prints comprehensive results

## Outputs Generated

1. **orchestrator_comparison.png**
   - Bar chart comparing all orchestrators across scenarios
   - Shows relative performance clearly

2. **learning_curves.png**
   - Time series showing accuracy convergence
   - Moving average with window=50
   - Demonstrates learning speed differences

## Extensions for Machine Teaching

This baseline implementation provides the foundation for the proposed machine teaching approach:

1. **Strategic Task Selection**: Instead of random stream, teacher selects informative tasks
2. **Efficient Capability Learning**: Minimize evaluations needed for accurate estimation
3. **Region-Aware Sampling**: Target uncertain regions for faster convergence
4. **Active Learning**: Query most informative agent-region pairs

The machine teaching extension would modify task generation to strategically probe agent capabilities rather than observing a passive stream.

## Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
```

## References

Bhatt, U., Kapoor, S., Upadhyay, M., Sucholutsky, I., Quinzan, F., Collins, K. M., 
Weller, A., Wilson, A. G., and Zafar, M. B. (2025). When should we orchestrate 
multiple agents? arXiv preprint arXiv:2503.13577.
