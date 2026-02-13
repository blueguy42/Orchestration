import numpy as np
from orchestration import OrchestrationFramework, Teacher

# 1. Setup Ground Truth Capabilities (from Source [2])
# Rows: Agents (A1-A4), Columns: Regions (R1-R3)
true_capabilities = np.array([
    [0.650, 0.076, 0.274],  # A1: Strong in R1
    [0.399, 0.298, 0.303],  # A2: Moderate
    [0.079, 0.852, 0.069],  # A3: Strong in R2
    [0.031, 0.091, 0.877]   # A4: Strong in R3
])

# 2. Initialize Framework
# Using 4 agents and 3 regions as specified in the synthetic setup [5]
orch = OrchestrationFramework(num_agents=4, num_regions=3)
teacher = Teacher()

print(f"{'Step':<5} | {'Region':<8} | {'Selected Agent':<15} | {'Result':<10} | {'Utility':<8}")
print("-" * 55)

# 3. Run Simulation for 10 steps
np.random.seed(42) # For reproducibility
for t in range(1, 11):
    # Teacher selects a task region (random baseline for now) [6, 7]
    current_region = teacher.select_task_region()
    region_name = f"R{current_region + 1}"
    
    # Orchestrator selects the best agent based on current utility [4, 8]
    selected_agent = orch.select_agent(current_region)
    utility = orch.compute_utility(selected_agent, current_region)
    
    # Simulate performance based on ground truth (Bernoulli trial) [9]
    success_prob = true_capabilities[selected_agent, current_region]
    is_correct = np.random.rand() < success_prob
    result_str = "CORRECT" if is_correct else "WRONG"
    
    # Update the framework with the outcome [10, 11]
    orch.update_performance(selected_agent, current_region, is_correct)
    
    print(f"{t:<5} | {region_name:<8} | Agent A{selected_agent+1:<10} | {result_str:<10} | {utility:.3f}")

# 4. Final Learned Estimates
print("\nFinal Learned Correctness Estimates (ct,km):")
for k in range(4):
    estimates = [orch.get_agent_correctness(k, m) for m in range(3)]
    print(f"Agent A{k+1}: {['%.2f' % e for e in estimates]}")