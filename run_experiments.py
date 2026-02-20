"""
Main script to run baseline synthetic experiments for multi-agent orchestration.
"""

import numpy as np
from synthetic_experiments import (
    SyntheticDataGenerator,
    create_agents_approximately_invariant,
    create_agents_dominant,
    create_agents_dominant_misaligned_cost,
    create_agents_varying,
    run_all_baselines,
    compute_appropriateness,
    plot_results,
    plot_learning_curves
)


def main():
    """Run all baseline experiments"""
    
    print("="*80)
    print("Multi-Agent Orchestration: Baseline Experiments")
    print("="*80)
    
    # Configuration
    M = 3  # Number of regions
    N = 1000  # Number of tasks
    seed = 42
    
    # Generate task stream
    print(f"\nGenerating task stream: N={N} tasks, M={M} regions")
    data_gen = SyntheticDataGenerator(M=M, seed=seed)
    tasks = data_gen.generate_task_stream(N)
    
    # Define scenarios
    scenarios = {
        'Approximately Invariant': create_agents_approximately_invariant(M),
        'Dominant': create_agents_dominant(M),
        'Dominant + Misaligned Cost': create_agents_dominant_misaligned_cost(M),
        'Varying': create_agents_varying(M)
    }
    
    # Run experiments for each scenario
    all_results = {}
    
    for scenario_name, agents in scenarios.items():
        print("\n" + "="*80)
        print(f"Scenario: {scenario_name}")
        print("="*80)
        
        # Compute and display appropriateness
        app = compute_appropriateness(agents, M)
        print(f"\nAppropriateness of Orchestration: {app:.3f}")
        print(f"  (Higher values indicate greater benefit from orchestration)")
        
        # Display agent capabilities
        print("\nAgent Capabilities (P(correct | region)):")
        for i, agent in enumerate(agents):
            caps = [f"{c:.3f}" for c in agent.capabilities]
            costs = [f"{c:.1f}" for c in agent.costs]
            print(f"  {agent.name}: Capabilities={caps}, Costs={costs}")
        
        # Run all orchestrators
        print("\nRunning orchestrators...")
        results = run_all_baselines(agents, tasks, verbose=True)
        all_results[scenario_name] = results
        
        # Print summary
        print("\n" + "-"*80)
        print("Summary:")
        best_orch = max(results.items(), key=lambda x: x[1]['overall_accuracy'])
        print(f"  Best Orchestrator: {best_orch[0]} "
              f"(Accuracy: {best_orch[1]['overall_accuracy']:.3f})")
    
    # Print overall comparison
    print("\n" + "="*80)
    print("OVERALL RESULTS COMPARISON")
    print("="*80)
    
    # Create comparison table
    orchestrator_names = list(next(iter(all_results.values())).keys())
    
    print("\n{:<30} {}".format("Scenario", " | ".join(f"{o[:15]:^15}" for o in orchestrator_names)))
    print("-" * (30 + 19 * len(orchestrator_names)))
    
    for scenario_name, results in all_results.items():
        accuracies = [f"{results[o]['overall_accuracy']:.3f}" for o in orchestrator_names]
        print("{:<30} {}".format(scenario_name, " | ".join(f"{a:^15}" for a in accuracies)))
    
    # Visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    # Plot comparison across scenarios
    print("\n1. Creating performance comparison plot...")
    fig1 = plot_results(all_results, output_path='output/orchestrator_comparison.png')
    print("   Saved to: output/orchestrator_comparison.png")
    
    # Plot learning curves for varying scenario
    print("\n2. Creating learning curves (Varying scenario)...")
    varying_agents = create_agents_varying(M)
    fig2 = plot_learning_curves(
        varying_agents, 
        tasks, 
        window_size=50,
        output_path='output/learning_curves.png'
    )
    print("   Saved to: output/learning_curves.png")
    
    print("\n" + "="*80)
    print("Experiments complete!")
    print("="*80)
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 80)
    
    for scenario_name, results in all_results.items():
        app = results[orchestrator_names[0]]['appropriateness']
        oracle_acc = results['OracleOrchestrator']['overall_accuracy']
        random_acc = results['RandomOrchestrator']['overall_accuracy']
        paper_acc = results['PaperOrchestrator']['overall_accuracy']
        
        improvement = (paper_acc - random_acc) / random_acc * 100 if random_acc > 0 else 0
        oracle_gap = (oracle_acc - paper_acc) / oracle_acc * 100 if oracle_acc > 0 else 0
        
        print(f"\n{scenario_name}:")
        print(f"  Appropriateness: {app:.3f}")
        print(f"  Random baseline: {random_acc:.3f}")
        print(f"  Paper method: {paper_acc:.3f} ({improvement:+.1f}% vs random)")
        print(f"  Oracle: {oracle_acc:.3f} (gap: {oracle_gap:.1f}%)")
    
    return all_results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run experiments
    results = main()
