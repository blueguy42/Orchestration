import numpy as np

class OrchestrationFramework:
    def __init__(self, num_agents, num_regions, alpha_prior=1.0, cost_matrix=None):
        """
        Initializes the orchestration framework with Dirichlet and Beta-Binomial priors.
        [1, 5, 6]
        """
        self.K = num_agents
        self.M = num_regions
        
        # Region probability estimates (Dirichlet) [1, 6]
        # alpha_m represents pseudo-counts for observing region Rm
        self.alpha_regions = np.full(self.M, alpha_prior)
        self.n_regions = np.zeros(self.M)
        
        # Agent correctness per region (Beta-Binomial) [2, 7]
        # alpha0: incorrect pseudo-count, alpha1: correct pseudo-count
        self.alpha0 = np.ones((self.K, self.M)) 
        self.alpha1 = np.ones((self.K, self.M))
        self.successes = np.zeros((self.K, self.M))
        self.failures = np.zeros((self.K, self.M))
        
        # Costs per agent per region [3, 4]
        if cost_matrix is not None:
            self.costs = cost_matrix
        else:
            self.costs = np.ones((self.K, self.M))

    def get_region_probs(self):
        """Calculates current estimate of region probabilities wt,m [7]."""
        total_counts = np.sum(self.n_regions + self.alpha_regions)
        return (self.n_regions + self.alpha_regions - 1) / (total_counts - self.M)

    def get_agent_correctness(self, k, m):
        """Calculates current estimate of agent k's correctness in region m (ct,km) [2]."""
        num = self.successes[k, m] + self.alpha1[k, m] - 1
        den = (self.successes[k, m] + self.failures[k, m]) + (self.alpha0[k, m] + self.alpha1[k, m]) - 2
        return num / den if den > 0 else 0.5

    def compute_utility(self, k, current_region_idx):
        """
        Computes the total empirical utility (lookahead correctness / cost) [3, 4, 8].
        """
        w = self.get_region_probs()
        # Correctness at current step t
        c_t = self.get_agent_correctness(k, current_region_idx)
        
        # Future long-running correctness (sum of w_m * c_km) [3, 8]
        future_correctness = sum(w[m] * self.get_agent_correctness(k, m) for m in range(self.M))
        
        # Onward utility [3, 9]
        onward_utility = c_t * future_correctness
        
        # Final utility adjusted by cost [4]
        return onward_utility / self.costs[k, current_region_idx]

    def select_agent(self, current_region_idx, feasible_mask=None):
        """Selects the agent with the highest utility among feasible agents [9, 10]."""
        utilities = [self.compute_utility(k, current_region_idx) for k in range(self.K)]
        
        if feasible_mask is not None:
            # Apply constraints: set utility to -inf for infeasible agents [10]
            utilities = [u if feasible_mask[k] else -np.inf for k, u in enumerate(utilities)]
            
        return np.argmax(utilities)

    def update_performance(self, agent_idx, region_idx, is_correct):
        """Updates the posterior based on observed performance [2, 11]."""
        self.n_regions[region_idx] += 1
        if is_correct:
            self.successes[agent_idx, region_idx] += 1
        else:
            self.failures[agent_idx, region_idx] += 1

class Teacher:
    """
    Strategically selects tasks (regions) to evaluate to reduce uncertainty [12, 13].
    """
    def select_task_region(self, strategy="random"):
        # Placeholder for strategies to reduce the "jagged capability frontier" [12, 14]
        if strategy == "random":
            return np.random.randint(0, 3) 
        # Future: Implement strategies like Uncertainty Sampling or Expected Information Gain [12]
        return 0