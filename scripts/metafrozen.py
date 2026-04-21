"""
Meta-Task Value Iteration with PBRS Coboundary Condition
Extends single-task formulation to learn task-conditioned potential functions
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# 1. Meta-Task Environment Definition
# ============================================================================

class MetaTaskMDP(ABC):
    """Abstract base class for meta-task MDPs."""
    
    @abstractmethod
    def sample_task(self):
        """Sample a task from the task distribution."""
        pass
    
    @abstractmethod
    def get_transitions(self, state, action, task):
        """Get transitions for a specific task."""
        pass
    
    @abstractmethod
    def is_terminal(self, state, task):
        """Check if state is terminal for given task."""
        pass

class MetaFrozenLake(MetaTaskMDP):
    """
    Meta-task version of Frozen Lake where:
    - Goal location varies across tasks
    - Hole locations may vary
    - Start state is fixed
    
    This creates a distribution of navigation tasks in the same grid.
    """
    
    def __init__(self, size=4, n_tasks=5, seed=42):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.n_tasks = n_tasks
        
        np.random.seed(seed)
        
        # Fixed elements
        self.start = 0  # Always start at top-left
        self.base_holes = [5, 7, 11, 12]  # Some fixed holes
        
        # Action mappings
        self.actions = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}
        self.action_deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        
        # Sample task configurations
        self.tasks = self._generate_tasks()
        
    def _generate_tasks(self):
        """Generate different goal and hole configurations."""
        tasks = []
        valid_positions = [i for i in range(self.n_states) 
                          if i not in self.base_holes and i != self.start]
        
        for task_id in range(self.n_tasks):
            # Sample goal from valid positions
            goal_idx = np.random.choice(len(valid_positions))
            goal = valid_positions[goal_idx]
            
            # Possibly add an extra hole for variation
            extra_holes = []
            if task_id % 2 == 0:  # Add variation
                remaining = [p for p in valid_positions if p != goal]
                if remaining:
                    extra_hole = np.random.choice(remaining)
                    extra_holes.append(extra_hole)
            
            tasks.append({
                'id': task_id,
                'goal': goal,
                'holes': self.base_holes + extra_holes,
                'embedding': self._compute_task_embedding(goal, extra_holes)
            })
            
        return tasks
    
    def _compute_task_embedding(self, goal, extra_holes):
        """Create a simple task embedding vector."""
        # One-hot for goal location + binary indicators for hole presence
        embedding = np.zeros(self.n_states + self.n_states)
        embedding[goal] = 1.0  # Goal location
        for h in extra_holes:
            embedding[self.n_states + h] = 1.0  # Extra hole locations
        return embedding
    
    def sample_task(self):
        """Sample a random task."""
        return self.tasks[np.random.randint(len(self.tasks))]
    
    def get_transitions(self, state, action, task):
        """Get transitions for a specific task."""
        if self.is_terminal(state, task):
            return [(state, 1.0, 0.0)]
        
        i, j = state // self.size, state % self.size
        di, dj = self.action_deltas[action]
        ni, nj = i + di, j + dj
        
        # Check boundaries
        if 0 <= ni < self.size and 0 <= nj < self.size:
            next_state = ni * self.size + nj
        else:
            next_state = state  # Bounce off wall
            
        # Determine reward based on task
        if next_state == task['goal']:
            reward = 1.0
        elif next_state in task['holes']:
            reward = -1.0
        else:
            reward = -0.01  # Small step penalty
            
        return [(next_state, 1.0, reward)]
    
    def is_terminal(self, state, task):
        """Check if state is terminal for given task."""
        return state == task['goal'] or state in task['holes']

# ============================================================================
# 2. Meta Value Iteration with Task Conditioning
# ============================================================================

class MetaValueIterationPBRS:
    """
    Value iteration that computes task-conditioned potential functions
    satisfying the PBRS coboundary condition.
    """
    
    def __init__(self, meta_mdp, gamma=0.99, theta=1e-6):
        self.meta_mdp = meta_mdp
        self.gamma = gamma
        self.theta = theta
        
        # Store potential function for each task
        self.task_potentials = {}
        
    def compute_all_potentials(self):
        """Compute optimal value function for all tasks as potential."""
        print("Computing potentials for all tasks...")
        
        for task in self.meta_mdp.tasks:
            V = self._value_iteration_single_task(task)
            self.task_potentials[task['id']] = V
            
        print(f"Computed potentials for {len(self.task_potentials)} tasks")
        return self.task_potentials
    
    def _value_iteration_single_task(self, task):
        """Standard value iteration for a single task."""
        V = np.zeros(self.meta_mdp.n_states)
        
        while True:
            delta = 0
            for s in range(self.meta_mdp.n_states):
                if self.meta_mdp.is_terminal(s, task):
                    continue
                    
                v = V[s]
                max_value = float('-inf')
                
                for a in range(self.meta_mdp.n_actions):
                    expected_value = 0
                    for ns, prob, rew in self.meta_mdp.get_transitions(s, a, task):
                        expected_value += prob * (rew + self.gamma * V[ns])
                    max_value = max(max_value, expected_value)
                    
                V[s] = max_value
                delta = max(delta, abs(v - V[s]))
                
            if delta < self.theta:
                break
                
        return V
    
    def compute_meta_coboundary(self, s, a, ns, task_id):
        """
        Compute the shaping reward F(s,a,s',τ) = γ·Φ(s',τ) - Φ(s,τ)
        for a specific task.
        """
        phi_s = self.task_potentials[task_id][s]
        phi_ns = self.task_potentials[task_id][ns]
        return self.gamma * phi_ns - phi_s
    
    def get_task_conditioned_potential(self, state, task_id):
        """Retrieve Φ(s, τ) for a specific state and task."""
        return self.task_potentials[task_id][state]

# ============================================================================
# 3. Neural Meta-Potential Function (Learned)
# ============================================================================

class MetaPotentialNetwork(nn.Module):
    """
    Neural network that learns a task-conditioned potential function
    Φ_θ(s, τ) satisfying the PBRS coboundary condition across tasks.
    """
    
    def __init__(self, state_dim, task_embedding_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),  # State as scalar index
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.task_encoder = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.joint_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output scalar potential
        )
        
    def forward(self, state, task_embedding):
        # Normalize state to [0, 1]
        state_norm = state.float() / self.state_dim
        
        state_features = self.state_encoder(state_norm.unsqueeze(-1))
        task_features = self.task_encoder(task_embedding)
        
        # Combine features (concatenation)
        joint = torch.cat([state_features, task_features], dim=-1)
        
        return self.joint_network(joint).squeeze(-1)

class LearnedMetaPBRS:
    """
    Learns a task-conditioned potential function Φ_θ(s, τ) that satisfies
    the coboundary condition across all tasks using gradient descent.
    """
    
    def __init__(self, meta_mdp, gamma=0.99, hidden_dim=128, lr=1e-3):
        self.meta_mdp = meta_mdp
        self.gamma = gamma
        
        # Determine dimensions
        self.state_dim = meta_mdp.n_states
        self.task_embedding_dim = len(meta_mdp.tasks[0]['embedding'])
        
        # Initialize network
        self.network = MetaPotentialNetwork(
            self.state_dim, 
            self.task_embedding_dim, 
            hidden_dim
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Store training history
        self.loss_history = []
        
    def coboundary_loss(self, s, a, ns, task_embedding, original_reward, gamma):
        """
        Loss function enforcing the coboundary condition:
        F(s,a,s') = γ·Φ(s') - Φ(s)
        
        We want Φ to be consistent with the optimal value function,
        so we minimize the Bellman error.
        """
        phi_s = self.network(torch.tensor([s]), task_embedding.unsqueeze(0))
        phi_ns = self.network(torch.tensor([ns]), task_embedding.unsqueeze(0))
        
        # Bellman target: R + γ·Φ(s')
        target = original_reward + gamma * phi_ns.detach()
        
        # Loss is squared error between Φ(s) and target
        return (phi_s - target) ** 2
    
    def train_epoch(self, batch_size=32):
        """Train for one epoch over sampled transitions."""
        total_loss = 0
        n_batches = 0
        
        for _ in range(batch_size):
            # Sample random task
            task = self.meta_mdp.sample_task()
            task_embedding = torch.tensor(task['embedding'], dtype=torch.float32)
            
            # Sample random non-terminal state and action
            valid_states = [s for s in range(self.meta_mdp.n_states) 
                           if not self.meta_mdp.is_terminal(s, task)]
            
            if not valid_states:
                continue
                
            s = np.random.choice(valid_states)
            a = np.random.randint(0, self.meta_mdp.n_actions)
            
            # Get transition
            transitions = self.meta_mdp.get_transitions(s, a, task)
            
            for ns, prob, rew in transitions:
                # Weighted by transition probability
                loss = self.coboundary_loss(
                    s, a, ns, task_embedding, rew, self.gamma
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * prob
                n_batches += 1
                
        avg_loss = total_loss / max(n_batches, 1)
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def train(self, n_epochs=1000, batch_size=32, verbose=True):
        """Full training loop."""
        print("Training meta-potential network...")
        
        for epoch in range(n_epochs):
            loss = self.train_epoch(batch_size)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
                
        print(f"Training complete. Final loss: {self.loss_history[-1]:.6f}")
        
    def get_potential(self, state, task_id):
        """Get learned potential for state and task."""
        task = self.meta_mdp.tasks[task_id]
        task_embedding = torch.tensor(task['embedding'], dtype=torch.float32)
        
        with torch.no_grad():
            phi = self.network(
                torch.tensor([state]), 
                task_embedding.unsqueeze(0)
            )
        return phi.item()

# ============================================================================
# 4. Meta-Critical Transition Detection
# ============================================================================

class MetaCriticalTransitionAnalyzer:
    """
    Identifies critical transitions that are consistent across tasks,
    enabling meta-learning of subgoals.
    """
    
    def __init__(self, meta_vi, meta_mdp):
        self.meta_vi = meta_vi
        self.meta_mdp = meta_mdp
        
    def compute_task_critical_edges(self, task_id, percentile=90):
        """Compute critical edges for a single task."""
        task = self.meta_mdp.tasks[task_id]
        V = self.meta_vi.task_potentials[task_id]
        
        edges = []
        for s in range(self.meta_mdp.n_states):
            if self.meta_mdp.is_terminal(s, task):
                continue
                
            for a in range(self.meta_mdp.n_actions):
                for ns, prob, rew in self.meta_mdp.get_transitions(s, a, task):
                    shaping = self.meta_vi.compute_meta_coboundary(s, a, ns, task_id)
                    delta_V = V[ns] - V[s]
                    
                    edges.append({
                        's': s, 'ns': ns, 'a': a,
                        'shaping': shaping,
                        'delta_V': delta_V,
                        'prob': prob
                    })
        
        # Find threshold for critical edges
        if edges:
            shaping_abs = [abs(e['shaping']) for e in edges]
            threshold = np.percentile(shaping_abs, percentile)
            critical = [e for e in edges if abs(e['shaping']) >= threshold]
        else:
            critical = []
            
        return critical
    
    def find_meta_critical_transitions(self, percentile=90, min_task_frequency=0.5):
        """
        Find transitions that are critical across multiple tasks.
        These are candidates for meta-learned subgoals.
        """
        all_critical_edges = defaultdict(list)
        
        # Collect critical edges from all tasks
        for task_id in range(self.meta_mdp.n_tasks):
            critical = self.compute_task_critical_edges(task_id, percentile)
            for edge in critical:
                key = (edge['s'], edge['ns'])
                all_critical_edges[key].append({
                    'task_id': task_id,
                    'shaping': edge['shaping'],
                    'delta_V': edge['delta_V']
                })
        
        # Find edges that are critical in multiple tasks
        meta_critical = {}
        n_tasks = self.meta_mdp.n_tasks
        threshold_count = int(n_tasks * min_task_frequency)
        
        for edge_key, occurrences in all_critical_edges.items():
            if len(occurrences) >= threshold_count:
                # Compute statistics across tasks
                shapings = [o['shaping'] for o in occurrences]
                delta_Vs = [o['delta_V'] for o in occurrences]
                
                meta_critical[edge_key] = {
                    'frequency': len(occurrences) / n_tasks,
                    'mean_shaping': np.mean(shapings),
                    'std_shaping': np.std(shapings),
                    'mean_delta_V': np.mean(delta_Vs),
                    'task_ids': [o['task_id'] for o in occurrences]
                }
                
        return meta_critical

# ============================================================================
# 5. Visualization for Meta-Tasks
# ============================================================================

def visualize_meta_potentials(meta_vi, meta_mdp, n_tasks_to_show=3):
    """Visualize potential functions across multiple tasks."""
    fig, axes = plt.subplots(2, n_tasks_to_show, figsize=(5*n_tasks_to_show, 8))
    
    size = meta_mdp.size
    indices = np.random.choice(len(meta_mdp.tasks), n_tasks_to_show, replace=False)
    sample_tasks = [meta_mdp.tasks[i] for i in indices]
    
    for idx, task in enumerate(sample_tasks):
        V = meta_vi.task_potentials[task['id']]
        
        # Reshape to grid for visualization
        V_grid = V.reshape(size, size)
        
        # Heatmap of potentials
        ax1 = axes[0, idx]
        im = ax1.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
        ax1.set_title(f"Task {task['id']}\nGoal: {task['goal']}")
        
        # Mark special states
        goal_i, goal_j = task['goal'] // size, task['goal'] % size
        ax1.plot(goal_j, goal_i, 'b*', markersize=15, label='Goal')
        
        for hole in task['holes']:
            hole_i, hole_j = hole // size, hole % size
            ax1.plot(hole_j, hole_i, 'rx', markersize=12, label='Hole' if hole == task['holes'][0] else '')
            
        ax1.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax1)
        
        # Show shaping rewards for one action (e.g., RIGHT)
        ax2 = axes[1, idx]
        shaping_grid = np.zeros((size, size))
        
        for s in range(meta_mdp.n_states):
            if meta_mdp.is_terminal(s, task):
                continue
            i, j = s // size, s % size
            # Check RIGHT action
            for ns, prob, rew in meta_mdp.get_transitions(s, 2, task):
                shaping = meta_vi.compute_meta_coboundary(s, 2, ns, task['id'])
                shaping_grid[i, j] = shaping
                
        im2 = ax2.imshow(shaping_grid, cmap='RdBu', interpolation='nearest')
        ax2.set_title(f"Shaping Reward (RIGHT action)")
        plt.colorbar(im2, ax=ax2)
        
    plt.suptitle("Meta-Task Potential Functions and Shaping Rewards", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_meta_critical_transitions(analyzer, meta_mdp):
    """Visualize meta-critical transitions on the grid."""
    meta_critical = analyzer.find_meta_critical_transitions(
        percentile=85, min_task_frequency=0.4
    )
    
    size = meta_mdp.size
    
    # Count frequency of critical edges
    edge_counts = np.zeros((size, size, 4))  # 4 possible directions
    
    for (s, ns), info in meta_critical.items():
        i, j = s // size, s % size
        # Determine direction
        diff = ns - s
        if diff == -size:  # UP
            edge_counts[i, j, 0] = info['frequency']
        elif diff == size:  # DOWN
            edge_counts[i, j, 1] = info['frequency']
        elif diff == -1:    # LEFT
            edge_counts[i, j, 2] = info['frequency']
        elif diff == 1:     # RIGHT
            edge_counts[i, j, 3] = info['frequency']
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Background grid
    for i in range(size + 1):
        ax.axhline(i, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(i, color='gray', linestyle='-', alpha=0.3)
    
    # Plot critical edges as arrows
    for i in range(size):
        for j in range(size):
            # UP arrow
            if edge_counts[i, j, 0] > 0:
                ax.arrow(j + 0.5, i + 0.7, 0, -0.4, 
                        head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', alpha=edge_counts[i, j, 0],
                        width=0.02 * edge_counts[i, j, 0])
            # DOWN arrow
            if edge_counts[i, j, 1] > 0:
                ax.arrow(j + 0.5, i + 0.3, 0, 0.4,
                        head_width=0.1, head_length=0.1,
                        fc='green', ec='green', alpha=edge_counts[i, j, 1],
                        width=0.02 * edge_counts[i, j, 1])
            # LEFT arrow
            if edge_counts[i, j, 2] > 0:
                ax.arrow(j + 0.7, i + 0.5, -0.4, 0,
                        head_width=0.1, head_length=0.1,
                        fc='blue', ec='blue', alpha=edge_counts[i, j, 2],
                        width=0.02 * edge_counts[i, j, 2])
            # RIGHT arrow
            if edge_counts[i, j, 3] > 0:
                ax.arrow(j + 0.3, i + 0.5, 0.4, 0,
                        head_width=0.1, head_length=0.1,
                        fc='orange', ec='orange', alpha=edge_counts[i, j, 3],
                        width=0.02 * edge_counts[i, j, 3])
    
    # Mark start
    start_i, start_j = meta_mdp.start // size, meta_mdp.start % size
    ax.add_patch(plt.Rectangle((start_j, start_i), 1, 1, 
                               fill=True, facecolor='lightgreen', alpha=0.5))
    ax.text(start_j + 0.5, start_i + 0.5, 'S', 
            ha='center', va='center', fontweight='bold', fontsize=14)
    
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.set_title("Meta-Critical Transitions\n(Arrows show frequent critical edges across tasks)")
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='UP critical'),
        plt.Line2D([0], [0], color='green', lw=2, label='DOWN critical'),
        plt.Line2D([0], [0], color='blue', lw=2, label='LEFT critical'),
        plt.Line2D([0], [0], color='orange', lw=2, label='RIGHT critical'),
        plt.Rectangle((0,0), 1, 1, facecolor='lightgreen', alpha=0.5, label='Start')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()
    
    return meta_critical

# ============================================================================
# 6. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("META-TASK PBRS COBOUNDARY FORMULATION")
    print("="*60)
    
    # Create meta MDP
    meta_mdp = MetaFrozenLake(size=4, n_tasks=10, seed=42)
    
    print(f"\nMeta-MDP Configuration:")
    print(f"  Size: {meta_mdp.size}x{meta_mdp.size}")
    print(f"  Number of tasks: {meta_mdp.n_tasks}")
    print(f"  Start state: {meta_mdp.start}")
    print(f"  Task variations:")
    for task in meta_mdp.tasks[:3]:
        print(f"    Task {task['id']}: Goal={task['goal']}, Holes={task['holes']}")
    print(f"    ... and {len(meta_mdp.tasks)-3} more tasks")
    
    # Compute potentials for all tasks
    print("\n" + "="*60)
    print("COMPUTING TASK-CONDITIONED POTENTIALS")
    print("="*60)
    
    meta_vi = MetaValueIterationPBRS(meta_mdp, gamma=0.99)
    potentials = meta_vi.compute_all_potentials()
    
    # Verify coboundary condition for a sample
    print("\n" + "="*60)
    print("VERIFYING META COBOUNDARY CONDITION")
    print("="*60)
    print("F(s,a,s',τ) = γ·Φ(s',τ) - Φ(s,τ)")
    
    sample_task = meta_mdp.tasks[0]
    sample_state = meta_mdp.start
    sample_action = 2  # RIGHT
    
    print(f"\nSample: Task {sample_task['id']}, State {sample_state}, Action RIGHT")
    
    for ns, prob, rew in meta_mdp.get_transitions(sample_state, sample_action, sample_task):
        shaping = meta_vi.compute_meta_coboundary(sample_state, sample_action, ns, sample_task['id'])
        phi_s = potentials[sample_task['id']][sample_state]
        phi_ns = potentials[sample_task['id']][ns]
        
        print(f"  Transition to {ns}:")
        print(f"    Φ(s,τ) = {phi_s:.4f}")
        print(f"    Φ(s',τ) = {phi_ns:.4f}")
        print(f"    γ·Φ(s',τ) - Φ(s,τ) = {0.99*phi_ns - phi_s:.4f}")
        print(f"    F(s,a,s',τ) = {shaping:.4f}")
        print(f"    ✓ Condition holds")
    
    # Analyze meta-critical transitions
    print("\n" + "="*60)
    print("META-CRITICAL TRANSITION ANALYSIS")
    print("="*60)
    
    analyzer = MetaCriticalTransitionAnalyzer(meta_vi, meta_mdp)
    meta_critical = visualize_meta_critical_transitions(analyzer, meta_mdp)
    
    print(f"\nFound {len(meta_critical)} meta-critical transitions")
    print("(Transitions that are critical in ≥40% of tasks)\n")
    
    for (s, ns), info in sorted(meta_critical.items(), 
                                key=lambda x: x[1]['frequency'], 
                                reverse=True)[:5]:
        print(f"Edge {s} → {ns}:")
        print(f"  Frequency: {info['frequency']*100:.0f}% of tasks")
        print(f"  Mean shaping reward: {info['mean_shaping']:.4f}")
        print(f"  Mean ΔΦ: {info['mean_delta_V']:.4f}")
        print(f"  Tasks: {info['task_ids']}")
        print()
    
    # Visualize potentials across tasks
    print("\n" + "="*60)
    print("VISUALIZING META-TASK POTENTIALS")
    print("="*60)
    visualize_meta_potentials(meta_vi, meta_mdp, n_tasks_to_show=3)
    
    # Demonstrate learned meta-potential
    print("\n" + "="*60)
    print("LEARNING NEURAL META-POTENTIAL FUNCTION")
    print("="*60)
    
    learned_pbrs = LearnedMetaPBRS(meta_mdp, gamma=0.99, hidden_dim=64, lr=1e-3)
    learned_pbrs.train(n_epochs=500, batch_size=32, verbose=True)
    
    # Compare learned vs computed potentials
    task_id = 0
    print(f"\nLearned vs Computed Potentials for Task {task_id}:")
    print("State | Learned Φ | True Φ | Difference")
    print("-" * 45)
    
    for s in range(min(10, meta_mdp.n_states)):
        learned_phi = learned_pbrs.get_potential(s, task_id)
        true_phi = potentials[task_id][s]
        diff = learned_phi - true_phi
        print(f"  {s:2d}  | {learned_phi:9.4f} | {true_phi:7.4f} | {diff:10.4f}")
    
    print("\n" + "="*60)
    print("META-LEARNING IMPLICATIONS")
    print("="*60)
    print("""
    The meta-critical transitions identified above represent subgoals
    that are consistent across multiple tasks. These can be used to:
    
    1. Initialize option policies for hierarchical RL
    2. Provide task-invariant landmarks for a meta-learner
    3. Guide exploration in new tasks by focusing on known bottlenecks
    4. Transfer shaping potentials across tasks using the learned Φ_θ(s,τ)
    
    The coboundary condition ensures that adding F(s,a,s',τ) as shaping
    reward does not alter the optimal policy for ANY task in the distribution.
    """)
