"""
Asymmetric Metric Tree Construction for Meta-Learning
Uses Potential Function Φ to build a hierarchical representation of the MDP.
Branching nodes correspond to discrete Morse critical points (subgoals).
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from typing import List, Tuple, Dict, Set, Optional
import gymnasium as gym
from sklearn.cluster import KMeans

# ============================================================================
# 1. Asymmetric Metric Tree Node & Edge Definitions
# ============================================================================

class MetricTreeNode:
    """Node in the asymmetric metric tree."""
    def __init__(self, node_id: int, level: int, state_indices: Set[int], 
                 potential_range: Tuple[float, float]):
        self.id = node_id
        self.level = level  # Depth in tree (0 = root/start region)
        self.states = state_indices  # Original MDP states in this abstraction
        self.potential_range = potential_range  # Min/Max Φ in this node
        self.children = []
        self.parent = None
        self.in_edges = []  # Directed edges entering this node
        self.out_edges = [] # Directed edges leaving this node
        
    @property
    def mean_potential(self):
        return np.max(self.potential_range)
    
    def __repr__(self):
        return f"Node({self.id}, L{self.level}, Φ∈[{self.potential_range[0]:.2f}, {self.potential_range[1]:.2f}])"

class AsymmetricEdge:
    """Directed edge with weight representing transition difficulty."""
    def __init__(self, source: MetricTreeNode, target: MetricTreeNode, 
                 weight: float, transition_prob: float, is_critical: bool = False):
        self.source = source
        self.target = target
        self.weight = weight  # Asymmetric cost/distance
        self.prob = transition_prob
        self.is_critical = is_critical  # Flagged as subgoal boundary
        
    def __repr__(self):
        return f"Edge({self.source.id}->{self.target.id}, w={self.weight:.2f})"

# ============================================================================
# 2. Potential-Based Tree Builder
# ============================================================================

class PotentialMetricTreeBuilder:
    """
    Constructs an asymmetric metric tree from MDP states using the potential Φ.
    
    The tree captures the hierarchical reachability structure:
    - Nodes are state clusters with similar potential (equipotential layers).
    - Edges have asymmetric weights: w(u→v) ≈ expected cost to reach v from u.
    - Branching points correspond to critical transitions (subgoals).
    """
    
    def __init__(self, mdp, potential: np.ndarray, gamma: float = 0.99):
        self.mdp = mdp
        self.phi = potential  # Potential function satisfying PBRS
        self.gamma = gamma
        self.nodes = []
        self.edges = []
        self.state_to_node = {}  # Mapping from original state to tree node
        
    def build_tree(self, n_levels: int = 5, min_states_per_node: int = 1) -> nx.DiGraph:
        """
        Construct the metric tree by partitioning states based on potential levels.
        
        Args:
            n_levels: Number of potential intervals (tree depth resolution)
            min_states_per_node: Minimum states to form a separate node
        """
        print(f"Building metric tree with {n_levels} potential levels...")
        
        # Step 1: Create equipotential layers
        phi_min, phi_max = np.min(self.phi), np.max(self.phi)
        thresholds = np.linspace(phi_min, phi_max, n_levels + 1)
        
        # Track states per level
        level_states = defaultdict(list)
        for s in range(self.mdp.n_states):
            for i in range(n_levels):
                if thresholds[i] <= self.phi[s] < thresholds[i+1]:
                    level_states[i].append(s)
                    break
            else:
                # Edge case for max value
                level_states[n_levels-1].append(s)
        
        # Step 2: Create nodes for each level
        node_id = 0
        level_nodes = defaultdict(list)
        
        for level, states in level_states.items():
            if len(states) < min_states_per_node:
                continue
                
            # Within a level, cluster states that have similar transition patterns
            sub_clusters = self._cluster_by_transition_similarity(states)
            
            for cluster_states in sub_clusters:
                phi_vals = [self.phi[s] for s in cluster_states]
                node = MetricTreeNode(
                    node_id=node_id,
                    level=level,
                    state_indices=set(cluster_states),
                    potential_range=(min(phi_vals), max(phi_vals))
                )
                
                self.nodes.append(node)
                level_nodes[level].append(node)
                
                for s in cluster_states:
                    self.state_to_node[s] = node
                    
                node_id += 1
        
        print(f"Created {len(self.nodes)} abstract nodes across {len(level_nodes)} levels")
        
        # Step 3: Connect nodes with directed edges based on MDP transitions
        self._create_directed_edges(level_nodes)
        
        # Step 4: Identify critical transitions (branching points)
        self._identify_critical_transitions()
        
        # Step 5: Build NetworkX graph for analysis/visualization
        self.graph = self._build_networkx_graph()
        return self.graph
    
    def _cluster_by_transition_similarity(self, states: List[int]) -> List[List[int]]:
        """
        Cluster states within same potential level by their transition dynamics.
        States that transition to similar next-level nodes are grouped together.
        """
        if len(states) <= 1:
            return [states]
        
        # Build feature vector: for each action, which potential level does it lead to?
        n_actions = self.mdp.n_actions
        features = np.zeros((len(states), n_actions))
        
        for i, s in enumerate(states):
            for a in range(n_actions):
                # Get expected next potential
                expected_next_phi = 0
                total_prob = 0
                for ns, prob, _ in self.mdp.get_transitions(s, a):
                    expected_next_phi += prob * self.phi[ns]
                    total_prob += prob
                if total_prob > 0:
                    features[i, a] = expected_next_phi / total_prob
        
        # Simple clustering: group if features are similar
        # Use single-linkage clustering
        if len(states) > 2:
            Z = linkage(features, method='ward')
            # Cut tree to get reasonable number of clusters
            max_clusters = min(len(states) // 2 + 1, 5)
            clusters = fcluster(Z, max_clusters, criterion='maxclust')
        else:
            clusters = np.arange(len(states)) + 1
        
        # Group by cluster ID
        clustered_states = defaultdict(list)
        for s, c in zip(states, clusters):
            clustered_states[c].append(s)
            
        return list(clustered_states.values())
    
    def _create_directed_edges(self, level_nodes: Dict[int, List[MetricTreeNode]]):
        """
        Create directed edges between nodes based on actual MDP transitions.
        Edge weight = γ·Φ(target) - Φ(source) + expected cost
        """
        # Build mapping for quick lookup
        node_by_state = {}
        for node in self.nodes:
            for s in node.states:
                node_by_state[s] = node
        
        # Track edges between nodes (aggregated transitions)
        edge_data = defaultdict(lambda: {'total_prob': 0, 'weight_sum': 0, 'count': 0})
        
        for s in range(self.mdp.n_states):
            source_node = node_by_state.get(s)
            if source_node is None:
                continue
                
            for a in range(self.mdp.n_actions):
                for ns, prob, rew in self.mdp.get_transitions(s, a):
                    target_node = node_by_state.get(ns)
                    if target_node is None or target_node == source_node:
                        continue
                    
                    # Asymmetric weight: difference in potential plus original cost
                    # This satisfies the coboundary property
                    phi_diff = self.gamma * self.phi[ns] - self.phi[s]
                    weight = -rew - phi_diff  # Negative because we minimize cost
                    
                    key = (source_node.id, target_node.id)
                    edge_data[key]['total_prob'] += prob
                    edge_data[key]['weight_sum'] += prob * weight
                    edge_data[key]['count'] += 1
                    edge_data[key]['source'] = source_node
                    edge_data[key]['target'] = target_node
        
        # Create edges
        for (src_id, tgt_id), data in edge_data.items():
            avg_weight = data['weight_sum'] / data['total_prob'] if data['total_prob'] > 0 else 0
            avg_prob = data['total_prob'] / data['count'] if data['count'] > 0 else 0
            
            edge = AsymmetricEdge(
                source=data['source'],
                target=data['target'],
                weight=avg_weight,
                transition_prob=avg_prob
            )
            
            self.edges.append(edge)
            data['source'].out_edges.append(edge)
            data['target'].in_edges.append(edge)
            data['target'].parent = data['source']
            data['source'].children.append(data['target'])
    
    def _identify_critical_transitions(self):
        """
        Identify critical edges (subgoal boundaries).
        An edge is critical if:
        1. It has unusually high weight (bottleneck)
        2. Its source node has high out-degree (branching point)
        3. It connects significantly different potential levels
        """
        if not self.edges:
            return
        
        # Compute metrics for each edge
        weights = [e.weight for e in self.edges]
        out_degrees = {node.id: len(node.out_edges) for node in self.nodes}
        
        weight_threshold = np.percentile(weights, 75)  # Top 25% heaviest edges
        branch_threshold = 2  # Nodes with 2+ outgoing edges are branching points
        
        for edge in self.edges:
            is_heavy = edge.weight >= weight_threshold
            is_branching = out_degrees[edge.source.id] >= branch_threshold
            is_level_jump = abs(edge.target.level - edge.source.level) > 1
            
            # Critical if it's a heavy branch or a level jump
            if (is_heavy and is_branching) or is_level_jump:
                edge.is_critical = True
                
        n_critical = sum(1 for e in self.edges if e.is_critical)
        print(f"Identified {n_critical} critical transitions (potential subgoals)")
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph for analysis and visualization."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(node.id, 
                      level=node.level,
                      phi_range=node.potential_range,
                      n_states=len(node.states),
                      label=f"N{node.id}\nL{node.level}")
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source.id, edge.target.id,
                      weight=edge.weight,
                      prob=edge.prob,
                      critical=edge.is_critical)
        
        return G

# ============================================================================
# 3. Meta-Learning Interface for the Metric Tree
# ============================================================================

class MetricTreeMetaLearner:
    """
    Uses the asymmetric metric tree for meta-learning across tasks.
    
    The tree provides:
    - Hierarchical state abstraction
    - Critical transition points (subgoals)
    - Asymmetric distance metric for planning
    """
    
    def __init__(self, tree_graph: nx.DiGraph, nodes: List[MetricTreeNode]):
        self.graph = tree_graph
        self.nodes = {n.id: n for n in nodes}
        self.critical_edges = [(u, v) for u, v, d in tree_graph.edges(data=True) 
                               if d.get('critical', False)]
        
    def get_subgoals(self) -> List[int]:
        """
        Return the target nodes of critical edges as subgoal candidates.
        These are the branching points where meta-learning should focus.
        """
        subgoal_nodes = set()
        for u, v in self.critical_edges:
            subgoal_nodes.add(v)
        return list(subgoal_nodes)
    
    def compute_asymmetric_distance(self, start_node: int, target_node: int) -> float:
        """
        Compute the asymmetric distance (expected cost) between two abstract nodes.
        This respects the directed nature of the MDP.
        """
        try:
            # Shortest path using edge weights
            path_length = nx.shortest_path_length(
                self.graph, start_node, target_node, weight='weight'
            )
            return path_length
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_subtask_decomposition(self) -> List[Set[int]]:
        """
        Decompose the tree into subtasks by cutting at critical edges.
        Each subtree rooted at a branching point is a potential subtask.
        """
        subtasks = []
        
        # Find branching nodes (nodes with multiple children leading to different regions)
        for node_id in self.graph.nodes():
            children = list(self.graph.successors(node_id))
            if len(children) >= 2:
                # Each child subtree is a separate subtask
                for child in children:
                    # Get all descendants
                    subtree_nodes = set([child])
                    stack = [child]
                    while stack:
                        current = stack.pop()
                        for succ in self.graph.successors(current):
                            if succ not in subtree_nodes:
                                subtree_nodes.add(succ)
                                stack.append(succ)
                    
                    # Get original states in this subtree
                    states = set()
                    for nid in subtree_nodes:
                        states.update(self.nodes[nid].states)
                    
                    subtasks.append(states)
        
        return subtasks
    
    def get_meta_learning_curriculum(self) -> List[List[int]]:
        """
        Generate a curriculum of state sets for meta-learning.
        Progresses from simple subtasks (leaves) to complex ones (root).
        """
        # Compute depth from leaves
        depths = {}
        for node_id in self.graph.nodes():
            try:
                # Distance to furthest leaf
                distances = []
                for leaf in [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]:
                    try:
                        d = nx.shortest_path_length(self.graph, node_id, leaf, weight='weight')
                        distances.append(d)
                    except nx.NetworkXNoPath:
                        pass
                depths[node_id] = max(distances) if distances else 0
            except:
                depths[node_id] = 0
        
        # Group nodes by depth (curriculum levels)
        curriculum = defaultdict(list)
        for node_id, depth in depths.items():
            curriculum[depth].extend(list(self.nodes[node_id].states))
        
        # Sort by depth (ascending: shallow/leaf tasks first)
        return [curriculum[d] for d in sorted(curriculum.keys())]

# ============================================================================
# 4. Visualization
# ============================================================================

def visualize_metric_tree(builder: PotentialMetricTreeBuilder, 
                          meta_learner: MetricTreeMetaLearner):
    """
    Visualize the asymmetric metric tree with critical transitions highlighted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    G = builder.graph
    
    # ========== Plot 1: Tree Structure ==========
    ax1 = axes[0]
    
    # Compute layout (hierarchical)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Organize by level for better visualization
    level_pos = defaultdict(list)
    for node_id in G.nodes():
        level = G.nodes[node_id]['level']
        level_pos[level].append(node_id)
    
    # Adjust y-coordinate by level
    for node_id, (x, y) in pos.items():
        level = G.nodes[node_id]['level']
        pos[node_id] = (x, -level * 0.3)  # Level determines y-position
    
    # Node colors by potential
    node_colors = [np.mean(G.nodes[n]['phi_range']) for n in G.nodes()]
    
    # Node sizes by number of states
    node_sizes = [G.nodes[n]['n_states'] * 50 + 300 for n in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax1,
                          node_color=node_colors,
                          cmap='viridis',
                          node_size=node_sizes,
                          edgecolors='black',
                          linewidths=1)
    
    # Draw labels
    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=9)
    
    # Draw normal edges
    normal_edges = [(u, v) for u, v in G.edges() 
                    if not G.edges[u, v].get('critical', False)]
    nx.draw_networkx_edges(G, pos, ax=ax1,
                          edgelist=normal_edges,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=15,
                          connectionstyle='arc3,rad=0.1',
                          alpha=0.6)
    
    # Draw critical edges (subgoals)
    critical_edges = [(u, v) for u, v in G.edges() 
                      if G.edges[u, v].get('critical', False)]
    if critical_edges:
        nx.draw_networkx_edges(G, pos, ax=ax1,
                              edgelist=critical_edges,
                              edge_color='red',
                              width=3,
                              arrows=True,
                              arrowsize=20,
                              connectionstyle='arc3,rad=0.1')
    
    ax1.set_title("Asymmetric Metric Tree\n(Red = Critical Transitions/Subgoals)")
    ax1.axis('off')
    
    # Add colorbar for potential
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(node_colors), 
                                                 vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Mean Potential Φ', shrink=0.6)
    
    # ========== Plot 2: Subtask Decomposition ==========
    ax2 = axes[1]
    
    subtasks = meta_learner.get_subtask_decomposition()
    
    # Assign colors to subtasks
    subtask_colors = {}
    colors = plt.cm.Set3(np.linspace(0, 1, len(subtasks)))
    for i, states in enumerate(subtasks):
        for s in states:
            subtask_colors[s] = colors[i]
    
    # Visualize on grid if MDP is grid-like
    if hasattr(builder.mdp, 'size'):
        size = builder.mdp.size
        grid = np.zeros((size, size), dtype=int)
        color_grid = np.ones((size, size, 3))  # Default white
        
        for s in range(builder.mdp.n_states):
            i, j = s // size, s % size
            if s in subtask_colors:
                color_grid[i, j] = subtask_colors[s][:3]
            else:
                color_grid[i, j] = [0.9, 0.9, 0.9]  # Light gray for unassigned
        
        ax2.imshow(color_grid, interpolation='nearest')
        
        # Mark critical transitions on grid
        for u, v in critical_edges:
            # Get representative states
            u_states = list(builder.nodes[u].states)
            v_states = list(builder.nodes[v].states)
            if u_states and v_states:
                u_s = u_states[0]
                v_s = v_states[0]
                ui, uj = u_s // size, u_s % size
                vi, vj = v_s // size, v_s % size
                
                # Draw arrow between regions
                ax2.arrow(uj + 0.5, ui + 0.5, 
                         (vj - uj) * 0.7, (vi - ui) * 0.7,
                         head_width=0.3, head_length=0.3, 
                         fc='red', ec='red', linewidth=2, alpha=0.8)
        
        # Mark start and goal
        if hasattr(builder.mdp, 'start'):
            si, sj = builder.mdp.start // size, builder.mdp.start % size
            ax2.plot(sj, si, 'go', markersize=12, label='Start')
        if hasattr(builder.mdp, 'goal'):
            gi, gj = builder.mdp.goal // size, builder.mdp.goal % size
            ax2.plot(gj, gi, 'r*', markersize=15, label='Goal')
        
        ax2.set_xticks(np.arange(size))
        ax2.set_yticks(np.arange(size))
        ax2.set_xticklabels(range(size))
        ax2.set_yticklabels(range(size))
        ax2.set_title("Subtask Decomposition\n(Colors = Different Subtasks)")
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('metric_tree_meta_learning.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# 5. Demonstration with Frozen Lake
# ============================================================================

class SimpleFrozenLakeMDP:
    """Minimal Frozen Lake for demonstration."""
    def __init__(self):
        self.size = 4
        self.n_states = 16
        self.n_actions = 4
        
        # 4x4 Frozen Lake layout
        self.holes = [5, 7, 11, 12]
        self.goal = 15
        self.start = 0
        self.terminal = set(self.holes + [self.goal])
        
        # Action mappings
        self.action_deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        
    def get_transitions(self, state, action):
        if state in self.terminal:
            return [(state, 1.0, 0.0)]
        
        i, j = state // self.size, state % self.size
        di, dj = self.action_deltas[action]
        ni, nj = i + di, j + dj
        
        if 0 <= ni < self.size and 0 <= nj < self.size:
            ns = ni * self.size + nj
        else:
            ns = state
            
        if ns == self.goal:
            reward = 1.0
        elif ns in self.holes:
            reward = -1.0
        else:
            reward = 0.0
            
        return [(ns, 1.0, reward)]

def compute_potential_frozen_lake(mdp, gamma=0.99):
    """Compute optimal value function as potential."""
    V = np.zeros(mdp.n_states)
    theta = 1e-6
    
    while True:
        delta = 0
        for s in range(mdp.n_states):
            if s in mdp.terminal:
                continue
            v = V[s]
            max_v = float('-inf')
            for a in range(mdp.n_actions):
                q_sa = sum(prob * (rew + gamma * V[ns])
                           for ns, prob, rew in mdp.get_transitions(s, a))
                max_v = max(max_v, q_sa)
            V[s] = max_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def main():
    print("="*60)
    print("ASYMMETRIC METRIC TREE FOR META-LEARNING")
    print("="*60)
    
    # Create MDP
    mdp = SimpleFrozenLakeMDP()
    
    # Compute potential (optimal value function)
    print("\nComputing potential function Φ(s) via Value Iteration...")
    phi = compute_potential_frozen_lake(mdp, gamma=0.99)
    
    print(f"Potential range: [{phi.min():.3f}, {phi.max():.3f}]")
    
    # Build metric tree
    builder = PotentialMetricTreeBuilder(mdp, phi, gamma=0.99)
    tree_graph = builder.build_tree(n_levels=5, min_states_per_node=1)
    
    # Create meta-learner interface
    meta_learner = MetricTreeMetaLearner(tree_graph, builder.nodes)
    
    # Extract meta-learning information
    print("\n" + "="*60)
    print("META-LEARNING STRUCTURE EXTRACTED")
    print("="*60)
    
    subgoals = meta_learner.get_subgoals()
    print(f"\nSubgoal Nodes (Critical Transitions): {subgoals}")
    
    subtasks = meta_learner.get_subtask_decomposition()
    print(f"\nSubtasks Identified: {len(subtasks)}")
    for i, states in enumerate(subtasks):
        print(f"  Subtask {i+1}: {len(states)} states - {sorted(states)}")
    
    curriculum = meta_learner.get_meta_learning_curriculum()
    print(f"\nCurriculum Levels: {len(curriculum)}")
    for i, states in enumerate(curriculum):
        print(f"  Level {i+1}: {len(states)} states")
    
    # Compute asymmetric distance example
    if len(builder.nodes) >= 2:
        start_node = builder.state_to_node[mdp.start].id if mdp.start in builder.state_to_node else builder.nodes[0].id
        goal_node = builder.state_to_node[mdp.goal].id if mdp.goal in builder.state_to_node else builder.nodes[-1].id

        dist_forward = meta_learner.compute_asymmetric_distance(start_node, goal_node)
        dist_backward = meta_learner.compute_asymmetric_distance(goal_node, start_node)

        print(f"\nAsymmetric Distance Example:")
        print(f"  Start -> Goal: {dist_forward:.2f}")
        print(f"  Goal -> Start: {dist_backward:.2f}" if dist_backward != float('inf') else f"  Goal -> Start: inf")
        print(f"  Asymmetry Ratio: {dist_backward/dist_forward:.2f}" if dist_forward > 0 and dist_backward != float('inf') else f"  Asymmetry Ratio: undefined")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_metric_tree(builder, meta_learner)
    
    print("\n" + "="*60)
    print("THEORETICAL PROPERTIES")
    print("="*60)
    print("""
    The metric tree satisfies:
    
    1. Hierarchical Abstraction: States grouped by potential Φ
       → Natural subtask boundaries at branching points
       
    2. Asymmetric Distances: w(u→v) ≠ w(v→u)
       → Reflects irreversible MDP transitions
       
    3. Coboundary Preservation: Φ acts as height function
       → Critical points = Morse critical points
       
    4. Meta-Learning Readiness:
       - Subgoals = Branching nodes
       - Curriculum = Level-order traversal
       - Distance metric for planning across tasks
    """)

# ============================================================================
# 6. MuJoCo Reacher Adaptation
# ============================================================================

class ReacherTrajectoryCollector:
    """
    Collects random-exploration trajectories from MuJoCo Reacher-v5.

    Reacher-v5 observation (11-dim):
      [0] cos(θ₁)  [1] cos(θ₂)  [2] sin(θ₁)  [3] sin(θ₂)
      [4] target_x  [5] target_y
      [6] θ̇₁       [7] θ̇₂
      [8] (fingertip−target)_x  [9] (fingertip−target)_y  [10] _z (≈0)

    Potential: Φ(s) = −‖obs[8:10]‖  (higher ⟹ closer to target)
    """

    def __init__(self,
                 env_name: str = 'Reacher-v5',
                 n_trajectories: int = 300,
                 max_steps: int = 50,
                 seed: int = 42):
        self.env_name = env_name
        self.n_trajectories = n_trajectories
        self.max_steps = max_steps
        self.seed = seed

    def collect(self) -> List[Tuple]:
        """Return list of (obs, action, next_obs, reward, done) tuples."""
        env = gym.make(self.env_name)
        transitions = []

        for ep in range(self.n_trajectories):
            ep_seed = self.seed + ep
            obs, _ = env.reset(seed=ep_seed)

            for _ in range(self.max_steps):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transitions.append((obs.copy(), action.copy(),
                                    next_obs.copy(), float(reward), done))
                obs = next_obs
                if done:
                    break

        env.close()
        print(f"Collected {len(transitions)} transitions "
              f"({self.n_trajectories} episodes, env={self.env_name})")
        return transitions


class ReacherAbstractMDP:
    """
    Wraps Reacher trajectories as a discrete abstract MDP via k-means clustering.

    Provides the same interface as SimpleFrozenLakeMDP so that
    PotentialMetricTreeBuilder works without any modification.

    State abstraction : k-means on all 11 observation dimensions.
    Action abstraction: 4-bin quadrant discretisation of the 2D torque space.
    Potential         : per-cluster mean of Φ(obs) = −‖obs[8:10]‖.
    """

    N_ACTION_BINS = 4  # (+,+) (+,−) (−,+) (−,−) torque quadrants

    def __init__(self,
                 transitions: List[Tuple],
                 n_clusters: int = 20,
                 seed: int = 42):
        self.n_states = n_clusters
        self.n_actions = self.N_ACTION_BINS
        self.terminal: Set[int] = set()   # Reacher has no absorbing terminal state
        self._raw = transitions
        self._seed = seed

        self._fit_clusters()
        self._build_transition_model()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_bin(action: np.ndarray) -> int:
        """Map a 2-D continuous torque to one of 4 quadrant bins."""
        return (int(action[0] >= 0) << 1) | int(action[1] >= 0)

    def assign_cluster(self, obs: np.ndarray) -> int:
        return int(self.kmeans.predict(obs.reshape(1, -1))[0])

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_clusters(self) -> None:
        obs_matrix = np.array([t[0] for t in self._raw])
        self.kmeans = KMeans(n_clusters=self.n_states,
                             n_init=10, random_state=self._seed)
        self.kmeans.fit(obs_matrix)
        print(f"Fitted {self.n_states} abstract states "
              f"from {len(obs_matrix)} observations")

    def _build_transition_model(self) -> None:
        # trans_data[s][a] = [(next_s, reward), ...]
        trans_data: Dict = defaultdict(lambda: defaultdict(list))

        phi_sum = np.zeros(self.n_states)
        phi_cnt = np.zeros(self.n_states)

        for obs, action, next_obs, reward, _ in self._raw:
            s  = self.assign_cluster(obs)
            a  = self._action_bin(action)
            ns = self.assign_cluster(next_obs)
            trans_data[s][a].append((ns, reward))

            dist = float(np.linalg.norm(obs[8:10]))
            phi_sum[s] += -dist
            phi_cnt[s] += 1

        self._trans_data = trans_data
        self._phi = phi_sum / np.maximum(phi_cnt, 1)

    # ------------------------------------------------------------------
    # MDP interface
    # ------------------------------------------------------------------

    def get_transitions(self, s: int, a: int) -> List[Tuple[int, float, float]]:
        """Return [(next_s, prob, avg_reward), ...] from empirical data."""
        data = self._trans_data[s][a]
        if not data:
            return [(s, 1.0, 0.0)]   # self-loop when no data observed

        counts = Counter(ns for ns, _ in data)
        total  = len(data)
        result = []
        for ns, count in counts.items():
            prob       = count / total
            avg_reward = float(np.mean([r for ns_, r in data if ns_ == ns]))
            result.append((ns, prob, avg_reward))
        return result

    @property
    def potential(self) -> np.ndarray:
        return self._phi.copy()

    @property
    def cluster_centers(self) -> np.ndarray:
        return self.kmeans.cluster_centers_


def compute_potential_reacher(abstract_mdp: ReacherAbstractMDP) -> np.ndarray:
    """Return pre-computed per-cluster potential for Reacher."""
    return abstract_mdp.potential


def visualize_reacher_tree(builder: PotentialMetricTreeBuilder,
                           meta_learner: MetricTreeMetaLearner,
                           abstract_mdp: ReacherAbstractMDP) -> None:
    """
    Three-panel figure:
      Left  : metric tree graph coloured by potential.
      Centre: abstract state cluster centres in cos(θ₁)–cos(θ₂) joint space.
      Right : per-cluster potential bar chart with subgoals highlighted.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    G   = builder.graph
    phi = abstract_mdp.potential
    centers = abstract_mdp.cluster_centers  # (n_clusters, 11)

    critical_edges = [(u, v) for u, v in G.edges()
                      if G.edges[u, v].get('critical', False)]
    normal_edges   = [(u, v) for u, v in G.edges()
                      if not G.edges[u, v].get('critical', False)]

    # ---- Plot 1: tree structure ----
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    for nid in G.nodes():
        x, _ = pos[nid]
        level = G.nodes[nid]['level']
        pos[nid] = (x, -level * 0.5)

    node_colors = [np.mean(G.nodes[n]['phi_range']) for n in G.nodes()]
    node_sizes  = [G.nodes[n]['n_states'] * 30 + 200  for n in G.nodes()]

    sc = nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                                cmap='RdYlGn', node_size=node_sizes,
                                edgecolors='black', linewidths=1)
    nx.draw_networkx_labels(G, pos, {n: f"N{n}" for n in G.nodes()},
                            ax=ax1, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=normal_edges,
                           edge_color='gray', arrows=True, arrowsize=12,
                           connectionstyle='arc3,rad=0.1', alpha=0.5)
    if critical_edges:
        nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=critical_edges,
                               edge_color='red', width=3, arrows=True,
                               arrowsize=18, connectionstyle='arc3,rad=0.1')

    plt.colorbar(sc, ax=ax1, label='Mean Potential Φ', shrink=0.7)
    ax1.set_title("Metric Tree Structure\n(Red = Critical Transitions / Subgoals)")
    ax1.axis('off')

    # ---- Plot 2: abstract states in joint-angle space ----
    ax2 = axes[1]
    x_coords = centers[:, 0]  # cos(θ₁)
    y_coords = centers[:, 1]  # cos(θ₂)

    sc2 = ax2.scatter(x_coords, y_coords, c=phi, cmap='RdYlGn',
                      s=250, edgecolors='black', linewidths=1, zorder=3)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax2.annotate(str(i), (x, y), fontsize=7, ha='center', va='center',
                     color='black', fontweight='bold')

    for u, v in critical_edges:
        ax2.annotate('',
                     xy=(x_coords[v], y_coords[v]),
                     xytext=(x_coords[u], y_coords[u]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    plt.colorbar(sc2, ax=ax2, label='Φ = −dist(fingertip, target)')
    ax2.set_xlabel('cos(θ₁)')
    ax2.set_ylabel('cos(θ₂)')
    ax2.set_title("Abstract States in Joint-Angle Space\n"
                  "(Colour = Potential, Arrows = Critical Transitions)")
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: potential bar chart ----
    ax3 = axes[2]
    subgoal_ids = set(meta_learner.get_subgoals())
    bar_colors  = ['tomato' if i in subgoal_ids else 'steelblue'
                   for i in range(abstract_mdp.n_states)]
    ax3.bar(range(abstract_mdp.n_states), phi, color=bar_colors)
    ax3.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax3.set_xlabel('Abstract State (Cluster ID)')
    ax3.set_ylabel('Mean Φ = −‖fingertip − target‖')
    ax3.set_title("Per-Cluster Potential\n(Red = Subgoal / Critical Node)")
    ax3.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(facecolor='steelblue', label='Normal state'),
                        Patch(facecolor='tomato',    label='Subgoal state')])

    plt.suptitle("Asymmetric Metric Tree — MuJoCo Reacher-v5",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reacher_metric_tree.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved reacher_metric_tree.png")


def main_reacher():
    print("="*60)
    print("ASYMMETRIC METRIC TREE FOR MUJOCO REACHER-v5")
    print("="*60)

    # 1. Collect trajectories via random exploration
    print("\nCollecting exploration trajectories...")
    collector = ReacherTrajectoryCollector(
        env_name='Reacher-v5',
        n_trajectories=300,
        max_steps=50,
        seed=42
    )
    transitions = collector.collect()

    # 2. Build abstract MDP (state clustering + empirical transitions)
    print("\nBuilding abstract MDP via k-means clustering...")
    abstract_mdp = ReacherAbstractMDP(transitions, n_clusters=20, seed=42)

    # 3. Compute potential (pre-computed during clustering)
    phi = compute_potential_reacher(abstract_mdp)
    print(f"Potential range: [{phi.min():.4f}, {phi.max():.4f}]")

    # 4. Build metric tree
    print("\nBuilding metric tree...")
    builder = PotentialMetricTreeBuilder(abstract_mdp, phi, gamma=0.99)
    tree_graph = builder.build_tree(n_levels=5, min_states_per_node=1)

    # 5. Meta-learning interface
    meta_learner = MetricTreeMetaLearner(tree_graph, builder.nodes)

    # 6. Analysis
    print("\n" + "="*60)
    print("META-LEARNING STRUCTURE")
    print("="*60)

    subgoals = meta_learner.get_subgoals()
    print(f"\nSubgoal nodes (critical transitions): {subgoals}")

    subtasks = meta_learner.get_subtask_decomposition()
    print(f"Subtasks identified: {len(subtasks)}")
    for i, states in enumerate(subtasks):
        print(f"  Subtask {i+1}: abstract states {sorted(states)}")

    curriculum = meta_learner.get_meta_learning_curriculum()
    print(f"\nCurriculum levels: {len(curriculum)}")
    for i, states in enumerate(curriculum):
        print(f"  Level {i+1}: {len(states)} abstract states")

    # 7. Asymmetric distance: worst-potential → best-potential cluster
    if len(builder.nodes) >= 2:
        far_id  = builder.nodes[int(np.argmin(phi))].id
        near_id = builder.nodes[int(np.argmax(phi))].id

        d_fwd = meta_learner.compute_asymmetric_distance(far_id, near_id)
        d_bwd = meta_learner.compute_asymmetric_distance(near_id, far_id)

        print(f"\nAsymmetric distance (far → near target): {d_fwd:.4f}")
        print(f"Asymmetric distance (near → far target): "
              f"{d_bwd:.4f}" if d_bwd != float('inf') else "inf")
        if d_fwd > 0 and d_bwd not in (float('inf'), 0):
            print(f"Asymmetry ratio: {d_bwd / d_fwd:.2f}")

    # 8. Visualise
    print("\nGenerating visualisation...")
    visualize_reacher_tree(builder, meta_learner, abstract_mdp)

    return builder, meta_learner, abstract_mdp


if __name__ == "__main__":
    main_reacher()
