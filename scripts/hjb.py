"""
Critical Transition Discovery for Meta-World via Reachability Embeddings and Poset Construction

This script implements a framework for discovering critical transitions (subgoals) in Meta-World
tasks by learning reachability embeddings and constructing a partial order over abstract states.
The approach is designed for meta-learning scenarios where identifying reusable transition
structure across tasks accelerates adaptation.

Based on concepts from:
- Hamilton-Jacobi reachability analysis for MDPs
- Reachability embeddings for state abstraction
- Option discovery via meta-learned subgoals [citation:7]

Requirements:
    pip install metaworld gymnasium numpy torch matplotlib scikit-learn networkx
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import metaworld
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration and Data Structures
# =============================================================================

@dataclass
class TrajectoryBatch:
    """Container for collected trajectory data."""
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    task_ids: List[int]

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class CriticalTransition:
    """Represents a discovered critical transition (subgoal) in the MDP."""
    state_prototype: np.ndarray       # Representative state for this transition
    descendant_clusters: Set[int]     # Cluster IDs reachable from this node
    predecessor_count: int            # Number of incoming edges in poset
    successor_count: int              # Number of outgoing edges in poset
    task_frequency: Dict[int, float]  # How often this transition appears per task
    importance_score: float           # Overall utility for meta-learning


class ReachabilityEmbedding(nn.Module):
    """
    Learned embedding that captures minimum-cost reachability relationships between states.

    The embedding distance between states approximates the minimum discounted cost
    to transition between them (the valuation Phi(S) concept).
    """

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.gamma = gamma

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Asymmetric predictor captures directed nature of reachability
        self.reachability_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def predict_reachability(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Predict probability that target is reachable from source with low cost."""
        combined = torch.cat([source_embeddings, target_embeddings], dim=-1)
        return self.reachability_predictor(combined)

    def compute_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        reachable_mask: torch.Tensor,
        horizon_weights: torch.Tensor
    ) -> torch.Tensor:
        source_emb = self.encode(states)
        target_emb = self.encode(next_states)
        predictions = self.predict_reachability(source_emb, target_emb).squeeze()
        loss = -horizon_weights * (
            reachable_mask * torch.log(predictions + 1e-8) +
            (1 - reachable_mask) * torch.log(1 - predictions + 1e-8)
        )
        return loss.mean()

    def train_step(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        reachable_mask: torch.Tensor,
        horizon_weights: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, next_states, reachable_mask, horizon_weights)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class ReachabilityPoset:
    """
    Partially ordered set constructed from reachability embeddings.

    Nodes represent abstract states (clusters); directed edges indicate reachability
    with low cost; the partial order defines prerequisite relationships.
    """

    def __init__(
        self,
        embedding_model: ReachabilityEmbedding,
        n_clusters: int = 20,
        reachability_threshold: float = 0.7
    ):
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.reachability_threshold = reachability_threshold

        self.graph = nx.DiGraph()
        self.cluster_centers = None
        self.kmeans = None
        self.state_to_cluster: Dict[int, int] = {}
        self.all_states: Optional[np.ndarray] = None  # stored for prototype lookup

    def build_from_trajectories(
        self,
        all_states: np.ndarray,
        trajectories: List[List[int]]
    ) -> nx.DiGraph:
        """
        Construct the poset by clustering embedded states and establishing
        reachability edges between clusters.
        """
        print("Building reachability poset...")
        self.all_states = all_states

        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                torch.FloatTensor(all_states)
            ).numpy()

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        cluster_labels = self.kmeans.fit_predict(embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_

        for i, label in enumerate(cluster_labels):
            self.state_to_cluster[i] = int(label)

        self.graph.clear()
        for c in range(self.n_clusters):
            self.graph.add_node(c, size=0)
        for label in cluster_labels:
            self.graph.nodes[label]['size'] += 1

        cluster_pairs = self._compute_cluster_reachability(
            embeddings, cluster_labels, trajectories
        )

        edge_weights: Dict[Tuple[int, int], float] = {}
        for c1, c2, weight in cluster_pairs:
            if weight > self.reachability_threshold and c1 != c2:
                edge_weights[(c1, c2)] = weight
                self.graph.add_edge(c1, c2, weight=weight)

        # Transitive reduction drops all attributes; restore node and edge data
        node_sizes = {n: self.graph.nodes[n]['size'] for n in self.graph.nodes()}
        reduced = nx.transitive_reduction(self.graph)
        for n in reduced.nodes():
            reduced.nodes[n]['size'] = node_sizes.get(n, 0)
        for u, v in reduced.edges():
            reduced[u][v]['weight'] = edge_weights.get((u, v), 0.0)
        self.graph = reduced

        print(f"Built poset with {self.n_clusters} nodes and {self.graph.number_of_edges()} edges")
        return self.graph

    def _compute_cluster_reachability(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        trajectories: List[List[int]]
    ) -> List[Tuple[int, int, float]]:
        n_clusters = self.n_clusters
        empirical_counts = np.zeros((n_clusters, n_clusters))
        embedding_scores = np.zeros((n_clusters, n_clusters))

        for traj in trajectories:
            for i in range(len(traj) - 1):
                c_from = cluster_labels[traj[i]]
                c_to = cluster_labels[traj[i + 1]]
                empirical_counts[c_from, c_to] += 1

        # Batch all cluster-pair predictions
        with torch.no_grad():
            centers = torch.FloatTensor(self.cluster_centers)
            for c1 in range(n_clusters):
                src = centers[c1].unsqueeze(0).expand(n_clusters, -1)
                scores = self.embedding_model.predict_reachability(src, centers).squeeze()
                embedding_scores[c1] = scores.numpy()

        empirical_norm = empirical_counts / (empirical_counts.sum(axis=1, keepdims=True) + 1e-8)
        combined_scores = 0.5 * empirical_norm + 0.5 * embedding_scores

        edges = []
        for c1 in range(n_clusters):
            for c2 in range(n_clusters):
                if c1 != c2 and combined_scores[c1, c2] > 0:
                    edges.append((c1, c2, float(combined_scores[c1, c2])))
        return edges

    def get_critical_transitions(self, min_importance: float = 0.1) -> List[CriticalTransition]:
        """
        Identify critical transitions from the poset structure.

        Critical nodes have high betweenness centrality (lie on many paths)
        and serve as prerequisites for many downstream states.
        """
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())

        critical_transitions = []
        for node in self.graph.nodes():
            importance = betweenness[node]
            if importance < min_importance:
                continue

            prototype = self._get_cluster_prototype(node)
            descendant_clusters = set(nx.descendants(self.graph, node))

            ct = CriticalTransition(
                state_prototype=prototype,
                descendant_clusters=descendant_clusters,
                predecessor_count=in_degree[node],
                successor_count=out_degree[node],
                task_frequency={},
                importance_score=importance
            )
            critical_transitions.append(ct)

        return sorted(critical_transitions, key=lambda x: x.importance_score, reverse=True)

    def _get_cluster_prototype(self, cluster_id: int) -> np.ndarray:
        """Return the state in this cluster closest to its embedding centroid."""
        if self.all_states is None:
            raise RuntimeError("build_from_trajectories must be called first")
        indices = [i for i, c in self.state_to_cluster.items() if c == cluster_id]
        if not indices:
            return np.zeros(self.all_states.shape[1])
        cluster_states = self.all_states[indices]
        with torch.no_grad():
            embs = self.embedding_model.encode(torch.FloatTensor(cluster_states)).numpy()
        center = self.cluster_centers[cluster_id]
        prototype_idx = int(np.argmin(np.linalg.norm(embs - center, axis=1)))
        return cluster_states[prototype_idx]

    def get_state_indices_for_clusters(self, cluster_ids: Set[int]) -> List[int]:
        """Return all state indices belonging to any of the given cluster IDs."""
        return [i for i, c in self.state_to_cluster.items() if c in cluster_ids]

    def visualize(self, save_path: str = "poset_visualization.png"):
        """Visualize the reachability poset."""
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        node_sizes = [self.graph.nodes[n]['size'] * 50 for n in self.graph.nodes()]
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        node_colors = [betweenness[n] for n in self.graph.nodes()]

        sc = nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='viridis',
            alpha=0.8
        )

        edge_weights = [
            self.graph[u][v].get('weight', 1.0) * 2
            for u, v in self.graph.edges()
        ]
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            width=edge_weights,
            alpha=0.5,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            arrowstyle='->'
        )

        labels = {n: f"C{n}" for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, ax=ax)

        plt.colorbar(sc, ax=ax, label='Betweenness Centrality')
        ax.set_title("Reachability Poset: Critical Transition Structure", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Poset visualization saved to {save_path}")


class MetaWorldCriticalTransitionLearner:
    """
    Main class for discovering critical transitions in Meta-World environments.

    Orchestrates:
    1. Data collection across multiple Meta-World tasks
    2. Reachability embedding training
    3. Poset construction and critical transition identification
    4. Analysis of discovered structure for meta-learning
    """

    def __init__(
        self,
        env_names: List[str],
        embedding_dim: int = 64,
        n_clusters: int = 20,
        gamma: float = 0.99,
        trajectories_per_task: int = 100,
        max_steps_per_trajectory: int = 200
    ):
        self.env_names = env_names
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.trajectories_per_task = trajectories_per_task
        self.max_steps_per_trajectory = max_steps_per_trajectory

        # Each ML1 instance is tied to a single task name
        self.envs: Dict[str, Any] = {}
        for name in env_names:
            ml1 = metaworld.ML1(name)
            env = ml1.train_classes[name]()
            task = random.choice(ml1.train_tasks)
            env.set_task(task)
            self.envs[name] = env

        sample_env = next(iter(self.envs.values()))
        self.state_dim = sample_env.observation_space.shape[0]

        self.embedding_model = ReachabilityEmbedding(
            state_dim=self.state_dim,
            embedding_dim=embedding_dim,
            gamma=gamma
        )

        self.poset: Optional[ReachabilityPoset] = None
        self.collected_data: Dict[str, TrajectoryBatch] = {}
        self.critical_transitions: List[CriticalTransition] = []

    def collect_trajectories(self, use_exploration: bool = True) -> Dict[str, TrajectoryBatch]:
        """
        Collect trajectories from all environments using a mixed exploration policy.
        """
        print(f"Collecting trajectories from {len(self.env_names)} environments...")

        for task_name, env in tqdm(self.envs.items(), desc="Collecting data"):
            states, next_states, actions, rewards, dones = [], [], [], [], []
            task_id = self.env_names.index(task_name)

            for _ in range(self.trajectories_per_task):
                obs, _ = env.reset()

                for _ in range(self.max_steps_per_trajectory):
                    if use_exploration and np.random.random() < 0.3:
                        action = env.action_space.sample()
                    else:
                        action = self._heuristic_action(obs)

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    states.append(obs)
                    next_states.append(next_obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)

                    obs = next_obs
                    if done:
                        break

            self.collected_data[task_name] = TrajectoryBatch(
                states=np.array(states),
                next_states=np.array(next_states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                dones=np.array(dones),
                task_ids=[task_id] * len(states)
            )
            print(f"  {task_name}: collected {len(states)} transitions")

        return self.collected_data

    def _heuristic_action(self, obs: np.ndarray) -> np.ndarray:
        """Move gripper toward the object (approximate heuristic)."""
        action = np.zeros(4)
        if len(obs) >= 6:
            gripper_pos = obs[:3]
            obj_pos = obs[3:6]
            action[:3] = np.clip((obj_pos - gripper_pos) * 0.5, -1, 1)
        else:
            action[:3] = np.random.uniform(-1, 1, 3)
        action[3] = np.random.uniform(-1, 1)
        return action

    def train_reachability_embedding(self, epochs: int = 100, batch_size: int = 256) -> List[float]:
        """Train the reachability embedding on collected trajectory data."""
        print("Training reachability embedding...")

        all_states, all_next_states, all_reachable, all_horizons = [], [], [], []

        for task_name, data in self.collected_data.items():
            n = len(data.states)

            for i in range(n):
                if not data.dones[i]:
                    all_states.append(data.states[i])
                    all_next_states.append(data.next_states[i])
                    all_reachable.append(1.0)
                    steps = self._estimate_steps_to_goal(data, i)
                    all_horizons.append(self.gamma ** steps)

            n_neg = min(n, 5000)
            for _ in range(n_neg):
                idx1, idx2 = np.random.randint(0, n, 2)
                if abs(idx1 - idx2) > 50:
                    all_states.append(data.states[idx1])
                    all_next_states.append(data.next_states[idx2])
                    all_reachable.append(0.0)
                    all_horizons.append(1.0)

        states_t = torch.FloatTensor(np.array(all_states))
        next_t = torch.FloatTensor(np.array(all_next_states))
        reach_t = torch.FloatTensor(np.array(all_reachable))
        horizon_t = torch.FloatTensor(np.array(all_horizons))

        n_total = len(states_t)
        losses = []

        for epoch in tqdm(range(epochs), desc="Training embedding"):
            indices = np.random.permutation(n_total)
            for start in range(0, n_total, batch_size):
                batch = indices[start:start + batch_size]
                self.embedding_model.train_step(
                    states_t[batch], next_t[batch], reach_t[batch], horizon_t[batch]
                )

            if epoch % 20 == 0:
                # Evaluate on a random subset to avoid OOM on large datasets
                eval_n = min(n_total, 4096)
                eval_idx = np.random.choice(n_total, eval_n, replace=False)
                with torch.no_grad():
                    val_loss = self.embedding_model.compute_loss(
                        states_t[eval_idx], next_t[eval_idx],
                        reach_t[eval_idx], horizon_t[eval_idx]
                    ).item()
                losses.append(val_loss)
                print(f"Epoch {epoch}: loss = {val_loss:.4f}")

        return losses

    def _estimate_steps_to_goal(self, data: TrajectoryBatch, index: int) -> int:
        """Steps remaining until episode termination from `index`."""
        done_indices = np.where(data.dones[index:])[0]
        if len(done_indices) == 0:
            return self.max_steps_per_trajectory
        return min(int(done_indices[0]) + 1, self.max_steps_per_trajectory)

    def discover_critical_transitions(self) -> List[CriticalTransition]:
        """Build the reachability poset and identify critical transitions."""
        all_states = []
        trajectories = []
        state_idx = 0

        for data in self.collected_data.values():
            current_traj: List[int] = []
            for i in range(len(data.states)):
                all_states.append(data.states[i])
                current_traj.append(state_idx)
                state_idx += 1

                if data.dones[i] or len(current_traj) >= self.max_steps_per_trajectory:
                    if len(current_traj) > 1:
                        trajectories.append(current_traj)
                    current_traj = []

            if len(current_traj) > 1:
                trajectories.append(current_traj)

        all_states_arr = np.array(all_states)

        self.poset = ReachabilityPoset(
            embedding_model=self.embedding_model,
            n_clusters=self.n_clusters
        )
        self.poset.build_from_trajectories(all_states_arr, trajectories)

        self.critical_transitions = self.poset.get_critical_transitions(min_importance=0.05)
        self._compute_task_frequencies()

        return self.critical_transitions

    def _compute_task_frequencies(self):
        """Compute how often each critical transition's cluster appears per task."""
        if self.poset is None:
            return

        for ct in self.critical_transitions:
            task_counts: Dict[int, int] = defaultdict(int)
            total = 0

            for task_name, data in self.collected_data.items():
                task_id = self.env_names.index(task_name)

                # Batch-encode all states for this task
                with torch.no_grad():
                    embs = self.embedding_model.encode(
                        torch.FloatTensor(data.states)
                    ).numpy()
                clusters = self.poset.kmeans.predict(embs)

                count = int(np.isin(clusters, list(ct.descendant_clusters)).sum())
                task_counts[task_id] = count
                total += count

            if total > 0:
                ct.task_frequency = {tid: cnt / total for tid, cnt in task_counts.items()}

    def analyze_critical_transitions(self) -> Dict[str, Any]:
        analysis = {
            "n_critical_transitions": len(self.critical_transitions),
            "top_transitions": [],
            "hierarchy_depth": self._compute_poset_depth(),
            "coverage": self._compute_coverage()
        }

        for ct in self.critical_transitions[:10]:
            analysis["top_transitions"].append({
                "importance": ct.importance_score,
                "predecessor_count": ct.predecessor_count,
                "successor_count": ct.successor_count,
                "task_distribution": ct.task_frequency,
                "is_task_agnostic": len(ct.task_frequency) > len(self.env_names) * 0.7
            })

        return analysis

    def _compute_poset_depth(self) -> int:
        if self.poset is None or self.poset.graph.number_of_nodes() == 0:
            return 0
        try:
            return nx.dag_longest_path_length(self.poset.graph)
        except nx.NetworkXError:
            return 0

    def _compute_coverage(self) -> float:
        if self.poset is None:
            return 0.0
        covered_clusters: Set[int] = set()
        for ct in self.critical_transitions:
            covered_clusters.update(ct.descendant_clusters)
        covered_states = len(self.poset.get_state_indices_for_clusters(covered_clusters))
        total_states = sum(len(d.states) for d in self.collected_data.values())
        return covered_states / total_states if total_states > 0 else 0.0

    def save_critical_transitions(self, filepath: str = "critical_transitions.pkl"):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'transitions': self.critical_transitions,
                'embedding_model_state': self.embedding_model.state_dict(),
                'cluster_centers': self.poset.cluster_centers if self.poset else None,
                'analysis': self.analyze_critical_transitions()
            }, f)
        print(f"Critical transitions saved to {filepath}")

    def run_full_pipeline(self) -> Dict[str, Any]:
        print("=" * 60)
        print("Critical Transition Discovery for Meta-World")
        print("=" * 60)

        self.collect_trajectories()
        self.train_reachability_embedding(epochs=50)
        critical_transitions = self.discover_critical_transitions()
        analysis = self.analyze_critical_transitions()

        if self.poset:
            self.poset.visualize()

        self.save_critical_transitions()

        print("\n" + "=" * 60)
        print("Discovery Complete!")
        print("=" * 60)
        print(f"Found {len(critical_transitions)} critical transitions")
        print(f"Poset depth: {analysis['hierarchy_depth']}")
        print(f"State coverage: {analysis['coverage']:.2%}")

        return analysis


# =============================================================================
# Meta-Learning Integration Utilities
# =============================================================================

class CriticalTransitionSubgoalGenerator:
    """
    Use discovered critical transitions as subgoals for meta-learning.

    Integrates with meta-RL algorithms by providing a library of reusable
    subgoals and a curriculum based on poset structure.
    """

    def __init__(self, critical_transitions_path: str):
        import pickle
        with open(critical_transitions_path, 'rb') as f:
            data = pickle.load(f)

        self.transitions: List[CriticalTransition] = data['transitions']
        self.cluster_centers: Optional[np.ndarray] = data['cluster_centers']
        self.analysis: Dict[str, Any] = data['analysis']
        self.transitions.sort(key=lambda x: x.importance_score, reverse=True)

    def get_subgoal_sequence(self, task_id: int, n_subgoals: int = 3) -> List[np.ndarray]:
        """Generate a subgoal sequence for a given task based on frequency and importance."""
        task_relevant = [
            ct for ct in self.transitions
            if task_id in ct.task_frequency and ct.task_frequency[task_id] > 0.1
        ]
        if len(task_relevant) < n_subgoals:
            task_relevant = [
                ct for ct in self.transitions
                if len(ct.task_frequency) > 1
            ][:n_subgoals]

        selected = sorted(task_relevant, key=lambda x: x.importance_score, reverse=True)[:n_subgoals]
        return [ct.state_prototype for ct in selected]

    def get_curriculum(self) -> List[List[int]]:
        """
        Order transitions by predecessor count so prerequisites come first.
        Returns groups of transition indices at the same depth level.
        """
        sorted_transitions = sorted(
            self.transitions,
            key=lambda x: (x.predecessor_count, -x.importance_score)
        )

        curriculum: List[List[int]] = []
        current_group: List[int] = []
        current_pred_count = -1

        for ct in sorted_transitions:
            if ct.predecessor_count != current_pred_count:
                if current_group:
                    curriculum.append(current_group)
                current_group = []
                current_pred_count = ct.predecessor_count
            current_group.append(self.transitions.index(ct))

        if current_group:
            curriculum.append(current_group)

        return curriculum

    def compute_subgoal_reward(
        self,
        current_state: np.ndarray,
        subgoal_state: np.ndarray,
        threshold: float = 0.1
    ) -> float:
        distance = np.linalg.norm(current_state - subgoal_state)
        return 1.0 if distance < threshold else -distance


# =============================================================================
# Example Usage
# =============================================================================

def main():
    meta_tasks = [
        'reach-v3',
        'push-v3',
        'pick-place-v3',
        'door-open-v3',
       ]

    learner = MetaWorldCriticalTransitionLearner(
        env_names=meta_tasks,
        embedding_dim=32,
        n_clusters=15,
        gamma=0.99,
        trajectories_per_task=50,
        max_steps_per_trajectory=150
    )

    analysis = learner.run_full_pipeline()

    print("\n" + "=" * 60)
    print("Critical Transition Analysis")
    print("=" * 60)

    for i, ct_info in enumerate(analysis['top_transitions'][:5]):
        print(f"\nTransition {i+1}:")
        print(f"  Importance: {ct_info['importance']:.3f}")
        print(f"  Predecessors: {ct_info['predecessor_count']}")
        print(f"  Successors: {ct_info['successor_count']}")
        print(f"  Task-agnostic: {ct_info['is_task_agnostic']}")
        if ct_info['task_distribution']:
            print(f"  Task coverage: {len(ct_info['task_distribution'])} tasks")

    print("\n" + "=" * 60)
    print("Meta-Learning Integration Example")
    print("=" * 60)

    subgoal_gen = CriticalTransitionSubgoalGenerator("critical_transitions.pkl")
    subgoals = subgoal_gen.get_subgoal_sequence(task_id=0, n_subgoals=3)
    print(f"\nSubgoals for reach-v2: {len(subgoals)} subgoals generated")

    curriculum = subgoal_gen.get_curriculum()
    print(f"Curriculum has {len(curriculum)} stages")


if __name__ == "__main__":
    main()
