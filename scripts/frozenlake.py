"""
Dual-Task Coboundary Graph Analysis for Frozen Lake
Two tasks: deterministic (non-slippery) and stochastic (slippery) via gymnasium.

Identifies the underlying common graph-induced simplicial complex from both
potential functions, characterised by Betti numbers β₀ (components) and β₁
(independent loops).

Complexity hierarchy:
  0-simplices : states
  1-simplices : ascending edges  Φ(s') > Φ(s)
  2-simplices : triangles fully supported by 1-simplices
Common complex : intersection of both tasks' 1-simplices (+ induced triangles)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from itertools import combinations
import gymnasium as gym

# ============================================================================
# 1. Gymnasium-backed Frozen Lake MDP
# ============================================================================

class GymnasiumFrozenLakeMDP:
    """
    Frozen Lake MDP backed by gymnasium's FrozenLake-v1.

    Provides get_transitions(s, a) → [(next_state, prob, reward), ...] by
    reading the environment's tabular transition matrix P directly, so
    slippery / non-slippery variants are supported without custom physics.

    Gymnasium P format: P[s][a] = [(prob, next_s, reward, terminated), ...]
    Identical next-states (possible in slippery mode) are aggregated.
    """

    _ACTION_NAMES = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}

    def __init__(self, is_slippery: bool = False):
        self.is_slippery = is_slippery
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode=None)
        raw = self.env.unwrapped

        self.n_states  = self.env.observation_space.n   # 16 for 4×4
        self.n_actions = self.env.action_space.n        # 4
        self.size      = len(raw.desc)                  # rows (= 4)
        self.desc      = raw.desc                       # (4,4) byte array
        self.actions   = self._ACTION_NAMES
        self._P        = raw.P                          # tabular transitions

        self.holes: list  = []
        self.goal:  int   = -1
        self.start: int   = -1
        for i in range(self.size):
            for j in range(self.size):
                s    = i * self.size + j
                cell = self.desc[i, j].decode()
                if   cell == 'H': self.holes.append(s)
                elif cell == 'G': self.goal  = s
                elif cell == 'S': self.start = s

        self.terminal_states: set = set(self.holes + [self.goal])
        self.name = ('Slippery (stochastic)' if is_slippery
                     else 'Non-slippery (deterministic)')

    def get_transitions(self, state: int, action: int):
        """Return [(next_state, prob, reward)] with duplicates aggregated."""
        agg: dict = defaultdict(lambda: [0.0, 0.0])   # ns → [Σprob, Σprob·r]
        for prob, ns, rew, _ in self._P[state][action]:
            agg[ns][0] += prob
            agg[ns][1] += prob * rew
        return [(ns, p, (w / p if p > 0 else 0.0))
                for ns, (p, w) in agg.items()]

    def close(self):
        self.env.close()


# ============================================================================
# 2. Potential via Value Iteration
# ============================================================================

def value_iteration_potential(mdp: GymnasiumFrozenLakeMDP,
                               gamma: float = 0.99,
                               theta: float = 1e-8) -> np.ndarray:
    """Optimal value function as potential Φ via synchronous value iteration."""
    V = np.zeros(mdp.n_states)
    while True:
        delta = 0.0
        for s in range(mdp.n_states):
            if s in mdp.terminal_states:
                continue
            v = V[s]
            max_q = float('-inf')
            for a in range(mdp.n_actions):
                q = sum(p * (r + gamma * V[ns])
                        for ns, p, r in mdp.get_transitions(s, a))
                max_q = max(max_q, q)
            V[s]  = max_q
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


# ============================================================================
# 3. Coboundary Graph
# ============================================================================

class CoboundaryGraphBuilder:
    """
    Builds the directed coboundary graph of the MDP w.r.t. potential Φ.
    Edge weight: F(s, s') = γ·Φ(s') − Φ(s)  (PBRS shaping reward).
    When multiple actions produce the same (s, s') pair, the highest-probability
    entry is retained.
    """

    def __init__(self, mdp: GymnasiumFrozenLakeMDP,
                 phi: np.ndarray, gamma: float = 0.99):
        self.mdp   = mdp
        self.phi   = phi
        self.gamma = gamma
        self.graph = nx.DiGraph()
        self._build()

    def _build(self):
        G = self.graph
        for s in range(self.mdp.n_states):
            G.add_node(s, phi=self.phi[s],
                       terminal=(s in self.mdp.terminal_states))
            if s in self.mdp.terminal_states:
                continue
            for a in range(self.mdp.n_actions):
                for ns, prob, rew in self.mdp.get_transitions(s, a):
                    shaping = self.gamma * self.phi[ns] - self.phi[s]
                    attrs   = dict(prob=prob, orig_reward=rew,
                                   shaping_reward=shaping,
                                   total_weight=rew + shaping, action=a)
                    if G.has_edge(s, ns):
                        if prob > G[s][ns]['prob']:
                            G.add_edge(s, ns, **attrs)
                    else:
                        G.add_edge(s, ns, **attrs)

    def critical_transitions(self, percentile: int = 85):
        """Return edges with |shaping_reward| above the given percentile."""
        vals = [abs(d['shaping_reward']) for _, _, d in self.graph.edges(data=True)]
        if not vals:
            return []
        thr = np.percentile(vals, percentile)
        return [(u, v, d) for u, v, d in self.graph.edges(data=True)
                if abs(d['shaping_reward']) >= thr]


# ============================================================================
# 4. Graph-Induced Simplicial Complex
# ============================================================================

def _ek(u: int, v: int) -> tuple:
    """Canonical undirected edge key (min, max)."""
    return (min(u, v), max(u, v))


class GraphInducedComplex:
    """
    Discrete simplicial complex induced by potential Φ on the transition graph.

    0-simplices : every state node
    1-simplices : undirected edges (s, s') where Φ(s') > Φ(s)  (ascending)
    2-simplices : triangles {u, v, w} fully supported by 1-simplices

    Betti numbers:
      β₀  = connected components of the 1-skeleton
      β₁  = independent 1-cycles not filled by 2-simplices
            (from Euler: β₁ = |E| − |V| + β₀ − |T|, assuming β₂ ≈ 0)
    """

    def __init__(self, builder: CoboundaryGraphBuilder):
        self.phi = builder.phi
        self._build(builder.graph)

    def _build(self, G: nx.DiGraph):
        self.simplices_0: set = set(G.nodes())

        # 1-simplices: ascending directed edges → undirected key
        self.simplices_1: set = {
            _ek(u, v)
            for u, v in G.edges()
            if self.phi[v] > self.phi[u]
        }

        # 2-simplices: triangles fully in simplices_1
        nodes = sorted(self.simplices_0)
        self.simplices_2: set = {
            (u, v, w)
            for u, v, w in combinations(nodes, 3)
            if _ek(u, v) in self.simplices_1
            and _ek(u, w) in self.simplices_1
            and _ek(v, w) in self.simplices_1
        }

    @property
    def betti_0(self) -> int:
        G = nx.Graph()
        G.add_nodes_from(self.simplices_0)
        G.add_edges_from(self.simplices_1)
        return nx.number_connected_components(G)

    @property
    def betti_1(self) -> int:
        n = len(self.simplices_0)
        e = len(self.simplices_1)
        t = len(self.simplices_2)
        return max(0, e - n + self.betti_0 - t)

    def summary(self) -> dict:
        return {
            '0-simplices': len(self.simplices_0),
            '1-simplices': len(self.simplices_1),
            '2-simplices': len(self.simplices_2),
            'β₀': self.betti_0,
            'β₁': self.betti_1,
        }


# ============================================================================
# 5. Common Complex Analyzer
# ============================================================================

class CommonComplexAnalyzer:
    """
    Identifies the common graph-induced simplicial complex shared by two tasks.

    An edge belongs to the common complex iff it is an ascending 1-simplex
    in both Φ₁ and Φ₂.  This captures the topological structure of the
    potential landscape that is invariant under slippery / non-slippery dynamics.

    The common complex is strictly contained in both individual complexes:
        K_common ⊆ K_task1 ∩ K_task2
    with equality for 0- and 1-cells; 2-cells are re-derived from common edges.
    """

    def __init__(self, cx1: GraphInducedComplex, cx2: GraphInducedComplex):
        self.cx1 = cx1
        self.cx2 = cx2
        self._build()

    def _build(self):
        self.simplices_0 = self.cx1.simplices_0 & self.cx2.simplices_0
        self.simplices_1 = self.cx1.simplices_1 & self.cx2.simplices_1

        nodes = sorted(self.simplices_0)
        self.simplices_2: set = {
            (u, v, w)
            for u, v, w in combinations(nodes, 3)
            if _ek(u, v) in self.simplices_1
            and _ek(u, w) in self.simplices_1
            and _ek(v, w) in self.simplices_1
        }

    @property
    def betti_0(self) -> int:
        G = nx.Graph()
        G.add_nodes_from(self.simplices_0)
        G.add_edges_from(self.simplices_1)
        return nx.number_connected_components(G)

    @property
    def betti_1(self) -> int:
        n = len(self.simplices_0)
        e = len(self.simplices_1)
        t = len(self.simplices_2)
        return max(0, e - n + self.betti_0 - t)

    def summary(self) -> dict:
        return {
            '0-simplices': len(self.simplices_0),
            '1-simplices': len(self.simplices_1),
            '2-simplices': len(self.simplices_2),
            'β₀': self.betti_0,
            'β₁': self.betti_1,
        }

    def task_exclusive(self, task: int) -> set:
        """1-simplices in one task's complex but absent from the common complex."""
        src = self.cx1.simplices_1 if task == 1 else self.cx2.simplices_1
        return src - self.simplices_1


# ============================================================================
# 6. Visualization helpers
# ============================================================================

def _grid_pos(size: int) -> dict:
    """NetworkX positions for a size×size grid: (col, −row)."""
    return {s: (s % size, -(s // size)) for s in range(size * size)}


def _draw_grid_panel(ax, pos, size, phi, edges, mdp: GymnasiumFrozenLakeMDP,
                     edge_color='steelblue', edge_width=2.0, alpha=0.85,
                     bg_edges=None, title=''):
    """Reusable panel: grid nodes coloured by phi, edges drawn on top."""
    n = size * size
    G = nx.Graph()
    G.add_nodes_from(range(n))

    node_colors = [phi[s] for s in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, cmap='RdYlGn',
                           node_size=500, edgecolors='black', linewidths=0.8)
    nx.draw_networkx_labels(G, pos, {s: str(s) for s in range(n)},
                            ax=ax, font_size=7)

    if bg_edges:
        nx.draw_networkx_edges(nx.Graph(list(bg_edges)), pos, ax=ax,
                               edge_color='lightgray', width=1.0, alpha=0.4)
    if edges:
        nx.draw_networkx_edges(nx.Graph(list(edges)), pos, ax=ax,
                               edge_color=edge_color, width=edge_width, alpha=alpha)

    # Marker overlays (Start, Goal, Holes)
    ax.plot(mdp.start % size, -(mdp.start // size),
            'go', markersize=11, zorder=5)
    ax.plot(mdp.goal  % size, -(mdp.goal  // size),
            'b*', markersize=14, zorder=5)
    for h in mdp.holes:
        ax.plot(h % size, -(h // size), 'rx', markersize=11,
                markeredgewidth=2, zorder=5)

    ax.set_title(title, fontsize=10)
    ax.axis('off')


# ============================================================================
# 7. Dual-task + common complex visualization
# ============================================================================

def visualize_potentials(mdp1, phi1, mdp2, phi2):
    """Side-by-side heat-maps of Φ₁ and Φ₂ on the 4×4 grid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mdp, phi in [(axes[0], mdp1, phi1), (axes[1], mdp2, phi2)]:
        size = mdp.size
        im   = ax.imshow(phi.reshape(size, size), cmap='RdYlGn',
                         interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Φ(s)')
        for s in range(mdp.n_states):
            i, j = s // size, s % size
            ax.text(j, i, f"{s}\n{phi[s]:.3f}", ha='center', va='center',
                    fontsize=7)
        ax.plot(mdp.start % size, mdp.start // size, 'go', markersize=10)
        ax.plot(mdp.goal  % size, mdp.goal  // size, 'b*', markersize=13)
        for h in mdp.holes:
            ax.plot(h % size, h // size, 'rx', markersize=11, markeredgewidth=2)
        ax.set_title(f"Φ — {mdp.name}", fontsize=10)
        ax.axis('off')
    plt.suptitle("Optimal Value Functions as Potentials", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dual_task_potentials.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_common_complex(common: CommonComplexAnalyzer,
                              cx1: GraphInducedComplex,
                              cx2: GraphInducedComplex,
                              mdp: GymnasiumFrozenLakeMDP,
                              phi1: np.ndarray, phi2: np.ndarray):
    """
    2×2 figure:
      TL – Task 1 ascending 1-simplices
      TR – Task 2 ascending 1-simplices
      BL – Common 1-simplices (intersection)
      BR – Common 2-simplices (filled) with Betti numbers
    """
    size    = mdp.size
    pos     = _grid_pos(size)
    mean_phi = (phi1 + phi2) / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    _draw_grid_panel(axes[0, 0], pos, size, phi1, cx1.simplices_1, mdp,
                     edge_color='steelblue', bg_edges=cx2.simplices_1,
                     title=f"Task 1 — {mdp.name}\n1-simplices: "
                           f"{len(cx1.simplices_1)}  "
                           f"2-simplices: {len(cx1.simplices_2)}  "
                           f"β₁={cx1.betti_1}")

    mdp2_ref = type('_', (), {
        'start': mdp.start, 'goal': mdp.goal,
        'holes': mdp.holes, 'size': size,
    })()
    _draw_grid_panel(axes[0, 1], pos, size, phi2, cx2.simplices_1, mdp2_ref,
                     edge_color='darkorange', bg_edges=cx1.simplices_1,
                     title=f"Task 2 — Slippery (stochastic)\n1-simplices: "
                           f"{len(cx2.simplices_1)}  "
                           f"2-simplices: {len(cx2.simplices_2)}  "
                           f"β₁={cx2.betti_1}")

    _draw_grid_panel(axes[1, 0], pos, size, mean_phi, common.simplices_1, mdp,
                     edge_color='forestgreen', edge_width=3.0,
                     title=f"Common 1-simplices (Φ₁ ∩ Φ₂)\n"
                           f"{len(common.simplices_1)} shared ascending edges  "
                           f"β₀={common.betti_0}  β₁={common.betti_1}")

    # Bottom-right: common complex with filled 2-simplices
    ax4 = axes[1, 1]
    _draw_grid_panel(ax4, pos, size, mean_phi, common.simplices_1, mdp,
                     edge_color='forestgreen', edge_width=3.0,
                     title=f"Common graph-induced complex\n"
                           f"2-simplices: {len(common.simplices_2)}  "
                           f"β₀={common.betti_0}  β₁={common.betti_1}")

    for u, v, w in common.simplices_2:
        xs = [pos[u][0], pos[v][0], pos[w][0], pos[u][0]]
        ys = [pos[u][1], pos[v][1], pos[w][1], pos[u][1]]
        ax4.fill(xs, ys, alpha=0.35, color='limegreen', zorder=1)

    s = common.summary()
    ax4.text(0.02, 0.04,
             f"0-simplices : {s['0-simplices']}\n"
             f"1-simplices : {s['1-simplices']}\n"
             f"2-simplices : {s['2-simplices']}\n"
             f"β₀  =  {s['β₀']}  (components)\n"
             f"β₁  =  {s['β₁']}  (loops)",
             transform=ax4.transAxes, fontsize=9,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], marker='o',  color='w', markerfacecolor='g',  markersize=10, label='Start'),
        Line2D([0], [0], marker='*',  color='w', markerfacecolor='b',  markersize=12, label='Goal'),
        Line2D([0], [0], marker='x',  color='r', markersize=10,        label='Hole'),
        Line2D([0], [0], color='steelblue',   lw=2, label='Task 1 only'),
        Line2D([0], [0], color='darkorange',  lw=2, label='Task 2 only'),
        Line2D([0], [0], color='forestgreen', lw=3, label='Common'),
        Patch(facecolor='limegreen', alpha=0.4, label='Common 2-simplex'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Graph-Induced Simplicial Complex — Common Structure Across Tasks",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('common_complex.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved common_complex.png")


# ============================================================================
# 8. Main
# ============================================================================

if __name__ == "__main__":
    gamma = 0.99

    print("=" * 60)
    print("DUAL-TASK FROZEN LAKE: COBOUNDARY + COMMON COMPLEX")
    print("=" * 60)

    # ── Task 1: non-slippery ──────────────────────────────────────
    print("\n[Task 1] Non-slippery FrozenLake-v1")
    mdp1 = GymnasiumFrozenLakeMDP(is_slippery=False)
    phi1 = value_iteration_potential(mdp1, gamma)
    print(f"  Φ range: [{phi1.min():.4f}, {phi1.max():.4f}]")
    b1 = CoboundaryGraphBuilder(mdp1, phi1, gamma)

    # ── Task 2: slippery ─────────────────────────────────────────
    print("\n[Task 2] Slippery FrozenLake-v1")
    mdp2 = GymnasiumFrozenLakeMDP(is_slippery=True)
    phi2 = value_iteration_potential(mdp2, gamma)
    print(f"  Φ range: [{phi2.min():.4f}, {phi2.max():.4f}]")
    b2 = CoboundaryGraphBuilder(mdp2, phi2, gamma)

    # ── Coboundary condition verification ────────────────────────
    print("\n" + "=" * 60)
    print("COBOUNDARY CONDITION  F(s,s') = γΦ(s') − Φ(s)")
    print("=" * 60)
    for tag, builder, phi in [("Task 1", b1, phi1), ("Task 2", b2, phi2)]:
        print(f"\n  {tag}:")
        for u, v, d in list(builder.graph.edges(data=True))[:3]:
            lhs = d['shaping_reward']
            rhs = gamma * phi[v] - phi[u]
            ok  = "✓" if np.isclose(lhs, rhs) else "✗"
            print(f"    {u}→{v}: F={lhs:.4f}  γΦ(s')−Φ(s)={rhs:.4f}  {ok}")

    # ── Critical transitions ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("CRITICAL TRANSITIONS (top-15% shaping reward)")
    print("=" * 60)
    for tag, builder in [("Task 1", b1), ("Task 2", b2)]:
        critical = builder.critical_transitions(percentile=85)
        print(f"\n  {tag}: {len(critical)} critical edges")
        for u, v, d in sorted(critical,
                               key=lambda x: abs(x[2]['shaping_reward']),
                               reverse=True)[:5]:
            print(f"    {u:2d}→{v:2d}  act={builder.mdp.actions[d['action']]}  "
                  f"F={d['shaping_reward']:+.4f}  R={d['orig_reward']:.3f}")

    # ── Graph-induced complexes ───────────────────────────────────
    print("\n" + "=" * 60)
    print("GRAPH-INDUCED SIMPLICIAL COMPLEXES")
    print("=" * 60)
    cx1 = GraphInducedComplex(b1)
    cx2 = GraphInducedComplex(b2)
    print(f"\n  Task 1 complex : {cx1.summary()}")
    print(f"  Task 2 complex : {cx2.summary()}")

    # ── Common complex ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMMON GRAPH-INDUCED COMPLEX  (Φ₁ ∩ Φ₂)")
    print("=" * 60)
    common = CommonComplexAnalyzer(cx1, cx2)
    print(f"\n  Common complex  : {common.summary()}")
    print(f"  Task-1-only edges: {len(common.task_exclusive(1))}")
    print(f"  Task-2-only edges: {len(common.task_exclusive(2))}")

    if common.simplices_1:
        print("\n  Common ascending edges:")
        for u, v in sorted(common.simplices_1):
            print(f"    {u:2d} — {v:2d}   ΔΦ₁={phi1[v]-phi1[u]:+.4f}   "
                  f"ΔΦ₂={phi2[v]-phi2[u]:+.4f}")

    if common.simplices_2:
        print(f"\n  Common 2-simplices (triangles): {sorted(common.simplices_2)}")

    # ── Visualisations ────────────────────────────────────────────
    print("\nGenerating visualisations...")
    visualize_potentials(mdp1, phi1, mdp2, phi2)
    visualize_common_complex(common, cx1, cx2, mdp1, phi1, phi2)

    mdp1.close()
    mdp2.close()
