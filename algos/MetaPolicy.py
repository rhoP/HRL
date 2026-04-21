"""
Meta-RL algorithm: hitting-time potential + discrete Morse skeleton + option framework.
Task distribution: MetaWorld ML1 / ML10 / ML45 benchmarks (v3 environments).

Phases (from docs/Meta-RL.md):
  0. Collect experience per task using SB3 (SAC/PPO).
  1. Build common Morse skeleton; critical states discovered from trajectory topology.
  2. Train V_H(s; c) hitting-time value nets for common critical states.
  3. Train sub-policies for each critical state with augmented rewards.
  4. Train meta-policy π_θ(c | s) over critical states (REINFORCE over options).
  5. Repeat.

Run as:
    python3 algos/MetaPolicy.py --benchmark ML10 --iterations 3 \\
        --landmarks 32 --meta-epochs 100 --save-dir results/run1

    # Demo-only from saved checkpoint:
    python3 algos/MetaPolicy.py --demo-only --load results/run1/best \\
        --benchmark ML10 --n-demos 10
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import metaworld
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env

from models.networks      import PolicyNetwork, ValueNetwork
from utils.replay_buffer  import ReplayBuffer
from utils.skeleton       import build_skeleton, refine_skeleton
from algos.HTTD           import train_hitting_time_value_net
from models.HTVNet        import HittingTimeValueNet
from utils.train          import train_sub_policy
from utils.utils          import compute_discounted_returns
from utils.checkpoint     import (save_checkpoint, load_checkpoint,
                                   restore_models, BestModelTracker,
                                   save_replay_buffer, load_replay_buffer)
from utils.viz            import (plot_training_curves, plot_skeleton_topology,
                                   evaluate_policy, run_demos,
                                   save_iteration_visuals)

_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from morse import RewardNormalizer


# ── MetaWorld state/action dimensions ─────────────────────────────────────

MW_STATE_DIM  = 39   # all v3 environments
MW_ACTION_DIM = 4    # all v3 environments (continuous)


# ── Task / Distribution ────────────────────────────────────────────────────

class MetaWorldTask:
    """
    Wraps a single MetaWorld (env_name, mw_task) pair.

    create_env() instantiates a fresh environment with the task's goal set.
    """

    def __init__(self, task_id: int, env_cls, mw_task):
        self.id       = task_id
        self.env_name = mw_task.env_name
        self._env_cls = env_cls
        self._mw_task = mw_task

    def create_env(self):
        env = self._env_cls()
        env.set_task(self._mw_task)
        return env



class MetaWorldTaskDistribution:
    """
    Uniform distribution over a list of MetaWorldTask objects.

    Factory classmethod build() constructs the distribution from an ML1 / ML10 /
    ML45 benchmark object.
    """

    def __init__(self, tasks: list):
        self.tasks = tasks

    def sample(self) -> MetaWorldTask:
        return self.tasks[np.random.randint(len(self.tasks))]

    @classmethod
    def from_env_names(cls, env_names: list, max_tasks_per_env: int = 5):
        """
        Build a distribution over specific MetaWorld env names.

        Example:
            dist = MetaWorldTaskDistribution.from_env_names(
                ['reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3']
            )
        """
        tasks   = []
        task_id = 0
        for env_name in env_names:
            ml1     = metaworld.ML1(env_name)
            env_cls = ml1.train_classes[env_name]
            for mw_task in ml1.train_tasks[:max_tasks_per_env]:
                tasks.append(MetaWorldTask(task_id, env_cls, mw_task))
                task_id += 1
        return cls(tasks)

    @classmethod
    def from_benchmark(cls, benchmark, max_tasks_per_env: int = 10):
        """
        Build a distribution from any MetaWorld benchmark (ML1, ML10, ML45).

        For ML1 this gives multiple goal-variants of a single task.
        For ML10/ML45 this samples up to max_tasks_per_env goals per task type.
        """
        tasks = []
        task_id = 0
        env_to_cls = benchmark.train_classes
        env_to_tasks: dict = {}
        for mw_task in benchmark.train_tasks:
            env_to_tasks.setdefault(mw_task.env_name, []).append(mw_task)

        for env_name, mw_task_list in env_to_tasks.items():
            env_cls = env_to_cls[env_name]
            chosen  = mw_task_list[:max_tasks_per_env]
            for mw_task in chosen:
                tasks.append(MetaWorldTask(task_id, env_cls, mw_task))
                task_id += 1

        return cls(tasks)


# ── Gymnasium-compatible wrapper ───────────────────────────────────────────

class MetaWorldGymWrapper(gym.Env):
    """
    Thin wrapper that makes a MetaWorld env look like a standard Gymnasium env
    so SB3 and other tools work without modification.
    """

    metadata = {"render_modes": []}

    def __init__(self, mw_task: MetaWorldTask):
        super().__init__()
        self._task      = mw_task
        self._env       = mw_task.create_env()
        raw_obs_space   = self._env.observation_space
        raw_act_space   = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low=raw_obs_space.low.astype(np.float32),
            high=raw_obs_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=raw_act_space.low.astype(np.float32),
            high=raw_act_space.high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset()
        return np.asarray(obs, dtype=np.float32).flatten(), info

    def step(self, action):
        obs, r, terminated, truncated, info = self._env.step(action)
        return np.asarray(obs, dtype=np.float32).flatten(), float(r), terminated, truncated, info

    def close(self):
        self._env.close()


# ── Sub-policy ─────────────────────────────────────────────────────────────

class SubPolicy:
    """
    An option that attempts to reach a specific subgoal state (continuous actions).
    """

    def __init__(
        self,
        subgoal_id,
        subgoal_state: np.ndarray,
        state_dim: int    = MW_STATE_DIM,
        action_dim: int   = MW_ACTION_DIM,
        discrete: bool    = False,
        threshold: float  = 0.1,
        max_steps: int    = 100,
        reach_bonus: float = 1.0,
        device: str       = "cpu",
    ):
        self.id          = subgoal_id
        self.c           = torch.tensor(
            np.asarray(subgoal_state, dtype=np.float32), device=device
        )
        self.threshold   = threshold
        self.max_steps   = max_steps
        self.reach_bonus = reach_bonus
        self.device      = device
        self.discrete    = discrete

        self.policy_net       = PolicyNetwork(state_dim, action_dim, discrete=discrete).to(device)
        self.value_net        = ValueNetwork(state_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim).to(device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

    def get_action(self, s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net.get_action(s)

    def is_terminated(self, s: torch.Tensor, done: bool, steps: int) -> bool:
        dist = torch.norm(s - self.c)
        return bool(dist < self.threshold) or done or (steps >= self.max_steps)

    def _soft_update_targets(self, tau: float = 0.005):
        for p, tp in zip(self.value_net.parameters(),
                         self.target_value_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ── Meta-policy networks ───────────────────────────────────────────────────

class MetaPolicy(nn.Module):
    """π_θ(c | s) — memoryless meta-policy that selects subgoal indices."""

    def __init__(self, state_dim: int, num_subgoals: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_subgoals),
        )

    def forward(self, s: torch.Tensor):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return torch.distributions.Categorical(logits=self.net(s))


class MetaValueNetwork(nn.Module):
    """V^meta(s) — value function for the meta-policy."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return self.net(s)


# ── Phase 0: SB3-based data collection ────────────────────────────────────

def phase0_collect_initial_data(
    task_distribution: MetaWorldTaskDistribution,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 5_000,
    algo: str = "SAC",
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """
    Use SAC (or PPO) from stable_baselines3 to collect transitions from each
    task in the distribution. Pushes all transitions into replay_buffer.

    Per the docs: "Initialize a set of algorithms (such as PPO, SAC) from
    stable_baselines3 for each task."
    """
    AlgoCls = SAC if algo.upper() == "SAC" else PPO

    for task in task_distribution.tasks:
        if verbose:
            print(f"  [Phase 0] Collecting from {task.env_name} (task {task.id}) "
                  f"with {algo} for {timesteps_per_task} steps...")

        def _make_env():
            return MetaWorldGymWrapper(task)

        vec_env = make_vec_env(_make_env, n_envs=1)
        model   = AlgoCls("MlpPolicy", vec_env, verbose=0,
                          device=device)
        model.learn(total_timesteps=timesteps_per_task)

        # Roll out the trained model and push transitions into the shared buffer
        obs = vec_env.reset()
        for _ in range(timesteps_per_task):
            action, _ = model.predict(obs, deterministic=False)
            obs_next, reward, done, info = vec_env.step(action)
            replay_buffer.push(
                obs[0], action[0], float(reward[0]),
                obs_next[0], bool(done[0]), task.id,
            )
            obs = obs_next if not done[0] else vec_env.reset()

        vec_env.close()

    if verbose:
        print(f"  [Phase 0] Buffer size after collection: {len(replay_buffer)}")


# ── Phase 1: Morse skeleton ────────────────────────────────────────────────

def phase1_build_skeleton(
    replay_buffer: ReplayBuffer,
    num_landmarks: int = 32,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Build common Morse skeleton from the shared replay buffer.
    Critical states are identified from trajectory topology; goal states are
    not provided explicitly — the model learns them from done/success signals.
    """
    state_dim = replay_buffer.state_dim
    if state_dim is None:
        raise RuntimeError("replay_buffer is empty — run Phase 0 first.")

    return build_skeleton(
        replay_buffer, state_dim=state_dim,
        num_landmarks=num_landmarks,
        device=device, verbose=verbose, **kwargs,
    )


# ── Phase 2: Hitting-time value nets ──────────────────────────────────────

def phase2_train_value_net(
    skeleton_data: dict,
    replay_buffer: ReplayBuffer,
    gamma: float  = 0.99,
    epochs: int   = 200,
    batch_size: int = 256,
    lr: float     = 1e-3,
    tau: float    = 0.005,
    device: str   = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Train V_H(s; c) for each common critical state via TD learning.
    Returns dict mapping c_id → HittingTimeValueNet.
    """
    state_dim       = replay_buffer.state_dim
    critical_states = skeleton_data["critical_states"]
    hitting_nets = {}
    all_losses   = []

    # Fit a single reward normalizer on the full buffer (shared across all c_id nets)
    reward_normalizer = RewardNormalizer()
    for ep in replay_buffer.iter_episodes():
        rw = ep["rewards"]
        reward_normalizer.update(rw.tolist() if hasattr(rw, "tolist") else list(rw))
    if verbose:
        print(f"  [Phase 2] Reward stats: mean={reward_normalizer.mean:.4f}  "
              f"std={reward_normalizer.std:.4f}")

    for c_id, c_np in critical_states.items():
        if verbose:
            print(f"  [Phase 2] Training V_H for critical state {c_id}...")
        c_t = torch.tensor(c_np, dtype=torch.float32, device=device)

        value_net  = HittingTimeValueNet(state_dim).to(device)
        target_net = HittingTimeValueNet(state_dim).to(device)
        target_net.load_state_dict(value_net.state_dict())
        optimizer  = torch.optim.Adam(value_net.parameters(), lr=lr)

        def S_proto_fn(task_id, _c=c_t):
            B = task_id.shape[0] if task_id.dim() > 0 else 1
            return _c.unsqueeze(0).expand(B, -1)

        run_losses = []
        for _ in range(epochs):
            if len(replay_buffer) >= batch_size:
                loss = train_hitting_time_value_net(
                    value_net, target_net, replay_buffer,
                    S_proto_fn, optimizer,
                    gamma=gamma, batch_size=batch_size, tau=tau,
                    reward_normalizer=reward_normalizer,
                )
                if not np.isnan(loss):
                    run_losses.append(loss)

        hitting_nets[c_id] = value_net
        all_losses.extend(run_losses)

    return hitting_nets, all_losses


# ── Phase 3 helpers ────────────────────────────────────────────────────────

def collect_sub_policy_rollouts(
    sp: "SubPolicy",
    task_distribution: "MetaWorldTaskDistribution",
    target_buffer: ReplayBuffer,
    num_episodes: int = 5,
    device: str = "cpu",
) -> int:
    """
    Roll out SubPolicy sp from fresh task resets and push transitions into
    target_buffer.  Returns the number of transitions collected.

    This mirrors collect_with_meta_policy but works at the sub-policy level:
    no meta-policy is needed; sp runs until its own termination condition.
    """
    collected = 0
    for _ in range(num_episodes):
        task = task_distribution.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        obs  = torch.tensor(np.asarray(obs, dtype=np.float32).flatten(), device=device)
        done = False
        T_c  = 0
        while not sp.is_terminated(obs, done, T_c):
            with torch.no_grad():
                a = sp.get_action(obs)
            a_np = a.cpu().numpy()
            obs_next, r_env, terminated, truncated, _ = env.step(a_np)
            done = terminated or truncated
            obs_next_np = np.asarray(obs_next, dtype=np.float32).flatten()
            target_buffer.push(
                obs.cpu().numpy(), a_np, r_env, obs_next_np, done, task.id,
            )
            obs  = torch.tensor(obs_next_np, device=device)
            T_c += 1
            collected += 1
            if done:
                break
        env.close()
    return collected


# ── Phase 3: Sub-policies ─────────────────────────────────────────────────

def phase3_train_sub_policies(
    skeleton_data: dict,
    replay_buffer: ReplayBuffer,
    hitting_nets: dict,
    task_distribution=None,
    state_dim: int    = MW_STATE_DIM,
    action_dim: int   = MW_ACTION_DIM,
    discrete: bool    = False,
    gamma: float      = 0.99,
    epochs: int       = 50,
    batch_size: int   = 256,
    beta_intrinsic: float = 0.1,
    reach_bonus: float    = 1.0,
    reach_threshold: float = 0.5,
    max_steps: int    = 100,
    collect_episodes: int  = 5,
    existing_sub_policies: dict = None,
    carry_over_policies: bool   = True,
    device: str       = "cpu",
    verbose: bool     = True,
) -> dict:
    """
    Train one SubPolicy per critical state discovered by the skeleton.

    When task_distribution is provided, on-policy rollouts are collected with
    the current sub-policy before training begins and mixed 50/50 with the
    off-policy SB3 data on every gradient step. This reduces the distribution
    gap that causes high Phase 3 variance.

    When carry_over_policies=True and existing_sub_policies is provided, any
    SubPolicy whose c_id already appears in existing_sub_policies is reused
    (continues training from its current weights) rather than being
    re-initialised from scratch.

    Set discrete=True for environments with Discrete action spaces.
    """
    critical_states = skeleton_data["critical_states"]
    sub_policies    = {}
    per_subgoal_losses: dict = {}

    for c_id, c_np in critical_states.items():
        reused = (
            carry_over_policies
            and existing_sub_policies is not None
            and c_id in existing_sub_policies
        )
        if verbose:
            tag = "continuing" if reused else "initialising"
            print(f"  [Phase 3] Sub-policy {c_id} ({tag})...")

        c_tensor = torch.tensor(c_np, dtype=torch.float32, device=device)

        if reused:
            sp = existing_sub_policies[c_id]
        else:
            sp = SubPolicy(
                c_id, c_np,
                state_dim=state_dim, action_dim=action_dim,
                discrete=discrete,
                threshold=reach_threshold,
                max_steps=max_steps,
                reach_bonus=reach_bonus,
                device=device,
            )

        # Collect on-policy sub-policy rollouts for 50/50 mixing
        sp_buffer = None
        if task_distribution is not None:
            sp_buffer = ReplayBuffer(
                capacity=collect_episodes * max_steps * 2,
                device=device,
            )
            n = collect_sub_policy_rollouts(
                sp, task_distribution, sp_buffer,
                num_episodes=collect_episodes, device=device,
            )
            if verbose:
                print(f"             collected {n} on-policy transitions "
                      f"({'50/50 mix active' if n >= batch_size // 2 else 'off-policy only — buffer too small'})")

        sp_metrics = train_sub_policy(
            sp, replay_buffer, c_tensor,
            hitting_time_net=hitting_nets.get(c_id),
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            beta_intrinsic=beta_intrinsic,
            reach_bonus=reach_bonus,
            reach_threshold=reach_threshold,
            discrete=discrete,
            on_policy_buffer=sp_buffer,
            device=device,
        )
        sub_policies[c_id]       = sp
        per_subgoal_losses[c_id] = sp_metrics

    all_pi_losses = [l for m in per_subgoal_losses.values() for l in m["pi_losses"]]
    all_v_losses  = [l for m in per_subgoal_losses.values() for l in m["v_losses"]]

    return sub_policies, all_pi_losses, all_v_losses


# ── Phase 4 helpers ────────────────────────────────────────────────────────

class ReturnNormalizer:
    """
    Welford online algorithm for running mean/variance of option returns.
    Used to normalise returns before advantage computation so the effective
    gradient scale stays bounded regardless of reward magnitude.
    """

    def __init__(self):
        self.mean  = 0.0
        self.M2    = 0.0
        self.count = 0

    def update(self, returns):
        for r in returns:
            self.count += 1
            delta      = r - self.mean
            self.mean += delta / self.count
            delta2     = r - self.mean
            self.M2   += delta * delta2

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / max(self.count - 1, 1)))

    def normalize(self, value: float) -> float:
        return (value - self.mean) / (self.std + 1e-8)


def check_sub_policy_convergence(
    sub_policies: dict,
    task_distribution,
    n_eval: int   = 5,
    max_steps: int = 100,
    threshold: float = 0.3,
    device: str   = "cpu",
) -> bool:
    """
    Sample n_eval tasks, run each sub-policy deterministically, and check
    whether the average subgoal-reach rate exceeds threshold.
    Returns False (skip meta-update) when sub-policies are not converged.
    """
    if not sub_policies:
        return False
    rates = []
    for c_id, sp in sub_policies.items():
        reached = 0
        for _ in range(n_eval):
            task = task_distribution.sample()
            env  = task.create_env()
            obs, _ = env.reset()
            obs = torch.tensor(
                np.asarray(obs, dtype=np.float32).flatten(), device=device
            )
            for step in range(max_steps):
                with torch.no_grad():
                    a = sp.policy_net.get_action(obs, deterministic=True)
                a_np = a.cpu().numpy()
                obs_next, _, terminated, truncated, _ = env.step(a_np)
                done = terminated or truncated
                obs_next_t = torch.tensor(
                    np.asarray(obs_next, dtype=np.float32).flatten(), device=device
                )
                if sp.is_terminated(obs_next_t, done, step):
                    reached += (not done)   # reached subgoal, not just env-done
                    break
                obs = obs_next_t
                if done:
                    break
            env.close()
        rates.append(reached / n_eval)
    avg = float(np.mean(rates))
    return avg >= threshold


def _meta_curriculum(iteration: int, total: int) -> dict:
    """
    Anneal meta-training hyperparameters over iterations:
      - option_stickiness_bonus: high early (prevents thrashing), decays to 0
      - min_option_steps:        starts at 10, decays to 2
      - entropy_coef:            starts low, rises to encourage later exploration
      - clip_epsilon:            PPO clip, fixed at 0.2
      - gae_lambda:              GAE mixing, fixed
    """
    progress = float(iteration) / max(total - 1, 1)
    return {
        "option_stickiness_bonus": 0.5 * (1.0 - progress),
        "min_option_steps":        max(2, int(10 * (1.0 - progress))),
        "entropy_coef":            0.01 * (1.0 + progress),
        "clip_epsilon":            0.2,
        "gae_lambda":              0.95,
    }


# ── Phase 4: Meta-policy ───────────────────────────────────────────────────

def _collect_meta_episode(
    meta_policy,
    sub_policies: dict,
    c_list: list,
    task_distribution,
    gamma: float,
    curriculum: dict,
    device: str,
) -> list:
    """
    Roll out one meta-episode. Returns list of
    (s_start, c_idx, log_prob_old, R_option, T_option).
    log_prob_old is stored for the PPO importance-ratio.
    """
    task = task_distribution.sample()
    env  = task.create_env()
    obs, _ = env.reset()
    obs  = torch.tensor(
        np.asarray(obs, dtype=np.float32).flatten(), device=device
    )
    done    = False
    episode = []
    stickiness = curriculum["option_stickiness_bonus"]
    min_steps  = curriculum["min_option_steps"]

    while not done:
        with torch.no_grad():
            dist          = meta_policy(obs)
            c_idx_t       = dist.sample().item()
            log_prob_old  = dist.log_prob(
                torch.tensor(c_idx_t, device=device)
            ).item()

        c_id = c_list[c_idx_t]
        sp   = sub_policies.get(c_id)
        s_start  = obs.clone()
        R_option = 0.0
        T_c      = 0

        # Always run at least min_option_steps to reduce policy churn
        with torch.no_grad():
            while True:
                terminate = (sp is not None and
                             sp.is_terminated(obs, done, T_c) and
                             T_c >= min_steps)
                if terminate or done:
                    break
                if sp is None:
                    a_np = env.action_space.sample()
                else:
                    a    = sp.get_action(obs)
                    a_np = a.cpu().numpy()
                obs_next, r_env, terminated, truncated, _ = env.step(a_np)
                done     = terminated or truncated
                obs_next = torch.tensor(
                    np.asarray(obs_next, dtype=np.float32).flatten(), device=device
                )
                # Stickiness bonus: reward staying with the current option
                R_option += (gamma ** T_c) * (r_env + stickiness / max(T_c + 1, 1))
                obs  = obs_next
                T_c += 1
                if done:
                    break

        episode.append((s_start, c_idx_t, log_prob_old, R_option, T_c))

    env.close()
    return episode


def phase4_train_meta_policy(
    sub_policies: dict,
    skeleton_data: dict,
    task_distribution: MetaWorldTaskDistribution,
    state_dim: int        = MW_STATE_DIM,
    gamma: float          = 0.99,
    meta_epochs: int      = 200,
    episodes_per_update: int = 4,
    ppo_epochs: int       = 4,
    clip_epsilon: float   = 0.2,
    target_entropy: float = 0.5,
    entropy_lr: float     = 0.01,
    convergence_threshold: float = 0.0,
    total_iterations: int = 1,
    current_iteration: int = 0,
    device: str           = "cpu",
    verbose: bool         = True,
):
    """
    Train MetaPolicy over options using a PPO-style clipped objective with:
      1. K-episode minibatch averaging (episodes_per_update=4)
      2. Advantage normalisation per minibatch
      3. Return normalisation via Welford running statistics
      4. PPO clipped surrogate loss (ppo_epochs inner updates per batch)
      5. Adaptive entropy regularisation
      6. Sub-policy convergence gate
      7. Meta-curriculum annealing over iterations

    Sub-policy parameters are frozen throughout.
    Returns (MetaPolicy, MetaValueNetwork, epoch_returns).
    """
    c_list       = list(skeleton_data["critical_states"].keys())
    num_subgoals = len(c_list)

    if num_subgoals == 0:
        print("[Phase 4] No subgoals — skipping meta-policy training.")
        return None, None, []

    # Sub-policy convergence gate
    if convergence_threshold > 0.0:
        if not check_sub_policy_convergence(
            sub_policies, task_distribution,
            threshold=convergence_threshold, device=device,
        ):
            print("[Phase 4] Sub-policies below convergence threshold — "
                  "skipping meta-update.")
            return None, None, []

    meta_policy    = MetaPolicy(state_dim, num_subgoals).to(device)
    meta_value_net = MetaValueNetwork(state_dim).to(device)
    opt_meta  = torch.optim.Adam(meta_policy.parameters(),    lr=1e-4)
    opt_metav = torch.optim.Adam(meta_value_net.parameters(), lr=1e-4)

    return_normalizer = ReturnNormalizer()
    entropy_coef      = 0.01
    epoch_returns: list = []

    for epoch in range(meta_epochs):
        curriculum = _meta_curriculum(
            epoch + current_iteration * meta_epochs,
            total_iterations * meta_epochs,
        )

        # ── Collect K episodes ─────────────────────────────────────────────
        batch_episodes = []
        for _ in range(episodes_per_update):
            ep = _collect_meta_episode(
                meta_policy, sub_policies, c_list,
                task_distribution, gamma, curriculum, device,
            )
            if ep:
                batch_episodes.append(ep)

        if not batch_episodes:
            continue

        # Update return normalizer on all observed option returns
        all_R = [e[3] for ep in batch_episodes for e in ep]
        return_normalizer.update(all_R)

        epoch_returns.append(float(np.mean(all_R)))

        # ── Compute advantages across the full batch ────────────────────────
        # (s, c_idx, log_prob_old, R, T) per step across all episodes
        flat: list = []
        for ep in batch_episodes:
            n = len(ep)
            for t, (s_t, c_idx_t, lp_old, R_t, T_t) in enumerate(ep):
                V_cur  = meta_value_net(s_t).item()
                V_next = meta_value_net(ep[t + 1][0]).item() \
                    if t < n - 1 else 0.0
                target = return_normalizer.normalize(
                    R_t + (gamma ** T_t) * V_next
                )
                advantage = target - return_normalizer.normalize(V_cur)
                flat.append((s_t, c_idx_t, lp_old, target, advantage))

        # Normalise advantages across the entire minibatch
        adv_arr = np.array([f[4] for f in flat], dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # ── PPO inner update loop ──────────────────────────────────────────
        for _ in range(ppo_epochs):
            pi_loss_total = torch.zeros(1, device=device)
            v_loss_total  = torch.zeros(1, device=device)
            entropy_total = torch.zeros(1, device=device)

            for i, (s_t, c_idx_t, lp_old, target, _) in enumerate(flat):
                adv = float(adv_arr[i])

                dist      = meta_policy(s_t)
                log_p_new = dist.log_prob(torch.tensor(c_idx_t, device=device))
                ratio     = torch.exp(log_p_new - lp_old)

                # Clipped surrogate objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon,
                                           1.0 + clip_epsilon) * adv
                pi_loss_total = pi_loss_total - torch.min(surr1, surr2)

                # Value loss
                V_cur = meta_value_net(s_t)
                v_loss_total = v_loss_total + F.mse_loss(
                    V_cur,
                    torch.tensor([[target]], dtype=torch.float32, device=device),
                )

                entropy_total = entropy_total + dist.entropy()

            N = max(len(flat), 1)
            pi_loss = pi_loss_total / N
            v_loss  = v_loss_total  / N
            entropy = entropy_total / N

            # Adaptive entropy coefficient
            if entropy.item() < target_entropy:
                entropy_coef *= (1.0 + entropy_lr)
            else:
                entropy_coef *= (1.0 - entropy_lr)
            entropy_coef = float(np.clip(entropy_coef, 1e-4, 0.5))

            total_loss = pi_loss - entropy_coef * entropy + 0.5 * v_loss

            opt_meta.zero_grad()
            opt_metav.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_policy.parameters(),    0.5)
            torch.nn.utils.clip_grad_norm_(meta_value_net.parameters(), 0.5)
            opt_meta.step()
            opt_metav.step()

        if verbose and (epoch + 1) % 50 == 0:
            avg_R = epoch_returns[-1]
            print(f"  [Phase 4] epoch {epoch+1}/{meta_epochs}: "
                  f"avg option return={avg_R:.4f}  "
                  f"entropy_coef={entropy_coef:.4f}")

    return meta_policy, meta_value_net, epoch_returns


# ── Return helper ──────────────────────────────────────────────────────────

def compute_discounted_meta_returns(rewards: list, durations: list,
                                    gamma: float) -> list:
    """G_t = R_t + γ^{T_t} · G_{t+1}, accounting for option duration."""
    returns = []
    G = 0.0
    for r, T in zip(reversed(rewards), reversed(durations)):
        G = float(r) + (gamma ** int(T)) * G
        returns.insert(0, G)
    return returns


# ── Collect with trained meta-policy ──────────────────────────────────────

def collect_with_meta_policy(
    meta_policy,
    sub_policies: dict,
    task_distribution: MetaWorldTaskDistribution,
    skeleton_data: dict,
    replay_buffer: ReplayBuffer,
    num_episodes: int = 20,
    gamma: float      = 0.99,
    device: str       = "cpu",
) -> None:
    """Roll out meta-policy + sub-policies and push transitions into replay_buffer."""
    c_list = list(skeleton_data["critical_states"].keys())

    for _ in range(num_episodes):
        task = task_distribution.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        obs  = torch.tensor(
            np.asarray(obs, dtype=np.float32).flatten(), device=device
        )
        done = False

        while not done:
            with torch.no_grad():
                dist  = meta_policy(obs)
                c_idx = dist.sample().item()
            c_id = c_list[c_idx]
            sp   = sub_policies.get(c_id)
            T_c  = 0

            while True:
                if sp is None or (sp is not None and sp.is_terminated(obs, done, T_c)):
                    break
                with torch.no_grad():
                    a    = sp.get_action(obs)
                a_np     = a.cpu().numpy()
                obs_next, r_env, terminated, truncated, _ = env.step(a_np)
                done     = terminated or truncated
                obs_next_np = np.asarray(obs_next, dtype=np.float32).flatten()
                replay_buffer.push(
                    obs.cpu().numpy(), a_np,
                    r_env, obs_next_np, done, task.id,
                )
                obs  = torch.tensor(obs_next_np, device=device)
                T_c += 1
                if done:
                    break

        env.close()


# ── Main orchestration ─────────────────────────────────────────────────────

def main_meta_rl_loop(
    task_distribution: MetaWorldTaskDistribution,
    state_dim: int          = MW_STATE_DIM,
    action_dim: int         = MW_ACTION_DIM,
    num_landmarks: int      = 32,
    num_iterations: int     = 3,
    refine_every: int       = 2,
    timesteps_per_task: int = 5_000,
    collect_episodes: int   = 20,
    gamma: float            = 0.99,
    sub_epochs: int         = 50,
    meta_epochs: int        = 200,
    algo: str               = "SAC",
    eval_episodes: int      = 10,
    n_demos: int            = 5,
    save_dir: str           = "results/meta_rl",
    device: str             = "cpu",
    verbose: bool           = True,
):
    """
    Full meta-RL pipeline with metrics logging, checkpointing, evaluation,
    and visualization:
        Phase 0 → Phase 1 → (Phase 2 → Phase 3 → Phase 4 →
            evaluate → checkpoint → visualize → collect → refine)*
    """
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Meta-RL pipeline  (MetaWorld)")
        print(f"  tasks: {len(task_distribution.tasks)}, "
              f"landmarks: {num_landmarks}, "
              f"iterations: {num_iterations}")
        print(f"  save_dir: {save_dir}")
        print("=" * 60)

    # Accumulated metrics across all iterations
    metrics = {
        "skeleton_train_losses": [],
        "phase2_losses":         [],
        "phase3_pi_losses":      [],
        "phase3_v_losses":       [],
        "phase4_returns":        [],
        "eval_success_rates":    [],
        "eval_returns":          [],
    }

    tracker = BestModelTracker(save_dir, higher_is_better=True)

    # ── Phase 0 ────────────────────────────────────────────────────────────
    rb_path = os.path.join(save_dir, "replay_buffer.npz")
    if os.path.exists(rb_path):
        if verbose:
            print("\n[Phase 0] Loading existing replay buffer...")
        rb = load_replay_buffer(rb_path, device=device)
        if verbose:
            print(f"  Loaded {len(rb)} transitions.")
    else:
        rb = ReplayBuffer(device=device)
        if verbose:
            print("\n[Phase 0] Collecting initial data with SB3...")
        phase0_collect_initial_data(
            task_distribution, rb,
            timesteps_per_task=timesteps_per_task,
            algo=algo, device=device, verbose=verbose,
        )
        save_replay_buffer(rb, rb_path)

    # ── Phase 1 ────────────────────────────────────────────────────────────
    if verbose:
        print("\n[Phase 1] Building common Morse skeleton...")
    skeleton = phase1_build_skeleton(
        rb, num_landmarks=num_landmarks,
        device=device, verbose=verbose,
    )
    metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
    n_sub = len(skeleton["critical_states"])
    if verbose:
        print(f"  Found {n_sub} critical state(s).")

    # Always save topology, even if no critical states found
    plot_skeleton_topology(
        skeleton, rb,
        os.path.join(save_dir, "topology_initial.png"),
    )

    if n_sub == 0:
        print("No subgoals found; aborting.")
        plot_training_curves(metrics, save_dir)
        return None, None, skeleton, metrics

    meta_policy     = None
    meta_value_net  = None
    sub_policies    = {}
    hitting_nets    = {}
    # Start True so Phase 2 always runs on the first iteration.
    sp_converged    = True

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─'*60}")

        # Phase 2 — re-train V_H only when sub-policies have converged to the
        # current shaping signal (or on the very first iteration).  Keeping V_H
        # frozen while sub-policies are still adapting prevents the shaping
        # reward from shifting under the learner's feet, which was the primary
        # cause of the large Phase 2→3 loss jump.
        if sp_converged or not hitting_nets:
            if verbose:
                reason = "first iteration" if not hitting_nets else "sub-policies converged"
                print(f"[Phase 2] Training hitting-time value nets ({reason})...")
            hitting_nets, p2_losses = phase2_train_value_net(
                skeleton, rb, gamma=gamma, device=device, verbose=verbose,
            )
            # Freeze weights so no code path can accidentally update V_H
            # while sub-policies are adapting to its shaping signal.
            for net in hitting_nets.values():
                net.requires_grad_(False)
            metrics["phase2_losses"].append(p2_losses)
            sp_converged = False   # sub-policies must re-converge to new V_H
        else:
            if verbose:
                print("[Phase 2] Skipping — V_H frozen until sub-policies converge.")
            metrics["phase2_losses"].append([])

        # Phase 3
        if verbose:
            print("[Phase 3] Training sub-policies...")
        sub_policies, p3_pi, p3_v = phase3_train_sub_policies(
            skeleton, rb, hitting_nets,
            task_distribution=task_distribution,
            state_dim=state_dim, action_dim=action_dim,
            gamma=gamma, epochs=sub_epochs,
            device=device, verbose=verbose,
            existing_sub_policies=sub_policies,
            carry_over_policies=True,
        )
        metrics["phase3_pi_losses"].append(p3_pi)
        metrics["phase3_v_losses"].append(p3_v)

        # Check whether sub-policies have converged; if so, Phase 2 will
        # re-train V_H on the next iteration with the enlarged buffer.
        sp_converged = check_sub_policy_convergence(
            sub_policies, task_distribution, device=device,
        )
        if verbose:
            print(f"  Sub-policy convergence check: "
                  f"{'converged — V_H will be updated next iteration' if sp_converged else 'not yet converged — V_H stays frozen'}")

        # Phase 4
        if verbose:
            print("[Phase 4] Training meta-policy...")
        meta_policy, meta_value_net, p4_returns = phase4_train_meta_policy(
            sub_policies, skeleton, task_distribution,
            state_dim=state_dim, gamma=gamma,
            meta_epochs=meta_epochs,
            total_iterations=num_iterations,
            current_iteration=iteration,
            device=device, verbose=verbose,
        )
        metrics["phase4_returns"].append(p4_returns)

        if meta_policy is None:
            continue

        # Evaluate
        if verbose:
            print("[Eval] Evaluating meta-policy...")
        eval_result = evaluate_policy(
            meta_policy, sub_policies, skeleton, task_distribution,
            n_episodes=eval_episodes, gamma=gamma, device=device,
        )
        metrics["eval_success_rates"].append(eval_result["success_rate"])
        metrics["eval_returns"].append(eval_result["avg_return"])
        if verbose:
            print(f"  success_rate={eval_result['success_rate']:.1%}  "
                  f"avg_return={eval_result['avg_return']:.4f}")
            for env_name, sr in eval_result["per_env"].items():
                print(f"    {env_name}: {sr:.1%}")

        # Checkpoint
        ckpt_dir = save_checkpoint(
            save_dir, iteration=iteration,
            meta_policy=meta_policy,
            meta_value_net=meta_value_net,
            sub_policies=sub_policies,
            hitting_nets=hitting_nets,
            skeleton_data=skeleton,
            replay_buffer=rb,
            metrics={
                "eval": eval_result,
                "p4_avg_return": float(np.mean(p4_returns)) if p4_returns else 0.0,
            },
        )
        improved = tracker.update(eval_result["success_rate"], ckpt_dir)
        if verbose and improved:
            print(f"  ★ New best model (success_rate="
                  f"{eval_result['success_rate']:.1%})")

        # Visualize
        save_iteration_visuals(skeleton, rb, metrics,
                               save_dir, iteration)

        # Collect with trained policy
        if verbose:
            print("[Collect] Rolling out meta-policy to gather more data...")
        collect_with_meta_policy(
            meta_policy, sub_policies, task_distribution,
            skeleton, rb,
            num_episodes=collect_episodes, device=device,
        )
        save_replay_buffer(rb, rb_path)
        if verbose:
            print(f"  Buffer size: {len(rb)}")

        # Periodic refinement
        if (iteration + 1) % refine_every == 0 and iteration < num_iterations - 1:
            if verbose:
                print("[Refine] Rebuilding skeleton on enlarged buffer...")
            skeleton = refine_skeleton(
                skeleton, rb,
                num_landmarks=num_landmarks,
                device=device, verbose=verbose,
            )
            metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
            n_sub = len(skeleton["critical_states"])
            if verbose:
                print(f"  Refined skeleton has {n_sub} critical state(s).")
            if n_sub == 0:
                print("  No subgoals after refinement; stopping.")
                break

    # Final training curves
    plot_training_curves(metrics, save_dir)

    # Run demos using best saved model if available
    best_dir = os.path.join(save_dir, "best")
    if os.path.isdir(best_dir) and n_demos > 0:
        if verbose:
            print(f"\n[Demo] Running {n_demos} demos with best model...")
        try:
            best_ckpt = load_checkpoint(best_dir, device=device)
            (demo_mp, _, demo_sp, _, demo_skel) = restore_models(
                best_ckpt, state_dim, action_dim, device=device,
            )
            demo_dir = os.path.join(save_dir, "demos")
            run_demos(
                demo_mp, demo_sp, demo_skel, task_distribution,
                save_dir=demo_dir, n_demos=n_demos,
                render=True, gamma=gamma, device=device,
            )
        except Exception as e:
            print(f"  [Demo] Could not run demos: {e}")

    if verbose:
        print("\nMeta-RL pipeline complete.")

    return meta_policy, sub_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────

_DEFAULT_TASKS = ["reach-v3", "push-v3",]#"pick-place-v3", "door-open-v3"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-RL on MetaWorld")
    parser.add_argument("--tasks",       nargs="+", default=_DEFAULT_TASKS,
                        metavar="ENV",
                        help="MetaWorld env names to train on "
                             "(default: reach-v3 push-v3 pick-place-v3 door-open-v3)")
    parser.add_argument("--max-tasks",   type=int, default=2,
                        help="Goal variants per env type")
    parser.add_argument("--iterations",  type=int, default=1)
    parser.add_argument("--landmarks",   type=int, default=100)
    parser.add_argument("--meta-epochs", type=int, default=200)
    parser.add_argument("--sub-epochs",  type=int, default=500)
    parser.add_argument("--timesteps",   type=int, default=5_000,
                        help="SB3 timesteps per task in Phase 0")
    parser.add_argument("--algo",        default="SAC", choices=["SAC", "PPO"])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--n-demos",     type=int, default=5)
    parser.add_argument("--save-dir",    default="results/meta_rl",
                        help="Directory for checkpoints, metrics and visuals")
    parser.add_argument("--load",        default=None, metavar="CKPT_DIR",
                        help="Load checkpoint from this directory and resume/demo")
    parser.add_argument("--demo-only",   action="store_true",
                        help="Skip training; run demos from --load checkpoint")
    parser.add_argument("--no-render",   action="store_true",
                        help="Disable rendering in demo mode")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    dist = MetaWorldTaskDistribution.from_env_names(args.tasks,
                                                    max_tasks_per_env=args.max_tasks)
    print(f"Task distribution: {len(dist.tasks)} tasks  "
          f"({', '.join(args.tasks)})")

    if args.demo_only:
        if args.load is None:
            parser.error("--demo-only requires --load <checkpoint_dir>")
        print(f"Loading checkpoint from {args.load} ...")
        ckpt = load_checkpoint(args.load, device=args.device)
        mp, mv, sp, hn, skel = restore_models(
            ckpt, MW_STATE_DIM, MW_ACTION_DIM, device=args.device,
        )
        demo_dir = os.path.join(args.save_dir, "demos")
        run_demos(
            mp, sp, skel, dist,
            save_dir=demo_dir,
            n_demos=args.n_demos,
            render=not args.no_render,
            device=args.device,
        )
    else:
        main_meta_rl_loop(
            task_distribution=dist,
            state_dim=MW_STATE_DIM,
            action_dim=MW_ACTION_DIM,
            num_landmarks=args.landmarks,
            num_iterations=args.iterations,
            timesteps_per_task=args.timesteps,
            gamma=0.99,
            sub_epochs=args.sub_epochs,
            meta_epochs=args.meta_epochs,
            algo=args.algo,
            eval_episodes=args.eval_episodes,
            n_demos=args.n_demos,
            save_dir=args.save_dir,
            device=args.device,
            verbose=True,
        )
