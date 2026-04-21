import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from collections import defaultdict, deque
import warnings
import argparse
import pickle
from typing import cast
import h5py


warnings.filterwarnings("ignore")


class Collector:
    """
    Collect trajectories from a trained SB3 policy.

    Trajectories are stored as a flat buffer (states, actions, rewards) with
    an episode-pointer array (ptrs) that marks the exclusive end of each
    episode in the flat arrays:

        episode i  →  states [ptrs[i-1] : ptrs[i]+1]   (includes terminal obs)
                       actions[ptrs[i-1] : ptrs[i]]
                       rewards[ptrs[i-1] : ptrs[i]]
    """

    def __init__(self, env_id, algo, model_path, max_length=1000, seed=42):
        self.env_id = env_id
        self.algo = algo
        self.model_path = model_path
        self.max_length = max_length
        self.seed = seed

        np.random.seed(seed)

        self.env = gym.make(env_id)

        algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG}
        self.model = algo_map[algo].load(model_path)

        # Flat trajectory buffers
        self.states = []  # length = total_steps + n_episodes (includes terminal obs)
        self.actions = []  # length = total_steps
        self.rewards = []  # length = total_steps
        self.dones = []  # length = total_steps; True where terminated or truncated
        self.ptrs = []  # ptrs[i] = exclusive end index in actions/rewards for episode i
        self.ptr = 0  # running step count across all episodes

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_trajectories(self, n_episodes=100):
        """Roll out n_episodes and append results to the internal buffers."""
        print(f"Collecting {n_episodes} trajectories from {self.env_id}")

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            self.states.append(obs.copy())

            for _ in range(self.max_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)

                self.actions.append(action)
                self.rewards.append(reward)
                self.ptr += 1

                if terminated or truncated:
                    self.dones.append(True)
                    self.states.append(obs.copy())  # final state on episode end
                    break

                else:
                    self.dones.append(False)
                    self.states.append(obs.copy())  # final state on max_length

            self.ptrs.append(self.ptr)

        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "ptrs": np.array(self.ptrs),
            "dones": np.array(self.dones),
        }

    # ------------------------------------------------------------------
    # Add a single trajectory
    # ------------------------------------------------------------------

    def add(self, trajectory):
        """
        Append a single pre-collected trajectory to the buffer.

        Args:
            trajectory: dict with keys
                "states"  – array of shape (T+1, obs_dim), includes terminal obs
                "actions" – array of shape (T, act_dim)
                "rewards" – array of shape (T,)
                "dones"   – bool array of shape (T,); last element True if episode
                            ended naturally or was truncated (optional, defaults
                            to all-False except the last step which is set True)
        """
        states = np.asarray(trajectory["states"])
        actions = np.asarray(trajectory["actions"])
        rewards = np.asarray(trajectory["rewards"])

        T = len(actions)
        if len(states) != T + 1:
            raise ValueError(f"states must have length T+1={T + 1}, got {len(states)}")
        if len(rewards) != T:
            raise ValueError(f"rewards must have length T={T}, got {len(rewards)}")

        if "dones" in trajectory:
            dones = np.asarray(trajectory["dones"], dtype=bool)
            if len(dones) != T:
                raise ValueError(f"dones must have length T={T}, got {len(dones)}")
        else:
            dones = np.zeros(T, dtype=bool)
            if T > 0:
                dones[-1] = True

        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.ptr += T
        self.ptrs.append(self.ptr)

    # ------------------------------------------------------------------
    # Sample trajectories
    # ------------------------------------------------------------------

    def sample(self, n):
        """
        Sample n trajectories uniformly at random without replacement.

        Returns:
            list of dicts, each with keys "states", "actions", "rewards"
        """
        n_eps = len(self.ptrs)
        if n_eps == 0:
            raise RuntimeError("No trajectories in buffer.")
        if n > n_eps:
            raise ValueError(f"Requested {n} trajectories but only {n_eps} are stored.")

        states_arr = np.array(self.states)
        actions_arr = np.array(self.actions)
        rewards_arr = np.array(self.rewards)
        dones_arr = np.array(self.dones, dtype=bool)
        ptrs_arr = np.array(self.ptrs)

        indices = np.random.choice(n_eps, size=n, replace=False)
        result = []
        for idx in indices:
            # Flat start/end in actions, rewards, and dones
            act_start = int(ptrs_arr[idx - 1]) if idx > 0 else 0
            act_end = int(ptrs_arr[idx])

            # States array has one extra entry per preceding episode
            obs_start = act_start + idx
            obs_end = act_end + idx + 1

            result.append(
                {
                    "states": states_arr[obs_start:obs_end],
                    "actions": actions_arr[act_start:act_end],
                    "rewards": rewards_arr[act_start:act_end],
                    "dones": dones_arr[act_start:act_end],
                }
            )

        return result

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save buffer to disk. Format inferred from file extension:
          .pkl / .pickle  →  Python pickle
          .h5  / .hdf5    →  HDF5 (compressed flat datasets + ptrs)
        """
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("h5", "hdf5"):
            self._save_hdf5(path)
        else:
            self._save_pickle(path)

        print(f"Saved {len(self.ptrs)} episodes ({self.ptr} steps) to {path}")

    def _save_pickle(self, path):
        data = {
            "env_id": self.env_id,
            "algo": self.algo,
            "model_path": self.model_path,
            "max_length": self.max_length,
            "seed": self.seed,
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones, dtype=bool),
            "ptrs": np.array(self.ptrs),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _save_hdf5(self, path):
        with h5py.File(path, "w") as f:
            f.attrs["env_id"] = self.env_id
            f.attrs["algo"] = self.algo
            f.attrs["model_path"] = self.model_path
            f.attrs["max_length"] = self.max_length
            f.attrs["seed"] = self.seed
            f.create_dataset("states", data=np.array(self.states), compression="gzip")
            f.create_dataset("actions", data=np.array(self.actions), compression="gzip")
            f.create_dataset("rewards", data=np.array(self.rewards), compression="gzip")
            f.create_dataset("dones", data=np.array(self.dones, dtype=bool))
            f.create_dataset("ptrs", data=np.array(self.ptrs))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path):
        """
        Load a previously saved buffer into this instance, replacing any
        existing data.
        """
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("h5", "hdf5"):
            data = self._load_hdf5(path)
        else:
            data = self._load_pickle(path)

        self.env_id = data["env_id"]
        self.algo = data["algo"]
        self.model_path = data["model_path"]
        self.max_length = int(data["max_length"])
        self.seed = int(data["seed"])
        self.states = list(data["states"])
        self.actions = list(data["actions"])
        self.rewards = list(data["rewards"])
        self.dones = list(data["dones"])
        self.ptrs = list(data["ptrs"])
        self.ptr = int(self.ptrs[-1]) if self.ptrs else 0

        print(f"Loaded {len(self.ptrs)} episodes ({self.ptr} steps) from {path}")

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_hdf5(path):
        with h5py.File(path, "r") as f:
            return {
                "env_id": str(f.attrs["env_id"]),
                "algo": str(f.attrs["algo"]),
                "model_path": str(f.attrs["model_path"]),
                "max_length": int(cast(int, f.attrs["max_length"])),
                "seed": int(cast(int, f.attrs["seed"])),
                "states": np.asarray(cast(h5py.Dataset, f["states"])),
                "actions": np.asarray(cast(h5py.Dataset, f["actions"])),
                "rewards": np.asarray(cast(h5py.Dataset, f["rewards"])),
                "dones": np.asarray(cast(h5py.Dataset, f["dones"])),
                "ptrs": np.asarray(cast(h5py.Dataset, f["ptrs"])),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Ant-v5")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="trajectories.pkl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    collector = Collector(
        args.env,
        args.algo,
        args.model_path,
        max_length=args.max_length,
        seed=args.seed,
    )
    collector.collect_trajectories(args.episodes)
    collector.save(args.output)

    sample = collector.sample(min(3, args.episodes))
    print(f"\nSample of {len(sample)} trajectories:")
    for i, traj in enumerate(sample):
        T = len(traj["actions"])
        R = float(traj["rewards"].sum())
        print(f"  [{i}]  {T} steps,  return={R:.2f}")


if __name__ == "__main__":
    main()
