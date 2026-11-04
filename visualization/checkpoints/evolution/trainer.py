"""Train a DQN policy for the continuous-action CarTrack environment.

The two-dimensional action space is discretized into a Cartesian grid. A run's
configuration is a JSON object with optional ``environment`` and ``training``
objects; environment keys are passed to :class:`car_rl.CarTrackEnv`, ``reward``
keys configure progress/collision/inactivity shaping, and training keys configure
the learner. Checkpoints contain the learner, optimizer,
replay buffer, and random-number generator states so a later run can continue.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from car_rl import CarTrackEnv


class QNetwork(nn.Module):
    """Small fully connected network mapping observations to action values."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        widths = [int(obs_dim), *(int(width) for width in hidden_sizes), int(n_actions)]
        layers: list[nn.Module] = []
        for layer_index, (in_width, out_width) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(in_width, out_width))
            if layer_index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class ReplayBuffer:
    """Fixed-capacity replay buffer with a checkpoint-friendly representation."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices], self.actions[indices], self.rewards[indices],
            self.next_states[indices], self.dones[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity, "obs_dim": self.obs_dim,
            "states": self.states, "actions": self.actions, "rewards": self.rewards,
            "next_states": self.next_states, "dones": self.dones,
            "position": self.position, "size": self.size,
        }

    @classmethod
    def from_state_dict(cls, data):
        buffer = cls(data["capacity"], data["obs_dim"])
        for name in ("states", "actions", "rewards", "next_states", "dones"):
            setattr(buffer, name, np.asarray(data[name]).copy())
        buffer.position, buffer.size = int(data["position"]), int(data["size"])
        return buffer


def make_actions(training: dict[str, Any]) -> np.ndarray:
    """Make evenly spaced force/steering commands from the configured bin counts."""
    force_bins = int(training.get("force_bins", 5))
    steering_bins = int(training.get("steering_bins", 5))
    if force_bins < 2 or steering_bins < 2:
        raise ValueError("force_bins and steering_bins must each be at least 2")
    forces = np.linspace(-1.0, 1.0, force_bins, dtype=np.float32)
    steering = np.linspace(-1.0, 1.0, steering_bins, dtype=np.float32)
    return np.asarray([(force, steer) for force in forces for steer in steering], dtype=np.float32)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trainer checkpoint onto CPU (including optimizer and replay state)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _resolve_checkpoint(parent: str | Path) -> Path:
    path = Path(parent)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"parent checkpoint not found: {path}")
    return path


def _merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _checkpoint_payload(*, model, target, optimizer, replay, config, actions,
                        global_step, episode, env_rng_state, rng, env=None,
                        state=None, episode_return=0.0, episode_steps=0):
    environment_state = None
    if env is not None and state is not None:
        environment_state = {
            "track": env.track.name,
            "car_state": env.unwrapped.state.copy(),
            "previous_s": float(env.unwrapped._prev_s),
            "progress": float(env.unwrapped.progress),
            "best_progress": float(env.unwrapped.best_progress),
            "steps": int(env.unwrapped.steps),
            "steps_without_improvement": int(env.unwrapped.steps_without_improvement),
            "observation": np.asarray(state, dtype=np.float32).copy(),
            "episode_return": float(episode_return),
            "episode_steps": int(episode_steps),
        }
    return {
        "format_version": 2,
        "online_state_dict": model.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay": replay.state_dict(),
        "config": config,
        "actions": actions,
        "obs_dim": replay.obs_dim,
        "global_step": global_step,
        "episode": episode,
        "environment_state": environment_state,
        "rng": {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "generator": rng.bit_generator.state,
            "environment": env_rng_state,
        },
    }


def train(config: dict[str, Any], output: str | Path, steps: int,
          parent: str | Path | None = None) -> Path:
    """Run ``steps`` additional environment transitions and save ``latest.pt``."""
    config = json.loads(json.dumps(config))  # normalize tuples/NumPy values for JSON
    parent_checkpoint = load_checkpoint(_resolve_checkpoint(parent)) if parent is not None else None
    if parent_checkpoint is not None:
        # Unspecified settings inherit from the parent run, while explicit settings
        # can change environment/reward and optimizer/schedule hyperparameters.
        config = _merge_config(parent_checkpoint.get("config", {}), config)
    environment_config = dict(config.get("environment", {}))
    training = dict(config.get("training", {}))
    reward_config = {
        "forward_progress_weight": 1.0,
        "collision_penalty": 10.0,
        "inactivity_penalty": 0.01,
        "inactivity_threshold": 1e-4,
        **config.get("reward", {}),
    }
    seed = int(training.get("seed", config.get("seed", 0)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)

    env = CarTrackEnv(**environment_config)
    initial_obs, _ = env.reset(seed=seed)
    obs_dim = int(env.observation_space.shape[0])
    actions = make_actions(training)
    hidden_sizes = tuple(training.get("hidden_sizes", (128, 128)))
    model = QNetwork(obs_dim, len(actions), hidden_sizes).cpu()
    target = QNetwork(obs_dim, len(actions), hidden_sizes).cpu()
    target.load_state_dict(model.state_dict())
    learning_rate = float(training.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay_capacity = int(training.get("replay_capacity", 100_000))
    if replay_capacity < 1:
        raise ValueError("replay_capacity must be positive")
    replay = ReplayBuffer(replay_capacity, obs_dim)
    global_step = 0
    episode = 0

    if parent_checkpoint is not None:
        checkpoint = parent_checkpoint
        if int(checkpoint["obs_dim"]) != obs_dim:
            raise ValueError(f"checkpoint observation size {checkpoint['obs_dim']} != {obs_dim}")
        if not np.array_equal(np.asarray(checkpoint["actions"]), actions):
            raise ValueError("configured action bins do not match checkpoint action grid")
        model.load_state_dict(checkpoint["online_state_dict"])
        target.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        replay = ReplayBuffer.from_state_dict(checkpoint["replay"])
        global_step, episode = int(checkpoint["global_step"]), int(checkpoint["episode"])
        saved_rng = checkpoint.get("rng", {})
        if "python" in saved_rng:
            random.setstate(saved_rng["python"])
        if "numpy_global" in saved_rng:
            np.random.set_state(saved_rng["numpy_global"])
        if "torch_cpu" in saved_rng:
            torch.set_rng_state(saved_rng["torch_cpu"])
        if "generator" in saved_rng:
            rng.bit_generator.state = saved_rng["generator"]
        # Restore the environment stream before resuming the saved active episode;
        # future randomized starts/tracks then continue from the saved stream.
        if "environment" in saved_rng:
            env.np_random.bit_generator.state = saved_rng["environment"]
        active_episode = checkpoint.get("environment_state")
        if active_episode is None:
            initial_obs, _ = env.reset()
        else:
            matching_tracks = [track for track in env.unwrapped.tracks
                               if track.name == active_episode["track"]]
            if not matching_tracks:
                raise ValueError(f"checkpoint track {active_episode['track']!r} is not configured")
            env.unwrapped.track = matching_tracks[0]
            env.unwrapped.state = np.asarray(active_episode["car_state"], dtype=float).copy()
            env.unwrapped._prev_s = float(active_episode["previous_s"])
            env.unwrapped.progress = float(active_episode["progress"])
            env.unwrapped.best_progress = float(active_episode["best_progress"])
            env.unwrapped.steps = int(active_episode["steps"])
            env.unwrapped.steps_without_improvement = int(active_episode["steps_without_improvement"])
            initial_obs = env.unwrapped._observe()

    gamma = float(training.get("gamma", 0.99))
    batch_size = int(training.get("batch_size", 64))
    learning_starts = int(training.get("learning_starts", 1_000))
    train_frequency = max(1, int(training.get("train_frequency", 4)))
    target_update_frequency = max(1, int(training.get("target_update_frequency", 1_000)))
    target_update_tau = float(training.get("target_update_tau", 0.0))
    if not 0.0 <= target_update_tau <= 1.0:
        raise ValueError("target_update_tau must be between 0 and 1")
    epsilon_start = float(training.get("epsilon_start", 1.0))
    epsilon_end = float(training.get("epsilon_end", 0.05))
    epsilon_decay_steps = max(1, int(training.get("epsilon_decay_steps", 50_000)))
    gradient_clip = float(training.get("gradient_clip", 10.0))
    active_episode = parent_checkpoint.get("environment_state") if parent_checkpoint is not None else None
    state = (np.asarray(active_episode["observation"], dtype=np.float32).copy()
             if active_episode is not None else initial_obs)
    episode_return = float(active_episode.get("episode_return", 0.0)) if active_episode else 0.0
    episode_steps = int(active_episode.get("episode_steps", 0)) if active_episode else 0
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(0, int(training.get("checkpoint_every", 10_000)))

    try:
        for local_step in range(int(steps)):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
                0.0, 1.0 - (global_step / epsilon_decay_steps)
            )
            if rng.random() < epsilon:
                action_index = int(rng.integers(len(actions)))
            else:
                with torch.no_grad():
                    q_values = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                    action_index = int(q_values.argmax(dim=1).item())
            previous_progress = float(env.unwrapped.progress)
            next_state, _environment_reward, terminated, truncated, info = env.step(actions[action_index])
            progress_delta = float(env.unwrapped.progress) - previous_progress
            reward = float(reward_config["forward_progress_weight"]) * progress_delta
            if progress_delta <= float(reward_config["inactivity_threshold"]):
                reward -= float(reward_config["inactivity_penalty"])
            if info.get("collision", False):
                reward -= float(reward_config["collision_penalty"])
            done = bool(terminated or truncated)
            replay.add(state, action_index, reward, next_state, done)
            state = next_state
            global_step += 1
            episode_steps += 1
            episode_return += float(reward)

            if global_step >= learning_starts and replay.size >= batch_size and global_step % train_frequency == 0:
                states, action_ids, rewards, next_states, dones = replay.sample(batch_size, rng)
                states_t = torch.as_tensor(states, dtype=torch.float32)
                action_t = torch.as_tensor(action_ids, dtype=torch.int64).unsqueeze(1)
                reward_t = torch.as_tensor(rewards, dtype=torch.float32)
                next_t = torch.as_tensor(next_states, dtype=torch.float32)
                done_t = torch.as_tensor(dones, dtype=torch.float32)
                current_q = model(states_t).gather(1, action_t).squeeze(1)
                with torch.no_grad():
                    # Double DQN reduces the positive bias from taking a max over
                    # noisy action values while keeping the evaluator policy unchanged.
                    next_actions = model(next_t).argmax(dim=1, keepdim=True)
                    next_q = target(next_t).gather(1, next_actions).squeeze(1)
                    target_q = reward_t + gamma * (1.0 - done_t) * next_q
                loss = nn.functional.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

                # Optional Polyak averaging keeps bootstrap targets from making
                # abrupt jumps. A zero tau preserves the original hard-update
                # schedule for older configurations and checkpoints.
                if target_update_tau > 0.0:
                    with torch.no_grad():
                        for target_parameter, online_parameter in zip(
                                target.parameters(), model.parameters()):
                            target_parameter.lerp_(online_parameter, target_update_tau)

            if target_update_tau == 0.0 and global_step % target_update_frequency == 0:
                target.load_state_dict(model.state_dict())

            if done:
                episode += 1
                print(f"episode={episode} step={global_step} length={episode_steps} return={episode_return:.3f}", flush=True)
                state, _ = env.reset()
                episode_return, episode_steps = 0.0, 0

            if checkpoint_every and (local_step + 1) % checkpoint_every == 0:
                payload = _checkpoint_payload(
                    model=model, target=target, optimizer=optimizer, replay=replay,
                    config=config, actions=actions, global_step=global_step,
                    episode=episode, env_rng_state=env.np_random.bit_generator.state, rng=rng,
                    env=env, state=state, episode_return=episode_return,
                    episode_steps=episode_steps,
                )
                torch.save(payload, out_dir / "latest.pt")
    finally:
        env.close()

    payload = _checkpoint_payload(
        model=model, target=target, optimizer=optimizer, replay=replay,
        config=config, actions=actions, global_step=global_step,
        episode=episode, env_rng_state=env.np_random.bit_generator.state, rng=rng,
        env=env, state=state, episode_return=episode_return,
        episode_steps=episode_steps,
    )
    final_path = out_dir / "latest.pt"
    torch.save(payload, final_path)
    return final_path


def _read_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("configuration JSON must contain an object")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON hyperparameter/environment configuration")
    parser.add_argument("--output", required=True, help="directory for latest.pt")
    parser.add_argument("--steps", type=int, required=True, help="additional environment steps")
    parser.add_argument("--parent", help="checkpoint file or directory to continue from")
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    result = train(_read_config(args.config), args.output, args.steps, args.parent)
    print(f"saved checkpoint: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
