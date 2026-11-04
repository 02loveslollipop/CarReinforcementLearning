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
    """Fixed-capacity replay buffer with checkpoint-friendly terminal masks.

    ``dones`` stores true terminal states for the Bellman target. Episode
    truncations are reset boundaries but remain bootstrappable.
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.collisions = np.zeros(capacity, dtype=np.bool_)
        self.start_groups = np.full(capacity, -1, dtype=np.int16)
        self.start_group_counts: dict[int, int] = {}
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, collision=False, start_group=-1):
        i = self.position
        old_group = int(self.start_groups[i])
        if old_group >= 0:
            self.start_group_counts[old_group] -= 1
            if self.start_group_counts[old_group] == 0:
                del self.start_group_counts[old_group]
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.collisions[i] = collision
        self.start_groups[i] = int(start_group)
        if int(start_group) >= 0:
            group = int(start_group)
            self.start_group_counts[group] = self.start_group_counts.get(group, 0) + 1
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator,
               collision_fraction: float = 0.0, collision_penalty: float = 0.0,
               start_balance_fraction: float = 0.0):
        """Sample uniformly, optionally reserving part of a batch for crashes.

        Older checkpoints do not store collision flags, so those are inferred
        from terminal transitions whose reward contains most of the collision
        cost. New transitions distinguish crashes from ordinary time limits.
        """
        collision_fraction = float(np.clip(collision_fraction, 0.0, 1.0))
        indices = rng.integers(0, self.size, size=batch_size)
        collision_quota = 0
        if collision_fraction and collision_penalty > 0.0:
            collision_indices = np.flatnonzero(self.collisions[:self.size])
            n_collision = min(batch_size, int(round(batch_size * collision_fraction)))
            if collision_indices.size and n_collision:
                collision_quota = n_collision
                # Use every distinct crash once when possible, then sample with
                # replacement to fill the quota. This keeps crash feedback
                # effective early in training, when only a few collisions have
                # entered replay, while preserving diversity as the pool grows.
                n_distinct = min(n_collision, collision_indices.size)
                indices[:n_distinct] = rng.choice(collision_indices, size=n_distinct, replace=False)
                if n_distinct < n_collision:
                    indices[n_distinct:n_collision] = rng.choice(
                        collision_indices, size=n_collision - n_distinct, replace=True
                    )
                # Keep the rest of the batch non-collision when possible. A
                # plain uniform fill can silently exceed the requested crash
                # fraction (and repeatedly duplicate the same few crashes).
                non_collision_indices = np.flatnonzero(~self.collisions[:self.size])
                if non_collision_indices.size and n_collision < batch_size:
                    indices[n_collision:] = rng.choice(
                        non_collision_indices, size=batch_size - n_collision, replace=True
                    )
        # New transitions carry the track/start group used at reset. Drawing
        # part of each batch evenly across represented groups prevents long
        # episodes or an easy track from monopolizing replay updates. Legacy
        # transitions use group -1 and remain available through uniform draws.
        balance_count = int(round(batch_size * np.clip(start_balance_fraction, 0.0, 1.0)))
        if balance_count:
            valid_groups = np.fromiter(self.start_group_counts, dtype=np.int16)
            if collision_quota and np.any(~self.collisions[:self.size]):
                balance_count = min(balance_count, batch_size - collision_quota)
            if valid_groups.size and balance_count:
                group_slots = rng.integers(0, valid_groups.size, size=balance_count)
                balanced_indices = np.empty(balance_count, dtype=np.int64)
                replay_groups = self.start_groups[:self.size]
                eligible = ~self.collisions[:self.size] if collision_quota else None
                # Build each group's exact index pool once, then draw directly.
                # This is uniform conditional sampling without per-item rejection.
                for group_slot in np.unique(group_slots):
                    group = valid_groups[group_slot]
                    slots = np.flatnonzero(group_slots == group_slot)
                    members = np.flatnonzero(replay_groups == group)
                    if eligible is not None:
                        eligible_members = members[eligible[members]]
                        if eligible_members.size:
                            members = eligible_members
                    balanced_indices[slots] = rng.choice(members, size=slots.size, replace=True)
                replace_slots = np.arange(collision_quota, batch_size)
                rng.shuffle(replace_slots)
                n_replace = min(balance_count, len(replace_slots))
                indices[replace_slots[:n_replace]] = balanced_indices[:n_replace]
        return (
            self.states[indices], self.actions[indices], self.rewards[indices],
            self.next_states[indices], self.dones[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity, "obs_dim": self.obs_dim,
            "states": self.states, "actions": self.actions, "rewards": self.rewards,
            "next_states": self.next_states, "dones": self.dones,
            "collisions": self.collisions, "start_groups": self.start_groups,
            "position": self.position, "size": self.size,
        }

    @classmethod
    def from_state_dict(cls, data, collision_penalty: float = 50.0):
        buffer = cls(data["capacity"], data["obs_dim"])
        for name in ("states", "actions", "rewards", "next_states", "dones"):
            setattr(buffer, name, np.asarray(data[name]).copy())
        if "collisions" in data:
            buffer.collisions = np.asarray(data["collisions"], dtype=np.bool_).copy()
        else:
            # Backward-compatible migration for checkpoints created before
            # collision labels were stored independently from terminal flags.
            buffer.collisions = (
                (buffer.dones > 0.5)
                & (buffer.rewards <= -0.5 * float(collision_penalty))
            )
        if "start_groups" in data:
            buffer.start_groups = np.asarray(data["start_groups"], dtype=np.int16).copy()
        buffer.position, buffer.size = int(data["position"]), int(data["size"])
        for group in buffer.start_groups[:buffer.size]:
            group = int(group)
            if group >= 0:
                buffer.start_group_counts[group] = buffer.start_group_counts.get(group, 0) + 1
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


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Replace a checkpoint only after its complete contents are on disk."""
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


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
                        global_step, episode, env_rng_state, rng, epsilon_step=0,
                        epsilon_schedule=None, env=None,
                        state=None, episode_return=0.0, episode_steps=0,
                        start_group=-1):
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
            "start_group": int(start_group),
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
        # Exploration has its own clock so a child of an older checkpoint can
        # run a fresh exploration schedule, while later continuations preserve it.
        "epsilon_step": int(epsilon_step),
        "epsilon_schedule": epsilon_schedule,
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
    track_sampling_weights = training.get("track_sampling_weights")
    configured_tracks = list(env.unwrapped.tracks)
    episode = 0
    current_start_group = -1
    start_anchors = training.get("start_fraction_anchors", [0.0, 0.25, 0.5, 0.75])
    if not isinstance(start_anchors, (list, tuple)) or not start_anchors:
        raise ValueError("start_fraction_anchors must be a non-empty list of fractions")
    start_anchors = [float(value) % 1.0 for value in start_anchors]
    start_jitter = float(training.get("start_fraction_jitter", 0.02))
    if not np.isfinite(start_jitter) or not 0.0 <= start_jitter <= 0.5:
        raise ValueError("start_fraction_jitter must be between 0 and 0.5")

    def episode_start_options(track, anchor_index=None):
        # Cycle through the evaluator's quarter-lap start points, with a small
        # reproducible random offset so the policy still sees varied starts.
        if anchor_index is None:
            anchor_index = episode % len(start_anchors)
        anchor = start_anchors[anchor_index % len(start_anchors)]
        fraction = (anchor + float(env.np_random.uniform(-start_jitter, start_jitter))) % 1.0
        return {"s0": fraction * track.length}

    if track_sampling_weights is not None:
        if not isinstance(track_sampling_weights, dict):
            raise ValueError("track_sampling_weights must map track names to weights")
        track_names = {track.name for track in configured_tracks}
        unknown_tracks = set(track_sampling_weights) - track_names
        if unknown_tracks:
            raise ValueError(f"track_sampling_weights contains unconfigured tracks: {sorted(unknown_tracks)}")
        weights = np.asarray(
            [float(track_sampling_weights.get(track.name, 0.0)) for track in configured_tracks],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("track_sampling_weights must be finite, non-negative, and have positive sum")
        track_probabilities = weights / weights.sum()
        # Equal track weights are common for evaluation-oriented training. In
        # that case cycle through the Cartesian product of tracks and start
        # anchors so a short run cannot accidentally neglect one combination.
        balanced_track_cycle = bool(np.allclose(track_probabilities, 1.0 / len(configured_tracks)))

        def reset_training_episode():
            nonlocal current_start_group
            # Keep the environment configured for all tracks outside reset.
            # When weights are equal, pair each track with each anchor in a
            # deterministic cycle that resumes from the checkpointed episode.
            if balanced_track_cycle:
                selected = episode % len(configured_tracks)
                anchor_index = (episode // len(configured_tracks)) % len(start_anchors)
            else:
                selected = int(env.np_random.choice(len(configured_tracks), p=track_probabilities))
                anchor_index = episode % len(start_anchors)
            current_start_group = selected * len(start_anchors) + anchor_index
            env.unwrapped.tracks = [configured_tracks[selected]]
            try:
                observation, reset_info = env.reset(
                    options=episode_start_options(configured_tracks[selected], anchor_index)
                )
            finally:
                env.unwrapped.tracks = configured_tracks
            return observation, reset_info

        initial_obs, _ = reset_training_episode()
    else:
        def reset_training_episode():
            nonlocal current_start_group
            # Cycle through track/start pairs. This preserves uniform coverage
            # while keeping assignment reproducible across continuations.
            selected = episode % len(configured_tracks)
            anchor_index = (episode // len(configured_tracks)) % len(start_anchors)
            current_start_group = selected * len(start_anchors) + anchor_index
            selected_track = configured_tracks[selected]
            env.unwrapped.tracks = [selected_track]
            try:
                return env.reset(options=episode_start_options(selected_track, anchor_index))
            finally:
                env.unwrapped.tracks = configured_tracks
        initial_obs, _ = reset_training_episode()
    obs_dim = int(env.observation_space.shape[0])
    actions = make_actions(training)
    force_bins = int(training.get("force_bins", 5))
    steering_bins = int(training.get("steering_bins", 5))
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
    epsilon_step = 0
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
        replay = ReplayBuffer.from_state_dict(
            checkpoint["replay"], float(reward_config["collision_penalty"])
        )
        global_step, episode = int(checkpoint["global_step"]), int(checkpoint["episode"])
        epsilon_step = int(checkpoint.get("epsilon_step", 0))
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
            initial_obs, _ = reset_training_episode()
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
            if "start_group" in active_episode:
                current_start_group = int(active_episode["start_group"])
            else:
                track_index = next(i for i, item in enumerate(configured_tracks)
                                   if item.name == active_episode["track"])
                anchor_index = ((episode // len(configured_tracks)) % len(start_anchors)
                                if track_sampling_weights is None or balanced_track_cycle
                                else episode % len(start_anchors))
                current_start_group = track_index * len(start_anchors) + anchor_index

    gamma = float(training.get("gamma", 0.99))
    batch_size = int(training.get("batch_size", 64))
    learning_starts = int(training.get("learning_starts", 1_000))
    train_frequency = max(1, int(training.get("train_frequency", 4)))
    target_update_frequency = max(1, int(training.get("target_update_frequency", 1_000)))
    epsilon_start = float(training.get("epsilon_start", 1.0))
    epsilon_end = float(training.get("epsilon_end", 0.05))
    epsilon_decay_steps = max(1, int(training.get("epsilon_decay_steps", 50_000)))
    epsilon_schedule = {
        "start": epsilon_start,
        "end": epsilon_end,
        "decay_steps": epsilon_decay_steps,
        "local_fraction": float(training.get("epsilon_local_fraction", 0.0)),
    }
    epsilon_local_fraction = float(epsilon_schedule["local_fraction"])
    if not 0.0 <= epsilon_local_fraction <= 1.0:
        raise ValueError("epsilon_local_fraction must be between 0 and 1")
    if parent_checkpoint is not None and parent_checkpoint.get("epsilon_schedule") != epsilon_schedule:
        # Start a new exploration phase when a child changes its schedule. The
        # stored schedule makes repeated continuations resume the same phase.
        epsilon_step = 0
    gradient_clip = float(training.get("gradient_clip", 10.0))
    lane_centering_weight = float(reward_config.get("lane_centering_weight", 0.0))
    slip_stability_weight = float(reward_config.get("slip_stability_weight", 0.0))
    slip_reference = max(float(reward_config.get("slip_reference", 0.3)), 1e-6)
    target_update_tau = float(training.get("target_update_tau", 0.0))
    if not 0.0 <= target_update_tau <= 1.0:
        raise ValueError("target_update_tau must be between 0 and 1")
    active_episode = parent_checkpoint.get("environment_state") if parent_checkpoint is not None else None
    state = (np.asarray(active_episode["observation"], dtype=np.float32).copy()
             if active_episode is not None else initial_obs)
    episode_return = float(active_episode.get("episode_return", 0.0)) if active_episode else 0.0
    episode_steps = int(active_episode.get("episode_steps", 0)) if active_episode else 0
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(0, int(training.get("checkpoint_every", 10_000)))
    collision_replay_fraction = float(training.get("collision_replay_fraction", 0.0))
    if not 0.0 <= collision_replay_fraction <= 1.0:
        raise ValueError("collision_replay_fraction must be between 0 and 1")

    try:
        for local_step in range(int(steps)):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
                0.0, 1.0 - (epsilon_step / epsilon_decay_steps)
            )
            with torch.no_grad():
                q_values = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                action_index = int(q_values.argmax(dim=1).item())
            if rng.random() < epsilon:
                # Most exploratory actions can stay close to the current
                # greedy command, allowing useful steering/force refinement
                # without making every exploration step a large perturbation.
                if rng.random() < epsilon_local_fraction:
                    force_index, steering_index = divmod(action_index, steering_bins)
                    nearby = [
                        force * steering_bins + steering
                        for force in range(max(0, force_index - 1), min(force_bins, force_index + 2))
                        for steering in range(max(0, steering_index - 1), min(steering_bins, steering_index + 2))
                        if force * steering_bins + steering != action_index
                    ]
                    action_index = int(rng.choice(nearby)) if nearby else int(rng.integers(len(actions)))
                else:
                    action_index = int(rng.integers(len(actions)))
            previous_progress = float(env.unwrapped.progress)
            previous_lateral = float(env.track.project(env.unwrapped.state[:2])[1])
            previous_u, previous_v = (float(value) for value in env.unwrapped.state[3:5])
            next_state, _environment_reward, terminated, truncated, info = env.step(actions[action_index])
            progress_delta = float(env.unwrapped.progress) - previous_progress
            reward = float(reward_config["forward_progress_weight"]) * progress_delta
            # Potential-based shaping rewards moving toward the track center and
            # discourages drifting toward the edge, using only existing state.
            if lane_centering_weight:
                half_width = max(float(env.track.half_width), 1e-6)
                old_offset = np.clip(previous_lateral / half_width, -1.0, 1.0)
                new_offset = np.clip(float(info["lateral"]) / half_width, -1.0, 1.0)
                old_potential = -lane_centering_weight * old_offset * old_offset
                new_potential = -lane_centering_weight * new_offset * new_offset
                reward += gamma * new_potential - old_potential
            # Potential-based slip shaping uses the already observed lateral
            # dynamics to reward recovery from unstable tire slip. Keeping it
            # as a discounted potential difference avoids imposing a constant
            # cost on every turn or favoring one track's geometry.
            if slip_stability_weight:
                previous_slip = np.arctan2(previous_v, max(previous_u, 1.0))
                next_u, next_v = (float(value) for value in env.unwrapped.state[3:5])
                next_slip = np.arctan2(next_v, max(next_u, 1.0))
                old_slip = np.clip(previous_slip / slip_reference, -1.0, 1.0)
                new_slip = np.clip(next_slip / slip_reference, -1.0, 1.0)
                old_potential = -slip_stability_weight * old_slip * old_slip
                new_potential = -slip_stability_weight * new_slip * new_slip
                reward += gamma * new_potential - old_potential
            if progress_delta <= float(reward_config["inactivity_threshold"]):
                reward -= float(reward_config["inactivity_penalty"])
            if info.get("collision", False):
                reward -= float(reward_config["collision_penalty"])
            done = bool(terminated or truncated)
            collision = bool(info.get("collision", False))
            # Truncations are reset boundaries for data collection, but their
            # final observation is still a valid state for the Bellman target.
            # Mask only true terminal states (collision or completed lap).
            replay.add(state, action_index, reward, next_state, bool(terminated), collision,
                       current_start_group)
            state = next_state
            global_step += 1
            epsilon_step += 1
            episode_steps += 1
            episode_return += float(reward)

            if global_step >= learning_starts and replay.size >= batch_size and global_step % train_frequency == 0:
                states, action_ids, rewards, next_states, dones = replay.sample(
                    batch_size, rng, collision_replay_fraction,
                    float(reward_config["collision_penalty"]),
                    float(training.get("replay_start_balance_fraction", 0.0)),
                )
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
                if target_update_tau:
                    with torch.no_grad():
                        for target_parameter, online_parameter in zip(target.parameters(), model.parameters()):
                            target_parameter.lerp_(online_parameter, target_update_tau)

            if not target_update_tau and global_step % target_update_frequency == 0:
                target.load_state_dict(model.state_dict())

            if done:
                episode += 1
                print(f"episode={episode} step={global_step} length={episode_steps} return={episode_return:.3f}", flush=True)
                state, _ = reset_training_episode()
                episode_return, episode_steps = 0.0, 0

            if checkpoint_every and (local_step + 1) % checkpoint_every == 0:
                payload = _checkpoint_payload(
                    model=model, target=target, optimizer=optimizer, replay=replay,
                    config=config, actions=actions, global_step=global_step,
                    episode=episode, epsilon_step=epsilon_step,
                    epsilon_schedule=epsilon_schedule,
                    env_rng_state=env.np_random.bit_generator.state, rng=rng,
                    env=env, state=state, episode_return=episode_return,
                    episode_steps=episode_steps, start_group=current_start_group,
                )
                _save_checkpoint(payload, out_dir / "latest.pt")
    finally:
        env.close()

    payload = _checkpoint_payload(
        model=model, target=target, optimizer=optimizer, replay=replay,
        config=config, actions=actions, global_step=global_step,
        episode=episode, epsilon_step=epsilon_step,
        epsilon_schedule=epsilon_schedule,
        env_rng_state=env.np_random.bit_generator.state, rng=rng,
        env=env, state=state, episode_return=episode_return,
        episode_steps=episode_steps, start_group=current_start_group,
    )
    final_path = out_dir / "latest.pt"
    _save_checkpoint(payload, final_path)
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
