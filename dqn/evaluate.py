"""Deterministic, standardized evaluation for saved DQN policies.

Run with ``python -m dqn.evaluate path/to/checkpoint.pt``. The evaluator runs
each of the three built-in tracks from four quarter-lap offsets and emits one
JSON document containing per-episode and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from car_rl.environment import CarTrackEnv


TRACKS = ("circuit", "oval", "clover")
START_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
FORWARD_SPEED_REFERENCE = 20.0


def _linear_layers(state_dict: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Return saved 2-D weights in model order, independent of module naming."""
    weights = [(key, value) for key, value in state_dict.items()
               if key.endswith("weight") and isinstance(value, torch.Tensor) and value.ndim == 2]

    def order(item: tuple[str, torch.Tensor]) -> tuple[Any, ...]:
        key = item[0]
        # ModuleList/Sequential paths contain layer indices. Keep the original
        # key as a deterministic tie breaker for custom but ordered names.
        numbers = tuple(int(part) for part in key.split(".") if part.isdigit())
        return (numbers, key)

    return sorted(weights, key=order)


def _build_q_network(state_dict: dict[str, Any], obs_dim: int, action_count: int) -> nn.Sequential:
    """Recreate a ReLU MLP and copy its Linear tensors by layer order."""
    saved_weights = _linear_layers(state_dict)
    if not saved_weights:
        raise ValueError("Checkpoint online_state_dict contains no Linear layer weights")
    dims = [int(saved_weights[0][1].shape[1])]
    for key, weight in saved_weights:
        if int(weight.shape[1]) != dims[-1]:
            raise ValueError(f"Non-contiguous Linear dimensions at {key!r}")
        dims.append(int(weight.shape[0]))
    if dims[0] != obs_dim or dims[-1] != action_count:
        raise ValueError(
            f"Checkpoint dimensions ({dims[0]} inputs, {dims[-1]} outputs) do not match "
            f"observation/action dimensions ({obs_dim}, {action_count})"
        )

    layers: list[nn.Module] = []
    linears: list[nn.Linear] = []
    for index, (in_features, out_features) in enumerate(zip(dims, dims[1:])):
        linear = nn.Linear(in_features, out_features)
        layers.append(linear)
        linears.append(linear)
        if index < len(dims) - 2:
            layers.append(nn.ReLU())

    with torch.no_grad():
        for layer, (key, weight) in zip(linears, saved_weights):
            layer.weight.copy_(weight)
            bias_key = key[:-len("weight")] + "bias"
            bias = state_dict.get(bias_key)
            if bias is None:
                layer.bias.zero_()
            else:
                layer.bias.copy_(bias)
    return nn.Sequential(*layers).eval()


def _load_checkpoint(path: str | Path) -> tuple[dict[str, Any], nn.Sequential, np.ndarray]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a mapping")

    state_dict = checkpoint.get("online_state_dict", checkpoint.get("net"))
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing online_state_dict")
    actions = checkpoint.get("actions")
    if actions is None:
        raise ValueError("Checkpoint is missing its discretized actions array")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) == 0:
        raise ValueError("Checkpoint actions must have shape (N, 2)")

    config = checkpoint.get("config", {})
    environment_config = config.get("environment", {}) if isinstance(config, dict) else {}
    obs_dim = int(checkpoint.get("obs_dim", 10 if environment_config.get("observe_dynamics", True) else 7))
    model = _build_q_network(state_dict, obs_dim, len(actions))
    return checkpoint, model, actions


def _evaluate_episode(
    model: nn.Module,
    actions: np.ndarray,
    track: str,
    start_fraction: float,
    obs_dim: int,
) -> dict[str, Any]:
    env = CarTrackEnv(track=track, observe_dynamics=(obs_dim != 7), random_start=False)
    try:
        start_s = env.track.length * start_fraction
        observation, _ = env.reset(seed=0, options={"s0": start_s})
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.inference_mode():
                q_values = model(torch.from_numpy(observation).unsqueeze(0))
                action_index = int(q_values.argmax(dim=1).item())
            observation, reward, terminated, truncated, info = env.step(actions[action_index])
            total_reward += reward

        # The environment's progress is signed and may include small backward
        # motion. Score only net forward progress, bounded to one lap.
        forward_distance = max(0.0, float(env.progress))
        progress_fraction = float(np.clip(forward_distance / env.track.length, 0.0, 1.0))
        elapsed = max(float(info["time"]), env.dt)
        mean_forward_speed = max(0.0, forward_distance / elapsed)
        speed_score = float(np.clip(mean_forward_speed / FORWARD_SPEED_REFERENCE, 0.0, 1.0))
        score = 0.8 * progress_fraction + 0.2 * speed_score
        return {
            "track": track,
            "start_fraction": start_fraction,
            "lap_completed": bool(info["lap_completed"]),
            "collision": bool(info["collision"]),
            "truncated": bool(truncated),
            "progress": progress_fraction,
            "forward_distance_m": forward_distance,
            "elapsed_seconds": elapsed,
            "mean_forward_speed_mps": mean_forward_speed,
            "return": float(total_reward),
            "score": float(score),
        }
    finally:
        env.close()


def evaluate(checkpoint_path: str | Path) -> dict[str, Any]:
    """Evaluate a checkpoint on all 12 fixed track/start combinations."""
    torch.set_num_threads(1)
    checkpoint, model, actions = _load_checkpoint(checkpoint_path)
    obs_dim = int(checkpoint.get("obs_dim", 10))
    episodes = [
        _evaluate_episode(model, actions, track, fraction, obs_dim)
        for track in TRACKS
        for fraction in START_FRACTIONS
    ]

    scores = [row["score"] for row in episodes]
    per_track = {}
    for track in TRACKS:
        rows = [row for row in episodes if row["track"] == track]
        per_track[track] = {
            "mean_score": float(np.mean([row["score"] for row in rows])),
            "mean_progress": float(np.mean([row["progress"] for row in rows])),
            "lap_completion_rate": float(np.mean([row["lap_completed"] for row in rows])),
            "collision_rate": float(np.mean([row["collision"] for row in rows])),
        }
    return {
        "checkpoint": str(checkpoint_path),
        "episodes": len(episodes),
        "score_definition": "0.8 * clip(forward_distance / track_length, 0, 1) + 0.2 * clip((forward_distance / elapsed_seconds) / 20, 0, 1)",
        "mean_score": float(np.mean(scores)),
        "mean_progress": float(np.mean([row["progress"] for row in episodes])),
        "lap_completion_rate": float(np.mean([row["lap_completed"] for row in episodes])),
        "collision_rate": float(np.mean([row["collision"] for row in episodes])),
        "truncation_rate": float(np.mean([row["truncated"] for row in episodes])),
        "per_track": per_track,
        "results": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="DQN checkpoint file")
    parser.add_argument("--output", type=Path, help="write JSON to this file (default: stdout)")
    args = parser.parse_args()
    result = evaluate(args.checkpoint)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
