"""Speed-focused evaluation on a fixed set of seeded random circuits.

The historical evaluator in :mod:`training.dqn.evaluate` remains the fixed
benchmark.  This module adds a larger, reproducible suite for comparing driving
speed without changing physics, track geometry, step duration, or termination
rules.  Tracks come from :func:`visualization.circuits.random_circuit`; every
candidate sees the same seed list, start fractions, and step limit.

Metric definitions
------------------
``lap_completion_rate``
    Completed laps / episodes.
``completed_lap_time_s``
    Mean of the environment's reported ``lap_time`` over completed laps only.
``completed_lap_speed_mps``
    Mean of ``track_length / lap_time`` over completed laps only.  This is the
    primary speed metric; it is uncapped and excludes failed episodes.
``mean_forward_speed_mps_all``
    Mean of ``max(0, forward_distance) / elapsed_seconds`` over *all* episodes,
    so crashes and timeouts cannot be hidden by averaging only successes.
``peak_speed_mps``
    Mean of each episode's maximum instantaneous speed (supporting context).

Selection protocol
------------------
First require a candidate not to regress against the baseline's fixed-benchmark
lap completion rate.  Then rank by ``completed_lap_speed_mps`` on the tuning
suite, and confirm promising candidates on the disjoint held-out suite.  The
held-out seeds must not be used for training, tuning, or parent selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from training.dqn.evaluate import _load_checkpoint
from utils.car_rl.environment import CarTrackEnv
from visualization.circuits import random_circuit

TUNE_SEEDS = tuple(range(1000, 1012))
HOLDOUT_SEEDS = tuple(range(2000, 2016))
START_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
MAX_STEPS = 3000
SUITES: dict[str, tuple[int, ...]] = {"tune": TUNE_SEEDS, "holdout": HOLDOUT_SEEDS}

SUITE_DEFINITION = {
    "generator": "visualization.circuits.random_circuit",
    "width_m": 12.0,
    "start_fractions": list(START_FRACTIONS),
    "start_rule": "s0 = start_fraction * track.length; reset(seed=0)",
    "max_steps": MAX_STEPS,
    "dt_s": 0.05,
    "no_progress_timeout_s": 10.0,
    "suites": {name: list(seeds) for name, seeds in SUITES.items()},
    "tune_seeds": list(TUNE_SEEDS),
    "holdout_seeds": list(HOLDOUT_SEEDS),
}


def build_tracks(seeds: Iterable[int]):
    """Return one deterministically generated track per seed."""
    return [(int(seed), random_circuit(np.random.default_rng(int(seed)), int(seed))) for seed in seeds]


def _run_episode(model, actions: np.ndarray, obs_dim: int, track, seed: int,
                 start_fraction: float, max_steps: int) -> dict[str, Any]:
    env = CarTrackEnv(track=track, observe_dynamics=(obs_dim != 7), random_start=False,
                      max_steps=max_steps)
    try:
        start_s = env.track.length * float(start_fraction)
        observation, _ = env.reset(seed=0, options={"s0": start_s})
        terminated = truncated = False
        peak_speed = 0.0
        total_reward = 0.0
        while not (terminated or truncated):
            with torch.inference_mode():
                q_values = model(torch.from_numpy(observation).unsqueeze(0))
                action_index = int(q_values.argmax(dim=1).item())
            observation, reward, terminated, truncated, info = env.step(actions[action_index])
            total_reward += float(reward)
            peak_speed = max(peak_speed, float(info["speed"]))

        forward_distance = max(0.0, float(env.progress))
        progress_fraction = float(np.clip(forward_distance / env.track.length, 0.0, 1.0))
        elapsed = max(float(info["time"]), env.dt)
        completed = bool(info["lap_completed"])
        lap_time = float(info["lap_time"]) if completed else None
        lap_speed = float(env.track.length / lap_time) if completed and lap_time else None
        return {
            "seed": int(seed),
            "track": track.name,
            "track_length_m": float(env.track.length),
            "start_fraction": float(start_fraction),
            "lap_completed": completed,
            "collision": bool(info["collision"]),
            "truncated": bool(truncated),
            "progress": progress_fraction,
            "forward_distance_m": forward_distance,
            "elapsed_seconds": elapsed,
            "mean_forward_speed_mps": forward_distance / elapsed,
            "completed_lap_time_s": lap_time,
            "completed_lap_speed_mps": lap_speed,
            "peak_speed_mps": peak_speed,
            "return": total_reward,
        }
    finally:
        env.close()


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["lap_completed"]]
    lap_times = [row["completed_lap_time_s"] for row in completed]
    lap_speeds = [row["completed_lap_speed_mps"] for row in completed]
    return {
        "episodes": len(rows),
        "completed_laps": len(completed),
        "lap_completion_rate": float(np.mean([row["lap_completed"] for row in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([row["collision"] for row in rows])) if rows else 0.0,
        "truncation_rate": float(np.mean([row["truncated"] for row in rows])) if rows else 0.0,
        "mean_progress": float(np.mean([row["progress"] for row in rows])) if rows else 0.0,
        "mean_forward_speed_mps_all": float(np.mean([row["mean_forward_speed_mps"] for row in rows])) if rows else 0.0,
        "completed_lap_time_s": float(np.mean(lap_times)) if lap_times else None,
        "completed_lap_time_median_s": float(np.median(lap_times)) if lap_times else None,
        "completed_lap_speed_mps": float(np.mean(lap_speeds)) if lap_speeds else None,
        "completed_lap_speed_median_mps": float(np.median(lap_speeds)) if lap_speeds else None,
        "peak_speed_mps": float(np.mean([row["peak_speed_mps"] for row in rows])) if rows else 0.0,
    }


def evaluate_suite(checkpoint_path: str | Path, suite: str = "tune",
                   seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Evaluate one checkpoint on a fixed random-circuit suite."""
    if seeds is None:
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
        seeds = SUITES[suite]
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    checkpoint, model, actions = _load_checkpoint(checkpoint_path)
    obs_dim = int(checkpoint.get("obs_dim", 10))
    rows = [
        _run_episode(model, actions, obs_dim, track, seed, fraction, MAX_STEPS)
        for seed, track in build_tracks(seeds)
        for fraction in START_FRACTIONS
    ]
    summary = _summary(rows)
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = _summary([row for row in rows if row["seed"] == seed])
    return {
        "checkpoint": str(checkpoint_path),
        "model_identifier": Path(checkpoint_path).name,
        "global_step": checkpoint.get("global_step"),
        "suite": suite,
        "seeds": list(seeds),
        "suite_definition": SUITE_DEFINITION,
        "summary": summary,
        "per_seed": per_seed,
        "results": rows,
    }


def _key(row: dict[str, Any]) -> tuple[int, float]:
    return int(row["seed"]), float(row["start_fraction"])


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Paired aggregate and per-episode comparison of two suite reports."""
    baseline_rows = {_key(row): row for row in baseline["results"]}
    candidate_rows = {_key(row): row for row in candidate["results"]}
    shared = sorted(baseline_rows.keys() & candidate_rows.keys())
    paired = []
    for key in shared:
        old, new = baseline_rows[key], candidate_rows[key]
        paired.append({
            "seed": key[0],
            "start_fraction": key[1],
            "completed_delta": int(new["lap_completed"]) - int(old["lap_completed"]),
            "lap_time_delta_s": (new["completed_lap_time_s"] - old["completed_lap_time_s"])
            if new["completed_lap_time_s"] is not None and old["completed_lap_time_s"] is not None else None,
            "lap_speed_delta_mps": (new["completed_lap_speed_mps"] - old["completed_lap_speed_mps"])
            if new["completed_lap_speed_mps"] is not None and old["completed_lap_speed_mps"] is not None else None,
            "mean_forward_speed_delta_mps": new["mean_forward_speed_mps"] - old["mean_forward_speed_mps"],
        })
    old_summary, new_summary = baseline["summary"], candidate["summary"]
    return {
        "baseline": baseline.get("checkpoint"),
        "candidate": candidate.get("checkpoint"),
        "paired_episodes": len(paired),
        "summary_delta": {
            name: (new_summary[name] - old_summary[name])
            if new_summary[name] is not None and old_summary[name] is not None else None
            for name in ("lap_completion_rate", "mean_progress", "mean_forward_speed_mps_all",
                         "completed_lap_time_s", "completed_lap_speed_mps", "peak_speed_mps")
        },
        "paired": paired,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="DQN checkpoint file")
    parser.add_argument("--suite", choices=sorted(SUITES), default="tune", help="fixed seed suite (default: tune)")
    parser.add_argument("--seeds", help="comma-separated seed override (for smoke tests)")
    parser.add_argument("--previous", type=Path, help="optional earlier report JSON to compare against")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip()) if args.seeds else None
    report = evaluate_suite(args.checkpoint, suite=args.suite, seeds=seeds)
    if args.previous is not None:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(previous, report)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
