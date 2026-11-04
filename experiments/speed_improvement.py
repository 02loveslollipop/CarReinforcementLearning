"""Controlled continuation experiments for faster, still-reliable DQN driving.

Each candidate starts from the same parent checkpoint, receives the same number
of additional environment steps, and is evaluated with the unchanged fixed
benchmark plus the tuning split of :mod:`training.dqn.speed_eval`.  The held-out
random-circuit split is intentionally not run here; evaluate finalists with
``python -m training.dqn.speed_eval ... --suite holdout``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from training.dqn.evaluate import evaluate
from training.dqn.speed_eval import evaluate_suite
from training.dqn.trainer import load_checkpoint, train

# Iteration 2: continuation stability first. The winner's lr=3e-4/train_freq=2 at
# epsilon floor collapses (iteration 1: 0/12). Vary one factor at a time from it:
# lower lr, slower training frequency, smaller batches, and reduced exploration.
VARIANTS: dict[str, dict] = {
    "control": {},
    "lr_1e5": {"training": {"learning_rate": 1e-5}},
    "lr_3e5": {"training": {"learning_rate": 3e-5}},
    "lr_1e4": {"training": {"learning_rate": 1e-4}},
    "tf_8": {"training": {"train_frequency": 8}},
    "tf_16": {"training": {"train_frequency": 16}},
    "tf_8_b64": {"training": {"train_frequency": 8, "batch_size": 64}},
    "lr_1e5_tf_8": {"training": {"learning_rate": 1e-5, "train_frequency": 8}},
    "lr_1e5_eps_end_001": {"training": {"learning_rate": 1e-5, "epsilon_end": 0.01}},
    "lr_1e5_tf_8_b64": {"training": {"learning_rate": 1e-5, "train_frequency": 8, "batch_size": 64}},
}
TRACKED_SOURCES = (
    "training/dqn/trainer.py",
    "training/dqn/evaluate.py",
    "training/dqn/speed_eval.py",
    "utils/car_rl/environment.py",
    "utils/car_rl/dynamics.py",
    "utils/car_rl/tracks.py",
    "visualization/circuits.py",
)


def _merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _code_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for name in TRACKED_SOURCES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def run_candidate(name: str, parent: str, steps: int, seed: int, run_dir: Path, root: Path,
                  clip: int = 1) -> dict:
    directory = run_dir / name / f"seed_{seed}"
    report_path = directory / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    overrides = VARIANTS[name]
    config = _merge({
        "training": {"seed": seed, "checkpoint_every": 0, "resume_rng": True},
    }, overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    started = time.perf_counter()
    checkpoint_path = train(config, directory / "checkpoint", steps, parent=parent)
    wall_seconds = time.perf_counter() - started
    checkpoint = load_checkpoint(checkpoint_path)
    fixed = evaluate(checkpoint_path)
    tune = evaluate_suite(checkpoint_path, suite="tune")
    report = {
        "variant": name,
        "seed": seed,
        "parent": str(parent),
        "checkpoint": str(checkpoint_path),
        "steps_requested": steps,
        "global_step": checkpoint.get("global_step"),
        "wall_seconds": wall_seconds,
        "code_hash": _code_hash(root),
        "config": checkpoint.get("config"),
        "fixed": fixed,
        "tune": tune,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def summarize(reports: list[dict], baseline_fixed_completion: float,
              baseline_tune: dict | None = None) -> dict:
    rows = []
    for report in reports:
        fixed = report["fixed"]
        tune = report["tune"]
        row = {
            "variant": report["variant"],
            "seed": report["seed"],
            "global_step": report["global_step"],
            "wall_seconds": report["wall_seconds"],
            "fixed_completion_rate": fixed["lap_completion_rate"],
            "fixed_mean_score": fixed["mean_score"],
            "fixed_mean_forward_speed_mps": sum(
                episode["mean_forward_speed_mps"] for episode in fixed["results"]) / len(fixed["results"]),
            "fixed_collisions": sum(bool(episode["collision"]) for episode in fixed["results"]),
            "tune_completion_rate": tune["summary"]["lap_completion_rate"],
            "tune_completed_laps": tune["summary"]["completed_laps"],
            "tune_completed_lap_speed_mps": tune["summary"]["completed_lap_speed_mps"],
            "tune_completed_lap_time_s": tune["summary"]["completed_lap_time_s"],
            "tune_mean_forward_speed_all_mps": tune["summary"]["mean_forward_speed_mps_all"],
            "tune_collision_rate": tune["summary"]["collision_rate"],
            "passes_fixed_reliability": fixed["lap_completion_rate"] >= baseline_fixed_completion,
        }
        if baseline_tune is not None:
            row["tune_speed_delta_mps"] = (
                tune["summary"]["completed_lap_speed_mps"] - baseline_tune["summary"]["completed_lap_speed_mps"]
                if tune["summary"]["completed_lap_speed_mps"] is not None
                and baseline_tune["summary"]["completed_lap_speed_mps"] is not None else None)
        rows.append(row)
    rows.sort(key=lambda row: (
        not row["passes_fixed_reliability"],
        -(row.get("tune_completed_lap_speed_mps") or -1.0),
    ))
    return {
        "baseline_fixed_completion_rate": baseline_fixed_completion,
        "selection_rule": (
            "Require fixed-benchmark lap completion >= baseline; then rank by uncapped "
            "completed-lap speed on the tuning suite. Confirm finalists on the held-out suite. "
            "Crashes are not allowed to win."
        ),
        "candidates": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True, help="parent checkpoint")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("experiments/runs/speed_improvement"))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--baseline-fixed", type=Path,
                        help="baseline fixed evaluation JSON (default: infer 10/12)")
    parser.add_argument("--baseline-tune", type=Path,
                        help="baseline tuning-suite report JSON for speed deltas")
    parser.add_argument("--summary", type=Path, help="write summary JSON here")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}; choose from {sorted(VARIANTS)}")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    baseline_fixed_completion = 10.0 / 12.0
    if args.baseline_fixed:
        baseline_fixed_completion = json.loads(
            args.baseline_fixed.read_text(encoding="utf-8"))["lap_completion_rate"]
    baseline_tune = (json.loads(args.baseline_tune.read_text(encoding="utf-8"))
                     if args.baseline_tune else None)

    reports = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(run_candidate, name, str(args.parent), args.steps, seed,
                        args.run_dir, root): (name, seed)
            for name in names for seed in seeds
        }
        for future in as_completed(futures):
            name, seed = futures[future]
            report = future.result()
            reports.append(report)
            print(f"finished {name} seed={seed} global_step={report['global_step']}", flush=True)

    summary = summarize(reports, baseline_fixed_completion, baseline_tune)
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if args.summary:
        args.summary.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.summary}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
