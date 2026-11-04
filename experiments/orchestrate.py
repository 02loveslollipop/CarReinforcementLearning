"""Persistent evolutionary runner for the CarTrack DQN trainer.

The runner creates isolated candidates, asks Codex to edit each candidate's
trainer/configuration, trains and evaluates candidates in a local container,
and writes enough state to resume after interruption.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "experiments" / "orchestrator_run"
IMAGE = "localhost/car-rl-benchmark:latest"
BASE_CONFIG: dict[str, Any] = {
    "environment": {"track": ["circuit", "oval", "clover"], "random_start": True},
    "training": {"seed": 0},
}
RUNNER_NAMES = ("podman", "docker")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _copy_code(source_root: Path, destination: Path) -> None:
    """Copy only the Python source needed by a candidate's fixed environment."""
    for package in ("dqn", "car_rl"):
        source = source_root / package
        target = destination / package
        target.mkdir(parents=True, exist_ok=True)
        for file in source.glob("*.py"):
            shutil.copy2(file, target / file.name)


def _restore_fixed_code(candidate: Path, immutable: Path) -> None:
    """Restore evaluator and environment files after an agent finishes."""
    for relative in ("dqn/__init__.py", "dqn/evaluate.py"):
        target = candidate / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(immutable / relative, target)
    for source in (immutable / "car_rl").glob("*.py"):
        shutil.copy2(source, candidate / "car_rl" / source.name)


def _snapshot_immutable(run_dir: Path) -> Path:
    immutable = run_dir / "immutable"
    if not (immutable / "dqn/evaluate.py").exists():
        if immutable.exists():
            shutil.rmtree(immutable)
        _copy_code(ROOT, immutable)
    return immutable


def _candidate_dir(run_dir: Path, generation: int, index: int) -> Path:
    return run_dir / "generations" / f"gen_{generation:03d}" / f"candidate_{index:03d}"


def _new_candidate(
    candidate: Path,
    immutable: Path,
    generation: int,
    index: int,
    parent: dict[str, Any] | None,
) -> None:
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)
    if parent:
        parent_dir = Path(parent["directory"])
        _copy_code(parent_dir, candidate)
        shutil.copy2(parent_dir / "config.json", candidate / "config.json")
        shutil.copy2(Path(parent["checkpoint"]), candidate / "parent.pt")
    else:
        _copy_code(immutable, candidate)
        config = json.loads(json.dumps(BASE_CONFIG))
        config["training"]["seed"] = generation * 10_000 + index
        _atomic_json(candidate / "config.json", config)
    _restore_fixed_code(candidate, immutable)
    (candidate / "output").mkdir()
    (candidate / "logs").mkdir()


def _select_parents(records: list[dict[str, Any]], seed: int, count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        (record for record in records if record.get("status") == "complete" and record.get("score") is not None),
        key=lambda record: (-float(record["score"]), int(record["index"])),
    )
    if len(ranked) < count:
        return []
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    remaining = ranked[:]
    while len(selected) < count:
        weights = [math.exp(-ranked.index(record) / 2.0) for record in remaining]
        choice = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        selected.append(remaining.pop(choice))
    return selected


def _select_two(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return _select_parents(records, seed, 2)


def _selection_probabilities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (record for record in records if record.get("status") == "complete" and record.get("score") is not None),
        key=lambda record: (-float(record["score"]), int(record["index"])),
    )
    normalizer = sum(math.exp(-rank / 2.0) for rank in range(len(ranked)))
    return [
        {"index": int(record["index"]), "rank": rank,
         "probability": math.exp(-rank / 2.0) / normalizer}
        for rank, record in enumerate(ranked)
    ]


def _strategy_text(candidate: Path) -> str:
    path = candidate / "strategy.md"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")[:12000]
    return "(No strategy.md was produced.)"


def _agent_prompt(
    generation: int,
    candidate: Path,
    parents: list[dict[str, Any]],
    assigned_parent: dict[str, Any] | None,
    steps: int,
    repair_error: str | None = None,
) -> str:
    allowed_context = ""
    if parents and assigned_parent is not None:
        reports = []
        for parent in parents:
            pdir = Path(parent["directory"])
            eval_data = _read_json(pdir / "evaluation.json", {}) or {}
            reports.append(
                f"### Candidate {parent['index']} (mean_score={parent['score']:.8f})\n"
                f"Evaluation: {json.dumps(eval_data, sort_keys=True)}\n"
                f"Strategy:\n{_strategy_text(pdir)}"
            )
        allowed_context = (
            "\nPrevious generation's selected winner reports:\n\n"
            + "\n\n".join(reports)
            + f"\n\nYour assigned parent's complete checkpoint is at `{candidate / 'parent.pt'}`. "
            "Its complete trainer source and config are the starting `dqn/trainer.py` and `config.json` here. "
            f"You are assigned parent candidate {assigned_parent['index']}.\n"
        )
    else:
        allowed_context = "\nThis is generation zero: start from the supplied baseline trainer and config, with no parent reports or checkpoint.\n"
    repair = f"\nA previous attempt failed. Repair the candidate using this error report:\n{repair_error}\n" if repair_error else ""
    return f"""You are designing one candidate DQN training strategy for CarTrack.

Edit only `dqn/trainer.py` and `config.json` in this workspace. Write `strategy.md`
describing the changes and the reasoning. Do not edit the environment, evaluator,
or any other source file. The evaluator and environment are immutable and will be
restored after your run.

The required command contract is:
`python -m dqn.trainer --config config.json --output output --steps {steps} --parent parent.pt`
for a child, or omit `--parent` for a fresh run. The trainer must accept the
existing CLI flags and save `output/latest.pt` as a checkpoint with
`online_state_dict`, `actions`, `obs_dim`, and `config` fields. The fixed evaluator
reconstructs a feed-forward ReLU MLP from Linear weights, so keep the policy a
ReLU MLP and preserve an evaluator-compatible checkpoint and action grid.
Training must support the configured three tracks and random starts.
Keep `environment.observe_dynamics` enabled so every candidate has the same
10-value observation vector. Keep `environment.track` set to all three tracks
and `environment.random_start` enabled.
{allowed_context}
{repair}
Improve the chance of high mean score across all fixed evaluation starts. Keep
changes reproducible and make sure a fresh run and a continuation both work.
"""


def _run_agent(
    candidate: Path,
    generation: int,
    parents: list[dict[str, Any]],
    assigned_parent: dict[str, Any] | None,
    attempt: int,
    steps: int,
    repair_error: str | None = None,
) -> tuple[bool, str]:
    log_path = candidate / "logs" / f"agent_attempt_{attempt}.log"
    prompt = _agent_prompt(generation, candidate, parents, assigned_parent, steps, repair_error)
    command = [
        "codex", "exec", "-m", "gpt-6-luna", "-c", 'model_reasoning_effort="high"',
        "--skip-git-repo-check", "--ephemeral", "--sandbox", "workspace-write",
        "-C", str(candidate), prompt,
    ]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                check=False, timeout=20 * 60,
            )
        return completed.returncode == 0, f"codex exit status {completed.returncode}"
    except subprocess.TimeoutExpired:
        return False, "codex agent timed out after 20 minutes"
    except FileNotFoundError as error:
        return False, str(error)


def _container_runtime() -> str:
    for name in RUNNER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("Neither podman nor docker is installed")


def _container_command(runtime: str, candidate: Path, command: list[str]) -> list[str]:
    return [
        runtime, "run", "--rm", "--network=none", "--security-opt", "label=disable",
        "-v", f"{candidate.resolve()}:/work", "-w", "/work", "-e", "PYTHONPATH=/work",
        IMAGE, *command,
    ]


def _run_logged(command: list[str], log_path: Path, cwd: Path = ROOT, timeout: int = 4 * 60 * 60) -> tuple[bool, str]:
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                check=False, timeout=timeout,
            )
        return result.returncode == 0, f"exit status {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} seconds"
    except FileNotFoundError as error:
        return False, str(error)


def _validate_candidate(candidate: Path) -> tuple[bool, str]:
    config = _read_json(candidate / "config.json")
    if not isinstance(config, dict):
        return False, "config.json is missing or is not a JSON object"
    environment = config.setdefault("environment", {})
    if not isinstance(environment, dict):
        return False, "config.environment must be an object"
    if environment.get("observe_dynamics", True) is not True:
        return False, "environment.observe_dynamics must stay enabled (10 observations)"
    if environment.get("track", ["circuit", "oval", "clover"]) != ["circuit", "oval", "clover"]:
        return False, "environment.track must include circuit, oval, and clover"
    if environment.get("random_start", True) is not True:
        return False, "environment.random_start must stay enabled"
    trainer = (candidate / "dqn/trainer.py").read_text(encoding="utf-8", errors="replace")
    if "ReLU" not in trainer and "relu" not in trainer:
        return False, "trainer source does not declare a ReLU MLP"
    if not (candidate / "strategy.md").is_file():
        return False, "agent did not create strategy.md"
    return True, "candidate source/config accepted"


def _train_and_evaluate(candidate: Path, runtime: str, steps: int, attempt: int, resume: bool = False) -> tuple[bool, str, float | None]:
    valid, reason = _validate_candidate(candidate)
    if not valid:
        return False, reason, None
    parent_path = "parent.pt" if (candidate / "parent.pt").exists() else None
    remaining = steps
    partial = candidate / "output/latest.pt"
    if resume and partial.exists():
        probe = ("import json,torch; from pathlib import Path; "
                 "load=lambda p: torch.load(p,map_location='cpu',weights_only=False); "
                 "print(json.dumps([int(load('parent.pt')['global_step']) if Path('parent.pt').exists() else 0, "
                 "int(load('output/latest.pt')['global_step'])]))")
        result = subprocess.run(_container_command(runtime, candidate, ["python", "-c", probe]),
                                cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, "cannot inspect interrupted checkpoint: " + result.stderr[-2000:], None
        try:
            base, current = json.loads(result.stdout.strip().splitlines()[-1])
            if not base <= current <= base + steps:
                raise ValueError("checkpoint step outside expected training interval")
        except (ValueError, KeyError, IndexError, TypeError) as error:
            return False, f"invalid interrupted checkpoint: {error}", None
        remaining = base + steps - current
        parent_path = "output/latest.pt"
    trainer_command = _container_command(
        runtime, candidate,
        ["python", "-m", "dqn.trainer", "--config", "config.json", "--output", "output",
         "--steps", str(remaining), *(["--parent", parent_path] if parent_path else [])],
    )
    if resume and parent_path == "output/latest.pt" and remaining == 0:
        trained, training_message = True, "recovered completed checkpoint"
    else:
        trained, training_message = _run_logged(
            trainer_command, candidate / "logs" / f"training_attempt_{attempt}.log"
        )
    _restore_fixed_code(candidate, candidate.parent.parent.parent / "immutable")
    if not trained:
        return False, f"training failed: {training_message}", None
    checkpoint = candidate / "output/latest.pt"
    if not checkpoint.exists():
        return False, "training did not create output/latest.pt", None
    eval_command = _container_command(
        runtime, candidate,
        ["python", "-m", "dqn.evaluate", "output/latest.pt", "--output", "evaluation.json"],
    )
    evaluated, eval_message = _run_logged(eval_command, candidate / "logs" / f"evaluation_attempt_{attempt}.log")
    if not evaluated:
        return False, f"fixed evaluator failed: {eval_message}", None
    report = _read_json(candidate / "evaluation.json")
    try:
        score = float(report["mean_score"])
        if not math.isfinite(score):
            raise ValueError("mean_score is not finite")
    except (TypeError, KeyError, ValueError) as error:
        return False, f"invalid evaluation result: {error}", None
    return True, "training and evaluation complete", score


def _save_best(run_dir: Path, candidate: Path, record: dict[str, Any]) -> None:
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate / "output/latest.pt", best_dir / "latest.pt")
    shutil.copy2(candidate / "config.json", best_dir / "config.json")
    shutil.copy2(candidate / "dqn/trainer.py", best_dir / "trainer.py")
    shutil.copy2(candidate / "evaluation.json", best_dir / "evaluation.json")
    _atomic_json(best_dir / "record.json", record)


def _status(record: dict[str, Any], key: str, value: str, state: dict[str, Any], state_path: Path) -> None:
    record[key] = value
    record["updated_at"] = time.time()
    _atomic_json(state_path, state)


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = _read_json(state_path)
    if state is None:
        state = {
            "version": 1,
            "seed": args.seed,
            "generations": args.generations,
            "population": args.population,
            "steps": args.steps,
            "parent_count": args.parent_count,
            "parent_run": str(args.parent_run.expanduser().resolve()) if args.parent_run else None,
            "generations_data": {},
            "best": None,
        }
        if args.parent_run:
            source_dir = args.parent_run.expanduser().resolve()
            source_state = _read_json(source_dir / "state.json")
            if not source_state or not source_state.get("best") or not (source_dir / "best/latest.pt").is_file():
                raise ValueError("Parent run must have a saved best checkpoint")
            shutil.copytree(source_dir / "best", run_dir / "best", dirs_exist_ok=True)
            state["best"] = dict(source_state["best"], source_run=str(source_dir))
        _atomic_json(state_path, state)
    else:
        args.seed = int(state["seed"])
        args.generations = int(state["generations"])
        args.population = int(state["population"])
        args.steps = int(state["steps"])
        args.parent_count = int(state.get("parent_count", 2))
        args.parent_run = Path(state["parent_run"]) if state.get("parent_run") else None

    immutable = _snapshot_immutable(run_dir)
    runtime = _container_runtime()
    max_generation = min(args.generations, 10)
    for generation in range(max_generation):
        generation_key = str(generation)
        gen_data = state["generations_data"].setdefault(generation_key, {"candidates": []})
        records: list[dict[str, Any]] = gen_data["candidates"]
        if not records:
            if generation == 0 and args.parent_run:
                source_state = _read_json(args.parent_run / "state.json")
                if not source_state or not source_state.get("generations_data"):
                    raise ValueError("Parent run has no completed generation state")
                source_generation = max(map(int, source_state["generations_data"]))
                previous = source_state["generations_data"][str(source_generation)]["candidates"]
                winners = _select_parents(previous, args.seed + source_generation + 1, max(1, math.ceil(len(previous) * 0.2)))
                if len(winners) != max(1, math.ceil(len(previous) * 0.2)):
                    raise ValueError("Parent run has too few successful candidates")
            elif generation == 0:
                winners: list[dict[str, Any]] = []
            else:
                previous = state["generations_data"].get(str(generation - 1), {}).get("candidates", [])
                winners = _select_parents(previous, args.seed + generation, args.parent_count)
                if len(winners) < args.parent_count:
                    print(f"Stopping before generation {generation}: too few successful parents.", flush=True)
                    break
            for index in range(args.population):
                assigned = winners[min(index * len(winners) // args.population, len(winners) - 1)] if winners else None
                candidate = _candidate_dir(run_dir, generation, index)
                _new_candidate(candidate, immutable, generation, index, assigned)
                records.append({
                    "index": index,
                    "directory": str(candidate),
                    "parent_index": assigned["index"] if assigned else None,
                    "agent_status": "pending",
                    "training_status": "pending",
                    "status": "pending",
                    "attempt": 0,
                })
            gen_data["selected_parents"] = [int(item["index"]) for item in winners]
            if winners:
                gen_data["selection_probabilities"] = _selection_probabilities(previous)
                if generation == 0:
                    gen_data["source_parent_run"] = str(args.parent_run)
                    gen_data["source_generation"] = source_generation
            _atomic_json(state_path, state)

        # Interrupted tasks marked as running are safe to restart. Completed stages persist.
        for record in records:
            if record.get("agent_status") == "running":
                record["agent_status"] = "pending"
            if record.get("training_status") == "running":
                record["training_status"] = "pending"
                record["resume_training"] = True
            if record.get("status") == "running":
                record["status"] = "pending"
            if record.get("status") == "repairing":
                record["status"] = "repair_pending"
                record["repair_resume"] = True

        parent_records = []
        if generation > 0 or (generation == 0 and args.parent_run):
            if generation == 0:
                source_state = _read_json(args.parent_run / "state.json")
                prior = source_state["generations_data"][str(gen_data["source_generation"])]["candidates"]
            else:
                prior = state["generations_data"][str(generation - 1)]["candidates"]
            selected_indices = gen_data.get("selected_parents", [])
            parent_records = [item for idx in selected_indices for item in prior if int(item["index"]) == int(idx)]

        for record in records:
            if record.get("agent_status") == "complete" or record.get("status") == "complete":
                continue
            record["attempt"] = int(record.get("attempt", 0)) + 1
            _status(record, "agent_status", "running", state, state_path)
        agent_tasks = [
            record for record in records
            if record.get("agent_status") == "running" and record.get("status") != "complete"
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for record in agent_tasks:
                cdir = Path(record["directory"])
                assigned = None
                if parent_records:
                    assigned = next((p for p in parent_records if int(p["index"]) == int(record["parent_index"])), None)
                futures[pool.submit(
                    _run_agent, cdir, generation, parent_records, assigned,
                    int(record["attempt"]), args.steps,
                )] = record
            for future in concurrent.futures.as_completed(futures):
                record = futures[future]
                cdir = Path(record["directory"])
                ok, message = future.result()
                _restore_fixed_code(cdir, immutable)
                record["agent_status"] = "complete" if ok else "failed"
                record["agent_message"] = message
                record["status"] = "pending" if ok else "failed"
                _atomic_json(state_path, state)

        train_tasks = [
            r for r in records
            if r.get("agent_status") == "complete" and r.get("status") not in ("complete", "repair_pending", "repairing")
            and (int(r.get("training_attempts", 0)) < 2 or r.get("resume_training"))
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, args.population)) as pool:
            futures = {}
            for record in train_tasks:
                resumed = record.pop("resume_training", False)
                if not resumed:
                    record["training_attempts"] = int(record.get("training_attempts", 0)) + 1
                record["training_status"] = "running"
                record["status"] = "running"
                _atomic_json(state_path, state)
                futures[pool.submit(
                    _train_and_evaluate, Path(record["directory"]), runtime, args.steps, int(record["attempt"]), resumed
                )] = record
            for future in concurrent.futures.as_completed(futures):
                record = futures[future]
                ok, message, score = future.result()
                record["training_status"] = "complete" if ok else "failed"
                record["status"] = "complete" if ok else "repair_pending"
                record.pop("resume_training", None)
                record["training_message"] = message
                if ok:
                    record["score"] = score
                    record["checkpoint"] = str(Path(record["directory"]) / "output/latest.pt")
                    best = state.get("best")
                    if best is None or score > float(best["score"]):
                        state["best"] = {
                            "generation": generation,
                            "index": int(record["index"]),
                            "score": score,
                            "directory": record["directory"],
                        }
                        _save_best(run_dir, Path(record["directory"]), state["best"])
                _atomic_json(state_path, state)

        # Give a failed candidate exactly one repair pass, with its error log.
        repair_tasks = [r for r in records if r.get("status") == "repair_pending"]
        if repair_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {}
                for record in repair_tasks:
                    if int(record.get("repair_attempts", 0)) == 0:
                        record["repair_attempts"] = 1
                    record["status"] = "repairing"
                    if not record.pop("repair_resume", False):
                        record["attempt"] = int(record.get("attempt", 0)) + 1
                    cdir = Path(record["directory"])
                    error_path = cdir / "logs" / f"training_attempt_{record['training_attempts']}.log"
                    eval_error_path = cdir / "logs" / f"evaluation_attempt_{record['training_attempts']}.log"
                    details = "\n".join(
                        path.read_text(encoding="utf-8", errors="replace")[-10000:]
                        for path in (error_path, eval_error_path) if path.exists()
                    ) or record.get("training_message", "No log was produced")
                    assigned = None
                    if parent_records:
                        assigned = next((p for p in parent_records if int(p["index"]) == int(record["parent_index"])), None)
                    futures[pool.submit(
                        _run_agent, cdir, generation, parent_records, assigned,
                        int(record["attempt"]), args.steps,
                        f"{record.get('training_message', '')}\n{details}",
                    )] = record
                    _atomic_json(state_path, state)
                for future in concurrent.futures.as_completed(futures):
                    record = futures[future]
                    cdir = Path(record["directory"])
                    repaired, message = future.result()
                    _restore_fixed_code(cdir, immutable)
                    record["repair_message"] = message
                    if repaired:
                        record["agent_status"] = "complete"
                        record["status"] = "pending"
                    else:
                        record["agent_status"] = "failed"
                        record["status"] = "failed"
                    _atomic_json(state_path, state)

            retry_tasks = [r for r in repair_tasks if r.get("status") == "pending"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, args.population)) as pool:
                futures = {}
                for record in retry_tasks:
                    record["training_attempts"] = int(record.get("training_attempts", 0)) + 1
                    record["training_status"] = "running"
                    record["status"] = "running"
                    _atomic_json(state_path, state)
                    futures[pool.submit(
                        _train_and_evaluate, Path(record["directory"]), runtime, args.steps, int(record["attempt"])
                    )] = record
                for future in concurrent.futures.as_completed(futures):
                    record = futures[future]
                    ok, message, score = future.result()
                    record["training_status"] = "complete" if ok else "failed"
                    record["status"] = "complete" if ok else "failed"
                    record["training_message"] = message
                    if ok:
                        record["score"] = score
                        record["checkpoint"] = str(Path(record["directory"]) / "output/latest.pt")
                        best = state.get("best")
                        if best is None or score > float(best["score"]):
                            state["best"] = {
                                "generation": generation,
                                "index": int(record["index"]),
                                "score": score,
                                "directory": record["directory"],
                            }
                            _save_best(run_dir, Path(record["directory"]), state["best"])
                    _atomic_json(state_path, state)

        successful = [record for record in records if record.get("status") == "complete"]
        print(f"generation {generation}: {len(successful)}/{len(records)} successful", flush=True)
        if len(successful) != args.population:
            print("Stopping: incomplete population; resume this run to retry before selecting parents.", flush=True)
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000, help="additional training transitions per candidate")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=10, help="candidate count in each generation")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=20261101, help="deterministic parent selection seed")
    parser.add_argument("--parent-run", type=Path, help="continue from the final generation of an earlier run")
    parser.add_argument("--parent-count", type=int, default=2, help="number of parents sampled for each new generation")
    args = parser.parse_args(argv)
    if args.steps < 0 or args.generations < 1 or args.population < 1 or not 1 <= args.parent_count <= args.population:
        parser.error("steps must be non-negative and generations/population must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
