"""Export a completed continuation's checkpoints and evaluation results."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.audit_run import audit


def export(run_dir: Path, destination: Path) -> bool:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text())
    generations = state["generations_data"]
    if len(generations) != state["generations"]:
        return False
    for generation in range(state["generations"]):
        candidates = generations[str(generation)]["candidates"]
        if len(candidates) != state["population"] or any(c["status"] != "complete" for c in candidates):
            return False
    issues = audit(run_dir)
    if issues:
        raise RuntimeError("Completed-run audit failed:\n" + "\n".join(issues))
    source = run_dir / "best"
    strategy = Path(state["best"]["directory"]) / "strategy.md"
    if not strategy.is_file():
        raise FileNotFoundError(f"Winning strategy is missing: {strategy}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("latest.pt", "trainer.py", "config.json", "evaluation.json", "record.json"):
        temporary = destination / (name + ".tmp")
        shutil.copy2(source / name, temporary)
        temporary.replace(destination / name)
    temporary = destination / "strategy.md.tmp"
    shutil.copy2(strategy, temporary)
    temporary.replace(destination / "strategy.md")
    def candidate_summary(candidate):
        report = json.loads((Path(candidate["directory"]) / "evaluation.json").read_text())
        return {"index": candidate["index"], "score": candidate["score"],
                "parent_index": candidate.get("parent_index"),
                "mean_progress": report["mean_progress"],
                "mean_forward_speed_mps": sum(r["mean_forward_speed_mps"] for r in report["results"]) / len(report["results"]),
                "lap_completion_rate": report["lap_completion_rate"]}

    summary = {
        "best": state["best"],
        "score_definition": "0.8 * clipped lap progress + 0.2 * clipped mean forward speed / 20",
        "generations": [
            {"generation": g + 1,
             "selected_parent_indices": generations[str(g)].get("selected_parents", []),
             "candidates": [candidate_summary(c) for c in generations[str(g)]["candidates"]]}
            for g in range(state["generations"])
        ],
    }
    temporary = destination / "results.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(destination / "results.json")
    for generation in range(state["generations"]):
        winner = max(generations[str(generation)]["candidates"], key=lambda c: c["score"])
        candidate_dir = Path(winner["directory"])
        target = destination / "generations" / f"generation_{generation + 1:02d}"
        target.mkdir(parents=True, exist_ok=True)
        for name, source_file in {
            "latest.pt": Path(winner["checkpoint"]),
            "trainer.py": candidate_dir / "dqn/trainer.py",
            "config.json": candidate_dir / "config.json",
            "strategy.md": candidate_dir / "strategy.md",
            "evaluation.json": candidate_dir / "evaluation.json",
        }.items():
            temporary = target / (name + ".tmp")
            shutil.copy2(source_file, temporary)
            temporary.replace(target / name)
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps({"run_dir": str(run_dir), "generations": state["generations"],
                                     "population": state["population"], "audit": "passed"}, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    print(f"Exported best checkpoint to {destination}", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    while not export(args.run_dir, args.destination):
        time.sleep(30)
