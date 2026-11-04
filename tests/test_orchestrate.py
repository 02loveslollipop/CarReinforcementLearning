from pathlib import Path

from orchestration.orchestrate import _agent_prompt, _select_parents, _train_and_evaluate
from orchestration.export_when_complete import export
from orchestration.supervise import complete, workers_alive
from types import SimpleNamespace
import json


def test_candidate_snapshot_preserves_standalone_imports(tmp_path):
    import os
    import subprocess
    import sys
    from orchestration.orchestrate import ROOT, _copy_code

    _copy_code(ROOT, tmp_path)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    subprocess.run(
        [sys.executable, "-c", "import dqn.trainer, dqn.evaluate, car_rl; import gymnasium as gym; gym.make('CarTrack-v0')"],
        cwd=tmp_path, env=env, check=True,
    )


def test_single_parent_selection_favors_rank_and_is_deterministic():
    records = [{"index": i, "status": "complete", "score": 5 - i} for i in range(5)]
    assert _select_parents(records, 42, 1) == _select_parents(records, 42, 1)
    choices = [_select_parents(records, seed, 1)[0]["index"] for seed in range(1000)]
    assert choices.count(0) > choices.count(1) > choices.count(4)


def test_first_continuation_generation_receives_parent_reports(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "strategy.md").write_text("parent strategy")
    (parent / "evaluation.json").write_text('{"mean_score": 0.5}')
    record = {"index": 3, "score": 0.5, "directory": str(parent)}
    prompt = _agent_prompt(0, Path(tmp_path / "child"), [record], record, 100)
    assert "parent strategy" in prompt
    assert "assigned parent candidate 3" in prompt
    assert "generation zero: start" not in prompt


def test_export_waits_for_all_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr('orchestration.export_when_complete.audit', lambda path: [])
    run = tmp_path / "run"
    run.mkdir()
    destination = tmp_path / "demo"
    state = {"generations": 1, "population": 1,
             "generations_data": {"0": {"candidates": [{"status": "running", "index": 0, "score": 0.5}]}}}
    (run / "state.json").write_text(json.dumps(state))
    assert not export(run, destination)
    assert not destination.exists()
    best = run / "best"
    best.mkdir()
    for name in ("latest.pt", "trainer.py", "config.json", "evaluation.json", "record.json"):
        (best / name).write_text(name)
    strategy = run / "winning_candidate"
    strategy.mkdir()
    (strategy / "strategy.md").write_text("winning strategy")
    (strategy / "evaluation.json").write_text(json.dumps({"mean_progress": 0.5, "lap_completion_rate": 0.0,
                                                           "results": [{"mean_forward_speed_mps": 4.0}]}))
    (strategy / "dqn").mkdir()
    (strategy / "dqn" / "trainer.py").write_text('trainer')
    (strategy / "config.json").write_text('{}')
    (strategy / "output").mkdir()
    (strategy / "output" / "latest.pt").write_text('weights')
    state["generations_data"]["0"]["candidates"][0]["directory"] = str(strategy)
    state["generations_data"]["0"]["candidates"][0]["checkpoint"] = str(strategy / "output" / "latest.pt")
    state["best"] = {"directory": str(strategy)}
    state["generations_data"]["0"]["candidates"][0]["status"] = "complete"
    (run / "state.json").write_text(json.dumps(state))
    assert export(run, destination)
    assert (destination / "latest.pt").read_text() == "latest.pt"
    assert (destination / "strategy.md").read_text() == "winning strategy"
    assert json.loads((destination / "results.json").read_text())["generations"][0]["candidates"][0]["score"] == 0.5
    assert (destination / "generations/generation_01/latest.pt").read_text() == "weights"
    assert json.loads((destination / "manifest.json").read_text())["audit"] == "passed"


def test_supervisor_requires_every_candidate_in_every_generation():
    state = {"generations": 2, "population": 1,
             "generations_data": {"0": {"candidates": [{"status": "complete"}]}}}
    assert not complete(state)
    state["generations_data"]["1"] = {"candidates": [{"status": "failed"}]}
    assert not complete(state)
    state["generations_data"]["1"]["candidates"][0]["status"] = "complete"
    assert complete(state)


def test_supervisor_detects_surviving_training_container(tmp_path, monkeypatch):
    monkeypatch.setattr('orchestration.supervise.shutil.which', lambda name: '/bin/podman')
    def run(command, **kwargs):
        if command[0] == 'ps':
            return SimpleNamespace(stdout='')
        if command[1] == 'ps':
            return SimpleNamespace(stdout='abc\n')
        return SimpleNamespace(returncode=0, stdout=json.dumps([{'Mounts': [{'Source': str(tmp_path / 'candidate')}]}]))
    monkeypatch.setattr('orchestration.supervise.subprocess.run', run)
    assert workers_alive(tmp_path)


def test_recovered_finished_checkpoint_is_evaluated_without_retraining(tmp_path, monkeypatch):
    (tmp_path / 'output').mkdir()
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'output/latest.pt').write_text('checkpoint')
    monkeypatch.setattr('orchestration.orchestrate._validate_candidate', lambda candidate: (True, 'ok'))
    monkeypatch.setattr('orchestration.orchestrate._restore_fixed_code', lambda *args: None)
    monkeypatch.setattr('orchestration.orchestrate.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='[100, 200]\n'))
    def logged(command, log_path):
        assert 'dqn.evaluate' in command
        (tmp_path / 'evaluation.json').write_text('{"mean_score": 0.5}')
        return True, 'ok'
    monkeypatch.setattr('orchestration.orchestrate._run_logged', logged)
    assert _train_and_evaluate(tmp_path, 'podman', 100, 1, resume=True) == (True, 'training and evaluation complete', .5)
