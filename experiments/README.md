# DQN experiments

Build the CPU training image with `podman build -t localhost/car-rl-benchmark:latest -f benchmarks/Containerfile benchmarks`.

Run `python3 experiments/orchestrate.py --run-dir experiments/runs/main` to launch ten generations of ten candidates. Resume an interrupted run with the same command; saved settings and completed candidate results are reused. Python 3.10+ is required for the host runner. The runner uses the logged-in Codex CLI with `gpt-6-luna` and high reasoning, and runs training in the container image. Each candidate receives 100,000 additional steps by default.

The run directory holds `state.json`, candidate source, agent and training logs, per-track evaluations, lineage and checkpoints. `best/` stores the best evaluated checkpoint, configuration, trainer and results. The score averages 12 fixed starts (three tracks, four starts each): 80% lap progress and 20% forward speed capped at 20 m/s. The selection probabilities favor higher-ranked candidates; the two sampled parents provide checkpoints and strategy reports to the next generation.

To continue the previous ten-candidate run for ten generations of five candidates, run:

```sh
python3 -u experiments/orchestrate.py --run-dir experiments/runs/continuation \
  --parent-run experiments/runs/main --population 5 --parent-count 1 \
  --generations 10 --seed 20261102
```

The first new generation samples two parents (20% of the preceding population of ten). Subsequent generations sample one parent (20% of five). Every child receives its assigned parent's checkpoint and the selected strategy reports. The previous run's best checkpoint is retained as the overall best until a new candidate scores higher. Resume an interrupted run with the same `--run-dir`; its stored settings are authoritative.

The runner trains up to four candidates concurrently and pauses at an incomplete population before selecting parents. `experiments/supervise.py` can restart an interrupted run and wait for reported agent-usage resets; run only one supervisor per run. `experiments/export_when_complete.py` audits every completed candidate and exports the best checkpoint, matching trainer, configuration, strategy, evaluation and per-candidate results after all generations finish. You can run the same audit directly with `python3 experiments/audit_run.py experiments/runs/continuation`.

To evaluate a saved checkpoint, run `podman run --rm --network=none --security-opt label=disable -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work localhost/car-rl-benchmark:latest python -m dqn.evaluate experiments/runs/main/best/latest.pt`.

To watch a checkpoint locally, install the project requirements in a Python 3.10+ environment and run `python -m dqn.drive experiments/runs/main/best/latest.pt --track circuit`. Use `--track oval` or `--track clover`, `--start 0.25` to select a quarter-lap offset, or `--headless` to print the outcome without a viewer.
