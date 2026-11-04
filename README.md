# Car reinforcement learning

A car-driving environment, DQN training and evaluation tools, an evolutionary experiment runner, and a live random-circuit visualizer.

## Test it

From the project root, install the dependencies, extract the trained weights, and launch the animation:

```sh
python -m pip install -r requirements.txt
python -m zipfile -e weights.zip .weights
python -m visualization .weights/latest.pt
```

The car drives on a new random circuit after each lap, collision, or timeout. Close the window to stop. A graphical desktop and an interactive Matplotlib backend are required.

## Project layout

- `training/dqn/` — DQN training, evaluation, and single-track driving.
- `orchestration/` — population experiments, supervision, auditing, and export. See [its guide](orchestration/README.md).
- `utils/car_rl/` — shared car dynamics, environment, circuits, and rendering.
- `visualization/` — continuous driving on randomly generated circuits. See [its guide](visualization/README.md).
- `benchmarks/`, `examples/`, and `tests/` — performance checks, a heuristic driver, and tests.
- `weights.zip` — the best checkpoint (`latest.pt`) and its configuration, strategy, and evaluation. The same checkpoint is available under `visualization/checkpoints/evolution/` together with the generation winners.

To evaluate the best checkpoint on the fixed benchmark:

```sh
python -m training.dqn.evaluate visualization/checkpoints/evolution/latest.pt
```

Run tests with `pytest`. The experiment runner stores generated runs under `experiments/runs/`, which is excluded from Git. Candidate workspaces retain their original `dqn` and `car_rl` package names so existing runs and saved trainers remain compatible.
