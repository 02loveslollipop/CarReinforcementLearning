# Random circuit visualizer

From the project root, install the dependencies and run the visualizer:

```sh
python -m pip install -r requirements.txt
python -m visualization
```

The window shows the whole circuit alongside a close view of the car and its sensors. Each lap completion, collision, or timeout starts a newly generated circuit. Close the window to stop. The default checkpoint is the best saved DQN in `visualization/checkpoints/evolution/latest.pt`.

The right and bottom panels show the DQN that chose the current action. The bar chart gives the Q-value for every discrete force/steering action; the highlighted bar is the chosen action and its force, steering, and Q-value are printed. Below it, heatmaps show every activation in order: the 10 observations, Hidden 1 after ReLU, and Hidden 2 after ReLU. The same 35 Q-values are the output-layer activations. Colors are diverging (blue negative, red positive) with each panel scaled to its own maximum absolute value; panel titles state the size and observed range. Hidden values are post-ReLU, not weights. The network values come from the single forward pass that selected the action, while the car view already shows the state after that action. Press `P` to print the exact action, Q-values, and every layer activation as JSON.

You can pass a different checkpoint, reproduce a circuit sequence, or change the update rate:

```sh
python -m visualization path/to/checkpoint.pt --seed 42 --fps 20
```

A graphical desktop and an interactive Matplotlib backend are required for the window. For a quick non-graphical check, run `python -m visualization --headless --episodes 3 --max-steps 100`. This prints an outcome for each generated circuit. The model was trained on the project's original tracks; its performance on new random circuits will vary.

For a speed-focused comparison on seeded random circuits, use:

```sh
python -m training.dqn.speed_eval visualization/checkpoints/evolution/latest.pt --suite tune --output reports/baseline_tune.json
python -m training.dqn.speed_eval visualization/checkpoints/evolution/latest.pt --suite holdout --output reports/baseline_holdout.json
```

The tuning suite (seeds 1000-1011) and a disjoint held-out suite (seeds 2000-2015) each use the four fixed start fractions `0, 0.25, 0.5, 0.75` and `max_steps=3000`. The reports record lap completion, completed-lap time, uncapped completed-lap speed (`track_length / lap_time`), mean forward speed over all episodes, peak speed, collisions, and truncations. Use the held-out suite only to confirm candidates; do not use it for training or parent selection.

To check the simulation in the existing container image from the project root:

```sh
podman run --rm --network=none --security-opt label=disable \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  localhost/car-rl-benchmark:latest \
  python -m visualization --headless --episodes 3 --max-steps 100
```
