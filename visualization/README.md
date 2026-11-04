# Random circuit visualizer

From the project root, install the dependencies and run the visualizer:

```sh
python -m pip install -r requirements.txt
python -m visualization
```

The window shows the whole circuit alongside a close view of the car and its sensors. Each lap completion, collision, or timeout starts a newly generated circuit. Close the window to stop. The default checkpoint is the best saved DQN in `visualization/checkpoints/evolution/latest.pt`.

You can pass a different checkpoint, reproduce a circuit sequence, or change the update rate:

```sh
python -m visualization path/to/checkpoint.pt --seed 42 --fps 20
```

A graphical desktop and an interactive Matplotlib backend are required for the window. For a quick non-graphical check, run `python -m visualization --headless --episodes 3 --max-steps 100`. This prints an outcome for each generated circuit. The model was trained on the project's original tracks; its performance on new random circuits will vary.

To check the simulation in the existing container image from the project root:

```sh
podman run --rm --network=none --security-opt label=disable \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  localhost/car-rl-benchmark:latest \
  python -m visualization --headless --episodes 3 --max-steps 100
```
