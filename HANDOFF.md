# Handoff: faster DQN driving and policy visualization

## Goal

Iteratively improve the car's **simulated driving speed**, particularly lap time and mean forward speed, while retaining reliable lap completion. Add a visualization of the selected action and the DQN's full layer activations. Compare models on identical circuits and starting conditions; do not confuse Matplotlib frame rate or training throughput with driving speed.

## Starting point

- The current best model is `visualization/checkpoints/evolution/latest.pt`, also packaged with metadata in `weights.zip`. Its matching configuration, actual trainer, strategy, and evaluation are in the same `evolution/` directory. Preserve this model as the baseline.
- The original fixed benchmark covers 12 episodes: circuit, oval, and clover, each at four fixed start fractions. Its recorded baseline is score **0.867842**, mean progress **0.911405**, lap completion **10/12**, collision rate **2/12**, and mean forward speed across episodes **13.872 m/s**. Circuit has two collisions; oval and clover finish all starts. See `visualization/checkpoints/evolution/evaluation.json` for per-episode details.
- Benchmark score is `0.8 * clip(progress, 0, 1) + 0.2 * clip(mean_forward_speed / 20, 0, 1)`. This mostly favors distance and stops rewarding speed above 20 m/s. Keep reporting it for comparison, but do not use it alone as the speed objective.
- The winning network takes 10 observations, has two 128-unit ReLU layers, and chooses among 35 force/steering combinations. Its Double DQN configuration includes gamma 0.995, learning rate 0.0003, target tau 0.01, and collision penalty 50. Its progress-based shaped reward replaces the environment's own speed reward; see the saved trainer and strategy rather than assuming `training/dqn/trainer.py` is identical to the winner.
- `utils/car_rl/environment.py` and `utils/car_rl/dynamics.py` define fixed physics and observations; `training/dqn/evaluate.py` is the fixed evaluator. Do not improve scores by changing physics, track geometry, step duration, termination rules, or evaluation conditions.
- The original ten-generation run and five-candidate continuation are stored locally under ignored `experiments/runs/` when available; they are not included in the published repository. The checked-in `evolution/` directory includes the winner and exported per-generation winners. Read `orchestration/README.md` before resuming or creating population runs. Archived candidate workspaces use the original `dqn` and `car_rl` names; the maintained project uses `training.dqn` and `utils.car_rl`. The saved winning trainer is an archived candidate source, so it is not directly runnable in the reorganized root without adapting its imports or using a compatible candidate workspace.

## Evaluation protocol

1. Re-run the unchanged fixed benchmark for the baseline and every contender. Use paired comparisons at every track/start and record completed-lap time, forward speed over the whole episode, progress, collisions, truncations, and completion rate. Compute completed-lap speed/time separately and always show the number of completed laps; excluding failures can make a reckless policy appear fast.
2. Build an additional **fixed, reproducible** set of randomly generated circuits and start positions, using the same saved seeds and episode limits for every candidate. Include a separate held-out set that is not used for training, hyperparameter choices, or parent selection. Save track-generation parameters/seeds, metrics, model identifier, config, training steps, and timing in machine-readable reports. `visualization/circuits.py` already generates seeded tracks; explicitly define how starts and max steps are selected.
3. Select candidates primarily by a stated reliability threshold and then by lap time or uncapped forward speed, rather than letting a fast crash win. Start by requiring no regression against the baseline's fixed-track completion rate; report the tradeoff and per-track results if no candidate meets it. Since 12 starts are a small sample, use multiple seeds and paired aggregate results on the random-track suite before claiming a general improvement.
4. Keep the fixed evaluator as a historical benchmark. Add a separate speed-oriented report or selection metric, document its exact formula and constraints, and use the same evaluation code for every model. Track peak speed only as supporting context; a high peak does not necessarily produce a faster lap.

## Iterative experiments

Establish a reproducible baseline first. In each iteration, run a small controlled set of candidates from the same starting checkpoint with the same training budget and evaluation suite. Change one main factor at a time where possible, repeat promising settings with multiple training seeds, and retain the best verified checkpoint separately from the most recent one. Log parent checkpoint, code/config, seed, wall time, environment steps, and results. Continue for multiple iterations as resources permit; report failed and regressed runs as well as winners.

Suggested tests, in order:

1. **Faster laps through reward shaping:** compare the winner's progress reward with small per-step time costs or explicit lap-time incentives, while testing a few collision-penalty values. Check reward scaling and termination effects so early crashes cannot earn an advantage. Compare against an unchanged continuation of the winner with the same steps and seeds; continuing training alone can degrade the policy.
2. **Continuation stability:** test modest learning-rate and exploration schedules, target update rates, and replay choices. The checkpoint includes optimizer, replay, random state, and an active episode; determine explicitly whether a run resumes these or resets them. The epsilon schedule uses global steps, so a resumed model may already be at its exploration floor. Keep comparisons fair and avoid changing several settings at once.
3. **Driving behavior:** inspect force, steering, braking, speed, corner position, and collision locations on identical episodes. Test whether more speed can be gained on straights or by braking earlier for corners. Consider geometry-diverse random training circuits and randomized starts, with fixed held-out circuits for generalization.
4. **Action and observation resolution:** examine whether the existing 5-by-7 grid limits throttle or steering control and whether sensor range or dynamics observations limit high-speed corner anticipation. Changes to action count, observation dimensions, or network shape make parent checkpoints and the evaluator incompatible unless an explicit migration or fresh-training path is implemented; treat them as separate experiments and document the compatibility boundary.

## Visualization deliverable

Extend `visualization/app.py` without breaking its continuous random-circuit loop, seed handling, headless mode, or close behavior:

- Show the action index and its force and steering values, and preferably the Q-values for all 35 actions with the chosen one clearly marked. Label them **Q-values**, not probabilities or confidence.
- Show every neuron activation in each layer, including the 10 inputs, both 128-unit hidden layers, and 35 outputs. A compact heatmap or per-layer grids are more readable than tens of thousands of edges. Clearly label layer names, scales, signs, and whether hidden values are before or after ReLU; do not imply that visualized values are network weights. Provide exact values via an inspectable view if practical.
- Capture activations and Q-values for the same observation used to select the action, using one inference pass under `torch.inference_mode()`. Make the display's timing clear because the car state advances after selection. Avoid changing model outputs, allocating new artists or registering hooks on every frame, leaking hooks across circuits, or imposing unnecessary work in headless mode.
- Cover action/activation correctness with a deterministic small model and check that rendering does not change the selected action or simulation state. Test circuit transitions and closing the window; inspect a real rendered frame and, when available, an interactive backend.

## Useful commands

Run from the repository root after installing `requirements.txt`:

```sh
python -m training.dqn.evaluate visualization/checkpoints/evolution/latest.pt
python -m visualization --seed 42
python -m visualization --headless --episodes 3 --seed 42
pytest -q
```

The CPU container can be built as described in `orchestration/README.md`; use it for reproducible training and non-graphical tests. Record actual commands, hardware, budgets, and results. Do not replace `weights.zip` or the default checkpoint until a candidate has passed the stated evaluation; if promoted, update both together and include its matching config and evaluation in the archive.
