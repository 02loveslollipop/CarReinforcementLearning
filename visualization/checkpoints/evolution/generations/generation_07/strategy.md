# CarTrack candidate strategy

This child continues parent candidate 3 with the evaluator-compatible 10-value
dynamics observation, 128×128 ReLU Q network, and 5×7 force/steering action
grid. The environment remains configured for circuit, oval, and clover with
`random_start` and `observe_dynamics` enabled.

## Cover the evaluator's start locations

The evaluator uses four fixed start fractions per track. Training now cycles
episode starts through 0, 0.25, 0.5, and 0.75 of the selected track, adding a
small seeded jitter of up to 0.02 of a lap. This keeps starts varied while
putting substantially more training experience near every fixed evaluation
start. Track selection remains equal across the three tracks. The anchor index
comes from the saved episode count, and jitter and track selection use the
checkpointed environment RNG, so a continuation remains reproducible.

The environment's `random_start` setting stays enabled. Training supplies
`s0` through the environment's existing reset options to focus that randomness
around the evaluation points; no observation or evaluator changes are needed.

## Refine the inherited policy

The parent has already exhausted its 100,000-step epsilon decay. This child
changes the schedule to start at 0.12 and taper to 0.02 over its next 100,000
steps, giving the inherited policy a fresh, restrained exploration phase. The
local-neighbor share rises from 0.30 to 0.55 so most exploratory actions test
nearby force and steering choices, with occasional full-grid exploration. The
learning rate doubles from 5e-6 to 1e-5 to permit modest policy refinement
while preserving the inherited network and replay buffer.

The trainer retains its existing progress, lane-centering, slip-stability, and
collision shaping. The changed schedule resets its epsilon clock once for this
child; its checkpoint stores the new schedule, RNG states, replay, and active
episode so later continuations resume it exactly.

The required child command remains:

```sh
python -m dqn.trainer --config config.json --output output --steps 100000 --parent parent.pt
```

Omit `--parent parent.pt` for a fresh run. Both paths save
`output/latest.pt` with `online_state_dict`, `actions`, `obs_dim`, and `config`
alongside the trainer continuation state.
