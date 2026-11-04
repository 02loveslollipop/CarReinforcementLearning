# CarTrack candidate strategy

This candidate continues parent candidate 3. It keeps the 10-value dynamics
observation, 128×128 ReLU Q network, 5×7 force/steering action grid, and the
three-track configuration with randomized starts enabled.

## Balance training over evaluation starts

The prior trainer selected a track randomly and independently cycled through
the four evaluation start fractions. That could leave some track/start pairs
underrepresented in a 100,000-step run. With equal track weights, episodes now
cycle deterministically through every track at each configured start anchor.
The assignment is derived from the checkpointed episode count, so a child or
later continuation keeps the same coverage sequence. Unequal user-supplied
track weights still use weighted random track selection.

Start jitter is reduced from 0.02 to 0.005 of a lap. The environment still has
`random_start` enabled and receives its start through the existing reset option;
the small offset retains variation while concentrating experience near the
evaluator's fixed quarter-lap starts.

## Refine the inherited policy

The learning rate is reduced from 1e-5 to 5e-6 to limit drift in the selected
parent's policy. Exploration starts at 0.08 and decays to 0.01 over 60,000
steps. Seventy percent of exploratory actions are neighboring grid actions,
which focuses most policy refinement on local force and steering adjustments
while retaining some full-grid exploration.

Reward shaping, replay, Double DQN targets, checkpoint state, and action
construction remain compatible with the parent and fixed evaluator. The
checkpoint continues to include `online_state_dict`, `actions`, `obs_dim`, and
`config` in `output/latest.pt`.

Run a child with:

```sh
python -m dqn.trainer --config config.json --output output --steps 100000 --parent parent.pt
```

Omit `--parent parent.pt` for a fresh run.
