# CarTrack continuation strategy

This candidate continues parent candidate 3 with the same 10-value dynamics
observation, 128×128 feed-forward ReLU network, 5×7 force and steering grid,
and equal training exposure to circuit, oval, and clover with random starts.
The environment configuration keeps `observe_dynamics` enabled and
`random_start` enabled.

## Explore useful nearby controls

The winner had no evaluation collisions, but made little forward progress on
many fixed starts. Its collision penalty, crash replay share, lane-centering
reward, optimizer, and network remain unchanged so continuation reuses replay
data on the same reward scale and preserves the learned crash avoidance.

Exploration now begins at 0.10 and decays to 0.02 over 60,000 child steps. For
75% of epsilon-selected actions, the trainer samples a neighboring force and
steering command around the current greedy action. The remaining epsilon
actions still sample the full grid, allowing the learner to discover controls
far from its current preference. Nearby exploration should provide more
targeted experience for refining movement while the full-grid branch preserves
broader action discovery. The schedule change starts a fresh exploration phase
when continuing from the parent; repeated continuations checkpoint and resume
that schedule and the random-number-generator states.

The policy remains the evaluator-compatible ReLU MLP and action grid. Both
fresh runs and `--parent` continuations use the existing CLI contract and save
`output/latest.pt` with `online_state_dict`, `actions`, `obs_dim`, and `config`
alongside the trainer state.
