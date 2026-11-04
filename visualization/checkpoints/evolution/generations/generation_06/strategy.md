# CarTrack candidate strategy

This candidate keeps the 10-value dynamics observation, the evaluator-compatible
128×128 ReLU Q network, the 5×7 force/steering grid, and the inherited replay
and optimizer state. The environment remains configured for circuit, oval, and
clover with random starts.

## Actually train on all configured tracks

The trainer temporarily reduces the environment's track list to the selected
track before resetting. `CarTrackEnv.reset()` only changes its active track
when that list has multiple entries, so the active track could remain circuit
for every episode. The trainer now sets the selected track explicitly before
the reset. This lets the existing equal-weight track sampler train on all
three tracks as intended, including during child continuation.

## Cover evaluation start regions

Training now cycles episode starts through four equal longitudinal bands per
track and draws a seeded random position inside each band. This retains random
starts while improving coverage of the four regions used by fixed evaluation
starts in finite runs. The episode counter is already checkpointed, and the
environment RNG state is saved, so the start sequence continues reproducibly.

Exploration increases slightly from 0.22 to 0.26 at the start of this child
phase, and the local-action share drops from 45% to 35%. More exploratory steps
therefore test actions across the full action grid while preserving local
refinement. The existing low learning rate, collision penalty, collision replay
share, and lane-centering shaping stay in place to limit disruption to the
inherited policy and retain its safety feedback.

Fresh training and continuation use the existing CLI and save
`output/latest.pt` with `online_state_dict`, `actions`, `obs_dim`, and `config`.
