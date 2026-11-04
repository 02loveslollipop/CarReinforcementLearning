# CarTrack continuation strategy

This candidate continues the assigned parent candidate 0. It keeps the 10-value
dynamics observation, the 128×128 feed-forward ReLU network, the 5×7 force and
steering action grid, and random starts across circuit, oval, and clover. Track
sampling remains equal to match the evaluator's equal weighting of tracks.

## Reduce risky policy drift and reinforce crash avoidance

The parent's mean score is low and it collided on half of circuit starts, every
oval start, and half of clover starts. The continuation lowers the learning
rate from 0.00005 to 0.00002 so replay can refine the inherited policy without
large parameter changes. Exploration starts at 0.05 and decays to 0.01 over
40,000 child steps, retaining some action discovery while spending most of the
run close to the learned greedy policy.

Collision replay rises from 30% to 40% of sampled transitions. The sampler now
fills the remaining batch from non-collision examples when available, so
uniform sampling cannot unexpectedly push the crash share above the configured
quota. This is useful early in training when the crash pool is small and replay
would otherwise duplicate those few transitions in both the crash and uniform
parts of a batch. The collision penalty stays at 50, preserving reward-scale
compatibility with the parent's replay.

Lane-centering shaping increases from 0.02 to 0.04 to give the learner a
stronger signal to move away from track edges before a collision. Progress
reward, inactivity settings, target-network update rate, architecture, action
grid, and equal track sampling stay fixed to limit unrelated changes. The seed
and checkpointed random-number-generator states keep runs and continuations
reproducible.

The trainer retains fresh-run and `--parent` paths. It saves
`output/latest.pt` with the evaluator-required `online_state_dict`, `actions`,
`obs_dim`, and `config` fields, and atomically replaces the checkpoint only
after serialization completes.
