# CarTrack candidate strategy

This candidate continues parent candidate 3. It preserves the 10-value
dynamics observation, 128×128 ReLU Q network, 5×7 force/steering action grid,
three configured tracks, randomized starts, reward shaping, optimizer settings,
and exploration schedule.

## Balance replay across evaluation starts

Training already cycles episode starts across the three tracks and four
quarter-lap anchors, with a small reproducible jitter. However, uniform replay
sampling weights each transition equally, so long episodes and easier
track/start pairs can dominate gradient updates. New transitions now store the
track/anchor group of their episode. Half of each minibatch is drawn evenly
across the groups represented in replay; the existing collision quota remains
in place. The other half stays uniformly sampled, preserving general replay
coverage and compatibility with old parent transitions.

Replay checkpoints now include group labels. Checkpoints from before this
change load with unlabeled transitions, which remain available through the
uniform portion of sampling. The active episode's group is also checkpointed so
continuations keep assigning its transitions consistently. Fresh runs begin
with the same deterministic track/start cycle.

Balanced samples now draw directly from each represented group's eligible
indices, removing the per-sample rejection loop while keeping uniform
conditional sampling and falling back to collisions only when a group has no
non-collision entries.

The policy architecture and action array remain unchanged, so the fixed
evaluator can reconstruct the saved ReLU MLP. The required policy weights,
actions, observation dimension, and merged config remain in `output/latest.pt`.

Run a child with:

```sh
python -m dqn.trainer --config config.json --output output --steps 100000 --parent parent.pt
```

Omit `--parent parent.pt` for a fresh run.
