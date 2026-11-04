# CarTrack candidate strategy

This candidate continues parent candidate 0 with the same 10-value dynamics
observation, 128×128 ReLU Q network, 5×7 force/steering action grid, and
quarter-lap anchors across circuit, oval, and clover. Random starts and
`observe_dynamics` remain enabled.

## Adapt while preserving broad coverage

Replay already protects collision transitions and allocates half its batch
across represented track/start pairs. This candidate keeps both mechanisms and
adds a recent-transition quota. After the protected crash and balanced-group
slots are filled, the sampler requests one quarter of a batch from the newest 20,000
transitions, capped by the remaining slots. Recent sampling prefers
non-collision examples, while the collision quota remains available. This
gives fine-tuning a path to learn from the behavior collected by the child
policy instead of relying almost entirely on the parent's older replay data.
The replay window follows the ring buffer and its position is already saved in
checkpoints, so sampling remains reproducible through continuations.

The balanced fraction is raised slightly to 0.55 so less common starts keep a
steady signal alongside recent experience. The learning rate is doubled to
1e-5 for useful but restrained adaptation. Exploration starts at 0.12 and
decays to 0.02 over 50,000 child steps; changing this schedule intentionally
starts a fresh exploration phase while preserving the stored RNG states.

## Compatibility

The policy remains a feed-forward ReLU MLP and the action grid is unchanged.
Checkpoints continue to include `online_state_dict`, `actions`, `obs_dim`, and
`config`, along with optimizer, replay, environment, and RNG state. The existing
CLI supports both fresh runs and continuations:

```sh
python -m dqn.trainer --config config.json --output output --steps 100000 --parent parent.pt
```

Omit `--parent parent.pt` for a fresh run.
