# CarTrack continuation strategy

This candidate continues parent candidate 1 with the same 10-value dynamics
observation, 128×128 ReLU Q network, 5×7 force/steering action grid, and equal
training weight for circuit, oval, and clover. Training continues to use random
starts on all three tracks.

## Explore around useful actions while retaining broad coverage

The parent's reported mean score was low across the fixed starts, with only
small forward progress and a 25% collision rate. This child gives the inherited
policy a larger but temporary exploration budget: epsilon decays from 0.10 to
0.02 over 70,000 child steps. On 80% of exploration events, the trainer samples
a neighboring force/steering grid point around the current greedy action. The
remaining exploration events still sample the full action grid. Local changes
can refine steering and speed choices without making every exploratory action a
large change; occasional global actions preserve broader discovery. The
exploration schedule and local-action fraction are checkpointed together, so a
later continuation resumes the same schedule reproducibly.

The collision penalty is reduced from 50 to 25, and collision transitions fill
15% rather than 20% of replay batches. The parent had both crashes and many
truncated low-progress episodes, so this moderates the influence of a crash
sample while keeping crashes strongly negative and replayed more often than
ordinary transitions. Progress reward, lane-centering shaping, learning rate,
network, optimizer cadence, track sampling, and action grid otherwise remain as
configured to focus this candidate on exploration and the progress/crash
balance.

The trainer retains deterministic seeded RNG state, parent checkpoint loading,
and `output/latest.pt` fields `online_state_dict`, `actions`, `obs_dim`, and
`config` for the fixed evaluator. It supports both fresh and continued runs.
