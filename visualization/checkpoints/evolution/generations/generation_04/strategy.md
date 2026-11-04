# CarTrack candidate 4 continuation strategy

This candidate continues selected parent candidate 2 from `parent.pt`. It keeps
the evaluator-compatible 10-value dynamics observation, the 128×128 ReLU Q
network, and the existing 5×7 force/steering action grid. Training samples
circuit, oval, and clover equally, with random starts enabled on all three.

## Reinforce safer progress

The parent made forward progress but collided on every fixed evaluation start.
This child increases the collision cost from 25 to 40 so a risky fast trajectory
has a stronger negative target. It also raises lane-centering potential shaping
from 0.02 to 0.08. That gives the policy a denser signal for moving away from
the track edge before a collision occurs. Collision transitions remain in
replay, with their reserved batch share reduced from 15% to 10% to keep the
learner focused on the much larger variety of ordinary driving states.

## Continue adapting the inherited policy

The learning rate rises from 1e-5 to 2e-5 so the inherited policy can adapt
more within this 100,000-step continuation. Epsilon starts at 0.14 and decays
to 0.025 over 60,000 steps. Three quarters of exploratory actions are sampled
near the greedy force/steering pair; the rest still cover the full grid. The
trainer records the schedule and RNG states in the checkpoint, so a later child
with the same schedule resumes reproducibly. Changing the schedule starts a
new exploration phase.

The trainer now checks that each run uses exactly the three required tracks,
random starts, and dynamics observations. It retains parent checkpoint loading,
the established CLI flags, and `output/latest.pt` fields
`online_state_dict`, `actions`, `obs_dim`, and `config`; fresh runs use the same
path without `--parent`.
