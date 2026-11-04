# CarTrack continuation strategy

This candidate continues parent candidate 1 while preserving the evaluator's
10-value observation, 128×128 ReLU MLP, and 5×7 force/steering action grid.
Training continues to sample circuit, oval, and clover with equal probability
and uses random starts. These settings keep the inherited Q weights compatible
and expose the policy to varied locations on all three tracks.

## Stabilize the inherited policy and make driving safer

The supplied evaluation has collisions on all three tracks, especially circuit
and clover. This child lowers the learning rate from 5e-6 to 2e-6 to reduce
large changes to the inherited policy. It also lowers epsilon from 0.20 to
0.10, decays it to 0.025 over 60,000 steps, and makes three quarters of
exploration choices neighboring actions. This keeps useful local steering and
force refinement while limiting disruptive full-grid exploration.

Lane-centering shaping increases from 0.08 to 0.16. The reward uses the
environment's existing lateral offset normalized by track half-width, so the
policy receives a denser signal for staying near the center without changing
observations or environment behavior. Collision replay is set to 20% of each
batch, providing consistent crash feedback while retaining the majority of
updates for ordinary driving. The raw collision penalty is reduced from 60 to
12; the additional configurable reward clip bounds the final composed reward
to [-1, 1]. This prevents an isolated crash target from overwhelming many
small progress and lane-centering updates, while every collision still gives
the strongest negative clipped reward.

The trainer applies `reward_clip` after composing forward progress, lane
potential shaping, inactivity cost, and collision penalty. It validates the
clip as finite and positive; omitting it retains the previous unbounded reward
behavior. The new setting is part of the saved config, so continuation runs
resume reproducibly. Existing checkpoint serialization retains model,
optimizer, replay, environment state, and Python, NumPy, and PyTorch RNG states.

The command contract remains `python -m dqn.trainer --config config.json
--output output --steps 100000 --parent parent.pt`; removing `--parent` starts
a fresh run with the same architecture and action grid. Both paths save
`output/latest.pt` with `online_state_dict`, `actions`, `obs_dim`, and `config`.
