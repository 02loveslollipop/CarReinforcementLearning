# CarTrack DQN strategy

This candidate keeps the parent's evaluator-compatible policy architecture and
action grid: a 10-input, two-hidden-layer 128-unit ReLU MLP with five force
levels and seven steering levels. The environment keeps dynamics observations
enabled, samples randomized starts, and includes circuit, oval, and clover.
Retaining the exact network dimensions and action list lets `--parent parent.pt`
continue from the assigned checkpoint.

## Learning changes

- Raise the collision penalty from 3 to 50 while retaining signed forward
  progress as the main reward and a small inactivity cost. The previous
  generation's stronger-penalty candidate reported substantially better oval
  completion than this parent, whose policies collided on every evaluation
  start. A larger terminal cost gives the learner a clearer signal against
  fast but unsafe steering.
- Use Double DQN targets, as in the starting trainer, to reduce overestimation
  across the 35 discrete actions.
- Use Polyak target updates with `target_update_tau=0.01`, applied after each
  optimizer step. This smooths value targets; `target_update_frequency` remains
  available as the hard-update fallback when tau is zero.
- Keep the 3e-4 learning rate, 128-transition batches, 2,000-step warmup,
  two-step training frequency, and gradient clipping at 5. These retain the
  existing stable update scale while replay gains diverse track and start
  experience.
- Decay epsilon from 1.0 to 0.05 over 60,000 global steps, preserving some
  exploration while exploiting more during later training. A continuation
  respects the saved global step and therefore resumes at the corresponding
  point in this schedule.

## Reproducibility and continuation

The seed remains fixed at 6. Checkpoints preserve online and target weights,
optimizer, replay, active episode, and random-number-generator states. The
trainer continues to validate observation size and the action grid when loading
a parent. `output/latest.pt` retains `online_state_dict`, `actions`, `obs_dim`,
and `config` for the fixed evaluator. The policy remains a feed-forward ReLU MLP
whose Linear weights can be reconstructed by that evaluator.
