# CarTrack DQN strategy

Promoted model from post-promotion iteration D (`D_steps2k/seed_11`).
Parent: `lr_1e5_eps_end_001` at `global_step=205000` (see `latest_iter2.pt`).
This candidate adds 2,000 steps with `learning_rate=1e-5`, `epsilon_end=0.01`,
`resume_rng=false` (fresh env/replay streams for true seed diversity, optimizer
and replay contents resumed), train_frequency 2, batch 128. It keeps the
10-input, 128x128 ReLU MLP and 35-action grid.

Reproducibility:
- Parent checkpoint: `experiments/runs/speed_improvement/lr_1e5_eps_end_001/seed_11/checkpoint/latest.pt`
- Candidate: `experiments/runs/speed_improvement/D_steps2k/seed_11/checkpoint/latest.pt`
- Evaluation: `evaluation.json` (fixed 12), `speed_tune.json` (48), `speed_holdout.json` (64).

Results vs previous best (12/12, tune 15.66, holdout 15.80):
- Fixed: 12/12, score 0.9653, 0 collisions
- Tune: 48/48 at 17.172 m/s
- Holdout: 64/64 at 17.293 m/s
