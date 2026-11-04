# Speed improvement experiments

## Current best

Best checkpoint: `visualization/checkpoints/evolution/latest.pt` (= `weights.zip`).
It is the post-promotion iteration **D** (`D_steps2k/seed_11`):
- Parent: `lr_1e5_eps_end_001` (global_step 205000), itself 5000 steps from the
  original evolution baseline at `lr=1e-5, epsilon_end=0.01`.
- Candidate: +2000 steps, `lr=1e-5, epsilon_end=0.01, resume_rng=false,
  train_frequency=2, batch_size=128`.
- Fixed: **12/12, score 0.9653, 0 collisions**, mean speed ~16.7 m/s.
- Tune (seeds 1000-1011, 48 eps): **48/48 at 17.172 m/s**.
- Holdout (seeds 2000-2015, 64 eps): **64/64 at 17.293 m/s**.

Previous best (iter2, `latest_iter2.pt`): 12/12, tune 15.659, holdout 15.802.
Original baseline (`latest_baseline.pt`): 10/12, tune 13.341, holdout 13.338.

## Evaluation protocol

- Fixed benchmark: unchanged `python -m training.dqn.evaluate <checkpoint>`.
- Speed suites: `python -m training.dqn.speed_eval <checkpoint> --suite tune|holdout`
  (fixed seeds, start fractions 0/0.25/0.5/0.75, max_steps 3000, dt 0.05).
  Metric: uncapped mean over completed laps of `track_length/lap_time`.
- Selection rule: require fixed completion **12/12** (no regression, in fact
  improvement over 10/12), then maximise tune completed-lap speed, confirm on
  holdout, and report completion counts and collisions. Crashes cannot win.

## Key lessons

1. **resume_rng=true makes seeds non-independent.** With the parent's saved RNG
   restored, all seeds give identical continuations (37 duplicate checkpoints).
   True seed diversity requires `resume_rng=false` (fresh env/replay streams,
   optimizer/replay contents resumed). Iteration 1's collapse at `lr=3e-4` was
   due to LR + epsilon floor, not RNG.
2. **Low LR is essential at the epsilon floor.** `lr=3e-4` collapses to 0/12.
   `lr=1e-5` with `epsilon_end=0.01` gives 12/12. Shorter budgets (1-2k) are
   safer than 5k-20k.
3. **Reward shaping (time penalty) is fragile.** Small penalties (0.005-0.02)
   sometimes help one seed but hurt reliability on others; larger ones collapse.
   `reset_replay=true` collapses (0/12).
4. **Speed vs reliability trade-off.** Faster configs (e.g. train_freq 8/batch 64)
   reach 16.8 tune but only 9/12 fixed. The gate requiring 12/12 fixed filters
   these out.
5. **Variance is high.** The same config with different seeds spans 0.73-0.97
   fixed score and 13.7-17.2 m/s tune. Single-seed wins must be confirmed on
   holdout and ideally repeated.

## Iterations after promotion (no improvement over D)

Iterations B-S (18 distinct configs, 60+ runs, each 2-5k steps, single-factor,
3 seeds each where possible, `resume_rng=false`):

- B: control vs time_penalty 0.005 at 5k (no beat)
- C: `epsilon_end=0.005`, `lr=5e-6` (no beat; best 16.01 tune)
- D: **2000 steps** (best 17.172 tune, 17.293 holdout) -- PROMOTED
- E: `lr=5e-6` 2k/3k (best 15.89)
- F: 1k steps (best 15.70)
- G: `lr=3e-6` 5k, time 2k (best 15.87)
- H: train_freq 8/4 batch 64 (fast but unreliable, best 16.23 tune at 11/12)
- I/J: repeats of D config across seeds 11-14 (same best, variance confirmed)
- K: 2k/3k + time (best 15.76)
- L: 1k, `lr=5e-6` 2k (best 15.72)
- M: 2k, `lr=1.5e-5` (collapse risk, best 16.97 at 7/12 -- rejected)
- N: collision penalty 30/70 (no beat)
- O: D config seeds 15-18 (two collapses, best 14.78)
- P: 1.5k/2.5k steps (best 15.77, often unreliable)
- Q: time 0.002/0.005 at 2k (best 15.92, unreliable)
- R: fresh 2k/3k repeats of D (same 16.39 max)
- S: `epsilon_end=0.02`, `lr=7e-6` (no beat)

All post-D candidates with 12/12 fixed scored **< 17.0 m/s** on tune.
Distinct no-improvement iterations after D: **>10**, so the stopping criterion
is met. Best is left as D.

## Files

- Best: `visualization/checkpoints/evolution/latest.pt` + `weights.zip`
  (`latest.pt`, `config.json`, `evaluation.json`, `strategy.md`).
- Previous best: `latest_iter2.pt`; original baseline: `latest_baseline.pt`.
- Reports: `evaluation.json` (fixed), `speed_tune.json`, `speed_holdout.json`.
- Per-candidate: `experiments/runs/speed_improvement/*/seed_*/report.json`
  (git-ignored, local only).

## Reproduce best

```sh
podman run -i --rm --security-opt label=disable   -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work localhost/car-rl-dev:latest   python -m training.dqn.evaluate visualization/checkpoints/evolution/latest.pt
podman run -i --rm --security-opt label=disable   -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work localhost/car-rl-dev:latest   python -m training.dqn.speed_eval visualization/checkpoints/evolution/latest.pt --suite tune
```
