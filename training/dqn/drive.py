"""Drive a saved DQN checkpoint on a chosen track.

Example: python -m training.dqn.drive experiments/runs/main/best/latest.pt --track circuit
"""
from __future__ import annotations

import argparse
import json

import torch

from utils.car_rl.environment import CarTrackEnv
from .evaluate import _load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='trained DQN checkpoint')
    parser.add_argument('--track', choices=('circuit', 'oval', 'clover'), default='circuit')
    parser.add_argument('--start', type=float, default=0.0, help='fraction of lap at which to start')
    parser.add_argument('--headless', action='store_true', help='run without opening a viewer')
    args = parser.parse_args()
    if not 0.0 <= args.start < 1.0:
        parser.error('--start must be at least 0 and less than 1')
    torch.set_num_threads(1)
    checkpoint, model, actions = _load_checkpoint(args.checkpoint)
    env = CarTrackEnv(track=args.track, observe_dynamics=(int(checkpoint.get('obs_dim', 10)) != 7),
                      render_mode=None if args.headless else 'human')
    try:
        obs, _ = env.reset(seed=0, options={'s0': env.track.length * args.start})
        done = False
        while not done:
            with torch.inference_mode():
                action = actions[int(model(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(json.dumps(info, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
