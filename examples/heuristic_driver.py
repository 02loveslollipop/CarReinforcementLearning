"""Fixed driver that uses only the 7 sensors. Used to validate the environment and as a baseline.

    python -m examples.heuristic_driver [track] [--render]
"""

import sys

import numpy as np

import utils.car_rl  # noqa: F401  (registers CarTrack-v0)
import gymnasium as gym


def driver(obs: np.ndarray, gain: float = 2.5, k_speed: float = 2.0) -> np.ndarray:
    """Steers toward the side with more room (positive angles = left) and accelerates based on the clear front."""
    left = obs[4] + obs[5] + 0.5 * obs[6]
    right = obs[2] + obs[1] + 0.5 * obs[0]
    steering = np.clip(gain * (left - right) / (left + right + 1e-3), -1, 1)
    front = min(obs[3], obs[2] + 0.2, obs[4] + 0.2)
    # Threshold 0.5: with the environment's base traction, force=0 already moves forward; more
    # is only needed when the front is well clear, and brake when it is close.
    force = np.clip(k_speed * (front - 0.5), -1, 1)
    return np.array([force, steering], dtype=np.float32)


if __name__ == "__main__":
    track = next((a for a in sys.argv[1:] if not a.startswith("--")), "circuit")
    mode = "human" if "--render" in sys.argv else None
    env = gym.make("CarTrack-v0", track=track, render_mode=mode)
    obs, info = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(driver(obs))
        total += r
        done = term or trunc
    print(f"track={info['track']} lap={info['lap_completed']} collision={info['collision']} "
          f"progress={info['progress']:.2f} t={info['time']:.1f}s return={total:.2f} truncated={trunc}")
    env.close()
