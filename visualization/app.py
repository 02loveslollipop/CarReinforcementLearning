"""Continuously drive a saved DQN on newly generated circuits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_rl import dynamics
from car_rl.environment import CarTrackEnv, SENSOR_ANGLES
from dqn.evaluate import _load_checkpoint
from .circuits import random_circuit

ROOT = Path(__file__).resolve().parents[1]


def default_checkpoint() -> Path:
    candidates = (
        ROOT / "visualization/checkpoints/evolution/latest.pt",
        ROOT / "experiments/runs/continuation/best/latest.pt",
        ROOT / "experiments/runs/main/best/latest.pt",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


class Simulation:
    """Own the policy and create a fresh environment after each episode."""

    def __init__(self, model, actions, obs_dim: int, seed: int | None = None, max_steps: int = 3000):
        self.model, self.actions, self.obs_dim = model, actions, obs_dim
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.env = None
        self.observation = None
        self.episode = 0
        self.finished = False
        self.previous_result = None
        self.new_circuit()

    def new_circuit(self):
        if self.env is not None:
            self.env.close()
        self.episode += 1
        track = random_circuit(self.rng, self.episode)
        self.env = CarTrackEnv(track=track, observe_dynamics=self.obs_dim != 7, max_steps=self.max_steps)
        self.observation, _ = self.env.reset()
        self.finished = False

    def step(self):
        """Advance once; start a fresh circuit on the step after an episode ends."""
        if self.finished:
            self.new_circuit()
        with torch.inference_mode():
            values = self.model(torch.from_numpy(self.observation).unsqueeze(0))
            action = self.actions[int(values.argmax(dim=1).item())]
        self.observation, _, terminated, truncated, info = self.env.step(action)
        self.finished = bool(terminated or truncated)
        if self.finished:
            outcome = ("lap completed" if info["lap_completed"] else
                       "collision" if info["collision"] else
                       "no progress" if self.env.steps_without_improvement >= self.env.max_no_improvement else
                       "time limit")
            self.previous_result = {
                "episode": self.episode,
                "track": self.env.track.name,
                "outcome": outcome,
                "progress": float(info["progress"]),
                "elapsed_seconds": float(info["time"]),
            }
        return self.finished

    def close(self):
        if self.env is not None:
            self.env.close()


class Window:
    """One Matplotlib window, reused across all generated circuits."""

    def __init__(self, simulation: Simulation, fps: float):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.simulation = simulation
        self.fig, (self.overview, self.chase) = plt.subplots(1, 2, figsize=(13, 7))
        self.fig.canvas.manager.set_window_title("DQN random circuits")
        self.fig.patch.set_facecolor("#161f2b")
        self.fig.subplots_adjust(left=0.04, right=0.97, top=0.86, bottom=0.09, wspace=0.12)
        self.header = self.fig.text(0.05, 0.95, "", color="white", fontsize=15, weight="bold", va="top")
        self.detail = self.fig.text(0.05, 0.90, "", color="#b9c8d5", fontsize=10, va="top")
        self.closed = False
        self.displayed_episode = 0
        self.timer = self.fig.canvas.new_timer(interval=max(1, round(1000 / fps)))
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.draw_track()
        self.draw_car()

    def draw_track(self):
        from matplotlib.patches import Polygon

        env = self.simulation.env
        track = env.track
        self.displayed_episode = self.simulation.episode
        self.markers = []
        self.bodies = []
        self.rays = []
        left, right = track.left_edge, track.right_edge
        road = np.vstack((left, left[:1], right[:1], right[::-1]))
        for ax, title in ((self.overview, "CIRCUIT"), (self.chase, "CAR VIEW")):
            ax.clear()
            ax.set_facecolor("#627987")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, color="#d8e6ed", fontsize=10, loc="left", pad=8)
            ax.add_patch(Polygon(road, closed=True, facecolor="#303e4a", edgecolor="none"))
            for edge in (left, right):
                closed = np.vstack((edge, edge[:1]))
                ax.plot(closed[:, 0], closed[:, 1], color="#dbe5e8", linewidth=1)
            ax.plot(*np.stack((left[0], right[0])).T, color="#ffffff", linewidth=2)
            marker, = ax.plot([], [], "o", color="#f06b55", markersize=6, zorder=6)
            body = Polygon(np.zeros((4, 2)), closed=True, facecolor="#51d39b", edgecolor="#0b151a", zorder=7)
            ax.add_patch(body)
            self.markers.append(marker)
            self.bodies.append(body)
            if ax is self.chase:
                self.rays = [ax.plot([], [], color="#bcebd8", linewidth=0.8, alpha=0.7)[0] for _ in SENSOR_ANGLES]
        extent = np.vstack((left, right))
        margin = 20
        self.overview.set_xlim(extent[:, 0].min() - margin, extent[:, 0].max() + margin)
        self.overview.set_ylim(extent[:, 1].min() - margin, extent[:, 1].max() + margin)

    def draw_car(self):
        env = self.simulation.env
        x, y, heading = env.state[:3]
        for marker, body in zip(self.markers, self.bodies):
            marker.set_data([x], [y])
            body.set_xy(dynamics.corners(env.state, env.car))
        self.chase.set_xlim(x - 35, x + 35)
        self.chase.set_ylim(y - 35, y + 35)
        for line, distance, angle in zip(self.rays, env._rays, SENSOR_ANGLES):
            direction = heading + angle
            line.set_data([x, x + distance * np.cos(direction)], [y, y + distance * np.sin(direction)])
        speed = float(np.hypot(env.state[3], env.state[4])) * 3.6
        self.header.set_text(f"Random circuit {self.simulation.episode}  •  {speed:.1f} km/h")
        prior = self.simulation.previous_result
        previous = (f"Previous: {prior['outcome']}  ·  {prior['progress']:.1%} lap" if prior else "Close the window to stop")
        self.detail.set_text(f"Lap progress: {max(0.0, env.progress / env.track.length):.1%}     {previous}")
        self.fig.canvas.draw_idle()

    def tick(self):
        if self.closed:
            return False
        self.simulation.step()
        if self.displayed_episode != self.simulation.episode:
            self.draw_track()
        self.draw_car()
        return True

    def on_close(self, _event):
        self.closed = True
        self.timer.stop()
        self.simulation.close()

    def show(self):
        try:
            self.timer.start()
            self.plt.show(block=True)
        finally:
            self.timer.stop()
            self.simulation.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, help="DQN checkpoint (defaults to the best saved model)")
    parser.add_argument("--seed", type=int, default=None, help="reproduce the same sequence of random circuits")
    parser.add_argument("--fps", type=float, default=20.0, help="visualization updates per second (default: 20)")
    parser.add_argument("--max-steps", type=int, default=3000, help="maximum simulation steps per circuit")
    parser.add_argument("--headless", action="store_true", help="run without a window, for testing")
    parser.add_argument("--episodes", type=int, default=1, help="number of headless episodes (default: 1)")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.fps) or args.fps <= 0 or args.max_steps < 1 or args.episodes < 1
            or (args.seed is not None and args.seed < 0)):
        parser.error("--fps, --max-steps and --episodes must be positive; --seed must be non-negative")
    checkpoint_path = (args.checkpoint or default_checkpoint()).expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}; pass a trained checkpoint path")
    if not args.headless:
        import matplotlib.pyplot as plt
        if plt.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
            parser.error("Matplotlib has no interactive display backend; run on a desktop or use --headless")
    torch.set_num_threads(1)
    _, model, actions = _load_checkpoint(checkpoint_path)
    simulation = Simulation(model, actions, model[0].in_features, seed=args.seed, max_steps=args.max_steps)
    if args.headless:
        try:
            completed = 0
            while completed < args.episodes:
                if simulation.step():
                    print(json.dumps(simulation.previous_result), flush=True)
                    completed += 1
        finally:
            simulation.close()
    else:
        Window(simulation, args.fps).show()


if __name__ == "__main__":
    main()
