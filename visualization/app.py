"""Continuously drive a saved DQN on newly generated circuits."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from utils.car_rl import dynamics
from utils.car_rl.environment import CarTrackEnv, SENSOR_ANGLES
from training.dqn.evaluate import _load_checkpoint
from .circuits import random_circuit

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_TYPES = {"ReLU", "LeakyReLU", "ELU", "SELU", "SiLU", "Mish", "Tanh", "Sigmoid"}
ACTION_COLOR = "#4f8fc0"
CHOSEN_COLOR = "#f2a541"


def default_checkpoint() -> Path:
    candidates = (
        ROOT / "visualization/checkpoints/evolution/latest.pt",
        ROOT / "experiments/runs/continuation/best/latest.pt",
        ROOT / "experiments/runs/main/best/latest.pt",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _flatten(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().reshape(-1).cpu().numpy().astype(np.float32)


def _forward_with_activations(model, observation: np.ndarray) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    """Run one forward pass and record every leaf-module output."""
    value = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
    records: list[tuple[str, torch.Tensor]] = []

    def apply(module, current):
        children = list(module.children())
        if not children:
            current = module(current)
            records.append((type(module).__name__, current))
            return current
        for child in children:
            current = apply(child, current)
        return current

    return apply(model, value), records


def _activation_layers(observation: np.ndarray, records, q_values: np.ndarray):
    """Describe the observation, post-activation hidden layers, and Q outputs."""
    hidden: list[tuple[str, np.ndarray]] = []
    for name, tensor in records[:-1]:
        if name in ACTIVATION_TYPES:
            hidden.append((name, _flatten(tensor)))
    if not hidden:
        hidden = [(name, _flatten(tensor)) for name, tensor in records[:-1]]

    layers: list[tuple[str, np.ndarray]] = [
        ("Input observation", np.asarray(observation, dtype=np.float32).reshape(-1)),
    ]
    for index, (name, values) in enumerate(hidden, start=1):
        stage = f"post-{name}" if name in ACTIVATION_TYPES else f"{name}, pre-activation"
        layers.append((f"Hidden {index} ({stage})", values))
    layers.append(("Output Q-values", np.asarray(q_values, dtype=np.float32)))
    return layers


def _grid_shape(size: int) -> tuple[int, int]:
    if size == 10:
        return 2, 5
    if size == 128:
        return 8, 16
    if size == 35:
        return 5, 7
    columns = max(1, int(np.ceil(np.sqrt(size))))
    return max(1, int(np.ceil(size / columns))), columns


@dataclass
class Decision:
    """One inference pass, captured before the environment applies the action."""

    observation: np.ndarray
    q_values: np.ndarray
    action_index: int
    action: np.ndarray
    activations: list[tuple[str, np.ndarray]] = field(default_factory=list)


class Simulation:
    """Own the policy and create a fresh environment after each episode."""

    def __init__(self, model, actions, obs_dim: int, seed: int | None = None, max_steps: int = 3000,
                 capture_activations: bool = False):
        self.model, self.actions, self.obs_dim = model, actions, obs_dim
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.capture_activations = capture_activations
        self.env = None
        self.observation = None
        self.last_decision: Decision | None = None
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

    def _decide(self) -> Decision:
        observation = self.observation
        with torch.inference_mode():
            if self.capture_activations:
                output, records = _forward_with_activations(self.model, observation)
            else:
                output = self.model(torch.from_numpy(observation).unsqueeze(0))
                records = []
            q_values = _flatten(output)
        action_index = int(q_values.argmax())
        return Decision(
            observation=np.array(observation, dtype=np.float32, copy=True),
            q_values=q_values,
            action_index=action_index,
            action=np.array(self.actions[action_index], dtype=np.float32, copy=True),
            activations=_activation_layers(observation, records, q_values) if records else [],
        )

    def step(self):
        """Advance once; start a fresh circuit on the step after an episode ends."""
        if self.finished:
            self.new_circuit()
        decision = self._decide()
        self.last_decision = decision
        self.observation, _, terminated, truncated, info = self.env.step(decision.action)
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

    def activation_snapshot(self) -> dict:
        """"Exact current observation, action, Q-values, and layer activations."""
        decision = self.last_decision
        if decision is None:
            return {}
        return {
            "episode": self.episode,
            "action_index": decision.action_index,
            "force": float(decision.action[0]),
            "steering": float(decision.action[1]),
            "q_values": [float(value) for value in decision.q_values],
            "layers": {
                label: {"size": int(values.size), "values": [float(value) for value in values]}
                for label, values in decision.activations
            },
        }

    def close(self):
        if self.env is not None:
            self.env.close()


class Window:
    """One Matplotlib window, reused across all generated circuits."""

    def __init__(self, simulation: Simulation, fps: float):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.simulation = simulation
        self.simulation.capture_activations = True
        self.fig = plt.figure(figsize=(17, 9))
        self.fig.canvas.manager.set_window_title("DQN random circuits")
        self.fig.patch.set_facecolor("#161f2b")
        grid = self.fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.88, bottom=0.07,
                                     wspace=0.12, hspace=0.30,
                                     width_ratios=(1.35, 1.1, 1.15), height_ratios=(1.35, 1.0))
        self.overview = self.fig.add_subplot(grid[0, 0])
        self.chase = self.fig.add_subplot(grid[0, 1])
        self.q_axes = self.fig.add_subplot(grid[0, 2])
        self.activation_area = grid[1, :]
        self.header = self.fig.text(0.05, 0.965, "", color="white", fontsize=15, weight="bold", va="top")
        self.detail = self.fig.text(0.05, 0.925, "", color="#b9c8d5", fontsize=10, va="top")
        self.activation_note = self.fig.text(
            0.5, 0.018,
            "Network values come from the single pass that chose this action; the car shown has already advanced by it. "
            "Hidden layers are post-ReLU. Colors: blue < 0 < red; each panel uses its own maximum |value|. "
            "Press P to print exact values.",
            color="#8fa5b5", fontsize=8, ha="center", va="bottom")
        self.closed = False
        self.displayed_episode = 0
        self.activation_axes = []
        self.activation_images = []
        self.activation_titles = []
        self.activation_texts = []
        self.timer = self.fig.canvas.new_timer(interval=max(1, round(1000 / fps)))
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._setup_q_axes(len(simulation.actions))
        self.draw_track()
        self.draw_car()

    def _style_axes(self, axes, title: str):
        axes.set_facecolor("#1b2632")
        axes.set_title(title, color="#d8e6ed", fontsize=9, loc="left", pad=6)
        axes.tick_params(colors="#b9c8d5", labelsize=7)
        for spine in axes.spines.values():
            spine.set_color("#425666")

    def _setup_q_axes(self, action_count: int):
        self._style_axes(self.q_axes, "Output layer: Q-values")
        self.q_axes.set_xlabel("action index (force × steering)", color="#b9c8d5", fontsize=7)
        self.q_axes.set_ylabel("Q-value", color="#b9c8d5", fontsize=7)
        self.q_bars = self.q_axes.bar(np.arange(action_count), np.zeros(action_count),
                                      color=ACTION_COLOR, edgecolor="none", width=0.8)
        self.q_axes.axhline(0.0, color="#7f93a3", linewidth=0.6)
        self.q_annotation = self.q_axes.text(
            0.02, 0.97, "", transform=self.q_axes.transAxes, color="#f6d6a5",
            fontsize=8, va="top", ha="left")

    def _ensure_activation_axes(self, layers):
        if len(self.activation_axes) == len(layers):
            return
        for axes in self.activation_axes:
            axes.remove()
        self.activation_axes, self.activation_images = [], []
        self.activation_titles, self.activation_texts = [], []
        if not layers:
            return
        grid = self.activation_area.subgridspec(1, len(layers), wspace=0.14)
        for column, (label, values) in enumerate(layers):
            axes = self.fig.add_subplot(grid[0, column])
            rows, columns = _grid_shape(int(values.size))
            image = axes.imshow(np.zeros((rows, columns), dtype=np.float32), cmap="RdBu_r",
                                vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")
            axes.set_xticks([])
            axes.set_yticks([])
            self._style_axes(axes, "")
            title = axes.title
            texts = []
            if int(values.size) <= 20:
                for index in range(int(values.size)):
                    row, col = divmod(index, columns)
                    texts.append(axes.text(col, row, "", color="#f1f5f8", fontsize=7,
                                           ha="center", va="center"))
            self.activation_axes.append(axes)
            self.activation_images.append(image)
            self.activation_titles.append(title)
            self.activation_texts.append(texts)

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

    def draw_network(self):
        decision = self.simulation.last_decision
        if decision is None:
            return
        q_values = decision.q_values
        if len(q_values) != len(self.q_bars):
            for bar in self.q_bars:
                bar.remove()
            self._setup_q_axes(len(q_values))
        for index, (bar, value) in enumerate(zip(self.q_bars, q_values)):
            bar.set_height(float(value))
            bar.set_color(CHOSEN_COLOR if index == decision.action_index else ACTION_COLOR)
        force, steering = (float(value) for value in decision.action)
        chosen_q = float(q_values[decision.action_index])
        self.q_axes.set_title(
            f"Output layer: Q-values ({len(q_values)} actions)  •  chosen #{decision.action_index}",
            color="#d8e6ed", fontsize=9, loc="left", pad=6)
        self.q_annotation.set_text(f"force {force:+.2f}\nsteering {steering:+.2f}\nQ {chosen_q:+.3f}")
        span = float(np.ptp(q_values)) or 1.0
        self.q_axes.set_ylim(float(q_values.min()) - 0.05 * span, float(q_values.max()) + 0.05 * span)

        layers = [layer for layer in decision.activations if not layer[0].startswith("Output")]
        self._ensure_activation_axes(layers)
        for (label, values), image, title, texts in zip(
                layers, self.activation_images, self.activation_titles, self.activation_texts):
            rows, columns = _grid_shape(int(values.size))
            image.set_data(np.asarray(values, dtype=np.float32).reshape(rows, columns))
            scale = max(float(np.abs(values).max()), 1e-6)
            image.set_clim(-scale, scale)
            title.set_text(f"{label} • n={values.size} • [{values.min():+.2f}, {values.max():+.2f}]")
            for text, value in zip(texts, values):
                text.set_text(f"{value:+.2f}")
                text.set_color("#0b151a" if abs(float(value)) < 0.55 * scale else "#f1f5f8")
        self.fig.canvas.draw_idle()

    def tick(self):
        if self.closed:
            return False
        self.simulation.step()
        if self.displayed_episode != self.simulation.episode:
            self.draw_track()
        self.draw_car()
        self.draw_network()
        return True

    def on_key(self, event):
        if event.key in {"p", "P"}:
            print(json.dumps(self.simulation.activation_snapshot(), indent=2), flush=True)

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
    simulation = Simulation(model, actions, model[0].in_features, seed=args.seed, max_steps=args.max_steps,
                            capture_activations=False)
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
