"""Matplotlib visualization (lazy import: the environment works without it)."""

from __future__ import annotations

import numpy as np

from . import dynamics


class Viewer:
    def __init__(self, track, car, mode: str, fps: int):
        import matplotlib

        if mode == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        self.plt, self.track, self.car, self.mode, self.fps = plt, track, car, mode, fps
        self.fig, self.ax = plt.subplots(figsize=(7, 7), dpi=90)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.mini = self.fig.add_axes([0.02, 0.02, 0.27, 0.27], zorder=10)
        background = "#6f7f8c"
        self.fig.patch.set_facecolor(background)
        left, right = track.left_edge, track.right_edge
        road = np.vstack([left, left[:1], right[:1], right[::-1]])
        for ax in (self.ax, self.mini):  # same track in the main view and the minimap
            ax.set_aspect("equal")
            ax.set_facecolor(background)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(Polygon(road, closed=True, fc="#3b4650", ec="none"))
            for edge in (left, right):
                closed = np.vstack([edge, edge[:1]])
                ax.plot(closed[:, 0], closed[:, 1], color="white", lw=2 if ax is self.ax else 1)
            ax.plot(*np.stack([left[0], right[0]]).T, color="white", lw=3 if ax is self.ax else 1.5)  # start line
        # Main view: frame centered on the car; the minimap shows the whole track.
        self.half = 30.0
        self.ax.axis("off")
        lim = np.vstack([left, right])
        margin = 15.0
        self.mini.set_xlim(lim[:, 0].min() - margin, lim[:, 0].max() + margin)
        self.mini.set_ylim(lim[:, 1].min() - margin, lim[:, 1].max() + margin)
        for sp in self.mini.spines.values():
            sp.set_edgecolor("white")
            sp.set_linewidth(1.5)
        self.point = self.mini.plot([], [], "o", color="#ff4d4d", ms=6, zorder=6)[0]
        self.body = Polygon(np.zeros((4, 2)), closed=True, fc="#2fbf5b", ec="black", zorder=5)
        self.ax.add_patch(self.body)
        self.lines = [self.ax.plot([], [], color="white", lw=0.8, alpha=0.9, zorder=4)[0] for _ in range(7)]
        self.text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", color="white", family="monospace")
        if mode == "human":
            plt.ion()
            self.fig.show()

    def draw(self, state, rays, angles):
        x, y = state[0], state[1]
        self.ax.set_xlim(x - self.half, x + self.half)
        self.ax.set_ylim(y - self.half, y + self.half)
        self.point.set_data([x], [y])
        self.body.set_xy(dynamics.corners(state, self.car))
        for ln, d, a in zip(self.lines, rays, angles):
            ang = state[2] + a
            ln.set_data([x, x + d * np.cos(ang)], [y, y + d * np.sin(ang)])
        self.text.set_text(f"v = {np.hypot(state[3], state[4]) * 3.6:5.1f} km/h")
        if self.mode == "human":
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(1.0 / self.fps)
            return None
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        self.plt.close(self.fig)
