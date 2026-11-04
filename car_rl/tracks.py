"""Closed tracks defined by a center line (periodic spline) and a width."""

from __future__ import annotations

import numpy as np


def _catmull_rom_closed(points: np.ndarray, samples_per_segment: int = 60) -> np.ndarray:
    n = len(points)
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    segments = []
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k) % n] for k in (-1, 0, 1, 2))
        segments.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    return np.vstack(segments)


def _rotate(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


class Track:
    """Closed circuit.

    Parameters
    ----------
    points : (K, 2) control points (meters) the center line passes through.
    width : total roadway width in meters.
    samples : number of equally spaced points of the center line.
    """

    def __init__(self, points, width: float = 12.0, samples: int = 900, name: str = "custom"):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError("points must have shape (K, 2) with K >= 4")
        self.name = name
        self.half_width = width / 2.0

        fine = _catmull_rom_closed(points)
        closed = np.vstack([fine, fine[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(cum[-1])
        self.n = samples
        self.ds = self.length / samples
        target = np.arange(samples) * self.ds
        self.center = np.stack(
            [np.interp(target, cum, closed[:, 0]), np.interp(target, cum, closed[:, 1])], axis=1
        )
        self.s = target

        # Tangents, (left) normals and curvature
        nxt = np.roll(self.center, -1, axis=0)
        prev = np.roll(self.center, 1, axis=0)
        tangent = nxt - prev
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        self.tangent = tangent
        self.normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        dh = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))
        self.curvature = dh / self.ds
        min_radius = 1.0 / max(np.abs(self.curvature).max(), 1e-9)
        if min_radius < self.half_width * 1.05:
            raise ValueError(
                f"Track '{name}': minimum curvature radius {min_radius:.1f} m < half width {self.half_width:.1f} m; "
                "the edges would cross. Smooth the track or reduce the width."
            )
        self.min_radius = min_radius

        self.left_edge = self.center + self.normal * self.half_width
        self.right_edge = self.center - self.normal * self.half_width
        a = np.vstack([self.left_edge, self.right_edge])
        b = np.vstack([np.roll(self.left_edge, -1, axis=0), np.roll(self.right_edge, -1, axis=0)])
        self._seg_a, self._seg_e = a, b - a

    # --------------------------------------------------------------- geometry
    def start_point(self, s: float = 0.0):
        """Position and heading (rad) on the center line at coordinate s."""
        i = int(round((s % self.length) / self.ds)) % self.n
        return self.center[i].copy(), float(np.arctan2(self.tangent[i, 1], self.tangent[i, 0]))

    def project(self, p: np.ndarray):
        """Project p onto the center line. Returns (s, signed lateral offset (+ left))."""
        d2 = np.sum((self.center - p) ** 2, axis=1)
        i = int(np.argmin(d2))
        best = None
        for j in (i - 1, i):  # segments (i-1 -> i) and (i -> i+1)
            a = self.center[j % self.n]
            e = self.center[(j + 1) % self.n] - a
            t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0.0, 1.0)
            q = a + t * e
            dist = np.linalg.norm(p - q)
            if best is None or dist < best[0]:
                sign = np.sign(e[0] * (p[1] - q[1]) - e[1] * (p[0] - q[0]))
                best = (dist, ((j % self.n) + t) * self.ds, sign * dist)
        return best[1], best[2]

    def ray_distances(self, pos: np.ndarray, heading: float, angles: np.ndarray, sensor_range: float) -> np.ndarray:
        """Distance from pos to the nearest edge along each ray (saturated at sensor_range)."""
        ang = heading + angles
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (R, 2)
        ao = self._seg_a - pos  # (M, 2)
        e = self._seg_e  # (M, 2)
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (R, M)
        cross_ae = ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]  # (M,)
        cross_ad = ao[None, :, 0] * d[:, None, 1] - ao[None, :, 1] * d[:, None, 0]  # (R, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross_ae[None, :] / denom
            u = cross_ad / denom
        valid = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, np.inf)
        return np.minimum(t.min(axis=1), sensor_range)


# ---------------------------------------------------------------- ready-made tracks
def _oval() -> Track:
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    return Track(np.stack([170 * np.cos(th), 90 * np.sin(th)], axis=1), width=12, name="oval")


def _circuit() -> Track:
    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    r = 150 * (1 + 0.28 * np.sin(2 * th + 0.6) + 0.18 * np.sin(3 * th + 1.9) + 0.07 * np.sin(5 * th))
    return Track(np.stack([r * np.cos(th), r * np.sin(th)], axis=1), width=12, name="circuit")


def _clover() -> Track:
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = 160 * (1 + 0.32 * np.cos(3 * th))
    return Track(np.stack([r * np.cos(th), r * np.sin(th)], axis=1), width=12, name="clover")


TRACKS = {"oval": _oval, "circuit": _circuit, "clover": _clover}


def create_track(spec) -> Track:
    """Accepts a name from TRACKS, a Track, or an array (K, 2) of control points."""
    if isinstance(spec, Track):
        return spec
    if isinstance(spec, str):
        if spec not in TRACKS:
            raise ValueError(f"Unknown track '{spec}'. Available: {sorted(TRACKS)}")
        return TRACKS[spec]()
    return Track(spec)
