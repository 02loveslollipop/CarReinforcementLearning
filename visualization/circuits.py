"""Generate smooth closed circuits for the visualizer."""
from __future__ import annotations

import numpy as np

from car_rl.tracks import Track


def random_circuit(rng: np.random.Generator, number: int = 1) -> Track:
    """Create a varied, non-intersecting circuit with a 12-metre-wide road.

    Ordered radial control points avoid crossing the centre line. The Track
    constructor rejects turns whose inside edge would fold over; a coarse
    separation check also rejects narrow gaps between distant track sections.
    """
    for _ in range(200):
        count = int(rng.integers(18, 25))
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amplitudes = rng.uniform([0.05, 0.02, 0.0], [0.19, 0.12, 0.04])
        radius = rng.uniform(140.0, 180.0) * (
            1.0 + amplitudes[0] * np.sin(2 * theta + phase[0])
            + amplitudes[1] * np.sin(3 * theta + phase[1])
            + amplitudes[2] * np.sin(4 * theta + phase[2])
        )
        aspect = rng.uniform(0.8, 1.2)
        points = np.column_stack((radius * np.cos(theta), aspect * radius * np.sin(theta)))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        try:
            track = Track(points @ rotation.T, width=12.0, name=f"random-{number}")
        except ValueError:
            continue
        # Exclude nearby points along the loop; the rest must have room for
        # both road edges. This avoids visually overlapping separate sections.
        center = track.center[::5]
        distance = np.linalg.norm(center[:, None] - center[None, :], axis=2)
        index = np.arange(len(center))
        gap = np.abs(index[:, None] - index[None, :])
        gap = np.minimum(gap, len(center) - gap)
        if np.all(distance[gap > 8] > 2 * track.half_width):
            return track
    raise RuntimeError("Could not generate a valid circuit after 200 attempts")
