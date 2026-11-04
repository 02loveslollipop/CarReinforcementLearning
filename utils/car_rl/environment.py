"""Gymnasium environment: a car with basic dynamics that must complete one lap of a track."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import dynamics
from .dynamics import CarParameters
from .tracks import Track, create_track

SENSOR_ANGLES = np.deg2rad([-65.0, -20.0, -10.0, 0.0, 10.0, 20.0, 65.0])


class CarTrackEnv(gym.Env):
    """
    Action (Box, 2): [force, steering], both in [-1, 1].
        force   is rescaled before being passed to the dynamics so that force=0 equals
            a base traction of `base_acceleration` (0.5 by default): the car does not stay
            still if the agent does nothing. force=1 requests maximum traction, force=-1 brakes
            fully; intermediate positive values increase traction above the base and
            negative values reduce it until it turns into braking.
        steering  target steering angle (the actual angle follows it with a rate limit).

    Observation (Box, 10): distance from each sensor to the track edge, normalized
        by `sensor_range` (0 = edge touching, 1 = nothing within range). With
        `observe_dynamics=True` (default), normalized u, v, r are appended (3 more values);
        with False only the rays.

    Internal state (`env.unwrapped.state`): [x, y, psi, u, v, r, delta] + position on the
        track (`info["s"]`, `info["lateral"]`).

    Reward per step:
        speed_weight * (u / reference_speed) * dt   (u = longitudinal speed)
        With the default values (speed_weight=1), going exactly at reference_speed gives +1 per
        second. There is no penalty for crashing: the episode simply ends there, so the
        reward stops growing.

    terminated: collision (any corner outside the roadway) or completed lap.
    truncated:  `max_steps` reached or no improvement of the best progress for `time_without_progress` s.

    `track` can be a name (see tracks.TRACKS), a Track, an array (K, 2) of control
    points, or a list of them: on each reset one is chosen at random.

    `initial_speed` (m/s) is the longitudinal speed at which the car starts on each reset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        track="circuit",
        render_mode: str | None = None,
        car: CarParameters | None = None,
        dt: float = 0.05,
        substeps: int = 5,
        sensor_range: float = 60.0,
        max_steps: int = 3000,
        time_without_progress: float = 10.0,
        random_start: bool = False,
        observe_dynamics: bool = True,
        initial_speed: float = 5.0,
        base_acceleration: float = 0.5,
        reference_speed: float = 20.0,
        speed_weight: float = 1.0,
    ):
        super().__init__()
        specs = track if isinstance(track, (list, tuple)) and not _is_points(track) else [track]
        self.tracks: list[Track] = [create_track(s) for s in specs]
        self.track: Track = self.tracks[0]
        self.car = car or CarParameters()
        self.dt, self.substeps = dt, substeps
        self.sensor_range = sensor_range
        self.max_steps = max_steps
        self.max_no_improvement = int(round(time_without_progress / dt))
        self.random_start = random_start
        self.observe_dynamics = observe_dynamics
        self.initial_speed = initial_speed
        self.base_acceleration = base_acceleration
        self.reference_speed = reference_speed
        self.speed_weight = speed_weight
        self.render_mode = render_mode
        self._viewer = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n_obs = len(SENSOR_ANGLES) + (3 if observe_dynamics else 0)
        low = np.concatenate([np.zeros(len(SENSOR_ANGLES)), -np.ones(n_obs - len(SENSOR_ANGLES))])
        self.observation_space = spaces.Box(low.astype(np.float32), np.ones(n_obs, np.float32), dtype=np.float32)

        self.state = np.zeros(7)
        self._rays = np.zeros(len(SENSOR_ANGLES))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if len(self.tracks) > 1:
            self.track = self.tracks[int(self.np_random.integers(len(self.tracks)))]
        s0 = float(self.np_random.uniform(0, self.track.length)) if self.random_start else 0.0
        if options and "s0" in options:
            s0 = float(options["s0"])
        pos, heading = self.track.start_point(s0)
        self.state = np.array([pos[0], pos[1], heading, self.initial_speed, 0.0, 0.0, 0.0])
        self._prev_s, _ = self.track.project(pos)
        self.progress = 0.0
        self.best_progress = 0.0
        self.steps = 0
        self.steps_without_improvement = 0
        return self._observe(), self._info(False, False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        force = _rescale_force(float(action[0]), self.base_acceleration)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.state = dynamics.step(self.state, force, action[1], self.car, h)
        self.steps += 1

        s, lateral = self.track.project(self.state[:2])
        ds = (s - self._prev_s + self.track.length / 2) % self.track.length - self.track.length / 2
        self._prev_s = s
        self.progress += ds

        collision = self._has_collision()
        lap = (not collision) and self.progress >= self.track.length

        reward = self.speed_weight * self.state[3] / self.reference_speed * self.dt

        if self.progress > self.best_progress + 1e-6:
            self.best_progress, self.steps_without_improvement = self.progress, 0
        else:
            self.steps_without_improvement += 1
        terminated = collision or lap
        stagnated = self.steps_without_improvement >= self.max_no_improvement
        truncated = (not terminated) and (self.steps >= self.max_steps or stagnated)

        obs = self._observe()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, self._info(collision, lap, s, lateral)

    def render(self):
        if self.render_mode is None:
            return None
        from .render import Viewer

        if self._viewer is None or self._viewer.track is not self.track:
            if self._viewer is not None:
                self._viewer.close()
            self._viewer = Viewer(self.track, self.car, self.render_mode, self.metadata["render_fps"])
        return self._viewer.draw(self.state, self._rays, SENSOR_ANGLES)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------ helpers
    def _observe(self) -> np.ndarray:
        self._rays = self.track.ray_distances(self.state[:2], self.state[2], SENSOR_ANGLES, self.sensor_range)
        obs = self._rays / self.sensor_range
        if self.observe_dynamics:
            u, v, r = self.state[3:6]
            obs = np.concatenate([obs, np.clip([u / 40.0, v / 10.0, r / 2.0], -1, 1)])
        return obs.astype(np.float32)

    def _has_collision(self) -> bool:
        for q in dynamics.corners(self.state, self.car):
            if abs(self.track.project(q)[1]) > self.track.half_width:
                return True
        return False

    def _info(self, collision, lap, s=None, lateral=None) -> dict:
        if s is None:
            s, lateral = self.track.project(self.state[:2])
        u, v = self.state[3], self.state[4]
        return {
            "s": s,
            "lateral": lateral,
            "progress": self.progress / self.track.length,
            "speed": math.hypot(u, v),
            "slip": math.atan2(v, max(u, 1.0)),
            "time": self.steps * self.dt,
            "collision": collision,
            "lap_completed": lap,
            "lap_time": self.steps * self.dt if lap else None,
            "track": self.track.name,
        }


def _rescale_force(action: float, base: float) -> float:
    """Rescales the action in [-1, 1] to the real force (also in [-1, 1]) received by the
    dynamics, so that action=0 equals a base traction of `base` instead of 0."""
    if action >= 0.0:
        return base + (1.0 - base) * action
    return base + (base + 1.0) * action


def _is_points(x) -> bool:
    try:
        a = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return False
    return a.ndim == 2 and a.shape[1] == 2
