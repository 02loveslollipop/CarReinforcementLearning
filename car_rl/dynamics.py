"""Dynamic bicycle model with saturated tires (friction circle).

State: [x, y, psi, u, v, r, delta]
  x, y   world position (m)
  psi    heading (rad)
  u, v   longitudinal and lateral speed in the car frame (m/s)
  r      yaw rate (rad/s)
  delta  actual front wheel angle (rad)

When more force is requested than the tire can provide (a fast corner, full
acceleration or braking in a curve) the lateral force saturates and the car slips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class CarParameters:
    mass: float = 1200.0
    inertia: float = 2500.0
    a: float = 1.3  # CG -> front axle
    b: float = 1.3  # CG -> rear axle
    length: float = 4.2
    width: float = 1.8
    cornering_stiffness: float = 90000.0  # N/rad per axle
    mu: float = 1.0
    engine_force: float = 6000.0  # N, rear-wheel traction
    brake_force: float = 14000.0  # N, 60% front / 40% rear
    front_brake_bias: float = 0.6
    c_air: float = 2.5  # N/(m/s)^2
    c_rolling: float = 0.015
    max_steering: float = 0.5  # rad
    steering_rate: float = 1.5  # rad/s


def _sat(requested_force: float, max_force: float) -> float:
    return max_force * math.tanh(requested_force / max_force) if max_force > 1e-6 else 0.0


def step(state: np.ndarray, force: float, steering: float, p: CarParameters, dt: float) -> np.ndarray:
    """Advance dt seconds. force in [-1, 1] (+ traction, - brake); steering in [-1, 1]."""
    x, y, psi, u, v, r, delta = (float(e) for e in state)
    fz = p.mass * G * 0.5  # a == b -> 50/50 split (no load transfer)
    target = float(np.clip(steering, -1.0, 1.0)) * p.max_steering
    delta += float(np.clip(target - delta, -p.steering_rate * dt, p.steering_rate * dt))

    cmd = float(np.clip(force, -1.0, 1.0))
    if cmd >= 0.0:
        fx_front, fx_rear = 0.0, cmd * p.engine_force
    else:
        fb = -cmd * p.brake_force * math.tanh(u / 0.5)
        fx_front, fx_rear = -fb * p.front_brake_bias, -fb * (1 - p.front_brake_bias)
    lim = p.mu * fz
    fx_front, fx_rear = max(-lim, min(lim, fx_front)), max(-lim, min(lim, fx_rear))

    u_eff = max(u, 3.0)  # avoids the slip-angle singularity at low speed
    alpha_f = math.atan2(v + p.a * r, u_eff) - delta
    alpha_r = math.atan2(v - p.b * r, u_eff)
    fy_front = -_sat(p.cornering_stiffness * alpha_f, math.sqrt(max(lim**2 - fx_front**2, 0.0)))
    fy_rear = -_sat(p.cornering_stiffness * alpha_r, math.sqrt(max(lim**2 - fx_rear**2, 0.0)))

    resist = p.c_air * u * abs(u) + p.c_rolling * p.mass * G * math.tanh(u / 0.5)
    cd, sd = math.cos(delta), math.sin(delta)
    du = ((fx_front * cd - fy_front * sd) + fx_rear - resist) / p.mass + v * r
    dv = ((fy_front * cd + fx_front * sd) + fy_rear) / p.mass - u * r
    dr = (p.a * (fy_front * cd + fx_front * sd) - p.b * fy_rear) / p.inertia

    u += du * dt
    v += dv * dt
    r += dr * dt
    if u < 0.0:  # no reverse gear
        u = 0.0
    psi += r * dt
    x += (u * math.cos(psi) - v * math.sin(psi)) * dt
    y += (u * math.sin(psi) + v * math.cos(psi)) * dt
    return np.array([x, y, psi, u, v, r, delta])


def corners(state: np.ndarray, p: CarParameters) -> np.ndarray:
    """The 4 corners of the car (4, 2) in the world."""
    x, y, psi = state[0], state[1], state[2]
    local = np.array(
        [[p.length / 2, p.width / 2], [p.length / 2, -p.width / 2], [-p.length / 2, -p.width / 2], [-p.length / 2, p.width / 2]]
    )
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], axis=1)
