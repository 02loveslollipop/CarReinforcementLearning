import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from utils.car_rl import CarTrackEnv, TRACKS
from examples.heuristic_driver import driver


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_check_env(track):
    check_env(CarTrackEnv(track=track), skip_render_check=True)


def test_observation_and_internal_state():
    env = CarTrackEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == (10,) and obs.dtype == np.float32
    assert env.unwrapped.state.shape == (7,)
    assert {"s", "lateral", "progress", "speed"} <= set(info)


def test_crashes_at_full_speed_on_straight():
    env = CarTrackEnv(track="oval")
    env.reset(seed=0)
    for _ in range(600):
        _, r, term, trunc, info = env.step([1.0, 0.0])
        if term or trunc:
            break
    assert term and info["collision"] and r > 0  # crashing ends the episode, it does not penalize the reward


def test_slip_in_fast_corner():
    env = CarTrackEnv(track="oval")
    env.reset(seed=0, options={"s0": 0})
    slip = 0.0
    for _ in range(400):
        _, _, term, trunc, info = env.step([1.0, 0.3])
        slip = max(slip, abs(info["slip"]))
        if term or trunc:
            break
    assert slip > 0.15


def test_full_lap_with_driver_and_positive_reward():
    env = CarTrackEnv(track="oval")
    obs, _ = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(driver(obs))
        total += r
        done = term or trunc
    assert info["lap_completed"] and info["lap_time"] > 0 and total > 0


def test_truncation_when_standing_still():
    env = CarTrackEnv(time_without_progress=2.0)
    env.reset(seed=0)
    for _ in range(100):
        _, _, term, trunc, _ = env.step([-1.0, 0.0])  # brakes: starts with initial_speed
        if trunc:
            break
    assert trunc and not term


def test_multiple_tracks_and_random_start():
    env = CarTrackEnv(track=["oval", "circuit", "clover"], random_start=True)
    seen = {env.reset(seed=i)[1]["track"] for i in range(30)}
    assert len(seen) == 3


def test_speed_in_observation():
    env = CarTrackEnv()
    env.reset(seed=0)
    for _ in range(40):
        obs, *_ = env.step([1.0, 0.0])
    assert obs[7] > 0.05  # normalized u grows when accelerating
    assert CarTrackEnv(observe_dynamics=False).reset(seed=0)[0].shape == (7,)
