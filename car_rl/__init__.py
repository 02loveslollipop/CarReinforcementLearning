from gymnasium.envs.registration import register

from .dynamics import CarParameters
from .environment import SENSOR_ANGLES, CarTrackEnv
from .tracks import TRACKS, Track, create_track

register(id="CarTrack-v0", entry_point="car_rl.environment:CarTrackEnv")

__all__ = ["CarTrackEnv", "CarParameters", "Track", "TRACKS", "create_track", "SENSOR_ANGLES"]
