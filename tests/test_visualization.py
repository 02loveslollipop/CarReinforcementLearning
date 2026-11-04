import numpy as np
import torch

from visualization.app import Simulation, Window
from visualization.circuits import random_circuit


class ConstantPolicy(torch.nn.Module):
    def forward(self, observation):
        return torch.zeros((observation.shape[0], 1))


def test_random_circuits_are_reproducible_and_distinct():
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)
    first = [random_circuit(first_rng, i) for i in range(3)]
    second = [random_circuit(second_rng, i) for i in range(3)]
    for a, b in zip(first, second):
        assert np.array_equal(a.center, b.center)
        assert np.isfinite(a.center).all()
        assert a.min_radius > a.half_width * 1.05
    assert not np.array_equal(first[0].center, first[1].center)


def test_episode_transition_reuses_window_and_stops_on_close():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backend_bases import CloseEvent

    simulation = Simulation(ConstantPolicy(), np.array([[-1.0, 0.0]], dtype=np.float32), 10, seed=1, max_steps=1)
    window = Window(simulation, fps=20)
    try:
        figure = window.fig
        first_track = simulation.env.track.center.copy()
        assert window.tick() is True
        assert simulation.finished and simulation.episode == 1
        assert window.tick() is True
        assert simulation.finished and simulation.episode == 2
        assert window.fig is figure
        assert not np.array_equal(first_track, simulation.env.track.center)
        figure.canvas.callbacks.process("close_event", CloseEvent("close_event", figure.canvas))
        assert window.closed
        assert window.tick() is False
    finally:
        window.on_close(None)
        window.plt.close(window.fig)
