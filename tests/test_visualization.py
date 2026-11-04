import numpy as np
import torch

from training.dqn.trainer import QNetwork, make_actions
from visualization.app import Simulation, Window
from visualization.circuits import random_circuit


class ConstantPolicy(torch.nn.Module):
    def forward(self, observation):
        return torch.zeros((observation.shape[0], 1))


def _small_policy():
    torch.manual_seed(7)
    actions = make_actions({"force_bins": 5, "steering_bins": 7})
    return QNetwork(10, len(actions), (16, 16)).eval(), actions


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


def test_decision_captures_one_pass_activations_and_q_values():
    model, actions = _small_policy()
    simulation = Simulation(model, actions, 10, seed=3, max_steps=20, capture_activations=True)
    try:
        simulation.step()
        decision = simulation.last_decision
        assert decision is not None
        assert decision.q_values.shape == (len(actions),)
        assert [values.size for _, values in decision.activations] == [10, 16, 16, len(actions)]
        assert decision.activations[0][0] == "Input observation"
        assert "post-ReLU" in decision.activations[1][0]
        assert decision.activations[-1][0] == "Output Q-values"

        with torch.inference_mode():
            x = torch.from_numpy(decision.observation).unsqueeze(0)
            hidden1 = model.layers[1](model.layers[0](x))
            hidden2 = model.layers[3](model.layers[2](hidden1))
            q_values = model.layers[4](hidden2)
        assert np.allclose(decision.activations[1][1], hidden1.numpy().ravel())
        assert np.allclose(decision.activations[2][1], hidden2.numpy().ravel())
        assert np.allclose(decision.q_values, q_values.numpy().ravel())
        assert decision.action_index == int(np.argmax(decision.q_values))
        assert np.array_equal(decision.action, actions[decision.action_index])
    finally:
        simulation.close()


def test_rendering_does_not_change_action_or_simulation_state():
    import matplotlib
    matplotlib.use("Agg", force=True)

    model, actions = _small_policy()
    rendered = Simulation(model, actions, 10, seed=11, max_steps=30)
    control = Simulation(model, actions, 10, seed=11, max_steps=30, capture_activations=False)
    window = Window(rendered, fps=20)
    try:
        assert rendered.capture_activations
        assert window.tick() is True
        control.step()
        assert rendered.last_decision is not None and control.last_decision is not None
        assert rendered.last_decision.action_index == control.last_decision.action_index
        assert np.array_equal(rendered.last_decision.q_values, control.last_decision.q_values)
        assert np.array_equal(rendered.env.state, control.env.state)
        assert np.array_equal(rendered.observation, control.observation)
        snapshot = rendered.activation_snapshot()
        assert snapshot["layers"]["Hidden 1 (post-ReLU)"]["size"] == 16
        assert len(snapshot["q_values"]) == len(actions)
    finally:
        window.on_close(None)
        window.plt.close(window.fig)
        control.close()
