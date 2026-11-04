import numpy as np
import torch

from dqn.trainer import QNetwork, make_actions, train, load_checkpoint
from dqn.evaluate import _build_q_network


def test_checkpoint_can_continue_and_evaluator_matches_policy(tmp_path):
    config = {'training': {'seed': 9, 'replay_capacity': 100, 'batch_size': 8,
                           'learning_starts': 8, 'hidden_sizes': [16, 16],
                           'checkpoint_every': 0}}
    first = train(config, tmp_path / 'first', 20)
    resumed = train({}, tmp_path / 'resumed', 10, parent=first)
    checkpoint = load_checkpoint(resumed)
    assert checkpoint['global_step'] == 30
    assert checkpoint['replay']['size'] == 30
    online = QNetwork(10, len(checkpoint['actions']), (16, 16))
    online.load_state_dict(checkpoint['online_state_dict'])
    independent = _build_q_network(checkpoint['online_state_dict'], 10, len(checkpoint['actions']))
    inputs = torch.randn(4, 10)
    assert torch.allclose(online(inputs), independent(inputs))
    assert np.array_equal(checkpoint['actions'], make_actions(config['training']))
