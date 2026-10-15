from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ray.rllib.agents.focus_common import FocusResponsibilityEngine
from ray.rllib.agents.qplex_focus.qplex_policy import (
    LearnedOccupancyModel,
    resolve_focus_config,
)


def _state():
    n_agents, n_targets = 2, 2
    state_dim = 13 + n_agents * 9 + n_targets * 14
    state = torch.zeros(1, 4, state_dim)
    future = state.clone()
    target_start = 13 + n_agents * 9
    for target in range(n_targets):
        start = target_start + target * 14
        future[..., start : start + 2] = 0.1
    return state, future


def _engine(mode):
    state, _ = _state()
    config = resolve_focus_config(
        {
            'focus': {
                'use_env_params': False,
                'belief_safeguard_mode': mode,
                'n_agents': 2,
                'n_targets': 2,
                'horizon': 3,
                'integral_mode': 'sigma',
                'sigma_method': 'legacy5',
                'use_action_selection': False,
            }
        }
    )
    model = LearnedOccupancyModel(
        state.size(-1),
        2,
        2,
        horizon=3,
        hidden_dim=16,
        persistence_initialization=config['belief_persistence_initialization'],
    )
    return FocusResponsibilityEngine(model, 2, 4, config)


def _observe_like_policy_custom_loss(engine, state, future):
    losses, valid, _mean, _std, diagnostics = engine.estimator.occupancy_model.nll(
        state, future
    )
    weights = torch.ones(len(losses)) / len(losses)
    time_valid = torch.ones_like(valid[0], dtype=torch.bool)
    return engine.observe_belief_diagnostics(
        diagnostics, valid, weights, valid_mask=time_valid
    )


def test_zero_mode_does_not_run_ema():
    state, future = _state()
    engine = _engine('zero')
    _observe_like_policy_custom_loss(engine, state, future)
    assert engine.get_runtime_state()['belief_error_ema_updates'] == 0


def test_policy_rollout_reads_gate_and_learner_updates_once():
    state, future = _state()
    engine = _engine('zero_ema')
    actions = torch.zeros(1, 4, 2, dtype=torch.long)
    mask = torch.ones(1, 4)
    engine(
        state,
        actions,
        valid_mask=mask,
        next_global_state=future,
        update_belief_ema=False,
    )
    assert engine.get_runtime_state()['belief_error_ema_updates'] == 0

    _observe_like_policy_custom_loss(engine, state, future)
    assert engine.get_runtime_state()['belief_error_ema_updates'] == 1
    assert engine.get_runtime_state()['learner_update'] == 1
    # RLlib TorchPolicy.get_weights calls this exact chain on every entry.
    converted = {
        key: value.cpu().detach().numpy()
        for key, value in engine.state_dict().items()
    }
    assert converted

    restored = _engine('zero_ema')
    restored.load_state_dict(engine.state_dict())
    assert restored.get_runtime_state() == engine.get_runtime_state()
    restored(
        state,
        actions,
        valid_mask=mask,
        next_global_state=future,
        update_belief_ema=False,
    )
    assert restored.get_runtime_state()['learner_update'] == 1

    legacy_state = {
        key: value
        for key, value in engine.state_dict().items()
        if not key.endswith('_state')
    }
    legacy_restored = _engine('zero_ema')
    legacy_restored.load_state_dict(legacy_state, strict=True)
    assert legacy_restored.get_runtime_state()['belief_error_ema_updates'] == 0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name}: ok')
