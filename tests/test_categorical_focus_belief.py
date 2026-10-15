"""Dependency-free smoke tests for the categorical FOCUS belief backend."""

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ray.rllib.agents.focus_common import FocusResponsibilityEngine
from ray.rllib.agents.qplex_focus.qplex_policy import (
    CategoricalOccupancyModel,
    _extract_target_positions,
    build_occupancy_model,
)
from hmvfe_mate_d.focus_adapter import HMVFEFocusResponsibilityEngine


def _state(batch=1, time=4, n_agents=2, n_targets=2):
    state_dim = 13 + n_agents * 9 + n_targets * 14
    state = torch.zeros(batch, time, state_dim)
    future = state.clone()
    target_start = 13 + n_agents * 9
    for target in range(n_targets):
        start = target_start + target * 14
        future[..., start : start + 2] = 0.1
    return state, future


def test_persistence_initialization_nll_and_gradient():
    state, future = _state()
    model = build_occupancy_model(
        state.size(-1),
        2,
        2,
        {
            'belief_distribution': 'categorical_grid',
            'horizon': 3,
            'belief_hidden_dim': 16,
            'belief_grid_size': 5,
            'belief_grid_max_delta': 200.0,
            'belief_persistence_initialization': True,
        },
    )
    assert isinstance(model, CategoricalOccupancyModel)
    probability, mean, std, logits = model.distribution(state)
    persistence = _extract_target_positions(state, 2, 2).unsqueeze(2).expand_as(mean)
    torch.testing.assert_close(mean, persistence, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.ones_like(probability[..., 0])
    )
    assert torch.isfinite(std).all() and torch.isfinite(logits).all()

    losses, masks, _, _, diagnostics = model.nll(state, future)
    assert all(torch.isfinite(loss).all() for loss in losses)
    assert all(mask.dtype == torch.bool for mask in masks)
    assert all(torch.isfinite(x).all() for x in diagnostics['categorical_nll'])
    assert all(torch.isfinite(x).all() for x in diagnostics['position_loss'])
    torch.stack([loss.mean() for loss in losses]).sum().backward()
    assert model.net[-1].weight.grad is not None
    assert model.net[-1].weight.grad.abs().sum() > 0.0


def test_full_distribution_credit_not_collapsed_to_mean():
    state, _ = _state(batch=1, time=1, n_agents=1, n_targets=1)
    model = CategoricalOccupancyModel(
        state.size(-1),
        1,
        1,
        horizon=1,
        hidden_dim=8,
        grid_size=3,
        max_delta=100.0,
        persistence_initialization=True,
    )
    engine = FocusResponsibilityEngine(
        model,
        1,
        1,
        {
            'n_targets': 1,
            'horizon': 1,
            'use_action_selection': False,
            'belief_grid_chunk_size': 2,
        },
    )
    grid = model.displacement_grid
    centre = ((grid[:, 0] == 0.0) & (grid[:, 1] == 0.0)).nonzero()[0, 0]
    left = ((grid[:, 0] == -100.0) & (grid[:, 1] == 0.0)).nonzero()[0, 0]
    right = ((grid[:, 0] == 100.0) & (grid[:, 1] == 0.0)).nonzero()[0, 0]
    centre_mass = torch.zeros(1, 1, 1, 1, model.num_cells)
    split_mass = torch.zeros_like(centre_mass)
    centre_mass[..., centre] = 1.0
    split_mass[..., left] = 0.5
    split_mass[..., right] = 0.5
    centre_delta = torch.einsum('bthnk,kd->bthnd', centre_mass, grid)
    split_delta = torch.einsum('bthnk,kd->bthnd', split_mass, grid)
    torch.testing.assert_close(centre_delta, split_delta)

    current_x = _extract_target_positions(state, 1, 1)[..., 0]

    def nonlinear_visibility(points, *_args, **_kwargs):
        displaced = (points[..., 0] - current_x.unsqueeze(2).unsqueeze(-1)).abs()
        return (displaced > 50.0).to(points.dtype).unsqueeze(2)

    engine.estimator._point_visibility = nonlinear_visibility
    actions = torch.zeros(1, 1, 1, dtype=torch.long)
    centre_gain = engine.estimator._credit_from_categorical_grid(
        centre_mass, grid, state, actions, 1
    )
    split_gain = engine.estimator._credit_from_categorical_grid(
        split_mass, grid, state, actions, 1
    )
    torch.testing.assert_close(centre_gain, torch.zeros_like(centre_gain))
    torch.testing.assert_close(split_gain, torch.ones_like(split_gain))


def test_end_to_end_focus_output_and_checkpoint_roundtrip():
    state, future = _state()
    config = {
        'belief_distribution': 'categorical_grid',
        'belief_mode': 'learned',
        'belief_grid_size': 5,
        'belief_grid_max_delta': 200.0,
        'belief_hidden_dim': 16,
        'belief_persistence_initialization': True,
        'n_targets': 2,
        'horizon': 3,
        'use_action_selection': False,
        'belief_grid_chunk_size': 4,
        'min_credit_signal': 1e-6,
    }
    model = build_occupancy_model(state.size(-1), 2, 2, config)
    engine = FocusResponsibilityEngine(model, 2, 4, config)
    action = torch.zeros(1, state.size(1), 2, dtype=torch.long)
    valid = torch.ones_like(action, dtype=torch.float32)
    output = engine(state, action, valid_mask=valid, next_global_state=future)
    assert torch.isfinite(output.rho).all()
    assert torch.isfinite(output.gains).all()
    assert torch.isfinite(output.belief_loss)
    assert engine.last_belief_stats['focus_belief_distribution_categorical'] == 1.0
    output.belief_loss.backward()
    assert model.net[-1].weight.grad.abs().sum() > 0.0

    restored_model = build_occupancy_model(state.size(-1), 2, 2, config)
    restored = FocusResponsibilityEngine(restored_model, 2, 4, config)
    restored.load_state_dict(engine.state_dict())
    restored_output = restored(
        state,
        action,
        valid_mask=valid,
        next_global_state=future,
        update_belief_ema=False,
    )
    torch.testing.assert_close(restored_output.rho, output.rho)


def test_hmvfe_adapter_builds_categorical_backend():
    state, future = _state()
    engine = HMVFEFocusResponsibilityEngine(
        num_cameras=2,
        num_targets=2,
        state_dim=state.size(-1),
        config={
            'enabled': True,
            'use_env_params': False,
            'belief_mode': 'learned',
            'belief_distribution': 'categorical_grid',
            'belief_safeguard_mode': 'zero',
            'belief_hidden_dim': 16,
            'belief_grid_size': 5,
            'belief_grid_max_delta': 200.0,
            'belief_grid_chunk_size': 4,
            'horizon': 3,
        },
    )
    assert isinstance(engine.estimator.occupancy_model, CategoricalOccupancyModel)
    mask = torch.ones(1, state.size(1), 2)
    output = engine.compute(
        global_state=state,
        next_global_state=future,
        valid_mask=mask,
    )
    assert torch.isfinite(output.rho).all()
    assert torch.isfinite(output.belief_loss)


if __name__ == '__main__':
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            function()
            print(f'{name}: ok')
