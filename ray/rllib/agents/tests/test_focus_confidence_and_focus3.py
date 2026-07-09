import math

import torch

from ray.rllib.agents.focus_utils import gated_focus_loss, resolve_confidence
from ray.rllib.agents.qplex_focus3.qplex_policy import (
    CellDiscreteBeliefModel,
    QPLEXFocus3Loss,
    _make_cell_grid,
)


def test_make_cell_grid_default_geometry():
    centers, bounds, subpoints = _make_cell_grid(grid_size=(10, 10), subpoints_per_cell=4)
    assert centers.shape == (100, 2)
    assert bounds.shape == (100, 4)
    assert subpoints.shape == (100, 4, 2)
    assert centers[:, 0].min() >= -1000.0
    assert centers[:, 0].max() <= 1000.0
    assert centers[:, 1].min() >= -1000.0
    assert centers[:, 1].max() <= 1000.0
    first_center = centers[0]
    expected_offsets = torch.tensor(
        [[-50.0, -50.0], [-50.0, 50.0], [50.0, -50.0], [50.0, 50.0]],
        dtype=centers.dtype,
    )
    assert torch.allclose(subpoints[0], first_center.unsqueeze(0) + expected_offsets)


def test_make_cell_grid_center_only():
    centers, _, subpoints = _make_cell_grid(grid_size=(2, 2), subpoints_per_cell=1)
    assert subpoints.shape == (4, 1, 2)
    assert torch.allclose(subpoints[:, 0, :], centers)


def test_gated_focus_loss_zero_and_one_confidence():
    per_step = torch.tensor([[1.0, 3.0]])
    valid = torch.tensor([[True, True]])
    reference = torch.tensor(0.0)
    zero = torch.zeros_like(per_step)
    one = torch.ones_like(per_step)
    assert gated_focus_loss(per_step, valid, zero, reference).item() == 0.0
    assert torch.allclose(gated_focus_loss(per_step, valid, one, reference), per_step.mean())


def test_entropy_and_loss_confidence_ordering():
    valid = torch.tensor([[True, True]])
    peaked = torch.tensor([[[[[0.98, 0.02]]], [[[0.50, 0.50]]]]])
    conf, mode = resolve_confidence(
        {"confidence_gate_enabled": True, "confidence_gate_mode": "entropy", "confidence_entropy_kappa": 2.0, "eps": 1e-8},
        valid,
        torch.ones(1, 2),
        default_mode="entropy",
        probs=peaked,
    )
    assert mode == "entropy"
    assert conf[0, 0] > conf[0, 1]

    losses = torch.tensor([[0.1, 10.0]])
    conf, mode = resolve_confidence(
        {"confidence_gate_enabled": True, "confidence_gate_mode": "loss", "confidence_loss_threshold": 1.0, "confidence_loss_temperature": 1.0, "eps": 1e-8},
        valid,
        torch.ones(1, 2),
        default_mode="loss",
        per_step_loss=losses,
    )
    assert mode == "loss"
    assert conf[0, 0] > conf[0, 1]


def test_focus3_belief_and_credit_shapes():
    n_agents = 2
    n_targets = 1
    state_dim = 13 + n_agents * 9 + n_targets * 14
    model = CellDiscreteBeliefModel(
        state_dim=(state_dim,),
        n_agents=n_agents,
        n_targets=n_targets,
        horizon=3,
        hidden_dim=8,
        grid_size=(4, 4),
        subpoints_per_cell=4,
    )
    state = torch.zeros(2, 5, state_dim)
    next_state = torch.zeros_like(state)
    mask = torch.ones(2, 5, n_agents)
    actions = torch.zeros(2, 5, n_agents, dtype=torch.long)
    logits = model(state)
    assert logits.shape == (2, 5, 3, n_targets, 16)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-4)

    loss = QPLEXFocus3Loss(
        None, None, None, None, n_agents, 2,
        focus_config={
            "enabled": True,
            "horizon": 3,
            "horizon_weights": "uniform",
            "cell_grid_size": [4, 4],
            "cell_subpoints_per_cell": 4,
            "cell_x_range": [-1000.0, 1000.0],
            "cell_y_range": [-1000.0, 1000.0],
            "min_credit_signal": 1e-6,
            "confidence_gate_enabled": True,
            "confidence_gate_mode": "auto",
            "eps": 1e-6,
        },
        belief_model=model,
    )
    rho, valid, total_g, belief_loss, belief_stats, confidence, confidence_mode = loss._focus_credit_target(
        state, next_state, actions, mask
    )
    assert rho.shape == (2, 5, n_agents)
    assert torch.allclose(rho.sum(dim=-1), torch.ones_like(rho[..., 0]), atol=1e-4)
    assert total_g.shape == (2, 5)
    assert confidence.shape == (2, 5)
    assert confidence_mode in {"entropy", "off", "loss"}
    assert torch.isfinite(belief_loss)
