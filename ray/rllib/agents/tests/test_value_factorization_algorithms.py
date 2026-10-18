"""Focused mathematical and gradient tests for the new CTDE algorithms."""

import copy

import numpy as np
import pytest
import torch
from gym.spaces import Box, Dict, Discrete, Tuple

from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.agents.concaveq.concaveq_policy import ConcaveQTorchPolicy
from ray.rllib.agents.facmac.facmac_policy import FACMACTorchPolicy
from ray.rllib.agents.focus_common.responsibility_engine import (
    FocusResponsibilityEngine,
)
from ray.rllib.agents.qplex_focus.qplex_policy import (
    CategoricalOccupancyModel,
    LearnedOccupancyModel,
    _extract_target_positions,
    build_occupancy_model,
    resolve_focus_config,
)
from ray.rllib.agents.qfree.qfree_policy import QFreeTorchPolicy
from ray.rllib.agents.resq.resq_policy import ResQTorchPolicy
from ray.rllib.agents.riskq.riskq_policy import RiskQTorchPolicy
from ray.rllib.agents.value_factorization.components import (
    ConcaveMixer,
    MonotonicMixer,
    RecurrentAgent,
    coordinate_ascent_actions,
    forward_preserving_scale,
    quantile_huber_loss,
    risk_distortion_weights,
    td_lambda_targets,
    unroll_current_and_next,
)
from ray.rllib.agents.value_factorization.policy_base import GroupedSequenceBatch


def _belief_state(batch=1, time=4, n_agents=2, n_targets=2):
    state_dim = 13 + n_agents * 9 + n_targets * 14
    state = torch.zeros(batch, time, state_dim)
    next_state = state.clone()
    target_start = 13 + n_agents * 9
    for target in range(n_targets):
        start = target_start + target * 14
        next_state[..., start : start + 2] = 0.1
    return state, next_state


def _safe_focus_config(**overrides):
    config = {
        "belief_mode": "learned",
        "n_targets": 2,
        "horizon": 3,
        "horizon_discount": 0.7,
        "integral_mode": "sigma",
        "sigma_method": "gauss_hermite",
        "sigma_order": 3,
        "use_action_selection": False,
        "belief_progressive_blend": True,
        "belief_blend_warmup_updates": 20,
        "belief_blend_ramp_updates": 80,
        "belief_performance_gate": True,
        "belief_error_ema_decay": 0.0,
        "belief_gate_full_improvement": 0.2,
        "min_credit_signal": 1e-6,
        "eps": 1e-8,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "mode,ema_gate,progressive_blend",
    [
        ("zero", False, False),
        ("zero_ema", True, False),
        ("zero_ema_blend", True, True),
    ],
)
def test_belief_safeguard_mode_mapping(mode, ema_gate, progressive_blend):
    config = resolve_focus_config(
        {
            "focus": {
                "use_env_params": False,
                "belief_safeguard_mode": mode,
            }
        }
    )
    assert config["belief_persistence_initialization"] is True
    assert config["belief_performance_gate"] is ema_gate
    assert config["belief_progressive_blend"] is progressive_blend


def test_invalid_belief_safeguard_mode_is_rejected():
    with pytest.raises(ValueError, match="belief_safeguard_mode"):
        resolve_focus_config(
            {
                "focus": {
                    "use_env_params": False,
                    "belief_safeguard_mode": "not-a-mode",
                }
            }
        )


def test_belief_persistence_initialization_is_exact_and_trainable():
    state, next_state = _belief_state()
    model = LearnedOccupancyModel(
        state.size(-1),
        n_agents=2,
        n_targets=2,
        horizon=3,
        hidden_dim=16,
        persistence_initialization=True,
    )
    mean, _ = model(state)
    persistence = _extract_target_positions(state, 2, 2).unsqueeze(2).expand_as(mean)
    torch.testing.assert_close(mean, persistence, rtol=0.0, atol=0.0)

    losses, _, _, _, _ = model.nll(state, next_state)
    torch.stack([loss.mean() for loss in losses]).sum().backward()
    assert model.net[-1].weight.grad is not None
    assert model.net[-1].weight.grad.abs().sum() > 0.0


def test_categorical_belief_starts_at_persistence_and_is_trainable():
    state, next_state = _belief_state()
    model = build_occupancy_model(
        state.size(-1),
        n_agents=2,
        n_targets=2,
        focus_config={
            "belief_distribution": "categorical_grid",
            "horizon": 3,
            "belief_hidden_dim": 16,
            "belief_grid_size": 5,
            "belief_grid_max_delta": 200.0,
            "belief_persistence_initialization": True,
        },
    )
    assert isinstance(model, CategoricalOccupancyModel)
    probabilities, mean, std, logits = model.distribution(state)
    persistence = _extract_target_positions(state, 2, 2).unsqueeze(2).expand_as(mean)
    torch.testing.assert_close(mean, persistence, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )
    assert torch.isfinite(std).all()
    assert torch.isfinite(logits).all()

    losses, valid, _, _, diagnostics = model.nll(state, next_state)
    assert all(torch.isfinite(loss).all() for loss in losses)
    assert all(mask.dtype == torch.bool for mask in valid)
    assert all(torch.isfinite(loss).all() for loss in diagnostics["categorical_nll"])
    assert all(torch.isfinite(loss).all() for loss in diagnostics["position_loss"])
    torch.stack([loss.mean() for loss in losses]).sum().backward()
    assert model.net[-1].weight.grad is not None
    assert model.net[-1].weight.grad.abs().sum() > 0.0


def test_categorical_focus_integrates_distribution_not_only_expected_position():
    state, _ = _belief_state(batch=1, time=1, n_agents=1, n_targets=1)
    model = CategoricalOccupancyModel(
        state.size(-1),
        n_agents=1,
        n_targets=1,
        horizon=1,
        hidden_dim=8,
        grid_size=3,
        max_delta=100.0,
        persistence_initialization=True,
    )
    engine = FocusResponsibilityEngine(
        model,
        n_agents=1,
        n_actions=1,
        focus_config={
            "n_targets": 1,
            "horizon": 1,
            "use_action_selection": False,
            "belief_grid_chunk_size": 2,
        },
    )

    # Two distributions have the same zero expected displacement.  The first
    # is concentrated at the centre, while the second puts equal mass at the
    # left/right cells.  A nonlinear visibility function must distinguish them.
    grid = model.displacement_grid
    centre = ((grid[:, 0] == 0.0) & (grid[:, 1] == 0.0)).nonzero()[0, 0]
    left = ((grid[:, 0] == -100.0) & (grid[:, 1] == 0.0)).nonzero()[0, 0]
    right = ((grid[:, 0] == 100.0) & (grid[:, 1] == 0.0)).nonzero()[0, 0]
    centre_mass = torch.zeros(1, 1, 1, 1, model.num_cells)
    split_mass = torch.zeros_like(centre_mass)
    centre_mass[..., centre] = 1.0
    split_mass[..., left] = 0.5
    split_mass[..., right] = 0.5
    centre_delta = torch.einsum("bthnk,kd->bthnd", centre_mass, grid)
    split_delta = torch.einsum("bthnk,kd->bthnd", split_mass, grid)
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


def test_focus_blend_uses_prior_ema_and_keeps_raw_belief_gradient():
    state, next_state = _belief_state()
    model = LearnedOccupancyModel(
        state.size(-1),
        n_agents=2,
        n_targets=2,
        horizon=3,
        hidden_dim=16,
        persistence_initialization=True,
    )
    engine = FocusResponsibilityEngine(
        model,
        n_agents=2,
        n_actions=4,
        focus_config=_safe_focus_config(),
    )
    # Prior batches say the learned mean is substantially better than
    # persistence.  The current batch has equal raw/persistence errors because
    # the head starts at zero; it must not turn off its own gate retroactively.
    engine.set_runtime_state(
        {
            "learner_update": 0,
            "belief_model_error_ema": 50.0,
            "belief_baseline_error_ema": 100.0,
            "belief_error_ema_updates": 1,
        }
    )
    actions = torch.zeros(1, state.size(1), 2, dtype=torch.long)
    valid = torch.ones_like(actions, dtype=torch.float32)
    output = engine(
        state,
        actions,
        valid_mask=valid,
        next_global_state=next_state,
        learner_update=60,
    )
    stats = engine.last_belief_stats
    assert stats["focus_belief_blend_schedule"] == pytest.approx(0.5)
    assert stats["focus_belief_performance_gate"] == pytest.approx(1.0)
    assert stats["focus_belief_effective_blend"] == pytest.approx(0.5)
    assert stats["focus_belief_gate_ema_updates"] == pytest.approx(1.0)
    assert stats["focus_belief_gate_ema_updates_after"] == pytest.approx(2.0)

    output.belief_loss.backward()
    assert model.net[-1].weight.grad is not None
    assert model.net[-1].weight.grad.abs().sum() > 0.0

    # With decay=0, the just-seen equal-error batch becomes the prior for the
    # next call and correctly closes the performance gate.
    engine(
        state,
        actions,
        valid_mask=valid,
        next_global_state=next_state,
        learner_update=61,
    )
    assert engine.last_belief_stats["focus_belief_performance_gate"] == pytest.approx(0.0)


def test_focus_gate_runtime_state_roundtrips():
    state, _ = _belief_state()
    first = FocusResponsibilityEngine(
        LearnedOccupancyModel(
            state.size(-1), 2, 2, persistence_initialization=True
        ),
        n_agents=2,
        n_actions=4,
        focus_config=_safe_focus_config(),
    )
    expected = {
        "learner_update": 73,
        "belief_model_error_ema": 80.0,
        "belief_baseline_error_ema": 100.0,
        "belief_error_ema_updates": 12,
    }
    first.set_runtime_state(expected)
    second = FocusResponsibilityEngine(
        LearnedOccupancyModel(
            state.size(-1), 2, 2, persistence_initialization=True
        ),
        n_agents=2,
        n_actions=4,
        focus_config=_safe_focus_config(),
    )
    second.set_runtime_state(first.get_runtime_state())
    assert second.get_runtime_state() == expected


def test_forward_preserving_focus_scaling_changes_only_gradient():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    weights = torch.tensor([[0.25, 1.75]])
    output = forward_preserving_scale(x, weights)
    torch.testing.assert_close(output, x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, weights)


def test_td_lambda_terminal_and_padding():
    reward = torch.tensor([[[1.0], [2.0], [99.0]]])
    terminal = torch.tensor([[[0.0], [1.0], [0.0]]])
    mask = torch.tensor([[[1.0], [1.0], [0.0]]])
    bootstrap = torch.full_like(reward, 10.0)
    target = td_lambda_targets(reward, terminal, mask, bootstrap, 0.9, 0.6)
    assert target[0, 1, 0].item() == pytest.approx(2.0)
    assert target[0, 2, 0].item() == 0.0
    assert torch.isfinite(target).all()


def test_monotonic_mixer_has_nonnegative_input_derivatives():
    torch.manual_seed(0)
    mixer = MonotonicMixer(3, 7, embed_dim=8)
    utility = torch.randn(5, 3, requires_grad=True)
    state = torch.randn(5, 7)
    mixer(utility, state).sum().backward()
    assert utility.grad is not None
    assert (utility.grad >= -1e-7).all()


@pytest.mark.parametrize("kind,param", [("neutral", 0.0), ("wang", 0.75), ("cvar", 0.25), ("cpw", 0.71), ("var", 0.5)])
def test_risk_distortion_weights_are_finite_nonnegative_quadrature(kind, param):
    weights = risk_distortion_weights(32, kind, param)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()


def test_wang_distortion_matches_official_midpoint_derivative():
    tau = (torch.arange(32, dtype=torch.float32) + 0.5) / 32.0
    z = np.sqrt(2.0) * torch.erfinv(2.0 * tau - 1.0)
    expected = torch.exp(-0.75 * z - 0.5 * 0.75 ** 2).clamp(0.0, 5.0) / 32.0
    torch.testing.assert_close(risk_distortion_weights(32, "wang", 0.75), expected)


def test_current_recurrent_outputs_do_not_depend_on_next_observations():
    torch.manual_seed(11)
    model = RecurrentAgent(7, 5, 3, hidden_dim=11)
    observations = torch.randn(2, 4, 3, 7)
    actions = torch.randint(5, (2, 4, 3))
    current_a, _, _ = unroll_current_and_next(
        model, observations, torch.randn_like(observations), actions
    )
    current_b, _, _ = unroll_current_and_next(
        model, observations, torch.randn_like(observations), actions
    )
    torch.testing.assert_close(current_a, current_b, rtol=0.0, atol=0.0)


def test_quantile_huber_has_finite_nonzero_gradient():
    prediction = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 2.0, 4.0]]])
    loss = quantile_huber_loss(prediction, target, torch.ones(1, 1))
    loss.backward()
    assert loss.item() > 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_concave_mixer_satisfies_jensen_inequality():
    torch.manual_seed(2)
    mixer = ConcaveMixer(3, 5, hidden_dims=(8, 8, 8))
    state = torch.randn(16, 5)
    x = torch.randn(16, 3)
    y = torch.randn(16, 3)
    theta = 0.37
    lhs = mixer(theta * x + (1.0 - theta) * y, state)
    rhs = theta * mixer(x, state) + (1.0 - theta) * mixer(y, state)
    assert (lhs + 1e-5 >= rhs).all()


def test_coordinate_ascent_never_decreases_joint_value():
    torch.manual_seed(3)
    values = torch.randn(4, 3, 5)
    mask = torch.ones_like(values)

    def evaluate(actions):
        if actions.dim() == 2:
            chosen = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            expanded = values.unsqueeze(1).expand(-1, actions.size(1), -1, -1)
            chosen = expanded.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return (-(chosen - 0.3).square().sum(dim=-1)).unsqueeze(-1)

    initial = values.argmax(dim=-1)
    before = evaluate(initial)
    result, after = coordinate_ascent_actions(values, mask, evaluate, initial)
    chunked_result, chunked_after = coordinate_ascent_actions(
        values, mask, evaluate, initial, candidate_chunk_size=2
    )
    assert result.shape == initial.shape
    assert (after + 1e-7 >= before).all()
    assert torch.equal(chunked_result, result)
    assert torch.equal(chunked_after, after)


def _spaces(n_agents=2, obs_dim=11, state_dim=64, n_actions=5):
    agent = Dict(
        {
            "obs": Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32),
            ENV_STATE: Box(-1.0, 1.0, shape=(state_dim,), dtype=np.float32),
        }
    )
    observations = Tuple(tuple(copy.deepcopy(agent) for _ in range(n_agents)))
    setattr(observations, "original_space", copy.deepcopy(observations))
    actions = Tuple(tuple(Discrete(n_actions) for _ in range(n_agents)))
    return observations, actions


def _batch(policy, batch_size=2, time=4):
    torch.manual_seed(7)
    actions = torch.randint(
        policy.n_actions, (batch_size, time, policy.n_agents), device=policy.device
    )
    mask = torch.ones(batch_size, time, 1, device=policy.device)
    terminated = torch.zeros_like(mask)
    terminated[:, -1] = 1.0
    return GroupedSequenceBatch(
        rewards=torch.randn(batch_size, time, 1, device=policy.device),
        actions=actions,
        terminated=terminated,
        mask=mask,
        obs=torch.randn(
            batch_size, time, policy.n_agents, policy.obs_dim, device=policy.device
        ),
        next_obs=torch.randn(
            batch_size, time, policy.n_agents, policy.obs_dim, device=policy.device
        ),
        action_mask=torch.ones(
            batch_size,
            time,
            policy.n_agents,
            policy.n_actions,
            device=policy.device,
        ),
        next_action_mask=torch.ones(
            batch_size,
            time,
            policy.n_agents,
            policy.n_actions,
            device=policy.device,
        ),
        state=torch.randn(batch_size, time, policy.state_dim, device=policy.device),
        next_state=torch.randn(batch_size, time, policy.state_dim, device=policy.device),
        seq_lens=np.full(batch_size, time),
    )


@pytest.mark.parametrize(
    "policy_class,overrides",
    [
        (FACMACTorchPolicy, {}),
        (ResQTorchPolicy, {}),
        (RiskQTorchPolicy, {"n_quantiles": 8, "n_target_quantiles": 8}),
        (QFreeTorchPolicy, {}),
        (ConcaveQTorchPolicy, {}),
    ],
)
def test_baseline_policy_one_update_is_finite_and_checkpoint_roundtrips(
    policy_class, overrides
):
    observations, actions = _spaces()
    config = {
        **overrides,
        "num_gpus": 0,
        "model": {"lstm_cell_size": 16, "max_seq_len": 64},
        "focus": {"enabled": False},
    }
    policy = policy_class(observations, actions, config)
    stats = policy._learn_sequence_batch(_batch(policy))
    for key, value in stats.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), key
    state = policy.get_state()
    clone = policy_class(observations, actions, config)
    clone.set_state(state)
    for name, values in policy.get_weights().items():
        for key, value in values.items():
            np.testing.assert_allclose(value, clone.get_weights()[name][key])


def test_concaveq_policy_value_mixer_is_fitted_to_joint_value():
    """Repeated updates must not maximize q^pi's parameters without a target."""

    torch.manual_seed(0)
    observations, actions = _spaces()
    policy = ConcaveQTorchPolicy(
        observations,
        actions,
        {
            "num_gpus": 0,
            "model": {"lstm_cell_size": 16, "max_seq_len": 64},
            "focus": {"enabled": False},
            "policy_value_num_samples": 4,
        },
    )
    batch = _batch(policy)
    losses = []
    for _ in range(20):
        stats = policy._learn_sequence_batch(batch)
        losses.append(float(stats["policy_value_loss"]))
        for key, value in stats.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), key

    # The old objective drove q^pi directly toward +infinity and made the
    # total loss increasingly negative.  q^pi is now anchored to E_pi[Q_tot]
    # by a non-negative finite regression loss.
    assert min(losses) >= 0.0
    assert max(losses) < 1e8
