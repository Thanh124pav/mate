from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmvfe_mate_d.focus_adapter import (
    HMVFEFocusAdapter,
    policy_loss_from_log_prob,
    weighted_actor_log_prob,
)
from hmvfe_mate_d.models import HMVFECoordinator


def test_camera_log_prob_decomposition_matches_joint_policy_output():
    torch.manual_seed(7)
    model = HMVFECoordinator(
        num_cameras=3,
        num_targets=4,
        table_size=32,
        num_fields=5,
        embedding_dim=4,
        num_experts=2,
        top_k=1,
        gating_hidden=8,
        mlp_hidden=8,
        mlp_layers=1,
        critic_reduction='learned',
        value_head_hidden=8,
    )
    obs = torch.randint(0, 32, (3, 4, 5)).float()
    _action, joint_log_prob, _entropy, _value, camera_log_prob = model.act(
        obs, deterministic=True, return_per_camera_log_prob=True
    )
    assert camera_log_prob.shape == (3,)
    torch.testing.assert_close(camera_log_prob.sum(dim=-1), joint_log_prob)


def test_focus_disabled_actor_path_matches_baseline_loss():
    camera_log_prob = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.25, -0.125]])
    joint_log_prob = camera_log_prob.sum(dim=-1)
    advantages = torch.tensor([0.7, -1.3])
    adapter = HMVFEFocusAdapter(eta=1.0)

    actor_log_prob, weights = weighted_actor_log_prob(joint_log_prob, camera_log_prob, adapter)

    assert weights is None
    torch.testing.assert_close(actor_log_prob, joint_log_prob)
    torch.testing.assert_close(
        policy_loss_from_log_prob(actor_log_prob, advantages),
        -(joint_log_prob * advantages).mean(),
    )


def test_eta_zero_recovers_unit_weights_and_joint_log_prob():
    camera_log_prob = torch.tensor([[-1.0, -2.0, -3.0]])
    joint_log_prob = camera_log_prob.sum(dim=-1)
    rho = torch.tensor([[0.7, 0.2, 0.1]])
    adapter = HMVFEFocusAdapter(eta=0.0)

    actor_log_prob, weights = weighted_actor_log_prob(
        joint_log_prob, camera_log_prob, adapter, rho=rho
    )

    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(actor_log_prob, joint_log_prob)


def test_uniform_responsibility_recovers_joint_log_prob():
    camera_log_prob = torch.tensor([[-1.0, -2.0, -3.0]])
    joint_log_prob = camera_log_prob.sum(dim=-1)
    rho = torch.full((1, 3), 1.0 / 3.0)
    adapter = HMVFEFocusAdapter(eta=1.0)

    actor_log_prob, weights = weighted_actor_log_prob(
        joint_log_prob, camera_log_prob, adapter, rho=rho
    )

    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(actor_log_prob, joint_log_prob)


def test_zero_confidence_recovers_joint_log_prob():
    camera_log_prob = torch.tensor([[-1.0, -2.0, -3.0]])
    joint_log_prob = camera_log_prob.sum(dim=-1)
    rho = torch.tensor([[0.7, 0.2, 0.1]])
    confidence = torch.zeros(1)
    adapter = HMVFEFocusAdapter(eta=1.0, use_confidence=True)

    actor_log_prob, weights = weighted_actor_log_prob(
        joint_log_prob, camera_log_prob, adapter, rho=rho, confidence=confidence
    )

    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(actor_log_prob, joint_log_prob)


def test_all_active_weight_mean_is_one():
    rho = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    adapter = HMVFEFocusAdapter(eta=0.75)

    weights = adapter.compute_weights(rho)

    torch.testing.assert_close(weights.mean(dim=-1), torch.ones(2))


def test_active_mask_normalizes_over_active_cameras_and_zeroes_inactive():
    rho = torch.tensor([[0.6, 0.3, 0.1]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    adapter = HMVFEFocusAdapter(eta=0.5)

    weights = adapter.compute_weights(rho, active_mask=mask)

    assert weights[0, 1].item() == 0.0
    active_weights = weights[mask.bool()]
    torch.testing.assert_close(active_weights.mean(), torch.tensor(1.0))


def test_actor_gradient_does_not_flow_into_rho():
    camera_log_prob = torch.nn.Parameter(torch.tensor([[-1.0, -2.0, -3.0]]))
    joint_log_prob = camera_log_prob.sum(dim=-1)
    rho = torch.tensor([[0.7, 0.2, 0.1]], requires_grad=True)
    adapter = HMVFEFocusAdapter(eta=1.0)

    actor_log_prob, _weights = weighted_actor_log_prob(
        joint_log_prob, camera_log_prob, adapter, rho=rho
    )
    loss = policy_loss_from_log_prob(actor_log_prob, torch.ones(1))
    loss.backward()

    assert camera_log_prob.grad is not None
    assert torch.isfinite(camera_log_prob.grad).all()
    assert rho.grad is None



def test_masked_policy_loss_uses_only_valid_focus_decisions():
    actor_log_prob = torch.tensor([10.0, 20.0, 30.0])
    advantages = torch.tensor([1.0, 1.0, 1000.0])
    mask = torch.tensor([True, True, False])

    loss = policy_loss_from_log_prob(actor_log_prob, advantages, mask=mask)

    torch.testing.assert_close(loss, torch.tensor(-15.0))


def test_masked_policy_loss_rejects_all_invalid_focus_decisions():
    actor_log_prob = torch.tensor([10.0, 20.0])
    advantages = torch.tensor([1.0, 1.0])
    mask = torch.zeros(2, dtype=torch.bool)

    try:
        policy_loss_from_log_prob(actor_log_prob, advantages, mask=mask)
    except RuntimeError as exc:
        assert 'no valid decisions' in str(exc)
    else:
        raise AssertionError('expected RuntimeError for an all-invalid FOCUS batch')

def test_non_uniform_weighting_changes_relative_camera_gradient():
    camera_terms = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
    joint_log_prob = camera_terms.sum(dim=-1)
    rho = torch.tensor([[0.75, 0.25]])
    adapter = HMVFEFocusAdapter(eta=1.0)

    actor_log_prob, weights = weighted_actor_log_prob(
        joint_log_prob, camera_terms, adapter, rho=rho
    )
    loss = policy_loss_from_log_prob(actor_log_prob, torch.ones(1))
    loss.backward()

    torch.testing.assert_close(camera_terms.grad.abs(), weights)
    assert camera_terms.grad[0, 0].abs() > camera_terms.grad[0, 1].abs()



def test_focus_weight_bounds_prevent_negative_or_oversized_weights():
    rho = torch.tensor([[0.99, 0.01, 0.0, 0.0]])
    adapter = HMVFEFocusAdapter(eta=2.0, weight_min=0.1, weight_max=3.0)

    weights = adapter.compute_weights(rho)

    assert weights.min().item() >= 0.1 - 1e-6
    assert weights.max().item() <= 3.0 + 1e-6
    torch.testing.assert_close(weights.mean(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6)


def test_focus_weight_bounds_preserve_active_mask_mean():
    rho = torch.tensor([[0.99, 0.01, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    adapter = HMVFEFocusAdapter(eta=2.0, weight_min=0.1, weight_max=3.0)

    weights = adapter.compute_weights(rho, active_mask=mask)

    assert weights[0, 2].item() == 0.0
    assert weights[0, 3].item() == 0.0
    active_weights = weights[mask.bool()]
    assert active_weights.min().item() >= 0.1 - 1e-6
    assert active_weights.max().item() <= 3.0 + 1e-6
    torch.testing.assert_close(active_weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

def test_fixed_seed_disabled_single_update_matches_baseline_formula():
    torch.manual_seed(13)
    model_focus_disabled = torch.nn.Linear(3, 2, bias=False)
    model_baseline = torch.nn.Linear(3, 2, bias=False)
    model_baseline.load_state_dict(model_focus_disabled.state_dict())
    opt_focus = torch.optim.SGD(model_focus_disabled.parameters(), lr=0.1)
    opt_base = torch.optim.SGD(model_baseline.parameters(), lr=0.1)
    x = torch.randn(5, 3)
    advantages = torch.randn(5)

    camera_focus = model_focus_disabled(x)
    joint_focus = camera_focus.sum(dim=-1)
    actor_log_prob, _ = weighted_actor_log_prob(
        joint_focus, camera_focus, HMVFEFocusAdapter(eta=1.0)
    )
    loss_focus = policy_loss_from_log_prob(actor_log_prob, advantages)
    opt_focus.zero_grad()
    loss_focus.backward()
    opt_focus.step()

    camera_base = model_baseline(x)
    joint_base = camera_base.sum(dim=-1)
    loss_base = -(joint_base * advantages.detach()).mean()
    opt_base.zero_grad()
    loss_base.backward()
    opt_base.step()

    for focus_param, base_param in zip(model_focus_disabled.parameters(), model_baseline.parameters()):
        torch.testing.assert_close(focus_param, base_param)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name}: ok')
