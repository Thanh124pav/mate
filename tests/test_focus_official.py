import pytest
import torch

from ray.rllib.agents.focus_official import (
    GradientFloor,
    RunningFeatureNormalizer,
    action_credit_loss,
    carrier_credit_loss,
    subset_action_matrix,
)


def test_subset_actions_use_mate_target_bit_order():
    bits = subset_action_matrix(8, 3)
    assert bits.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]


def test_action_credit_prefers_unique_high_gain_target():
    pair_gains = torch.tensor([[[[0.0, 0.10, 0.35]]]])
    q_values = torch.zeros(1, 1, 1, 8, requires_grad=True)
    loss, action_gains = action_credit_loss(
        q_values, pair_gains, torch.ones(1, 1, dtype=torch.bool)
    )
    assert action_gains[0, 0, 0].argmax().item() == 3  # binary 011 -> targets 2+3
    loss.backward()
    # Adding target 3 must be encouraged more strongly than adding target 1.
    assert q_values.grad[0, 0, 0, 1] < q_values.grad[0, 0, 0, 4]


def test_carrier_credit_is_minimal_when_carrier_matches_rho():
    rho = torch.tensor([[[0.7, 0.2, 0.1]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    matched, probability = carrier_credit_loss(rho, rho.clone(), valid)
    uniform, _ = carrier_credit_loss(rho, torch.ones_like(rho), valid)
    assert torch.allclose(probability, rho)
    assert matched < uniform


def test_gradient_floor_raises_small_action_gradient_to_ten_percent():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    floor = GradientFloor(ratio=0.1, ema_decay=0.9)
    td_loss = parameter.square()  # gradient norm 2
    action_loss = 0.001 * parameter.square()  # gradient norm .002
    scaled, stats = floor.scale(td_loss, action_loss, [parameter])
    scaled_grad = torch.autograd.grad(scaled, parameter)[0].abs().item()
    assert stats.scale == pytest.approx(100.0, rel=1e-4)
    assert scaled_grad == pytest.approx(0.2, rel=1e-4)
    assert stats.active


def test_running_normalizer_updates_and_clips():
    normalizer = RunningFeatureNormalizer(2, clip=5.0)
    batch = torch.tensor([[[1.0, 10.0], [3.0, 14.0]]])
    normalized = normalizer(batch)
    assert normalizer.mean.tolist() == pytest.approx([2.0, 12.0])
    assert normalized.mean(dim=(0, 1)).tolist() == pytest.approx([0.0, 0.0])
    normalizer.eval()
    clipped = normalizer(torch.tensor([[[1000.0, -1000.0]]]))
    assert clipped.abs().max().item() == pytest.approx(5.0)
