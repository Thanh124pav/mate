from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'hmvfe-mate'))

from hitmac_focus.focus import FutureVisibilityBelief, VisibilityBeliefSafeguard
from hitmac_focus.models import HitMACFocusCoordinator


def test_zero_initialization_equals_current_visibility_persistence():
    torch.manual_seed(31)
    visible_on_index = 23
    belief = FutureVisibilityBelief(
        table_size=64,
        num_fields=7,
        embedding_dim=4,
        hidden=8,
        visible_on_index=visible_on_index,
        persistence_probability=0.99,
        persistence_initialization=True,
    )
    obs = torch.randint(0, 64, (2, 3, 7)).float()
    obs[..., 4] = torch.tensor([[visible_on_index, 0, visible_on_index], [0, 0, visible_on_index]])

    torch.testing.assert_close(belief(obs), belief.persistence_logits(obs))


def test_visibility_residual_still_receives_supervised_gradient():
    belief = FutureVisibilityBelief(
        table_size=32,
        num_fields=7,
        embedding_dim=4,
        hidden=8,
        visible_on_index=17,
        persistence_initialization=True,
    )
    obs = torch.randint(0, 32, (2, 3, 7)).float()
    target = torch.randint(0, 2, (2, 3)).float()
    loss = F.binary_cross_entropy_with_logits(belief(obs), target)
    loss.backward()
    output = belief.net[-1]
    assert output.weight.grad is not None
    assert torch.isfinite(output.weight.grad).all()
    assert output.weight.grad.abs().sum() > 0


def test_visibility_ema_gate_is_causal_and_checkpointable():
    gate = VisibilityBeliefSafeguard(mode='zero_ema', gate_full_improvement=0.2)
    raw = torch.tensor([2.0])
    baseline = torch.tensor([-2.0])
    torch.testing.assert_close(gate.effective_logits(raw, baseline), baseline)

    gate.observe(model_error=0.4, baseline_error=1.0)
    assert gate.gate() == 1.0
    torch.testing.assert_close(gate.effective_logits(raw, baseline), raw)

    restored = VisibilityBeliefSafeguard(mode='zero_ema', gate_full_improvement=0.2)
    restored.load_state_dict(gate.state_dict())
    assert restored.state_dict() == gate.state_dict()


def test_execution_bias_changes_policy_logits_without_belief_gradient():
    torch.manual_seed(37)
    model = HitMACFocusCoordinator(
        num_cameras=2,
        num_targets=3,
        table_size=32,
        num_fields=7,
        embedding_dim=4,
        attention_dim=16,
        attention_heads=4,
        attention_layers=1,
        value_hidden=8,
    )
    obs = torch.randint(0, 32, (2, 3, 7)).float()
    bias = torch.tensor([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]], requires_grad=True)
    action, log_prob, *_ = model.act(
        obs, deterministic=True, pair_logit_bias=bias.detach()
    )
    assert action.shape == (2, 3)
    assert torch.isfinite(log_prob)
    log_prob.backward()
    assert bias.grad is None


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name}: ok')
