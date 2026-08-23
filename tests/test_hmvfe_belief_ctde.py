"""Focused tests for HMVFE's optional belief-state CTDE path."""

import torch

from hmvfe_mate_d.models import HMVFECoordinator


def _model(enabled=True):
    return HMVFECoordinator(
        num_cameras=2,
        num_targets=3,
        table_size=32,
        num_fields=5,
        embedding_dim=4,
        num_experts=2,
        top_k=1,
        gating_hidden=8,
        mlp_hidden=8,
        mlp_layers=1,
        critic_reduction="learned",
        value_head_hidden=8,
        global_state_dim=7,
        belief_enabled=enabled,
        belief_hidden_dim=8,
    )


def test_belief_actor_does_not_read_privileged_critic_state():
    torch.manual_seed(3)
    model = _model(enabled=True).eval()
    obs = torch.randint(0, 32, (2, 3, 5))
    state_a = torch.full((7,), -0.5)
    state_b = torch.full((7,), 0.5)

    probs_a, _ = model(obs, critic_state=state_a)
    probs_b, _ = model(obs, critic_state=state_b)

    # True state changes only the centralized critic input; actor probabilities
    # must remain a function of the local observation and predicted belief.
    assert torch.allclose(probs_a, probs_b, atol=1e-7, rtol=0.0)


def test_belief_loss_backpropagates_to_belief_head():
    torch.manual_seed(4)
    model = _model(enabled=True)
    obs = torch.randint(0, 32, (2, 3, 5))
    prediction = model.belief_from_observation(obs)
    target = torch.zeros_like(prediction)
    loss = model.belief_loss(target, prediction=prediction)
    loss.backward()

    gradients = [p.grad for p in model.belief_head.parameters() if p.requires_grad]
    assert gradients
    assert any(g is not None and torch.isfinite(g).all() for g in gradients)


def test_disabled_belief_preserves_native_model_shape():
    model = _model(enabled=False)
    obs = torch.randint(0, 32, (2, 3, 5))
    probs, value = model(obs)
    assert probs.shape == (2, 3)
    assert value.shape == (1,)
    assert model.belief_head is None
    assert model.belief_actor_head is None
