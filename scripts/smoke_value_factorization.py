#!/usr/bin/env python3
"""Dependency-free component/policy smoke tests for the new algorithms."""

import argparse
import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete, Tuple

from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.agents.concaveq.concaveq_policy import (
    ConcaveQFocusTorchPolicy,
    ConcaveQTorchPolicy,
)
from ray.rllib.agents.facmac.facmac_policy import (
    FACMACFocusTorchPolicy,
    FACMACTorchPolicy,
)
from ray.rllib.agents.qfree.qfree_policy import QFreeFocusTorchPolicy, QFreeTorchPolicy
from ray.rllib.agents.resq.resq_policy import ResQFocusTorchPolicy, ResQTorchPolicy
from ray.rllib.agents.riskq.riskq_policy import RiskQFocusTorchPolicy, RiskQTorchPolicy
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


BASELINES = [
    ("FACMAC", FACMACTorchPolicy, {}),
    ("ResQ", ResQTorchPolicy, {}),
    ("RiskQ", RiskQTorchPolicy, {"n_quantiles": 8, "n_target_quantiles": 8}),
    ("QFree", QFreeTorchPolicy, {}),
    ("ConcaveQ", ConcaveQTorchPolicy, {}),
]

FOCUS_VARIANTS = [
    ("FACMAC+FOCUS", FACMACFocusTorchPolicy, {}),
    ("ResQ+FOCUS", ResQFocusTorchPolicy, {}),
    ("RiskQ+FOCUS", RiskQFocusTorchPolicy, {"n_quantiles": 8, "n_target_quantiles": 8}),
    ("QFree+FOCUS", QFreeFocusTorchPolicy, {}),
    ("ConcaveQ+FOCUS", ConcaveQFocusTorchPolicy, {}),
]


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def spaces(n_agents=2, obs_dim=11, state_dim=64, n_actions=5):
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


def synthetic_batch(policy, batch_size=2, time=4):
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
            batch_size, time, policy.n_agents, policy.n_actions, device=policy.device
        ),
        next_action_mask=torch.ones(
            batch_size, time, policy.n_agents, policy.n_actions, device=policy.device
        ),
        state=torch.empty(
            batch_size, time, policy.state_dim, device=policy.device
        ).uniform_(-0.8, 0.8),
        next_state=torch.empty(
            batch_size, time, policy.state_dim, device=policy.device
        ).uniform_(-0.8, 0.8),
        seq_lens=np.full(batch_size, time),
    )


def component_checks():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    weight = torch.tensor([[0.25, 1.75]])
    scaled = forward_preserving_scale(x, weight)
    require(torch.equal(scaled, x), "FOCUS scaling changed its forward value")
    scaled.sum().backward()
    require(torch.allclose(x.grad, weight), "FOCUS scaling did not reweight gradients")

    reward = torch.tensor([[[1.0], [2.0], [99.0]]])
    terminal = torch.tensor([[[0.0], [1.0], [0.0]]])
    mask = torch.tensor([[[1.0], [1.0], [0.0]]])
    target = td_lambda_targets(reward, terminal, mask, torch.full_like(reward, 10.0), 0.9, 0.6)
    require(abs(target[0, 1, 0].item() - 2.0) < 1e-6, "terminal bootstrapped")
    require(target[0, 2, 0].item() == 0.0, "padded target was not zero")

    monotonic = MonotonicMixer(3, 7, embed_dim=8)
    utility = torch.randn(5, 3, requires_grad=True)
    monotonic(utility, torch.randn(5, 7)).sum().backward()
    require((utility.grad >= -1e-7).all(), "QMIX derivative is negative")

    risk_weights = {}
    for kind, parameter in (
        ("neutral", 0.0), ("wang", 0.75), ("cvar", 0.25), ("cpw", 0.71), ("var", 0.5)
    ):
        risk_weight = risk_distortion_weights(32, kind, parameter)
        risk_weights[kind] = risk_weight
        require(torch.isfinite(risk_weight).all(), f"{kind} risk weights are non-finite")
        require((risk_weight >= 0).all(), f"{kind} risk weights are negative")
    require(
        torch.allclose(risk_weights["neutral"], torch.full((32,), 1.0 / 32.0)),
        "neutral RiskQ weights are not uniform quantile masses",
    )
    tau = (torch.arange(32, dtype=torch.float32) + 0.5) / 32.0
    z = math.sqrt(2.0) * torch.erfinv(2.0 * tau - 1.0)
    expected_wang = torch.exp(-0.75 * z - 0.5 * 0.75 ** 2).clamp(0.0, 5.0) / 32.0
    require(
        torch.allclose(risk_weights["wang"], expected_wang),
        "Wang weights differ from the official distortion derivative",
    )

    predicted = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
    quantile_huber_loss(
        predicted, torch.tensor([[[1.0, 2.0, 4.0]]]), torch.ones(1, 1)
    ).backward()
    require(predicted.grad.abs().sum() > 0, "quantile loss has no gradient")

    concave = ConcaveMixer(3, 5, hidden_dims=(8, 8, 8))
    state = torch.randn(16, 5)
    first, second, interpolation = torch.randn(16, 3), torch.randn(16, 3), 0.37
    lhs = concave(interpolation * first + (1.0 - interpolation) * second, state)
    rhs = interpolation * concave(first, state) + (1.0 - interpolation) * concave(second, state)
    require((lhs + 1e-5 >= rhs).all(), "concave mixer violates Jensen inequality")

    values = torch.randn(4, 3, 5)
    initial = values.argmax(dim=-1)

    def evaluate(actions):
        if actions.dim() == 2:
            chosen = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            expanded = values.unsqueeze(1).expand(-1, actions.size(1), -1, -1)
            chosen = expanded.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return (-(chosen - 0.3).square().sum(dim=-1)).unsqueeze(-1)

    before = evaluate(initial)
    result, after = coordinate_ascent_actions(
        values, torch.ones_like(values), evaluate, initial
    )
    chunked_result, chunked_after = coordinate_ascent_actions(
        values,
        torch.ones_like(values),
        evaluate,
        initial,
        candidate_chunk_size=2,
    )
    require((after + 1e-7 >= before).all(), "coordinate sweep decreased joint value")
    require(
        torch.equal(chunked_result, result) and torch.equal(chunked_after, after),
        "chunked coordinate sweep changed the exact optimizer result",
    )

    recurrent = RecurrentAgent(7, 5, 3, hidden_dim=11)
    observations = torch.randn(2, 4, 3, 7)
    actions = torch.randint(5, (2, 4, 3))
    current_a, _, _ = unroll_current_and_next(
        recurrent, observations, torch.randn_like(observations), actions
    )
    current_b, _, _ = unroll_current_and_next(
        recurrent, observations, torch.randn_like(observations), actions
    )
    require(
        torch.equal(current_a, current_b),
        "current recurrent outputs depend on future observations",
    )


def run_policy(name, policy_class, overrides, focus=False):
    # Full FOCUS geometry for 2 cameras and 8 targets needs 143 state fields.
    state_dim = 143 if focus else 64
    observation_space, action_space = spaces(state_dim=state_dim)
    config = {
        **overrides,
        "num_gpus": 0,
        "model": {"lstm_cell_size": 16, "max_seq_len": 64},
    }
    if focus:
        config["env_config"] = {
            "config": {
                "camera": {"location_random_range": [[0, 0]] * 2},
                "target": {"location_random_range": [[0, 0]] * 8},
            }
        }
        config["focus"] = {
            "enabled": True,
            "n_agents": 2,
            "n_targets": 8,
            "integral_mode": "sigma",
            "sigma_method": "legacy5",
            "belief_hidden_dim": 32,
            "horizon": 1,
            "use_action_selection": False,
        }
    else:
        config["focus"] = {"enabled": False}
    policy = policy_class(observation_space, action_space, config)
    stats = policy._learn_sequence_batch(
        synthetic_batch(policy, batch_size=1 if focus else 2, time=2 if focus else 4)
    )
    for key, value in stats.items():
        if torch.is_tensor(value):
            require(torch.isfinite(value).all(), f"{name}: non-finite statistic {key}")
        elif isinstance(value, (float, int)):
            require(math.isfinite(float(value)), f"{name}: non-finite statistic {key}")

    # Execution must depend only on local histories, even though centralized
    # state is present in the grouped training observation.
    local = np.random.randn(3, policy.n_agents, policy.obs_dim).astype(np.float32)
    available = np.ones(
        (3, policy.n_agents, policy.n_actions), dtype=np.float32
    )
    central = [np.zeros((3, policy.state_dim), dtype=np.float32)]

    def unpack_without_changing_local(_):
        return local, available, central[0]

    policy._unpack_observation = unpack_without_changing_local
    first_actions = policy.compute_actions(
        np.zeros((3, 1), dtype=np.float32), explore=False
    )[0]
    central[0] = np.random.randn(3, policy.state_dim).astype(np.float32)
    second_actions = policy.compute_actions(
        np.zeros((3, 1), dtype=np.float32), explore=False
    )[0]
    require(
        all(np.array_equal(a, b) for a, b in zip(first_actions, second_actions)),
        f"{name}: decentralized action changed with centralized state",
    )

    state = policy.get_state()
    with tempfile.TemporaryDirectory(prefix="mate-vf-smoke-") as directory:
        path = Path(directory) / "policy.pt"
        torch.save(state, path)
        restored_state = torch.load(path, map_location="cpu")
    clone = policy_class(observation_space, action_space, config)
    clone.set_state(restored_state)
    original = policy.get_weights()
    restored = clone.get_weights()
    for module_name, module_state in original.items():
        for parameter_name, value in module_state.items():
            require(
                np.allclose(
                    value,
                    restored[module_name][parameter_name],
                    equal_nan=True,
                ),
                f"{name}: checkpoint mismatch {module_name}.{parameter_name}",
            )
    print(f"PASS {name}: update, finite gradients/stats, checkpoint round-trip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus", action="store_true", help="also run all FOCUS variants")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    component_checks()
    print("PASS shared components")
    for item in BASELINES:
        run_policy(*item, focus=False)
    if args.focus:
        for item in FOCUS_VARIANTS:
            run_policy(*item, focus=True)


if __name__ == "__main__":
    main()
