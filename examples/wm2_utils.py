import numpy as np
from gym import spaces
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.agents.qplex_wm2.world_model_v2 import LatentWorldModel


torch, nn = try_import_torch()


def discrete_action_count(action_space):
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, spaces.MultiDiscrete):
        return int(action_space.nvec.reshape(-1)[0])
    if isinstance(action_space, spaces.MultiBinary):
        return 2
    if isinstance(action_space, spaces.Dict) and 'action' in action_space.spaces:
        return discrete_action_count(action_space.spaces['action'])
    if isinstance(action_space, spaces.Tuple) and action_space.spaces:
        return discrete_action_count(action_space.spaces[0])
    return None


def world_model_feature_dim(config):
    if not config.get('enabled', True):
        return 0
    return int(config.get('stoch_dim', 32)) + int(config.get('deter_dim', 128))


def make_local_world_model(local_obs_dim, state_dim, n_actions, config):
    if not config.get('enabled', True):
        return None
    return LatentWorldModel(
        obs_size=local_obs_dim,
        state_dim=state_dim,
        n_agents=1,
        n_actions=n_actions,
        stoch_dim=config.get('stoch_dim', 32),
        deter_dim=config.get('deter_dim', 128),
        hidden_dim=config.get('hidden_dim', 128),
        action_embed_dim=config.get('action_embed_dim', 16),
        embed_dim=config.get('embed_dim', 128),
        imagination_horizon=config.get('imagination_horizon', 5),
        kl_coeff=config.get('kl_coeff', 1.0),
        free_nats=config.get('free_nats', 1.0),
    )


def encode_local_obs(world_model, local_obs):
    if world_model is None:
        return local_obs.new_empty((*local_obs.shape[:-1], 0))
    with torch.no_grad():
        leading = local_obs.shape[:-1]
        obs = local_obs.reshape(-1, 1, local_obs.shape[-1])
        feature = world_model.encode_obs(obs)
        return feature.reshape(*leading, feature.shape[-1])


def _primary_discrete_actions(actions):
    if isinstance(actions, dict):
        actions = actions.get('action')
    if actions is None:
        return None
    if actions.ndim > 1:
        actions = actions.reshape(actions.shape[0], -1)[:, 0]
    return actions.long()


def wm2_auxiliary_loss(
    world_model,
    loss_inputs,
    reference_loss,
    flat_obs_dim,
    local_obs_slice,
    local_obs_dim,
    state_slice,
    state_dim,
    n_actions,
    coeff=0.5,
):
    if world_model is None or coeff <= 0.0:
        return torch.zeros_like(reference_loss), {}
    if SampleBatch.CUR_OBS not in loss_inputs or SampleBatch.ACTIONS not in loss_inputs:
        return torch.zeros_like(reference_loss), {}

    obs = loss_inputs[SampleBatch.CUR_OBS].float()
    if obs.ndim != 2 or obs.size(-1) != flat_obs_dim:
        return torch.zeros_like(reference_loss), {}

    actions = _primary_discrete_actions(loss_inputs[SampleBatch.ACTIONS])
    if actions is None:
        return torch.zeros_like(reference_loss), {}
    actions = actions.to(obs.device).clamp(min=0, max=n_actions - 1)

    rewards = loss_inputs.get(SampleBatch.REWARDS)
    if rewards is None:
        rewards = obs.new_zeros(obs.shape[0])
    else:
        rewards = rewards.float().to(obs.device)

    seq_lens = loss_inputs.get(SampleBatch.SEQ_LENS)
    if seq_lens is None:
        B, T = 1, obs.shape[0]
        valid_mask = obs.new_ones(B, T)
    else:
        seq_lens = seq_lens.long().to(obs.device)
        B = int(seq_lens.numel())
        if B <= 0 or obs.shape[0] % B != 0:
            return torch.zeros_like(reference_loss), {}
        T = obs.shape[0] // B
        valid_mask = torch.arange(T, device=obs.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        valid_mask = valid_mask.float()

    local_obs = obs[:, local_obs_slice].reshape(B, T, 1, local_obs_dim)
    state = obs[:, state_slice].reshape(B, T, state_dim)
    actions = actions.reshape(B, T, 1)
    rewards = rewards.reshape(B, T, 1)

    wm_loss, _, wm_stats = world_model.compute_loss(
        local_obs,
        actions,
        state,
        rewards,
        valid_mask,
    )
    weighted_loss = float(coeff) * wm_loss
    stats = {
        'wm2_loss': weighted_loss.detach(),
        'wm2_coeff': torch.tensor(float(coeff), dtype=weighted_loss.dtype, device=weighted_loss.device),
    }
    for key, value in wm_stats.items():
        stats[f'wm2_{key}'] = torch.tensor(value, dtype=weighted_loss.dtype, device=weighted_loss.device)
    return weighted_loss, stats


DEFAULT_WM2_CONFIG = {
    'enabled': True,
    'stoch_dim': 32,
    'deter_dim': 128,
    'hidden_dim': 128,
    'action_embed_dim': 16,
    'embed_dim': 128,
    'imagination_horizon': 5,
    'kl_coeff': 1.0,
    'free_nats': 1.0,
    'wm_loss_weight': 0.5,
}