import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.agents.qplex_v2.mixers import Qatten_Weight


class Focus2LambdaWeight(nn.Module):
    """QPLEX-FOCUS2 lambda factorization: lambda_i = sum_h k_h p_hi m_hi."""

    def __init__(self, args, n_agents, n_actions, state_shape, num_kernel):
        super(Focus2LambdaWeight, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = n_agents * n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.num_kernel = num_kernel
        hidden = getattr(args, "adv_hypernet_embed", 64)
        layers = getattr(args, "adv_hypernet_layers", 2)

        self.key_extractors = nn.ModuleList()
        self.resp_extractors = nn.ModuleList()
        self.mag_extractors = nn.ModuleList()
        for _ in range(num_kernel):
            self.key_extractors.append(self._mlp(self.state_dim, 1, hidden, layers))
            self.resp_extractors.append(self._mlp(self.state_dim, n_agents, hidden, layers))
            self.mag_extractors.append(
                self._mlp(self.state_action_dim, n_agents, hidden, layers)
            )

    @staticmethod
    def _mlp(in_dim, out_dim, hidden, layers):
        if layers <= 1:
            return nn.Linear(in_dim, out_dim)
        modules = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(max(0, layers - 2)):
            modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        modules.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*modules)

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_action = torch.cat([states, actions], dim=-1)
        eps = 1e-10

        lambda_heads = []
        p_heads = []
        key_heads = []
        mag_heads = []
        for key_net, resp_net, mag_net in zip(
            self.key_extractors, self.resp_extractors, self.mag_extractors
        ):
            key = F.softplus(key_net(states)) + eps
            p = F.softmax(resp_net(states) / math.log(self.n_agents), dim=-1)
            mag = 1.0 + torch.tanh(mag_net(state_action))
            lambda_heads.append(key * p * mag)
            p_heads.append(p)
            key_heads.append(key)
            mag_heads.append(mag)

        lambda_w = torch.stack(lambda_heads, dim=1).sum(dim=1)
        keys = torch.stack(key_heads, dim=1)
        p_stack = torch.stack(p_heads, dim=1)
        p_dist = (keys * p_stack).sum(dim=1) / (keys.sum(dim=1) + eps)
        mag = torch.stack(mag_heads, dim=1).mean(dim=1)
        return lambda_w, p_dist, mag


class Focus2DuelMixer(nn.Module):
    """QPLEX mixer exposing p_i for FOCUS2 KL regularization."""

    def __init__(self, args, n_agents, n_actions, state_shape, mixing_embed_dim,
                 ffn_hidden_dim, n_kernel):
        super(Focus2DuelMixer, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = n_agents * n_actions
        self.attention_weight = Qatten_Weight(
            args, n_agents, state_shape, n_actions, mixing_embed_dim,
            ffn_hidden_dim, n_kernel
        )
        self.lambda_weight = Focus2LambdaWeight(
            args, n_agents, n_actions, state_shape, num_kernel=n_kernel
        )

    def calc_v(self, agent_qs):
        return torch.sum(agent_qs.view(-1, self.n_agents), dim=-1)

    def calc_adv(self, agent_qs, states, actions, max_action_vals, return_credit=False):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_action_vals = max_action_vals.view(-1, self.n_agents)
        adv_q = (agent_qs - max_action_vals).detach()
        lambda_w, p_dist, mag = self.lambda_weight(states, actions)
        if self.args.is_minus_one:
            adv_tot = torch.sum(adv_q * (lambda_w - 1.0), dim=-1)
        else:
            adv_tot = torch.sum(adv_q * lambda_w, dim=-1)
        if return_credit:
            return adv_tot, lambda_w, p_dist, mag
        return adv_tot

    def forward(self, agent_qs, states, actions=None, max_action_vals=None,
                is_v=False, return_credit=False):
        bs = agent_qs.size(0)
        w_final, v, _, _ = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents) / self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if is_v:
            out = self.calc_v(agent_qs).view(bs, -1, 1)
            if return_credit:
                return out, None, None
            return out

        max_action_vals = max_action_vals.view(-1, self.n_agents)
        max_action_vals = w_final * max_action_vals + v
        out = self.calc_adv(
            agent_qs, states, actions, max_action_vals, return_credit=return_credit
        )
        if return_credit:
            adv_tot, lambda_w, p_dist, _ = out
            return (
                adv_tot.view(bs, -1, 1),
                lambda_w.view(bs, -1, self.n_agents),
                p_dist.view(bs, -1, self.n_agents),
            )
        return out.view(bs, -1, 1)
