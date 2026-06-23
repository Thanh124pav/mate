import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ray.rllib.agents.qplex_v2.mixers import LamdaWeight, Qatten_Weight


class FocusDuelMixer(nn.Module):
    """QPLEX_V2 mixer that can expose lambda_i for FOCUS credit regularization."""

    def __init__(self, args, n_agents, n_actions, state_shape, mixing_embed_dim, ffn_hidden_dim, n_kernel):
        super(FocusDuelMixer, self).__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = self.n_agents * self.n_actions
        self.embed_dim = mixing_embed_dim

        self.attention_weight = Qatten_Weight(
            args, n_agents, state_shape, n_actions, mixing_embed_dim, ffn_hidden_dim, n_kernel
        )
        self.si_weight = LamdaWeight(args, n_agents, n_actions, state_shape, num_kernel=n_kernel)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        return torch.sum(agent_qs, dim=-1)

    def lambda_weights(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        return self.si_weight(states, actions).view(-1, self.n_agents)

    def credit_prior(self, states):
        """Return the softmax agent prior p_i before action modulation."""
        states = states.reshape(-1, self.state_dim)
        head_priors = []
        head_keys = []
        scale_factor = math.log(self.n_agents)
        for key_ext, agent_ext in zip(self.si_weight.key_extractors, self.si_weight.agents_extractors):
            key = torch.abs(key_ext(states)) + 1e-10
            prior = F.softmax(agent_ext(states) / scale_factor, dim=-1)
            head_keys.append(key)
            head_priors.append(prior)

        priors = torch.stack(head_priors, dim=1)
        keys = torch.stack(head_keys, dim=1)
        prior = (priors * keys).sum(dim=1) / (keys.sum(dim=1) + 1e-10)
        return prior

    def calc_adv(self, agent_qs, states, actions, max_action_vals, return_lambda=False):
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_action_vals = max_action_vals.view(-1, self.n_agents)

        adv_q = (agent_qs - max_action_vals).view(-1, self.n_agents).detach()
        adv_w_final = self.lambda_weights(states, actions)

        if self.args.is_minus_one:
            adv_tot = torch.sum(adv_q * (adv_w_final - 1.0), dim=1)
        else:
            adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
        if return_lambda:
            return adv_tot, adv_w_final
        return adv_tot

    def forward(
        self,
        agent_qs,
        states,
        actions=None,
        max_action_vals=None,
        is_v=False,
        return_lambda=False,
    ):
        bs = agent_qs.size(0)

        w_final, v, _, _ = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v

        if is_v:
            y = self.calc_v(agent_qs)
            v_tot = y.view(bs, -1, 1)
            if return_lambda:
                return v_tot, None
            return v_tot

        max_action_vals = max_action_vals.view(-1, self.n_agents)
        max_action_vals = w_final * max_action_vals + v
        out = self.calc_adv(
            agent_qs,
            states,
            actions=actions,
            max_action_vals=max_action_vals,
            return_lambda=return_lambda,
        )
        if return_lambda:
            y, lambda_w = out
            return y.view(bs, -1, 1), lambda_w.view(bs, -1, self.n_agents)
        return out.view(bs, -1, 1)
