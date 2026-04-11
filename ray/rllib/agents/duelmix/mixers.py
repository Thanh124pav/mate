"""DuelMIX mixing networks.

DuelMIX decomposes Q_tot = V_tot + A_tot where:
  - V_tot uses UNRESTRICTED weights (full expressiveness)
  - A_tot uses POSITIVE weights (monotonicity for IGM)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LamdaWeight(nn.Module):
    """Advantage mixing weights (positive).

    Produces per-agent positive weights for advantage mixing via
    multi-head attention-like mechanism with abs(key) * softmax(agents) * (tanh(action)+1).
    """

    def __init__(self, args, n_agents, n_actions, state_shape, num_kernel):
        super().__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.num_kernel = num_kernel

        adv_hypernet_embed = getattr(args, "adv_hypernet_embed", 64)
        adv_hypernet_layers = getattr(args, "adv_hypernet_layers", 2)

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        for _ in range(self.num_kernel):
            if adv_hypernet_layers == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))
            elif adv_hypernet_layers == 2:
                self.key_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, 1),
                ))
                self.agents_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
                self.action_extractors.append(nn.Sequential(
                    nn.Linear(self.state_action_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
            else:
                self.key_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, 1),
                ))
                self.agents_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
                self.action_extractors.append(nn.Sequential(
                    nn.Linear(self.state_action_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = torch.cat([states, actions], dim=1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [a_ext(states) for a_ext in self.agents_extractors]
        all_head_action = [s_ext(data) for s_ext in self.action_extractors]

        head_attend_weights = []
        for curr_key, curr_agents, curr_action in zip(
            all_head_key, all_head_agents, all_head_action
        ):
            x_key = torch.abs(curr_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.softmax(curr_agents, dim=-1)
            x_action = torch.tanh(curr_action) + 1
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = torch.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = torch.sum(head_attend, dim=1)
        return head_attend


class DuelMixMixer(nn.Module):
    """DuelMIX mixer: unrestricted V mixing + monotonic A mixing.

    Q_tot = V_tot + A_tot
    V_tot = Σ w'_i(s) * V_i(s) + b_v(s)     [UNRESTRICTED w']
    A_tot = Σ λ_i(s,u) * A_i(s)              [POSITIVE λ]

    Each agent's V and A are first transformed by state-dependent weights:
        V_i(s) = w_i(s) * V_i(h_i) + b_i(s)
        A_i(s) = w_i(s) * A_i(h_i, u_i)       [no bias for advantage]
    """

    def __init__(self, args, n_agents, n_actions, state_shape,
                 mixing_embed_dim, ffn_hidden_dim, num_kernel):
        super().__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = mixing_embed_dim

        # --- Transformation layer (state-dependent per-agent weights) ---
        self.w_transform = nn.Sequential(
            nn.Linear(self.state_dim, ffn_hidden_dim), nn.ReLU(),
            nn.Linear(ffn_hidden_dim, n_agents),
        )
        self.b_transform = nn.Sequential(
            nn.Linear(self.state_dim, ffn_hidden_dim), nn.ReLU(),
            nn.Linear(ffn_hidden_dim, n_agents),
        )

        # --- Value mixing (UNRESTRICTED weights — key DuelMIX feature) ---
        self.v_mix_w = nn.Sequential(
            nn.Linear(self.state_dim, mixing_embed_dim), nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents),
        )
        self.v_mix_bias = nn.Sequential(
            nn.Linear(self.state_dim, mixing_embed_dim), nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

        # --- Advantage mixing (POSITIVE weights) ---
        self.lambda_weight = LamdaWeight(
            args, n_agents, n_actions, state_shape, num_kernel=num_kernel
        )

    def forward(self, agent_vs, agent_as=None, states=None,
                actions=None, max_action_advs=None, is_v=False):
        """Compute V_tot or A_tot.

        Args:
            agent_vs: [B, T, n_agents] per-agent V values.
            agent_as: [B, T, n_agents] per-agent advantage values (for chosen actions).
            states: [B, T, n_agents, obs_size] or [B, T, state_dim].
            actions: [B, T, n_agents, n_actions] one-hot (for A mixing).
            max_action_advs: [B, T, n_agents] max advantages (for centering).
            is_v: if True, compute V_tot only.

        Returns:
            [B, T, 1] mixed value.
        """
        bs = agent_vs.size(0)
        states_flat = states.reshape(-1, self.state_dim)

        # State-dependent transformation weights
        w_t = torch.abs(self.w_transform(states_flat)) + 1e-10  # positive
        w_t = w_t.view(-1, self.n_agents)
        b_t = self.b_transform(states_flat).view(-1, self.n_agents)

        if is_v:
            # Transform V: V_i(s) = w_i(s)*V_i + b_i(s)
            agent_vs_flat = agent_vs.view(-1, self.n_agents)
            transformed_v = w_t * agent_vs_flat + b_t

            # Mix V (UNRESTRICTED weights)
            v_weights = self.v_mix_w(states_flat).view(-1, self.n_agents)
            v_bias = self.v_mix_bias(states_flat).view(-1, 1)
            v_tot = torch.sum(v_weights * transformed_v, dim=-1, keepdim=True) + v_bias
            return v_tot.view(bs, -1, 1)
        else:
            # Transform A: A_i(s) = w_i(s)*A_i (no bias)
            agent_as_flat = agent_as.view(-1, self.n_agents)
            transformed_a = w_t * agent_as_flat

            if max_action_advs is not None:
                max_advs_flat = max_action_advs.view(-1, self.n_agents)
                transformed_max = w_t * max_advs_flat
                adv_q = (transformed_a - transformed_max).detach()
            else:
                adv_q = transformed_a.detach()

            # Lambda weights (positive)
            lambda_w = self.lambda_weight(states_flat, actions)
            lambda_w = lambda_w.view(-1, self.n_agents)

            is_minus_one = getattr(self.args, "is_minus_one", True)
            if is_minus_one:
                a_tot = torch.sum(adv_q * (lambda_w - 1.0), dim=-1, keepdim=True)
            else:
                a_tot = torch.sum(adv_q * lambda_w, dim=-1, keepdim=True)

            return a_tot.view(bs, -1, 1)
