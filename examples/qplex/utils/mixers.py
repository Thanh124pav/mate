import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class LamdaWeight(nn.Module):
    def __init__(self, args, n_agents, n_actions, state_shape, num_kernel):
        super(LamdaWeight, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        adv_hypernet_embed = self.args.adv_hypernet_embed
        for i in range(self.num_kernel):  # multi-head attention 
            # Each kernel having a Key NN, Agent NN, Action NN, each of them will be added to key_extractors, agents_extractors, action_extractors
            # key NN: state_dim -> 1, Agent NN: state_dim -> n_agents, Action NN: state_dim + action_dim -> n_agents
            if getattr(args, "adv_hypernet_layers", 1) == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))  # key
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  # agent
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 2:
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 3:
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
            else:
                raise Exception("Error setting number of adv hypernet layers.")

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim) # [B*T, state_dim]
        actions = actions.reshape(-1, self.action_dim) # [B*T, action_dim]
        data = torch.cat([states, actions], dim=1) # [B*T, state_dim + action_dim]

        all_head_key = [k_ext(states) for k_ext in self.key_extractors] # [num_kernel,]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors] # [num_kernel, n_agents]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors] # [num_kernel, n_agents]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = torch.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = torch.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = torch.sum(head_attend, dim=1)

        return head_attend
    





class DuelMixer(nn.Module):
    def __init__(self, args, n_agents, n_actions, state_shape, mixing_embed_dim, ffn_hidden_dim, n_kernel):
        self.args = args
        self.n_agents = n_agents 
        self.n_actions = n_actions 
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = n_agents + n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.embed_dim = mixing_embed_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_hidden_dim, self.n_agents)
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_hidden_dim, self.n_agents)
        )
        self.lamda_weight = LamdaWeight(args, n_agents,n_actions, state_shape, num_kernel=n_kernel)
    def calc_v(self, agent_qs):
        agent_vs = agent_qs.view(-1, self.n_agents)
        V_tot = torch.sum(agent_vs, dim=-1)
        return V_tot
    
    def calc_adv(self, agent_qs, states, actions, max_action_vals):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_action_vals = max_action_vals(-1, self.n_agents)
        adv_q = (agent_qs - max_action_vals).view(-1, self.n_agents).detach()
        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
        return adv_tot
    def calc(self, agent_qs, states, actions=None, max_action_vals=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_action_vals)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_action_vals=None, is_v=False):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w_final = self.hyper_w_final(states)
        w_final = torch.abs(w_final)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)
        v = v.view(-1, self.n_agents)

        if self.args.weighted_head:
            agent_qs = w_final * agent_qs + v
        if not is_v:
            max_action_vals = max_action_vals.view(-1, self.n_agents)
            if self.args.weighted_head:
                max_action_vals = w_final * max_action_vals + v

        y = self.calc(agent_qs, states, actions=actions, max_action_vals=max_action_vals, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot
