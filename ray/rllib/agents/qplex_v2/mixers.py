import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.framework import try_import_torch
import math
torch, nn = try_import_torch()


class LamdaWeight(nn.Module):
    def __init__(self, args, n_agents, n_actions, state_shape, num_kernel):
        super(LamdaWeight, self).__init__()
        self.args = args
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
            x_key = torch.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10 # (B*T, n_agents)
            scale_factor = math.sqrt(self.n_agents) 
            x_agents = F.softmax(curr_head_agents, dim=-1) # (B*T, n_agents)
            x_action = F.tanh(curr_head_action) + 1 #B*T, n_agents)
            agent_action_weights = (x_agents * x_action)
            weights = x_key * agent_action_weights # (B*T, n_agents)
            head_attend_weights.append(weights)

        head_attend = torch.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = torch.sum(head_attend, dim=1) 
        
        return head_attend

# class DuelMixerV2(nn.Module):
#     def __init__(self, args, n_agents, n_actions, state_shape, mixing_embed_dim, ffn_hidden_dim, n_kernel):
#         super(DuelMixerV2, self).__init__()
#         self.args = args
#         self.n_agents = n_agents 
#         self.n_actions = n_actions 
#         self.state_dim = int(np.prod(state_shape))
#         self.action_dim = n_agents * n_actions
#         self.state_action_dim = self.state_dim + self.action_dim + 1
#         self.embed_dim = mixing_embed_dim
#         self.ffn_hidden_dim = ffn_hidden_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(self.state_dim, self.ffn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.ffn_hidden_dim, self.n_agents)
#         )
#         self.V = nn.Sequential(
#             nn.Linear(self.state_dim, self.ffn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.ffn_hidden_dim, self.n_agents)
#         )
#         self.lamda_weight = LamdaWeight(args, n_agents,n_actions, state_shape, num_kernel=n_kernel)
#     def calc_v(self, agent_qs):
#         agent_vs = agent_qs.view(-1, self.n_agents)
#         V_tot = torch.sum(agent_vs, dim=-1)
#         return V_tot
    
#     def calc_adv(self, agent_qs, states, actions, max_action_vals):
#         states = states.reshape(-1, self.state_dim)
#         actions = actions.reshape(-1, self.action_dim)
#         agent_qs = agent_qs.view(-1, self.n_agents)
#         max_action_vals = max_action_vals.view(-1, self.n_agents)
#         adv_q = (agent_qs - max_action_vals).view(-1, self.n_agents).detach()
#         adv_w_final = self.lamda_weight(states, actions)
#         adv_w_final = adv_w_final.view(-1, self.n_agents)

#         if self.args.is_minus_one:
#             adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
#         else:
#             adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
#         return adv_tot
#     def calc(self, agent_qs, states, actions=None, max_action_vals=None, is_v=False):
#         if is_v:
#             v_tot = self.calc_v(agent_qs)
#             return v_tot
#         else:
#             adv_tot = self.calc_adv(agent_qs, states, actions, max_action_vals)
#             return adv_tot

#     def forward(self, agent_qs, states, actions=None, max_action_vals=None, is_v=False):
#         bs = agent_qs.size(0)
#         states = states.reshape(-1, self.state_dim)
#         agent_qs = agent_qs.view(-1, self.n_agents)

#         w_final = self.mlp(states)
#         w_final = torch.abs(w_final)
#         w_final = w_final.view(-1, self.n_agents) + 1e-10
#         v = self.V(states)
#         v = v.view(-1, self.n_agents)
#         if self.args.weighted_head:
#             agent_qs = w_final * agent_qs + v
#         if not is_v:
#             max_action_vals = max_action_vals.view(-1, self.n_agents)
#             if self.args.weighted_head:
#                 max_action_vals = w_final * max_action_vals + v

#         y = self.calc(agent_qs, states, actions=actions, max_action_vals=max_action_vals, is_v=is_v)
#         v_tot = y.view(bs, -1, 1)

#         return v_tot


class DuelMixerV2(nn.Module):
    def __init__(self, args, n_agents, n_actions, state_shape, mixing_embed_dim, ffn_hidden_dim, n_kernel):
        super(DuelMixerV2, self).__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = self.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.embed_dim = mixing_embed_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.attention_weight = Qatten_Weight(args, n_agents, state_shape, n_actions, mixing_embed_dim, ffn_hidden_dim, n_kernel)
        self.si_weight = LamdaWeight(args, n_agents,n_actions, state_shape, num_kernel=n_kernel)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = torch.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_action_vals):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_action_vals = max_action_vals.view(-1, self.n_agents)

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

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents)  + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_action_vals = max_action_vals.view(-1, self.n_agents)
            max_action_vals = w_final * max_action_vals + v

        y = self.calc(agent_qs, states, actions=actions, max_action_vals=max_action_vals, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot


class Qatten_Weight(nn.Module):
    def __init__(self, args, n_agents, state_shape, n_actions, mixing_embed_dim, ffn_hidden_dim, n_head ):
        super(Qatten_Weight, self).__init__()

        self.name = 'qatten_weight'
        self.args = args
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.unit_dim = 1
        self.n_actions = n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = n_head  # attention head num

        self.embed_dim = mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef if hasattr(args, 'attend_reg_coef') else 0.001
        self.nonlinear = args.nonlinear if hasattr(args, 'nonlinear') else False
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        hypernet_embed = ffn_hidden_dim
        for i in range(self.n_head):  # multi-head attention
            selector_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim, bias=False))
            self.selector_extractors.append(selector_nn)  # query
            if self.nonlinear:  # add qs
                self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
            else:
                self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                              nn.ReLU(),
                                              nn.Linear(hypernet_embed, self.n_head))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.nonlinear:
            unit_states = torch.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = torch.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                      torch.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            self.mask_dead = self.args.mask_dead if hasattr(self.args, 'mask_dead') else False
            if self.mask_dead:
                # actions: (episode_batch, episode_length - 1, agent_num, 1)
                actions = actions.reshape(-1, 1, self.n_agents)
                # actions: (batch_size, 1, agent_num)
                scaled_attend_logits[actions == 0] = -99999999  # action == 0 means the unit is dead
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = torch.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V(states).view(-1, 1)  # v: (bs, 1)
        # head_qs: [head_num, bs, 1]
        if self.args.weighted_head:
            w_head = torch.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
            # w_head = self.hyper_w_head(states)
            # w_head = w_head*w_head
            # w_head = self.hyper_w_head(states)
            w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)  # w_head: (bs, head_num, self.n_agents)
            #w_head = F.softplus(w_head, dim=1)
            head_attend *= w_head
        head_attend *= np.sqrt(self.n_head)
        head_attend = torch.sum(head_attend, dim=1)
        if not hasattr(self.args, 'state_bias'):
            setattr(self.args, 'state_bias', True)
        if not self.args.state_bias:
            v *= 0.

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]

        return head_attend, v, attend_mag_regs, head_entropies