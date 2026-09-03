"""SPECTra agent model: SAQA attention + GRU + policy decoupling."""

import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class SAQAModule(nn.Module):
    """Single-Agent Query Attention.

    Uses agent's own embedding as query and all entity embeddings as keys/values.
    Complexity: O(n_entities * embed_dim) — linear, not quadratic.
    """

    def __init__(self, own_dim, entity_dim, embed_dim, n_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Embedding layers
        self.E_own = nn.Linear(own_dim, embed_dim)
        self.E_entity = nn.Linear(entity_dim, embed_dim)

        # Attention projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, own_obs, entity_obs):
        """
        Args:
            own_obs: [B, own_dim] agent's own observation
            entity_obs: [B, n_entities, entity_dim] entity observations

        Returns:
            [B, embed_dim] attended embedding
        """
        B = own_obs.size(0)
        n_entities = entity_obs.size(1)

        # Embed
        own_embed = self.E_own(own_obs)          # [B, embed_dim]
        ent_embed = self.E_entity(entity_obs)    # [B, n_entities, embed_dim]

        # Query from own, Keys/Values from entities
        Q = self.W_q(own_embed).unsqueeze(1)     # [B, 1, embed_dim]
        K = self.W_k(ent_embed)                  # [B, n_entities, embed_dim]
        V = self.W_v(ent_embed)                  # [B, n_entities, embed_dim]

        # Multi-head reshape
        Q = Q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, n_entities, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, n_entities, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, 1, n_ent]
        attn = torch.softmax(attn, dim=-1)
        attended = torch.matmul(attn, V)  # [B, heads, 1, head_dim]

        # Merge heads
        attended = attended.transpose(1, 2).reshape(B, self.embed_dim)
        attended = self.W_o(attended)

        # Residual + LayerNorm
        output = self.layer_norm(own_embed + attended)
        return output


class SPECTraRNNModel(TorchModelV2, nn.Module):
    """SPECTra agent model with SAQA and GRU.

    Parses flat observation into own_obs and entity_obs using configured dimensions.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.n_agents = model_config["n_agents"]

        # Entity dimensions from config (with sensible defaults for MATE 4v5)
        self.own_dim = model_config.get("own_obs_dim", self.obs_size)
        entity_dim = model_config.get("entity_obs_dim", self.own_dim)
        embed_dim = model_config.get("entity_embed_dim", 64)
        n_heads = model_config.get("n_attention_heads", 4)

        # SAQA attention
        self.saqa = SAQAModule(self.own_dim, entity_dim, embed_dim, n_heads)

        # Fallback: if obs can't be split into entities, use simple FC
        self.use_saqa = model_config.get("use_saqa", False)

        if self.use_saqa:
            self.fc1 = nn.Linear(embed_dim, self.rnn_hidden_dim)
        else:
            # Simple FC fallback (works with any obs structure)
            self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

    @override(ModelV2)
    def get_initial_state(self):
        return [
            self.fc1.weight.new(self.n_agents, self.rnn_hidden_dim).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        obs = input_dict["obs_flat"].float()

        if self.use_saqa:
            own_obs, entity_obs = self._parse_obs(obs)
            attended = self.saqa(own_obs, entity_obs)
            x = nn.functional.relu(self.fc1(attended))
        else:
            x = nn.functional.relu(self.fc1(obs))

        h_in = hidden_state[0].reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, [h]

    def _parse_obs(self, obs):
        """Split flat observation into own_obs and entity_obs."""
        own = obs[:, :self.own_dim]
        entity_flat = obs[:, self.own_dim:]

        entity_dim = self.saqa.E_entity.in_features
        n_entities = entity_flat.size(1) // entity_dim
        if n_entities > 0:
            entity_obs = entity_flat[:, :n_entities * entity_dim].reshape(
                obs.size(0), n_entities, entity_dim
            )
        else:
            # No entities: use own obs as single entity
            entity_obs = own.unsqueeze(1)

        return own, entity_obs


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
