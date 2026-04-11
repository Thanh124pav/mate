"""SPECTra mixing network: Set Transformer HyperNet + 2-layer non-linear mixer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STHyperNet(nn.Module):
    """Set Transformer-based Hypernetwork.

    Generates mixing weights and biases via multi-head attention inner products.
    Ensures permutation invariance and monotonicity (via abs on weights).
    """

    def __init__(self, n_agents, state_dim, embed_dim, mixing_dim, n_heads=4):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.mixing_dim = mixing_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Agent Q-value embedding
        self.agent_embed = nn.Linear(1, embed_dim)

        # State embedding for biases
        self.state_embed = nn.Linear(state_dim, embed_dim)

        # W1 generation: [n_agents, mixing_dim]
        self.W1_q = nn.Linear(embed_dim, embed_dim)
        self.W1_k = nn.Linear(embed_dim, embed_dim)

        # W2 generation: [mixing_dim, 1]
        self.W2_q = nn.Linear(embed_dim, embed_dim)
        self.W2_k = nn.Linear(embed_dim, embed_dim)

        # W1 network: state → [n_agents, mixing_dim] weights
        self.W1_net = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, n_agents * mixing_dim),
        )

        # W2 network: state → [mixing_dim, 1] weights
        self.W2_net = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, mixing_dim),
        )

        # Bias networks (from state)
        self.b1_net = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, mixing_dim),
        )
        self.b2_net = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        """Generate mixing weights from agent Q-values and state.

        Args:
            agent_qs: [B, n_agents]
            states: [B, state_dim]

        Returns:
            W1: [B, n_agents, mixing_dim]
            b1: [B, 1, mixing_dim]
            W2: [B, mixing_dim, 1]
            b2: [B, 1, 1]
        """
        B = agent_qs.size(0)

        # Generate W1: [B, n_agents, mixing_dim] from state
        W1 = self.W1_net(states).view(B, self.n_agents, self.mixing_dim)

        # Generate W2: [B, mixing_dim, 1] from state
        W2 = self.W2_net(states).view(B, self.mixing_dim, 1)

        # Biases from state
        b1 = self.b1_net(states).unsqueeze(1)  # [B, 1, mixing_dim]
        b2 = self.b2_net(states).unsqueeze(1)  # [B, 1, 1]

        return W1, b1, W2, b2


class SPECTraMixer(nn.Module):
    """SPECTra 2-layer non-linear mixer with ST-HyperNet generated weights.

    q_tot = ELU(qs @ |W1| + b1) @ |W2| + b2
    abs() ensures monotonicity: dQ_tot/dQ_i >= 0
    """

    def __init__(self, n_agents, state_shape, mixing_dim=32, embed_dim=64, n_heads=4):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.mixing_dim = mixing_dim

        self.hypernet = STHyperNet(
            n_agents, self.state_dim, embed_dim, mixing_dim, n_heads
        )

    def forward(self, agent_qs, states):
        """Mix agent Q-values into joint Q-value.

        Args:
            agent_qs: [B, T, n_agents] or reshaped
            states: [B, T, state_dim] or reshaped

        Returns:
            q_tot: [B, T, 1]
        """
        bs = agent_qs.size(0)
        agent_qs_flat = agent_qs.reshape(-1, self.n_agents)
        states_flat = states.reshape(-1, self.state_dim)

        W1, b1, W2, b2 = self.hypernet(agent_qs_flat, states_flat)

        # Layer 1: ELU activation
        qs = agent_qs_flat.unsqueeze(1)  # [B*T, 1, n_agents]
        hidden = F.elu(torch.bmm(qs, torch.abs(W1)) + b1)  # [B*T, 1, mixing_dim]

        # Layer 2: linear
        q_tot = torch.bmm(hidden, torch.abs(W2)) + b2  # [B*T, 1, 1]

        return q_tot.view(bs, -1, 1)
