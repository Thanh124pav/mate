"""World Model for predicting next target positions.

Architecture: Transformer Encoder + Mixture of Experts (MoE)

Input: per-camera local observations → each camera produces its own prediction.
Label: ground truth target positions from the global state.
The model learns to map relative (per-camera) observations to absolute positions.

Per-token features (1 target as seen by 1 camera): 10 dims
    - target_pos (x, y): 2          — relative to camera, normalized
    - target_vel (vx, vy): 2        — estimated from consecutive timesteps
    - sight_range: 1
    - is_loaded: 1
    - visible_flag: 1
    - nearest_wh_direction (dx, dy): 2  — normalized direction to nearest warehouse
    - nearest_wh_distance: 1

Camera context (conditioning for MoE router): 18 dims
    - camera_pos (x, y): 2          — from self_state, enables relative→absolute
    - camera_rest: 7                — remaining self_state features
    - agent_id: 1                   — camera identity for MoE routing
    - warehouse_locs: 8             — 4 warehouses × 2 coords
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MATE observation layout constants
PRESERVED_DIM = 13
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PRIVATE = 14   # 6 + NUM_WAREHOUSES * 2
WAREHOUSE_START_IDX = 4         # index in preserved data
NUM_WAREHOUSES = 4
TARGET_OBS_START = 22           # PRESERVED_DIM + CAMERA_STATE_DIM_PRIVATE
TARGET_OBS_DIM = 5              # x, y, sight_range, is_loaded, visible_flag

TOKEN_DIM = 10                  # per-target token feature dimension
CAMERA_CONTEXT_DIM = 18         # camera conditioning dimension


# =============================================================================
# Mixture of Experts
# =============================================================================

class ExpertFFN(nn.Module):
    """Single expert: a small feed-forward network."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """MoE layer with camera-aware routing.

    Total experts = n_cameras × alpha.
    Each group of `alpha` experts specializes in one camera's perspective.
    The router uses token features + camera context (including agent_id)
    to select top_k experts per token.
    """

    def __init__(self, d_model, d_ff, n_cameras, alpha, top_k, camera_context_dim):
        super().__init__()
        self.n_experts = n_cameras * alpha
        self.alpha = alpha
        self.top_k = min(top_k, self.n_experts)
        self.d_model = d_model

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(self.n_experts)
        ])
        self.router = nn.Linear(d_model + camera_context_dim, self.n_experts)

    def forward(self, x, camera_context):
        """
        Args:
            x: [B, n_targets, d_model]
            camera_context: [B, camera_context_dim]

        Returns:
            output: [B, n_targets, d_model]
        """
        B, S, D = x.shape

        # Router input: token features + camera context (broadcast to all tokens)
        ctx = camera_context.unsqueeze(1).expand(-1, S, -1)   # [B, S, ctx_dim]
        gate_logits = self.router(torch.cat([x, ctx], dim=-1))  # [B, S, n_experts]

        # Top-k expert selection
        top_k_val, top_k_idx = gate_logits.topk(self.top_k, dim=-1)  # [B, S, top_k]
        top_k_weights = F.softmax(top_k_val, dim=-1)                 # [B, S, top_k]

        # Compute all expert outputs (feasible for small n_experts ≤ 16)
        expert_out = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )  # [B, S, n_experts, D]

        # Gather selected experts and combine
        idx_exp = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, S, top_k, D]
        selected = torch.gather(expert_out, 2, idx_exp)           # [B, S, top_k, D]
        output = (selected * top_k_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, D]

        return output


# =============================================================================
# Transformer Layer with MoE
# =============================================================================

class TransformerMoELayer(nn.Module):
    """Single Transformer encoder layer: Self-Attention + MoE FFN."""

    def __init__(self, d_model, n_heads, d_ff, n_cameras, alpha, top_k,
                 camera_context_dim, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = MixtureOfExperts(d_model, d_ff, n_cameras, alpha, top_k, camera_context_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, camera_context):
        """
        Args:
            x: [B, n_targets, d_model]
            camera_context: [B, camera_context_dim]

        Returns:
            x: [B, n_targets, d_model]
        """
        # Self-attention (all targets attend to all — visible_flag is a feature)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # MoE FFN
        moe_out = self.moe(x, camera_context)
        x = self.norm2(x + self.dropout(moe_out))

        return x


# =============================================================================
# World Model
# =============================================================================

class TargetWorldModel(nn.Module):
    """Transformer + MoE world model for target position prediction.

    Each camera's local observation produces its own prediction.
    During training, predictions are supervised by global state (absolute positions).
    The camera_pos in the input allows the model to learn relative → absolute mapping.

    Args:
        n_cameras: number of camera agents
        n_targets: number of targets in the environment
        n_warehouses: number of warehouses (default 4)
        d_model: transformer hidden dimension (default 64)
        n_heads: number of attention heads (default 4)
        n_layers: number of transformer layers (default 1)
        d_ff: FFN hidden dimension in each MoE expert (default 64)
        alpha: number of experts per camera (default 2)
        top_k: number of experts activated per token (default 2)
        dropout: dropout rate (default 0.0)
    """

    def __init__(self, n_cameras, n_targets, n_warehouses=NUM_WAREHOUSES,
                 d_model=64, n_heads=4, n_layers=1, d_ff=64,
                 alpha=2, top_k=2, dropout=0.0):
        super(TargetWorldModel, self).__init__()

        self.n_cameras = n_cameras
        self.n_targets = n_targets
        self.n_warehouses = n_warehouses
        self.d_model = d_model

        # Input projection: token_dim → d_model
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)

        # Transformer encoder layers with MoE
        self.layers = nn.ModuleList([
            TransformerMoELayer(
                d_model, n_heads, d_ff, n_cameras, alpha, top_k,
                CAMERA_CONTEXT_DIM, dropout,
            )
            for _ in range(n_layers)
        ])

        # Output head: d_model → 2 (predicted absolute x, y)
        self.output_head = nn.Linear(d_model, 2)

        # For state augmentation
        self.output_dim = n_targets * 2

        # Indices for extracting target positions from global state
        self.target_global_start = PRESERVED_DIM + n_cameras * CAMERA_STATE_DIM_PRIVATE

    # -----------------------------------------------------------------
    # Feature extraction from local observations
    # -----------------------------------------------------------------

    def extract_features(self, obs, prev_obs=None):
        """Extract per-camera, per-target token features from local observations.

        Args:
            obs: [B, T, n_cameras, obs_size] — local observations
            prev_obs: [B, T, n_cameras, obs_size] or None — for velocity estimation.
                      If None, velocity is computed from consecutive timesteps within obs.

        Returns:
            tokens: [B*T*n_cameras, n_targets, TOKEN_DIM=10]
            camera_ctx: [B*T*n_cameras, CAMERA_CONTEXT_DIM=18]
            vis_flags: [B*T*n_cameras, n_targets]
        """
        B, T, C, obs_size = obs.shape

        # --- Target raw features: [B, T, C, n_targets, 5] ---
        t_start = TARGET_OBS_START
        raw = obs[:, :, :, t_start:t_start + self.n_targets * TARGET_OBS_DIM]
        raw = raw.reshape(B, T, C, self.n_targets, TARGET_OBS_DIM)

        target_pos = raw[..., :2]          # [B, T, C, n_targets, 2]
        sight_range = raw[..., 2:3]        # [B, T, C, n_targets, 1]
        is_loaded = raw[..., 3:4]          # [B, T, C, n_targets, 1]
        visible = raw[..., 4:5]            # [B, T, C, n_targets, 1]
        vis_flags = raw[..., 4]            # [B, T, C, n_targets]

        # --- Velocity ---
        if prev_obs is not None:
            prev_raw = prev_obs[:, :, :, t_start:t_start + self.n_targets * TARGET_OBS_DIM]
            prev_raw = prev_raw.reshape(B, T, C, self.n_targets, TARGET_OBS_DIM)
            velocity = target_pos - prev_raw[..., :2]
        else:
            velocity = torch.zeros_like(target_pos)
            velocity[:, 1:] = target_pos[:, 1:] - target_pos[:, :-1]

        # --- Warehouse positions: [B, T, C, n_warehouses, 2] ---
        wh_start = WAREHOUSE_START_IDX
        wh_flat = obs[:, :, :, wh_start:wh_start + self.n_warehouses * 2]
        wh = wh_flat.reshape(B, T, C, self.n_warehouses, 2)

        # --- Nearest warehouse direction and distance per target ---
        # target_pos: [B,T,C, n_targets, 1, 2]  vs  wh: [B,T,C, 1, n_wh, 2]
        diff = wh.unsqueeze(3) - target_pos.unsqueeze(4)     # [B,T,C, n_targets, n_wh, 2]
        dist = diff.norm(dim=-1)                               # [B,T,C, n_targets, n_wh]
        nearest_idx = dist.argmin(dim=-1, keepdim=True)        # [B,T,C, n_targets, 1]

        nearest_dir = torch.gather(
            diff, 4, nearest_idx.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 2)
        ).squeeze(4)                                            # [B,T,C, n_targets, 2]
        nearest_dist = torch.gather(dist, 4, nearest_idx).squeeze(-1)  # [B,T,C, n_targets]
        nearest_dir_norm = nearest_dir / (nearest_dist.unsqueeze(-1) + 1e-8)

        # --- Assemble tokens: [B, T, C, n_targets, 10] ---
        tokens = torch.cat([
            target_pos,                    # 2
            velocity,                      # 2
            sight_range,                   # 1
            is_loaded,                     # 1
            visible,                       # 1
            nearest_dir_norm,              # 2
            nearest_dist.unsqueeze(-1),    # 1
        ], dim=-1)                         # total = 10

        # --- Camera context: [B, T, C, 18] ---
        camera_pos = obs[:, :, :, 13:15]                                  # 2
        camera_rest = obs[:, :, :, 15:22]                                 # 7
        agent_id = obs[:, :, :, 3:4]                                      # 1
        wh_ctx = obs[:, :, :, wh_start:wh_start + self.n_warehouses * 2]  # 8
        camera_ctx = torch.cat([camera_pos, camera_rest, agent_id, wh_ctx], dim=-1)  # 18

        # --- Flatten batch dims ---
        tokens = tokens.reshape(B * T * C, self.n_targets, TOKEN_DIM)
        camera_ctx = camera_ctx.reshape(B * T * C, CAMERA_CONTEXT_DIM)
        vis_flags = vis_flags.reshape(B * T * C, self.n_targets)

        return tokens, camera_ctx, vis_flags

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, tokens, camera_context):
        """Run Transformer + MoE on token features.

        Args:
            tokens: [B, n_targets, TOKEN_DIM]
            camera_context: [B, CAMERA_CONTEXT_DIM]

        Returns:
            predicted_pos: [B, n_targets, 2] — predicted absolute next positions
        """
        x = self.token_proj(tokens)  # [B, n_targets, d_model]

        for layer in self.layers:
            x = layer(x, camera_context)

        return self.output_head(x)  # [B, n_targets, 2]

    # -----------------------------------------------------------------
    # Global state helpers (for labels)
    # -----------------------------------------------------------------

    def extract_global_target_positions(self, state):
        """Extract absolute target (x, y) from the env global state.

        Args:
            state: [B, T, state_dim]

        Returns:
            positions: [B, T, n_targets, 2]
        """
        positions = []
        for i in range(self.n_targets):
            start = self.target_global_start + i * TARGET_STATE_DIM_PRIVATE
            positions.append(state[:, :, start:start + 2])
        return torch.stack(positions, dim=2)  # [B, T, n_targets, 2]

    # -----------------------------------------------------------------
    # Global state helpers (camera positions for reward shaping)
    # -----------------------------------------------------------------

    def extract_camera_positions(self, state):
        """Extract absolute camera (x, y) positions from the env global state.

        Args:
            state: [B, T, state_dim]

        Returns:
            positions: [B, T, n_cameras, 2]
        """
        positions = []
        for i in range(self.n_cameras):
            start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE
            positions.append(state[:, :, start:start + 2])
        return torch.stack(positions, dim=2)

    # -----------------------------------------------------------------
    # Predict: per-camera (for obs augmentation)
    # -----------------------------------------------------------------

    def predict_per_camera(self, obs, prev_obs=None):
        """Predict next target positions per camera (not aggregated).

        Args:
            obs: [B, T, n_cameras, obs_size]
            prev_obs: [B, T, n_cameras, obs_size] or None

        Returns:
            per_camera_pred: [B, T, n_cameras, n_targets, 2]
        """
        B, T, C, _ = obs.shape
        tokens, ctx, vis = self.extract_features(obs, prev_obs)
        pred = self.forward(tokens, ctx)                # [B*T*C, n_targets, 2]
        return pred.reshape(B, T, C, self.n_targets, 2)

    # -----------------------------------------------------------------
    # Predict: aggregated across cameras (for state augmentation)
    # -----------------------------------------------------------------

    def predict(self, obs, prev_obs=None):
        """Predict next target positions, aggregated across cameras.

        Args:
            obs: [B, T, n_cameras, obs_size]
            prev_obs: [B, T, n_cameras, obs_size] or None

        Returns:
            agg_predictions: [B, T, n_targets, 2]
        """
        B, T, C, _ = obs.shape
        tokens, ctx, vis = self.extract_features(obs, prev_obs)
        pred = self.forward(tokens, ctx).reshape(B, T, C, self.n_targets, 2)
        vis = vis.reshape(B, T, C, self.n_targets)

        vis_w = vis.unsqueeze(-1)
        agg = (pred * vis_w).sum(dim=2) / vis_w.sum(dim=2).clamp(min=1)
        return agg

    # -----------------------------------------------------------------
    # Training: compute loss + per-camera & aggregated predictions
    # -----------------------------------------------------------------

    def compute_loss(self, obs, state, next_state, mask):
        """Compute world model prediction loss.

        Input: per-camera local observations (relative coords).
        Labels: ground truth from global state (absolute coords).

        Args:
            obs: [B, T, n_cameras, obs_size]
            state: [B, T, state_dim] — current global state
            next_state: [B, T, state_dim] — next global state (labels)
            mask: [B, T]

        Returns:
            wm_loss: scalar — masked MSE on visible targets
            agg_pred: [B, T, n_targets, 2] — aggregated (for state augmentation)
            per_cam_pred: [B, T, n_cameras, n_targets, 2] — per-camera (for obs augmentation)
        """
        B, T, C, _ = obs.shape

        tokens, ctx, vis = self.extract_features(obs)
        pred = self.forward(tokens, ctx)                 # [B*T*C, n_targets, 2]
        pred = pred.reshape(B, T, C, self.n_targets, 2)
        vis = vis.reshape(B, T, C, self.n_targets)

        # Labels from next global state
        labels = self.extract_global_target_positions(next_state)
        labels_exp = labels.unsqueeze(2).expand_as(pred)

        # MSE loss weighted by visibility
        error = (pred - labels_exp.detach()) ** 2
        error = error.mean(dim=-1)                                 # [B, T, C, n_targets]
        vis_error = (error * vis).sum(dim=(2, 3))
        vis_count = vis.sum(dim=(2, 3)).clamp(min=1)
        per_step_loss = vis_error / vis_count
        wm_loss = (per_step_loss * mask).sum() / mask.sum().clamp(min=1)

        # Aggregated predictions
        vis_w = vis.unsqueeze(-1)
        agg_pred = (pred * vis_w).sum(dim=2) / vis_w.sum(dim=2).clamp(min=1)

        return wm_loss, agg_pred, pred

    # -----------------------------------------------------------------
    # State augmentation
    # -----------------------------------------------------------------

    def augment_state(self, state_seq, predictions):
        """Concatenate aggregated predictions to global state for the mixer.

        Args:
            state_seq: [B, T, state_dim]
            predictions: [B, T, n_targets, 2]

        Returns:
            augmented: [B, T, state_dim + n_targets * 2]
        """
        B, T = state_seq.shape[:2]
        return torch.cat([state_seq, predictions.reshape(B, T, -1)], dim=-1)
