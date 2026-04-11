"""Stochastic World Model — predicts Gaussian mixture distributions of next target positions.

Instead of a single point prediction, outputs a mixture of K Gaussians per target:
  Each mode k: (mean_x, mean_y, std_x, std_y, weight_k)

This creates a "probability region" around each target, representing where it might go.
Camera coverage is measured as the fraction of this probability region within the camera FOV.

Shared encoder approach: prediction head branches off the Q-network's GRU hidden state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MATE observation layout constants
PRESERVED_DIM = 13
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PRIVATE = 14


class StochasticPredictionHead(nn.Module):
    """Predicts Gaussian mixture distributions for next target positions.

    Shared encoder: takes GRU hidden state as input.
    Per-agent prediction → all agents predict the same targets.

    Output per target: K modes × (mean_x, mean_y, log_std_x, log_std_y, log_weight)

    Args:
        hidden_dim: GRU hidden state dimension
        n_targets: number of targets
        n_modes: number of Gaussian modes per target (default 3)
        pred_hidden: prediction head hidden dimension
    """

    def __init__(self, hidden_dim, n_targets, n_modes=3, pred_hidden=128):
        super().__init__()
        self.n_targets = n_targets
        self.n_modes = n_modes
        # Per target: n_modes × 5 (mean_x, mean_y, log_std_x, log_std_y, log_weight)
        output_dim = n_targets * n_modes * 5
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, output_dim),
        )

    def forward(self, hidden):
        """
        Args:
            hidden: [B, hidden_dim] — GRU hidden state

        Returns:
            means: [B, n_targets, n_modes, 2] — mode centers
            stds: [B, n_targets, n_modes, 2] — mode spreads (positive)
            weights: [B, n_targets, n_modes] — mode weights (sum to 1)
        """
        out = self.net(hidden)
        out = out.view(-1, self.n_targets, self.n_modes, 5)

        means = out[..., :2]             # [B, n_targets, n_modes, 2]
        log_stds = out[..., 2:4]         # [B, n_targets, n_modes, 2]
        log_weights = out[..., 4]        # [B, n_targets, n_modes]

        stds = torch.exp(log_stds).clamp(min=0.01, max=2.0)
        weights = F.softmax(log_weights, dim=-1)

        return means, stds, weights

    def nll_loss(self, hidden, target_positions):
        """Negative log-likelihood loss of actual target positions under predicted mixture.

        Args:
            hidden: [B, hidden_dim]
            target_positions: [B, n_targets, 2] — actual next target positions

        Returns:
            nll: [B] — per-sample NLL (lower is better)
        """
        means, stds, weights = self.forward(hidden)
        # means: [B, N_t, K, 2], stds: [B, N_t, K, 2], weights: [B, N_t, K]

        # Expand target positions for broadcasting: [B, N_t, 1, 2]
        pos = target_positions.unsqueeze(2)

        # Log probability under each Gaussian mode
        # log N(x; mu, sigma) = -0.5 * ((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
        log_probs = (
            -0.5 * ((pos - means) / stds) ** 2
            - torch.log(stds)
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1)  # [B, N_t, K] — sum over x,y dimensions

        # Log mixture probability: log(sum_k w_k * N_k)
        log_mix = torch.logsumexp(
            torch.log(weights + 1e-8) + log_probs, dim=-1
        )  # [B, N_t]

        # NLL: average over targets
        nll = -log_mix.mean(dim=-1)  # [B]
        return nll

    def sample(self, hidden, n_samples=20):
        """Sample points from predicted distributions (for coverage computation).

        Args:
            hidden: [B, hidden_dim]
            n_samples: number of samples per target

        Returns:
            samples: [B, n_targets, n_samples, 2]
        """
        means, stds, weights = self.forward(hidden)
        B, N_t, K, _ = means.shape

        # Sample mode indices: [B, N_t, n_samples]
        mode_idx = torch.multinomial(
            weights.reshape(B * N_t, K), n_samples, replacement=True
        ).reshape(B, N_t, n_samples)

        # Gather means and stds for sampled modes
        mode_idx_exp = mode_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # [B, N_t, n_samples, 2]
        means_exp = means.unsqueeze(2).expand(-1, -1, n_samples, -1, -1)
        stds_exp = stds.unsqueeze(2).expand(-1, -1, n_samples, -1, -1)

        sampled_means = torch.gather(means_exp, 3, mode_idx_exp.unsqueeze(3)).squeeze(3)
        sampled_stds = torch.gather(stds_exp, 3, mode_idx_exp.unsqueeze(3)).squeeze(3)

        # Sample from Gaussian
        samples = sampled_means + sampled_stds * torch.randn_like(sampled_means)
        return samples  # [B, n_targets, n_samples, 2]


"""
Camera private state bounds (for denormalization):
  [0:2] = x, y           ∈ [-2000, 2000]
  [2]   = radius          ∈ [0, 1000]
  [3:5] = vx, vy          ∈ [-2000, 2000]  (= sight_range * cos/sin(orientation))
  [5]   = viewing_angle    ∈ [0, 180]      (degrees)
  [6]   = max_sight_range  ∈ [0, 2000]
  [7]   = rotation_step    ∈ [0, 180]
  [8]   = zooming_step     ∈ [0, 180]
"""
_CAM_LOW = torch.tensor([-2000., -2000., 0., -2000., -2000., 0., 0., 0., 0.])
_CAM_HIGH = torch.tensor([2000., 2000., 1000., 2000., 2000., 180., 2000., 180., 180.])


def _denorm_camera(state_normalized, idx):
    """Denormalize a camera state field from [-1,1] back to original range.

    normalized = 2 * (raw - low) / (high - low) - 1
    raw = (normalized + 1) / 2 * (high - low) + low
    """
    low = _CAM_LOW[idx].to(state_normalized.device)
    high = _CAM_HIGH[idx].to(state_normalized.device)
    return (state_normalized + 1.0) / 2.0 * (high - low) + low


def extract_camera_fov(state, n_agents):
    """Extract denormalized camera FOV parameters from global state.

    Args:
        state: [B, T, state_dim] — normalized global state

    Returns:
        positions: [B, T, C, 2] — real coords ∈ [-2000, 2000]
        orientations: [B, T, C] — radians
        sight_ranges: [B, T, C] — real sight range
        half_viewing_angles: [B, T, C] — half viewing angle in radians
    """
    positions = []
    orientations = []
    sight_ranges = []
    half_angles = []

    for i in range(n_agents):
        start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE

        # Denormalize each field
        x = _denorm_camera(state[:, :, start], 0)
        y = _denorm_camera(state[:, :, start + 1], 1)
        vx = _denorm_camera(state[:, :, start + 3], 3)  # sr * cos(orient)
        vy = _denorm_camera(state[:, :, start + 4], 4)  # sr * sin(orient)
        va = _denorm_camera(state[:, :, start + 5], 5)  # viewing_angle in degrees

        pos = torch.stack([x, y], dim=-1)
        orient = torch.atan2(vy, vx)           # radians
        sr = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
        half_va = va * (3.14159265 / 180.0) / 2.0  # half-angle in radians

        positions.append(pos)
        orientations.append(orient)
        sight_ranges.append(sr)
        half_angles.append(half_va)

    return (
        torch.stack(positions, dim=2),
        torch.stack(orientations, dim=2),
        torch.stack(sight_ranges, dim=2),
        torch.stack(half_angles, dim=2),
    )


def _denorm_target_pos(state, n_agents, n_targets):
    """Extract and denormalize target positions from global state.

    Returns real coords ∈ [-2000, 2000].
    """
    # Target state low/high for x,y are [-2000, 2000] (same as camera)
    target_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    positions = []
    for i in range(n_targets):
        start = target_start + i * TARGET_STATE_DIM_PRIVATE
        x = (state[:, :, start] + 1.0) / 2.0 * 4000.0 - 2000.0
        y = (state[:, :, start + 1] + 1.0) / 2.0 * 4000.0 - 2000.0
        positions.append(torch.stack([x, y], dim=-1))
    return torch.stack(positions, dim=2)


def compute_probability_coverage(camera_positions, camera_orientations,
                                  camera_sight_ranges, camera_half_angles,
                                  mode_means_denorm, mode_weights):
    """Compute coverage of predicted future region (convex hull vertices) by camera FOVs.

    Each target's future region is defined by K mode means (convex hull vertices).
    A vertex is covered if for ANY camera:
      1. distance(vertex, camera) < sight_range
      2. |angle(vertex - camera) - orientation| < half_viewing_angle

    Coverage = sum of covered vertices weighted by mode probability.

    All inputs in REAL coordinates (denormalized).

    Args:
        camera_positions: [B, C, 2] — real coords
        camera_orientations: [B, C] — radians
        camera_sight_ranges: [B, C] — real sight range
        camera_half_angles: [B, C] — half viewing angle in radians
        mode_means_denorm: [B, n_targets, K, 2] — predicted positions in real coords
        mode_weights: [B, n_targets, K] — probability per mode

    Returns:
        coverage: [B, n_targets]
    """
    B, N_t, K, _ = mode_means_denorm.shape
    C = camera_positions.shape[1]

    # Vector from each camera to each mode mean
    # mode_means: [B, N_t, K, 1, 2], cameras: [B, 1, 1, C, 2]
    diff = mode_means_denorm.unsqueeze(3) - camera_positions.unsqueeze(1).unsqueeze(2)
    dist = diff.norm(dim=-1)  # [B, N_t, K, C]

    # Check 1: within sight range
    sr = camera_sight_ranges.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
    in_range = dist < sr

    # Check 2: within viewing angle
    angle_to_point = torch.atan2(diff[..., 1], diff[..., 0])  # [B, N_t, K, C]
    cam_orient = camera_orientations.unsqueeze(1).unsqueeze(2)
    angle_diff = angle_to_point - cam_orient
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)).abs()
    half_va = camera_half_angles.unsqueeze(1).unsqueeze(2)
    in_angle = angle_diff < half_va

    # Vertex covered if both conditions met for ANY camera
    covered = (in_range & in_angle).any(dim=-1).float()  # [B, N_t, K]

    # Weighted coverage
    coverage = (covered * mode_weights).sum(dim=-1)  # [B, N_t]
    return coverage


def extract_target_positions(state, n_agents, n_targets):
    """Extract absolute target (x, y) from env global state."""
    target_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    positions = []
    for i in range(n_targets):
        start = target_start + i * TARGET_STATE_DIM_PRIVATE
        positions.append(state[:, :, start:start + 2])
    return torch.stack(positions, dim=2)


def extract_camera_positions(state, n_agents):
    """Extract absolute camera (x, y) from env global state."""
    positions = []
    for i in range(n_agents):
        start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE
        positions.append(state[:, :, start:start + 2])
    return torch.stack(positions, dim=2)
