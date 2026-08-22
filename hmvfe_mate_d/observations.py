"""MATE-extended discretised coordinator observation for HMVFE (variant B).

The base ``hmvfe_mate`` builder is faithful to the paper's 5 fields
``(camera id, target id, distance bin, angle bin, visibility)``. That is a clean
"HMVFE-as-published" input, but it leaves MATE-specific signal on the table:
MATE cameras also observe **obstacles** (occlusion of the camera->target ray) and
each **target's cargo** state, which the HiT-MAC MATE integration (``hit_mac/``)
feeds to its coordinator. To benchmark HMVFE on the *same information* as the
other MATE algorithms (fairest comparison + best on-MATE performance), this
variant adds two discretised fields, exploiting the FM+MoE's native ability to
absorb extra categorical fields:

    idx[7] = (camera id, target id, distance bin, angle bin,
              visibility, cargo, occlusion bin)

New fields (vs the base 5):
  * **cargo**      -- ``target.is_loaded`` in {0, 1} (0 when invisible/unknown).
  * **occlusion**  -- how clear the camera->target line of sight is of obstacles,
    binned from the min radial clearance of any obstacle to the ray (the same
    geometric occlusion measure ``hit_mac`` computes). Bin 0 = blocked/tight,
    top bin = fully clear / no obstacle on the ray.

Fairness is preserved: every field is built ONLY from each camera's own
observation + its visibility masks (targets AND obstacles), never from the
unwrapped environment state. Invisible pairs get distance/angle/occlusion bin 0
and visibility = cargo = 0.

Geometry is MATE-4v8-9 (``TERRAIN_SIZE=1000``, camera ``max_sight_range=1500``,
9 obstacles, FoV up to 180 deg); bin counts are hyperparameters.
"""

from __future__ import annotations

import numpy as np

import mate
from mate.agents import utils as agent_utils


__all__ = ['DiscretizedCoordinatorObservationBuilder']


class DiscretizedCoordinatorObservationBuilder:
    """Build HMVFE's MATE-extended ``[N_cam, N_tgt, 7]`` field-index tensor."""

    NUM_FIELDS = 7

    def __init__(
        self,
        num_cameras: int,
        num_targets: int,
        num_obstacles: int,
        num_distance_bins: int = 16,
        num_angle_bins: int = 16,
        num_occlusion_bins: int = 8,
    ) -> None:
        self.num_cameras = int(num_cameras)
        self.num_targets = int(num_targets)
        self.num_obstacles = int(num_obstacles)
        self.num_distance_bins = int(num_distance_bins)
        self.num_angle_bins = int(num_angle_bins)
        self.num_occlusion_bins = int(num_occlusion_bins)
        self.slices = mate.camera_observation_slices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )
        self.terrain_scale = float(mate.TERRAIN_SIZE)

        # Field offsets into one shared embedding table.
        self.off_camera = 0
        self.off_target = self.off_camera + self.num_cameras
        self.off_distance = self.off_target + self.num_targets
        self.off_angle = self.off_distance + self.num_distance_bins
        self.off_visible = self.off_angle + self.num_angle_bins
        self.off_cargo = self.off_visible + 2               # visibility in {0, 1}
        self.off_occlusion = self.off_cargo + 2             # cargo in {0, 1}
        self.table_size = self.off_occlusion + self.num_occlusion_bins

    @property
    def num_fields(self) -> int:
        return self.NUM_FIELDS

    def build(self, joint_observation: np.ndarray) -> np.ndarray:
        """Map the camera joint observation ``[N_cam, obs_dim]`` to field indices.

        Returned as float32 (the env pipeline stacks/casts obs to float32); the
        model casts back to long for the embedding look-up.
        """

        joint_observation = np.asarray(joint_observation, dtype=np.float64)
        out = np.zeros((self.num_cameras, self.num_targets, self.NUM_FIELDS), dtype=np.int64)
        for c, obs in enumerate(joint_observation):
            out[c] = self._camera_fields(c, obs)
        return out.astype(np.float32)

    # -- extraction ------------------------------------------------------------

    def _extract(self, observation: np.ndarray):
        camera_state = agent_utils.CameraStatePrivate(
            observation[self.slices['self_state']], index=0
        )
        opponents = observation[self.slices['opponent_states_with_mask']].reshape(
            self.num_targets, agent_utils.TargetStatePublic.DIM + 1
        )
        target_states = tuple(
            agent_utils.TargetStatePublic(state=row[:-1], index=t)
            for t, row in enumerate(opponents)
        )
        target_mask = opponents[:, -1].astype(bool)

        if self.num_obstacles > 0:
            obstacles = observation[self.slices['obstacle_states_with_mask']].reshape(
                self.num_obstacles, agent_utils.ObstacleState.DIM + 1
            )
            obstacle_states = tuple(
                agent_utils.ObstacleState(state=row[:-1], index=o)
                for o, row in enumerate(obstacles)
            )
            obstacle_mask = obstacles[:, -1].astype(bool)
        else:
            obstacle_states, obstacle_mask = (), np.zeros((0,), dtype=bool)

        return camera_state, target_states, target_mask, obstacle_states, obstacle_mask

    def _camera_fields(self, camera_index: int, observation: np.ndarray) -> np.ndarray:
        camera_state, target_states, target_mask, obstacle_states, obstacle_mask = self._extract(
            observation
        )
        camera_loc = np.asarray(camera_state.location, dtype=np.float64)
        max_range = max(float(camera_state.max_sight_range), 1.0)

        fields = np.zeros((self.num_targets, self.NUM_FIELDS), dtype=np.int64)
        for t, target_state in enumerate(target_states):
            visible = bool(target_mask[t])
            fields[t, 0] = self.off_camera + camera_index
            fields[t, 1] = self.off_target + t

            if not visible:
                fields[t, 2] = self.off_distance    # bin 0 == unknown / out of sight
                fields[t, 3] = self.off_angle
                fields[t, 4] = self.off_visible + 0
                fields[t, 5] = self.off_cargo + 0
                fields[t, 6] = self.off_occlusion   # bin 0 == unknown
                continue

            offset = np.asarray(target_state.location, dtype=np.float64) - camera_loc
            distance = float(np.linalg.norm(offset))
            bearing = mate.normalize_angle(
                mate.arctan2_deg(offset[-1], offset[0]) - camera_state.orientation
            )
            d_ratio = min(distance / max_range, 1.0 - 1e-9)
            a_ratio = min(abs(bearing) / 180.0, 1.0 - 1e-9)

            fields[t, 2] = self.off_distance + int(d_ratio * self.num_distance_bins)
            fields[t, 3] = self.off_angle + int(a_ratio * self.num_angle_bins)
            fields[t, 4] = self.off_visible + 1
            fields[t, 5] = self.off_cargo + int(bool(target_state.is_loaded))
            fields[t, 6] = self.off_occlusion + self._occlusion_bin(
                camera_loc, offset, distance, obstacle_states, obstacle_mask, max_range
            )

        return fields

    # -- occlusion (adapted from hit_mac's obstacle features) ------------------

    def _occlusion_bin(
        self, camera_loc, target_offset, target_distance, obstacle_states, obstacle_mask, max_range
    ) -> int:
        """Bin the min obstacle clearance along the camera->target ray.

        Clearance = radial distance of an obstacle centre to the ray minus its
        radius, over obstacles whose projection falls between the camera and the
        target. Negative => the ray is blocked; large => clear. Mapped to a ratio
        in [0, 1) and binned; the top bin means "no obstacle on the ray".
        """

        if self.num_obstacles == 0 or not np.any(obstacle_mask) or target_distance <= 1e-8:
            return self.num_occlusion_bins - 1  # nothing in the way -> fully clear

        ray_dir = np.asarray(target_offset, dtype=np.float64) / target_distance
        min_clearance = np.inf
        for obstacle_state, is_visible in zip(obstacle_states, obstacle_mask):
            if not is_visible:
                continue
            obstacle_offset = np.asarray(obstacle_state.location, dtype=np.float64) - camera_loc
            projection = float(np.dot(obstacle_offset, ray_dir))
            if 0.0 <= projection <= target_distance:
                radial = float(np.linalg.norm(obstacle_offset - projection * ray_dir))
                min_clearance = min(min_clearance, radial - float(obstacle_state.radius))

        if not np.isfinite(min_clearance):
            return self.num_occlusion_bins - 1  # no obstacle intersects the ray

        ratio = np.clip(min_clearance / max_range, 0.0, 1.0 - 1e-9)  # <0 (blocked) -> 0
        return int(ratio * self.num_occlusion_bins)
