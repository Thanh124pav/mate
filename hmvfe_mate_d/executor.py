"""Geometric low-level executor, vendored from ``examples/hrl/wrappers.py``.

This is a verbatim copy of the ``track`` / ``executor`` logic inside MATE's
``HierarchicalCamera``. It is duplicated here (rather than imported) only because
importing anything under ``examples`` drags in RLlib/Ray via
``examples.utils``. **Keep this in sync with ``HierarchicalCamera.track`` so the
low-level controller stays byte-identical to the other HRL baselines** -- that
identity is what makes the benchmark fair.

Given a per-(camera, target) selection and the camera's current view mask, each
camera turns/zooms toward the centroid of the targets it both *selected* and can
*see*; if that set is empty it falls back to ``action_space.low`` (exactly as the
canonical wrapper does).
"""

from __future__ import annotations

import numpy as np

import mate


__all__ = ['track', 'joint_executor']


def track(camera, targets):
    """Primitive 2-DOF action that points ``camera`` at the centroid of ``targets``."""

    if len(targets) == 0:
        return camera.action_space.low

    center = np.mean([target.location for target in targets], axis=0)

    def best_orientation():
        direction = center - camera.location
        return mate.arctan2_deg(direction[-1], direction[0])

    def best_viewing_angle():
        distance = np.linalg.norm(center - camera.location)

        if distance * (1.0 + mate.sin_deg(camera.min_viewing_angle / 2.0)) >= camera.max_sight_range:
            return camera.min_viewing_angle

        area_product = camera.viewing_angle * np.square(camera.sight_range)
        if distance <= np.sqrt(area_product / 180.0) / 2.0:
            return min(180.0, mate.MAX_CAMERA_VIEWING_ANGLE)

        best = min(180.0, mate.MAX_CAMERA_VIEWING_ANGLE)
        for _ in range(20):
            sight_range = distance * (1.0 + mate.sin_deg(min(best / 2.0, 90.0)))
            best = area_product / np.square(sight_range)
        return np.clip(best, a_min=camera.min_viewing_angle, a_max=mate.MAX_CAMERA_VIEWING_ANGLE)

    return np.asarray(
        [
            mate.normalize_angle(best_orientation() - camera.orientation),
            best_viewing_angle() - camera.viewing_angle,
        ]
    ).clip(min=camera.action_space.low, max=camera.action_space.high)


def joint_executor(selection, view_mask, cameras, targets):
    """Map a ``[N_cam, N_tgt]`` selection to primitive camera actions ``[N_cam, 2]``.

    ``selection`` and ``view_mask`` are boolean arrays of shape ``[N_cam, N_tgt]``.
    A target is only tracked if it is both selected and currently visible.
    """

    selection = np.asarray(selection, dtype=np.bool8)
    view_mask = np.asarray(view_mask, dtype=np.bool8)

    actions = []
    for c, camera in enumerate(cameras):
        target_bits = np.logical_and(selection[c], view_mask[c])
        selected = [targets[t] for t in np.flatnonzero(target_bits)]
        actions.append(track(camera, selected))

    return np.asarray(actions, dtype=np.float64)
