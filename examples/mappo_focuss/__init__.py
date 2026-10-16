"""MAPPO camera agents with FOCUSS responsibility-weighted actor updates."""

from examples.mappo_focuss import camera
from examples.mappo_focuss.camera import MAPPOFocussCameraAgent

CameraAgent = MAPPOFocussCameraAgent

__all__ = ["CameraAgent", "MAPPOFocussCameraAgent", "camera"]
