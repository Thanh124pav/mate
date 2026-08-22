"""MAPPO camera agents with FOCUS responsibility-weighted actor updates."""

from examples.mappo_focus import camera
from examples.mappo_focus.camera import MAPPOFocusCameraAgent

CameraAgent = MAPPOFocusCameraAgent

__all__ = ["CameraAgent", "MAPPOFocusCameraAgent", "camera"]
