"""MAPPO+FOCUS camera agent package."""

from examples.mappo_focus.camera.agent import MAPPOFocusCameraAgent
from examples.mappo_focus.camera.config import config

CameraAgent = MAPPOFocusCameraAgent

__all__ = ["CameraAgent", "MAPPOFocusCameraAgent", "config"]
