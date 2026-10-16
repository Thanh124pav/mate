"""MAPPO+FOCUSS camera agent package."""

from examples.mappo_focuss.camera.agent import MAPPOFocussCameraAgent
from examples.mappo_focuss.camera.config import config

CameraAgent = MAPPOFocussCameraAgent

__all__ = ["CameraAgent", "MAPPOFocussCameraAgent", "config"]
