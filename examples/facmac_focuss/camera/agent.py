from ray.rllib.agents.facmac import FACMACFocusTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class FACMACFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = FACMACFocusTorchPolicy
    DEFAULT_CONFIG = config
