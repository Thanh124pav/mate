from ray.rllib.agents.facmac import FACMACFocussTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class FACMACFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = FACMACFocussTorchPolicy
    DEFAULT_CONFIG = config
