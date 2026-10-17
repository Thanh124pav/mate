from ray.rllib.agents.riskq import RiskQFocusTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class RiskQFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = RiskQFocusTorchPolicy
    DEFAULT_CONFIG = config
