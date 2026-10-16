from ray.rllib.agents.riskq import RiskQFocussTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class RiskQFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = RiskQFocussTorchPolicy
    DEFAULT_CONFIG = config
