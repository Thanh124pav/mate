from ray.rllib.agents.riskq import RiskQTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class RiskQCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = RiskQTorchPolicy
    DEFAULT_CONFIG = config
