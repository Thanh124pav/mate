from ray.rllib.agents.qfree import QFreeFocusTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class QFreeFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = QFreeFocusTorchPolicy
    DEFAULT_CONFIG = config
