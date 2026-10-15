from ray.rllib.agents.qfree import QFreeFocusTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class QFreeFocusCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = QFreeFocusTorchPolicy
    DEFAULT_CONFIG = config
