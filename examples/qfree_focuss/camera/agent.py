from ray.rllib.agents.qfree import QFreeFocussTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class QFreeFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = QFreeFocussTorchPolicy
    DEFAULT_CONFIG = config
