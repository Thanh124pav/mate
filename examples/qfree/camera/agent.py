from ray.rllib.agents.qfree import QFreeTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class QFreeCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = QFreeTorchPolicy
    DEFAULT_CONFIG = config
