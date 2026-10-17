from ray.rllib.agents.riskq import FOCUS_CONFIG
from examples.value_factorization.camera.config import build_camera_config, make_env

config = build_camera_config(FOCUS_CONFIG)
