from ray.rllib.agents.resq import DEFAULT_CONFIG
from examples.value_factorization.camera.config import build_camera_config, make_env

config = build_camera_config(DEFAULT_CONFIG)
