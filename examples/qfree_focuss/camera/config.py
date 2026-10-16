from ray.rllib.agents.qfree import FOCUSS_CONFIG
from examples.value_factorization.camera.config import build_camera_config, make_env

config = build_camera_config(FOCUSS_CONFIG)
