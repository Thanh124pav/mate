from pathlib import Path

from examples.value_factorization.camera.train import main_for
from .config import config


def main():
    return main_for("RESQ_FOCUSS", "RESQ_FOCUSS", config, __package__, Path(__file__).parent / "ray_results")
