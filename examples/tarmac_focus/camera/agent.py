import copy

from examples.tarmac.camera.agent import TarMACCameraAgent
from examples.tarmac_focus.camera.config import config as _config
from examples.tarmac_focus.camera.config import make_env as _make_env


class TarMACFocusCameraAgent(TarMACCameraAgent):
    """TarMAC Camera Agent with FOCUS responsibility-weighted actor updates."""

    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(self, config=None, checkpoint_path=None, make_env=_make_env, seed=None):
        super().__init__(
            config=config, checkpoint_path=checkpoint_path, make_env=make_env, seed=seed
        )
