import copy

from examples.mappo.camera.agent import MAPPOCameraAgent
from examples.smpe2_focus.camera.config import config as _config
from examples.smpe2_focus.camera.config import make_env as _make_env


class SMPE2FocusCameraAgent(MAPPOCameraAgent):
    """SMPE2_FOCUS inference agent with decentralized execution by default."""

    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(
        self,
        config=None,
        checkpoint_path=None,
        make_env=_make_env,
        seed=None,
        decentralized_execution=True,
    ):
        if config is None:
            config = copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            config = copy.deepcopy(config)
        if decentralized_execution:
            focus = config["model"]["custom_model_config"].setdefault("focus", {})
            focus["action_bias_eta"] = 0.0
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            make_env=make_env,
            seed=seed,
        )
