"""Matched MATE camera environment/configuration for all ten comparisons."""

import copy

from gym import spaces
from ray import tune
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hrl.wrappers import DiscreteMultiSelection, HierarchicalCamera
from examples.utils import (
    CustomMetricCallback,
    FrameSkip,
    RLlibMultiAgentAPI,
    RLlibMultiAgentCentralizedTraining,
)


ENV_NAME = "mate-hrl.value-factorization.camera"


def target_agent_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    base_env = mate.make(
        env_config.get("env_id", "MultiAgentTracking-v0"),
        config=env_config.get("config"),
        **env_config.get("config_overrides", {}),
    )
    if str(env_config.get("enhanced_observation", None)).lower() != "none":
        base_env = mate.EnhancedObservation(
            base_env, team=env_config["enhanced_observation"]
        )
    control_mode = str(env_config.get("control_mode", "hierarchical")).lower()
    if control_mode == "direct":
        levels = env_config.get("discrete_levels")
        if levels is None:
            raise ValueError("direct camera control requires discrete_levels")
        base_env = mate.DiscreteCamera(base_env, levels=levels)
    elif control_mode != "hierarchical":
        raise ValueError(f"unknown camera control_mode: {control_mode}")
    target_agent = env_config.get("opponent_agent_factory", target_agent_factory)()
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)
    if "reward_coefficients" in env_config:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=env_config["reward_coefficients"],
            reduction=env_config.get("reward_reduction", "none"),
        )
    frame_skip = int(env_config.get("frame_skip", 1))
    if control_mode == "hierarchical":
        # Match the existing HRL-QPLEX interface: the learned discrete action
        # selects a subset of targets and the established analytic executor
        # converts that selection into continuous pan/zoom commands.
        multi_selection = bool(env_config.get("multi_selection", True))
        env = HierarchicalCamera(
            env, multi_selection=multi_selection, frame_skip=frame_skip
        )
        if multi_selection:
            env = DiscreteMultiSelection(env)
    elif frame_skip > 1:
        env = FrameSkip(env, frame_skip=frame_skip)
    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, "original_space", copy.deepcopy(observation_space))
    return env.with_agent_groups(
        groups={"camera": env.agent_ids},
        obs_space=observation_space,
        act_space=action_space,
    )


tune.register_env(ENV_NAME, make_env)


def build_camera_config(algorithm_config, seed=0):
    """Apply one matched MATE setup without changing algorithm objectives."""

    config = copy.deepcopy(algorithm_config)
    config.update(
        {
            "framework": "torch",
            "seed": int(seed),
            "env": ENV_NAME,
            "env_config": {
                "env_id": "MultiAgentTracking-v0",
                "config": "MATE-4v8-9.yaml",
                "config_overrides": {"reward_type": "dense"},
                "reward_coefficients": {"coverage_rate": 1.0},
                "reward_reduction": "mean",
                "control_mode": "hierarchical",
                "multi_selection": True,
                "frame_skip": 5,
                "enhanced_observation": "none",
                "opponent_agent_factory": target_agent_factory,
            },
            "disable_env_checking": True,
            "horizon": 500,
            "callbacks": CustomMetricCallback,
            "normalize_actions": True,
            "explore": True,
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 0,
            # RLlib 1.12 measures these in environment transitions rather than
            # PyMARL episodes.  1024 provides several full MATE episodes/update.
            "train_batch_size": 1024,
            "buffer_size": 500,
            # Match the established HRL-QPLEX reporting/evaluation cadence.
            # This changes neither replay sampling nor the learner objective.
            "timesteps_per_iteration": 5120,
            "learning_starts": 5000,
            "metrics_num_episodes_for_smoothing": 25,
        }
    )
    config["model"] = {
        **MODEL_DEFAULTS,
        **copy.deepcopy(algorithm_config.get("model", {})),
        "lstm_cell_size": int(
            algorithm_config.get("model", {}).get("lstm_cell_size", 64)
        ),
        "max_seq_len": 10000,
    }
    return config
