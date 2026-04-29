"""DuelMix + RSSM World Model (WM2) Trainer."""

from typing import Type

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.simple_q import SimpleQTrainer
from ray.rllib.agents.duelmix_wm2.duelmix_policy import DuelMixWM2TorchPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import SimpleReplayBuffer, Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator

DEFAULT_CONFIG = with_common_config({
    "mixer": "duelmix_wm2",
    "mixing_embed_dim": 64,
    "double_q": True,
    "batch_mode": "complete_episodes",
    "world_model_v2": {
        "stoch_dim": 32,
        "deter_dim": 128,
        "hidden_dim": 128,
        "action_embed_dim": 16,
        "embed_dim": 128,
        "imagination_horizon": 5,
        "kl_coeff": 1.0,
        "free_nats": 1.0,
        "wm_loss_weight": 0.5,
        "reward_bonus_coeff": 0.1,
        "reward_bonus_scale": 0.5,
        "use_imagination_targets": False,
        "imagination_loss_weight": 0.1,
        "ema_decay": 0.995,
    },
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "epsilon_timesteps": 40000,
    },
    "evaluation_interval": None,
    "evaluation_duration": 10,
    "evaluation_config": {"explore": False},
    "timesteps_per_iteration": 1000,
    "target_network_update_freq": 500,
    "buffer_size": 1000,
    "lr": 0.0005,
    "optim_alpha": 0.99,
    "optim_eps": 0.00001,
    "grad_norm_clipping": 10,
    "learning_starts": 1000,
    "rollout_fragment_length": 4,
    "train_batch_size": 32,
    "num_workers": 0,
    "worker_side_prioritization": False,
    "min_time_s_per_reporting": 1,
    "model": {"lstm_cell_size": 64, "max_seq_len": 999999},
    "framework": "torch",
})


class DuelMixWM2Trainer(SimpleQTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(SimpleQTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)
        if config["framework"] != "torch":
            raise ValueError("Only framework=torch supported for DuelMixWM2Trainer!")

    @override(SimpleQTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return DuelMixWM2TorchPolicy

    @staticmethod
    @override(SimpleQTrainer)
    def execution_plan(workers: WorkerSet, config: TrainerConfigDict, **kwargs) -> LocalIterator[dict]:
        assert len(kwargs) == 0
        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        replay_buffer = SimpleReplayBuffer(config["buffer_size"])
        store_op = rollouts.for_each(StoreToReplayBuffer(local_buffer=replay_buffer))
        train_op = (
            Replay(local_buffer=replay_buffer)
            .combine(ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            ))
            .for_each(TrainOneStep(workers))
            .for_each(UpdateTargetNetwork(workers, config["target_network_update_freq"]))
        )
        merged_op = Concurrently([store_op, train_op], mode="round_robin", output_indexes=[1])
        return StandardMetricsReporting(merged_op, workers, config)
