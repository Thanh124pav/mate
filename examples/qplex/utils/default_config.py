from typing import Type

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.simple_q import SimpleQTrainer
from ray.rllib.agents.qplex.qplex_policy import QPLEXTorchPolicy
from ray.rllib.agents.qplex.qplex import DEFAULT_CONFIG
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import (
    SimpleReplayBuffer,
    Replay,
    StoreToReplayBuffer,
)
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator


class QPlexTrainer(SimpleQTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(SimpleQTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        if config["framework"] != "torch":
            raise ValueError("Only `framework=torch` supported so far for QMixTrainer!")

    @override(SimpleQTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return QPLEXTorchPolicy

    @staticmethod
    @override(SimpleQTrainer)
    def execution_plan(
        workers: WorkerSet, config: TrainerConfigDict, **kwargs
    ) -> LocalIterator[dict]:
        assert (
            len(kwargs) == 0
        ), "QMIX execution_plan does NOT take any additional parameters"

        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        replay_buffer = SimpleReplayBuffer(config["buffer_size"])

        store_op = rollouts.for_each(StoreToReplayBuffer(local_buffer=replay_buffer))

        train_op = (
            Replay(local_buffer=replay_buffer)
            .combine(
                ConcatBatches(
                    min_batch_size=config["train_batch_size"],
                    count_steps_by=config["multiagent"]["count_steps_by"],
                )
            )
            .for_each(TrainOneStep(workers))
            .for_each(
                UpdateTargetNetwork(workers, config["target_network_update_freq"])
            )
        )

        merged_op = Concurrently(
            [store_op, train_op], mode="round_robin", output_indexes=[1]
        )

        return StandardMetricsReporting(merged_op, workers, config)
