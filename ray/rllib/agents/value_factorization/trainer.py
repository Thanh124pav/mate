"""Replay-based RLlib trainer shared by the new CTDE policies."""

from ray.rllib.agents.dqn.simple_q import SimpleQTrainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, _get_shared_metrics
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import (
    SimpleReplayBuffer,
    StoreToReplayBuffer,
)
from ray.rllib.execution.rollout_ops import ConcatBatches, ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics


def replay_after_warmup(local_buffer, learning_starts):
    """Yield replay non-blockingly once enough environment steps exist.

    ``LocalIterator.filter(WaitUntilTimestepsElapsed(...))`` repeatedly drops
    replay samples while the predicate is false.  Since a non-empty replay
    buffer never yields ``_NextValueNotReady``, that filter can monopolize a
    round-robin concurrent iterator and prevent further rollout collection.
    """

    threshold = int(learning_starts)

    def generate(_):
        while True:
            sampled = _get_shared_metrics().counters[STEPS_SAMPLED_COUNTER]
            if sampled <= threshold:
                yield _NextValueNotReady()
                continue
            item = local_buffer.replay()
            yield item if item is not None else _NextValueNotReady()

    return LocalIterator(generate, SharedMetrics())


class ValueFactorizationTrainer(SimpleQTrainer):
    """The execution plan used by episode-replay PyMARL-style learners."""

    @override(SimpleQTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)
        if config["framework"] != "torch":
            raise ValueError("Only framework=torch is supported")
        if config.get("batch_mode") != "complete_episodes":
            raise ValueError("paper-faithful recurrent targets require complete episodes")

    @staticmethod
    @override(SimpleQTrainer)
    def execution_plan(
        workers: WorkerSet, config: TrainerConfigDict, **kwargs
    ) -> LocalIterator[dict]:
        if kwargs:
            raise ValueError("ValueFactorizationTrainer takes no execution-plan kwargs")
        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        replay_buffer = SimpleReplayBuffer(config["buffer_size"])
        store = rollouts.for_each(StoreToReplayBuffer(local_buffer=replay_buffer))
        train = (
            replay_after_warmup(replay_buffer, config["learning_starts"])
            .combine(
                ConcatBatches(
                    min_batch_size=config["train_batch_size"],
                    count_steps_by=config["multiagent"]["count_steps_by"],
                )
            )
            .for_each(TrainOneStep(workers))
        )
        merged = Concurrently([store, train], mode="round_robin", output_indexes=[1])
        return StandardMetricsReporting(merged, workers, config)
