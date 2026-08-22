import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.focus_common import (
    FOCUS_CONFIDENCE,
    FOCUS_GAIN,
    FOCUS_RHO,
    FOCUS_RHO_ENTROPY,
    FOCUS_TOTAL_GAIN,
    FOCUS_VALID,
    FOCUS_WEIGHT,
)
from ray.rllib.evaluation.postprocessing import (
    compute_gae_for_sample_batch,
    Postprocessing,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import (
    EntropyCoeffSchedule,
    LearningRateSchedule,
    TorchPolicy,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class PPOTorchPolicy(TorchPolicy, LearningRateSchedule, EntropyCoeffSchedule):
    """PyTorch policy class used with PPOTrainer."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        setup_config(self, observation_space, action_space, config)

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            batch = compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
            add_focus = getattr(self.model, "add_focus_to_trajectory", None)
            if callable(add_focus):
                batch = add_focus(self, batch, other_agent_batches, episode)
            return batch

    # TODO: Add method to Policy base class (as the new way of defining loss
    #  functions (instead of passing 'loss` to the super's constructor)).
    @override(TorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        if FOCUS_WEIGHT in train_batch:
            focus_weight = train_batch[FOCUS_WEIGHT].detach().to(
                dtype=surrogate_loss.dtype, device=surrogate_loss.device
            )
            if FOCUS_VALID in train_batch:
                focus_valid = train_batch[FOCUS_VALID].bool().to(surrogate_loss.device)
            else:
                focus_valid = torch.ones_like(focus_weight, dtype=torch.bool)
            effective_focus_weight = torch.where(
                focus_valid, focus_weight, torch.ones_like(focus_weight)
            )
        else:
            effective_focus_weight = torch.ones_like(surrogate_loss)

        weighted_surrogate_loss = effective_focus_weight * surrogate_loss
        mean_native_policy_loss = reduce_mean_valid(-surrogate_loss)
        mean_policy_loss = reduce_mean_valid(-weighted_surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            vf_loss_clipped = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(
            -weighted_surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_native_policy_loss"] = mean_native_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function()
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        if FOCUS_WEIGHT in train_batch:
            focus_values = (
                effective_focus_weight[mask] if mask is not None else effective_focus_weight
            )
            model.tower_stats["focus_weight_mean"] = focus_values.mean()
            model.tower_stats["focus_weight_std"] = focus_values.std(unbiased=False)
            model.tower_stats["focus_weight_min"] = focus_values.min()
            model.tower_stats["focus_weight_max"] = focus_values.max()
            model.tower_stats["focus_actor_loss_delta"] = (
                mean_policy_loss - mean_native_policy_loss
            )
            if FOCUS_VALID in train_batch:
                focus_valid_float = train_batch[FOCUS_VALID].float().to(logp_ratio.device)
                valid_values = focus_valid_float[mask] if mask is not None else focus_valid_float
                model.tower_stats["focus_valid_fraction"] = valid_values.mean()
            if FOCUS_CONFIDENCE in train_batch:
                confidence_values = train_batch[FOCUS_CONFIDENCE].float().to(logp_ratio.device)
                confidence_values = confidence_values[mask] if mask is not None else confidence_values
                model.tower_stats["focus_confidence_mean"] = confidence_values.mean()
            if FOCUS_GAIN in train_batch:
                gain_values = train_batch[FOCUS_GAIN].float().to(logp_ratio.device)
                gain_values = gain_values[mask] if mask is not None else gain_values
                model.tower_stats["focus_gain_mean"] = gain_values.mean()
            if FOCUS_TOTAL_GAIN in train_batch:
                total_gain_values = train_batch[FOCUS_TOTAL_GAIN].float().to(logp_ratio.device)
                total_gain_values = total_gain_values[mask] if mask is not None else total_gain_values
                model.tower_stats["focus_total_gain_mean"] = total_gain_values.mean()
            if FOCUS_RHO in train_batch:
                rho_values = train_batch[FOCUS_RHO].float().to(logp_ratio.device)
                rho_values = rho_values[mask] if mask is not None else rho_values
                model.tower_stats["focus_rho_min"] = rho_values.min()
                model.tower_stats["focus_rho_max"] = rho_values.max()
            if FOCUS_RHO_ENTROPY in train_batch:
                entropy_values = train_batch[FOCUS_RHO_ENTROPY].float().to(logp_ratio.device)
                entropy_values = entropy_values[mask] if mask is not None else entropy_values
                model.tower_stats["focus_rho_entropy"] = entropy_values.mean()

        return total_loss

    def _value(self, **input_dict):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if self.config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            input_dict = self._lazy_tensor_dict(input_dict)
            model_out, _ = self.model(input_dict)
            # [0] = remove the batch dim.
            return self.model.value_function()[0].item()
        # When not doing GAE, we do not require the value function's output.
        else:
            return 0.0

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        # Return the current KL value.
        return self.kl_coeff

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_actions_computed").
    @override(TorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicy)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_losses_computed").
    @override(TorchPolicy)
    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = {
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": self.cur_lr,
            "total_loss": torch.mean(torch.stack(self.get_tower_stats("total_loss"))),
            "policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_policy_loss"))
            ),
            "native_policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_native_policy_loss"))
            ),
            "vf_loss": torch.mean(torch.stack(self.get_tower_stats("mean_vf_loss"))),
            "vf_explained_var": torch.mean(
                torch.stack(self.get_tower_stats("vf_explained_var"))
            ),
            "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
            "entropy": torch.mean(torch.stack(self.get_tower_stats("mean_entropy"))),
            "entropy_coeff": self.entropy_coeff,
        }
        optional_focus_stats = {
            "focus/weight_mean": "focus_weight_mean",
            "focus/weight_std": "focus_weight_std",
            "focus/weight_min": "focus_weight_min",
            "focus/weight_max": "focus_weight_max",
            "focus/confidence_mean": "focus_confidence_mean",
            "focus/valid_fraction": "focus_valid_fraction",
            "focus/gain_mean": "focus_gain_mean",
            "focus/total_gain_mean": "focus_total_gain_mean",
            "focus/rho_entropy": "focus_rho_entropy",
            "focus/rho_max": "focus_rho_max",
            "focus/rho_min": "focus_rho_min",
            "focus/native_actor_loss": "mean_native_policy_loss",
            "focus/weighted_actor_loss": "mean_policy_loss",
            "focus/actor_loss_delta": "focus_actor_loss_delta",
            "focus/action_prior_loss": "focus_action_prior_loss",
            "focus/action_prior_coeff": "focus_action_prior_coeff",
            "focus/action_prior_target_entropy": "focus_action_prior_target_entropy",
            "focus/action_prior_weight_mean": "focus_action_prior_weight_mean",
            "focus/action_prior_positive_fraction": "focus_action_prior_positive_fraction",
            "focus/belief_state_loss": "focus_belief_state_loss",
            "focus/belief_state_coeff": "focus_belief_state_coeff",
            "focus/belief_state_mae": "focus_belief_state_mae",
            "focus/belief_loss": "focus_belief_loss",
            "focus/beta_belief": "focus_beta_belief",
        }
        for public_key, tower_key in optional_focus_stats.items():
            try:
                values = self.get_tower_stats(tower_key)
            except AssertionError:
                continue
            if values:
                stats[public_key] = torch.mean(torch.stack(values))
        return convert_to_numpy(stats)

    # TODO: Make lr-schedule and entropy-schedule Plugin-style functionalities
    #  that can be added (via the config) to any Trainer/Policy.
    @override(TorchPolicy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )

    @override(TorchPolicy)
    def get_state(self) -> Union[Dict[str, TensorType], List[TensorType]]:
        state = super().get_state()
        # Add current kl-coeff value.
        state["current_kl_coeff"] = self.kl_coeff
        return state

    @override(TorchPolicy)
    def set_state(self, state: dict) -> None:
        # Set current kl-coeff value first.
        self.kl_coeff = state.pop("current_kl_coeff", self.config["kl_coeff"])
        # Call super's set_state with rest of the state dict.
        super().set_state(state)
