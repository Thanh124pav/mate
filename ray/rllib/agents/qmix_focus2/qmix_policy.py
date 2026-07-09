import copy

from ray.rllib.agents.qmix_focus.qmix_policy import QMixLoss, QMixTorchPolicy
from ray.rllib.agents.qplex_focus.qplex_policy import resolve_focus_config
from ray.rllib.agents.qplex_focus2.qplex_policy import (
    MixtureGaussianBeliefModel,
    QMCDiscreteBeliefModel,
    QPLEXFocus2Loss,
)


def _build_focus2_belief_model(policy, focus_config, algorithm_name):
    common = dict(
        state_dim=policy.env_global_state_shape,
        n_agents=policy.n_agents,
        n_targets=int(focus_config.get("n_targets", 8)),
        horizon=int(focus_config.get("horizon", 9)),
        hidden_dim=int(focus_config.get("belief_hidden_dim", 512)),
        num_points=int(focus_config.get("qmc_num_points", 128)),
        x_range=focus_config.get("qmc_x_range", (-1000.0, 1000.0)),
        y_range=focus_config.get("qmc_y_range", (-1000.0, 1000.0)),
        seed=int(focus_config.get("qmc_seed", 0)),
    )
    belief_type = focus_config.get("belief_type", "qmc_discrete")
    if belief_type == "qmc_discrete":
        return QMCDiscreteBeliefModel(
            **common,
            soft_label_sigma=float(focus_config.get("qmc_soft_label_sigma", 100.0)),
        ).to(policy.device)
    if belief_type == "mixture_gaussian":
        return MixtureGaussianBeliefModel(
            **common,
            components=int(focus_config.get("mixture_components", 4)),
            min_std=float(focus_config.get("mixture_min_std", 25.0)),
            max_delta=float(focus_config.get("mixture_max_delta", 400.0)),
        ).to(policy.device)
    raise ValueError(f"Unknown {algorithm_name} belief_type: {belief_type}")


class QMixFocus2Loss(QMixLoss):
    def __init__(self, model, target_model, mixer, target_mixer, n_agents, n_actions,
                 double_q=True, gamma=0.99, focus_config=None, belief_model=None):
        super().__init__(
            model, target_model, mixer, target_mixer, n_agents, n_actions,
            double_q, gamma, focus_config, occupancy_model=None,
        )
        self.focus_helper = QPLEXFocus2Loss(
            None, None, None, None, n_agents, n_actions,
            focus_config=self.focus_config, belief_model=belief_model,
        )


class QMixFocus2TorchPolicy(QMixTorchPolicy):
    def __init__(self, obs_space, action_space, config):
        from ray.rllib.agents.qmix_focus2.qmix import DEFAULT_CONFIG

        config = copy.deepcopy(dict(DEFAULT_CONFIG, **config))
        bootstrap_config = copy.deepcopy(config)
        bootstrap_focus = copy.deepcopy(bootstrap_config.get("focus", {}))
        bootstrap_focus["enabled"] = False
        bootstrap_config["focus"] = bootstrap_focus
        super().__init__(obs_space, action_space, bootstrap_config)

        self.config = config
        focus_config = resolve_focus_config(self.config)
        self.occupancy_model = None
        if focus_config.get("enabled", True):
            self.occupancy_model = _build_focus2_belief_model(
                self, focus_config, "QMIX_FOCUS2"
            )

        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        if self.occupancy_model:
            self.params += list(self.occupancy_model.parameters())

        self.loss = QMixFocus2Loss(
            self.model, self.target_model, self.mixer, self.target_mixer,
            self.n_agents, self.n_actions, self.config["double_q"],
            self.config["gamma"], focus_config, self.occupancy_model,
        )
        from torch.optim import RMSprop

        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )
