from ray.rllib.utils.framework import try_import_torch


torch, _ = try_import_torch(error=True)


class MAPPOFocusAdapter:
    """Maps FOCUS responsibility to mean-preserving PPO actor weights."""

    def __init__(self, eta):
        self.eta = float(eta)

    def responsibility_to_weight(self, rho, n_agents, confidence=None):
        if confidence is None:
            eta_t = self.eta
        else:
            eta_t = self.eta * confidence.detach().unsqueeze(-1)

        return 1.0 - eta_t + eta_t * int(n_agents) * rho.detach()
