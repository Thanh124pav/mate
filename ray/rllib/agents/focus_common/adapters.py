from ray.rllib.utils.framework import try_import_torch


torch, _ = try_import_torch(error=True)


class ValueDecompositionFocusAdapter:
    """Backbone-neutral transformations for value-decomposition mixers."""

    @staticmethod
    def normalize(rho, n_agents, valid=None):
        if rho is None:
            return None
        n_agents = int(n_agents)
        if rho.size(-1) != n_agents:
            raise ValueError(
                f"expected rho last dimension {n_agents}, got {rho.size(-1)}"
            )
        eps = torch.finfo(rho.dtype).eps
        rho = rho.clamp_min(0.0)
        rho = rho / rho.sum(dim=-1, keepdim=True).clamp_min(eps)
        if valid is not None:
            valid = valid.to(device=rho.device, dtype=torch.bool)
            uniform = torch.full_like(rho, 1.0 / n_agents)
            rho = torch.where(valid.unsqueeze(-1), rho, uniform)
        return rho.detach()

    @classmethod
    def qmix_scale(cls, agent_qs, rho, n_agents=None):
        """Return per-agent QMIX/VDN scale; uniform rho leaves Q unchanged."""
        n_agents = int(n_agents if n_agents is not None else agent_qs.size(-1))
        rho = cls.normalize(rho, n_agents)
        return rho * n_agents

    @classmethod
    def duelmix_allocation(cls, rho, n_agents):
        """Return normalized positive advantage allocation for DuelMIX."""
        return cls.normalize(rho, n_agents)


class MAPPOFocusAdapter:
    """Maps FOCUS responsibility to mean-preserving PPO actor weights."""

    def __init__(self, eta):
        self.eta = float(eta)

    def responsibility_to_weight(self, rho, n_agents, confidence=None):
        if confidence is None:
            eta_t = self.eta
        else:
            eta_t = self.eta * confidence.detach().unsqueeze(-1)

        rho = ValueDecompositionFocusAdapter.normalize(rho, int(n_agents))
        return 1.0 - eta_t + eta_t * int(n_agents) * rho
