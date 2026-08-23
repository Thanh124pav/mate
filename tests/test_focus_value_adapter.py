"""Tests for the backbone-neutral value-decomposition FOCUS adapter."""

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_adapter_module():
    framework = types.ModuleType("ray.rllib.utils.framework")
    framework.try_import_torch = lambda error=False: (torch, torch.nn)
    for name in ("ray", "ray.rllib", "ray.rllib.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["ray.rllib.utils.framework"] = framework
    path = ROOT / "ray/rllib/agents/focus_common/adapters.py"
    spec = importlib.util.spec_from_file_location("focus_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ValueDecompositionFocusAdapter


ValueDecompositionFocusAdapter = _load_adapter_module()


def test_qmix_uniform_scale_is_identity():
    values = torch.tensor([[[1.0, -2.0, 3.0]]])
    rho = torch.full((1, 1, 3), 1.0 / 3.0)
    scale = ValueDecompositionFocusAdapter.qmix_scale(values, rho)
    assert torch.allclose(scale, torch.ones_like(values))


def test_qmix_scale_normalizes_nonuniform_responsibility():
    values = torch.zeros((2, 4, 3))
    rho = torch.tensor([[[2.0, 1.0, 1.0]]] * 2).expand(2, 4, 3)
    scale = ValueDecompositionFocusAdapter.qmix_scale(values, rho)
    assert torch.allclose(scale[0, 0], torch.tensor([1.5, 0.75, 0.75]))
    assert torch.allclose(scale.mean(dim=-1), torch.ones((2, 4)))


def test_invalid_responsibility_falls_back_to_uniform():
    rho = torch.tensor([[[0.9, 0.1]]])
    valid = torch.tensor([[False]])
    normalized = ValueDecompositionFocusAdapter.normalize(rho, 2, valid=valid)
    assert torch.allclose(normalized, torch.full_like(rho, 0.5))
    assert not normalized.requires_grad
