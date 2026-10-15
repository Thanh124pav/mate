#!/usr/bin/env python3
"""Audit whether each backbone/FOCUS pair is a controlled comparison.

Expected adapter differences (FOCUS subtree, algorithm registration name, and
FOCUS model/mixer class) are ignored.  Every other differing scalar is emitted
as a parity violation and should be either fixed or explicitly justified in the
paper protocol before training.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PAIR_MODULES = {
    "ctde-value/qplex": (
        "examples.hrl.qplex.camera.config",
        "examples.hrl.qplex_focus.camera.config",
    ),
    "ctde-value/qmix": (
        "examples.hrl.qmix.camera.config",
        "examples.hrl.qmix_focus.camera.config",
    ),
    "ctde-value/duelmix": (
        "examples.hrl.duelmix.camera.config",
        "examples.hrl.duelmix_focus.camera.config",
    ),
    "ctde-policy/mappo": (
        "examples.mappo.camera.config",
        "examples.mappo_focus.camera.config",
    ),
    "ctde-policy/tarmac": (
        "examples.tarmac.camera.config",
        "examples.tarmac_focus.camera.config",
    ),
    "ctde-policy/smpe2": (
        "examples.smpe2.camera.config",
        "examples.smpe2_focus.camera.config",
    ),
    "ctce-policy/hmvfe": (
        "examples.hmvfe.camera.config",
        "examples.hmvfe_focus.camera.config",
    ),
}


IGNORED_EXACT = {
    "env",
    "mixer",
    "model.custom_model",
}

IGNORED_PREFIXES = (
    "focus.",
    "model.custom_model_config.focus.",
    "model.custom_model_config.env_config.",
    "env_config.opponent_agent_factory",
    "output_dir",
    "run_name",
    "wandb_group",
    "wandb_name",
    "wandb_tags",
)


def stable_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"{module}.{name}"
    return repr(value)


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value[key], path))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            result.update(flatten(item, path))
        return result
    return {prefix: stable_value(value)}


def ignored(path: str) -> bool:
    if path in IGNORED_EXACT:
        return True
    if path.startswith("focus_"):
        return True
    return any(path == prefix.rstrip(".") or path.startswith(prefix) for prefix in IGNORED_PREFIXES)


def load_config(module_name: str) -> Any:
    return importlib.import_module(module_name).config


def compare(base: Any, focus: Any) -> Iterable[Dict[str, Any]]:
    left = flatten(base)
    right = flatten(focus)
    for path in sorted(set(left) | set(right)):
        if ignored(path):
            continue
        base_value = left.get(path, "<missing>")
        focus_value = right.get(path, "<missing>")
        if base_value != focus_value:
            yield {"path": path, "backbone": base_value, "focus": focus_value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/focus_parity/config_parity.json"),
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Return a non-zero status when uncontrolled differences exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: Dict[str, Any] = {}
    violations = 0
    for name, (base_module, focus_module) in PAIR_MODULES.items():
        try:
            differences = list(compare(load_config(base_module), load_config(focus_module)))
            report[name] = {
                "backbone_module": base_module,
                "focus_module": focus_module,
                "status": "pass" if not differences else "fail",
                "differences": differences,
            }
            violations += len(differences)
        except Exception as exc:  # keep the audit useful when one optional model cannot import
            report[name] = {
                "backbone_module": base_module,
                "focus_module": focus_module,
                "status": "import_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            violations += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    for name, result in report.items():
        differences = result.get("differences", [])
        print(f"{name}: {result['status']} ({len(differences)} uncontrolled differences)")
        for difference in differences:
            print(
                f"  {difference['path']}: {difference['backbone']!r} -> {difference['focus']!r}"
            )
        if "error" in result:
            print(f"  {result['error']}")

    if args.fail_on_diff and violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
