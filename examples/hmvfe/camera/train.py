#!/usr/bin/env python3
"""Train the HMVFE camera coordinator on MATE.

This module mirrors the repo's ``examples.<algorithm>.camera.train`` entry point
shape while delegating the ray-free implementation to ``hmvfe_mate_d``.
Defaults live in ``examples.hmvfe.camera.config`` and CLI flags override them.

Run:
    python -m examples.hmvfe.camera.train --env-config MATE-4v8-9.yaml
"""

import dataclasses

from examples.hmvfe.camera.config import config as default_config
from hmvfe_mate_d.__main__ import parse_args
from hmvfe_mate_d.config import HMVFEConfig
from hmvfe_mate_d.trainer import train as train_hmvfe


def merge_config(base: HMVFEConfig, args) -> HMVFEConfig:
    values = base.to_dict()
    for field in dataclasses.fields(HMVFEConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    return HMVFEConfig(**values)


def main() -> None:
    args = parse_args(prog='python -m examples.hmvfe.camera.train')
    if args.eval_only:
        raise SystemExit(
            'examples.hmvfe.camera.train is training-only; use '
            '`python -m hmvfe_mate_d --eval-only --load <checkpoint.pt>` for evaluation.'
        )
    config = merge_config(default_config, args)
    config.focus_enabled = False
    config.focus_strict = False
    if config.wandb_mode != 'disabled':
        config.wandb_project = config.wandb_project or 'mate-camera'
        config.wandb_group = config.wandb_group or f'hmvfe.camera.{config.run_name or "hmvfe"}'
        config.wandb_name = config.wandb_name or config.run_name
    train_hmvfe(config)


if __name__ == '__main__':
    main()
