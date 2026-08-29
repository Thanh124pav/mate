#!/usr/bin/env python3
"""Train the FOCUS-HMVFE camera coordinator on MATE.

This wrapper keeps the repo's ``examples.<algorithm>.camera.train`` shape while
delegating the ray-free implementation to ``hmvfe_mate_d``. Defaults live in
``examples.hmvfe_focus.camera.config`` and CLI flags override them.

Run:
    python -m examples.hmvfe_focus.camera.train --env-config MATE-4v8-9.yaml
"""

import dataclasses

from examples.hmvfe_focus.camera.config import config as default_config
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
    args = parse_args(prog='python -m examples.hmvfe_focus.camera.train')
    if args.eval_only:
        raise SystemExit(
            'examples.hmvfe_focus.camera.train is training-only; use '
            '`python -m hmvfe_mate_d --eval-only --load <checkpoint.pt>` for evaluation.'
        )
    config = merge_config(default_config, args)
    config.focus_enabled = True
    config.focus_strict = True
    if config.focus_mode == 'uniform':
        raise SystemExit(
            'examples.hmvfe_focus disallows --focus-mode uniform because it is a vanilla-HMVFE parity mode.'
        )
    if float(config.focus_eta) == 0.0:
        raise SystemExit('examples.hmvfe_focus disallows --focus-eta 0 because it disables FOCUS weighting.')
    if config.run_name is None:
        config.run_name = f'hmvfe-focus-{config.focus_mode}'
    if config.wandb_mode != 'disabled':
        config.wandb_project = config.wandb_project or 'mate-camera'
        config.wandb_group = config.wandb_group or f'hmvfe_focus.camera.{config.run_name}'
        config.wandb_name = config.wandb_name or config.run_name
    train_hmvfe(config)


if __name__ == '__main__':
    main()
