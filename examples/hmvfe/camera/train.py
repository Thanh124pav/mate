#!/usr/bin/env python3
"""Train the HMVFE camera coordinator on MATE.

This module mirrors the repo's ``examples.<algorithm>.camera.train`` entry point
shape while delegating the ray-free implementation to ``hmvfe_mate_d``.

Run:
    python -m examples.hmvfe.camera.train --env-config MATE-4v8-9.yaml
"""

from hmvfe_mate_d.__main__ import build_config, parse_args
from hmvfe_mate_d.trainer import train as train_hmvfe


def main() -> None:
    args = parse_args(prog='python -m examples.hmvfe.camera.train')
    if args.eval_only:
        raise SystemExit(
            'examples.hmvfe.camera.train is training-only; use '
            '`python -m hmvfe_mate_d --eval-only --load <checkpoint.pt>` for evaluation.'
        )
    train_hmvfe(build_config(args))


if __name__ == '__main__':
    main()
