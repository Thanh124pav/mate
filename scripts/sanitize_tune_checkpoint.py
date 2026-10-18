#!/usr/bin/env python3
"""Copy a legacy Tune checkpoint while moving nested torch tensors to CPU."""

import argparse
import pickle
import shutil
from pathlib import Path

import torch


def cpu_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return type(value)((key, cpu_safe(item)) for key, item in value.items())
    if isinstance(value, list):
        return [cpu_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_safe(item) for item in value)
    if isinstance(value, bytes):
        try:
            nested = pickle.loads(value)
        except Exception:
            return value
        return pickle.dumps(cpu_safe(nested), protocol=pickle.HIGHEST_PROTOCOL)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payloads = [
        path
        for path in args.source.glob("checkpoint-*")
        if path.is_file() and not path.name.endswith(".tune_metadata")
    ]
    if len(payloads) != 1:
        raise ValueError(f"expected one payload in {args.source}, found {payloads}")
    source_payload = payloads[0]
    state = pickle.loads(source_payload.read_bytes())
    args.destination.mkdir(parents=True, exist_ok=True)
    destination_payload = args.destination / source_payload.name
    destination_payload.write_bytes(
        pickle.dumps(cpu_safe(state), protocol=pickle.HIGHEST_PROTOCOL)
    )
    for suffix in (".tune_metadata",):
        companion = source_payload.with_name(source_payload.name + suffix)
        if companion.is_file():
            shutil.copy2(companion, args.destination / companion.name)
    marker = args.source / ".is_checkpoint"
    if marker.exists():
        shutil.copy2(marker, args.destination / marker.name)
    result_history = args.source.parent / "result.json"
    if result_history.is_file():
        shutil.copy2(result_history, args.destination.parent / result_history.name)
    print(destination_payload)


if __name__ == "__main__":
    main()
