"""Extract the parameter key schema from the generated Hydra run config.

Used to hand the paper-extraction agent (run after the code-generation agent) a fixed list
of parameter names to look up in the paper, without exposing it to the actual config values
(run_id/optuna are excluded — they aren't paper-reportable experiment parameters).

Usage:
  python3 extract_config_keys.py --config <REPRO_DIR>/config/run/reproduction.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_EXCLUDED_KEYS = {"run_id", "optuna"}


def extract_keys(config_path: Path) -> list[str]:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        return []
    return [k for k in data if k not in _EXCLUDED_KEYS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"::error::Config file not found: {args.config}", file=sys.stderr)
        raise SystemExit(1)

    for key in extract_keys(args.config):
        print(key)


if __name__ == "__main__":
    main()
