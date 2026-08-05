"""Orchestrator for a single run_id.

Per the repository contract this file only resolves mode overrides and hands
off to the executor as a subprocess; the scoring logic lives in
src/inference.py. The task is inference-only -- nothing is trained.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Scale per mode. Same dataset and same predictor family throughout; only the
# number of systems and the bootstrap depth shrink.
MODE_OVERRIDES: dict[str, dict[str, object]] = {
    "sanity": {"experiment.n_bootstrap": 0},
    "pilot": {"experiment.n_bootstrap": 200},
    "full": {"experiment.n_bootstrap": 1000},
}


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = str(cfg.mode)
    if mode not in MODE_OVERRIDES:
        raise SystemExit(
            f"unknown mode={mode!r}; expected one of {sorted(MODE_OVERRIDES)}"
        )

    overrides = [
        f"run={cfg.run.run_id}",
        f"mode={mode}",
        f"results_dir={cfg.results_dir}",
        # W&B stays at whatever the config or the caller set; the contract
        # defaults it to online and only an explicit override turns it off.
        f"wandb.mode={cfg.wandb.mode}",
    ]
    for key, value in MODE_OVERRIDES[mode].items():
        overrides.append(f"{key}={value}")

    # Forward whatever else the caller passed on the command line, last, so an
    # explicit override always beats the mode default.
    forwarded = [
        o
        for o in HydraConfig.get().overrides.task
        if not o.split("=", 1)[0].lstrip("+~").startswith(("run", "mode", "results_dir"))
    ]
    overrides.extend(forwarded)

    cmd = [sys.executable, "-u", "-m", "src.inference", *overrides]
    logger.info("dispatching: %s", " ".join(cmd))
    logger.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
