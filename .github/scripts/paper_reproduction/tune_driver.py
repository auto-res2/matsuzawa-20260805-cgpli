"""Fixed Optuna driver for parameter tuning — reuses the original reproduction's Hydra code.

Tuning generates no code. This driver reads the original job's Hydra run config
(config/run/reproduction.yaml) for the `optuna` search space + objective, then runs the reproduction's
`src/main.py` with Hydra CLI parameter overrides for n_trials, re-runs the best trial, and writes
tuning_figure.png + result.json under the tuning job's outputs/.

Invocation of the reproduction (same Hydra entry point as the run workflow):
  <python> -u -m src.main run=reproduction run.<name>=<value> ... results_dir=<dir>
  → writes <dir>/result.json with "metrics": [{"name": ..., "value": ...}, ...]

Usage:
  python tune_driver.py --original-run-dir . --tune-output-dir .reproduction/tuning \
      --python .venv/bin/python [--n-trials 20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import yaml  # noqa: E402


def _load_config(original_run_dir: Path) -> tuple[dict, str, str, int]:
    cfg = yaml.safe_load(
        (original_run_dir / "config" / "run" / "reproduction.yaml").read_text()
    )
    optuna_cfg = (cfg or {}).get("optuna") or {}
    search_space = optuna_cfg.get("search_space") or {}
    objective = optuna_cfg.get("objective")
    direction = optuna_cfg.get("direction", "maximize")
    n_trials = int(optuna_cfg.get("n_trials", 20))
    if not search_space or not objective:
        raise SystemExit(
            "config/run/reproduction.yaml must declare optuna.search_space and optuna.objective"
        )
    _validate_search_space(search_space)
    return search_space, objective, direction, n_trials


def _validate_search_space(search_space: dict) -> None:
    for name, spec in search_space.items():
        typ = spec.get("type", "float")
        if typ == "categorical":
            if "choices" not in spec:
                raise SystemExit(
                    f"optuna.search_space.{name}: type=categorical requires 'choices'"
                )
        elif "low" not in spec or "high" not in spec:
            raise SystemExit(
                f"optuna.search_space.{name}: type={typ} requires 'low' and 'high'"
            )


def _suggest(trial: optuna.Trial, name: str, spec: dict) -> Any:
    typ = spec.get("type", "float")
    if typ == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    low, high, log = spec["low"], spec["high"], bool(spec.get("log", False))
    if typ == "int":
        return trial.suggest_int(name, int(low), int(high), log=log)
    return trial.suggest_float(name, float(low), float(high), log=log)


def _run_experiment(
    python: str, original_run_dir: Path, overrides: dict, out_dir: Path
) -> Path:
    """Run src/main.py with Hydra overrides; return the path to the trial's result.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-u", "-m", "src.main", "run=reproduction"]
    cmd += [f"run.{name}={value}" for name, value in overrides.items()]
    cmd += [f"results_dir={out_dir.resolve()}"]
    result = subprocess.run(cmd, cwd=original_run_dir, check=False)
    if result.returncode != 0:
        print(f"::warning::src.main exited with code {result.returncode} (cwd={out_dir})")
    return out_dir / "result.json"


def _read_metric(result_path: Path, metric: str) -> float | None:
    try:
        data = json.loads(result_path.read_text())
    except FileNotFoundError:
        print(f"::warning::{result_path} was not created (src.main likely crashed)")
        return None
    except json.JSONDecodeError as e:
        print(f"::warning::{result_path} is not valid JSON: {e}")
        return None
    for m in data.get("metrics", []):
        if isinstance(m, dict) and m.get("name") == metric:
            try:
                return float(str(m.get("value")).strip())
            except (ValueError, TypeError):
                print(
                    f"::warning::metric '{metric}' in {result_path} is not numeric: {m.get('value')!r}"
                )
                return None
    print(f"::warning::metric '{metric}' not found in {result_path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run-dir", required=True, type=Path)
    parser.add_argument("--tune-output-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, help="python that runs src.main")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    search_space, metric, direction, n_trials = _load_config(args.original_run_dir)
    if args.n_trials is not None:
        n_trials = args.n_trials
    args.tune_output_dir.mkdir(parents=True, exist_ok=True)
    trials_root = args.tune_output_dir / "trials"

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            name: _suggest(trial, name, spec) for name, spec in search_space.items()
        }
        result_path = _run_experiment(
            args.python,
            args.original_run_dir,
            overrides,
            trials_root / f"trial_{trial.number}",
        )
        value = _read_metric(result_path, metric)
        if value is None:
            raise optuna.TrialPruned()
        return value

    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    best_result = _run_experiment(
        args.python, args.original_run_dir, study.best_params, args.tune_output_dir
    )
    best_value = _read_metric(best_result, metric)

    _plot(study, metric, direction, args.tune_output_dir / "tuning_figure.png")

    summary = (
        f"## Parameter tuning\n\n"
        f"| Item | Value |\n|---|---|\n"
        f"| Method | Optuna (TPE) |\n| Trials | {len(study.trials)} |\n"
        f"| Objective | {metric} ({direction}) |\n"
        f"| Best trial | #{study.best_trial.number} → {study.best_value} |\n\n"
        f"## Best parameters\n\n```json\n{json.dumps(study.best_params, indent=2)}\n```\n\n"
        f"## Final performance\n\n{metric} = {best_value} (best params re-run)\n"
    )
    (args.tune_output_dir / "result.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "artifacts": ["tuning_figure.png"],
                "best_params": study.best_params,
                "best_value": study.best_value,
                "objective": {"metric": metric, "direction": direction},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"best_trial=#{study.best_trial.number} best_value={study.best_value}")


def _plot(study: optuna.Study, metric: str, direction: str, path: Path) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far, cur = [], None
    for v in values:
        cur = v if cur is None else (max(cur, v) if direction == "maximize" else min(cur, v))
        best_so_far.append(cur)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(values)), values, "o", alpha=0.5, label="trial")
    ax.plot(range(len(best_so_far)), best_so_far, "-", label="running best")
    ax.set_xlabel("trial")
    ax.set_ylabel(metric)
    ax.set_title(f"Optuna tuning ({direction} {metric})")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
