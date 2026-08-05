"""Independent cross-run aggregation and figures.

Not called from main.py. Reads finished runs from the W&B API by display
name, falling back to the on-disk metrics.json a run leaves behind so the
comparison still works when W&B was unavailable.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "spearman_correlation"
# Higher rank fidelity is better, so the proposed method wins by exceeding
# the best baseline.
HIGHER_IS_BETTER = True


def _from_wandb(run_id: str, entity: str, project: str) -> dict | None:
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(
            f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
        )
        if not len(runs):
            return None
        return dict(runs[0].summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B lookup failed for %s: %s", run_id, exc)
        return None


def _from_disk(results_dir: Path, run_id: str) -> dict | None:
    p = results_dir / run_id / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_run(results_dir: Path, run_id: str, entity: str, project: str) -> dict:
    disk = _from_disk(results_dir, run_id)
    if disk is not None:
        return disk
    remote = _from_wandb(run_id, entity, project)
    if remote is None:
        raise FileNotFoundError(f"no metrics found for run_id={run_id}")
    return {
        "run_id": run_id,
        PRIMARY_METRIC: remote.get(PRIMARY_METRIC, float("nan")),
        "per_predictor": {},
    }


def _bar(ax, labels, values, title, ylabel):
    ax.bar(range(len(labels)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def per_run_figures(results_dir: Path, run_id: str, metrics: dict) -> list[Path]:
    out: list[Path] = []
    per = metrics.get("per_predictor") or {}
    if not per:
        return out
    d = results_dir / run_id
    d.mkdir(parents=True, exist_ok=True)

    names = list(per)
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, names, [per[n]["value"] for n in names],
         f"{run_id}: aggregate score per predictor", metrics.get("rule", "score"))
    fig.tight_layout()
    p = d / f"{run_id}_aggregate_by_predictor.pdf"
    fig.savefig(p); plt.close(fig); out.append(p)

    # The diagnostic plot: aggregate score against true quality. A faithful
    # metric puts every point on a monotone curve.
    util = metrics.get("a_priori_utility") or {}
    if util:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        xs = [util.get(n, np.nan) for n in names]
        ys = [per[n]["value"] for n in names]
        ax.scatter(xs, ys, color="#C44E52", zorder=3)
        for n, x, y in zip(names, xs, ys):
            ax.annotate(n, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("a priori utility (coverage x geometric quality x validity)")
        ax.set_ylabel(f"{metrics.get('rule','score')} aggregate")
        ax.set_title(f"{run_id}: rank fidelity rho={metrics.get(PRIMARY_METRIC, float('nan')):.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = d / f"{run_id}_utility_vs_score.pdf"
        fig.savefig(p); plt.close(fig); out.append(p)
    return out


def comparison_figures(results_dir: Path, loaded: dict[str, dict]) -> list[Path]:
    out: list[Path] = []
    d = results_dir / "comparison"
    d.mkdir(parents=True, exist_ok=True)

    run_ids = list(loaded)
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [loaded[r].get(PRIMARY_METRIC, np.nan) for r in run_ids]
    lo = [loaded[r].get("spearman_ci_low", np.nan) for r in run_ids]
    hi = [loaded[r].get("spearman_ci_high", np.nan) for r in run_ids]
    err = None
    if not all(np.isnan(lo)):
        err = np.abs(np.vstack([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]))
    labels = [loaded[r].get("rule", r) for r in run_ids]
    ax.bar(range(len(run_ids)), vals, yerr=err, capsize=4, color="#55A868")
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Spearman rank fidelity")
    ax.set_title("Does the aggregation rank predictors correctly?", fontsize=10)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = d / "comparison_spearman_correlation.pdf"
    fig.savefig(p); plt.close(fig); out.append(p)

    # Overlay every rule's per-predictor curve on shared axes.
    preds: list[str] = []
    for m in loaded.values():
        for n in (m.get("per_predictor") or {}):
            if n not in preds:
                preds.append(n)
    if preds:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rid in run_ids:
            per = loaded[rid].get("per_predictor") or {}
            ys = [per.get(n, {}).get("value", np.nan) for n in preds]
            ax.plot(range(len(preds)), ys, marker="o", label=loaded[rid].get("rule", rid))
        util = next((m.get("a_priori_utility") for m in loaded.values() if m.get("a_priori_utility")), None)
        if util:
            ax.plot(range(len(preds)), [util.get(n, np.nan) for n in preds],
                    marker="x", ls="--", color="black", label="a priori utility")
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels(preds, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("aggregate score")
        ax.set_title("Aggregate score per predictor, by aggregation rule", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = d / "comparison_score_by_predictor.pdf"
        fig.savefig(p); plt.close(fig); out.append(p)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("kv", nargs="*", help="key=value pairs (hydra-style)")
    args = ap.parse_args()
    kv = dict(p.split("=", 1) for p in args.kv if "=" in p)

    results_dir = Path(kv.get("results_dir", ".research/results"))
    run_ids = kv.get("run_ids", "[]")
    try:
        ids = ast.literal_eval(run_ids)
    except (ValueError, SyntaxError):
        ids = [s for s in run_ids.strip("[]").replace('"', "").split(",") if s]
    entity = kv.get("entity", "airas")
    project = kv.get("project", "cgpli-plinder")

    loaded: dict[str, dict] = {}
    written: list[Path] = []
    for rid in ids:
        m = load_run(results_dir, rid, entity, project)
        loaded[rid] = m
        d = results_dir / rid
        d.mkdir(parents=True, exist_ok=True)
        p = d / "metrics.json"
        p.write_text(json.dumps(m, indent=2, default=float))
        written.append(p)
        written.extend(per_run_figures(results_dir, rid, m))

    written.extend(comparison_figures(results_dir, loaded))

    by_run = {r: loaded[r].get(PRIMARY_METRIC, float("nan")) for r in loaded}
    proposed = {r: v for r, v in by_run.items() if r.startswith("proposed")}
    baseline = {r: v for r, v in by_run.items() if not r.startswith("proposed")}
    pick = max if HIGHER_IS_BETTER else min
    best_proposed = pick(proposed.values()) if proposed else float("nan")
    best_baseline = pick(baseline.values()) if baseline else float("nan")
    agg = {
        "primary_metric": PRIMARY_METRIC,
        "higher_is_better": HIGHER_IS_BETTER,
        "metrics": by_run,
        "rules": {r: loaded[r].get("rule", r) for r in loaded},
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": float(best_proposed - best_baseline),
        "per_predictor": {r: loaded[r].get("per_predictor", {}) for r in loaded},
        "a_priori_utility": next(
            (m.get("a_priori_utility") for m in loaded.values() if m.get("a_priori_utility")), {}
        ),
    }
    d = results_dir / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "aggregated_metrics.json"
    p.write_text(json.dumps(agg, indent=2, default=float))
    written.append(p)

    for w in written:
        print(w, flush=True)
    print(f"GAP {PRIMARY_METRIC}: {agg['gap']:+.6f} "
          f"(proposed={best_proposed:.6f} baseline={best_baseline:.6f})", flush=True)


if __name__ == "__main__":
    main()
