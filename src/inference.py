"""Single-run executor: score the predictor family, aggregate, report.

One process = one aggregation rule (one `run_id`). Every run recomputes the
per-system scores from scratch rather than reading a shared cache, so no two
run_ids can accidentally report the same numbers for the wrong reason.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign
from scipy.stats import spearmanr

from src import model as predictors
from src import preprocess

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Per-ligand scorers
# --------------------------------------------------------------------------
def lddt_pli(
    receptor: np.ndarray,
    contacts: np.ndarray,
    ref_distances: np.ndarray,
    model_coords: np.ndarray,
    thresholds: tuple[float, ...] = preprocess.LDDT_THRESHOLDS,
) -> float:
    """Fraction of reference protein-ligand contacts preserved by the pose.

    Superposition-free by construction: it compares interatomic distances,
    never absolute positions. That is why PLINDER adopted it, and also why a
    blob sitting anywhere in the pocket scores well.
    """
    if len(contacts) == 0:
        return float("nan")
    model_d = np.linalg.norm(
        receptor[contacts[:, 0]] - model_coords[contacts[:, 1]], axis=-1
    )
    delta = np.abs(model_d - ref_distances)
    return float(np.mean([np.mean(delta < t) for t in thresholds]))


def symmetry_rmsd(ref_mol: Chem.Mol, model_mol: Chem.Mol) -> float:
    """Symmetry-corrected heavy-atom RMSD, computed in place.

    No superposition is applied: the receptor is identical between reference
    and model, so the poses are already in a common frame. This is the
    BiSyRMSD quantity for the fixed-receptor case.
    """
    try:
        return float(rdMolAlign.CalcRMS(model_mol, ref_mol))
    except Exception:  # noqa: BLE001 - fall back to naive correspondence
        a = model_mol.GetConformer().GetPositions()
        b = ref_mol.GetConformer().GetPositions()
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


class ValidityChecker:
    """PoseBusters wrapper. Pure python, so architecture-neutral."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._busters: dict[str, object] = {}
        self.errors = 0

    def _buster(self, config: str):
        if config not in self._busters:
            from posebusters import PoseBusters

            self._busters[config] = PoseBusters(config=config)
        return self._busters[config]

    def check(self, mol: Chem.Mol, receptor_pdb: Path | None) -> tuple[bool, dict]:
        if not self.enabled:
            return True, {}
        config = "dock" if receptor_pdb is not None else "mol"
        try:
            buster = self._buster(config)
            kwargs = {"mol_cond": str(receptor_pdb)} if receptor_pdb else {}
            df = buster.bust([mol], **kwargs)
            row = df.iloc[0].to_dict()
            checks = {k: bool(v) for k, v in row.items() if isinstance(v, (bool, np.bool_))}
            return (all(checks.values()) if checks else False), checks
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            logger.warning("posebusters failed: %s", exc)
            return False, {}


# --------------------------------------------------------------------------
# Aggregation rules -- the object under study
# --------------------------------------------------------------------------
def aggregate(rule: str, scores: list[float | None], valid: list[bool], weights: list[float]) -> float:
    """Turn per-ligand lDDT-PLI into one number for a predictor.

    `scores[i] is None` means the predictor emitted nothing assignable for
    reference ligand i. The three rules differ only in how they treat those
    entries and the validity flags.
    """
    n_ref = len(scores)
    if n_ref == 0:
        return float("nan")

    if rule == "pli_ave":
        # plinder.eval get_average_ligand_scores: skip unassigned, ignore validity.
        vals = [s for s in scores if s is not None and np.isfinite(s)]
        return float(np.mean(vals)) if vals else float("nan")

    if rule == "pli_wave":
        # The heavy-atom-weighted variant plinder.eval also emits. Shares the
        # coverage blindness; included to show weighting is not the culprit.
        num = sum(w * s for s, w, in zip(scores, weights) if s is not None and np.isfinite(s))
        den = sum(w for s, w in zip(scores, weights) if s is not None and np.isfinite(s))
        return float(num / den) if den > 0 else float("nan")

    if rule == "cg_pli":
        # Proposed: unassigned contributes 0 (coverage gating) and each score
        # is gated by physical validity. Denominator is every reference ligand.
        total = 0.0
        for s, v in zip(scores, valid):
            if s is None or not np.isfinite(s) or not v:
                continue
            total += s
        return float(total / n_ref)

    raise ValueError(f"unknown aggregation rule: {rule}")


AGGREGATION_BY_RUN = {
    "cg_pli": "cg_pli",
    "pli_ave": "pli_ave",
    "pli_wave": "pli_wave",
}


def rank_fidelity(values: dict[str, float]) -> float:
    """Spearman between a rule's ordering and the a priori utility ordering."""
    names = [n for n in values if np.isfinite(values[n]) and n in predictors.A_PRIORI_UTILITY]
    if len(names) < 3:
        return float("nan")
    truth = [predictors.A_PRIORI_UTILITY[n] for n in names]
    got = [values[n] for n in names]
    return float(spearmanr(truth, got).statistic)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------
def score_all(systems: list, cfg: DictConfig) -> pd.DataFrame:
    """Score every (system, predictor) pair on all three axes."""
    checker = ValidityChecker(enabled=bool(cfg.experiment.get("posebusters", True)))
    n_orient = int(cfg.experiment.get("n_orientations", 16))
    rows: list[dict] = []

    for idx, system in enumerate(systems):
        contacts, ref_d = preprocess.reference_contacts(system)
        if len(contacts) == 0:
            logger.warning("%s has no reference contacts, skipping", system.system_id)
            continue
        rng = np.random.default_rng(int(cfg.experiment.get("seed", 42)) + idx)

        def score_fn(coords: np.ndarray) -> float:
            return lddt_pli(system.receptor_coords, contacts, ref_d, coords)

        poses = predictors.generate_poses(system, rng, score_fn, n_orientations=n_orient)
        n_heavy = system.ligand.GetNumAtoms()

        # Check the crystal reference first. Deposited ligands routinely fail
        # some PoseBusters checks (strained conformers fail the energy-ratio
        # test in particular), so an absolute validity gate would zero the
        # ground truth itself. The gate we actually apply is therefore
        # relative: a pose is gated out only for failing a check that the
        # reference pose passes. The absolute rate is still recorded, because
        # how often the reference fails is a result in its own right.
        ref_valid, ref_checks = checker.check(poses["reference"], system.receptor_pdb)

        for name, mol in poses.items():
            coords = mol.GetConformer().GetPositions()
            if name == "reference":
                valid, checks = ref_valid, ref_checks
            else:
                valid, checks = checker.check(mol, system.receptor_pdb)
            expected = [k for k, v in ref_checks.items() if v]
            valid_rel = all(checks.get(k, False) for k in expected) if expected else valid
            rows.append(
                {
                    "system_id": system.system_id,
                    "predictor": name,
                    "lddt_pli": score_fn(coords),
                    "bisy_rmsd": symmetry_rmsd(system.ligand, mol),
                    "pb_valid": valid,
                    "pb_valid_rel": bool(valid_rel),
                    "n_heavy_atoms": n_heavy,
                    "answered": True,
                    **{f"pb::{k}": v for k, v in checks.items()},
                }
            )
        if (idx + 1) % 20 == 0:
            logger.info("scored %d/%d systems", idx + 1, len(systems))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no systems produced any scores")

    # abstain-p reuses the reference pose but answers only on the easiest
    # fraction of systems, so it isolates coverage gaming from pose quality.
    ref = df[df.predictor == "reference"].set_index("system_id")
    sids = ref.index.tolist()
    difficulty = ref["n_heavy_atoms"].to_numpy()
    extra: list[dict] = []
    for frac in predictors.ABSTAIN_FRACTIONS:
        mask = predictors.abstention_mask(sids, difficulty, frac)
        for sid in sids:
            r = ref.loc[sid]
            answered = mask[sid]
            extra.append(
                {
                    "system_id": sid,
                    "predictor": f"abstain-{frac:g}",
                    "lddt_pli": float(r["lddt_pli"]) if answered else None,
                    "bisy_rmsd": float(r["bisy_rmsd"]) if answered else None,
                    "pb_valid": bool(r["pb_valid"]) if answered else False,
                    "pb_valid_rel": bool(r["pb_valid_rel"]) if answered else False,
                    "n_heavy_atoms": int(r["n_heavy_atoms"]),
                    "answered": answered,
                }
            )
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def summarise(df: pd.DataFrame, rule: str) -> dict:
    """Apply one aggregation rule and report every reporting axis."""
    per_predictor: dict[str, dict] = {}
    for name in predictors.ALL_PREDICTORS:
        sub = df[df.predictor == name]
        if sub.empty:
            continue
        scores = [None if not a else (None if pd.isna(s) else float(s))
                  for s, a in zip(sub["lddt_pli"], sub["answered"])]
        valid = [bool(v) for v in sub["pb_valid_rel"]]
        weights = [float(w) for w in sub["n_heavy_atoms"]]
        answered = sub["answered"].to_numpy().astype(bool)
        rmsd = sub["bisy_rmsd"].to_numpy(dtype=float)
        per_predictor[name] = {
            "value": aggregate(rule, scores, valid, weights),
            "pli_ave": aggregate("pli_ave", scores, valid, weights),
            "coverage": float(np.mean(answered)),
            "pb_valid_rate": float(np.mean([bool(v) for v, a in zip(sub["pb_valid"], answered) if a]))
            if answered.any() else 0.0,
            "pb_valid_rel_rate": float(np.mean([v for v, a in zip(valid, answered) if a]))
            if answered.any() else 0.0,
            "median_bisy_rmsd": float(np.nanmedian(rmsd[answered])) if answered.any() else float("nan"),
            "n_systems": int(len(sub)),
        }
    values = {k: v["value"] for k, v in per_predictor.items()}
    return {
        "rule": rule,
        "spearman_correlation": rank_fidelity(values),
        "per_predictor": per_predictor,
        "a_priori_utility": predictors.A_PRIORI_UTILITY,
    }


def bootstrap_spearman(df: pd.DataFrame, rule: str, n_boot: int, seed: int) -> dict:
    """Percentile CI for the rank fidelity, resampling systems."""
    sids = sorted(df.system_id.unique())
    if n_boot <= 0 or len(sids) < 3:
        return {}
    rng = np.random.default_rng(seed)
    by_sid = {s: g for s, g in df.groupby("system_id")}
    stats: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(len(sids), size=len(sids), replace=True)
        sample = pd.concat([by_sid[sids[i]] for i in pick], ignore_index=True)
        stats.append(summarise(sample, rule)["spearman_correlation"])
    arr = np.asarray([s for s in stats if np.isfinite(s)])
    if arr.size == 0:
        return {}
    return {
        "spearman_ci_low": float(np.percentile(arr, 2.5)),
        "spearman_ci_high": float(np.percentile(arr, 97.5)),
        "n_bootstrap": int(arr.size),
    }


def _emit_validation(mode: str, summary: dict, df: pd.DataFrame) -> bool:
    """Print the machine-parsed verdict lines AGENTS.md requires."""
    prefix = "SANITY" if mode == "sanity" else "PILOT"
    reasons: list[str] = []

    values = {k: v["value"] for k, v in summary["per_predictor"].items()}
    finite = [v for v in values.values() if np.isfinite(v)]
    n_ops = int(len(df))

    if not finite:
        reasons.append("missing_metrics")
    if not np.isfinite(summary["spearman_correlation"]):
        reasons.append("nonfinite_primary_metric")
    if len(set(np.round(finite, 12))) <= 1:
        reasons.append("all_predictors_identical")
    min_ops = 5 if mode == "sanity" else 50
    if n_ops < min_ops:
        reasons.append(f"too_few_operations_{n_ops}")
    # The reference pose must score essentially perfectly; if it does not,
    # the scorer itself is broken and every downstream number is meaningless.
    ref_val = summary["per_predictor"].get("reference", {}).get("pli_ave", float("nan"))
    if not (np.isfinite(ref_val) and ref_val > 0.99):
        reasons.append(f"reference_not_selfconsistent_{ref_val:.4f}")
    # CG-PLI is dominated by PLI_ave by construction; violating that is a bug.
    for name, rec in summary["per_predictor"].items():
        if summary["rule"] == "cg_pli" and np.isfinite(rec["value"]) and np.isfinite(rec["pli_ave"]):
            if rec["value"] > rec["pli_ave"] + 1e-9:
                reasons.append(f"cgpli_exceeds_pliave_{name}")
                break

    ok = not reasons
    print(f"{prefix}_VALIDATION: " + ("PASS" if ok else f"FAIL reason={','.join(reasons)}"), flush=True)
    payload = {
        "operations": n_ops,
        "status": "ok" if ok else "failed",
        "primary_metric": "spearman_correlation",
        "primary_metric_value": summary["spearman_correlation"],
        "n_predictors": len(values),
        "reference_pli_ave": ref_val,
    }
    print(f"{prefix}_VALIDATION_SUMMARY: {json.dumps(payload, default=float)}", flush=True)
    return ok


def run(cfg: DictConfig) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = str(cfg.mode)
    rule = AGGREGATION_BY_RUN[str(cfg.run.aggregation)]
    results_dir = Path(str(cfg.results_dir)) / str(cfg.run.run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = preprocess.resolve_cache_dir(str(cfg.experiment.get("cache_dir", ".cache")))

    wb = None
    if str(cfg.wandb.mode) != "disabled":
        try:
            import wandb

            project = str(cfg.wandb.project)
            if mode in ("sanity", "pilot"):
                project = f"{project}-{mode}"
            wb = wandb.init(
                entity=str(cfg.wandb.entity),
                project=project,
                name=str(cfg.run.run_id),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=str(cfg.wandb.mode),
            )
            print(f"WANDB_RUN_URL: {wb.url}", flush=True)
        except Exception as exc:  # noqa: BLE001 - never let telemetry kill a run
            logger.warning("wandb init failed, continuing offline: %s", exc)
            wb = None

    systems = preprocess.build_systems(mode, dict(cfg.experiment), cache_dir)
    df = score_all(systems, cfg)
    df.to_parquet(results_dir / "per_system_scores.parquet", index=False)

    summary = summarise(df, rule)
    n_boot = int(cfg.experiment.get("n_bootstrap", 0 if mode == "sanity" else 1000))
    summary.update(bootstrap_spearman(df, rule, n_boot, int(cfg.experiment.get("seed", 42))))
    summary["mode"] = mode
    summary["run_id"] = str(cfg.run.run_id)
    summary["n_systems"] = int(df.system_id.nunique())

    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    if wb is not None:
        flat = {"spearman_correlation": summary["spearman_correlation"],
                "n_systems": summary["n_systems"]}
        for name, rec in summary["per_predictor"].items():
            for k, v in rec.items():
                flat[f"{name}/{k}"] = v
        for k in ("spearman_ci_low", "spearman_ci_high"):
            if k in summary:
                flat[k] = summary[k]
        wb.log(flat)
        wb.summary.update(flat)
        wb.finish()

    ok = True
    if mode in ("sanity", "pilot"):
        ok = _emit_validation(mode, summary, df)
    print(f"PRIMARY_METRIC spearman_correlation={summary['spearman_correlation']:.6f}", flush=True)
    if not ok:
        sys.exit(1)
    return summary


@hydra.main(version_base=None, config_path="../config", config_name="config")
def _main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    _main()
