"""PLINDER data access for the CG-PLI experiment.

The MLSB subset of the PLINDER test split is 346 single-ligand systems. We
only need the holo reference: the receptor heavy atoms and the reference
ligand. Everything is fetched anonymously over HTTPS from the public bucket;
no gcloud client and no GCP project are required.
"""

from __future__ import annotations

import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

PLINDER_BASE = "https://storage.googleapis.com/plinder/2024-06/v2"
MLSB_MANIFEST = f"{PLINDER_BASE}/splits/mlsb/inputs.parquet"

# Contacts closer than this in the reference complex define the support that
# lDDT-PLI is computed over. 6.0 A is the OpenStructure / PLINDER default.
INCLUSION_RADIUS = 6.0
# lDDT tolerance thresholds, averaged over. Also the OST default.
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass
class System:
    """One holo protein-ligand complex, reduced to what scoring needs."""

    system_id: str
    receptor_coords: np.ndarray  # (N, 3) heavy-atom coordinates
    receptor_pdb: Path | None  # kept on disk for PoseBusters' protein checks
    ligand: Chem.Mol  # reference ligand with a 3D conformer, Hs removed

    def pocket_centroid(self) -> np.ndarray:
        """Centroid of receptor atoms lining the reference ligand.

        This is the point a degenerate 'blackhole' predictor collapses onto.
        Using the pocket rather than the ligand centroid keeps that predictor
        honest: it only needs to know where the pocket is, which is
        information a real pocket-conditioned predictor also has.
        """
        lig = self.ligand.GetConformer().GetPositions()
        d = np.linalg.norm(
            self.receptor_coords[:, None, :] - lig[None, :, :], axis=-1
        )
        lining = self.receptor_coords[d.min(axis=1) < 8.0]
        if len(lining) == 0:
            return lig.mean(axis=0)
        return lining.mean(axis=0)


def _cache_path(cache_dir: Path, name: str) -> Path:
    p = cache_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str, dest: Path, timeout: int = 900) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    logger.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def load_manifest(cache_dir: Path) -> pd.DataFrame:
    """The MLSB subset manifest: 346 single-ligand PLINDER test systems."""
    dest = _cache_path(cache_dir, "plinder/inputs.parquet")
    _download(MLSB_MANIFEST, dest)
    df = pd.read_parquet(dest)
    if "system_id" not in df.columns:
        raise RuntimeError(
            f"unexpected manifest schema, columns={list(df.columns)}"
        )
    return df.sort_values("system_id").reset_index(drop=True)


def _shard_for(system_id: str) -> str:
    """systems/ is sharded into ~1060 zips by the 2nd-3rd char of system_id."""
    return system_id[1:3]


def fetch_system(system_id: str, cache_dir: Path) -> System | None:
    """Fetch and parse one system. Returns None if it cannot be assembled."""
    shard = _shard_for(system_id)
    zpath = _cache_path(cache_dir, f"plinder/systems/{shard}.zip")
    try:
        _download(f"{PLINDER_BASE}/systems/{shard}.zip", zpath)
    except Exception as exc:  # noqa: BLE001 - a missing shard must not abort
        logger.warning("shard %s download failed: %s", shard, exc)
        return None

    outdir = _cache_path(cache_dir, f"plinder/extracted/{system_id}")
    rec_pdb = outdir / "receptor.pdb"
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.startswith(system_id + "/")]
            if not names:
                logger.warning("%s not present in shard %s", system_id, shard)
                return None
            rec_name = next(
                (n for n in names if n.endswith("/receptor.pdb")), None
            )
            sdf_names = sorted(
                n for n in names if "/ligand_files/" in n and n.endswith(".sdf")
            )
            if rec_name is None or not sdf_names:
                logger.warning("%s missing receptor.pdb or ligand sdf", system_id)
                return None
            outdir.mkdir(parents=True, exist_ok=True)
            rec_pdb.write_bytes(zf.read(rec_name))
            sdf_bytes = zf.read(sdf_names[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s extraction failed: %s", system_id, exc)
        return None

    mol = Chem.MolFromMolBlock(sdf_bytes.decode("utf-8"), removeHs=True)
    if mol is None or mol.GetNumConformers() == 0 or mol.GetNumAtoms() < 4:
        logger.warning("%s ligand unusable", system_id)
        return None

    coords = _receptor_heavy_atoms(rec_pdb)
    if coords is None or len(coords) < 10:
        logger.warning("%s receptor unusable", system_id)
        return None

    return System(system_id, coords, rec_pdb, mol)


def _receptor_heavy_atoms(pdb_path: Path) -> np.ndarray | None:
    try:
        st = gemmi.read_structure(str(pdb_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gemmi failed on %s: %s", pdb_path, exc)
        return None
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    pts = [
        [a.pos.x, a.pos.y, a.pos.z]
        for model in st
        for chain in model
        for res in chain
        for a in res
    ]
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float64)


def synthetic_system(seed: int = 0) -> System:
    """A self-contained system for sanity mode: no network, no PLINDER.

    A small amide ligand is embedded with ETKDG and wrapped in a shell of
    pseudo-receptor atoms placed just outside contact range, so the reference
    pose has a non-empty contact support and the degenerate predictors have
    somewhere to collapse to.
    """
    rng = np.random.default_rng(seed)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)NCCO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed + 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("ETKDG failed to embed the sanity-mode ligand")
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    lig = mol.GetConformer().GetPositions()
    centre = lig.mean(axis=0)
    # Shell of receptor atoms 4.5-7.0 A from the ligand centroid, which puts
    # most of them inside the 6.0 A contact radius of some ligand atom.
    direction = rng.normal(size=(220, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radii = rng.uniform(4.5, 7.0, size=(220, 1))
    receptor = centre + direction * radii
    return System(f"synthetic-{seed}", receptor, None, mol)


def build_systems(mode: str, data_cfg: dict, cache_dir: Path) -> list[System]:
    """Assemble the evaluation set for a mode.

    sanity is fully synthetic so it runs locally on CPU with no network.
    pilot and full pull real PLINDER systems.
    """
    if mode == "sanity":
        n = int(data_cfg.get("n_systems_sanity", 4))
        return [synthetic_system(seed=i) for i in range(n)]

    n = int(
        data_cfg.get("n_systems_pilot", 20)
        if mode == "pilot"
        else data_cfg.get("n_systems_full", 346)
    )
    manifest = load_manifest(cache_dir)
    wanted = manifest["system_id"].tolist()

    # Fetch in shard order so each zip is downloaded once and reused.
    wanted.sort(key=lambda s: (_shard_for(s), s))
    wanted = wanted[: int(n)]

    # Prefetch the shards concurrently. Serial fetching of the ~7 s shards was
    # taking over half an hour for the full subset, which alone exceeded the
    # executor's activity heartbeat timeout; the downloads are latency-bound,
    # not bandwidth-bound, so a modest pool collapses that to a few minutes.
    shards = sorted({_shard_for(s) for s in wanted})
    logger.info("prefetching %d shards for %d systems", len(shards), len(wanted))
    workers = int(cfg_workers) if (cfg_workers := data_cfg.get("download_workers", 12)) else 12
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download,
                f"{PLINDER_BASE}/systems/{sh}.zip",
                _cache_path(cache_dir, f"plinder/systems/{sh}.zip"),
            ): sh
            for sh in shards
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - a missing shard is survivable
                logger.warning("shard %s failed: %s", futures[fut], exc)
            if done % 25 == 0:
                logger.info("prefetched %d/%d shards", done, len(shards))

    systems: list[System] = []
    for sid in wanted:
        s = fetch_system(sid, cache_dir)
        if s is not None:
            systems.append(s)
    logger.info("assembled %d systems for mode=%s", len(systems), mode)
    if not systems:
        raise RuntimeError("no PLINDER systems could be assembled")
    return systems


def reference_contacts(system: System) -> tuple[np.ndarray, np.ndarray]:
    """Receptor/ligand index pairs in contact, and their reference distances.

    This is the support C_ref that lDDT-PLI is computed over. It is a property
    of the reference complex alone, so it is identical for every predictor,
    which is what makes the per-predictor scores comparable.
    """
    lig = system.ligand.GetConformer().GetPositions()
    d = np.linalg.norm(
        system.receptor_coords[:, None, :] - lig[None, :, :], axis=-1
    )
    pi, li = np.nonzero(d < INCLUSION_RADIUS)
    return np.stack([pi, li], axis=1), d[pi, li]


def resolve_cache_dir(raw: str | os.PathLike[str]) -> Path:
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
