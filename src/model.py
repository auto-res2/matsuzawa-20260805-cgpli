"""The controlled predictor family.

There is no learned model here. To ask whether a metric ranks methods
correctly we need methods whose true quality is known rather than estimated,
so the family below is constructed: each member's geometric error and its
coverage are set by us, and the a priori utility ordering follows from the
construction rather than from any measurement.

The two degenerate members matter most. `blackhole` is the classic collapse
onto the pocket centroid. `blackhole_busted` is the interesting one: it is a
chemically valid conformer whose orientation is actively chosen to maximise
lDDT-PLI. It is a literal model of an agent that optimises the headline
metric, which is the threat this study is about.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

PERTURBATION_ANGSTROMS = (0.5, 1.0, 2.0, 4.0, 8.0)
ABSTAIN_FRACTIONS = (0.25, 0.5)

# Utility a user of a docking method would assign, fixed a priori by the
# construction and NOT derived from any score computed in this experiment.
#   U = coverage * exp(-rmsd_true / 2) * chemically_valid
# The exponential is a smooth stand-in for the conventional 2 A success
# threshold. decoy/blackhole/blackhole_busted carry no information about the
# true pose beyond the pocket location, so their geometric term is 0.
# This encodes a normative claim -- a method that answers half the time is
# worth half as much, and an impossible pose is worth nothing -- which the
# paper states explicitly rather than deriving from any metric.
def _a_priori_utility() -> dict[str, float]:
    u: dict[str, float] = {"reference": 1.0}
    for k in PERTURBATION_ANGSTROMS:
        u[f"perturbed-{k:g}"] = float(np.exp(-k / 2.0))
    for p in ABSTAIN_FRACTIONS:
        u[f"abstain-{p:g}"] = float(p)
    u["decoy"] = 0.0
    u["blackhole_busted"] = 0.0
    u["blackhole"] = 0.0
    return u


A_PRIORI_UTILITY = _a_priori_utility()
POSE_PREDICTORS = (
    ["reference"]
    + [f"perturbed-{k:g}" for k in PERTURBATION_ANGSTROMS]
    + ["decoy", "blackhole_busted", "blackhole"]
)
ABSTAIN_PREDICTORS = [f"abstain-{p:g}" for p in ABSTAIN_FRACTIONS]
ALL_PREDICTORS = POSE_PREDICTORS + ABSTAIN_PREDICTORS


def _with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return out


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _fresh_conformer(mol: Chem.Mol, seed: int) -> np.ndarray | None:
    """An independently generated, chemically relaxed conformer."""
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(work, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(work, maxIters=400)
    except Exception:  # noqa: BLE001 - MMFF has no params for some ligands
        pass
    work = Chem.RemoveHs(work)
    if work.GetNumAtoms() != mol.GetNumAtoms():
        return None
    return work.GetConformer().GetPositions()


def generate_poses(
    system,
    rng: np.random.Generator,
    score_fn: Callable[[np.ndarray], float],
    n_orientations: int = 16,
) -> dict[str, Chem.Mol | None]:
    """Build every pose-emitting predictor's output for one system.

    `score_fn` maps candidate ligand coordinates to lDDT-PLI and is used only
    by `blackhole_busted`, which searches orientations to maximise it.
    """
    ref_mol = system.ligand
    ref = ref_mol.GetConformer().GetPositions()
    centre = system.pocket_centroid()
    out: dict[str, Chem.Mol | None] = {"reference": _with_coords(ref_mol, ref)}

    # Pure translation of magnitude k gives a symmetry-corrected RMSD of
    # exactly k, so the geometric error of these members is exact, not
    # approximate.
    for k in PERTURBATION_ANGSTROMS:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        out[f"perturbed-{k:g}"] = _with_coords(ref_mol, ref + direction * k)

    relaxed = _fresh_conformer(ref_mol, seed=int(rng.integers(1, 2**31 - 1)))
    if relaxed is None:
        # Fall back to the reference geometry so the predictor still emits a
        # pose; it is still placed at the pocket centroid in a random
        # orientation, so it carries no information about the true pose.
        relaxed = ref.copy()
    relaxed = relaxed - relaxed.mean(axis=0)

    # decoy: one arbitrary orientation at the pocket centroid.
    out["decoy"] = _with_coords(ref_mol, relaxed @ _random_rotation(rng).T + centre)

    # blackhole_busted: the same chemically valid conformer, but the
    # orientation is chosen to maximise lDDT-PLI. This is the adversary.
    best_coords, best_score = None, -np.inf
    for _ in range(n_orientations):
        cand = relaxed @ _random_rotation(rng).T + centre
        s = score_fn(cand)
        if s > best_score:
            best_score, best_coords = s, cand
    out["blackhole_busted"] = _with_coords(ref_mol, best_coords)

    # blackhole: every heavy atom collapsed onto the pocket centroid. The
    # jitter only avoids exactly coincident atoms, which crashes some
    # cheminformatics code; it is far below any bond length.
    collapsed = centre + rng.normal(scale=0.05, size=(ref_mol.GetNumAtoms(), 3))
    out["blackhole"] = _with_coords(ref_mol, collapsed)
    return out


def abstention_mask(
    system_ids: list[str], difficulty: np.ndarray, fraction: float
) -> dict[str, bool]:
    """Which systems an abstain-p predictor answers on.

    It answers on the easiest `fraction` of systems and emits nothing
    elsewhere. Difficulty is the ligand heavy-atom count: larger ligands have
    more rotatable degrees of freedom and are harder to place, so this is a
    predictor-side proxy that needs no access to the answer.
    """
    order = np.argsort(difficulty, kind="stable")
    n_answer = max(1, int(round(fraction * len(system_ids))))
    answered = {system_ids[i] for i in order[:n_answer]}
    return {sid: (sid in answered) for sid in system_ids}
