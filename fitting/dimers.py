"""Dimer reference set for refitting the van der Waals parameters.

For every pair of probe molecules, a few rigid relative orientations are
scanned over the intermolecular separation. The interaction energy is

    E_int = E(AB) - E(A) - E(B) + BSSE correction

at the reference level, where the basis-set superposition error is taken from
the dispersion-free parent functional (wB97X for wB97X-D):

    BSSE correction = [E0(A) - E0(A; ghost B)] + [E0(B) - E0(B; ghost A)]

GAMESS's empirical-dispersion code crashes on ghost atoms and the -D term of
wB97X-D cannot be switched off, so the ghost runs cannot use wB97X-D itself.
BSSE is a basis-set effect, so the parent functional is a close proxy.
Per point: ``_ab`` (dimer), ``_a`` and ``_b`` (ghost runs, E0). Per pair:
``_ma``/``_mb`` (isolated monomers) and ``_ma0``/``_mb0`` (same, E0).

Monomers are rigid, so in the force field E_int is purely the intermolecular
Lennard-Jones + Coulomb energy, which is what the vdW fit adjusts.

Job ids: ``dm_PPP`` is one pair; its calculations are
``dm_PPP_oO_dD_ab`` / ``_a`` / ``_b`` (orientation O, distance D).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

# One or two small probes per element / functional group.
PROBES = [
    "[HH]", "C", "CC", "C=C", "C#C", "N", "O", "F", "[SiH4]", "P", "S", "Cl",
    "Br", "I", "CF", "CCl", "CBr", "CI", "C=O", "CO", "CN", "N#N", "O=C=O",
    "FB(F)F",
]
PARTNER = "C"  # every probe is also paired with methane
N_ORIENTATIONS = 3
# Separation as a multiple of the closest contact in units of summed vdW radii.
SCALES = (0.85, 0.95, 1.05, 1.20, 1.40, 1.80)
SEED = 20261112


def pairs() -> list[tuple[str, str]]:
    out = [(p, p) for p in PROBES]
    out += [(p, PARTNER) for p in PROBES if p != PARTNER]
    return out


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
    ])


def _contact(xa, xb, ra, rb) -> float:
    """Smallest interatomic distance in units of the summed vdW radii."""
    d = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=2)
    return float(np.min(d / (ra[:, None] + rb[None, :])))


def dimer_geometries(mol_a, mol_b, index: int):
    """Yield (orientation, distance index, coords_a, coords_b)."""
    table = Chem.GetPeriodicTable()
    xa0 = mol_a.GetConformer().GetPositions()
    xb0 = mol_b.GetConformer().GetPositions()
    ra = np.array([table.GetRvdw(a.GetAtomicNum()) for a in mol_a.GetAtoms()])
    rb = np.array([table.GetRvdw(a.GetAtomicNum()) for a in mol_b.GetAtoms()])
    rng = np.random.RandomState(SEED + index)
    for o in range(N_ORIENTATIONS):
        xa = (xa0 - xa0.mean(axis=0)) @ _random_rotation(rng).T
        xb = (xb0 - xb0.mean(axis=0)) @ _random_rotation(rng).T
        axis = np.array([1.0, 0.0, 0.0])
        for d, scale in enumerate(SCALES):
            lo, hi = 0.0, 40.0  # bisection on the centre-to-centre shift
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _contact(xa, xb + mid * axis, ra, rb) < scale:
                    lo = mid
                else:
                    hi = mid
            yield o, d, xa, xb + hi * axis


def combined_mol(mol_a, mol_b, xa, xb):
    """One RDKit molecule holding both monomers at the given coordinates."""
    from rdkit.Geometry import Point3D

    combo = Chem.CombineMols(mol_a, mol_b)
    Chem.SanitizeMol(combo)  # CombineMols drops the ring information
    conf = combo.GetConformer()
    for i, (x, y, z) in enumerate(np.vstack([xa, xb])):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return combo
