"""Atom typing and parameter assignment (the only RDKit-dependent engine part).

``assign(mol, params)`` types every atom with the SMARTS list of the parameter
set and looks every term up in its tables. Anything not found is collected
and reported in one :class:`MissingParameterError`; nothing is guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import COULOMB, System
from .parameters import (
    MissingParameterError,
    ParameterSet,
    UnsupportedAtomError,
    angle_key,
    bond_key,
    default_parameters,
    torsion_key,
)

_PATTERN_CACHE: Dict[str, object] = {}


def _pattern(smarts: str):
    from rdkit import Chem

    pat = _PATTERN_CACHE.get(smarts)
    if pat is None:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise ValueError(f"invalid SMARTS in parameter file: {smarts}")
        _PATTERN_CACHE[smarts] = pat
    return pat


def atom_types(mol, params: ParameterSet) -> List[str]:
    """Type every atom; later entries of the type list override earlier ones."""
    types: List[Optional[str]] = [None] * mol.GetNumAtoms()
    for entry in params.atom_types:
        for match in mol.GetSubstructMatches(_pattern(entry["smarts"]), maxMatches=100000):
            types[match[0]] = entry["type"]
    untyped = [
        f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i, t in enumerate(types) if t is None
    ]
    if untyped:
        raise UnsupportedAtomError(
            "PMEFF26 has no atom type for: " + ", ".join(untyped[:8])
            + ". This molecule is outside the fitted parameter set."
        )
    return types  # type: ignore[return-value]


def bond_symbol(bond) -> str:
    if bond.GetIsAromatic():
        return ":"
    return {1.0: "-", 2.0: "=", 3.0: "#"}.get(bond.GetBondTypeAsDouble(), "-")


@dataclass
class Assignment:
    """A fully parameterized molecule plus the labels the fit needs."""

    system: System
    types: List[str]
    charges: np.ndarray
    bond_keys: List[str]
    angle_keys: List[str]
    torsion_keys: List[str]  # one per proper-torsion term
    improper_centers: List[int]
    pi_bond_keys: List[str]
    # Estimated-DFT-energy offset in Hartree, or None when the parameter set
    # has no increment for one of this molecule's types.
    energy_offset: Optional[float]
    # Terms without parameters (only non-empty with allow_missing=True).
    missing: List[str]


def _small_ring(rings, i: int, j: int, k: int) -> int:
    for size in (3, 4):
        for ring in rings:
            if len(ring) == size and i in ring and j in ring and k in ring:
                return size
    return 0


def assign(mol, params: Optional[ParameterSet] = None, allow_missing: bool = False) -> Assignment:
    """Build the :class:`System` of *mol*.

    With ``allow_missing=True`` (used only by the fitting tools) unknown
    bonded terms get zero force constants and unknown charges zero increments
    instead of raising; the plugin never sets it.
    """
    from rdkit import Chem

    params = params or default_parameters()
    types = atom_types(mol, params)
    n = mol.GetNumAtoms()
    missing: List[str] = []
    table = Chem.GetPeriodicTable()
    masses = np.array([table.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()])
    nbrs = [sorted(nb.GetIdx() for nb in a.GetNeighbors()) for a in mol.GetAtoms()]
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    symbol: Dict[Tuple[int, int], str] = {}

    charges = np.array([float(a.GetFormalCharge()) for a in mol.GetAtoms()])
    bond_idx, bond_r0, bond_k, bond_keys = [], [], [], []
    pi_base = [params.pi_electrons(t) for t in types]
    # A pi-type atom with no pi-type neighbour (sulfoxide O, ...) has nothing
    # to conjugate with and stays out of the pi system.
    pi_atoms = [
        i for i, e in enumerate(pi_base)
        if e is not None and any(pi_base[j] is not None for j in nbrs[i])
    ]
    pi_local = {i: pos for pos, i in enumerate(pi_atoms)}
    pi_bond, pi_beta, pi_delta, pi_rref, pi_bond_keys = [], [], [], [], []
    pi_bond_of: Dict[Tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sym = bond_symbol(bond)
        if i in pi_local and j in pi_local:
            sym = "~"
            pkey = bond_key(types[i], types[j], sym)
            pentry = params.pi_bonds.get(pkey)
            if pentry is None:
                missing.append(f"pi bond {pkey}")
                pentry = (0.0, 0.0, 1.4)
            pi_bond_of[(i, j)] = pi_bond_of[(j, i)] = len(pi_bond)
            pi_bond.append((pi_local[i], pi_local[j]))
            pi_beta.append(pentry[0])
            pi_delta.append(pentry[1])
            pi_rref.append(pentry[2])
            pi_bond_keys.append(pkey)
        symbol[(i, j)] = symbol[(j, i)] = sym
        key = bond_key(types[i], types[j], sym)
        entry = params.bonds.get(key)
        if entry is None:
            missing.append(f"bond {key}")
            entry = (1.0, 0.0)
        bond_idx.append((i, j))
        bond_r0.append(entry[0])
        bond_k.append(entry[1])
        bond_keys.append(key)
        if types[i] != types[j]:
            inc = params.bci.get(key)
            if inc is None:
                missing.append(f"charge increment {key}")
                inc = 0.0
            first = i if types[i] < types[j] else j
            second = j if first == i else i
            charges[first] += inc
            charges[second] -= inc

    angle_idx, angle_t0, angle_k, angle_n, angle_keys = [], [], [], [], []
    for j in range(n):
        for a_pos, i in enumerate(nbrs[j]):
            for k in nbrs[j][a_pos + 1 :]:
                key = angle_key(types[i], types[j], types[k], _small_ring(rings, i, j, k))
                entry = params.angles.get(key)
                if entry is None:
                    missing.append(f"angle {key}")
                    entry = (109.5, 0.0)
                angle_idx.append((i, j, k))
                if entry[0] == "periodic":  # ["periodic", n, k]
                    angle_t0.append(0.0)
                    angle_n.append(int(entry[1]))
                    angle_k.append(entry[2])
                else:
                    angle_t0.append(math.radians(entry[0]))
                    angle_n.append(0)
                    angle_k.append(entry[1])
                angle_keys.append(key)

    tors_idx, tors_v, tors_n, tors_g, tors_imp, torsion_keys = [], [], [], [], [], []
    pi_path_idx: List[Tuple[int, int, int, int]] = []
    pi_path_bond: List[int] = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if params.is_linear(types[j]) or params.is_linear(types[k]):
            continue
        paths = [
            (i, l) for i in nbrs[j] if i != k for l in nbrs[k] if l not in (j, i)
        ]
        if (j, k) in pi_bond_of:
            for i, l in paths:
                pi_path_idx.append((i, j, k, l))
                pi_path_bond.append(pi_bond_of[(j, k)])
        for i, l in paths:
            sym = symbol[(j, k)]
            found = params.torsion(types[i], types[j], types[k], types[l], sym)
            key = torsion_key(types[i], types[j], types[k], types[l], sym)
            if found is None:
                missing.append(f"torsion {key}")
                continue
            terms, generic = found
            for period, barrier, gamma in terms:
                tors_idx.append((i, j, k, l))
                tors_v.append(barrier / len(paths) if generic else barrier)
                tors_n.append(float(period))
                tors_g.append(math.radians(gamma))
                tors_imp.append(False)
                torsion_keys.append(key)

    improper_centers = []
    for j in range(n):
        if len(nbrs[j]) != 3 or not params.is_planar(types[j]):
            continue
        barrier = params.impropers.get(types[j])
        if barrier is None:
            missing.append(f"improper {types[j]}")
            barrier = 0.0
        improper_centers.append(j)
        a, b, c = nbrs[j]
        for quad in ((a, b, j, c), (b, c, j, a), (c, a, j, b)):
            tors_idx.append(quad)
            tors_v.append(barrier / 3.0)
            tors_n.append(2.0)
            tors_g.append(math.pi)
            tors_imp.append(True)

    for t in sorted(set(types)):
        if t not in params.vdw:
            missing.append(f"vdw {t}")
    pi_occ = [pi_base[i] - mol.GetAtomWithIdx(i).GetFormalCharge() for i in pi_atoms]
    pi_alpha = [float(params.type_info[types[i]].get("pi_alpha", 0.0)) for i in pi_atoms]
    if pi_atoms and (min(pi_occ) < 0 or sum(pi_occ) % 2):
        raise UnsupportedAtomError(
            "PMEFF26: the pi system of this molecule is not closed-shell "
            f"({sum(pi_occ)} pi electrons); it is outside the parameter set."
        )
    if missing and not allow_missing:
        raise MissingParameterError(missing)

    excluded = set()
    pairs14 = set()
    for i in range(n):
        for j in nbrs[i]:
            excluded.add((min(i, j), max(i, j)))
            for k in nbrs[j]:
                if k == i:
                    continue
                excluded.add((min(i, k), max(i, k)))
                for l in nbrs[k]:
                    if l not in (i, j):
                        pairs14.add((min(i, l), max(i, l)))
    pairs14 -= excluded
    pair_idx, pair_rmin, pair_eps, pair_qq = [], [], [], []
    for i in range(n):
        ri, ei = params.vdw.get(types[i], (0.0, 0.0))
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            rj, ej = params.vdw.get(types[j], (0.0, 0.0))
            is14 = (i, j) in pairs14
            pair_idx.append((i, j))
            pair_rmin.append(ri + rj)
            pair_eps.append(math.sqrt(ei * ej) * (params.scale14_vdw if is14 else 1.0))
            pair_qq.append(
                COULOMB * charges[i] * charges[j] * (params.scale14_elec if is14 else 1.0)
            )

    def arr(values, shape=None, dtype=float):
        out = np.array(values, dtype=dtype)
        return out.reshape(shape) if shape is not None else out

    system = System(
        masses=masses,
        bond_idx=arr(bond_idx, (-1, 2), int),
        bond_r0=arr(bond_r0),
        bond_k=arr(bond_k),
        angle_idx=arr(angle_idx, (-1, 3), int),
        angle_t0=arr(angle_t0),
        angle_k=arr(angle_k),
        angle_n=arr(angle_n, None, int),
        tors_idx=arr(tors_idx, (-1, 4), int),
        tors_v=arr(tors_v),
        tors_n=arr(tors_n),
        tors_gamma=arr(tors_g),
        tors_improper=arr(tors_imp, None, bool),
        pair_idx=arr(pair_idx, (-1, 2), int),
        pair_rmin=arr(pair_rmin),
        pair_eps=arr(pair_eps),
        pair_qq=arr(pair_qq),
        pi_atoms=arr(pi_atoms, None, int),
        pi_alpha=arr(pi_alpha),
        pi_occ=arr(pi_occ),
        pi_bond=arr(pi_bond, (-1, 2), int),
        pi_beta=arr(pi_beta),
        pi_delta=arr(pi_delta),
        pi_rref=arr(pi_rref),
        pi_path_idx=arr(pi_path_idx, (-1, 4), int),
        pi_path_bond=arr(pi_path_bond, None, int),
    )
    offset: Optional[float] = None
    if params.inc_types:
        try:
            offset = sum(params.inc_types[t] for t in types) + sum(
                params.inc_bonds.get(k, 0.0) for k in bond_keys
            )
        except KeyError:
            offset = None
    return Assignment(
        system, types, charges, bond_keys, angle_keys, torsion_keys,
        improper_centers, pi_bond_keys, offset, sorted(set(missing)),
    )
