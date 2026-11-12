#!/usr/bin/env python3
"""Fit the PMEFF26 parameter file to the collected reference data.

    python -m fitting.fit            # reads reference/reference_data.json (+ .npz)

Writes ``pmeff26_plugin/parameters/pmeff26.json`` and ``reference/fit_report.md``.
Deterministic and non-iterative, so it can run unattended:

1. **Charges** - bond-charge increments by linear least squares on the DFT
   dipole moments, lightly restrained to the Lowdin charges.
2. **Bonded terms** - the energy is linear in (k, k*x0) for bonds and angles
   and in the barriers of torsions and impropers, so one bounded linear
   least-squares problem matches, for every training molecule at its DFT
   minimum, the force-field gradient to zero and the force-field Hessian to
   the DFT Hessian (after subtracting the fixed non-bonded part).
3. **Torsion scans** - for every scanned bond class the three Fourier
   barriers are refit to the relaxed-scan energy profiles.
4. **Energy increments** - per-type and per-bond constants that reference the
   strain energy to the DFT total energy.

A term whose data do not determine it is left out of the file (and listed in
the report); molecules that need it stay unsupported. Nothing is guessed.
Only training molecules are used; validation molecules appear only in the
report. The pi-electron term is not fitted by this script (``pi.enabled``
stays false), so bonds are keyed by formal bond order.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, lsq_linear

from pmeff26_plugin.assign import assign
from pmeff26_plugin.core import LINEAR_THRESHOLD, _pi_term
from pmeff26_plugin.core import (
    System,
    energy_and_gradient,
    frequencies_from_hessian,
    hessian,
    optimize,
)
from pmeff26_plugin.parameters import (
    DEFAULT_FILE,
    MissingParameterError,
    ParameterError,
    ParameterSet,
)

from . import level

LEVEL = "wB97X-D/def2-TZVP (GAMESS)"
REPO = Path(__file__).resolve().parents[1]
REF = level.REF
if level.NAME:
    LEVEL = f"{level.NAME}/def2-TZVP (GAMESS)"
HARTREE_KCAL = 627.5095
DEBYE_PER_EA = 4.80321
W_GRAD = 20.0  # weight of gradient rows relative to Hessian rows
RIDGE_TORSION = 0.05
RIDGE_SB = 0.02
SB_MAX = 250.0  # stretch-bend constants beyond this are not physical
PI_BETA0, PI_DELTA0 = 50.0, 2.0  # starting resonance integral (kcal/mol), decay (1/A)


def mol_of(record):
    """RDKit molecule of a record with the DFT geometry as its conformer."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(record["coords"]):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol


def generic(tkey: str) -> str:
    return "*|" + tkey.split("|")[1] + "|*"


def provisional(skeleton: dict, mols: list, use_pi: bool = False) -> ParameterSet:
    """Skeleton plus a unit placeholder for every term the molecules need."""
    data = copy.deepcopy(skeleton)
    data["pi"] = {"enabled": use_pi, "bonds": {}}
    for table in ("bonds", "angles", "torsions", "impropers", "bci"):
        data[table] = {}
    data["increments"] = {"types": {}, "bonds": {}}
    for mol in mols:
        a = assign(mol, ParameterSet(data), allow_missing=True)
        for item in a.missing:
            kind, key = item.rsplit(" ", 1)
            if kind == "bond":
                data["bonds"][key] = [1.0, 0.0]
            elif kind == "pi bond":
                data["pi"]["bonds"][key] = [0.0, PI_DELTA0, 1.4]  # beta 0: term off
            elif kind == "angle":
                data["angles"][key] = [100.0, 0.0]
            elif kind == "torsion":
                data["torsions"][generic(key)] = [[1, 1.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 0.0]]
            elif kind == "improper":
                data["impropers"][key] = 1.0
            elif kind == "charge increment":
                data["bci"][key] = 0.0
    return ParameterSet(data)


# --- 1. charges -------------------------------------------------------------


def fit_charges(train: list, prov: ParameterSet) -> dict:
    keys = sorted(prov.bci)
    col = {k: n for n, k in enumerate(keys)}
    rows, rhs = [], []
    for rec, mol in train:
        a = assign(mol, prov, allow_missing=True)
        x = np.array(rec["coords"])
        n = len(x)
        inc = np.zeros((n, len(keys)))  # dq_i / d(bci_key)
        for (i, j), key in zip(a.system.bond_idx, a.bond_keys):
            if a.types[i] == a.types[j] or key not in col:
                continue
            first, second = (i, j) if a.types[i] < a.types[j] else (j, i)
            inc[first, col[key]] += 1.0
            inc[second, col[key]] -= 1.0
        formal = np.array([at.GetFormalCharge() for at in mol.GetAtoms()], float)
        if rec.get("dipole") and n > 1:
            centred = x - x.mean(axis=0)
            rows.append(DEBYE_PER_EA * centred.T @ inc)
            rhs.append(np.array(rec["dipole"]) - DEBYE_PER_EA * centred.T @ formal)
        if rec.get("lowdin"):
            rows.append(0.3 * inc)
            rhs.append(0.3 * (np.array(rec["lowdin"]) - formal))
    if not keys or not rows:
        return {}
    a_mat, b_vec = np.vstack(rows), np.concatenate(rhs)
    a_mat = np.vstack([a_mat, 1e-3 * np.eye(len(keys))])
    b_vec = np.concatenate([b_vec, np.zeros(len(keys))])
    sol = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    used = np.abs(np.vstack(rows)).sum(axis=0) > 0
    return {k: round(float(v), 5) for k, v, u in zip(keys, sol, used) if u}


# --- 1b. van der Waals --------------------------------------------------------


def vdw_class(type_name: str, params: ParameterSet) -> str:
    """Types sharing one Lennard-Jones pair: the element, or polar hydrogen."""
    return "hx" if type_name == "hx" else params.type_info[type_name]["element"]


def fit_vdw(dimers: list, data: dict):
    """Refit Lennard-Jones (rmin/2, eps) per class to the dimer curves.

    Charges are held at the fitted bond-charge increments. Returns the RMSE
    (kcal/mol, over points with E_int < 5) before and after, and the classes
    that were refitted; classes without dimer data keep their UFF values.
    """
    from rdkit import Chem

    params = ParameterSet(data)
    points = []  # (class index i, class index j, r, coulomb, e_ref)
    names: dict = {}
    for rec in dimers:
        mol = Chem.MolFromMolBlock(rec["molblock"], removeHs=False)
        a = assign(mol, params, allow_missing=True)
        cls = [names.setdefault(vdw_class(t, params), len(names)) for t in a.types]
        n_a = rec["n_a"]
        ia, ib = np.meshgrid(np.arange(n_a), np.arange(n_a, len(cls)), indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        ci, cj = np.array(cls)[ia], np.array(cls)[ib]
        qq = 332.0637 * a.charges[ia] * a.charges[ib]
        for pt in rec["points"]:
            x = np.array(pt["coords"])
            r = np.linalg.norm(x[ia] - x[ib], axis=1)
            points.append((ci, cj, r, float(np.sum(qq / r)), pt["e_int"]))
    if not points:
        return None
    classes = sorted(names, key=names.get)
    first_type = {}
    for t in params.type_info:
        first_type.setdefault(vdw_class(t, params), t)
    start = np.array([params.vdw[first_type[c]] for c in classes])  # (rmin/2, eps)
    x0 = np.concatenate([start[:, 0], np.log(start[:, 1])])
    e_ref = np.array([p[4] for p in points])
    weight = 1.0 / np.sqrt(1.0 + np.maximum(e_ref, 0.0) ** 2)  # soften the wall
    n = len(classes)

    def model(v):
        rm, eps = v[:n], np.exp(v[n:])
        out = np.empty(len(points))
        for k, (ci, cj, r, coul, _e) in enumerate(points):
            s6 = ((rm[ci] + rm[cj]) / r) ** 6
            out[k] = np.sum(np.sqrt(eps[ci] * eps[cj]) * (s6 * s6 - 2.0 * s6)) + coul
        return out

    def residual(v):
        restraint = np.concatenate([0.3 * (v[:n] - x0[:n]), 0.05 * (v[n:] - x0[n:])])
        return np.concatenate([weight * (model(v) - e_ref), restraint])

    lower = np.concatenate([np.full(n, 0.4), np.full(n, math.log(1e-4))])
    upper = np.concatenate([np.full(n, 2.8), np.full(n, math.log(2.0))])
    sol = least_squares(residual, x0, bounds=(lower, upper))
    low = e_ref < 5.0

    def rmse(v):
        return float(np.sqrt(np.mean((model(v) - e_ref)[low] ** 2))) if low.any() else float("nan")

    fitted = {c: [round(float(sol.x[k]), 4), round(float(math.exp(sol.x[n + k])), 5)]
              for k, c in enumerate(classes)}
    for t in params.type_info:
        if vdw_class(t, params) in fitted:
            data["vdw"][t] = fitted[vdw_class(t, params)]
    return {"before": rmse(x0), "after": rmse(sol.x), "classes": classes, "points": len(points)}


# --- 2. bonded terms --------------------------------------------------------


def _sub(system: System, **arrays) -> System:
    """A system holding only the given term arrays (same masses)."""
    return System(masses=system.masses, **arrays)


def _grad_hess(x, system):
    _, g = energy_and_gradient(x, system)
    return g.ravel(), hessian(x, system)


def angle_classes(train: list, prov: ParameterSet) -> dict:
    """angle key -> 'harmonic' | 'linear' | 'periodic', from the DFT angles."""
    values = defaultdict(list)
    for rec, mol in train:
        a = assign(mol, prov, allow_missing=True)
        x = np.array(rec["coords"])
        for (i, j, k), key in zip(a.system.angle_idx, a.angle_keys):
            u, v = x[i] - x[j], x[k] - x[j]
            c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
            values[key].append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    out = {}
    for key, vals in values.items():
        if min(vals) >= 172.0:
            out[key] = "linear"
        elif max(vals) - min(vals) > 60.0:
            out[key] = "periodic"
        else:
            out[key] = "harmonic"
    return out


def basis_functions(x, a, classes):
    """Yield (column name, lower bound, gradient, Hessian) for one molecule."""
    s = a.system
    bkeys = np.array(a.bond_keys)
    for key in sorted(set(a.bond_keys)):
        sel = bkeys == key
        n = int(sel.sum())
        ga, ha = _grad_hess(x, _sub(s, bond_idx=s.bond_idx[sel], bond_r0=np.zeros(n), bond_k=np.ones(n)))
        g1, h1 = _grad_hess(x, _sub(s, bond_idx=s.bond_idx[sel], bond_r0=np.ones(n), bond_k=np.ones(n)))
        yield ("bond", key, "a"), 0.0, ga, ha
        yield ("bond", key, "b"), -np.inf, g1 - ga, h1 - ha
    akeys = np.array(a.angle_keys)
    for key in sorted(set(a.angle_keys)):
        sel = akeys == key
        n = int(sel.sum())
        idx = s.angle_idx[sel]
        cls = classes[key]
        if cls == "linear":
            g, h = _grad_hess(x, _sub(s, angle_idx=idx, angle_t0=np.full(n, math.pi), angle_k=np.ones(n), angle_n=np.zeros(n, int)))
            yield ("angle", key, "lin"), 0.0, g, h
        elif cls == "periodic":
            g, h = _grad_hess(x, _sub(s, angle_idx=idx, angle_t0=np.zeros(n), angle_k=np.ones(n), angle_n=np.full(n, 4)))
            yield ("angle", key, "per"), 0.0, g, h
        else:
            zero = np.zeros(n, int)
            ga, ha = _grad_hess(x, _sub(s, angle_idx=idx, angle_t0=np.zeros(n), angle_k=np.ones(n), angle_n=zero))
            g1, h1 = _grad_hess(x, _sub(s, angle_idx=idx, angle_t0=np.ones(n), angle_k=np.ones(n), angle_n=zero))
            yield ("angle", key, "a"), 0.0, ga, ha
            yield ("angle", key, "b"), -np.inf, g1 - ga, h1 - ha
    n_prop = len(a.torsion_keys)
    gkeys = np.array([generic(k) for k in a.torsion_keys])
    for key in sorted(set(gkeys)):
        for period in (1, 2, 3):
            sel = np.zeros(len(s.tors_v), bool)
            sel[:n_prop] = (gkeys == key) & (s.tors_n[:n_prop] == period)
            if not sel.any():
                continue
            g, h = _grad_hess(x, _sub(s, tors_idx=s.tors_idx[sel], tors_v=s.tors_v[sel], tors_n=s.tors_n[sel], tors_gamma=s.tors_gamma[sel], tors_improper=np.zeros(int(sel.sum()), bool)))
            yield ("torsion", key, period), -np.inf, g, h
    imp_types = [a.types[c] for c in a.improper_centers for _ in range(3)]
    for t in sorted(set(imp_types)):
        sel = np.zeros(len(s.tors_v), bool)
        sel[n_prop:] = np.array(imp_types) == t
        g, h = _grad_hess(x, _sub(s, tors_idx=s.tors_idx[sel], tors_v=s.tors_v[sel], tors_n=s.tors_n[sel], tors_gamma=s.tors_gamma[sel], tors_improper=np.ones(int(sel.sum()), bool)))
        yield ("improper", t, ""), 0.0, g, h


def bonded_system(train: list, hessians: dict, prov: ParameterSet, classes: dict):
    """The linear system of the bonded fit: columns, bounds, matrix, target.

    ``spans`` lists (assignment, coordinates, first row) per molecule, in the
    row layout [W_GRAD * gradient, upper triangle of the Hessian].
    """
    columns: dict = {}
    lower: list = []
    spans: list = []
    rows_i, cols_j, vals, rhs = [], [], [], []
    offset = 0
    for rec, mol in train:
        if rec["id"] not in hessians:
            continue
        x = np.array(rec["coords"])
        a = assign(mol, prov, allow_missing=True)
        s = a.system
        n3 = x.size
        tri = np.triu_indices(n3)
        g_nb, h_nb = _grad_hess(x, _sub(s, pair_idx=s.pair_idx, pair_rmin=s.pair_rmin, pair_eps=s.pair_eps, pair_qq=s.pair_qq))
        target = np.concatenate([-W_GRAD * g_nb, (hessians[rec["id"]] - h_nb)[tri]])
        for name, lb, g, h in basis_functions(x, a, classes):
            if name not in columns:
                columns[name] = len(columns)
                lower.append(lb)
            vec = np.concatenate([W_GRAD * g, h[tri]])
            nz = np.nonzero(np.abs(vec) > 1e-10)[0]
            rows_i.extend(offset + nz)
            cols_j.extend([columns[name]] * len(nz))
            vals.extend(vec[nz])
        rhs.append(target)
        spans.append((a, x, offset))
        offset += len(target)
    for name, col in columns.items():  # ridge on torsion barriers
        if name[0] == "torsion":
            rows_i.append(offset)
            cols_j.append(col)
            vals.append(RIDGE_TORSION)
            rhs.append(np.zeros(1))
            offset += 1
    a_mat = sparse.csr_matrix((vals, (rows_i, cols_j)), shape=(offset, len(columns)))
    b_vec = np.concatenate(rhs)
    return columns, np.array(lower), a_mat, b_vec, spans


def _column_scaled(a_mat):
    norms = np.sqrt(np.asarray(a_mat.multiply(a_mat).sum(axis=0)).ravel())
    norms[norms == 0.0] = 1.0
    return a_mat @ sparse.diags(1.0 / norms), norms


def solve_bonded(a_mat, b_vec, lower) -> np.ndarray:
    # Scale every column to unit norm: the unknowns span four orders of
    # magnitude (bond k ~ 1e3, torsion barriers ~ 1), which stalls LSMR.
    scaled, norms = _column_scaled(a_mat)
    lb = np.array(lower) * norms  # 0 and -inf are unchanged by the scaling
    result = lsq_linear(scaled, b_vec, bounds=(lb, np.inf), lsq_solver="lsmr",
                        lsmr_tol=1e-12, tol=1e-12, max_iter=500, lsmr_maxiter=20000)
    return result.x / norms


def fit_bonded(train: list, hessians: dict, prov: ParameterSet, classes: dict):
    columns, lower, a_mat, b_vec, _spans = bonded_system(train, hessians, prov, classes)
    return columns, solve_bonded(a_mat, b_vec, lower)


# --- 2a. stretch-bend cross terms -------------------------------------------


def _local_grad_hess(x, system, atoms, delta=1e-4):
    """Gradient and Hessian of terms that involve only *atoms*."""
    hess = np.zeros((x.size, x.size))
    y = x.copy()
    for atom in atoms:
        for c in range(3):
            keep = y[atom, c]
            y[atom, c] = keep + delta
            gp = energy_and_gradient(y, system)[1].ravel()
            y[atom, c] = keep - delta
            gm = energy_and_gradient(y, system)[1].ravel()
            y[atom, c] = keep
            hess[3 * atom + c] = (gp - gm) / (2.0 * delta)
    return energy_and_gradient(x, system)[1].ravel(), 0.5 * (hess + hess.T)


def stretch_bend_columns(spans: list, bonds: dict, angles: dict, n_rows: int):
    """Basis columns of the stretch-bend constants at fixed r0 / theta0.

    One column per (angle key, slot): slot 0 is the bond to the key's first
    end type, slot 1 the bond to its second; equal end types share slot 0.
    """
    names: dict = {}
    rows, cols, vals = [], [], []
    for a, x, offset in spans:
        s = a.system
        r0_of = {}
        for (i, j), key in zip(s.bond_idx, a.bond_keys):
            if key in bonds:
                r0_of[(i, j)] = r0_of[(j, i)] = bonds[key][0]
        groups = defaultdict(list)
        for (i, j, k), key in zip(s.angle_idx, a.angle_keys):
            entry = angles.get(key)
            if entry is None or entry[0] == "periodic" or math.radians(entry[0]) >= LINEAR_THRESHOLD:
                continue
            if (i, j) not in r0_of or (k, j) not in r0_of:
                continue
            base = ((i, j, k), (r0_of[(i, j)], r0_of[(k, j)]), math.radians(entry[0]))
            ti, tk = a.types[i], a.types[k]
            if ti == tk:
                groups[(key, 0)].append(base + ((1.0, 1.0),))
            else:
                on_i, on_k = (1.0, 0.0), (0.0, 1.0)
                groups[(key, 0)].append(base + (on_i if ti < tk else on_k,))
                groups[(key, 1)].append(base + (on_k if ti < tk else on_i,))
        tri = np.triu_indices(x.size)
        for name, items in sorted(groups.items()):
            idx = np.array([it[0] for it in items])
            sub = _sub(s, sb_idx=idx, sb_r0=np.array([it[1] for it in items]),
                       sb_t0=np.array([it[2] for it in items]),
                       sb_k=np.array([it[3] for it in items]))
            g, h = _local_grad_hess(x, sub, sorted(set(idx.ravel().tolist())))
            vec = np.concatenate([W_GRAD * g, h[tri]])
            nz = np.nonzero(np.abs(vec) > 1e-10)[0]
            col = names.setdefault(name, len(names))
            rows.extend(offset + nz)
            cols.extend([col] * len(nz))
            vals.extend(vec[nz])
    return names, sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, len(names)))


def fit_stretch_bend(a_mat, b_vec, lower, columns: dict, spans: list, sol, rounds: int = 2):
    """Re-solve the bonded fit with stretch-bend columns added.

    The cross term is linear in its constants only at fixed r0 / theta0, so
    those are taken from the previous solution and the joint solve repeated.
    Returns (solution of the original columns, table, dropped terms).
    """
    n_cols = a_mat.shape[1]
    names, k_sb = {}, np.zeros(0)
    for _ in range(rounds):
        bonds, angles = tables_from_solution(columns, sol)[:2]
        names, sb_mat = stretch_bend_columns(spans, bonds, angles, a_mat.shape[0])
        if not names:
            return sol, {}, []
        n_sb = len(names)
        ridge = sparse.hstack([sparse.csr_matrix((n_sb, n_cols)), RIDGE_SB * sparse.identity(n_sb)])
        full = sparse.vstack([sparse.hstack([a_mat, sb_mat]), ridge]).tocsr()
        both = solve_bonded(full, np.concatenate([b_vec, np.zeros(n_sb)]),
                            np.concatenate([lower, np.full(n_sb, -np.inf)]))
        sol, k_sb = both[:n_cols], both[n_cols:]
    raw: dict = {}
    for (key, slot), col in names.items():
        raw.setdefault(key, [None, None])[slot] = float(k_sb[col])
    table, dropped = {}, []
    for key, (k0, k1) in raw.items():
        k1 = k0 if k1 is None else k1
        if max(abs(k0), abs(k1)) > SB_MAX:
            dropped.append(f"stretch-bend {key} (k={k0:.0f}, {k1:.0f})")
        else:
            table[key] = [round(k0, 3), round(k1, 3)]
    return sol, table, dropped


# --- 2b. pi-electron term -----------------------------------------------------


def pi_pair(params: ParameterSet, t1: str, t2: str) -> str:
    """Class of a pi bond: its two elements. All type pairs of a class share beta."""
    return "~".join(sorted((params.type_info[t1]["element"], params.type_info[t2]["element"])))


def pi_site(params: ParameterSet, t: str) -> str:
    """Class of a pi atom: element + electrons it contributes. Shares alpha."""
    info = params.type_info[t]
    return f"{info['element']}{info['pi_electrons']}"


def _pi_grad_hess(x, system, atoms, delta=1e-4):
    """Gradient and Hessian of the pi term alone (only *atoms* move it)."""
    def grad(y):
        g = np.zeros_like(y)
        _pi_term(y, system, g)
        return g.ravel()

    hess = np.zeros((x.size, x.size))
    y = x.copy()
    for atom in atoms:
        for c in range(3):
            keep = y[atom, c]
            y[atom, c] = keep + delta
            gp = grad(y)
            y[atom, c] = keep - delta
            gm = grad(y)
            y[atom, c] = keep
            hess[3 * atom + c] = (gp - gm) / (2.0 * delta)
    return grad(x), 0.5 * (hess + hess.T)


def fit_pi(prov: ParameterSet, a_mat, b_vec, spans: list, max_nfev: int = 15, log=print):
    """Fit the Hueckel pi parameters by variable projection.

    For trial pi parameters the pi gradient and Hessian are subtracted from the
    targets and the bonded terms are re-solved linearly (unbounded, by
    projection); the remaining residual is minimized. Unknowns: one resonance
    integral ``beta`` per element pair, one on-site energy ``alpha`` per
    (element, electron count) other than carbon, and one global decay
    ``delta``. ``rref`` is not fitted: it is the mean DFT length of the pi
    bonds of the pair. Returns None when no training molecule has a pi system.
    """
    mols, lengths = [], defaultdict(list)
    for a, x, offset in spans:
        s = a.system
        if not len(s.pi_bond):
            continue
        ends = s.pi_atoms[s.pi_bond]
        pairs = [pi_pair(prov, a.types[i], a.types[j]) for i, j in ends]
        for (i, j), name in zip(ends, pairs):
            lengths[name].append(float(np.linalg.norm(x[i] - x[j])))
        sites = [pi_site(prov, a.types[i]) for i in s.pi_atoms]
        moved = sorted(set(s.pi_atoms.tolist()) | set(s.pi_path_idx.ravel().tolist()))
        mols.append((s, x, offset, pairs, sites, moved, np.triu_indices(x.size)))
    if not mols:
        return None
    pair_names = sorted(lengths)
    rref = {name: float(np.mean(v)) for name, v in lengths.items()}
    site_names = sorted({c for m in mols for c in m[4]} - {"C1"})
    n_pair, n_site = len(pair_names), len(site_names)
    pair_col = {n: k for k, n in enumerate(pair_names)}
    site_col = {n: n_pair + k for k, n in enumerate(site_names)}

    def systems(p):
        for s, x, offset, pairs, sites, moved, tri in mols:
            yield _sub(
                s, pi_atoms=s.pi_atoms, pi_occ=s.pi_occ, pi_bond=s.pi_bond,
                pi_path_idx=s.pi_path_idx, pi_path_bond=s.pi_path_bond,
                pi_beta=np.array([p[pair_col[n]] for n in pairs]),
                pi_delta=np.full(len(pairs), p[-1]),
                pi_rref=np.array([rref[n] for n in pairs]),
                pi_alpha=np.array([p[site_col[c]] if c in site_col else 0.0 for c in sites]),
            ), x, offset, moved, tri

    def pi_vector(p):
        out = np.zeros(len(b_vec))
        for s, x, offset, moved, tri in systems(p):
            g, h = _pi_grad_hess(x, s, moved)
            vec = np.concatenate([W_GRAD * g, h[tri]])
            out[offset : offset + len(vec)] = vec
        return out

    scaled, _norms = _column_scaled(a_mat)
    val, vec = np.linalg.eigh((scaled.T @ scaled).toarray())
    keep = val > 1e-10 * val.max()
    val, vec = val[keep], vec[:, keep]

    def project(t):  # the part of t the bonded terms cannot absorb
        return t - scaled @ (vec @ ((vec.T @ (scaled.T @ t)) / val))

    p0 = np.concatenate([np.full(n_pair, PI_BETA0), np.zeros(n_site), [PI_DELTA0]])
    lower = np.concatenate([np.full(n_pair, 1.0), np.full(n_site, -80.0), [0.5]])
    upper = np.concatenate([np.full(n_pair, 300.0), np.full(n_site, 80.0), [5.0]])
    # Weak pull to the start values settles parameters the data do not determine.
    pull = np.concatenate([np.full(n_pair, 0.02), np.full(n_site, 0.05), [1.0]])

    def residual(p):
        return np.concatenate([project(b_vec - pi_vector(p)), pull * (p - p0)])

    log(f"pi fit: {len(mols)} molecules, {len(p0)} parameters")
    sol = least_squares(residual, p0, bounds=(lower, upper), diff_step=1e-3,
                        x_scale=np.concatenate([np.full(n_pair, 20.0), np.full(n_site, 10.0), [0.5]]),
                        max_nfev=max_nfev)
    p = sol.x
    n = len(b_vec)
    return {
        "beta": {name: float(p[pair_col[name]]) for name in pair_names},
        "rref": rref, "delta": float(p[-1]),
        "alpha": {c: float(p[site_col[c]]) for c in site_names},
        "vector": pi_vector(p), "molecules": len(mols),
        "before": float(np.sqrt(np.mean(project(b_vec) ** 2))),
        "after": float(np.sqrt(np.mean(sol.fun[:n] ** 2))),
    }


def tables_from_solution(columns: dict, sol: np.ndarray):
    bonds, angles, torsions, impropers, dropped = {}, {}, defaultdict(list), {}, []
    val = {name: float(sol[col]) for name, col in columns.items()}
    for (kind, key, part), v in val.items():
        if kind == "bond" and part == "a":
            b = val.get(("bond", key, "b"), 0.0)
            if v < 20.0 or not 0.5 < b / v < 3.5:
                dropped.append(f"bond {key} (k={v:.1f})")
            else:
                bonds[key] = [round(b / v, 4), round(v, 2)]
        elif kind == "angle" and part == "a":
            b = val.get(("angle", key, "b"), 0.0)
            theta = math.degrees(b / v) if v > 1e-9 else 0.0
            if v < 2.0 or not 50.0 < theta < 180.0:
                dropped.append(f"angle {key} (k={v:.1f})")
            else:
                angles[key] = [round(theta, 3), round(v, 2)]
        elif kind == "angle" and part == "lin":
            if v < 1.0:
                dropped.append(f"angle {key} (linear, k={v:.2f})")
            else:
                angles[key] = [180.0, round(v, 2)]
        elif kind == "angle" and part == "per":
            if v < 1.0:
                dropped.append(f"angle {key} (periodic, k={v:.2f})")
            else:
                angles[key] = ["periodic", 4, round(v, 2)]
        elif kind == "torsion":
            torsions[key].append([part, round(abs(v), 4), 0.0 if v >= 0 else 180.0])
        elif kind == "improper":
            impropers[key] = round(v, 3)
    return bonds, angles, {k: sorted(v) for k, v in torsions.items()}, impropers, dropped


# --- 3. torsion scans -------------------------------------------------------


def fit_scans(scans: list, data: dict) -> dict:
    """Refit generic torsion barriers of scanned bond classes; return RMSEs."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    groups = defaultdict(list)
    for rec in scans:
        mol = Chem.MolFromMolBlock(rec["molblock"], removeHs=False)
        try:
            a = assign(mol, ParameterSet(data))
        except ParameterError:
            continue
        j, k = rec["dihedral"][1], rec["dihedral"][2]
        n_prop = len(a.torsion_keys)
        idx = a.system.tors_idx[:n_prop]
        on_bond = ((idx[:, 1] == j) & (idx[:, 2] == k)) | ((idx[:, 1] == k) & (idx[:, 2] == j))
        keys = {generic(a.torsion_keys[t]) for t in np.nonzero(on_bond)[0]}
        if len(keys) != 1:
            continue
        key = keys.pop()
        paths = np.unique(idx[on_bond], axis=0)
        rest = copy.copy(a.system)
        rest.tors_v = a.system.tors_v.copy()
        rest.tors_v[:n_prop][on_bond] = 0.0
        e_dft, e_rest, basis = [], [], []
        from pmeff26_plugin.core import _dihedral

        for pt in rec["points"]:
            x = np.array(pt["coords"])
            e_dft.append(pt["energy"] * HARTREE_KCAL)
            e_rest.append(energy_and_gradient(x, rest)[0])
            phi = _dihedral(x, paths)[0]
            basis.append([np.mean(0.5 * (1.0 + np.cos(n * phi))) for n in (1, 2, 3)])
        y = np.array(e_dft) - np.array(e_rest)
        groups[key].append((np.array(basis), y - y.mean(), rec["id"]))
    rmse = {}
    for key, items in groups.items():
        rows, ys = [], []
        for n_scan, (basis, y, _id) in enumerate(items):
            const = np.zeros((len(y), len(items)))
            const[:, n_scan] = 1.0
            rows.append(np.hstack([basis, const]))
            ys.append(y)
        a_mat, y_all = np.vstack(rows), np.concatenate(ys)
        sol = np.linalg.lstsq(a_mat, y_all, rcond=None)[0]
        data["torsions"][key] = [
            [n, round(abs(float(v)), 4), 0.0 if v >= 0 else 180.0]
            for n, v in zip((1, 2, 3), sol[:3])
        ]
        rmse[key] = float(np.sqrt(np.mean((a_mat @ sol - y_all) ** 2)))
    return rmse


# --- 4. energy increments ---------------------------------------------------


def fit_increments(train: list, data: dict) -> float | None:
    params = ParameterSet(data)
    names: dict = {}
    rows, ys = [], []
    for rec, mol in train:
        try:
            a = assign(mol, params)
        except ParameterError:
            continue
        e_ff = energy_and_gradient(np.array(rec["coords"]), a.system)[0]
        counts = defaultdict(float)
        for t in a.types:
            counts[("t", t)] += 1.0
        for k in a.bond_keys:
            counts[("b", k)] += 1.0
        for name in counts:
            names.setdefault(name, len(names))
        rows.append(counts)
        ys.append(rec["energy"] - e_ff / HARTREE_KCAL)
    if not rows:
        return None
    a_mat = np.zeros((len(rows), len(names)))
    for r, counts in enumerate(rows):
        for name, c in counts.items():
            a_mat[r, names[name]] = c
    y = np.array(ys)
    # Tiny ridge on the bond columns picks the solution carried by atom types.
    reg = np.diag([0.0 if name[0] == "t" else 1e-4 for name in names])
    sol = np.linalg.lstsq(np.vstack([a_mat, reg]), np.concatenate([y, np.zeros(len(names))]), rcond=None)[0]
    data["increments"] = {
        "types": {n[1]: float(sol[c]) for n, c in names.items() if n[0] == "t"},
        "bonds": {n[1]: float(sol[c]) for n, c in names.items() if n[0] == "b"},
    }
    return float(np.sqrt(np.mean((a_mat @ sol - y) ** 2))) * HARTREE_KCAL


# --- report -----------------------------------------------------------------


def evaluate(items: list, hessians: dict, params: ParameterSet) -> dict:
    """Optimize each molecule from its DFT geometry and compare with DFT."""
    bond_err, angle_err, freq_err, covered, refused = [], [], [], 0, []
    for rec, mol in items:
        try:
            a = assign(mol, params)
        except ParameterError as exc:
            refused.append((rec["id"], rec["smiles"], getattr(exc, "missing", [str(exc)])[:3]))
            continue
        covered += 1
        x0 = np.array(rec["coords"])
        if len(x0) < 2:
            continue
        x1, _ = optimize(x0, a.system, max_iter=5000, f_tol=1e-3)
        for i, j in a.system.bond_idx:
            bond_err.append(np.linalg.norm(x1[i] - x1[j]) - np.linalg.norm(x0[i] - x0[j]))

        def ang(x, i, j, k):
            u, v = x[i] - x[j], x[k] - x[j]
            return math.degrees(math.acos(max(-1.0, min(1.0, np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))))

        for i, j, k in a.system.angle_idx:
            angle_err.append(ang(x1, i, j, k) - ang(x0, i, j, k))
        if rec["id"] in hessians and len(x0) > 2:
            f_dft = frequencies_from_hessian(hessians[rec["id"]], a.system.masses)
            f_ff = frequencies_from_hessian(hessian(x1, a.system), a.system.masses)
            n_vib = max(len(f_dft) - 6, 0)
            if n_vib:
                freq_err.extend(np.sort(f_ff)[-n_vib:] - np.sort(f_dft)[-n_vib:])

    def rms(v):
        return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")

    return {"n": len(items), "covered": covered, "refused": refused,
            "bond_rmsd": rms(bond_err), "angle_rmsd": rms(angle_err), "freq_rmsd": rms(freq_err)}


def fit(records: list, hessians: dict, skeleton: dict, use_pi: bool = True,
        use_sb: bool = True):
    """Return (parameter dict, report text)."""
    usable = [r for r in records if r["usable"]]
    probe = copy.deepcopy(skeleton)
    probe["pi"] = {"enabled": use_pi, "bonds": {}}
    probe, untyped, mols = ParameterSet(probe), [], {}
    for r in usable:
        if r["kind"] in ("train", "validate"):
            try:
                mol = mol_of(r)
                assign(mol, probe, allow_missing=True)
                mols[r["id"]] = mol
            except ParameterError as exc:
                untyped.append((r, str(exc)))
    train = [(r, mols[r["id"]]) for r in usable if r["kind"] == "train" and r["id"] in mols]
    valid = [(r, mols[r["id"]]) for r in usable if r["kind"] == "validate" and r["id"] in mols]
    scans = [r for r in usable if r["kind"] == "scan"]
    dimers = [r for r in usable if r["kind"] == "dimer"]
    if not train:
        raise SystemExit("no usable training molecules in the reference data")

    prov = provisional(skeleton, [m for _r, m in train], use_pi)
    bci = fit_charges(train, prov)
    prov.data["bci"] = {k: bci.get(k, 0.0) for k in prov.bci}
    vdw_stats = fit_vdw(dimers, prov.data)  # updates prov.data["vdw"] in place
    prov = ParameterSet(prov.data)
    classes = angle_classes(train, prov)
    columns, lower, a_mat, b_vec, spans = bonded_system(train, hessians, prov, classes)
    pi_fit = fit_pi(prov, a_mat, b_vec, spans) if use_pi else None
    target = b_vec - pi_fit["vector"] if pi_fit else b_vec
    sol = solve_bonded(a_mat, target, lower)
    cross, cross_dropped = {}, []
    if use_sb:
        sol, cross, cross_dropped = fit_stretch_bend(a_mat, target, lower, columns, spans, sol)
    bonds, angles, torsions, impropers, dropped = tables_from_solution(columns, sol)
    cross = {k: v for k, v in cross.items() if k in angles}
    dropped += cross_dropped

    data = copy.deepcopy(skeleton)
    data.update(
        version="0.1.0",
        reference=f"{LEVEL}; {len(train)} training molecules, {len(scans)} torsion scans",
        bci=bci, bonds=bonds, angles=angles, torsions=torsions, impropers=impropers,
    )
    data["stretch_bend"] = cross
    data["pi"] = {"enabled": False, "bonds": {}}
    if pi_fit:
        table = {}
        for key in prov.pi_bonds:
            name = pi_pair(prov, *key.split("~"))
            if name in pi_fit["beta"]:
                table[key] = [round(pi_fit["beta"][name], 3), round(pi_fit["delta"], 4),
                              round(pi_fit["rref"][name], 4)]
        data["pi"] = {"enabled": True, "bonds": table}
        for entry in data["atom_types"]:
            if entry.get("pi_electrons") is not None:
                entry["pi_alpha"] = round(pi_fit["alpha"].get(pi_site(prov, entry["type"]), 0.0), 3)
    data["increments"] = {"types": {}, "bonds": {}}
    data["vdw"] = copy.deepcopy(prov.data["vdw"])
    if vdw_stats:
        data["vdw_source"] = (
            "Lennard-Jones refit to counterpoise-corrected DFT dimer curves for: "
            + ", ".join(vdw_stats["classes"])
            + "; all other elements keep the UFF values (RDKit Params.cpp)"
        )
    scan_rmse = fit_scans(scans, data)
    inc_rmse = fit_increments(train, data)

    params = ParameterSet(data)
    stats = {"train": evaluate(train, hessians, params), "validate": evaluate(valid, hessians, params)}
    data["fit_statistics"] = {
        name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "refused"}
        for name, s in stats.items()
    }
    data["fit_statistics"]["increment_rmse_kcal"] = inc_rmse

    lines = [
        "# PMEFF26 fit report", "",
        f"Reference: {LEVEL}.  Training molecules: {len(train)}; validation: {len(valid)}; scans: {len(scans)}.",
        f"Parameters: {len(bonds)} bonds, {len(angles)} angles, {len(torsions)} torsion classes, "
        f"{len(impropers)} impropers, {len(bci)} charge increments.", "",
        "| set | molecules | covered | bond RMSD (A) | angle RMSD (deg) | frequency RMSD (cm-1) |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['n']} | {s['covered']} | {s['bond_rmsd']:.4f} | {s['angle_rmsd']:.2f} | {s['freq_rmsd']:.0f} |")
    lines += ["", "Errors are force-field-optimized geometry and frequencies against DFT. "
              "The validation row is the honest accuracy figure: those molecules were not fitted.", ""]
    if inc_rmse is not None:
        lines.append(f"DFT-referenced energy increments: training RMSE {inc_rmse:.2f} kcal/mol.")
    if vdw_stats:
        lines += ["", f"van der Waals refit on {vdw_stats['points']} dimer points "
                  f"({len(dimers)} pairs): interaction-energy RMSE "
                  f"{vdw_stats['before']:.2f} -> {vdw_stats['after']:.2f} kcal/mol "
                  f"(points with E_int < 5). Classes: {', '.join(vdw_stats['classes'])}."]
    else:
        lines += ["", "van der Waals: no dimer data yet; UFF values used unchanged."]
    lines += ["", f"Stretch-bend cross terms: {len(cross)} angle classes."
              if use_sb else "Stretch-bend cross terms: switched off."]
    if pi_fit:
        lines += ["", f"pi-electron term fitted on {pi_fit['molecules']} molecules: unexplained "
                  f"gradient/Hessian residual {pi_fit['before']:.3f} -> {pi_fit['after']:.3f} "
                  f"(RMS; before = conventional terms only). delta = {pi_fit['delta']:.3f} 1/A.", "",
                  "| pi bond class | beta (kcal/mol) | rref (A) |", "|---|---|---|"]
        lines += [f"| {n} | {b:.1f} | {pi_fit['rref'][n]:.3f} |" for n, b in sorted(pi_fit["beta"].items())]
        if pi_fit["alpha"]:
            lines += ["", "On-site energies relative to carbon (kcal/mol): "
                      + ", ".join(f"{c} {v:+.1f}" for c, v in sorted(pi_fit["alpha"].items())) + "."]
    else:
        lines += ["", "pi-electron term: switched off."]
    if scan_rmse:
        lines += ["", "## Torsion scan fit (RMSE, kcal/mol)", ""]
        lines += [f"- `{k}`: {v:.2f}" for k, v in sorted(scan_rmse.items())]
    if dropped:
        lines += ["", "## Terms left out (not determined by the data)", ""]
        lines += [f"- {d}" for d in sorted(dropped)]
    for name, s in stats.items():
        if s["refused"]:
            lines += ["", f"## {name}: molecules not covered", ""]
            lines += [f"- {i} `{smi}`: {', '.join(m)}" for i, smi, m in s["refused"]]
    unused = [r for r in records if not r["usable"]]
    if unused or untyped:
        lines += ["", "## Reference records not used", ""]
        lines += [f"- {r['id']} `{r['smiles']}`: {r['reason']}" for r in unused]
        lines += [f"- {r['id']} `{r['smiles']}`: {why}" for r, why in untyped]
    return data, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    records = json.loads((REF / "reference_data.json").read_text(encoding="utf-8"))
    with np.load(REF / "reference_hessians.npz") as npz:
        hessians = {k: npz[k] for k in npz.files}
    from . import init_parameters

    skeleton = init_parameters.skeleton()
    argv = sys.argv[1:] if argv is None else argv
    data, report = fit(records, hessians, skeleton, use_pi="--no-pi" not in argv,
                       use_sb="--no-sb" not in argv)
    level.PARAMETER_FILE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    (REF / "fit_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print("wrote", level.PARAMETER_FILE, "and", REF / "fit_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
