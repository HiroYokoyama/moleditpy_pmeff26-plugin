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


def provisional(skeleton: dict, mols: list) -> ParameterSet:
    """Skeleton plus a unit placeholder for every term the molecules need."""
    data = copy.deepcopy(skeleton)
    data["pi"] = {"enabled": False, "bonds": {}}
    for table in ("bonds", "angles", "torsions", "impropers", "bci"):
        data[table] = {}
    data["increments"] = {"types": {}, "bonds": {}}
    for mol in mols:
        a = assign(mol, ParameterSet(data), allow_missing=True)
        for item in a.missing:
            kind, key = item.rsplit(" ", 1)
            if kind == "bond":
                data["bonds"][key] = [1.0, 0.0]
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


def fit_bonded(train: list, hessians: dict, prov: ParameterSet, classes: dict):
    columns: dict = {}
    lower: list = []
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
    # Scale every column to unit norm: the unknowns span four orders of
    # magnitude (bond k ~ 1e3, torsion barriers ~ 1), which stalls LSMR.
    norms = np.sqrt(np.asarray(a_mat.multiply(a_mat).sum(axis=0)).ravel())
    norms[norms == 0.0] = 1.0
    scaled = a_mat @ sparse.diags(1.0 / norms)
    lb = np.array(lower) * norms  # 0 and -inf are unchanged by the scaling
    result = lsq_linear(scaled, b_vec, bounds=(lb, np.inf), lsq_solver="lsmr",
                        lsmr_tol=1e-12, tol=1e-12, max_iter=500, lsmr_maxiter=20000)
    return columns, result.x / norms


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


def fit(records: list, hessians: dict, skeleton: dict):
    """Return (parameter dict, report text)."""
    usable = [r for r in records if r["usable"]]
    train = [(r, mol_of(r)) for r in usable if r["kind"] == "train"]
    valid = [(r, mol_of(r)) for r in usable if r["kind"] == "validate"]
    scans = [r for r in usable if r["kind"] == "scan"]
    dimers = [r for r in usable if r["kind"] == "dimer"]
    if not train:
        raise SystemExit("no usable training molecules in the reference data")

    prov = provisional(skeleton, [m for _r, m in train])
    bci = fit_charges(train, prov)
    prov.data["bci"] = {k: bci.get(k, 0.0) for k in prov.bci}
    vdw_stats = fit_vdw(dimers, prov.data)  # updates prov.data["vdw"] in place
    prov = ParameterSet(prov.data)
    classes = angle_classes(train, prov)
    columns, sol = fit_bonded(train, hessians, prov, classes)
    bonds, angles, torsions, impropers, dropped = tables_from_solution(columns, sol)

    data = copy.deepcopy(skeleton)
    data.update(
        version="0.1.0",
        reference=f"{LEVEL}; {len(train)} training molecules, {len(scans)} torsion scans",
        bci=bci, bonds=bonds, angles=angles, torsions=torsions, impropers=impropers,
    )
    data["pi"] = {"enabled": False, "bonds": {}}
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
    if unused:
        lines += ["", "## Reference records not used", ""]
        lines += [f"- {r['id']} `{r['smiles']}`: {r['reason']}" for r in unused]
    return data, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    records = json.loads((REF / "reference_data.json").read_text(encoding="utf-8"))
    with np.load(REF / "reference_hessians.npz") as npz:
        hessians = {k: npz[k] for k in npz.files}
    from . import init_parameters

    skeleton = init_parameters.skeleton()
    data, report = fit(records, hessians, skeleton)
    level.PARAMETER_FILE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    (REF / "fit_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print("wrote", level.PARAMETER_FILE, "and", REF / "fit_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
