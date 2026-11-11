#!/usr/bin/env python3
"""Build the GAMESS reference job pack for the PMEFF26 fit.

Embeds every molecule of :mod:`fitting.dataset` with RDKit (fixed seed),
writes one GAMESS input per job with inline coordinates and basis, and records the exact
RDKit molecule (atom order = coordinate order) in ``manifest.json`` so DFT
results map back onto bonds, bond orders and hybridizations without any
re-perception.

    python -m fitting.make_pack --out reference/pack [--only tr va sc] [--limit N]

Job ids: ``tr_NNN`` (training, opt + freq), ``va_NNN`` (validation, opt +
freq), ``sc_NNN`` (relaxed torsion scan).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from . import dataset, gamess

SEED = 20261112
SCAN_START = -170.0
SCAN_STEP = 15.0


_VERTICES = {
    "sqp": [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
    "oct": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)],
}


def _embed_with_hint(mol, params, hint: str, smiles: str) -> None:
    """Embed with the metal's donor atoms pinned to ideal polyhedron vertices.

    Donors are sorted by element so identical ligands sit on adjacent vertices
    (the cis isomer of a square-planar MA2B2 complex).
    """
    from rdkit.Geometry import Point3D

    metal = next(a for a in mol.GetAtoms() if a.GetSymbol() in dataset.METAL_ELEMENTS)
    donors = sorted(metal.GetNeighbors(), key=lambda a: (a.GetAtomicNum(), a.GetIdx()))
    vertices = _VERTICES[hint]
    if len(donors) != len(vertices):
        raise ValueError(f"{hint} needs {len(vertices)} donors: {smiles}")
    table = Chem.GetPeriodicTable()
    coord_map = {metal.GetIdx(): Point3D(0.0, 0.0, 0.0)}
    for donor, (x, y, z) in zip(donors, vertices):
        d = table.GetRcovalent(metal.GetAtomicNum()) + table.GetRcovalent(donor.GetAtomicNum())
        coord_map[donor.GetIdx()] = Point3D(x * d, y * d, z * d)
    params.SetCoordMap(coord_map)
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) == 0:
        return
    # Distance geometry failed (e.g. hexacarbonyls): lay each ligand out along
    # its vertex ray. Only valid for unbranched ligands such as CO or CN.
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal.GetIdx(), Point3D(0.0, 0.0, 0.0))
    for donor, (x, y, z) in zip(donors, vertices):
        prev, atom, dist = metal, donor, 0.0
        while atom is not None:
            dist += 0.9 * (table.GetRcovalent(prev.GetAtomicNum()) + table.GetRcovalent(atom.GetAtomicNum()))
            conf.SetAtomPosition(atom.GetIdx(), Point3D(x * dist, y * dist, z * dist))
            nxt = [n for n in atom.GetNeighbors() if n.GetIdx() != prev.GetIdx()]
            if len(nxt) > 1:
                raise ValueError(f"embedding failed (branched ligand): {smiles}")
            prev, atom = atom, (nxt[0] if nxt else None)
    mol.AddConformer(conf, assignId=True)


def build_mol(smiles: str) -> Chem.Mol:
    """Return an H-complete molecule with one MMFF/UFF-relaxed 3D conformer."""
    smiles, _, hint = smiles.partition("|")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad SMILES: {smiles}")
    if Chem.GetFormalCharge(mol) != 0:
        raise ValueError(f"not neutral: {smiles}")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError(f"radical: {smiles}")
    bad = {a.GetSymbol() for a in mol.GetAtoms()} - dataset.ALLOWED_ELEMENTS - dataset.METAL_ELEMENTS
    if bad:
        raise ValueError(f"elements outside the fit {sorted(bad)}: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    if hint:
        _embed_with_hint(mol, params, hint, smiles)
    else:
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise ValueError(f"embedding failed: {smiles}")
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            elif AllChem.UFFHasAllMoleculeParams(mol):
                AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception:  # pragma: no cover - a cleanup failure is not fatal
            pass
    # GAMESS's Cartesian optimizer aborts on exactly linear or planar starts
    # (wrong count of zero modes in its guess Hessian): break the symmetry.
    rng = np.random.RandomState(SEED)
    conf = mol.GetConformer()
    for idx in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(idx)
        d = rng.uniform(-0.02, 0.02, 3)
        conf.SetAtomPosition(idx, (p.x + d[0], p.y + d[1], p.z + d[2]))
    return mol


def n_heavy(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def has_metal(mol: Chem.Mol) -> bool:
    return any(a.GetSymbol() in dataset.METAL_ELEMENTS for a in mol.GetAtoms())


def nprocs_for(mol: Chem.Mol) -> int:
    return 4 if (has_metal(mol) and mol.GetNumAtoms() > 3) or n_heavy(mol) > 6 else 2


FUNCTIONAL = {"functional": "wB97X-D", "d3bj": False}


def write_input(path: Path, job_id: str, mol: Chem.Mol, hessian: bool,
                frozen=None, value: float = 0.0) -> None:
    conf = mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()]
    text = gamess.gamess_input(
        job_id,
        [a.GetSymbol() for a in mol.GetAtoms()],
        [a.GetAtomicNum() for a in mol.GetAtoms()],
        coords,
        hessian=hessian,
        frozen_dihedral=frozen,
        frozen_value=value,
        **FUNCTIONAL,
    )
    path.write_text(text, encoding="ascii", newline="\n")


def first_h(mol: Chem.Mol, idx: int, exclude: int) -> int:
    for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
        if nbr.GetAtomicNum() == 1 and nbr.GetIdx() != exclude:
            return nbr.GetIdx()
    raise ValueError(f"atom {idx} has no hydrogen")


def is_c3_rotor(mol: Chem.Mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    hs = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
    return atom.GetDegree() == 4 and hs == 3


def record(job_id: str, kind: str, smiles: str, mol: Chem.Mol) -> dict:
    return {
        "id": job_id,
        "kind": kind,
        "smiles": smiles,
        "canonical": Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "n_atoms": mol.GetNumAtoms(),
        "n_heavy": n_heavy(mol),
        "nprocs": nprocs_for(mol),
        "molblock": Chem.MolToMolBlock(mol),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["tr", "va", "sc", "mt"])
    ap.add_argument("--limit", type=int, default=0, help="first N of each set")
    ap.add_argument("--ids", nargs="*", default=None, help="keep only these job ids")
    ap.add_argument("--functional", default="wB97X-D", help="GAMESS DFTTYP")
    ap.add_argument("--d3bj", action="store_true", help="add Grimme D3(BJ)")
    args = ap.parse_args(argv)

    FUNCTIONAL.update(functional=args.functional, d3bj=args.d3bj)
    inp_dir = args.out / "inp"
    inp_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    seen: dict[str, str] = {}
    problems: list[str] = []

    for prefix, smiles_list, kind in (
        ("tr", dataset.train_smiles(), "train"),
        ("va", dataset.validate_smiles(), "validate"),
        ("mt", dataset.metal_smiles(), "train"),
    ):
        for n, smiles in enumerate(smiles_list, 1):
            job_id = f"{prefix}_{n:03d}"
            try:
                mol = build_mol(smiles)
            except ValueError as exc:
                problems.append(f"{job_id}: {exc}")
                continue
            rec = record(job_id, kind, smiles, mol)
            if rec["canonical"] in seen:
                problems.append(
                    f"{job_id}: duplicate of {seen[rec['canonical']]} ({smiles})"
                )
                continue
            seen[rec["canonical"]] = job_id
            if prefix not in args.only or (args.limit and n > args.limit):
                continue
            write_input(inp_dir / f"{job_id}.inp", job_id, mol, hessian=True)
            manifest.append(rec)

    for n, (label, smiles, quad) in enumerate(dataset.SCANS, 1):
        job_id = f"sc_{n:03d}"
        try:
            mol = build_mol(smiles)
            i, j, k, l = quad
            if i == "H":
                i = first_h(mol, j, k)
            if l == "H":
                l = first_h(mol, k, j)
            if mol.GetBondBetweenAtoms(j, k) is None or any(
                mol.GetBondBetweenAtoms(a, b) is None for a, b in ((i, j), (k, l))
            ):
                raise ValueError(f"dihedral {quad} is not a bonded path: {smiles}")
            npts = 9 if (is_c3_rotor(mol, j) or is_c3_rotor(mol, k)) else 24
            rdMolTransforms.SetDihedralDeg(mol.GetConformer(), i, j, k, l, SCAN_START)
        except ValueError as exc:
            problems.append(f"{job_id} ({label}): {exc}")
            continue
        if "sc" not in args.only or (args.limit and n > args.limit):
            continue
        values = [SCAN_START + SCAN_STEP * p for p in range(npts)]
        write_input(inp_dir / f"{job_id}.inp", job_id, mol, hessian=False,
                    frozen=(i, j, k, l), value=values[0])
        (inp_dir / f"{job_id}.scan").write_text(
            json.dumps(values), encoding="ascii", newline="\n"
        )
        rec = record(job_id, "scan", smiles, mol)
        rec.update(label=label, dihedral=[i, j, k, l], n_points=npts)
        manifest.append(rec)

    if args.ids is not None:
        wanted = set(args.ids)
        manifest = [r for r in manifest if r["id"] in wanted]
        for path in list(inp_dir.iterdir()):
            if path.stem not in wanted:
                path.unlink()
    # Small jobs first so data accrues early; nprocs groups stay together.
    manifest.sort(key=lambda r: (r["nprocs"], r["kind"] == "scan", r["n_atoms"]))
    for nprocs in (2, 4):
        ids = [r["id"] for r in manifest if r["nprocs"] == nprocs]
        (args.out / f"order_np{nprocs}.txt").write_text(
            "".join(f"{x}\n" for x in ids), encoding="ascii", newline="\n"
        )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline="\n"
    )

    kinds = {}
    for r in manifest:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(manifest)} jobs to {args.out}: {kinds}")
    for p in problems:
        print("PROBLEM", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
