#!/usr/bin/env python3
"""Download and extract the GAMESS reference results into one data set.

    python -m fitting.collect JOB_ID [JOB_ID ...]   # fetch from the host, then parse
    python -m fitting.collect                        # parse what is already in reference/raw

For each Job Manager job the script lists the job's remote ``results/``
directory, downloads every result file not yet in ``reference/raw/`` (works
while the job is still running), parses them and writes

* ``reference/reference_data.json`` - one record per molecule / scan, and
* ``reference/reference_hessians.npz`` - Cartesian Hessians (kcal/mol/A^2).

A record is flagged ``usable: false`` (with a ``reason``) when the job failed,
the optimized structure changed its connectivity, or it has an imaginary
frequency. Nothing is dropped silently: the summary lists every such case.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "reference"
RAW = REF / "raw"
HARTREE_KCAL = 627.5095
BOHR = 0.52917721
HESS_UNIT = HARTREE_KCAL / (BOHR * BOHR)  # Eh/bohr^2 -> kcal/mol/A^2

_ATOM = re.compile(
    r"^\s*(\S+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_ENERGY = re.compile(r"FINAL R-\S+ ENERGY IS\s+(-?\d+\.\d+)")


def _client():
    from .run_all import client

    return client()


def fetch(job_id: str) -> int:
    """Download the result files of one job that are not in reference/raw yet."""
    cl = _client()
    job = cl.request("GET", f"/jobs/{job_id}")
    job = job.get("job", job)
    host, remote = job["host_id"], job["remote_dir"].rstrip("/") + "/results"
    listing = cl.request("GET", f"/hosts/{host}/files?path={remote}&depth=1")
    names = [n for n in listing.get("entries", listing.get("files", [])) if not n.endswith("/")]
    RAW.mkdir(parents=True, exist_ok=True)
    want = [
        n for n in names
        if n.endswith((".gz", ".json", ".done", ".failed")) and not (RAW / n).exists()
    ]
    for start in range(0, len(want), 100):
        chunk = want[start : start + 100]
        cl.request(
            "POST",
            f"/hosts/{host}/download",
            {"paths": [f"{remote}/{n}" for n in chunk], "into": str(RAW)},
        )
    print(f"{job_id}: {len(names)} remote result files, {len(want)} downloaded")
    return len(want)


def load_manifest() -> dict:
    """Job id -> manifest record, from every reference/pack*/manifest.json."""
    records: dict = {}
    paths = sorted(REF.glob("pack*/manifest.json"))
    if not paths:
        from . import make_pack

        make_pack.main(["--out", str(REF / "pack")])
        paths = sorted(REF.glob("pack*/manifest.json"))
    for path in paths:
        for rec in json.loads(path.read_text(encoding="utf-8")):
            records.setdefault(rec["id"], rec)
    return records


def _read(path: Path) -> str:
    with gzip.open(path, "rt", errors="replace") as fh:
        return fh.read()


def parse_output(text: str) -> dict | None:
    """Final energy, geometry, dipole, charges and imaginary-mode count."""
    pos = text.find("EQUILIBRIUM GEOMETRY LOCATED")
    if pos < 0:
        return None
    energies = _ENERGY.findall(text[:pos])
    symbols, coords = [], []
    started = False
    for line in text[pos:].splitlines():
        m = _ATOM.match(line)
        if m:
            started = True
            symbols.append(m.group(1))
            coords.append([float(m.group(3)), float(m.group(4)), float(m.group(5))])
        elif started:
            break
    if not energies or not coords:
        return None
    out = {"energy": float(energies[-1]), "symbols": symbols, "coords": coords}
    tail = text[pos:]
    m = re.search(r"/D/\s+\(DEBYE\)\s*\n\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", tail)
    out["dipole"] = [float(g) for g in m.groups()] if m else None
    lowdin, mulliken = [], []
    q = tail.find("TOTAL MULLIKEN AND LOWDIN ATOMIC POPULATIONS")
    if q >= 0:
        for line in tail[q:].splitlines()[2:]:
            parts = line.split()
            if len(parts) != 6:
                break
            mulliken.append(float(parts[3]))
            lowdin.append(float(parts[5]))
    n = len(coords)
    out["mulliken"] = mulliken if len(mulliken) == n else None
    out["lowdin"] = lowdin if len(lowdin) == n else None
    # Imaginary modes are printed with an "I" after the wavenumber.
    imag = 0
    for line in tail.splitlines():
        if line.lstrip().startswith("FREQUENCY:"):
            for value, flag in re.findall(r"(\d+\.\d+)(\s+I)?", line.split(":", 1)[1]):
                if flag and float(value) > 20.0:
                    imag += 1
    out["n_imag"] = imag
    return out


def parse_hessian(dat_text: str, n_atoms: int) -> np.ndarray | None:
    """The last $HESS group of a GAMESS .dat file, in kcal/mol/A^2."""
    start = dat_text.rfind(" $HESS")
    if start < 0:
        return None
    end = dat_text.find(" $END", start)
    lines = dat_text[start:end].splitlines()[2:]  # skip "$HESS" and "ENERGY IS"
    n3 = 3 * n_atoms
    values: list[float] = []
    for line in lines:
        body = line[5:]
        values.extend(float(body[c : c + 15]) for c in range(0, len(body.rstrip()), 15))
    if len(values) != n3 * n3:
        return None
    hess = np.array(values).reshape(n3, n3)
    return 0.5 * (hess + hess.T) * HESS_UNIT


def connectivity_ok(molblock: str, coords: np.ndarray) -> bool:
    """True when every bond survived and no new bond-length contact appeared."""
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=False)
    table = Chem.GetPeriodicTable()
    rcov = [table.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()]
    bonded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))
        if np.linalg.norm(coords[i] - coords[j]) > 1.35 * (rcov[i] + rcov[j]):
            return False
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in bonded and np.linalg.norm(coords[i] - coords[j]) < 0.85 * (
                rcov[i] + rcov[j]
            ):
                return False
    return True


def collect() -> tuple[list, dict]:
    manifest = load_manifest()
    records, hessians = [], {}
    for job_id, rec in sorted(manifest.items()):
        entry = {k: rec[k] for k in ("id", "kind", "smiles", "molblock", "n_atoms")}
        scan_path = RAW / f"{job_id}.scan.json"
        out_path = RAW / f"{job_id}.out.gz"
        if rec["kind"] == "scan":
            if not scan_path.exists():
                continue
            points = [
                {"value": p["value"], "energy": p["energy"],
                 "coords": [list(a[2:]) for a in p["atoms"]]}
                for p in json.loads(scan_path.read_text()) if p["energy"] is not None
            ]
            entry.update(label=rec["label"], dihedral=rec["dihedral"], points=points,
                         usable=len(points) >= 4,
                         reason="" if len(points) >= 4 else "too few converged points")
            records.append(entry)
            continue
        if not out_path.exists():
            continue
        parsed = parse_output(_read(out_path))
        if parsed is None:
            entry.update(usable=False, reason="optimization did not converge")
            records.append(entry)
            continue
        coords = np.array(parsed["coords"])
        entry.update(parsed)
        reason = ""
        if len(coords) != rec["n_atoms"]:
            reason = "atom count mismatch"
        elif not connectivity_ok(rec["molblock"], coords):
            reason = "connectivity changed during optimization"
        elif parsed["n_imag"]:
            reason = f"{parsed['n_imag']} imaginary frequency(ies)"
        dat_path = RAW / f"{job_id}.dat.gz"
        hess = parse_hessian(_read(dat_path), len(coords)) if dat_path.exists() else None
        if hess is not None and len(coords) > 1:
            hessians[job_id] = hess
        elif len(coords) > 1 and not reason:
            reason = "no Hessian"
        entry.update(usable=not reason, reason=reason)
        records.append(entry)
    return records, hessians


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for job_id in argv:
        fetch(job_id)
    records, hessians = collect()
    REF.mkdir(exist_ok=True)
    (REF / "reference_data.json").write_text(json.dumps(records), encoding="utf-8")
    np.savez_compressed(REF / "reference_hessians.npz", **hessians)
    usable = sum(1 for r in records if r["usable"])
    print(f"{len(records)} records ({usable} usable), {len(hessians)} Hessians")
    print("wrote", REF / "reference_data.json", "and reference_hessians.npz")
    for r in records:
        if not r["usable"]:
            print(f"  NOT USED {r['id']} {r['smiles']}: {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
