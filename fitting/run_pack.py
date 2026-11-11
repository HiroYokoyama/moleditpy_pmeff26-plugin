#!/usr/bin/env python3
"""Remote driver for the PEFF26 reference pack (standard library only).

Runs on the compute host, in the directory holding ``pack.tgz``:

    python3 run_pack.py pack.tgz

Every job in ``order_np2.txt`` runs 4 at a time on 2 cores, then every job in
``order_np4.txt`` 2 at a time on 4 cores, through the host's ``gamess``
launcher (``gamess name.inp ncores`` -> ``name.out``). A job with a
``.scan`` file is a relaxed torsion scan: its points run one after another,
each starting from the previous optimized geometry.

Only small result files are kept, gzipped, in ``results/``. Re-running the
script skips jobs that already finished, so an interrupted run resumes.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time

HOME = os.path.expanduser("~")
GAMESS = os.path.join(HOME, "qc-progs", "bin", "gamess")
RESTART = glob.glob(os.path.join(HOME, "qc-progs", "gamess-*", "restart"))
ROOT = os.getcwd()
RESULTS = os.path.join(ROOT, "results")
LOCK = threading.Lock()

_ATOM = re.compile(
    r"^\s*(\S+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_ENERGY = re.compile(r"FINAL R-\S+ ENERGY IS\s+(-?\d+\.\d+)")


def log(msg: str) -> None:
    with LOCK, open(os.path.join(ROOT, "progress.log"), "a") as fh:
        fh.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def run_gamess(workdir: str, name: str, ncores: int) -> str:
    """Run one GAMESS job; return the path of its output file."""
    for stale in glob.glob(os.path.join(HOME, "work", "gamess-scr", name + ".*")):
        os.remove(stale)
    subprocess.run([GAMESS, name + ".inp", str(ncores)], cwd=workdir, check=False)
    return os.path.join(workdir, name + ".out")


def parse_final(out_path: str):
    """Return (energy, [(symbol, charge, x, y, z)]) of the located minimum."""
    try:
        text = open(out_path, errors="replace").read()
    except OSError:
        return None
    pos = text.find("EQUILIBRIUM GEOMETRY LOCATED")
    if pos < 0:
        return None
    energies = _ENERGY.findall(text[:pos])
    atoms = []
    started = False
    for line in text[pos:].splitlines():
        m = _ATOM.match(line)
        if m:
            started = True
            atoms.append((m.group(1), m.group(2)) + tuple(float(g) for g in m.groups()[2:]))
        elif started:
            break
    if not energies or not atoms:
        return None
    return float(energies[-1]), atoms


def keep(path: str, dest_name: str) -> None:
    if os.path.exists(path):
        with open(path, "rb") as src, gzip.open(
            os.path.join(RESULTS, dest_name + ".gz"), "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)


def dat_file(name: str) -> str:
    for d in RESTART:
        p = os.path.join(d, name + ".dat")
        if os.path.exists(p):
            return p
    return ""


def cleanup(workdir: str, name: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
    for d in RESTART:
        for p in glob.glob(os.path.join(d, name + ".*")):
            os.remove(p)
    for p in glob.glob(os.path.join(HOME, "work", "gamess-scr", name + ".*")):
        os.remove(p)


def finish(job: str, ok: bool) -> None:
    open(os.path.join(RESULTS, job + (".done" if ok else ".failed")), "w").close()
    log(f"{job} {'done' if ok else 'FAILED'}")


def run_opt(job: str, ncores: int) -> None:
    workdir = os.path.join(ROOT, "work", job)
    os.makedirs(workdir, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "inp", job + ".inp"), workdir)
    out = run_gamess(workdir, job, ncores)
    text = open(out, errors="replace").read() if os.path.exists(out) else ""
    ok = "EQUILIBRIUM GEOMETRY LOCATED" in text and "EXECUTION OF GAMESS TERMINATED NORMALLY" in text
    keep(out, job + ".out")
    keep(dat_file(job), job + ".dat")
    cleanup(workdir, job)
    finish(job, ok)


def run_scan(job: str, ncores: int) -> None:
    values = json.load(open(os.path.join(ROOT, "inp", job + ".scan")))
    template = open(os.path.join(ROOT, "inp", job + ".inp")).read().splitlines()
    atom_rows = [n for n, line in enumerate(template) if _ATOM.match(line)]
    points = []
    coords = None
    for p, value in enumerate(values):
        name = f"{job}_p{p:02d}"
        lines = list(template)
        for n, line in enumerate(lines):
            if "FVALUE(1)=" in line:
                lines[n] = re.sub(r"FVALUE\(1\)=\S+", f"FVALUE(1)={value:.4f}", line)
        if coords is not None:
            for row, (sym, chg, x, y, z) in zip(atom_rows, coords):
                lines[row] = f"{sym:<2s} {chg:>5s} {x:16.10f} {y:16.10f} {z:16.10f}"
        workdir = os.path.join(ROOT, "work", name)
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, name + ".inp"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        out = run_gamess(workdir, name, ncores)
        final = parse_final(out)
        if final is None:
            keep(out, name + ".out")  # keep the evidence of a failed point
            points.append({"value": value, "energy": None, "atoms": None})
        else:
            energy, coords = final
            points.append({"value": value, "energy": energy, "atoms": coords})
        cleanup(workdir, name)
        with open(os.path.join(RESULTS, job + ".scan.json"), "w") as fh:
            json.dump(points, fh)
    finish(job, all(pt["energy"] is not None for pt in points))


def run_job(job: str, ncores: int) -> None:
    if os.path.exists(os.path.join(RESULTS, job + ".done")):
        return
    try:
        if os.path.exists(os.path.join(ROOT, "inp", job + ".scan")):
            run_scan(job, ncores)
        else:
            run_opt(job, ncores)
    except Exception as exc:  # one bad job must not stop the pack
        log(f"{job} ERROR {exc!r}")
        finish(job, False)


def main() -> int:
    with tarfile.open(sys.argv[1]) as tar:
        tar.extractall(ROOT)
    os.makedirs(RESULTS, exist_ok=True)
    for fname, ncores, workers in (("order_np2.txt", 2, 4), ("order_np4.txt", 4, 2)):
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            continue
        jobs = open(path).read().split()
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda j: run_job(j, ncores), jobs))
    with tarfile.open(os.path.join(ROOT, "results.tgz"), "w:gz") as tar:
        tar.add(RESULTS, arcname="results")
    done = len(glob.glob(os.path.join(RESULTS, "*.done")))
    failed = len(glob.glob(os.path.join(RESULTS, "*.failed")))
    print(f"finished: {done} done, {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
