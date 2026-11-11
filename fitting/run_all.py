#!/usr/bin/env python3
"""Run the whole PEFF26 reference calculation in one go.

Builds every GAMESS input, packs them, and submits ONE job to the compute host
through the MoleditPy Job Manager local API (the Job Manager must be running
with *Local API* switched on):

    python -m fitting.run_all                 # everything (train + validate + scans)
    python -m fitting.run_all --smoke         # 4 quick jobs to check the setup
    python -m fitting.run_all --fetch JOB_ID  # download results into reference/raw

The job runs ``python3 run_pack.py pack.tgz`` on the host; it resumes where it
stopped if resubmitted in the same remote directory. Progress is in
``progress.log`` in the job's remote directory (Job Manager > log / files).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE_SMILES = {"O", "CI", "CC"}
SMOKE_SCAN = "ethane"


def client():
    for cand in (REPO.parent / "moleditpy_job_manager", Path.home() / ".moleditpy" / "plugins"):
        if (cand / "job_manager" / "api_client.py").exists():
            sys.path.insert(0, str(cand))
            break
    from job_manager.api_client import JobManagerClient  # pylint: disable=import-error

    return JobManagerClient()


def build_pack(smoke: bool) -> Path:
    from . import make_pack

    out = REPO / "reference" / ("smoke" if smoke else "pack")
    shutil.rmtree(out, ignore_errors=True)
    if make_pack.main(["--out", str(out)]) != 0:
        raise SystemExit("dataset problems reported above; fix them first")
    if smoke:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        keep = [
            r for r in manifest
            if (r["kind"] == "train" and r["smiles"] in SMOKE_SMILES)
            or (r["kind"] == "scan" and r["label"] == SMOKE_SCAN)
        ]
        ids = {r["id"] for r in keep}
        for p in (out / "inp").iterdir():
            if p.stem not in ids:
                p.unlink()
        (out / "order_np2.txt").write_text("".join(f"{r['id']}\n" for r in keep), newline="\n")
        (out / "order_np4.txt").write_text("", newline="\n")
        (out / "manifest.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
    with tarfile.open(out / "pack.tgz", "w:gz") as tar:
        for name in ("inp", "order_np2.txt", "order_np4.txt"):
            tar.add(out / name, arcname=name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="1J1NFIR")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the pack only")
    ap.add_argument("--fetch", metavar="JOB_ID")
    args = ap.parse_args(argv)

    if args.fetch:
        into = REPO / "reference" / "raw"
        into.mkdir(parents=True, exist_ok=True)
        reply = client().download(args.fetch, wait=True, into=str(into))
        print("downloaded:", reply.get("files"))
        return 0

    out = build_pack(args.smoke)
    print("pack:", out / "pack.tgz")
    if args.dry_run:
        return 0
    job = client().submit(
        host=args.host,
        name="peff26-ref-smoke" if args.smoke else "peff26-reference",
        files=[str(out / "pack.tgz"), str(Path(__file__).with_name("run_pack.py"))],
        command="python3 run_pack.py pack.tgz",
        cpus_per_task=8,
        memory="32GB",
        fetch_globs=["results.tgz", "progress.log"],
        auto_download=False,
    )
    job = job.get("job", job)
    print("submitted job", job["id"], "to", args.host)
    print("fetch results with: python -m fitting.run_all --fetch", job["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
