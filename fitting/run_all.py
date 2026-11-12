#!/usr/bin/env python3
"""Run the whole PMEFF26 reference calculation in one go.

Builds every GAMESS input, packs them, and submits ONE job to the compute host (``--host NAME`` or ``PMEFF26_HOST``)
through the MoleditPy Job Manager local API (the Job Manager must be running
with *Local API* switched on):

    python -m fitting.run_all                 # everything (train + validate + scans)
    python -m fitting.run_all --smoke         # 4 quick jobs to check the setup
    python -m fitting.run_all --retry JOB_ID  # rerun only what FAILED in that job
    python -m fitting.run_all --fetch JOB_ID  # download results into reference/raw

The job runs ``python3 run_pack.py pack.tgz`` on the host; it resumes where it
stopped if resubmitted in the same remote directory. Progress is in
``progress.log`` in the job's remote directory (Job Manager > log / files).
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
import tarfile
from pathlib import Path

from . import level

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


def failed_ids(job_id: str) -> list[str]:
    """Job ids the remote driver logged as FAILED for Job Manager job *job_id*."""
    reply = client().request(
        "GET", f"/jobs/{job_id}/log?file=progress.log&lines=5000"
    )
    text = reply.get("text") or reply.get("log") or ""
    if isinstance(text, list):
        text = chr(10).join(text)
    state = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3] in ("done", "FAILED"):
            state[parts[2]] = parts[3]
    return sorted(j for j, s in state.items() if s == "FAILED")


def build_pack(smoke: bool, ids: list[str] | None = None, only: list[str] | None = None,
               extra: list[str] | None = None) -> Path:
    from . import make_pack

    name = "smoke" if smoke else "retry" if ids else "pack-" + "-".join(only) if only else "pack"
    out = level.REF / name
    shutil.rmtree(out, ignore_errors=True)
    argv = ["--out", str(out)] + (["--ids", *ids] if ids else []) + (["--only", *only] if only else []) + (extra or [])
    if make_pack.main(argv) != 0:
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
    ap.add_argument("--host", default=os.environ.get("PMEFF26_HOST"),
                    help="Job Manager host name (or set PMEFF26_HOST)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the pack only")
    ap.add_argument("--fetch", metavar="JOB_ID")
    ap.add_argument("--ids", nargs="*", help="submit only these job ids")
    ap.add_argument("--force", action="store_true", help="start now, ahead of the host queue (small checks only)")
    ap.add_argument("--functional", default="wB97X-D", help="GAMESS DFTTYP (use with PMEFF26_LEVEL)")
    ap.add_argument("--d3bj", action="store_true", help="add Grimme D3(BJ) dispersion")
    ap.add_argument("--only", nargs="*", help="sets to run: tr va sc mt (default all)")
    ap.add_argument("--retry", metavar="JOB_ID",
                    help="resubmit only the jobs that FAILED in this earlier run")
    args = ap.parse_args(argv)

    if args.fetch:
        into = level.REF / "raw"
        into.mkdir(parents=True, exist_ok=True)
        reply = client().download(args.fetch, wait=True, into=str(into))
        print("downloaded:", reply.get("files"))
        return 0

    ids = args.ids or None
    if args.retry:
        ids = failed_ids(args.retry)
        print(f"{len(ids)} failed jobs to retry")
        if not ids:
            return 0
    extra = ["--functional", args.functional] + (["--d3bj"] if args.d3bj else [])
    out = build_pack(args.smoke, ids, args.only, extra)
    if not args.host and not args.dry_run:
        raise SystemExit("give --host NAME (a Job Manager host) or set PMEFF26_HOST")
    print("pack:", out / "pack.tgz")
    if args.dry_run:
        return 0
    job = client().submit(
        host=args.host,
        name="pmeff26-ref-smoke" if args.smoke else "pmeff26-ref-retry" + level.SUFFIX if ids else "pmeff26-ref-" + "-".join(args.only) + level.SUFFIX if args.only else "pmeff26-reference" + level.SUFFIX,
        files=[str(out / "pack.tgz"), str(Path(__file__).with_name("run_pack.py"))],
        command="python3 run_pack.py pack.tgz",
        cpus_per_task=8,
        memory="32GB",
        fetch_globs=["results.tgz", "progress.log"],
        auto_download=False,
        **({"force_run": True} if args.force else {}),
    )
    job = job.get("job", job)
    print("submitted job", job["id"], "to", args.host)
    if not args.smoke:  # build_parameters reads this list
        with open(level.JOBS_FILE, "a", encoding="ascii") as fh:
            fh.write(job["id"] + " " + job.get("name", "") + chr(10))
    print("fetch results with: python -m fitting.run_all --fetch", job["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
