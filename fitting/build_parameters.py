#!/usr/bin/env python3
"""One command from the compute host to a fitted parameter file.

    python -m fitting.build_parameters              # jobs listed in fitting/jobs.txt
    python -m fitting.build_parameters JOB_ID ...   # these Job Manager jobs
    python -m fitting.build_parameters --no-fetch   # only what is in reference/raw

Downloads every finished result of the reference jobs (the jobs may still be
running), extracts them into ``reference/reference_data.json``, fits the
parameters and writes ``pmeff26_plugin/parameters/pmeff26.json`` plus
``reference/fit_report.md``. Safe to re-run at any time: each run refits from
all results available so far. Needs the Job Manager with its Local API on,
unless ``--no-fetch`` is given.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import collect, fit

JOBS_FILE = Path(__file__).with_name("jobs.txt")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fetch = "--no-fetch" not in argv
    jobs = [a for a in argv if not a.startswith("--")]
    if not jobs and JOBS_FILE.exists():
        jobs = [ln.split()[0] for ln in JOBS_FILE.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    collect.main(jobs if fetch else [])
    return fit.main([])


if __name__ == "__main__":
    sys.exit(main())
