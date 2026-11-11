"""Which reference level the fitting tools work on.

The default level is wB97X-D/def2-TZVP. To build a second, independent
reference set with another functional, set the environment variable
``PMEFF26_LEVEL`` to a short name before running any fitting command, e.g.

    PMEFF26_LEVEL=b3lyp python -m fitting.run_all --functional B3LYP --d3bj
    PMEFF26_LEVEL=b3lyp python -m fitting.build_parameters

A named level keeps everything apart from the default one: its data live in
``reference-<name>/``, its job ids in ``fitting/jobs-<name>.txt`` and its fit
is written to ``pmeff26_plugin/parameters/pmeff26-<name>.json`` (the plugin
only loads ``pmeff26.json``; copy the file over to adopt a level).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAME = re.sub(r"[^a-z0-9_-]", "", os.environ.get("PMEFF26_LEVEL", "").lower())
SUFFIX = f"-{NAME}" if NAME else ""
REF = REPO / f"reference{SUFFIX}"
JOBS_FILE = Path(__file__).with_name(f"jobs{SUFFIX}.txt")
PARAMETER_FILE = REPO / "pmeff26_plugin" / "parameters" / f"pmeff26{SUFFIX}.json"
