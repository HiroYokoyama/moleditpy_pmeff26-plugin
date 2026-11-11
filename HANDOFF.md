# PMEFF26 handoff

State on 2026-11-12. Read this first when picking the project up again.

## Where things stand

| Part | State |
|---|---|
| Engine (`pmeff26_plugin/core.py`, `assign.py`, `parameters.py`, `forcefield.py`) | Done, tested |
| Plugin wiring (`pmeff26_plugin/__init__.py`) | Done; GUI dialogs not re-tested after the rewrite |
| Parameter file (`pmeff26_plugin/parameters/pmeff26.json`) | **Unfitted skeleton**: types + vdW only, so every molecule is refused |
| Reference calculations | **Running** on host `1J1NFIR` (GAMESS, wB97X-D/def2-TZVP) |
| Result extraction (`fitting/collect.py`) | Done, tested on real GAMESS output |
| Fit (`fitting/fit.py`) | Done, tested on synthetic data and a 3-molecule real smoke set |
| pi-electron term | In the engine and tested, but **not fitted** (`pi.enabled` is false) |
| GitHub | Private repo `HiroYokoyama/moleditpy_pmeff26-plugin`; CI is manual-only |

## What to do next, in order

Everything runs from the repo root with the MoleditPy Job Manager open and its
Local API switched on.

1. **Check progress** of the two reference jobs in the Job Manager
   (`pmeff26-reference`, `pmeff26-ref-mt`). Their ids are in `fitting/jobs.txt`.

2. **Retry the fast failures** once the main job has finished. GAMESS rejected
   exactly linear/planar starting geometries; the inputs are fixed now:

   ```bash
   python -m fitting.run_all --retry 4f16990769e4
   ```

   The retry job id is appended to `fitting/jobs.txt` automatically.

3. **Build the parameter file** (can be run at any time, also while jobs are
   still running; it uses whatever has finished):

   ```bash
   python -m fitting.build_parameters
   ```

   This downloads the results, writes `reference/reference_data.json`, fits,
   and writes `pmeff26_plugin/parameters/pmeff26.json` and
   `reference/fit_report.md`.

4. **Read `reference/fit_report.md`.** The *validate* row is the accuracy
   figure (those molecules are never fitted). Also check the lists of terms
   left out, molecules not covered and reference records not used.

5. **Commit the fitted parameter file**, update the README status block with
   the validation numbers, restore the CI triggers in
   `.github/workflows/test.yml`, and bump/tag only when asked.

## Commands

| Command | What it does |
|---|---|
| `python -m fitting.run_all` | Build all GAMESS inputs and submit them as one job |
| `python -m fitting.run_all --only mt` | Submit one set (`tr`, `va`, `sc`, `mt`) |
| `python -m fitting.run_all --retry JOB_ID` | Resubmit only the jobs that failed |
| `python -m fitting.build_parameters` | Download + extract + fit, in one go |
| `python -m fitting.collect [JOB_ID ...]` | Download + extract only |
| `python -m fitting.fit` | Fit only, from `reference/reference_data.json` |
| `python -m fitting.init_parameters --force` | Reset the parameter file to the unfitted skeleton |
| `python -m pytest tests -q -n auto` | Tests (29 pass) |

## Known gaps and risks

- **Run time.** GAMESS has no analytic DFT Hessian and no density fitting.
  After 2.5 h, 55 of 549 main-set jobs (the smallest) were done. The full set
  will take many days on one 8-core machine; the metal set queues behind it.
- **Metal jobs are untested.** No metal GAMESS input has run yet. SCF
  convergence for the Cr, Ni, Pd and Pt complexes is not verified.
- **pi term not fitted.** `fit.py` fits the conventional terms only, keyed by
  formal bond order. Fitting the pi parameters (`beta`, `delta`, `rref`,
  `pi_alpha`) is a nonlinear problem and is not written.
- **No cross terms** (stretch-bend etc.). On the 3-molecule smoke fit the
  frequency RMSD was 116 cm^-1 while geometry was 0.0007 A; expect geometry to
  be much better than frequencies.
- **vdW is not fitted** (UFF values, hard-coded from RDKit's table) and there
  is no intermolecular reference data. Dispersion is only the r^-6 part of the
  Lennard-Jones term.
- **Torsion fits use DFT scan geometries** with force-field single points, not
  force-field-relaxed scans.
- **Metal scope.** Neutral closed-shell species only; no pi-complexes, no
  5-coordinate centres, cis isomers only for Pd/Pt, no held-out metal
  validation molecules.
- **Reference level.** wB97X-D, not wB97X-D3(BJ) (GAMESS does not have it).
  ORCA was deliberately not used (licence question).
- **GUI.** The old Qt dialog tests were removed with the rewrite; the
  calculation and result dialogs are reused unchanged but were not re-run.

## Layout

```
pmeff26_plugin/        plugin package (what gets released)
  core.py              energy, gradient, optimizer, vibrations (NumPy only)
  assign.py            SMARTS typing + parameter lookup; raises if anything is missing
  parameters.py        parameter-file loader, key formats, errors
  parameters/pmeff26.json   the parameter file
fitting/               reference-data and fitting tools (not released)
  dataset.py           molecule sets: TRAIN, VALIDATE, SCANS, METAL
  make_pack.py         GAMESS inputs;  run_pack.py  remote driver
  run_all.py           submit;  collect.py  extract;  fit.py  fit
  build_parameters.py  collect + fit in one command
reference/             downloaded results and reports (git-ignored)
```
