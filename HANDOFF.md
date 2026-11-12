# PMEFF26 handoff

State on 2026-11-12. Read this first when picking the project up again.

## Where things stand

| Part | State |
|---|---|
| Engine (`pmeff26_plugin/core.py`, `assign.py`, `parameters.py`, `forcefield.py`) | Done, tested |
| Plugin wiring (`pmeff26_plugin/__init__.py`) | Done; GUI dialogs not re-tested after the rewrite |
| Parameter file (`pmeff26_plugin/parameters/pmeff26.json`) | **Unfitted skeleton**: types + vdW only, so every molecule is refused |
| Reference calculations | **Running** on the compute host (GAMESS, wB97X-D/def2-TZVP) |
| Result extraction (`fitting/collect.py`) | Done, tested on real GAMESS output |
| Fit (`fitting/fit.py`) | Done, tested on synthetic data and a 3-molecule real smoke set |
| pi-electron term | Engine and fit done, tested on synthetic data; not yet run on real DFT data |
| GitHub | Private repo `HiroYokoyama/moleditpy_pmeff26-plugin`; CI is manual-only |

## What to do next, in order

Everything runs from the repo root with the MoleditPy Job Manager open and its
Local API switched on. Submitting commands need the Job Manager host name:
set `PMEFF26_HOST` or pass `--host NAME`.

1. **Check progress** of the reference jobs in the Job Manager
   (`pmeff26-reference`, `pmeff26-ref-mt`, `pmeff26-ref-retry`,
   `pmeff26-ref-dm`). Their ids are in `fitting/jobs.txt`; the host runs them
   one after another.

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
| `python -m fitting.run_all --only mt` | Submit one set (`tr`, `va`, `sc`, `mt`, `dm`) |
| `python -m fitting.run_all --ids ID ... --force` | Submit chosen calculations only; `--force` jumps the host queue |
| `python -m fitting.run_all --retry JOB_ID` | Resubmit only the jobs that failed |
| `python -m fitting.build_parameters` | Download + extract + fit, in one go |
| `python -m fitting.collect [JOB_ID ...]` | Download + extract only |
| `python -m fitting.fit` | Fit only, from `reference/reference_data.json` |
| `python -m fitting.fit --no-pi` | Same, with the pi term off (bonds keyed by formal order) |
| `python -m fitting.init_parameters --force` | Reset the parameter file to the unfitted skeleton |
| `python -m pytest tests -q -n auto` | Tests (34 pass) |

## Another DFT level

Set `PMEFF26_LEVEL` to keep a second reference set fully separate
(`reference-<name>/`, `fitting/jobs-<name>.txt`, `parameters/pmeff26-<name>.json`):

```bash
PMEFF26_LEVEL=b3lyp python -m fitting.run_all --functional B3LYP --d3bj
```

```bash
PMEFF26_LEVEL=b3lyp python -m fitting.build_parameters
```

In PowerShell set it first with `$env:PMEFF26_LEVEL = "b3lyp"`. The plugin only
loads `pmeff26.json`; copy a level's file over it to adopt that level. This
refits charges, bonded terms and, when the `dm` set has run at that level,
the van der Waals values.

## Known gaps and risks

- **Run time.** GAMESS has no analytic DFT Hessian and no density fitting.
  After 2.5 h, 55 of 549 main-set jobs (the smallest) were done. The full set
  will take many days on one 8-core machine; the metal set queues behind it.
- **Metal jobs are untested.** No metal GAMESS input has run yet. SCF
  convergence for the Cr, Ni, Pd and Pt complexes is not verified.
- **pi term is fitted but unproven on real data.** `fit.py` fits it by default
  (`fit_pi`: one `beta` per element pair, one `pi_alpha` per element/electron
  count relative to carbon, one global `delta`; `rref` is the mean DFT length
  of the pair's pi bonds). Bonds between pi types are then keyed `a~b`. On
  synthetic conjugated hydrocarbons it beats the conventional fit (held-out
  hexatriene bond RMSD 0.0005 vs 0.0073 A). On the real set, compare the
  validation rows of a normal run and a `--no-pi` run and keep the better
  file. The pi fit is the slow step (estimated tens of minutes on the full
  set). Systems with a half-filled degenerate pi level (cyclobutadiene) are
  not smooth and are not handled specially.
- **No cross terms** (stretch-bend etc.). On the 3-molecule smoke fit the
  frequency RMSD was 116 cm^-1 while geometry was 0.0007 A; expect geometry to
  be much better than frequencies.
- **vdW refit is queued, not done.** The dimer set (`fitting/dimers.py`: 47
  pairs of 24 probes, 3 orientations x 6 separations, job `pmeff26-ref-dm`)
  gives counterpoise-corrected interaction energies; `fit.py` refits
  (rmin/2, eps) per element plus polar hydrogen (`hx`) to them, restrained to
  the UFF start values. Metals and anything without dimer data keep UFF.
  Verified so far on one water-dimer point only (-0.96 kcal/mol, BSSE 0.41).
  Dispersion is only the r^-6 part of the Lennard-Jones term.
- **Counterpoise uses wB97X, not wB97X-D.** GAMESS crashes on ghost atoms in
  its empirical-dispersion code and ignores `DC=.FALSE.` for wB97X-D, so the
  BSSE correction is computed with the parent functional.
- **Privacy.** Downloaded outputs are scrubbed of the account and host name
  before commit. Commit `77854f9` in the history holds unscrubbed outputs; the
  owner has accepted that (host name and home path may be public).
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
  dataset.py           molecule sets: TRAIN, VALIDATE, SCANS, METAL;  dimers.py  vdW dimer set
  make_pack.py         GAMESS inputs;  run_pack.py  remote driver
  run_all.py           submit;  collect.py  extract;  fit.py  fit
  build_parameters.py  collect + fit in one command
reference/             downloaded GAMESS outputs (raw/*.out.gz), extracted data and reports;
                       committed, except job packs and the large .dat orbital files
```
