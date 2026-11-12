# PMEFF26 handoff

State on 2026-11-12. This file is written so that another person or model can
finish the project without the earlier conversation. Read it top to bottom
once; sections 3 and 4 are the work that is left.

## 1. What this project is

PMEFF26 is a MoleditPy plugin: a force field fitted to DFT that runs in pure
Python (NumPy + RDKit). The code is finished. What is missing is the fitted
parameter file, because the DFT reference calculations are still running on
the owner's workstation.

Until the fit is run, `pmeff26_plugin/parameters/pmeff26.json` is an
**unfitted skeleton** (atom types + UFF van der Waals only) and the plugin
refuses every molecule. That is intended behaviour, not a bug.

## 2. Rules set by the owner (do not break these)

- **Never use ORCA** for reference data (licence concern). The reference
  program is **GAMESS**, run on the owner's workstation through the MoleditPy
  Job Manager. Job Manager host name: `1J1NFIR`.
- **No fallback parameters.** A molecule that needs a term missing from the
  parameter file is refused (`MissingParameterError`). Never add generic
  estimates or a fallback to the older PMEFF plugin.
- **All parameters live in the external JSON file**, never in code. The one
  exception is the UFF van der Waals start table in
  `fitting/init_parameters.py` (copied from RDKit `Code/ForceField/UFF/Params.cpp`).
- **Commit often**, small commits, pushed to `main`. No emoji anywhere. End
  commit messages with the co-author line your tool is configured to add.
- **Do not bump the version, tag or release** unless the owner asks.
- **Scripts must run unattended.** The owner's AI quota is limited: prefer
  one command that does everything over interactive steps.
- GAMESS outputs are committed exactly as downloaded (host name and home path
  in them are accepted as public). The repo is public.
- Tests: `python -m pytest tests -q -n auto` (38 pass). Line endings do not
  matter in this repo.

## 3. Current state

| Part | State |
|---|---|
| Engine (`pmeff26_plugin/`) | Done, tested |
| Reference pipeline (`fitting/`) | Done, tested on synthetic data and a few real GAMESS outputs |
| Parameter file | Unfitted skeleton |
| Reference calculations | Running / queued on `1J1NFIR` (see below) |
| pi-electron term, stretch-bend terms, vdW refit | Fit code done, tested on synthetic data only |
| Plugin GUI (dialogs) | Not re-run since the engine rewrite |
| GitHub | Public, `HiroYokoyama/moleditpy_pmeff26-plugin`; CI runs on push and pull request |

Reference jobs (ids are in `fitting/jobs.txt`; the host runs them one after
another, in this order):

| Job id | Name | Content | State on 2026-11-12 |
|---|---|---|---|
| `4f16990769e4` | `pmeff26-reference` | 411 training + 75 validation molecules, 63 torsion scans | Running; 56 of 549 done after the first hours. Expect many days |
| `c67e6569e89f` | `pmeff26-ref-mt` | 64 metal complexes | Queued. **No metal input has ever run** |
| `faa0c635e664` | `pmeff26-ref-retry` | 29 molecules that failed in the main run | Queued |
| `933992e9982d` | `pmeff26-ref-dm` | 47 dimer pairs for the vdW refit (2,538 dimer points + monomers) | Queued |

Level of theory: wB97X-D/def2-TZVP, RHF closed shell, ECPs for I/Pd/Sn/Pt/Hg.
GAMESS has no analytic DFT Hessian and no density fitting, hence the run time.

## 4. What is left to do, in order

Everything runs from the repo root. Steps that talk to the workstation need
the **MoleditPy Job Manager application open with its Local API switched on**
(ask the owner if a connection is refused) and the host name:
`PMEFF26_HOST=1J1NFIR` or `--host 1J1NFIR`.

### Step 1. Check progress

Either look at the jobs in the Job Manager window, or from the sibling repo
`../moleditpy_job_manager`:

```bash
python -m job_manager.api_client log 4f16990769e4 --file progress.log --lines 20
```

`progress.log` has one line per finished calculation (`<id> done` or
`<id> FAILED`). You do not have to wait for everything: step 2 works on
whatever has finished.

### Step 2. Build the parameter file

```bash
python -m fitting.build_parameters
```

This downloads every finished result of the jobs in `fitting/jobs.txt` into
`reference/raw/`, writes `reference/reference_data.json` and
`reference_hessians.npz`, fits, and writes
`pmeff26_plugin/parameters/pmeff26.json` and `reference/fit_report.md`.
It is safe to re-run at any time. `--no-fetch` skips the download.

The fit has **never run on more than a handful of real molecules**. On the
full set expect it to be slow (the pi step alone is estimated at tens of
minutes; nothing was measured) and expect first-time problems. Fix them in
`fitting/fit.py`, add a test, commit.

### Step 3. Read `reference/fit_report.md`

- The **validate** row is the accuracy figure (those molecules are never
  fitted). The train row only shows the fit converged.
- "Terms left out": parameters the data did not determine. Molecules needing
  them are refused. A long list means the training set lacks coverage.
- "molecules not covered": validation/training molecules the final file
  refuses, with the missing terms.
- "Reference records not used": failed jobs, changed connectivity, imaginary
  frequencies.

### Step 4. Decide on the optional terms

Run the fit three more times and compare the **validate** rows (bond RMSD,
angle RMSD, frequency RMSD, molecules covered):

```bash
python -m fitting.fit --no-pi
```

```bash
python -m fitting.fit --no-sb
```

```bash
python -m fitting.fit --no-pi --no-sb
```

Each run overwrites the parameter file and the report, so copy the report
aside between runs. Keep the variant with the best validation numbers; if
the pi term or stretch-bend does not clearly help on real data, say so
plainly in the README rather than keeping it for novelty. Finish by
re-running the chosen variant so the committed file matches the report.

### Step 5. Retry failed calculations

After a job has finished, resubmit what failed in it:

```bash
PMEFF26_HOST=1J1NFIR python -m fitting.run_all --retry 4f16990769e4
```

The new job id is appended to `fitting/jobs.txt` automatically. Then repeat
step 2. Known failure causes and fixes are in section 6. A molecule that
keeps failing is simply left out; report it, do not fake data.

### Step 6. Check the metals

When `c67e6569e89f` has run, look at how many of the 64 converged and whether
the optimized structures kept their connectivity (the collector flags this).
There are **no held-out metal molecules**, so metal accuracy is unmeasured:
state that in the README, or add validation complexes to
`fitting/dataset.py` (`METAL`) and submit them with `--only mt --ids ...`.

### Step 7. Test the plugin in MoleditPy

With a fitted file, load the plugin in MoleditPy and run the optimizer,
"PMEFF26 Single-Point Energy" and "PMEFF26 Vibrational Analysis" on a few
molecules. The calculation and result dialogs were reused unchanged from the
older plugin but not re-run after the rewrite. The energy component list now
includes `cross` and `pi`; check the result dialog shows them sensibly.

### Step 8. Finish

- Commit the fitted `pmeff26.json`, `reference/` data and `fit_report.md`.
- Replace the README status block with the real validation numbers and the
  honest list of what is not covered.
- Keep CI green (`.github/workflows/test.yml`: tests on Python 3.11-3.13 and
  pylint with `--fail-under=9.0`).
- Do not tag or release until the owner asks.

**Done means:** a fitted parameter file whose validation numbers are in the
README, all tests green in CI, and the plugin verified once in the GUI.

## 5. Commands

| Command | What it does |
|---|---|
| `python -m fitting.build_parameters` | Download + extract + fit, in one go (`--no-fetch`, `--no-pi`, `--no-sb`) |
| `python -m fitting.collect [JOB_ID ...]` | Download + extract only |
| `python -m fitting.fit` | Fit only, from `reference/reference_data.json` (`--no-pi`, `--no-sb`) |
| `python -m fitting.run_all` | Build all GAMESS inputs and submit them as one job |
| `python -m fitting.run_all --only dm` | Submit one set: `tr` train, `va` validate, `sc` scans, `mt` metals, `dm` dimers |
| `python -m fitting.run_all --retry JOB_ID` | Resubmit only what failed in that job |
| `python -m fitting.run_all --ids ID ... --force` | Submit chosen calculations; `--force` jumps the host queue (small checks only) |
| `python -m fitting.run_all --dry-run` | Build the job pack without submitting |
| `python -m fitting.init_parameters --force` | Reset the parameter file to the unfitted skeleton |
| `python -m pytest tests -q -n auto` | Tests |

A second DFT level can be kept fully separate with `PMEFF26_LEVEL=<name>`
(`reference-<name>/`, `fitting/jobs-<name>.txt`,
`parameters/pmeff26-<name>.json`), for example
`PMEFF26_LEVEL=b3lyp python -m fitting.run_all --functional B3LYP --d3bj`.
The plugin only loads `pmeff26.json`.

## 6. Things that already went wrong (so you do not repeat them)

- **GAMESS aborts with "THERE ARE NOT 5 OR 6 TRANS/ROT MODES"** when the start
  geometry is exactly linear or planar. `make_pack.build_mol` now adds a
  +-0.02 A jitter; the 29 affected molecules are in the retry job.
- **Ghost atoms crash GAMESS's empirical-dispersion code**, and `DC=.FALSE.`
  is ignored for wB97X-D. Counterpoise corrections for the dimers are
  therefore computed with the parent functional wB97X:
  `E_int = (E_AB - E_A - E_B) + (E0_A - E0_A(ghost B)) + (E0_B - E0_B(ghost A))`.
  One water-dimer point was checked: -0.96 kcal/mol, BSSE part 0.41.
- **GAMESS specifics:** launcher `~/qc-progs/bin/gamess name.inp ncores`;
  basis sets are written inline from `fitting/def2_tzvp_gamess.json`
  (`ISPHER=1`); Hessians are semi-numerical (`HSSEND=.TRUE.`) and read from
  the last `$HESS` group of the `.dat` file; torsion scans use DLC
  coordinates with one frozen dihedral, one job per point.
- **Metal inputs** needed charge-separated SMILES for Al adducts, `O=C=[M]`
  carbonyls and geometry hints (`|sqp`, `|oct` in `dataset.METAL`) because
  RDKit cannot embed them unaided.
- **pi term stability:** the resonance integral is the bounded form
  `beta * (1 - tanh(delta * (r - rref)))`; an exponential form collapsed
  bonds. Molecules with a half-filled degenerate pi level (cyclobutadiene)
  are not smooth and are not handled specially.
- **Shell pitfall on this Windows machine:** Python code passed through a bash
  heredoc gets its `\n` escapes mangled, and quotes inside can break the
  heredoc. Write a patch script to a file and run it instead.
- **Job Manager API:** file listings use the reply key `entries`; the
  collector re-downloads an id only if it is not marked `.done` locally.
- **Outputs fetched before 2026-11-12 were scrubbed** of the account and host
  name ("redacted", `/home/user`); later ones are committed as downloaded.
  The numbers are identical either way.

## 7. The model, briefly

Units: kcal/mol, Angstrom, radians internally, cm^-1 for frequencies.
Energy terms (`core.COMPONENTS`): `bond`, `angle`, `cross`, `torsion`,
`improper`, `vdw`, `elec`, `pi`.

- Bonds harmonic; angles harmonic, linear (`k (1 + cos)`), or periodic
  (`(k/n^2)(1 - cos n theta)`) for square-planar/octahedral metals.
- `cross`: stretch-bend `(k1 dr1 + k2 dr2) dtheta` per harmonic angle;
  optional table `stretch_bend` (no entry = no cross term, not a refusal).
- Torsions as Fourier series, generic per central bond (`*|b-c|*`) with the
  barrier divided over the paths; impropers as 2-fold torsions.
- Lennard-Jones 12-6 (Lorentz-Berthelot) + Coulomb with bond-charge
  increments; 1-4 scaling in `scale14`.
- `pi`: a Hueckel Hamiltonian over pi-type atoms,
  `H_ij = -beta (1 - tanh(delta (r - rref))) * twist`, energy
  `2 * sum(occupied eigenvalues) - sum(n_i alpha_i)`, exact gradient. With
  the term enabled, a bond between two pi types is keyed `a~b` whatever its
  formal order. A pi-type atom with no pi-type neighbour is left out.
- DFT-referenced energy increments per atom type and bond (Hartree) turn the
  force-field energy into an estimate of the DFT total energy.

How the fit works (`fitting/fit.py`, in call order inside `fit()`):

1. `fit_charges`: bond-charge increments by linear least squares on DFT
   dipoles (weak Loewdin restraint).
2. `fit_vdw`: Lennard-Jones per element (+ polar H `hx`) on the dimer curves,
   restrained to UFF. Skipped without dimer data.
3. `bonded_system`: one sparse linear system matching force-field gradient
   (target zero, weight 20) and Hessian to DFT at each DFT minimum, after
   subtracting the non-bonded part.
4. `fit_pi`: variable projection over that system. One `beta` per element
   pair, one `pi_alpha` per element/electron count (carbon = 0), one global
   `delta`; `rref` = mean DFT length of the pair's pi bonds.
5. `solve_bonded` then `fit_stretch_bend` (two joint rounds).
6. `fit_scans`: torsion barriers refit to the DFT scans (force-field single
   points on DFT scan geometries, not relaxed scans).
7. `fit_increments`, then `evaluate` on train and validate.

Synthetic-data results so far (the reference was generated by the same
model, so these show the fit works, not real accuracy): pi fit, held-out
hexatriene bond RMSD 0.0005 A against 0.0073 A without the pi term;
stretch-bend constants recovered within 2 kcal/mol/A/rad.

Parameter-file key formats are documented at the top of
`pmeff26_plugin/parameters.py`.

## 8. Known limits to state honestly in the README

- Neutral closed-shell molecules only; no ions beyond formal-charge pairs, no
  radicals, no pi-complexes, no 5-coordinate metal centres; cis isomers only
  for Pd/Pt.
- Reference level is wB97X-D, not wB97X-D3(BJ) (GAMESS lacks it).
- Dispersion is only the r^-6 part of the Lennard-Jones term; metals keep UFF
  van der Waals values.
- No stretch-stretch or bend-bend cross terms; 1-4 scaling is fixed, not
  fitted.
- Geometry is expected to be better than frequencies.

## 9. Layout

```
pmeff26_plugin/        plugin package (what gets released)
  core.py              energy, gradient, optimizer, vibrations (NumPy only)
  assign.py            SMARTS typing + parameter lookup; raises if anything is missing
  parameters.py        parameter-file loader, key formats, errors
  forcefield.py        RDKit-facing helpers used by the plugin
  parameters/pmeff26.json   the parameter file
fitting/               reference-data and fitting tools (not released)
  dataset.py           molecule sets TRAIN, VALIDATE, SCANS, METAL;  dimers.py  vdW dimer set
  gamess.py            GAMESS input writer;  make_pack.py  builds the job pack
  run_pack.py          driver that runs on the workstation (stdlib only)
  run_all.py           submit;  collect.py  download + extract;  fit.py  fit
  build_parameters.py  collect + fit in one command
  init_parameters.py   atom types + UFF vdW table -> unfitted skeleton
  jobs.txt             Job Manager ids of the reference jobs
reference/             raw/ GAMESS outputs (committed), extracted data, fit report;
                       job packs and the large .dat orbital files are git-ignored
tests/                 engine, plugin, fit, pi-fit and stretch-bend tests
```
