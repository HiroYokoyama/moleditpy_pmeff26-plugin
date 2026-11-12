# PMEFF26

[![Tests](https://github.com/HiroYokoyama/moleditpy_pmeff26-plugin/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_pmeff26-plugin/actions/workflows/test.yml)

A [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor) plugin
providing **PMEFF26**, a DFT-fitted molecular force field that runs entirely in
Python (NumPy + RDKit).

> **Status: engine complete, parameters not fitted yet.** The reference
> wB97X-D/def2-TZVP calculations are still running. Until
> `pmeff26_plugin/parameters/pmeff26.json` is filled by the fit, PMEFF26
> refuses every molecule. No accuracy figures exist yet; they will come from
> the held-out validation set and be reported here.

## What PMEFF26 is

- **A typed force field.** Atoms are typed by an ordered SMARTS list; bonds,
  angles, torsions, out-of-plane terms, Lennard-Jones and bond-charge-increment
  electrostatics are looked up by type. Energies are in kcal/mol, frequencies
  in cm⁻¹.
- **All parameters in an external file.** `parameters/pmeff26.json` holds the
  atom types and every table. The code contains no chemical constants.
- **No fallback.** A molecule that needs a term the file does not contain is
  refused with a list of what is missing. PMEFF26 never substitutes a generic
  estimate, because an energy built on a guessed parameter is not accurate.
- **A variational π-electron term.** Conjugated atoms carry a small
  Hückel-type Hamiltonian whose resonance integrals depend on bond length and
  twist; its energy, with exact analytic gradients, is part of the force
  field. Bond-length alternation, aromatic stabilization and rotation barriers
  about conjugated bonds come from that one term, so all π bonds between two
  atom types share a single σ parameter instead of one per bond order.
- **DFT-referenced energies.** Fitted per-type increments turn the strain
  energy into an estimate of the DFT total energy, so isomers can be compared.

Scope of the first parameter set: neutral, closed-shell molecules of H, B, C,
N, O, F, Si, P, S, Cl, Br and I, plus a first metal set (Li, Na, K, Mg, Ca, Al,
Ge, Sn, Ti, Cr, Ni, Cu, Zn, Pd, Pt, Hg) in the coordination numbers that were
calculated. Ions, radicals, open-shell metal centres, pi-complexes and
5-coordinate centres are refused. Square-planar and octahedral centres use a
periodic angle term with minima at 90 and 180 degrees.

## Where PMEFF26 stands

PMEFF26 is a classical force field with a fixed functional form. It is meant
to sit between the generic force fields (UFF, MMFF94) and DFT: geometries and
conformer energies closer to DFT than a generic force field gives, in a
fraction of a second on a CPU. It is a tool for preparing DFT work (starting
geometries, conformer pre-screening), not a replacement for DFT.

It is **not** a machine-learned potential and does not aim at that accuracy.
A machine-learned potential learns the potential surface itself with a very
large number of parameters; PMEFF26 only places DFT-fitted constants into a
set of fixed expressions, and what those expressions cannot describe stays
wrong however much data is added.

| | PMEFF26 | Machine-learned potentials |
|---|---|---|
| Equilibrium geometries | Comparable accuracy is the goal | Good |
| Vibrational frequencies | Helped by the stretch-bend term; expected to stay behind | Good |
| Conformer energies | Usable within the fitted torsion classes | Good |
| Bond breaking, reactions, transition states | Not possible (harmonic bonds) | Possible inside the training domain |
| Polarization, charge transfer | Not possible (fixed charges) | Possible in some models |
| Chemistry outside the parameter set | Refused | Runs, with unknown error |
| Speed on a CPU | Fast | Slow; normally needs a GPU |
| Dependencies | NumPy and RDKit | A deep-learning framework and model weights |

Compared with a generic force field, PMEFF26 adds the variational π term,
stretch-bend coupling, parameters fitted to one consistent DFT level, and a
refusal instead of a guess when a parameter is missing. The comparison above
describes the design; measured accuracy will be reported here once the fit
has run on the reference data.

## What it adds to MoleditPy

| Feature | Where |
|---|---|
| **PMEFF26** geometry optimization | Right-click the *Optimize 3D* button |
| **PMEFF26 Single-Point Energy** (per-term table, estimated DFT energy) | *Analysis* menu |
| **PMEFF26 Vibrational Analysis** (cm⁻¹, minimum/saddle verdict) | *Analysis* menu |
| **PMEFF26 Settings…** (optimization step budget) | *Settings* menu |

Calculations run in a background thread with a cancel button; cancelling
keeps the original geometry.

## Known limitations

- The Lennard-Jones parameters start from the per-element UFF values
  (hard-coded from RDKit's UFF table, `Code/ForceField/UFF/Params.cpp`). The
  main-group elements and polar hydrogen are refit to counterpoise-corrected
  DFT dimer curves once that reference set has finished; metals keep UFF.
- Dispersion is only the r⁻⁶ part of that Lennard-Jones term; there is no
  separate D3-style correction.
- The metal set has no held-out validation molecules yet.
- Charges are fixed bond-charge increments; there is no polarization.
- The reference functional is ωB97X-D (GAMESS has no ωB97X-D3(BJ)).

## Fitting

The reference pipeline lives in `fitting/`. With the MoleditPy Job Manager
running (Local API on), two commands take it from nothing to a fitted
parameter file:

```bash
python -m fitting.run_all
```

```bash
python -m fitting.build_parameters
```

The first builds every GAMESS input (ωB97X-D/def2-TZVP) and submits them as one
job. The second downloads whatever has finished, extracts energies,
geometries, Hessians, dipoles and torsion scans, fits the parameters and
writes the parameter file plus `reference/fit_report.md`. The fit matches the
force-field gradient and Hessian to DFT at every training minimum by linear
least squares; held-out validation molecules are only used for the report.
The π term is not fitted by this script yet.

See [HANDOFF.md](HANDOFF.md) for the current state, the remaining steps and
the known gaps.

## Development

```bash
python -m pytest tests/ -q -n auto
```

```bash
pylint pmeff26_plugin/
```

## Origin and license

PMEFF26 started from PMEFF 2.0.0 (same author); the engine was rewritten.
LGPL-3.0-or-later. See [LICENSE](LICENSE), [COPYING](COPYING) and
[LICENSING.md](LICENSING.md).
