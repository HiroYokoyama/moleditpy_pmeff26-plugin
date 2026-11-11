"""PMEFF26 - a DFT-fitted force field for MoleditPy.

PMEFF26 is a typed molecular-mechanics force field whose every parameter is
read from an external file (``parameters/pmeff26.json``) fitted to
wB97X-D/def2-TZVP reference calculations, plus a variational pi-electron
term for conjugated systems. It covers neutral closed-shell molecules of
H, B, C, N, O, F, Si, P, S, Cl, Br and I. A molecule that needs a parameter
the file does not contain is refused: there is no generic fallback.

Registers an **Optimize 3D** method, a **single-point energy** tool, a
**vibrational analysis** tool (frequencies in cm^-1) and a settings entry.
See ``core.py`` for the energy function and ``assign.py`` for the typing.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

PLUGIN_NAME = "PMEFF26 Plugin"
PLUGIN_VERSION = "0.1.0"
# Must equal the GitHub username (the moleditpy registry enforces this).
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "PMEFF26 - a DFT-fitted, typed force field with a variational pi-electron "
    "term for main-group organic molecules (H, B, C, N, O, F, Si, P, S, Cl, "
    "Br, I). Adds a geometry optimizer, a single-point energy tool and a "
    "vibrational analysis. Molecules outside the fitted parameter set are "
    "refused rather than approximated."
)
PLUGIN_CATEGORY = "Optimization"
PLUGIN_TAGS = ["Optimization"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
PLUGIN_DEPENDENCIES = ["numpy", "rdkit"]

_OPT_METHOD_NAME = f"PMEFF26 (v{PLUGIN_VERSION})"
_MAX_ITER = 2000
_SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
_DEFAULT_SETTINGS = {"max_iterations": _MAX_ITER}

logger = logging.getLogger(__name__)


def load_settings() -> dict:
    """Read settings.json, falling back to defaults on any problem."""
    settings = dict(_DEFAULT_SETTINGS)
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            settings.update(data)
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logger.warning("PMEFF26: could not read %s; using defaults.", _SETTINGS_FILE)
    return settings


def save_settings(settings: dict) -> None:
    """Write settings.json atomically (temp file + replace)."""
    try:
        fd, tmp = tempfile.mkstemp(dir=str(_SETTINGS_FILE.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
        os.replace(tmp, _SETTINGS_FILE)
    except OSError:
        logger.exception("PMEFF26: failed to write %s", _SETTINGS_FILE)


def _iteration_budget() -> int:
    try:
        return max(1, min(100000, int(load_settings().get("max_iterations", _MAX_ITER))))
    except (TypeError, ValueError, OverflowError):
        return _MAX_ITER


def initialize(context):
    """Register PMEFF26's optimization method, analysis tools and settings."""
    context.register_optimization_method(
        _OPT_METHOD_NAME, lambda mol: _optimize(mol, context)
    )
    context.add_analysis_tool(
        "PMEFF26 Single-Point Energy", lambda: _show_energy(context)
    )
    context.add_analysis_tool(
        "PMEFF26 Vibrational Analysis", lambda: _check_minimum(context)
    )
    context.add_menu_action(
        "Settings/PMEFF26 Setting",
        lambda: _open_settings(context),
        text="PMEFF26 Settings…",
    )


def _host_window(context):
    getter = getattr(context, "get_main_window", None)
    if callable(getter):
        return getter()
    return getattr(context, "main_window", None)


def _open_settings(context) -> None:
    """Ask for the optimization step budget (the only user setting)."""
    try:
        from PySide6.QtWidgets import QInputDialog
    except ImportError:
        return
    settings = load_settings()
    value, ok = QInputDialog.getInt(
        _host_window(context),
        "PMEFF26 Settings",
        "Maximum optimization steps:",
        _iteration_budget(),
        1,
        100000,
    )
    if ok:
        settings["max_iterations"] = int(value)
        save_settings(settings)
        _status(context, "PMEFF26 settings saved.", 3000)


def _refused(context, exc) -> None:
    """Report a molecule the parameter set does not cover."""
    _status(context, str(exc), 10000)
    _display_result(
        context,
        "PMEFF26: molecule not supported",
        str(exc),
        ["Missing parameter"],
        [[m] for m in getattr(exc, "missing", [])],
        "PMEFF26 only computes molecules whose every term was fitted to "
        "reference DFT data. It does not substitute approximate parameters.",
    )


def _optimize(mol, context) -> bool:
    """Relax *mol* in place with PMEFF26. Returns True on success."""
    from .forcefield import OptimizationCancelled, ParameterError, optimize_rdkit_mol

    max_iter = _iteration_budget()
    try:
        working, output = _calculate(
            context,
            "PMEFF26 Optimization",
            mol,
            lambda copy, report, cancel: optimize_rdkit_mol(
                copy, max_iter=max_iter, progress_callback=report, cancel_callback=cancel
            ),
        )
        success, result = output
    except OptimizationCancelled:
        _status(context, "PMEFF26: cancelled; original geometry preserved.", 5000)
        return False
    except ParameterError as exc:
        _refused(context, exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF26 optimization failed")
        _status(context, f"PMEFF26 optimization failed: {exc}", 5000)
        return False

    if not success:
        _status(context, "PMEFF26: no 3D geometry to optimize.", 4000)
        return False

    from rdkit import Chem

    current = getattr(context, "current_molecule", None)
    if isinstance(current, Chem.Mol) and current is not mol:  # pylint: disable=no-member
        _status(context, "PMEFF26: molecule changed during calculation; result discarded.", 5000)
        return False
    if mol is not None and working.GetNumConformers():
        target, source = mol.GetConformer(), working.GetConformer()
        for i in range(mol.GetNumAtoms()):
            target.SetAtomPosition(i, source.GetAtomPosition(i))

    if result is not None:
        state = "converged" if result.converged else "stopped (max iterations)"
        _status(
            context,
            f"PMEFF26 {state} in {result.steps} steps "
            f"(E = {result.energy:.3f} kcal/mol, |F|max = {result.max_force:.4f}).",
            5000,
        )
    else:
        _status(context, "PMEFF26: nothing to optimize (single atom).", 3000)
    return True


def _energy_report(mol):
    """Worker-side: components plus the DFT-referenced estimate, or None."""
    from .forcefield import conformer_coords, energy_components, parameterize

    coords = conformer_coords(mol)
    if coords is None:
        return None
    assignment = parameterize(mol)
    comp = energy_components(coords, assignment.system)
    comp["energy_offset"] = assignment.energy_offset
    return comp


def _show_energy(context) -> None:
    """Report the PMEFF26 single-point energy of the current molecule."""
    from .forcefield import OptimizationCancelled, ParameterError

    mol = context.current_molecule
    if mol is None:
        _status(context, "PMEFF26: no molecule loaded.", 3000)
        return
    try:
        _working, comp = _calculate(
            context,
            "PMEFF26 Single-Point Energy",
            mol,
            lambda copy, _report, _cancel: _energy_report(copy),
        )
    except OptimizationCancelled:
        return
    except ParameterError as exc:
        _refused(context, exc)
        return
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF26 energy evaluation failed")
        _status(context, f"PMEFF26 energy failed: {exc}", 5000)
        return
    if comp is None:
        _status(context, "PMEFF26: molecule has no 3D coordinates.", 4000)
        return

    offset = comp.pop("energy_offset")
    summary = f"Force-field energy: {comp['total']:.4f} kcal/mol"
    if offset is not None:
        estimate = offset + comp["total"] / 627.5095
        summary += f"\nEstimated DFT energy: {estimate:.6f} Eh"
    rows = [[k, f"{v:.4f}"] for k, v in comp.items() if k != "total"]
    _status(context, summary.replace("\n", "; "), 8000)
    _display_result(
        context,
        "PMEFF26 Single-Point Energy",
        summary,
        ["Term", "Energy (kcal/mol)"],
        rows,
        "The force-field energy is a strain energy relative to the fitted "
        "reference values. The estimated DFT energy adds fitted per-type "
        "increments and is only as accurate as the fit statistics in the "
        "parameter file.",
    )


def _check_minimum(context) -> None:
    """Report harmonic frequencies and whether the geometry is a minimum."""
    from .forcefield import OptimizationCancelled, ParameterError, check_minimum

    mol = context.current_molecule
    if mol is None:
        _status(context, "PMEFF26: no molecule loaded.", 3000)
        return
    try:
        _working, result = _calculate(
            context,
            "PMEFF26 Vibrational Analysis",
            mol,
            lambda copy, report, cancel: check_minimum(
                copy, progress_callback=report, cancel_callback=cancel
            ),
        )
    except OptimizationCancelled:
        return
    except ParameterError as exc:
        _refused(context, exc)
        return
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF26 vibrational analysis failed")
        _status(context, f"PMEFF26 vibrational analysis failed: {exc}", 5000)
        return
    if result is None:
        _status(context, "PMEFF26: molecule has no 3D coordinates.", 4000)
        return

    stationary = result["max_force"] < 1e-2
    if result["num_imaginary"]:
        verdict = f"{result['num_imaginary']} imaginary mode(s): not a minimum."
    elif stationary:
        verdict = "No imaginary modes: a minimum of the PMEFF26 surface."
    else:
        verdict = "No imaginary modes, but the geometry is not optimized (residual forces)."
    _status(context, f"PMEFF26 vibrational analysis: {verdict}", 8000)
    rows = [
        [str(i), f"{freq:.1f}", "Imaginary" if freq < -result["imaginary_threshold"] else "Real"]
        for i, freq in enumerate(result["frequencies"], 1)
    ]
    _display_result(
        context,
        "PMEFF26 Vibrational Analysis",
        f"{verdict}\nMaximum force: {result['max_force']:.5f} kcal/mol/Å",
        ["Mode", "Wavenumber (cm⁻¹)", "Classification"],
        rows,
        "Harmonic frequencies of the PMEFF26 surface with real atomic masses. "
        "Negative values are imaginary modes.",
    )


def _calculate(context, title, mol, operation):
    """Run *operation* on a copy of the molecule, off the GUI thread if possible."""
    from rdkit import Chem

    working = Chem.Mol(mol) if mol is not None else None  # pylint: disable=no-member

    def job(report, cancel):
        return operation(working, report, cancel)

    try:
        from .calculation_dialog import run_job
    except ImportError:  # headless installations need no Qt
        output = job(lambda *_args: None, lambda: False)
    else:
        output = run_job(title, job, _host_window(context))
    return working, output


def _display_result(context, title, summary, headers, rows, note=""):
    try:
        from .result_dialog import show_result
    except ImportError:
        return True
    return show_result(title, summary, headers, rows, note, _host_window(context))


def _status(context, message: str, timeout: int = 3000) -> None:
    show = getattr(context, "show_status_message", None)
    if callable(show):
        show(message, timeout)
    else:  # pragma: no cover - only hit with a bare context
        logger.info(message)
