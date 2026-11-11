"""PMEFF26 public engine API: RDKit molecule in, energies and geometries out.

Thin boundary over :mod:`.assign` (typing + parameter lookup from the external
parameter file) and :mod:`.core` (energy, gradient, optimizer, vibrations).
Molecules the parameter set does not cover raise
:class:`~pmeff26_plugin.parameters.ParameterError`; nothing is approximated.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .assign import Assignment, assign
from .core import (  # noqa: F401  (re-exported)
    OptimizationCancelled,
    OptimizationProgress,
    OptimizeResult,
    System,
    energy_and_gradient,
    energy_components,
    hessian,
    optimize,
    vibrational_analysis,
)
from .parameters import (  # noqa: F401  (re-exported)
    MissingParameterError,
    ParameterError,
    ParameterSet,
    UnsupportedAtomError,
    default_parameters,
)

logger = logging.getLogger(__name__)


def conformer_coords(mol: Any) -> Optional[np.ndarray]:
    """Return the (N, 3) coordinates of the molecule's conformer, or None."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


def parameterize(mol: Any, params: Optional[ParameterSet] = None) -> Assignment:
    """Type *mol* and assign every parameter (raises if any is missing)."""
    return assign(mol, params)


def compute_energy_components(mol: Any, params: Optional[ParameterSet] = None):
    """Per-term energies in kcal/mol (plus ``total``), or None without 3D."""
    coords = conformer_coords(mol)
    if coords is None:
        return None
    return energy_components(coords, parameterize(mol, params).system)


def compute_energy(mol: Any, params: Optional[ParameterSet] = None) -> Optional[float]:
    comp = compute_energy_components(mol, params)
    return None if comp is None else comp["total"]


def check_minimum(
    mol: Any,
    params: Optional[ParameterSet] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """Harmonic frequencies (cm^-1) at the current geometry, or None without 3D."""
    coords = conformer_coords(mol)
    if coords is None:
        return None
    return vibrational_analysis(
        coords,
        parameterize(mol, params).system,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def optimize_rdkit_mol(
    mol: Any,
    max_iter: int = 2000,
    f_tol: float = 1e-3,
    params: Optional[ParameterSet] = None,
    progress_callback: Optional[Callable[[OptimizationProgress], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, Optional[OptimizeResult]]:
    """Optimize the conformer of *mol* in place.

    Returns ``(False, None)`` without a conformer, ``(True, None)`` for a
    single atom, else ``(True, OptimizeResult)``. The conformer is written
    only after the optimizer returns, so cancellation keeps the input geometry.
    """
    coords = conformer_coords(mol)
    if coords is None:
        return False, None
    if mol.GetNumAtoms() < 2:
        return True, None
    system = parameterize(mol, params).system
    new_coords, result = optimize(
        coords,
        system,
        max_iter=max_iter,
        f_tol=f_tol,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    if not np.all(np.isfinite(new_coords)):
        logger.error("PMEFF26: optimization produced non-finite coordinates.")
        return False, result
    from rdkit.Geometry import Point3D

    conf = mol.GetConformer()
    for idx, (x, y, z) in enumerate(new_coords):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    return True, result
