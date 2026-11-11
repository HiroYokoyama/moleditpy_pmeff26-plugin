"""PMEFF26 numeric core: energy, analytic gradient, optimizer, vibrations.

A conventional class-I molecular-mechanics energy function in real units
(kcal/mol, Angstrom, radian, elementary charge, amu):

    E = sum_bonds     1/2 k (r - r0)^2
      + sum_angles    1/2 k (theta - theta0)^2      [k (1 + cos theta) if linear]
                      (k/n^2) (1 - cos(n theta))    [periodic: metal centres whose
                                                     ligands sit at 90 and 180 deg]
      + sum_torsions  1/2 V (1 + cos(n phi - gamma))   (propers and impropers)
      + sum_pairs     eps [(rmin/r)^12 - 2 (rmin/r)^6] + 332.0637 qi qj / r
      + E_pi          variational pi-electron energy (see below)

The pi term is what sets PMEFF26 apart from a conventional force field. The
conjugated atoms carry a Hueckel-type Hamiltonian whose resonance integrals
depend on the geometry,

    H_ii = alpha_i
    H_ij = -beta_ij [1 - tanh(delta_ij (r_ij - rref_ij))] t_ij
    t_ij = sqrt(<cos^2 phi>_ij + eps^2) / sqrt(1 + eps^2)

(<cos^2 phi> averaged over the dihedrals around bond i-j), and

    E_pi = 2 sum_occ e_k - sum_i n_i alpha_i .

Its gradient is exact by the Hellmann-Feynman theorem, dE = tr(P dH), so
The radial factor is bounded (0..2 beta), so a compressed bond can never
gain more pi energy than the harmonic sigma term costs. Bond-length
alternation, aromatic stabilization and the rotation barriers of
conjugated bonds all come from one energy expression instead of a separate
bond type per bond order.

This module is NumPy-only and knows nothing about chemistry: a
:class:`System` is a set of index/parameter arrays. Every parameter comes
from the external parameter file through :mod:`pmeff26_plugin.assign`; there
are no built-in defaults and no fallback rules here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.polynomial import chebyshev as _cheb

COULOMB = 332.0637  # kcal*Angstrom/(mol*e^2)
LINEAR_THRESHOLD = np.radians(175.0)
# sqrt(kcal/mol/Angstrom^2/amu) -> cm^-1
WAVENUMBER_FACTOR = 108.5913

COMPONENTS = ("bond", "angle", "torsion", "improper", "vdw", "elec", "pi")
PI_TWIST_EPS = 0.05


def _empty(shape, dtype=float):
    return np.zeros(shape, dtype=dtype)


@dataclass
class System:
    """Index and parameter arrays of one molecule (all terms fully assigned)."""

    masses: np.ndarray
    bond_idx: np.ndarray = field(default_factory=lambda: _empty((0, 2), int))
    bond_r0: np.ndarray = field(default_factory=lambda: _empty(0))
    bond_k: np.ndarray = field(default_factory=lambda: _empty(0))
    angle_idx: np.ndarray = field(default_factory=lambda: _empty((0, 3), int))
    angle_t0: np.ndarray = field(default_factory=lambda: _empty(0))
    angle_k: np.ndarray = field(default_factory=lambda: _empty(0))
    # 0 = harmonic; n > 0 = periodic form (k/n^2)(1 - cos(n theta))
    angle_n: np.ndarray = field(default_factory=lambda: _empty(0, int))
    tors_idx: np.ndarray = field(default_factory=lambda: _empty((0, 4), int))
    tors_v: np.ndarray = field(default_factory=lambda: _empty(0))
    tors_n: np.ndarray = field(default_factory=lambda: _empty(0))
    tors_gamma: np.ndarray = field(default_factory=lambda: _empty(0))
    # True for improper (out-of-plane) terms; only used to split the report.
    tors_improper: np.ndarray = field(default_factory=lambda: _empty(0, bool))
    pair_idx: np.ndarray = field(default_factory=lambda: _empty((0, 2), int))
    pair_rmin: np.ndarray = field(default_factory=lambda: _empty(0))
    pair_eps: np.ndarray = field(default_factory=lambda: _empty(0))
    pair_qq: np.ndarray = field(default_factory=lambda: _empty(0))  # incl. COULOMB
    # pi-electron system (empty = term inactive)
    pi_atoms: np.ndarray = field(default_factory=lambda: _empty(0, int))
    pi_alpha: np.ndarray = field(default_factory=lambda: _empty(0))
    pi_occ: np.ndarray = field(default_factory=lambda: _empty(0))  # electrons/atom
    pi_bond: np.ndarray = field(default_factory=lambda: _empty((0, 2), int))  # local
    pi_beta: np.ndarray = field(default_factory=lambda: _empty(0))
    pi_delta: np.ndarray = field(default_factory=lambda: _empty(0))
    pi_rref: np.ndarray = field(default_factory=lambda: _empty(0))
    pi_path_idx: np.ndarray = field(default_factory=lambda: _empty((0, 4), int))
    pi_path_bond: np.ndarray = field(default_factory=lambda: _empty(0, int))

    @property
    def num_atoms(self) -> int:
        return len(self.masses)


def _dihedral(x: np.ndarray, idx: np.ndarray):
    """Return (phi, dphi/dri, dphi/drj, dphi/drk, dphi/drl) for each quadruple."""
    ri, rj, rk, rl = (x[idx[:, c]] for c in range(4))
    f = ri - rj
    g = rj - rk
    h = rl - rk
    a = np.cross(f, g)
    b = np.cross(h, g)
    a2 = np.einsum("ij,ij->i", a, a)
    b2 = np.einsum("ij,ij->i", b, b)
    gn = np.linalg.norm(g, axis=1)
    ok = (a2 > 1e-12) & (b2 > 1e-12) & (gn > 1e-8)
    a2s = np.where(ok, a2, 1.0)
    b2s = np.where(ok, b2, 1.0)
    gns = np.where(ok, gn, 1.0)
    phi = np.arctan2(
        np.einsum("ij,ij->i", np.cross(b, a), g) / gns, np.einsum("ij,ij->i", a, b)
    )
    fg = np.einsum("ij,ij->i", f, g)
    hg = np.einsum("ij,ij->i", h, g)
    di = -(gns / a2s)[:, None] * a
    dl = (gns / b2s)[:, None] * b
    ca = (fg / (a2s * gns))[:, None] * a
    cb = (hg / (b2s * gns))[:, None] * b
    dj = -di + ca - cb
    dk = -dl - ca + cb
    mask = ok[:, None]
    return (
        np.where(ok, phi, 0.0),
        np.where(mask, di, 0.0),
        np.where(mask, dj, 0.0),
        np.where(mask, dk, 0.0),
        np.where(mask, dl, 0.0),
        ok,
    )


def pi_hamiltonian(x: np.ndarray, system: System):
    """Return the pi Hamiltonian and the intermediates its gradient needs."""
    a = system.pi_atoms[system.pi_bond[:, 0]]
    b = system.pi_atoms[system.pi_bond[:, 1]]
    d = x[a] - x[b]
    r = np.linalg.norm(d, axis=1)
    th = np.tanh(system.pi_delta * (r - system.pi_rref))
    radial = system.pi_beta * (1.0 - th)
    dradial = -system.pi_beta * system.pi_delta * (1.0 - th * th)
    nb = len(system.pi_bond)
    c2 = np.ones(nb)
    dih = None
    if len(system.pi_path_idx):
        dih = _dihedral(x, system.pi_path_idx)
        count = np.bincount(system.pi_path_bond, minlength=nb).astype(float)
        total = np.bincount(
            system.pi_path_bond, weights=np.cos(dih[0]) ** 2, minlength=nb
        )
        has = count > 0
        c2 = np.where(has, total / np.where(has, count, 1.0), 1.0)
    norm = np.sqrt(1.0 + PI_TWIST_EPS**2)
    twist = np.sqrt(c2 + PI_TWIST_EPS**2) / norm
    ham = np.diag(system.pi_alpha.astype(float))
    off = -radial * twist
    ham[system.pi_bond[:, 0], system.pi_bond[:, 1]] = off
    ham[system.pi_bond[:, 1], system.pi_bond[:, 0]] = off
    return ham, radial, twist, c2, dih, (a, b, d, r, norm, dradial)


def pi_bond_orders(coords: np.ndarray, system: System) -> np.ndarray:
    """Mobile pi bond order of every pi bond at *coords* (diagnostic)."""
    if not len(system.pi_atoms):
        return _empty(0)
    ham = pi_hamiltonian(np.asarray(coords, dtype=float), system)[0]
    _, vec = np.linalg.eigh(ham)
    occ = vec[:, : int(round(system.pi_occ.sum())) // 2]
    dens = 2.0 * occ @ occ.T
    return dens[system.pi_bond[:, 0], system.pi_bond[:, 1]]


def _pi_term(x: np.ndarray, system: System, grad: np.ndarray) -> float:
    ham, radial, twist, c2, dih, (a, b, d, r, norm, dradial) = pi_hamiltonian(x, system)
    val, vec = np.linalg.eigh(ham)
    n_occ = int(round(system.pi_occ.sum())) // 2
    occ = vec[:, :n_occ]
    energy = 2.0 * float(val[:n_occ].sum()) - float(system.pi_occ @ system.pi_alpha)
    dens = 2.0 * occ @ occ.T
    # dE/dH_ab for one symmetric off-diagonal element
    w = 2.0 * dens[system.pi_bond[:, 0], system.pi_bond[:, 1]]
    de_dr = -w * dradial * twist
    g = (de_dr / np.maximum(r, 1e-10))[:, None] * d
    np.add.at(grad, a, g)
    np.add.at(grad, b, -g)
    if dih is not None:
        phi, di, dj, dk, dl, ok = dih
        nb = len(system.pi_bond)
        count = np.bincount(system.pi_path_bond, minlength=nb).astype(float)
        de_dc2 = w * (-radial) / (2.0 * np.sqrt(c2 + PI_TWIST_EPS**2) * norm)
        coef = (de_dc2 / np.maximum(count, 1.0))[system.pi_path_bond] * (
            -np.sin(2.0 * phi)
        )
        coef = np.where(ok, coef, 0.0)[:, None]
        for col, dphi in enumerate((di, dj, dk, dl)):
            np.add.at(grad, system.pi_path_idx[:, col], coef * dphi)
    return energy


def energy_and_gradient(
    coords: np.ndarray, system: System, components: Optional[Dict[str, float]] = None
) -> Tuple[float, np.ndarray]:
    """Return the total energy (kcal/mol) and its analytic gradient (N, 3)."""
    x = np.asarray(coords, dtype=float)
    grad = np.zeros_like(x)
    parts = dict.fromkeys(COMPONENTS, 0.0)

    if len(system.bond_idx):
        i, j = system.bond_idx[:, 0], system.bond_idx[:, 1]
        d = x[i] - x[j]
        r = np.linalg.norm(d, axis=1)
        dr = r - system.bond_r0
        parts["bond"] = float(0.5 * np.sum(system.bond_k * dr * dr))
        g = (system.bond_k * dr / np.maximum(r, 1e-10))[:, None] * d
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)

    if len(system.angle_idx):
        i, j, k = (system.angle_idx[:, c] for c in range(3))
        u = x[i] - x[j]
        v = x[k] - x[j]
        ru = np.maximum(np.linalg.norm(u, axis=1), 1e-10)
        rv = np.maximum(np.linalg.norm(v, axis=1), 1e-10)
        cos_t = np.clip(np.einsum("ij,ij->i", u, v) / (ru * rv), -1.0, 1.0)
        dcos_i = (v / (ru * rv)[:, None]) - (cos_t / ru**2)[:, None] * u
        dcos_k = (u / (ru * rv)[:, None]) - (cos_t / rv**2)[:, None] * v
        linear = system.angle_t0 >= LINEAR_THRESHOLD
        theta = np.arccos(cos_t)
        dt = theta - system.angle_t0
        sin_t = np.maximum(np.sqrt(1.0 - cos_t * cos_t), 1e-6)
        e_harm = 0.5 * system.angle_k * dt * dt
        de_dcos_harm = -system.angle_k * dt / sin_t
        e_lin = system.angle_k * (1.0 + cos_t)
        e_ang = np.where(linear, e_lin, e_harm)
        de_dcos = np.where(linear, system.angle_k, de_dcos_harm)
        if len(system.angle_n) == len(e_ang):
            for n in np.unique(system.angle_n[system.angle_n > 0]):
                sel = system.angle_n == n
                coef = np.zeros(int(n) + 1)
                coef[-1] = 1.0  # Chebyshev T_n(cos theta) = cos(n theta)
                scale = system.angle_k[sel] / float(n * n)
                e_ang[sel] = scale * (1.0 - _cheb.chebval(cos_t[sel], coef))
                de_dcos[sel] = -scale * _cheb.chebval(cos_t[sel], _cheb.chebder(coef))
        parts["angle"] = float(np.sum(e_ang))
        de_dcos = de_dcos[:, None]
        gi = de_dcos * dcos_i
        gk = de_dcos * dcos_k
        np.add.at(grad, i, gi)
        np.add.at(grad, k, gk)
        np.add.at(grad, j, -(gi + gk))

    if len(system.tors_idx):
        phi, di, dj, dk, dl, ok = _dihedral(x, system.tors_idx)
        arg = system.tors_n * phi - system.tors_gamma
        e = np.where(ok, 0.5 * system.tors_v * (1.0 + np.cos(arg)), 0.0)
        imp = system.tors_improper
        parts["improper"] = float(np.sum(e[imp]))
        parts["torsion"] = float(np.sum(e[~imp]))
        de = (-0.5 * system.tors_v * system.tors_n * np.sin(arg))[:, None]
        for col, dphi in enumerate((di, dj, dk, dl)):
            np.add.at(grad, system.tors_idx[:, col], de * dphi)

    if len(system.pair_idx):
        i, j = system.pair_idx[:, 0], system.pair_idx[:, 1]
        d = x[i] - x[j]
        r2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-8)
        r = np.sqrt(r2)
        s6 = (system.pair_rmin * system.pair_rmin / r2) ** 3
        e_vdw = system.pair_eps * (s6 * s6 - 2.0 * s6)
        e_el = system.pair_qq / r
        parts["vdw"] = float(np.sum(e_vdw))
        parts["elec"] = float(np.sum(e_el))
        de_dr = -12.0 * system.pair_eps * (s6 * s6 - s6) / r - e_el / r
        g = (de_dr / r)[:, None] * d
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)

    if len(system.pi_atoms):
        parts["pi"] = _pi_term(x, system, grad)

    if components is not None:
        components.update(parts)
    return float(sum(parts.values())), grad


def energy_components(coords: np.ndarray, system: System) -> Dict[str, float]:
    parts: Dict[str, float] = {}
    total, _ = energy_and_gradient(coords, system, parts)
    parts["total"] = total
    return parts


# --- optimizer --------------------------------------------------------------


@dataclass
class OptimizeResult:
    converged: bool
    energy: float
    steps: int
    max_force: float


@dataclass
class OptimizationProgress:
    step: int
    max_iter: int
    energy: float
    max_force: float
    phase: str


class OptimizationCancelled(Exception):
    """Raised when the caller's cancel callback asks to stop."""


def _max_force(grad: np.ndarray) -> float:
    return float(np.sqrt(np.max(np.einsum("ij,ij->i", grad, grad)))) if len(grad) else 0.0


def _clamp(step: np.ndarray, max_step: float) -> np.ndarray:
    longest = float(np.sqrt(np.max(np.einsum("ij,ij->i", step, step))))
    return step * (max_step / longest) if longest > max_step else step


def optimize(
    coords: np.ndarray,
    system: System,
    max_iter: int = 2000,
    f_tol: float = 1e-3,
    max_step: float = 0.2,
    progress_callback: Optional[Callable[[OptimizationProgress], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, OptimizeResult]:
    """Minimize the energy: FIRE far from the minimum, then L-BFGS.

    *f_tol* is the largest per-atom force (kcal/mol/Angstrom) at convergence.
    Returns ``(coords, OptimizeResult)``; the input array is not modified.
    """
    x = np.array(coords, dtype=float)
    e, g = energy_and_gradient(x, system)
    step = 0

    def tick(phase: str) -> None:
        if cancel_callback is not None and cancel_callback():
            raise OptimizationCancelled("PMEFF26 optimization cancelled")
        if progress_callback is not None:
            progress_callback(OptimizationProgress(step, max_iter, e, _max_force(g), phase))

    # Phase 1: FIRE while forces are large (clashes, bad starting geometry).
    crossover = 20.0
    v = np.zeros_like(x)
    dt, alpha, n_pos = 0.02, 0.1, 0
    while step < max_iter and _max_force(g) > max(crossover, f_tol):
        tick("FIRE")
        f = -g
        if float(np.sum(f * v)) > 0.0:
            n_pos += 1
            if n_pos > 5:
                dt, alpha = min(dt * 1.1, 0.1), alpha * 0.99
        else:
            v[:] = 0.0
            dt, alpha, n_pos = max(dt * 0.5, 1e-4), 0.1, 0
        v += dt * f
        fn = np.linalg.norm(f)
        if fn > 0:
            v = (1.0 - alpha) * v + alpha * np.linalg.norm(v) * f / fn
        x = x + _clamp(dt * v, max_step)
        e, g = energy_and_gradient(x, system)
        step += 1

    # Phase 2: L-BFGS with Armijo backtracking.
    hist_s: list = []
    hist_y: list = []
    while step < max_iter and _max_force(g) > f_tol:
        tick("L-BFGS")
        q = g.ravel().copy()
        alphas = []
        for s, y in reversed(list(zip(hist_s, hist_y))):
            rho = 1.0 / float(y @ s)
            a = rho * float(s @ q)
            alphas.append((a, rho, s, y))
            q -= a * y
        if hist_s:
            q *= float(hist_s[-1] @ hist_y[-1]) / float(hist_y[-1] @ hist_y[-1])
        else:
            q *= 1.0 / max(float(np.abs(g).max()), 1.0) * 0.1
        for a, rho, s, y in reversed(alphas):
            q += s * (a - rho * float(y @ q))
        direction = -q.reshape(x.shape)
        slope = float(np.sum(direction * g))
        if slope >= 0.0:  # not a descent direction: restart from steepest descent
            hist_s, hist_y = [], []
            direction = -g * (0.1 / max(float(np.abs(g).max()), 1.0))
            slope = float(np.sum(direction * g))
        direction = _clamp(direction, max_step)
        slope = float(np.sum(direction * g))
        t = 1.0
        for _ in range(25):
            x_new = x + t * direction
            e_new, g_new = energy_and_gradient(x_new, system)
            if e_new <= e + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            if not hist_s:
                break  # line search failed even along steepest descent
            hist_s, hist_y = [], []
            step += 1
            continue
        s_vec = (x_new - x).ravel()
        y_vec = (g_new - g).ravel()
        if float(s_vec @ y_vec) > 1e-12:
            hist_s.append(s_vec)
            hist_y.append(y_vec)
            if len(hist_s) > 12:
                hist_s.pop(0)
                hist_y.pop(0)
        x, e, g = x_new, e_new, g_new
        step += 1

    fmax = _max_force(g)
    return x, OptimizeResult(fmax <= f_tol, float(e), step, fmax)


# --- vibrations -------------------------------------------------------------


def hessian(
    coords: np.ndarray,
    system: System,
    delta: float = 1e-4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Cartesian Hessian (kcal/mol/Angstrom^2) by central differences of the gradient."""
    x = np.array(coords, dtype=float)
    n3 = x.size
    hess = np.zeros((n3, n3))
    flat = x.ravel()
    for c in range(n3):
        if cancel_callback is not None and cancel_callback():
            raise OptimizationCancelled("PMEFF26 vibrational analysis cancelled")
        keep = flat[c]
        flat[c] = keep + delta
        _, gp = energy_and_gradient(x, system)
        flat[c] = keep - delta
        _, gm = energy_and_gradient(x, system)
        flat[c] = keep
        hess[c] = (gp - gm).ravel() / (2.0 * delta)
        if progress_callback is not None:
            progress_callback(c + 1, n3)
    return 0.5 * (hess + hess.T)


def frequencies_from_hessian(hess: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Harmonic wavenumbers (cm^-1, negative = imaginary), ascending, all 3N."""
    w = np.repeat(1.0 / np.sqrt(np.asarray(masses, dtype=float)), 3)
    eig = np.linalg.eigvalsh(hess * np.outer(w, w))
    return np.sign(eig) * np.sqrt(np.abs(eig)) * WAVENUMBER_FACTOR


def vibrational_analysis(coords: np.ndarray, system: System, **kwargs) -> dict:
    """Frequencies in cm^-1 and a minimum/saddle verdict at *coords*."""
    freqs = frequencies_from_hessian(hessian(coords, system, **kwargs), system.masses)
    _, grad = energy_and_gradient(coords, system)
    x = np.asarray(coords, dtype=float)
    centered = x - x.mean(axis=0)
    rank = np.linalg.matrix_rank(centered, tol=1e-6) if len(x) > 1 else 0
    n_rigid = 3 if len(x) == 1 else (5 if rank <= 1 else 6)
    n_rigid = min(n_rigid, len(freqs))
    order = np.argsort(np.abs(freqs))
    vib = np.sort(np.delete(freqs, order[:n_rigid]))
    threshold = 20.0  # cm^-1: below this an imaginary value is numerical noise
    n_imag = int(np.sum(vib < -threshold))
    return {
        "frequencies": vib,
        "num_imaginary": n_imag,
        "is_minimum": n_imag == 0,
        "max_force": _max_force(grad),
        "imaginary_threshold": threshold,
    }
