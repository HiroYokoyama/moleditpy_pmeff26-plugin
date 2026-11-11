"""PMEFF26 parameter set: loaded from an external JSON file, never built in.

The file holds everything chemical: the atom types (ordered SMARTS, the last
match wins), and one table per energy term. A lookup that finds nothing is an
error - PMEFF26 has no generic rules to fall back on, because an energy
computed with a guessed parameter is not accurate.

Key formats (types are the names from ``atom_types``):

* bond      ``a-b``, ``a=b``, ``a#b``, ``a:b`` (aromatic); types sorted
* angle     ``a|center|c`` with the two ends sorted; ``@3`` / ``@4`` appended
            when the three atoms share a 3- or 4-membered ring
            value ``[theta0, k]``, or ``["periodic", n, k]`` for the form
            ``(k/n^2)(1 - cos(n theta))`` used at square-planar/octahedral metals
* torsion   ``a|b-c|d`` (central bond written like a bond key); the generic
            form ``*|b-c|*`` gives a per-bond barrier shared by all its paths
* improper  center type -> barrier of the 2-fold out-of-plane term
* vdw       type -> ``[rmin/2, epsilon]``
* bci       bond key -> charge moved from the second type to the first
* pi        ``{"enabled": true, "bonds": {"a~b": [beta, delta, rref]}}``; atom
            types taking part carry ``pi_electrons`` and ``pi_alpha``. When
            enabled, a bond between two pi types is written ``a~b`` whatever
            its formal order - the pi term supplies the order dependence
* increments ``{"types": {...}, "bonds": {...}}`` in Hartree: fitted
            contributions that reference the energy to the DFT total energy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_FILE = Path(__file__).resolve().parent / "parameters" / "pmeff26.json"
BOND_SYMBOLS = {1.0: "-", 2.0: "=", 3.0: "#", 1.5: ":"}


class ParameterError(Exception):
    """Base class for parameter-set problems."""


class UnsupportedAtomError(ParameterError):
    """An atom matched no atom type of the parameter set."""


class MissingParameterError(ParameterError):
    """The parameter set has no entry for terms this molecule needs."""

    def __init__(self, missing: Sequence[str]):
        self.missing = sorted(set(missing))
        shown = ", ".join(self.missing[:8])
        more = f" (+{len(self.missing) - 8} more)" if len(self.missing) > 8 else ""
        super().__init__(
            f"PMEFF26 has no parameters for: {shown}{more}. "
            "This molecule is outside the fitted parameter set; no approximate "
            "fallback is used."
        )


def bond_key(a: str, b: str, symbol: str) -> str:
    a, b = sorted((a, b))
    return f"{a}{symbol}{b}"


def angle_key(a: str, center: str, c: str, ring: int = 0) -> str:
    a, c = sorted((a, c))
    return f"{a}|{center}|{c}" + (f"@{ring}" if ring else "")


def torsion_key(a: str, b: str, c: str, d: str, symbol: str) -> str:
    if (b, a) > (c, d):
        a, b, c, d = d, c, b, a
    return f"{a}|{b}{symbol}{c}|{d}"


class ParameterSet:
    """An immutable view of one parameter file."""

    def __init__(self, data: dict, source: str = "<memory>"):
        self.data = data
        self.source = source
        self.name = data.get("name", "PMEFF26")
        self.version = data.get("version", "?")
        self.atom_types: List[dict] = list(data.get("atom_types", []))
        self.type_info: Dict[str, dict] = {t["type"]: t for t in self.atom_types}
        self.bonds: Dict[str, list] = data.get("bonds", {})
        self.angles: Dict[str, list] = data.get("angles", {})
        self.torsions: Dict[str, list] = data.get("torsions", {})
        self.impropers: Dict[str, float] = data.get("impropers", {})
        self.vdw: Dict[str, list] = data.get("vdw", {})
        self.bci: Dict[str, float] = data.get("bci", {})
        pi = data.get("pi", {})
        self.pi_enabled = bool(pi.get("enabled"))
        self.pi_bonds: Dict[str, list] = pi.get("bonds", {})
        inc = data.get("increments", {})
        self.inc_types: Dict[str, float] = inc.get("types", {})
        self.inc_bonds: Dict[str, float] = inc.get("bonds", {})
        scale = data.get("scale14", {})
        self.scale14_vdw = float(scale.get("vdw", 0.5))
        self.scale14_elec = float(scale.get("elec", 1.0 / 1.2))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ParameterSet":
        path = Path(path) if path is not None else DEFAULT_FILE
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ParameterError(f"cannot read parameter file {path}: {exc}") from exc
        if not isinstance(data, dict) or "atom_types" not in data:
            raise ParameterError(f"{path} is not a PMEFF26 parameter file")
        return cls(data, str(path))

    def is_linear(self, type_name: str) -> bool:
        return bool(self.type_info.get(type_name, {}).get("linear"))

    def is_planar(self, type_name: str) -> bool:
        return bool(self.type_info.get(type_name, {}).get("planar"))

    def pi_electrons(self, type_name: str) -> Optional[int]:
        """Base pi-electron count of a type, or None if it is not a pi type."""
        if not self.pi_enabled:
            return None
        return self.type_info.get(type_name, {}).get("pi_electrons")

    def torsion(
        self, a: str, b: str, c: str, d: str, symbol: str
    ) -> Optional[Tuple[list, bool]]:
        """Return (terms, generic) for a path, or None when not parameterized."""
        exact = self.torsions.get(torsion_key(a, b, c, d, symbol))
        if exact is not None:
            return exact, False
        generic = self.torsions.get(torsion_key("*", b, c, "*", symbol))
        if generic is not None:
            return generic, True
        return None


_CACHE: Dict[str, Tuple[float, ParameterSet]] = {}


def default_parameters() -> ParameterSet:
    """The shipped parameter file, reloaded when it changes on disk."""
    path = DEFAULT_FILE
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise ParameterError(f"parameter file {path} is missing") from exc
    cached = _CACHE.get(str(path))
    if cached is None or cached[0] != mtime:
        _CACHE[str(path)] = (mtime, ParameterSet.load(path))
    return _CACHE[str(path)][1]
