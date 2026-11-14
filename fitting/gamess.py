"""GAMESS (US) input writer for the PMEFF26 reference calculations.

Level: wB97X-D / def2-TZVP (def2 ECP on iodine), spherical harmonics, RHF,
neutral singlets. The basis is written inline from ``def2_tzvp_gamess.json``
(Basis Set Exchange data) because GAMESS's built-in Karlsruhe sets stop at Ar.
"""

from __future__ import annotations

import json
from pathlib import Path

_BASIS = json.loads(
    (Path(__file__).with_name("def2_tzvp_gamess.json")).read_text(encoding="utf-8")
)["elements"]

LEVEL = "wB97X-D/def2-TZVP (GAMESS)"


def gamess_input(
    title: str,
    symbols: list[str],
    atomic_numbers: list[int],
    coords: list[tuple[float, float, float]],
    hessian: bool,
    frozen_dihedral: tuple[int, int, int, int] | None = None,
    frozen_value: float = 0.0,
    functional: str = "wB97X-D",
    d3bj: bool = False,
    single_point: bool = False,
    ghosts: frozenset = frozenset(),
    dispersion_off: bool = False,
) -> str:
    """Return a GAMESS optimization input (optionally + Hessian, or constrained)."""
    if dispersion_off and functional.upper() == "WB97X-D":
        # GAMESS cannot switch the -D term of wB97X-D off (and its dispersion
        # code crashes on ghost atoms): use the dispersion-free wB97X instead.
        functional = "wB97X"
    use_ecp = any(_BASIS[s]["ecp"] and n not in ghosts for n, s in enumerate(symbols))
    n = len(symbols)
    contrl = [
        f" $CONTRL SCFTYP=RHF RUNTYP={'ENERGY' if single_point else 'OPTIMIZE'} DFTTYP={functional} ISPHER=1",
        "  MAXIT=200 ICHARG=0 MULT=1",
    ]
    # Delocalized internal coordinates for every optimization of 4+ atoms: the
    # Cartesian optimizer's Badger guess Hessian is singular for many ordinary
    # molecules and GAMESS aborts at step 0 ("NOT 5 OR 6 TRANS/ROT MODES").
    use_dlc = not single_point and (frozen_dihedral is not None or n >= 4)
    if use_dlc:
        contrl.append(f"  NZVAR={3 * n - 6}")
    if use_ecp:
        contrl.append("  PP=READ")
    contrl[-1] += " $END"
    lines = contrl + [
        " $SYSTEM MWORDS=300 $END",
        " $SCF DIRSCF=.TRUE. NCONV=7 $END",
        " $DFT NRAD=99 NLEB=590"
        + ("" if dispersion_off else " DC=.TRUE. IDCVER=4" if d3bj else "")
        + " $END",
    ]
    if single_point:
        pass
    elif frozen_dihedral is not None:
        i, j, k, l = (x + 1 for x in frozen_dihedral)
        lines += [
            " $ZMAT DLC=.TRUE. AUTO=.TRUE.",
            f"  IFZMAT(1)=3,{i},{j},{k},{l} FVALUE(1)={frozen_value:.4f} $END",
            " $STATPT OPTTOL=1.0E-4 NSTEP=300 $END",
        ]
    else:
        if use_dlc:
            lines.append(" $ZMAT DLC=.TRUE. AUTO=.TRUE. $END")
        if hessian and n > 1:
            lines += [
                " $STATPT OPTTOL=1.0E-5 NSTEP=300 HSSEND=.TRUE. $END",
                " $FORCE METHOD=SEMINUM NVIB=2 $END",
            ]
        else:
            lines.append(" $STATPT OPTTOL=1.0E-5 NSTEP=300 $END")
    lines += [" $DATA", title, "C1"]
    for n, (sym, z, (x, y, zc)) in enumerate(zip(symbols, atomic_numbers, coords)):
        # A ghost atom (counterpoise correction) keeps its basis functions but
        # has no nuclear charge and no electrons.
        charge = 0.0 if n in ghosts else float(z)
        lines.append(f"{sym:<2s} {charge:5.1f} {x:16.10f} {y:16.10f} {zc:16.10f}")
        lines.extend(_BASIS[sym]["basis"])
        lines.append("")
    lines.append(" $END")
    if use_ecp:
        lines.append(" $ECP")
        seen: set[str] = set()
        for n, sym in enumerate(symbols):
            ecp = _BASIS[sym]["ecp"]
            if not ecp or n in ghosts:
                lines.append(f"{sym}-ECP NONE")
            elif sym in seen:
                lines.append(f"{sym}-ECP")
            else:
                seen.add(sym)
                lines.extend(ecp)
        lines.append(" $END")
    return "\n".join(lines) + "\n"
