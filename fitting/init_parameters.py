#!/usr/bin/env python3
"""Write the unfitted PMEFF26 parameter skeleton.

Creates ``pmeff26_plugin/parameters/pmeff26.json`` with the atom-type
definitions, the 1-4 scale factors and the van der Waals table. The bonded
tables and charge increments start empty and are filled by ``fitting.fit``
from the reference DFT data; until then the engine refuses every molecule.

The Lennard-Jones parameters are the per-element UFF values (Rappe et al.,
J. Am. Chem. Soc. 114, 10024 (1992)), hard-coded below from RDKit's UFF table.
They are *not* refit: the reference set has no intermolecular data.

    python -m fitting.init_parameters [--force]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "pmeff26_plugin" / "parameters" / "pmeff26.json"

# Ordered: a later entry overrides an earlier one. "linear" types carry no
# torsions around their bonds; "planar" 3-coordinate types need an improper.
ATOM_TYPES = [
    ("h", "[#1]", "H", {}),
    # Polar hydrogen: its own van der Waals class (hydrogen bonds).
    ("hx", "[#1][#7,#8,#9]", "H", {}),
    ("b2", "[#5X3]", "B", {"planar": True, "pi_electrons": 0, "pi_alpha": 0.0}),
    ("b4", "[#5X4]", "B", {}),
    ("c3", "[#6X4]", "C", {}),
    ("c3r4", "[#6X4;r4]", "C", {}),
    ("c3r3", "[#6X4;r3]", "C", {}),
    ("c2", "[#6X3]", "C", {"planar": True, "pi_electrons": 1, "pi_alpha": 0.0}),
    ("c2r4", "[#6X3;r4]", "C", {"planar": True, "pi_electrons": 1, "pi_alpha": 0.0}),
    ("c2r3", "[#6X3;r3]", "C", {"planar": True, "pi_electrons": 1, "pi_alpha": 0.0}),
    ("ca", "[c]", "C", {"planar": True, "pi_electrons": 1, "pi_alpha": 0.0}),
    ("c1", "[#6X2]", "C", {"linear": True}),
    ("c0", "[#6X1]", "C", {"linear": True}),
    ("n3", "[#7X3]", "N", {}),
    ("n3r3", "[#7X3;r3]", "N", {}),
    ("nh", "[#7X3;!r3]-[#6X3,c,#6X2,#7X2,#5X3]", "N", {"pi_electrons": 2, "pi_alpha": 0.0}),
    ("nam", "[#7X3;!r3]-[#6X3,#16X4,#15X4]=[#8,#16,#7]", "N", {"planar": True, "pi_electrons": 2, "pi_alpha": 0.0}),
    ("nb", "[nX3]", "N", {"planar": True, "pi_electrons": 2, "pi_alpha": 0.0}),
    ("no", "[#7X3+]", "N", {"planar": True, "pi_electrons": 2, "pi_alpha": 0.0}),
    ("n4", "[#7X4]", "N", {}),
    ("n2", "[#7X2]", "N", {"pi_electrons": 1, "pi_alpha": 0.0}),
    ("na", "[nX2]", "N", {"pi_electrons": 1, "pi_alpha": 0.0}),
    ("n1", "[#7X1]", "N", {"linear": True}),
    ("n1c", "[#7X2+,#7X2-]", "N", {"linear": True}),
    ("n1c", "[#7X2](=*)=*", "N", {"linear": True}),
    ("ow", "[#8X2H2]", "O", {}),
    ("oh", "[#8X2H1]", "O", {}),
    ("os", "[#8X2H0]", "O", {}),
    ("osr3", "[#8X2;r3]", "O", {}),
    ("oa", "[o]", "O", {"pi_electrons": 2, "pi_alpha": 0.0}),
    ("o", "[#8X1]", "O", {"pi_electrons": 1, "pi_alpha": 0.0}),
    ("f", "[#9]", "F", {}),
    ("si", "[#14X4]", "Si", {}),
    ("p3", "[#15X3]", "P", {}),
    ("p4", "[#15X4]", "P", {}),
    ("p5", "[#15X5]", "P", {}),
    ("p2", "[#15X2]", "P", {}),
    ("p1", "[#15X1]", "P", {"linear": True}),
    ("s2", "[#16X2]", "S", {}),
    ("s2r3", "[#16X2;r3]", "S", {}),
    ("sa", "[s]", "S", {"pi_electrons": 2, "pi_alpha": 0.0}),
    ("s1", "[#16X1]", "S", {"pi_electrons": 1, "pi_alpha": 0.0}),
    ("s3", "[#16X3]", "S", {}),
    ("s4", "[#16X4]", "S", {}),
    ("s6", "[#16X6]", "S", {}),
    ("cl", "[#17]", "Cl", {}),
    ("br", "[#35]", "Br", {}),
    ("i", "[#53]", "I", {}),
    # Metals: typed by element and coordination number. Neutral closed-shell
    # species only; no pi-complexes and no 5-coordinate centres.
    ("li1", "[#3X1]", "Li", {}),
    ("na1", "[#11X1]", "Na", {}),
    ("k1", "[#19X1]", "K", {}),
    ("mg2", "[#12X2]", "Mg", {"linear": True}),
    ("ca2", "[#20X2]", "Ca", {}),
    ("al3", "[#13X3]", "Al", {"planar": True}),
    ("al4", "[#13X4]", "Al", {}),
    ("ge4", "[#32X4]", "Ge", {}),
    ("sn4", "[#50X4]", "Sn", {}),
    ("ti4", "[#22X4]", "Ti", {}),
    ("cr6", "[#24X6]", "Cr", {}),
    ("ni4", "[#28X4]", "Ni", {}),
    ("cu1", "[#29X1]", "Cu", {}),
    ("cu2", "[#29X2]", "Cu", {"linear": True}),
    ("zn2", "[#30X2]", "Zn", {"linear": True}),
    ("zn4", "[#30X4]", "Zn", {}),
    ("hg2", "[#80X2]", "Hg", {"linear": True}),
    ("pd4", "[#46X4]", "Pd", {}),
    ("pt4", "[#78X4]", "Pt", {}),
]

# UFF van der Waals parameters: x1 (Angstrom, the like-pair minimum distance)
# and D1 (kcal/mol, the well depth). Values are those of RDKit's UFF table,
# Code/ForceField/UFF/Params.cpp (RDKit Release_2026_03; columns x1 and D1 of
# the atom label given in the comment), which reproduces Table I of
# Rappe et al., J. Am. Chem. Soc. 114, 10024 (1992). They are per element in
# UFF, so the label only records which row was read. Not refit.
UFF_VDW = {
    "H": (2.886, 0.044),  # H_
    "B": (4.083, 0.180),  # B_2
    "C": (3.851, 0.105),  # C_3
    "N": (3.660, 0.069),  # N_3
    "O": (3.500, 0.060),  # O_3
    "F": (3.364, 0.050),  # F_
    "Si": (4.295, 0.402),  # Si3
    "P": (4.147, 0.305),  # P_3+3
    "S": (4.035, 0.274),  # S_3+2
    "Cl": (3.947, 0.227),  # Cl
    "Br": (4.189, 0.251),  # Br
    "I": (4.500, 0.339),  # I_
    "Li": (2.451, 0.025),  # Li
    "Na": (2.983, 0.030),  # Na
    "K": (3.812, 0.035),  # K_
    "Mg": (3.021, 0.111),  # Mg3+2
    "Ca": (3.399, 0.238),  # Ca6+2
    "Al": (4.499, 0.505),  # Al3
    "Ge": (4.280, 0.379),  # Ge3
    "Sn": (4.392, 0.567),  # Sn3
    "Ti": (3.175, 0.017),  # Ti3+4
    "Cr": (3.023, 0.015),  # Cr6+3
    "Ni": (2.834, 0.015),  # Ni4+2
    "Cu": (3.495, 0.005),  # Cu3+1
    "Zn": (2.763, 0.124),  # Zn3+2
    "Pd": (2.899, 0.048),  # Pd4+2
    "Pt": (2.754, 0.080),  # Pt4+2
    "Hg": (2.705, 0.385),  # Hg1+2
}


def uff_lj(symbol: str) -> list[float]:
    """Return [rmin/2, epsilon] of the UFF like-pair potential for *symbol*."""
    x1, d1 = UFF_VDW[symbol]
    return [round(x1 / 2.0, 4), d1]


def skeleton() -> dict:
    """The unfitted parameter file: types, scale factors and the vdW table."""
    lj = {el: uff_lj(el) for el in UFF_VDW}
    return {
        "name": "PMEFF26",
        "version": "0.0.0-unfitted",
        "units": "kcal/mol, Angstrom, degree, e; bond/angle k in E = 1/2 k x^2",
        "reference": "not fitted yet",
        "vdw_source": "UFF per-element Lennard-Jones, values from RDKit "
        "Code/ForceField/UFF/Params.cpp (Rappe et al. 1992 Table I); not refit",
        "scale14": {"vdw": 0.5, "elec": 0.8333},
        "atom_types": [
            dict({"type": name, "smarts": smarts, "element": el}, **flags)
            for name, smarts, el, flags in ATOM_TYPES
        ],
        "vdw": {name: lj[el] for name, _s, el, _f in ATOM_TYPES},
        "pi": {"enabled": False, "bonds": {}},
        "increments": {"types": {}, "bonds": {}},
        "bci": {},
        "bonds": {},
        "angles": {},
        "torsions": {},
        "impropers": {},
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if OUT.exists() and "--force" not in argv:
        print(f"{OUT} exists; pass --force to overwrite (this discards fitted values)")
        return 1
    data = skeleton()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + chr(10), encoding="utf-8", newline=chr(10))
    print(f"wrote {OUT}: {len(data['atom_types'])} atom-type rules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
