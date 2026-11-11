#!/usr/bin/env python3
"""Write the unfitted PMEFF26 parameter skeleton.

Creates ``pmeff26_plugin/parameters/pmeff26.json`` with the atom-type
definitions, the 1-4 scale factors and the van der Waals table. The bonded
tables and charge increments start empty and are filled by ``fitting.fit``
from the reference DFT data; until then the engine refuses every molecule.

The Lennard-Jones parameters are the per-element UFF values (Rappe et al.,
J. Am. Chem. Soc. 114, 10024 (1992)), read from RDKit's UFF implementation.
They are *not* refit: the reference set has no intermolecular data.

    python -m fitting.init_parameters [--force]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

OUT = Path(__file__).resolve().parents[1] / "pmeff26_plugin" / "parameters" / "pmeff26.json"

# Ordered: a later entry overrides an earlier one. "linear" types carry no
# torsions around their bonds; "planar" 3-coordinate types need an improper.
ATOM_TYPES = [
    ("h", "[#1]", "H", {}),
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
]

# One simple molecule per element to read RDKit's UFF like-pair LJ parameters.
_PROBE = {
    "H": "[HH]", "B": "BB", "C": "CC", "N": "NN", "O": "OO", "F": "FF",
    "Si": "[SiH3][SiH3]", "P": "PP", "S": "SS", "Cl": "ClCl", "Br": "BrBr", "I": "II",
}


def uff_lj(symbol: str) -> list[float]:
    """Return [rmin/2, epsilon] of the UFF like-pair potential for *symbol*."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_PROBE[symbol]))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol][:2]
    x_ij, d_ij = rdForceFieldHelpers.GetUFFVdWParams(mol, idx[0], idx[1])
    return [round(x_ij / 2.0, 4), round(d_ij, 4)]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if OUT.exists() and "--force" not in argv:
        print(f"{OUT} exists; pass --force to overwrite (this discards fitted values)")
        return 1
    lj = {el: uff_lj(el) for el in _PROBE}
    data = {
        "name": "PMEFF26",
        "version": "0.0.0-unfitted",
        "units": "kcal/mol, Angstrom, degree, e; bond/angle k in E = 1/2 k x^2",
        "reference": "not fitted yet",
        "vdw_source": "UFF per-element Lennard-Jones (Rappe et al. 1992), not refit",
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}: {len(data['atom_types'])} atom-type rules, LJ {lj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
