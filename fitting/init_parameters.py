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

# One simple molecule per element to read RDKit's UFF like-pair LJ parameters.
_PROBE = {
    "H": "[HH]", "B": "BB", "C": "CC", "N": "NN", "O": "OO", "F": "FF",
    "Si": "[SiH3][SiH3]", "P": "PP", "S": "SS", "Cl": "ClCl", "Br": "BrBr", "I": "II",
    "Li": "[Li]Cl.[Li]Cl", "Na": "[Na]Cl.[Na]Cl", "K": "[K]Cl.[K]Cl",
    "Mg": "Cl[Mg]Cl.Cl[Mg]Cl", "Al": "Cl[Al](Cl)Cl.Cl[Al](Cl)Cl",
    "Ge": "[GeH4].[GeH4]", "Sn": "[SnH4].[SnH4]",
    "Ti": "Cl[Ti](Cl)(Cl)Cl.Cl[Ti](Cl)(Cl)Cl", "Hg": "Cl[Hg]Cl.Cl[Hg]Cl",
}


# UFF x_I (Angstrom) and D_I (kcal/mol) for metals RDKit's UFF typer does not
# type in the probe molecules. Entered from the published table (Rappe et al.
# 1992, Table I); NOT machine-verified against an implementation.
_UFF_TABLE = {
    "Ca": (3.399, 0.238), "Cr": (3.023, 0.015), "Ni": (2.834, 0.015),
    "Cu": (3.495, 0.005), "Zn": (2.763, 0.124), "Pd": (2.899, 0.048),
    "Pt": (2.754, 0.080),
}


def uff_lj(symbol: str) -> list[float]:
    """Return [rmin/2, epsilon] of the UFF like-pair potential for *symbol*."""
    if symbol in _UFF_TABLE:
        x_i, d_i = _UFF_TABLE[symbol]
        return [round(x_i / 2.0, 4), d_i]
    mol = Chem.AddHs(Chem.MolFromSmiles(_PROBE[symbol]))
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol][:2]
    x_ij, d_ij = rdForceFieldHelpers.GetUFFVdWParams(mol, idx[0], idx[1])
    return [round(x_ij / 2.0, 4), round(d_ij, 4)]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if OUT.exists() and "--force" not in argv:
        print(f"{OUT} exists; pass --force to overwrite (this discards fitted values)")
        return 1
    lj = {el: uff_lj(el) for el in list(_PROBE) + list(_UFF_TABLE)}
    data = {
        "name": "PMEFF26",
        "version": "0.0.0-unfitted",
        "units": "kcal/mol, Angstrom, degree, e; bond/angle k in E = 1/2 k x^2",
        "reference": "not fitted yet",
        "vdw_source": "UFF per-element Lennard-Jones (Rappe et al. 1992), not refit; "
        "Ca, Cr, Ni, Cu, Zn, Pd, Pt entered from the published table, unverified",
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
