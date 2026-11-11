"""Shared fixtures: a small hand-written parameter set for engine tests.

The values are plausible but NOT fitted; they exist only so the engine's
mathematics can be tested independently of the shipped parameter file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_PARAMETERS = {
    "name": "PMEFF26-test",
    "version": "test",
    "scale14": {"vdw": 0.5, "elec": 0.8333},
    "atom_types": [
        {"type": "h", "smarts": "[#1]", "element": "H"},
        {"type": "c3", "smarts": "[#6X4]", "element": "C"},
        {"type": "c2", "smarts": "[#6X3]", "element": "C", "planar": True,
         "pi_electrons": 1, "pi_alpha": 0.0},
        {"type": "ca", "smarts": "[c]", "element": "C", "planar": True,
         "pi_electrons": 1, "pi_alpha": 0.0},
        {"type": "c1", "smarts": "[#6X2]", "element": "C", "linear": True},
        {"type": "ow", "smarts": "[#8X2H2]", "element": "O"},
        {"type": "oh", "smarts": "[#8X2H1]", "element": "O"},
    ],
    "vdw": {"h": [1.443, 0.044], "c3": [1.9255, 0.105], "c2": [1.9255, 0.105],
            "ca": [1.9255, 0.105], "c1": [1.9255, 0.105], "ow": [1.75, 0.06],
            "oh": [1.75, 0.06]},
    "bci": {"c3-h": -0.05, "c2-h": -0.08, "ca-h": -0.1, "h-ow": 0.4, "c1-h": -0.2,
            "c2-c3": 0.02},
    "bonds": {"c3-c3": [1.53, 620.0], "c3-h": [1.093, 680.0], "c2-h": [1.085, 720.0],
              "ca-h": [1.084, 730.0], "h-ow": [0.96, 1100.0], "c2~c2": [1.50, 600.0],
              "ca~ca": [1.50, 600.0], "c1#c1": [1.20, 2200.0], "c1-h": [1.062, 800.0],
              "c2-c3": [1.50, 640.0]},
    "angles": {"h|c3|h": [107.8, 70.0], "c3|c3|h": [110.7, 90.0],
               "c3|c3|c3": [112.0, 110.0], "h|ow|h": [104.5, 100.0],
               "h|c2|h": [117.0, 70.0], "c2|c2|h": [121.0, 80.0],
               "c2|c2|c2": [124.0, 120.0], "ca|ca|ca": [120.0, 130.0],
               "ca|ca|h": [120.0, 80.0], "c1|c1|h": [180.0, 40.0],
               "c2|c3|h": [110.5, 90.0], "c2|c2|c3": [124.5, 110.0],
               "c3|c2|h": [116.0, 80.0]},
    "torsions": {"*|c3-c3|*": [[3, 2.9, 0.0]], "*|c2~c2|*": [[2, 4.0, 180.0]],
                 "*|ca~ca|*": [[2, 6.0, 180.0]], "*|c2-c3|*": [[3, 2.0, 0.0]]},
    "impropers": {"c2": 6.0, "ca": 6.0},
    "pi": {"enabled": True,
           "bonds": {"c2~c2": [60.0, 2.0, 1.40], "ca~ca": [60.0, 2.0, 1.40]}},
    "increments": {"types": {"h": -0.5, "c3": -37.8}, "bonds": {}},
}


@pytest.fixture(scope="session")
def params():
    from pmeff26_plugin.parameters import ParameterSet

    return ParameterSet(TEST_PARAMETERS, "test")


@pytest.fixture(scope="session")
def embed():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def _embed(smiles: str, seed: int = 7):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0, smiles
        return mol

    return _embed
