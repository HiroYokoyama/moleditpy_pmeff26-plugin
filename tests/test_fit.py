"""End-to-end check of the fit: recover known parameters from synthetic data.

A ground-truth parameter set plays the role of DFT: molecules are optimized
with it, their Hessians, dipoles and energies become the "reference data",
and the fit must recover the parameters starting from the bare skeleton.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("scipy")

from fitting import fit as fitmod  # noqa: E402
from fitting.init_parameters import skeleton  # noqa: E402
from pmeff26_plugin.assign import assign  # noqa: E402
from pmeff26_plugin.core import energy_and_gradient, hessian, optimize  # noqa: E402
from pmeff26_plugin.parameters import ParameterSet  # noqa: E402

TRUTH = {
    "bci": {"c3-h": -0.05, "h-ow": 0.40},
    "bonds": {"c3-c3": [1.53, 620.0], "c3-h": [1.093, 680.0], "h-ow": [0.96, 1100.0]},
    "angles": {"h|c3|h": [107.8, 70.0], "c3|c3|h": [110.7, 90.0],
               "c3|c3|c3": [112.0, 110.0], "h|ow|h": [104.5, 100.0]},
    "torsions": {"*|c3-c3|*": [[3, 2.9, 0.0]]},
}


@pytest.fixture(scope="module")
def synthetic():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    data = skeleton()
    data.update(copy.deepcopy(TRUTH))
    truth = ParameterSet(data)
    records, hessians = [], {}
    for n, smiles in enumerate(["CC", "CCC", "CCCC", "CC(C)C", "CCCCC", "O"]):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        a = assign(mol, truth)
        x, result = optimize(mol.GetConformer().GetPositions(), a.system, max_iter=8000, f_tol=1e-5)
        assert result.converged
        centred = x - x.mean(axis=0)
        job = f"tr_{n:03d}"
        records.append({
            "id": job, "kind": "validate" if smiles == "CCCCC" else "train",
            "smiles": smiles, "molblock": Chem.MolToMolBlock(mol), "n_atoms": len(x),
            "coords": x.tolist(), "energy": -40.0 * n + result.energy / 627.5095,
            "dipole": (4.80321 * centred.T @ a.charges).tolist(), "lowdin": None,
            "usable": True, "reason": "",
        })
        hessians[job] = hessian(x, a.system)
    return records, hessians


def test_fit_recovers_the_generating_parameters(synthetic):
    records, hessians = synthetic
    data, report = fitmod.fit(records, hessians, skeleton())
    for key, (r0, k) in TRUTH["bonds"].items():
        assert data["bonds"][key][0] == pytest.approx(r0, abs=2e-3), key
        assert data["bonds"][key][1] == pytest.approx(k, rel=0.03), key
    for key, (theta, k) in TRUTH["angles"].items():
        assert data["angles"][key][0] == pytest.approx(theta, abs=0.5), key
        assert data["angles"][key][1] == pytest.approx(k, rel=0.10), key
    three_fold = [t for t in data["torsions"]["*|c3-c3|*"] if t[0] == 3][0]
    assert three_fold[1] == pytest.approx(2.9, rel=0.15) and three_fold[2] == 0.0
    assert data["bci"]["c3-h"] == pytest.approx(-0.05, abs=5e-3)
    assert data["bci"]["h-ow"] == pytest.approx(0.40, abs=5e-3)
    stats = data["fit_statistics"]
    assert stats["train"]["bond_rmsd"] < 2e-3 and stats["train"]["angle_rmsd"] < 0.3
    # The held-out molecule (pentane) was not fitted and must still be accurate.
    assert stats["validate"]["covered"] == 1
    assert stats["validate"]["bond_rmsd"] < 3e-3
    assert "validation" in report


def test_fitted_file_refuses_what_the_data_did_not_cover(synthetic):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from pmeff26_plugin.parameters import MissingParameterError

    records, hessians = synthetic
    data, _ = fitmod.fit(records, hessians, skeleton())
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    with pytest.raises(MissingParameterError):
        assign(mol, ParameterSet(data))
    assert energy_and_gradient  # imported for parity with the engine tests
