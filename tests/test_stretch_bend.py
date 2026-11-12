"""Stretch-bend cross term: engine gradient and recovery by the fit."""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("scipy")

from fitting import fit as fitmod  # noqa: E402
from fitting.init_parameters import skeleton  # noqa: E402
from pmeff26_plugin.assign import assign  # noqa: E402
from pmeff26_plugin.core import energy_and_gradient, energy_components, hessian, optimize  # noqa: E402
from pmeff26_plugin.parameters import ParameterSet  # noqa: E402

TRUTH = {
    "bci": {"c3-h": -0.05, "hx-ow": 0.40},
    "bonds": {"c3-c3": [1.53, 620.0], "c3-h": [1.093, 680.0], "hx-ow": [0.96, 1100.0]},
    "angles": {"h|c3|h": [107.8, 70.0], "c3|c3|h": [110.7, 90.0],
               "c3|c3|c3": [112.0, 110.0], "hx|ow|hx": [104.5, 100.0]},
    "torsions": {"*|c3-c3|*": [[3, 2.9, 0.0]]},
    "stretch_bend": {"h|c3|h": [12.0, 12.0], "c3|c3|h": [25.0, 10.0],
                     "c3|c3|c3": [30.0, 30.0], "hx|ow|hx": [35.0, 35.0]},
}
SMILES = ["CC", "CCC", "CCCC", "CC(C)C", "CCCCC", "O"]


def _truth():
    data = skeleton()
    data.update(copy.deepcopy(TRUTH))
    return ParameterSet(data)


def _embedded(smiles, seed=11):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


@pytest.fixture(scope="module")
def synthetic():
    from rdkit import Chem

    truth = _truth()
    records, hessians = [], {}
    for n, smiles in enumerate(SMILES):
        mol = _embedded(smiles)
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


def test_cross_term_gradient_matches_finite_differences():
    mol = _embedded("CCC")
    a = assign(mol, _truth())
    assert len(a.system.sb_idx) == len(a.system.angle_idx)
    x = mol.GetConformer().GetPositions() + 0.05 * np.random.RandomState(1).normal(size=(11, 3))
    assert abs(energy_components(x, a.system)["cross"]) > 1e-3
    _, grad = energy_and_gradient(x, a.system)
    numeric = np.zeros_like(x)
    for i in range(x.shape[0]):
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, c] += 1e-5
            xm[i, c] -= 1e-5
            numeric[i, c] = (energy_and_gradient(xp, a.system)[0]
                             - energy_and_gradient(xm, a.system)[0]) / 2e-5
    assert np.allclose(grad, numeric, atol=1e-5)


def test_cross_term_uses_the_key_order_for_unequal_ends():
    """In c3|c3|h the first constant belongs to the C-C bond, whatever the atom order."""
    mol = _embedded("CC")
    s = assign(mol, _truth()).system
    types = ["c3", "c3"] + ["h"] * 6
    for (i, _j, k), (k1, k2) in zip(s.sb_idx, s.sb_k):
        if {types[i], types[k]} == {"c3", "h"}:
            assert (k1, k2) == ((25.0, 10.0) if types[i] == "c3" else (10.0, 25.0))


def test_fit_recovers_stretch_bend_constants(synthetic):
    records, hessians = synthetic
    data, report = fitmod.fit(records, hessians, skeleton())
    for key, (k1, k2) in TRUTH["stretch_bend"].items():
        got = data["stretch_bend"][key]
        assert got[0] == pytest.approx(k1, abs=2.0), key
        assert got[1] == pytest.approx(k2, abs=2.0), key
    for key, (r0, k) in TRUTH["bonds"].items():
        assert data["bonds"][key][0] == pytest.approx(r0, abs=2e-3), key
    stats = data["fit_statistics"]
    assert stats["validate"]["bond_rmsd"] < 2e-3 and stats["validate"]["freq_rmsd"] < 10.0
    assert "Stretch-bend cross terms: 4 angle classes" in report


def test_cross_terms_improve_frequencies_over_a_fit_without_them(synthetic):
    records, hessians = synthetic
    with_sb = fitmod.fit(records, hessians, skeleton())[0]["fit_statistics"]
    data, report = fitmod.fit(records, hessians, skeleton(), use_sb=False)
    assert data["stretch_bend"] == {} and "switched off" in report
    assert with_sb["validate"]["freq_rmsd"] < 0.5 * data["fit_statistics"]["validate"]["freq_rmsd"]
