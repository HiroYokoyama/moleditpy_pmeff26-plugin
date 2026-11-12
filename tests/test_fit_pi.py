"""The pi-term fit on synthetic data generated with a known pi model.

Conjugated hydrocarbons are optimized with a ground-truth parameter set whose
bond-order dependence comes only from the Hueckel pi term (one sigma bond
``c2~c2`` for formal single and double bonds alike). The fit must find pi
parameters that reproduce those structures, including a held-out molecule.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("scipy")

from fitting import fit as fitmod  # noqa: E402
from fitting.init_parameters import skeleton  # noqa: E402
from pmeff26_plugin.assign import assign  # noqa: E402
from pmeff26_plugin.core import hessian, optimize  # noqa: E402
from pmeff26_plugin.parameters import ParameterSet  # noqa: E402

TRUTH = {
    "bci": {"c2-h": -0.08, "ca-h": -0.10},
    "bonds": {"c2~c2": [1.50, 600.0], "ca~ca": [1.50, 600.0], "c2-h": [1.087, 720.0],
              "ca-h": [1.085, 730.0]},
    "angles": {"c2|c2|h": [121.0, 75.0], "h|c2|h": [117.0, 60.0], "c2|c2|c2": [124.0, 120.0],
               "ca|ca|ca": [120.0, 130.0], "ca|ca|h": [120.0, 80.0]},
    "torsions": {"*|c2~c2|*": [[2, 3.0, 180.0]], "*|ca~ca|*": [[2, 5.0, 180.0]]},
    "impropers": {"c2": 6.0, "ca": 6.0},
    "pi": {"enabled": True,
           "bonds": {"c2~c2": [45.0, 1.8, 1.40], "ca~ca": [45.0, 1.8, 1.40]}},
}
MOLECULES = ["C=C", "C=CC=C", "c1ccccc1", "C=CC=CC=C"]  # the last one is held out


@pytest.fixture(scope="module")
def synthetic():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    data = skeleton()
    data.update(copy.deepcopy(TRUTH))
    truth = ParameterSet(data)
    records, hessians = [], {}
    for n, smiles in enumerate(MOLECULES):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=7)
        AllChem.MMFFOptimizeMolecule(mol)
        a = assign(mol, truth)
        x, result = optimize(mol.GetConformer().GetPositions(), a.system, max_iter=8000, f_tol=1e-5)
        assert result.converged, smiles
        centred = x - x.mean(axis=0)
        job = f"tr_{n:03d}"
        records.append({
            "id": job, "kind": "validate" if smiles == MOLECULES[-1] else "train",
            "smiles": smiles, "molblock": Chem.MolToMolBlock(mol), "n_atoms": len(x),
            "coords": x.tolist(), "energy": -40.0 * n + result.energy / 627.5095,
            "dipole": (4.80321 * centred.T @ a.charges).tolist(), "lowdin": None,
            "usable": True, "reason": "",
        })
        hessians[job] = hessian(x, a.system)
    return records, hessians


def test_pi_fit_reproduces_conjugated_structures(synthetic):
    records, hessians = synthetic
    data, report = fitmod.fit(records, hessians, skeleton())
    assert data["pi"]["enabled"]
    beta, delta, rref = data["pi"]["bonds"]["c2~c2"]
    assert data["pi"]["bonds"]["ca~ca"] == [beta, delta, rref]  # one class per element pair
    assert 1.25 < rref < 1.50 and beta > 5.0
    stats = data["fit_statistics"]
    assert stats["train"]["covered"] == 3 and stats["train"]["bond_rmsd"] < 4e-3
    # Hexatriene was not fitted: its bond alternation must come from the pi term.
    assert stats["validate"]["covered"] == 1 and stats["validate"]["bond_rmsd"] < 8e-3
    assert "pi-electron term fitted on 3 molecules" in report


def test_pi_fit_beats_one_sigma_bond_without_pi(synthetic):
    """With beta = 0 a single c2~c2 bond cannot give both bond lengths of butadiene."""
    records, hessians = synthetic
    train = [(r, fitmod.mol_of(r)) for r in records if r["kind"] == "train"]
    prov = fitmod.provisional(skeleton(), [m for _r, m in train], use_pi=True)
    classes = fitmod.angle_classes(train, prov)
    _cols, _lower, a_mat, b_vec, spans = fitmod.bonded_system(train, hessians, prov, classes)
    result = fitmod.fit_pi(prov, a_mat, b_vec, spans, log=lambda *_: None)
    assert result["after"] < 0.3 * result["before"]


def test_no_pi_switch_keeps_formal_bond_orders(synthetic):
    records, hessians = synthetic
    data, report = fitmod.fit(records, hessians, skeleton(), use_pi=False)
    assert not data["pi"]["enabled"] and "c2=c2" in data["bonds"]
    assert "switched off" in report


def test_isolated_pi_type_atom_is_left_out_of_the_pi_system(params):
    """A lone pi-type atom must not make the electron count odd (and refuse the molecule)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    data = copy.deepcopy(params.data)
    for entry in data["atom_types"]:
        if entry["type"] == "oh":  # pretend hydroxyl O is a one-electron pi type
            entry.update(pi_electrons=1, pi_alpha=0.0)
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    a = assign(mol, ParameterSet(data), allow_missing=True)
    assert len(a.system.pi_atoms) == 0
    assert np.isfinite(a.system.masses).all()
