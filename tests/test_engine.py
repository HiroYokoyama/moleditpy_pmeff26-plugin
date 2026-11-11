"""Engine tests: gradients, optimizer, pi term, vibrations, no-fallback errors."""

from __future__ import annotations

import numpy as np
import pytest

from pmeff26_plugin import forcefield as ff
from pmeff26_plugin.assign import assign
from pmeff26_plugin.core import (
    energy_and_gradient,
    energy_components,
    optimize,
    pi_bond_orders,
    vibrational_analysis,
)
from pmeff26_plugin.parameters import (
    MissingParameterError,
    ParameterSet,
    UnsupportedAtomError,
    angle_key,
    bond_key,
    default_parameters,
    torsion_key,
)


def _coords(mol, jitter=0.0, seed=1):
    x = ff.conformer_coords(mol)
    if jitter:
        x = x + np.random.RandomState(seed).normal(0.0, jitter, x.shape)
    return x


def _numeric_gradient(x, system, h=1e-5):
    g = np.zeros_like(x)
    flat = x.ravel()
    for c in range(flat.size):
        keep = flat[c]
        flat[c] = keep + h
        ep, _ = energy_and_gradient(x, system)
        flat[c] = keep - h
        em, _ = energy_and_gradient(x, system)
        flat[c] = keep
        g.ravel()[c] = (ep - em) / (2 * h)
    return g


@pytest.mark.parametrize(
    "smiles", ["CC", "CCC", "O", "C=C", "C=CC=C", "c1ccccc1", "C#C", "CC=C"]
)
def test_analytic_gradient_matches_numeric(smiles, params, embed):
    mol = embed(smiles)
    system = assign(mol, params).system
    x = _coords(mol, jitter=0.05)
    _, g = energy_and_gradient(x, system)
    assert np.allclose(g, _numeric_gradient(x, system), atol=2e-4, rtol=1e-5)


def test_components_sum_to_total(params, embed):
    mol = embed("C=CC=C")
    comp = energy_components(_coords(mol), assign(mol, params).system)
    assert comp["total"] == pytest.approx(
        sum(v for k, v in comp.items() if k != "total")
    )
    assert comp["pi"] < 0.0


def test_optimizer_converges_and_lowers_energy(params, embed):
    mol = embed("CCC")
    system = assign(mol, params).system
    x0 = _coords(mol, jitter=0.15)
    e0, _ = energy_and_gradient(x0, system)
    x1, result = optimize(x0, system, max_iter=3000, f_tol=1e-3)
    assert result.converged and result.max_force <= 1e-3
    assert result.energy < e0
    cc = np.linalg.norm(x1[0] - x1[1])
    assert 1.50 < cc < 1.56


def test_optimize_rdkit_mol_writes_coordinates(params, embed):
    mol = embed("CC")
    before = _coords(mol).copy()
    ok, result = ff.optimize_rdkit_mol(mol, params=params)
    assert ok and result.converged
    assert not np.allclose(before, _coords(mol))


def test_cancel_keeps_input_geometry(params, embed):
    mol = embed("CCC")
    before = _coords(mol).copy()
    with pytest.raises(ff.OptimizationCancelled):
        ff.optimize_rdkit_mol(mol, params=params, cancel_callback=lambda: True)
    assert np.array_equal(before, _coords(mol))


def test_benzene_pi_bond_orders_are_equal(params, embed):
    mol = embed("c1ccccc1")
    system = assign(mol, params).system
    x, result = optimize(_coords(mol), system, max_iter=3000)
    assert result.converged
    orders = pi_bond_orders(x, system)
    assert np.allclose(orders, 2.0 / 3.0, atol=1e-3)
    ring = [np.linalg.norm(x[i] - x[(i + 1) % 6]) for i in range(6)]
    assert max(ring) - min(ring) < 1e-3


def test_butadiene_bond_alternation_from_one_bond_type(params, embed):
    # All three C-C bonds share the single "c2~c2" parameter; the pi term
    # alone must make the formal double bonds shorter than the central one.
    mol = embed("C=CC=C")
    a = assign(mol, params)
    assert set(k for k in a.bond_keys if "~" in k) == {"c2~c2"}
    x, result = optimize(_coords(mol), a.system, max_iter=3000)
    assert result.converged
    orders = pi_bond_orders(x, a.system)
    terminal = np.linalg.norm(x[0] - x[1])
    central = np.linalg.norm(x[1] - x[2])
    assert central - terminal > 0.03
    assert orders.max() > 0.8 and orders.min() < 0.6


def test_twisting_a_double_bond_costs_pi_energy(params, embed):
    from rdkit.Chem import rdMolTransforms

    mol = embed("C=C")
    system = assign(mol, params).system
    h_a = [n.GetIdx() for n in mol.GetAtomWithIdx(0).GetNeighbors() if n.GetIdx() != 1][0]
    h_b = [n.GetIdx() for n in mol.GetAtomWithIdx(1).GetNeighbors() if n.GetIdx() != 0][0]
    energies = []
    for angle in (0.0, 45.0, 90.0):
        rdMolTransforms.SetDihedralDeg(mol.GetConformer(), h_a, 0, 1, h_b, angle)
        energies.append(energy_components(_coords(mol), system)["pi"])
    assert energies[0] < energies[1] < energies[2]


def test_water_has_three_real_vibrations(params, embed):
    mol = embed("O")
    system = assign(mol, params).system
    x, result = optimize(_coords(mol), system, f_tol=1e-5)
    assert result.converged
    vib = vibrational_analysis(x, system)
    assert vib["is_minimum"] and len(vib["frequencies"]) == 3
    assert 1000.0 < vib["frequencies"][0] < 2500.0  # bend, cm^-1
    assert 3000.0 < vib["frequencies"][-1] < 4500.0  # O-H stretch, cm^-1


def test_energy_offset_only_when_every_increment_exists(params, embed):
    assert assign(embed("CC"), params).energy_offset == pytest.approx(
        2 * -37.8 + 6 * -0.5
    )
    assert assign(embed("O"), params).energy_offset is None


# --- no fallback ------------------------------------------------------------


def test_missing_parameters_are_refused_not_guessed(params, embed):
    with pytest.raises(MissingParameterError) as err:
        assign(embed("CO"), params)
    assert "bond c3-oh" in err.value.missing
    assert any(m.startswith("angle") for m in err.value.missing)
    assert "no approximate" in str(err.value)


def test_untyped_atom_is_refused(params, embed):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCl"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    with pytest.raises(UnsupportedAtomError):
        assign(mol, params)


def test_open_shell_pi_system_is_refused(params, embed):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("[CH2]C=C"))  # allyl radical
    AllChem.EmbedMolecule(mol, randomSeed=1)
    with pytest.raises(UnsupportedAtomError):
        assign(mol, params)


def test_shipped_parameter_file_loads_and_has_no_hidden_defaults(embed):
    shipped = default_parameters()
    assert isinstance(shipped, ParameterSet)
    assert {t["type"] for t in shipped.atom_types} >= {"h", "c3", "c2", "ca", "o"}
    for entry in shipped.atom_types:
        assert entry["type"] in shipped.vdw
    if not shipped.bonds:  # unfitted skeleton: every molecule must be refused
        with pytest.raises(MissingParameterError):
            assign(embed("CC"), shipped)


def test_keys_are_canonical():
    assert bond_key("h", "c3", "-") == bond_key("c3", "h", "-") == "c3-h"
    assert angle_key("h", "c3", "c3") == angle_key("c3", "c3", "h") == "c3|c3|h"
    assert angle_key("c3r3", "c3r3", "c3r3", 3).endswith("@3")
    assert torsion_key("h", "c3", "c2", "c2", "-") == torsion_key(
        "c2", "c2", "c3", "h", "-"
    )


# --- metals -----------------------------------------------------------------


def test_periodic_angle_has_minima_at_90_and_180_and_exact_gradient():
    from pmeff26_plugin.core import System

    system = System(
        masses=np.ones(3),
        angle_idx=np.array([[0, 1, 2]]),
        angle_t0=np.zeros(1),
        angle_k=np.array([80.0]),
        angle_n=np.array([4]),
    )

    def energy(deg):
        t = np.radians(deg)
        x = np.array([[2.0, 0, 0], [0, 0, 0], [2.0 * np.cos(t), 2.0 * np.sin(t), 0]])
        return energy_and_gradient(x, system)[0], x

    assert energy(90.0)[0] == pytest.approx(0.0, abs=1e-9)
    assert energy(180.0)[0] == pytest.approx(0.0, abs=1e-9)
    assert energy(135.0)[0] == pytest.approx(80.0 / 16 * 2)
    x = energy(117.0)[1] + 0.01
    _, g = energy_and_gradient(x, system)
    assert np.allclose(g, _numeric_gradient(x, system), atol=1e-6)


def test_shipped_types_cover_the_metal_set():
    from rdkit import Chem

    from pmeff26_plugin.assign import atom_types

    shipped = default_parameters()
    for smiles, expected in (
        ("Cl[Zn]Cl", "zn2"),
        ("Cl[Ti](Cl)(Cl)Cl", "ti4"),
        ("[NH3]->[Pt](Cl)(Cl)<-[NH3]", "pt4"),
        ("O=C=[Ni](=C=O)(=C=O)=C=O", "ni4"),
    ):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert expected in atom_types(mol, shipped), smiles
    iron = Chem.MolFromSmiles("Cl[Fe]Cl")
    with pytest.raises(UnsupportedAtomError):
        atom_types(iron, shipped)
