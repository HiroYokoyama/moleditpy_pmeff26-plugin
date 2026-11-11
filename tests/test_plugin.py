"""Plugin entry-point tests (headless; a mock PluginContext)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pmeff26_plugin as plugin


def test_initialize_registers_everything_and_no_metal_override():
    ctx = MagicMock()
    plugin.initialize(ctx)
    name = ctx.register_optimization_method.call_args[0][0]
    assert name.startswith("PMEFF26") and plugin.PLUGIN_VERSION in name
    tools = [c[0][0] for c in ctx.add_analysis_tool.call_args_list]
    assert tools == ["PMEFF26 Single-Point Energy", "PMEFF26 Vibrational Analysis"]
    menus = [c[0][0] for c in ctx.add_menu_action.call_args_list]
    assert menus == ["Settings/PMEFF26 Setting"]
    assert not any("Override" in m for m in menus)


def test_energy_report_refuses_unparameterized_molecule(embed, monkeypatch, params):
    from pmeff26_plugin import forcefield
    from pmeff26_plugin.parameters import MissingParameterError

    monkeypatch.setattr(forcefield, "parameterize", lambda mol: forcefield.assign(mol, params))
    ok = plugin._energy_report(embed("CC"))
    assert ok["energy_offset"] is not None and "pi" in ok
    try:
        plugin._energy_report(embed("CO"))
    except MissingParameterError as exc:
        assert exc.missing
    else:
        raise AssertionError("methanol must be refused by the test parameter set")


def test_iteration_budget_is_bounded(monkeypatch):
    monkeypatch.setattr(plugin, "load_settings", lambda: {"max_iterations": "junk"})
    assert plugin._iteration_budget() == plugin._MAX_ITER
    monkeypatch.setattr(plugin, "load_settings", lambda: {"max_iterations": 10**9})
    assert plugin._iteration_budget() == 100000
