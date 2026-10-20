"""Conservation and chronological dispatch checks for an additional battery."""

import numpy as np
import pytest

from fluvius_cgan.battery import simulate_battery


def _day(*values, width=24):
    return np.pad(np.asarray(values, dtype=float), (0, width - len(values)))[None, :]


def test_known_efficiency_and_conservation_with_leftover_energy():
    x = _day(-2., 1.)
    original = x.copy()
    result = simulate_battery(x, 1., eta_charge=.8, eta_discharge=.5)
    metrics = result["per_profile"]
    np.testing.assert_allclose(result["post_net_kwh"][0, :2], [0., .2])
    assert metrics["baseline_grid_import_kwh"] == pytest.approx([1.])
    assert metrics["baseline_grid_export_kwh"] == pytest.approx([2.])
    assert metrics["grid_import_reduction_kwh"] == pytest.approx([.8])
    assert metrics["grid_export_reduction_kwh"] == pytest.approx([2.])
    assert metrics["charge_loss_kwh"] == pytest.approx([.4])
    assert metrics["discharge_loss_kwh"] == pytest.approx([.8])
    assert metrics["ac_throughput_kwh"] == pytest.approx([2.8])
    assert metrics["end_soc_kwh"] == pytest.approx([0.])
    np.testing.assert_array_equal(x, original)
    surplus = simulate_battery(_day(-2.), 1., eta_charge=.8, eta_discharge=.5)["per_profile"]
    assert surplus["end_soc_kwh"] == pytest.approx([1.6])
    assert surplus["conversion_loss_kwh"] == pytest.approx([.4])
    assert surplus["grid_import_reduction_kwh"] == pytest.approx([0.])


def test_defaults_are_fixed_and_round_trip_efficiency_is_point_nine():
    result = simulate_battery(_day(-1., 1.), 1.)
    scenario, metrics = result["scenario"], result["per_profile"]
    assert scenario["capacity_kwh"] == 10
    assert scenario["charge_power_kw"] == scenario["discharge_power_kw"] == 4
    assert scenario["eta_charge"] == scenario["eta_discharge"] == pytest.approx(np.sqrt(.9))
    assert scenario["round_trip_efficiency"] == pytest.approx(.9)
    assert metrics["grid_import_reduction_kwh"] == pytest.approx([.9])
    assert metrics["conversion_loss_kwh"] == pytest.approx([.1])
    assert metrics["post_peak_import_power_kw"] == pytest.approx([.1])
    assert not scenario["grid_charging_allowed"]
    assert "unknown" in scenario["observed_data_caveat"]


def test_grid_import_cannot_charge_and_each_day_starts_empty():
    x = np.vstack([_day(-5.), _day(5.), np.zeros((1, 24))])
    result = simulate_battery(x, 1., eta_charge=1, eta_discharge=1)
    np.testing.assert_array_equal(result["soc_kwh"][:, 0], 0)
    np.testing.assert_array_equal(result["soc_kwh"][1:], 0)
    np.testing.assert_array_equal(result["post_net_kwh"][1:], x[1:])
    np.testing.assert_array_equal(result["per_profile"]["end_soc_kwh"], [4., 0., 0.])


def test_dispatch_depends_on_order_and_never_borrows_future_export():
    x = np.vstack([_day(-2., 2.), _day(2., -2.)])
    result = simulate_battery(x, 1., eta_charge=1, eta_discharge=1)
    np.testing.assert_array_equal(result["per_profile"]["grid_import_reduction_kwh"], [2., 0.])
    np.testing.assert_array_equal(result["per_profile"]["end_soc_kwh"], [0., 2.])


def test_capacity_and_charge_discharge_power_limits_apply_at_each_interval():
    result = simulate_battery(_day(-8., -8., 8., 8.), 1., capacity_kwh=3.,
                              charge_power_kw=2., discharge_power_kw=1.,
                              eta_charge=1, eta_discharge=1)
    np.testing.assert_array_equal(result["soc_kwh"][0, :5], [0., 2., 3., 2., 1.])
    np.testing.assert_array_equal(result["post_net_kwh"][0, :4], [-6., -7., 7., 7.])
    assert result["per_profile"]["post_peak_import_power_kw"] == pytest.approx([7.])


def test_capacity_headroom_accounts_for_charging_loss_and_disabled_discharge():
    result = simulate_battery(_day(-4., 4.), 1., capacity_kwh=1.5,
                              eta_charge=.5, eta_discharge=.8, discharge_power_kw=0.)
    np.testing.assert_array_equal(result["post_net_kwh"][0, :2], [-1., 4.])
    assert result["per_profile"]["charge_input_kwh"] == pytest.approx([3.])
    assert result["per_profile"]["end_soc_kwh"] == pytest.approx([1.5])
    assert result["per_profile"]["conversion_loss_kwh"] == pytest.approx([1.5])
    assert result["per_profile"]["discharge_delivered_kwh"] == pytest.approx([0.])


def test_resolution_uses_interval_energy_and_reports_power_consistently():
    hourly = _day(-8., -8., 8., 8.)
    quarter_hourly = np.repeat(hourly / 4, 4, axis=1)
    a = simulate_battery(hourly, 1.)
    b = simulate_battery(quarter_hourly, .25)
    for key in a["per_profile"]:
        if key == "post_peak_import_power_kw":
            continue
        np.testing.assert_allclose(a["per_profile"][key], b["per_profile"][key], atol=1e-13)
    # The battery empties inside the last hour. Hourly averaging masks the
    # quarter-hour peak; interval resolution must remain part of the scenario.
    assert a["per_profile"]["post_peak_import_power_kw"] == pytest.approx([4.8])
    assert b["per_profile"]["post_peak_import_power_kw"] == pytest.approx([7.2])
    np.testing.assert_allclose(a["post_net_kwh"], b["post_net_kwh"].reshape(1, 24, 4).sum(axis=2), atol=1e-13)
    limited = simulate_battery(_day(-2., 2., width=96), .25,
                               eta_charge=1, eta_discharge=1)
    np.testing.assert_array_equal(limited["post_net_kwh"][0, :2], [-1., 1.])
    assert limited["per_profile"]["post_peak_import_power_kw"] == pytest.approx([4.])


def test_random_profiles_conserve_energy_and_stay_within_physical_bounds():
    x = np.random.default_rng(17).normal(0, 3, size=(50, 96))
    result = simulate_battery(x, .25, capacity_kwh=2.3, eta_charge=.73, eta_discharge=.87)
    m, soc, post = result["per_profile"], result["soc_kwh"], result["post_net_kwh"]
    assert np.all((soc >= 0) & (soc <= 2.3))
    assert np.all(m["conversion_loss_kwh"] >= 0)
    assert np.all(post[x > 0] >= 0) and np.all(post[x < 0] <= 0)
    assert np.all(np.abs(post) <= np.abs(x) + 1e-14)
    np.testing.assert_allclose(m["charge_input_kwh"],
                               m["discharge_delivered_kwh"] + m["end_soc_kwh"] + m["conversion_loss_kwh"], atol=1e-12)
    np.testing.assert_allclose(post.sum(axis=1) - x.sum(axis=1),
                               m["end_soc_kwh"] + m["conversion_loss_kwh"], atol=1e-12)
    np.testing.assert_allclose(m["grid_import_reduction_kwh"], m["discharge_delivered_kwh"], atol=1e-12)
    np.testing.assert_allclose(m["grid_export_reduction_kwh"], m["charge_input_kwh"], atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"capacity_kwh":0}, {"charge_power_kw":0}])
def test_disabled_battery_preserves_profiles(kwargs):
    x = _day(-4., 4.)
    result = simulate_battery(x, 1., **kwargs)
    np.testing.assert_array_equal(result["post_net_kwh"], x)
    np.testing.assert_array_equal(result["soc_kwh"], 0)


@pytest.mark.parametrize("bad", [[], np.zeros((0,24)), np.zeros((2,23)), np.zeros(24),
                                 np.full((1,24), np.nan), np.full((1,24), np.inf),
                                 np.ones((1,24), dtype=bool), np.ones((1,24), dtype=complex),
                                 np.full((1,24), "1")])
def test_invalid_profiles_raise(bad):
    with pytest.raises(ValueError):
        simulate_battery(bad, 1.)


@pytest.mark.parametrize("hours", [0, -1, .25, np.inf, np.nan, True, "1"])
def test_invalid_or_inconsistent_interval_raises(hours):
    with pytest.raises(ValueError):
        simulate_battery(_day(1.), hours)


@pytest.mark.parametrize("name", ["capacity_kwh", "charge_power_kw", "discharge_power_kw", "eta_charge", "eta_discharge"])
@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, True, "1"])
def test_invalid_parameters_raise(name, bad):
    with pytest.raises(ValueError):
        simulate_battery(_day(1.), 1., **{name:bad})


@pytest.mark.parametrize("name", ["eta_charge", "eta_discharge"])
@pytest.mark.parametrize("bad", [0, 1.01])
def test_efficiencies_must_be_positive_and_at_most_one(name, bad):
    with pytest.raises(ValueError):
        simulate_battery(_day(1.), 1., **{name:bad})
