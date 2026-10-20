"""A hypothetical additional battery charged only from observed net export.

This is a secondary application-fidelity diagnostic for signed net import, not
an estimate of PV generation, self-consumption ratios, costs, annual operation,
or real hardware performance. Existing batteries and the source of observed
export are unknown. Each complete 24-hour row is an independent episode starting
empty; remaining stored energy is reported without crediting a future day.
"""

from __future__ import annotations

from numbers import Real

import numpy as np


_DEFAULT_EFFICIENCY = float(np.sqrt(.9))


def _parameter(value, name, *, allow_zero=False, efficiency=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    if efficiency and value > 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def simulate_battery(profiles, interval_hours: float, *, capacity_kwh: float = 10.,
                     charge_power_kw: float = 4., discharge_power_kw: float = 4.,
                     eta_charge: float = _DEFAULT_EFFICIENCY,
                     eta_discharge: float = _DEFAULT_EFFICIENCY) -> dict:
    """Dispatch an additional export-charged battery on finite [N,24/96] rows.

    Inputs and ``post_net_kwh`` are signed kWh/interval: positive import,
    negative export. ``interval_hours`` must make each row exactly 24 hours.
    Power limits refer to the grid-facing energy entering/leaving the battery;
    charging and discharging efficiencies apply on the stored-energy side.
    Capacity may be zero to disable the device; each power limit may be zero to
    disable that flow direction. Efficiencies are in (0,1], and no grid charging
    or battery export is allowed.

    Returns scenario metadata, NumPy per-profile metric arrays, the revised net
    profiles, and ``soc_kwh`` with the initial state in column zero. AC throughput
    is charge input plus delivered discharge, not an equivalent-cycle measure.
    Conversion loss excludes leftover SOC. Conservation per row is
    charge_input = discharge_delivered + end_soc + conversion_loss.
    Only SOC roundoff within a float64 tolerance is corrected to physical bounds;
    input profiles and revised net flows are never clipped.
    """
    try:
        values = np.asarray(profiles)
    except (TypeError, ValueError) as error:
        raise ValueError("profiles must be finite real [N,24] or [N,96] values") from error
    if values.ndim != 2 or len(values) == 0 or values.shape[1] not in (24, 96):
        raise ValueError("profiles must be a nonempty [N,24] or [N,96] array")
    if values.dtype.kind not in "fiu":
        raise ValueError("profiles must contain real numeric values")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("profiles must contain only finite values")
    hours = _parameter(interval_hours, "interval_hours")
    if not np.isclose(hours * values.shape[1], 24., rtol=0, atol=1e-12):
        raise ValueError("interval_hours and profile width must describe a complete 24-hour day")
    capacity = _parameter(capacity_kwh, "capacity_kwh", allow_zero=True)
    charge_power = _parameter(charge_power_kw, "charge_power_kw", allow_zero=True)
    discharge_power = _parameter(discharge_power_kw, "discharge_power_kw", allow_zero=True)
    charge_efficiency = _parameter(eta_charge, "eta_charge", efficiency=True)
    discharge_efficiency = _parameter(eta_discharge, "eta_discharge", efficiency=True)

    post_net = np.empty_like(values)
    soc_trace = np.zeros((len(values), values.shape[1] + 1), dtype=np.float64)
    charge_input = np.zeros(len(values))
    discharge_delivered = np.zeros(len(values))
    tolerance = 64 * np.finfo(np.float64).eps * max(1., capacity)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for slot in range(values.shape[1]):
                net, soc = values[:, slot], soc_trace[:, slot]
                charge = np.minimum(np.minimum(np.maximum(-net, 0.), charge_power * hours),
                                    (capacity - soc) / charge_efficiency)
                delivered = np.minimum(np.minimum(np.maximum(net, 0.), discharge_power * hours),
                                       soc * discharge_efficiency)
                next_soc = soc + charge * charge_efficiency - delivered / discharge_efficiency
                if np.any(next_soc < -tolerance) or np.any(next_soc > capacity + tolerance):
                    raise ArithmeticError("battery state exceeded physical bounds beyond roundoff")
                soc_trace[:, slot + 1] = np.clip(next_soc, 0., capacity)
                post_net[:, slot] = net + charge - delivered
                charge_input += charge
                discharge_delivered += delivered
            baseline_import = np.maximum(values, 0.).sum(axis=1)
            baseline_export = np.maximum(-values, 0.).sum(axis=1)
            post_import = np.maximum(post_net, 0.).sum(axis=1)
            post_export = np.maximum(-post_net, 0.).sum(axis=1)
            charge_loss = charge_input * (1 - charge_efficiency)
            discharge_loss = discharge_delivered * (1 / discharge_efficiency - 1)
            per_profile = {
                "baseline_grid_import_kwh": baseline_import,
                "baseline_grid_export_kwh": baseline_export,
                "post_grid_import_kwh": post_import,
                "post_grid_export_kwh": post_export,
                "grid_import_reduction_kwh": baseline_import - post_import,
                "grid_export_reduction_kwh": baseline_export - post_export,
                "post_peak_import_power_kw": np.maximum(post_net, 0.).max(axis=1) / hours,
                "end_soc_kwh": soc_trace[:, -1].copy(),
                "charge_input_kwh": charge_input,
                "discharge_delivered_kwh": discharge_delivered,
                "ac_throughput_kwh": charge_input + discharge_delivered,
                "charge_loss_kwh": charge_loss,
                "discharge_loss_kwh": discharge_loss,
                "conversion_loss_kwh": charge_loss + discharge_loss,
            }
    except FloatingPointError as error:
        raise ValueError("battery calculation exceeds the supported float64 numeric range") from error
    scenario = {
        "name": "hypothetical_additional_battery_charged_only_from_observed_net_export",
        "version": 1,
        "capacity_kwh": capacity,
        "charge_power_kw": charge_power,
        "discharge_power_kw": discharge_power,
        "eta_charge": charge_efficiency,
        "eta_discharge": discharge_efficiency,
        "round_trip_efficiency": charge_efficiency * discharge_efficiency,
        "interval_hours": hours,
        "episode_hours": 24.,
        "initial_soc_kwh": 0.,
        "boundary_policy": "Independent daily episodes; reset SOC to zero each row, report leftover SOC without future-day credit",
        "grid_charging_allowed": False,
        "battery_export_allowed": False,
        "throughput_definition": "Grid-facing charge input plus delivered discharge, in kWh",
        "scope": "Secondary application-fidelity diagnostic; no annual-operation, cost, PV-attribution or real-hardware claim",
        "observed_data_caveat": "Gross generation and any existing batteries are unknown from signed net import/export",
    }
    return {"scenario": scenario, "per_profile": per_profile,
            "post_net_kwh": post_net, "soc_kwh": soc_trace}
