"""Per-profile load-shape key performance indicators (vectorised over rows).

Two families are provided:

``price_kpis``  the five load-shape parameters of Price (2010, LBNL-3713E) as
                described in Li et al. (2025, Sec. 2.4):
                near-peak  = 97.5th percentile of the day,
                near-base  = 2.5th percentile,
                high-load duration = hours with load closer to near-peak than
                             to near-base (x > (peak + base) / 2),
                rise time  = hours from the last time the load is close to base
                             (x <= base + 0.05 (peak - base)) before the first
                             high-load hour, to that first high-load hour,
                fall time  = hours from the last high-load hour to the next
                             time the load is close to base.
                Rise and fall are measured within the calendar day; when no
                close-to-base hour exists before (after) the high-load period
                the day boundary is used.

``reference_kpis``  the definitions actually implemented in cGAN_EB utils.py
                (2nd-highest / 2nd-lowest value, count of hours above their
                midpoint, and the |idxmax - idxmin| constructions for rise and
                fall), reproduced only for comparability with published values.

``extra_kpis``  daily energy, maximum, minimum, peak hour, largest up/down
                ramps and exported energy (sum of negative net load).
"""
from __future__ import annotations

import numpy as np

PRICE = ["near_peak", "near_base", "high_load_duration", "rise_time", "fall_time"]
REFERENCE = ["ref_peak", "ref_base", "ref_duration", "ref_rise", "ref_fall"]
EXTRA = ["daily_energy", "evening_energy", "max_load", "min_load", "peak_hour", "max_ramp_up", "max_ramp_down", "export_energy"]
DISCRETE = {"high_load_duration", "rise_time", "fall_time", "ref_duration", "ref_rise", "ref_fall", "peak_hour"}
UNITS = {k: "kWh" for k in PRICE + REFERENCE + EXTRA}
UNITS.update({k: "h" for k in DISCRETE})


def price_kpis(X: np.ndarray, dt: float = 1.0) -> dict[str, np.ndarray]:
    X = np.asarray(X, np.float64)
    n, T = X.shape
    peak = np.percentile(X, 97.5, axis=1)
    base = np.percentile(X, 2.5, axis=1)
    high = X > ((peak + base) / 2)[:, None]
    close = X <= (base + 0.05 * (peak - base))[:, None]
    t = np.arange(T)

    has_high = high.any(1)
    first_high = np.where(has_high, high.argmax(1), 0)
    last_high = np.where(has_high, T - 1 - high[:, ::-1].argmax(1), 0)
    # last close-to-base index <= first_high (else 0)
    before = close & (t[None, :] <= first_high[:, None])
    rise_start = np.where(before.any(1), T - 1 - before[:, ::-1].argmax(1), 0)
    # first close-to-base index >= last_high (else T - 1)
    after = close & (t[None, :] >= last_high[:, None])
    fall_end = np.where(after.any(1), after.argmax(1), T - 1)

    return {
        "near_peak": peak,
        "near_base": base,
        "high_load_duration": high.sum(1) * dt,
        "rise_time": np.where(has_high, first_high - rise_start, 0) * dt,
        "fall_time": np.where(has_high, fall_end - last_high, 0) * dt,
    }


def reference_kpis(X: np.ndarray, dt: float = 1.0) -> dict[str, np.ndarray]:
    X = np.asarray(X, np.float64)
    s = np.sort(X, axis=1)
    peak2, base2 = s[:, -2], s[:, 1]
    mid = (peak2 + base2) / 2
    dp = np.abs(X - peak2[:, None])
    db = np.abs(X - base2[:, None])
    return {
        "ref_peak": peak2,
        "ref_base": base2,
        "ref_duration": (X > mid[:, None]).sum(1) * dt,
        "ref_rise": np.abs(dp.argmax(1) - dp.argmin(1)) * dt,
        "ref_fall": np.abs(db.argmax(1) - db.argmin(1)) * dt,
    }


def extra_kpis(X: np.ndarray, dt: float = 1.0) -> dict[str, np.ndarray]:
    X = np.asarray(X, np.float64)
    d = np.diff(X, axis=1)
    t = np.arange(X.shape[1]) * dt
    evening = (t >= 17) & (t < 22)
    return {
        "daily_energy": X.sum(1),
        "evening_energy": X[:, evening].sum(1),
        "max_load": X.max(1),
        "min_load": X.min(1),
        "peak_hour": X.argmax(1) * dt,
        "max_ramp_up": d.max(1),
        "max_ramp_down": d.min(1),
        "export_energy": np.clip(-X, 0, None).sum(1),
    }


def all_kpis(X: np.ndarray, dt: float = 1.0) -> dict[str, np.ndarray]:
    out = price_kpis(X, dt)
    out.update(reference_kpis(X, dt))
    out.update(extra_kpis(X, dt))
    return out
