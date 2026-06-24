"""Remaining-distance-to-end-of-life — RUL expressed in kilometres.

The model combines two things we already produce:
  • the SoH degradation forecast  -> months until SoH reaches an end-of-life threshold (TIME-to-EoL);
  • the vehicle's recent usage rate (km/month from the odometer).

    remaining_km(eol) = km_per_month × months_until_SoH_reaches(eol)

Rationale: in this fleet degradation is calendar/condition-driven, not mileage-driven (same km ≠ same
SoH; usage proxies mislead). So the battery ages out over TIME, and kilometres accrue at the vehicle's
usage rate — a high-utilisation vehicle therefore delivers MORE total km before the same calendar-driven
SoH threshold than a low-utilisation one.

EoL thresholds (SoH %):
  80  end of FIRST life (range no longer ideal; second-life trigger)
  70  reduced-range end of usable first life
  60  true end of life (battery effectively spent)

Returns per threshold: km remaining (int), 0 if the vehicle is already at/below it, or None if the
forecast does not reach it within its horizon (report as ">horizon") / usage rate is unknown.
"""
from __future__ import annotations
import numpy as np

MS_PER_MONTH = 30.4375 * 86400 * 1000.0
EOLS = (80, 70, 60)


def km_per_month(ages, odos):
    """Average km/month from per-month (age_months, odometer) pairs. None if not derivable.
    Robust to the NaN/zero odometer months (e.g. Mahindra's intellicar-only months)."""
    a = np.asarray(ages, dtype="float64"); o = np.asarray(odos, dtype="float64")
    m = np.isfinite(a) & np.isfinite(o) & (o > 0)
    a, o = a[m], o[m]
    if len(a) < 2:
        return None
    order = np.argsort(a); a, o = a[order], o[order]
    span = a[-1] - a[0]
    if span <= 0:
        return None
    return max((o[-1] - o[0]) / span, 0.0)


def remaining_km(obs, fc, kmpm, eols=EOLS):
    """obs, fc: lists of [ms, soh] (calculated history + forecast); kmpm: km/month.
    Returns {eol: km|0|None}. The SoH trajectory is non-increasing, so the first crossing is unique."""
    pts = list(obs or []) + list(fc or [])
    if len(pts) < 2 or kmpm is None or kmpm <= 0 or not obs:
        return {e: None for e in eols}
    xs = np.array([p[0] for p in pts], dtype="float64")
    ys = np.array([p[1] for p in pts], dtype="float64")
    now_ms, cur = float(obs[-1][0]), float(obs[-1][1])
    out = {}
    for e in eols:
        if cur <= e:
            out[e] = 0                                   # already at/below this threshold
            continue
        below = np.where((ys <= e) & (xs >= now_ms))[0]
        if len(below) == 0:
            out[e] = None                                # not reached within the forecast horizon
            continue
        j = below[0]
        x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
        cross = x1 if y0 == y1 else x0 + (x1 - x0) * (y0 - e) / (y0 - y1)
        months = max((cross - now_ms) / MS_PER_MONTH, 0.0)
        out[e] = int(round(kmpm * months))
    return out


def headline(rem, cur_soh):
    """Pick the most relevant remaining-km number to surface: to 80% if the vehicle is still above it,
    otherwise to 60% (true EoL). Returns (km|None, label)."""
    if cur_soh > 80 and rem.get(80) is not None:
        return rem[80], "80% (end of first life)"
    if rem.get(60):
        return rem[60], "60% (end of life)"
    if rem.get(70):
        return rem[70], "70%"
    return None, None
