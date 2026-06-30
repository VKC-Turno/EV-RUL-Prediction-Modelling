"""SoH-pipeline artifact audit.

The handful of completely-aged vehicles are 100% of the model's end-of-life signal, so if their SoH
trajectory is a measurement artifact rather than real aging, the training target is corrupt. This module
flags, per vehicle, three artifact patterns we verified by hand on the aged cohort:

  CLIFF       - a single-month SoH drop >= CLIFF_PP. Li-ion cannot lose ~6pp of capacity in one month;
                this is a BMS capacity RE-ESTIMATION jump (e.g. Euler 217380: flat 90.6 for 4 months then
                -24.6pp). It corrupts the monthly delta-SoH TARGET the rate model trains on (the trajectory/
                rate is untrustworthy even if the endpoint is roughly right).

  STUCK_FLOOR - the series ENDS in a run of >= FLOOR_RUN identical values, at the vehicle's minimum, after a
                real drop (>= MIN_DROP from the start). A held/stale value (e.g. Euler 217158: stepped to
                79.0 then reported exactly 79.0 for 18 months). The trajectory has stopped updating.
                NOTE: where no raw signal exists (Euler/Bajaj) a few of these may be genuine recent
                plateaus - treat as "suspected", confirm against raw telemetry.

  ISO_FLOOR   - (only where soh_raw exists, i.e. Mahindra) the monotone isotonic envelope sits >= ISO_GAP
                below the raw coulomb signal, i.e. the raw measurement RECOVERED but the envelope pinned it
                low (e.g. Mahindra H48636: raw climbs back to 81% but iso freezes at 78.5%). The ENDPOINT
                is biased low -> a premature near-EoL reading.

A vehicle can trip several. `tainted` = any of them.
"""
import numpy as np
import pandas as pd

CLIFF_PP = 6.0       # single-month drop >= this is non-physical (re-estimation jump)
FLOOR_RUN = 5        # trailing identical-value run >= this many months = held value
MIN_DROP = 3.0       # ... only counts as a STUCK_FLOOR if the vehicle actually dropped this much overall
ISO_GAP = 2.0        # raw exceeds the isotonic envelope by >= this many pp = envelope pinned the endpoint low


def _trailing_run(s):
    """Length of the trailing run of byte-identical values."""
    n = 1
    for i in range(len(s) - 1, 0, -1):
        if s[i] == s[i - 1]:
            n += 1
        else:
            break
    return n


def audit(F, eol):
    """F: feature rows with columns vin, age_months, soh, and optionally soh_raw.
    Returns one row per vehicle with the artifact flags + the raw measurements behind them."""
    has_raw = "soh_raw" in F.columns
    rows = []
    for v, gg in F.groupby("vin"):
        gg = gg.sort_values("age_months")
        s = gg["soh"].to_numpy()
        if len(s) < 2:
            continue
        d = np.diff(s)
        worst = float(-d.min())                       # biggest one-month drop (pp)
        tail = _trailing_run(s)
        cliff = worst >= CLIFF_PP
        stuck = bool(tail >= FLOOR_RUN and s[-1] == s.min() and (s[0] - s[-1]) >= MIN_DROP)
        iso_gap, isofl = 0.0, False
        if has_raw:
            raw = gg["soh_raw"].to_numpy()
            iso_gap = float(np.max(raw - s))          # how far the envelope sits below raw
            isofl = iso_gap >= ISO_GAP
        rows.append(dict(vin=v, n=int(len(s)), worst_drop=round(worst, 1), tail_run=int(tail),
                         iso_gap=round(iso_gap, 1), CLIFF=cliff, STUCK_FLOOR=stuck, ISO_FLOOR=isofl,
                         tainted=bool(cliff or stuck or isofl), aged=bool(s.min() <= eol)))
    return pd.DataFrame(rows)


def summary(F, eol):
    """Fleet-level counts per artifact type + the completely-aged taint."""
    R = audit(F, eol)
    ag = R[R.aged]
    return dict(n=len(R), clean=int((~R.tainted).sum()), tainted=int(R.tainted.sum()),
                cliff=int(R.CLIFF.sum()), stuck=int(R.STUCK_FLOOR.sum()), iso=int(R.ISO_FLOOR.sum()),
                aged=len(ag), aged_tainted=int(ag.tainted.sum()), table=R)
