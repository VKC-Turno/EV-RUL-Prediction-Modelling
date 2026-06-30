"""Curated training-set selection.

The current training signal is dominated by artifact-driven early-failures (see src/soh_audit.py). This
module instead picks vehicles by what they teach AFTER the SoH is robustly cleaned: it √t-smooths each
vehicle (Theil-Sen line on sqrt(age), which ignores cliffs / stuck-floors), projects to the warranty
deadline, and buckets:

  GRACEFUL     - aged or near-warranty AND projects to SURVIVE (>= EoL) AND shows real decline (>= MIN_DECLINE).
                 The "ages gracefully and survives" examples we have ~0 of under the raw reached-EoL definition.
  FLAT         - projects safe with little decline (< FLAT_DECLINE): a genuine flat negative example.
  PROBABLE_OOR - projects >= EoL but young / not near warranty and still declining: a supporting survivor.
  AT_RISK      - projects below EoL even after cleaning: a genuine end-of-life example (the real, rarer ones).
  EXCLUDED     - too few months / too short a span to fit a trend (data-thin).

"Good training data" = GRACEFUL + FLAT + PROBABLE_OOR. AT_RISK is kept too (real decline), just labelled.
Thresholds are module constants so the dashboard and any retrain share one definition.
"""
import numpy as np
import pandas as pd

GRACE_AGE_FRAC = 0.6     # "near warranty" = current age >= this fraction of the warranty term
MIN_DECLINE = 3.0        # GRACEFUL must show >= this much smoothed decline (else it's just FLAT)
FLAT_DECLINE = 3.0       # FLAT = smoothed decline below this
MIN_MONTHS = 4           # need this many SoH months ...
MIN_SPAN = 4             # ... over at least this many months of age span, to fit a trend

GOOD = ("GRACEFUL", "FLAT", "PROBABLE_OOR")


def theilsen(x, y):
    """Robust line y ~ intercept + slope*x via the median of pairwise slopes (resists cliff/stuck outliers)."""
    n = len(x)
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(n) for j in range(i + 1, n) if x[j] != x[i]]
    if not sl:
        return 0.0, float(np.median(y))
    s = float(np.median(sl))
    return s, float(np.median(y - s * x))


def curate(F, eol, warr_mo):
    """F: rows with vin, age_months, soh. eol: EoL %. warr_mo: warranty term in months.
    Returns one row per vehicle: bucket + the smoothed projection behind it."""
    rows = []
    for v, gg in F.groupby("vin"):
        gg = gg.sort_values("age_months")
        a = gg["age_months"].to_numpy(float)
        s = gg["soh"].to_numpy(float)
        n = len(a)
        span = float(a.max() - a.min()) if n else 0.0
        age = float(a.max()) if n else 0.0
        raw_aged = bool(n and s.min() <= eol)
        if n < MIN_MONTHS or span < MIN_SPAN:
            rows.append(dict(vin=v, bucket="EXCLUDED", proj=np.nan, sm_now=np.nan, decline=np.nan,
                             age=age, raw_aged=raw_aged))
            continue
        sl, ic = theilsen(np.sqrt(a), s)
        if sl > 0:                                       # degradation is monotone NON-increasing; a fitted
            sl, ic = 0.0, float(np.median(s))            # rise is just a flat/noisy vehicle -> hold it flat
        fit = lambda t: ic + sl * np.sqrt(t)             # slope<=0 & sqrt(age) increasing => never rises
        sm_now = float(fit(a.max()))                     # smoothed CURRENT SoH (anchor the projection here)
        proj = float(fit(warr_mo))                       # √t projection to the warranty deadline
        decline = float(fit(a.min()) - sm_now)           # smoothed decline observed so far
        near = age >= GRACE_AGE_FRAC * warr_mo
        oor = proj >= eol                                # out-of-risk: projected to survive warranty
        if oor and (raw_aged or near) and decline >= MIN_DECLINE:
            b = "GRACEFUL"
        elif oor and decline < FLAT_DECLINE:
            b = "FLAT"
        elif oor:
            b = "PROBABLE_OOR"
        else:
            b = "AT_RISK"
        rows.append(dict(vin=v, bucket=b, proj=round(proj, 1), sm_now=round(sm_now, 1),
                         decline=round(decline, 1), age=age, raw_aged=raw_aged))
    return pd.DataFrame(rows)


def summary(F, eol, warr_mo):
    R = curate(F, eol, warr_mo)
    vc = R.bucket.value_counts()
    return dict(table=R, counts={k: int(vc.get(k, 0)) for k in
                                 ("GRACEFUL", "FLAT", "PROBABLE_OOR", "AT_RISK", "EXCLUDED")},
                good=int(R.bucket.isin(GOOD).sum()), n=len(R))
