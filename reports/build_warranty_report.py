#!/usr/bin/env python3
"""Generate the **Warranty Predictions** PDF  ->  reports/warranty_predictions_report.pdf

Numbers computed 2026-06-22: Mahindra from data/mahindra/soh/warranty_risk.csv (free-run SoH to the
warranty boundary), Euler from dashboard/build_dashboard.build_euler (5-yr), Bajaj km from the feature
table. Regenerate:  .venv/bin/python reports/build_warranty_report.py
"""
import os
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.chdir(Path(__file__).resolve().parent.parent)
DATE = "22 Jun 2026"
INK, MUT, TEAL, GREEN, AMBER, RED = "#14202e", "#5b6b7d", "#1f9e8f", "#2ec16b", "#e0922b", "#d4504e"
A4 = (8.27, 11.69)
LH = 0.0188

# ── verified warranty numbers (see module docstring for provenance) ──
MH = [("Treo", 79, 3, 89, 9), ("Zor Grand", 14, 5, 21, 11)]   # (model, n, warr_yr, survive%, at_risk)
MH_FLEET, MH_HIGHCONF, MH_KM_BINDS, MH_N = 78, 81, 0, 95
EU = dict(n=86, ok=39, watch=4, atrisk=43, warr_yr=5, well_obs=80, well_surv=48)
WKM = 120000
# remaining-distance-to-EoL fleet medians (km), validated 2026-06 via src/rul_km.py + the forecasts
REM = [
    ("Euler",    1500, 19000, 30000, 37000, "full coverage"),
    ("Bajaj",    4160,  None, 27000, 139000, "all already <80%; 60% is a long, low-confidence extrapolation"),
    ("Mahindra", None,  None,  None,  None,  "n/a for most — odometer too sparse to derive a usage rate"),
]


def textpage(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=19, weight="bold", color=INK, va="top")
    y = 0.915
    if subtitle:
        fig.text(0.08, y, subtitle, fontsize=10.5, color=MUT, va="top")
    y = 0.88
    fig.text(0.08, 0.045, f"Battery SoH/RUL programme · {DATE}", fontsize=8, color=MUT, va="top")
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012; fig.text(0.08, y, txt, fontsize=12.5, weight="bold", color=TEAL, va="top"); y -= 0.032
        elif kind == "b":
            for line in txt:
                for i, seg in enumerate(textwrap.wrap(line, 96)):
                    if i == 0:
                        fig.text(0.095, y, "•", fontsize=10, color=TEAL, va="top")
                    fig.text(0.115, y, seg, fontsize=9.7, color=INK, va="top"); y -= LH
                y -= 0.004
            y -= 0.004
        elif kind == "p":
            for seg in textwrap.wrap(txt, 102):
                fig.text(0.08, y, seg, fontsize=9.7, color=INK, va="top"); y -= LH
            y -= 0.010
    plt.axis("off"); pdf.savefig(fig); plt.close(fig)


def style_table(ax, cells, head, hl_col=None, hl_rows=()):
    tbl = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    items = tbl.get_celld()
    for (r, c), cell in items.items():
        cell.set_edgecolor("#dde5ee")
        if r == 0:
            cell.set_facecolor(TEAL); cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#fff5e9" if (r-1) in hl_rows else "#f4f8fb")
    return tbl


def mahindra_page(pdf):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, "Mahindra — warranty survival by model/term", fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.915, "Free-run SoH to each vehicle's warranty boundary; survive = projected SoH ≥ 80% at expiry.",
             fontsize=10, color=MUT, va="top")
    ax = fig.add_axes([0.08, 0.66, 0.84, 0.18]); ax.axis("off")
    cells = [[m, str(n), f"{wy} yr", f"{s}%", str(ar)] for m, n, wy, s, ar in MH]
    style_table(ax, cells, ["Model", "Vehicles", "Warranty", "Survive ≥80%", "At-risk"], hl_rows=(1,))
    # bar: survival by model
    axb = fig.add_axes([0.12, 0.36, 0.78, 0.22])
    names = [f"{m}\n({wy} yr)" for m, n, wy, s, ar in MH]; vals = [s for *_, s, ar in MH]
    cols = [GREEN if v >= 80 else RED for v in vals]
    bars = axb.bar(names, vals, color=cols, width=0.5)
    for b, v in zip(bars, vals):
        axb.text(b.get_x()+b.get_width()/2, v+1.5, f"{v}%", ha="center", fontsize=11, weight="bold", color=INK)
    axb.axhline(80, color=AMBER, ls="--", lw=1)
    axb.text(0.5, 92, "working assumption: ~90% survive", ha="center", fontsize=8.5, color=AMBER)
    axb.set_ylim(0, 100); axb.set_ylabel("% surviving warranty", fontsize=9)
    axb.set_title("Warranty survival is term-dependent, not a single fleet number", fontsize=10, color=INK)
    axb.spines[["top", "right"]].set_visible(False); axb.tick_params(labelsize=9)
    fig.text(0.08, 0.30, "Key findings", fontsize=12, weight="bold", color=TEAL, va="top")
    y = 0.275
    for line in [
        f"Treo (3-yr term, n=79): ~89% survive — the working '~90% survive' assumption holds; short term is easy.",
        f"Zor Grand (5-yr term, n=14): only ~21% survive — the 5-year term is where the real liability sits.",
        f"Fleet-wide {MH_FLEET}% survive ({MH_HIGHCONF}% among the high-confidence ≥8-month vehicles).",
        f"km is NEVER the binding limit ({MH_KM_BINDS}/{MH_N}) — low utilisation means calendar/condition aging drives risk.",
        "Action: split warranty exposure by model/term; the Zor Grand 5-yr cohort is the one to provision for."]:
        for i, seg in enumerate(textwrap.wrap(line, 96)):
            if i == 0: fig.text(0.095, y, "•", fontsize=10, color=TEAL, va="top")
            fig.text(0.115, y, seg, fontsize=9.5, color=INK, va="top"); y -= LH
        y -= 0.004
    fig.text(0.08, 0.05, "Source: data/mahindra/soh/warranty_risk.csv (free-run XGBoost degradation model).",
             fontsize=8, color=MUT, va="top")
    pdf.savefig(fig); plt.close(fig)


def euler_page(pdf):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, "Euler — 5-year warranty exposure", fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.915, "BMS-capacity SoH forecast to the 5-yr boundary; this is an aged resale cohort.",
             fontsize=10, color=MUT, va="top")
    axb = fig.add_axes([0.14, 0.52, 0.72, 0.30])
    cats = ["OK\n(≥80%)", "WATCH\n(78–80%)", "AT-RISK\n(<78%)"]; vals = [EU["ok"], EU["watch"], EU["atrisk"]]
    cols = [GREEN, AMBER, RED]
    bars = axb.bar(cats, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        axb.text(b.get_x()+b.get_width()/2, v+0.6, str(v), ha="center", fontsize=12, weight="bold", color=INK)
    axb.set_ylabel("vehicles", fontsize=9); axb.set_title(f"Projected status at 5-yr warranty (n={EU['n']})", fontsize=10, color=INK)
    axb.spines[["top", "right"]].set_visible(False); axb.tick_params(labelsize=9)
    fig.text(0.08, 0.44, "Key findings", fontsize=12, weight="bold", color=TEAL, va="top")
    y = 0.415
    for line in [
        f"Only ~{round(EU['ok']/EU['n']*100)}% ({EU['ok']}/{EU['n']}) are projected to survive the 5-yr warranty above 80% SoH.",
        f"{EU['atrisk']} vehicles AT-RISK, {EU['watch']} WATCH — high exposure, but expected: this is an aged resale cohort.",
        f"Among well-observed vehicles (≥8 months), ~{EU['well_surv']}% survive.",
        "Caveat: 5-yr term + already-degraded fleet ⇒ the survival rate is far lower than a fresh fleet would show; "
        "read it as exposure on an old cohort, not a build-quality verdict.",
        "Action: prioritise these for resale/second-life routing before SoH erodes resale value further."]:
        for i, seg in enumerate(textwrap.wrap(line, 96)):
            if i == 0: fig.text(0.095, y, "•", fontsize=10, color=TEAL, va="top")
            fig.text(0.115, y, seg, fontsize=9.5, color=INK, va="top"); y -= LH
        y -= 0.004
    fig.text(0.08, 0.05, "Source: dashboard/build_dashboard.build_euler (BMS-capacity SoH + degradation forecast).",
             fontsize=8, color=MUT, va="top")
    pdf.savefig(fig); plt.close(fig)


def bajaj_page(pdf):
    b = pd.read_parquet("data/bajaj/features/feature_table.parquet").sort_values(["vin", "age_months"])
    rows = []
    for v, g in b.groupby("vin"):
        g = g.reset_index(drop=True); span = max(g.age_months.iloc[-1]-g.age_months.iloc[0], 1e-9)
        odo = g.odo_max.iloc[-1]; kmpm = (odo-g.odo_max.iloc[0])/span
        mo = 0.0 if odo >= WKM else ((WKM-odo)/kmpm if kmpm > 0 else np.inf)
        rows.append((v[-8:], odo, kmpm, mo, odo >= WKM))
    d = pd.DataFrame(rows, columns=["vin", "odo", "kmpm", "mo", "past"]).sort_values("mo")
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, "Bajaj — km-warranty runway & cycle RUL", fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.915, "High-mileage cargo fleet with a real BMS cycle counter. Here the km limit binds — the "
             "opposite of Mahindra.", fontsize=10, color=MUT, va="top")
    axb = fig.add_axes([0.20, 0.46, 0.70, 0.36])
    vals = [0 if p else m for m, p in zip(d.mo, d.past)]
    cols = [RED if p else AMBER for p in d.past]
    axb.barh(range(len(d)), vals, color=cols)
    axb.set_yticks(range(len(d))); axb.set_yticklabels(d.vin, fontsize=7.5)
    axb.set_xlabel("months until 120,000 km warranty limit", fontsize=9)
    axb.set_title("km-warranty runway (red = already past 120k km)", fontsize=10, color=INK)
    axb.spines[["top", "right"]].set_visible(False); axb.tick_params(labelsize=8)
    npast = int(d.past.sum())
    fig.text(0.08, 0.38, "Key findings", fontsize=12, weight="bold", color=TEAL, va="top")
    y = 0.355
    for line in [
        f"{npast}/{len(d)} vehicles have ALREADY exceeded the 120,000 km warranty limit.",
        f"Usage is heavy: {d.kmpm.min():.0f}–{d.kmpm.max():.0f} km/month; the rest reach 120k in a median "
        f"~{d[~d.past].mo.median():.1f} months.",
        "Cycle-based RUL: at current usage, ~3 months / ~138 more charge cycles to 70% SoH (median).",
        "For Bajaj the km limit is the binding constraint — opposite of the Mahindra fleet where time/SoH binds.",
        "Action: km-warranty provisioning matters here; flag the units already past 120k."]:
        for i, seg in enumerate(textwrap.wrap(line, 96)):
            if i == 0: fig.text(0.095, y, "•", fontsize=10, color=TEAL, va="top")
            fig.text(0.115, y, seg, fontsize=9.5, color=INK, va="top"); y -= LH
        y -= 0.004
    fig.text(0.08, 0.05, "Source: data/bajaj/features/feature_table.parquet (odometer + cycle counter). Warranty 3 yr / "
             "120k km (no spec sheet; inferred).", fontsize=8, color=MUT, va="top")
    pdf.savefig(fig); plt.close(fig)


def remaining_km_page(pdf):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, "Remaining distance to End-of-Life (RUL in km)", fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.915, "remaining km  =  recent km/month  ×  months until the SoH forecast reaches the EoL threshold.",
             fontsize=10, color=MUT, va="top")
    # table
    ax = fig.add_axes([0.08, 0.64, 0.84, 0.20]); ax.axis("off")
    fmt = lambda x: "n/a" if x is None else f"{x:,}"
    cells = [[o, fmt(km), fmt(t80), fmt(t70), fmt(t60)] for o, km, t80, t70, t60, _ in REM]
    style_table(ax, cells, ["OEM", "km/month", "to 80% (EoFL)", "to 70%", "to 60% (EoL)"], hl_rows=(2,))
    # grouped bar (Euler & Bajaj; Mahindra n/a)
    axb = fig.add_axes([0.12, 0.34, 0.78, 0.22])
    thr = ["to 80%", "to 70%", "to 60%"]; x = np.arange(3)
    eu = [19000, 30000, 37000]; bj = [np.nan, 27000, 139000]
    axb.bar(x-0.18, eu, width=0.34, color=TEAL, label="Euler")
    axb.bar(x+0.18, bj, width=0.34, color=AMBER, label="Bajaj")
    for xi, v in zip(x-0.18, eu): axb.text(xi, v+2000, f"{v//1000}k", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x+0.18, bj):
        if not np.isnan(v): axb.text(xi, v+2000, f"{v//1000}k", ha="center", fontsize=8, color=INK)
    axb.set_xticks(x); axb.set_xticklabels(thr); axb.set_ylabel("median remaining km", fontsize=9)
    axb.set_title("Median remaining km to each EoL threshold (Mahindra n/a — sparse odometer)", fontsize=9.5, color=INK)
    axb.legend(fontsize=9, frameon=False); axb.spines[["top", "right"]].set_visible(False); axb.tick_params(labelsize=8)
    fig.text(0.08, 0.28, "Why km, and the caveats", fontsize=12, weight="bold", color=TEAL, va="top")
    y = 0.255
    for line in [
        "Degradation is calendar/condition-driven, not mileage-driven — so the battery ages out over TIME and km "
        "accrue at the usage rate. A high-utilisation vehicle therefore delivers MORE total km before the same SoH.",
        "EoL thresholds: 80% = end of first life (second-life trigger); 70%; 60% = true end of life.",
        "Mahindra: remaining-km is n/a for most vehicles — the odometer is logged too sparsely to derive km/month.",
        "Long extrapolations (to 60%) are low-confidence, especially Bajaj (√t fit far past its ~10-month window).",
        "'Beyond horizon' (in the dashboard) means the vehicle does not reach the threshold within the modelled window."]:
        for i, seg in enumerate(textwrap.wrap(line, 98)):
            if i == 0: fig.text(0.095, y, "•", fontsize=10, color=TEAL, va="top")
            fig.text(0.115, y, seg, fontsize=9.3, color=INK, va="top"); y -= LH
        y -= 0.004
    fig.text(0.08, 0.05, "Source: src/rul_km.py (km/month × months-to-EoL from the per-OEM SoH forecast). Surfaced live in "
             "the dashboard.", fontsize=8, color=MUT, va="top")
    pdf.savefig(fig); plt.close(fig)


def main():
    Path("reports").mkdir(exist_ok=True)
    out = "reports/warranty_predictions_report.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(0.08, 0.74, "Warranty Predictions", fontsize=30, weight="bold", color=INK)
        fig.text(0.08, 0.685, "SoH-to-warranty survival & km-limit exposure", fontsize=18, color=TEAL)
        fig.text(0.08, 0.64, f"Mahindra · Euler · Bajaj    |    {DATE}", fontsize=12, color=MUT)
        fig.add_axes([0.08, 0.60, 0.84, 0.002]).axis("off")
        fig.text(0.08, 0.55, "Scope", fontsize=12, weight="bold", color=INK, va="top")
        fig.text(0.08, 0.515,
                 "For every vehicle we project SoH forward to its warranty boundary and flag whether it falls below\n"
                 "the 80% end-of-first-life threshold before expiry — and separately whether the km limit binds first.\n"
                 "Results are split by model/term, because warranty exposure is term-dependent, not a single number.",
                 fontsize=10.5, color=INK, va="top", linespacing=1.5)
        fig.text(0.08, 0.06, "Generated by reports/build_warranty_report.py", fontsize=8, color=MUT)
        plt.axis("off"); pdf.savefig(fig); plt.close(fig)

        textpage(pdf, "Methodology", subtitle="How warranty survival is predicted", blocks=[
            ("h", "Prediction"),
            ("b", ["Free-run the per-OEM degradation model from each vehicle's last observation to its warranty boundary.",
                   "Survive = projected SoH stays ≥ 80% (end-of-first-life) through the warranty term.",
                   "Status tiers: OK ≥80% · WATCH 78–80% (within forecast tolerance) · AT-RISK <78%."]),
            ("h", "Two limits, whichever binds first"),
            ("b", ["Time: warranty years from registration (Mahindra Treo 3 yr / Zor Grand 5 yr; Euler 5 yr; Bajaj 3 yr).",
                   "Distance: the km cap (120,000 km). We check which limit a vehicle hits first.",
                   "Mahindra is low-mileage (time binds); Bajaj is high-mileage (km binds)."]),
            ("h", "Confidence"),
            ("b", ["Vehicles with <8 months of history, or a warranty horizon > 2× current age, are low-confidence.",
                   "Two 'Ape E-Xtra FX' (a Piaggio model mislabelled into the Mahindra feed) are excluded from the split."]),
            ("h", "How to read it"),
            ("p", "Survival is reported per model/term. On short terms most vehicles pass; the long (5-yr) terms and the "
                  "aged cohorts are where exposure concentrates."),
        ])
        mahindra_page(pdf)
        euler_page(pdf)
        bajaj_page(pdf)
        remaining_km_page(pdf)
        textpage(pdf, "Summary", subtitle="Where the warranty exposure sits", blocks=[
            ("h", "Headline"),
            ("b", ["Mahindra Treo (3 yr): ~89% survive — assumption holds. Zor Grand (5 yr): ~21% survive — the liability.",
                   "Euler (5 yr, aged resale cohort): only ~45% projected to survive above 80%.",
                   "Bajaj (high-mileage): 2/16 already past the 120k km limit; km binds, not time."]),
            ("h", "The cross-cutting lesson"),
            ("b", ["Warranty exposure is term- and usage-dependent — never a single fleet percentage.",
                   "Low-mileage fleets (Mahindra) are bound by the time/SoH limit; km essentially never binds.",
                   "High-mileage fleets (Bajaj) are bound by the km limit; SoH is secondary."]),
            ("h", "Actions"),
            ("b", ["Provision warranty reserves by model/term — concentrate on Mahindra Zor Grand (5 yr) and the Euler cohort.",
                   "For Bajaj, track the km limit and flag units already past 120k.",
                   "Exclude/flag low-confidence vehicles (<8 mo history) before making firm calls."]),
        ])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
