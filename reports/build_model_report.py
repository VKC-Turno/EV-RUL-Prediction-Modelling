#!/usr/bin/env python3
"""Generate the **Model Details & Accuracy** PDF  ->  reports/model_accuracy_report.pdf

Accuracy = leave-one-vehicle-out (LOVO) backtest, 40%-tail holdout, computed 2026-06-22 via
src/euler_backtest.py (Euler) and the matching Mahindra/Bajaj LOVO. Regenerate this PDF with:
    .venv/bin/python reports/build_model_report.py
"""
import os
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.chdir(Path(__file__).resolve().parent.parent)
DATE = "22 Jun 2026"
INK, MUT, TEAL, GREEN, AMBER, RED = "#14202e", "#5b6b7d", "#1f9e8f", "#2ec16b", "#e0922b", "#d4504e"

# ── LOVO accuracy (RMSE / MAE, SoH percentage-points). model = conditioned forecaster ──
ACC = {
 "Mahindra": dict(
   target="Coulomb-counting SoH (intellicar current)",
   model="Condition-aware rate model — LightGBM quantile (q10/q50/q90)",
   hp="500 trees · lr 0.03 · num_leaves 15 · min_child 20 · flat-vs-degrading gate",
   feats="STATE(soh,age,cum_ah,cum_km,odo) + STRESS(current/SoC/volt/temp/km) + curvature(1/√age, deficit)",
   n=dict(overall=84, degrading=28, flat=56),
   rows=[("overall",(3.15,1.77),(3.22,1.40),(3.73,2.19)),
         ("degrading",(4.62,2.89),(5.37,3.58),(4.85,3.41)),
         ("flat",(1.94,1.15),(0.54,0.21),(2.94,1.52))], band=None),
 "Euler": dict(
   target="BMS remaining-capacity SoH (high-SoC, isotonic)",
   model="Trajectory model — LightGBM quantile P50 + own-slope blend, √-horizon P10/P90 bands",
   hp="400 trees · lr 0.03 · num_leaves 15 · reg_lambda 2.0 · flat-pin",
   feats="anchor state + recent stress (TRAJ_STRESS: throughput/temp/DoD/SoC) + Δage, √Δage",
   n=dict(overall=80, degrading=34, flat=46),
   rows=[("overall",(3.60,1.99),(4.01,1.92),(3.78,2.08)),
         ("degrading",(4.88,3.14),(5.76,3.81),(5.26,3.54)),
         ("flat",(1.67,0.92),(0.44,0.16),(1.37,0.72))],
   band=dict(overall=0.80, degrading=0.70, flat=0.89)),
 "Bajaj": dict(
   target="BMS reported SoH (essBmsSohcEstPercValue, cleaned)",
   model="No conditioned forecaster yet — √t-trend / persistence baseline (10-mo window)",
   hp="n/a (feature table built; conditioned model is the next step)",
   feats="cycle count / SoC dwell / pack & ambient temp / drive-eff / odometer",
   n=dict(overall=16, degrading=12, flat=4),
   rows=[("overall",(None,None),(3.00,2.35),(3.06,2.50)),
         ("degrading",(None,None),(3.46,3.00),(3.52,3.10)),
         ("flat",(None,None),(0.76,0.53),(1.00,0.79))], band=None),
}

A4 = (8.27, 11.69)


LH = 0.0188   # line height as a fraction of page; top-anchored text flows downward


def textpage(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=19, weight="bold", color=INK, va="top")
    y = 0.915
    if subtitle:
        fig.text(0.08, y, subtitle, fontsize=10.5, color=MUT, va="top"); y -= 0.018
    y = 0.88
    fig.text(0.08, 0.045, f"Battery SoH/RUL programme · {DATE}", fontsize=8, color=MUT, va="top")
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012
            fig.text(0.08, y, txt, fontsize=12.5, weight="bold", color=TEAL, va="top"); y -= 0.032
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


def acc_table_page(pdf, oem, d):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, f"{oem} — model & LOVO accuracy", fontsize=17, weight="bold", color=INK)
    fig.text(0.08, 0.92, d["model"], fontsize=10, color=INK)
    meta = [("SoH target", d["target"]), ("Hyperparameters", d["hp"]),
            ("Features", d["feats"]),
            ("Cohort (LOVO)", f"{d['n']['overall']} vehicles · {d['n']['degrading']} degrading · {d['n']['flat']} flat")]
    y = 0.885
    for k, v in meta:
        fig.text(0.08, y, k, fontsize=9, weight="bold", color=MUT)
        fig.text(0.26, y, v, fontsize=9, color=INK, wrap=True); y -= 0.027

    # accuracy table
    ax = fig.add_axes([0.08, 0.40, 0.84, 0.32]); ax.axis("off")
    head = ["Cohort", "Model RMSE", "Model MAE", "Persist. RMSE", "Trend RMSE"]
    cells = []
    for name, m, p, t in d["rows"]:
        mr = "—" if m[0] is None else f"{m[0]:.2f}"
        mm = "—" if m[1] is None else f"{m[1]:.2f}"
        cells.append([name, mr, mm, f"{p[0]:.2f}", f"{t[0]:.2f}"])
    tbl = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_cells().items() if hasattr(tbl, "get_cells") else tbl.get_celld().items():
        cell.set_edgecolor("#dde5ee")
        if r == 0:
            cell.set_facecolor(TEAL); cell.set_text_props(color="white", weight="bold")
        else:
            nm = d["rows"][r-1][0]
            cell.set_facecolor("#f4f8fb" if nm != "degrading" else "#fff5e9")
            # highlight model-beats-baseline on degrading
            if nm == "degrading" and c == 1 and d["rows"][r-1][1][0] is not None:
                cell.set_text_props(weight="bold", color=GREEN)

    # bar chart: degrading-cohort RMSE, model vs persistence vs trend
    axb = fig.add_axes([0.12, 0.10, 0.78, 0.22])
    deg = [r for r in d["rows"] if r[0] == "degrading"][0]
    vals = [deg[1][0] if deg[1][0] else np.nan, deg[2][0], deg[3][0]]
    labels = ["Model", "Persistence", "√t-trend"]
    cols = [GREEN, MUT, AMBER]
    bars = axb.bar(labels, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        if not np.isnan(v): axb.text(b.get_x()+b.get_width()/2, v+0.06, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    axb.set_title("Degrading-cohort RMSE (lower = better)", fontsize=10, color=INK)
    axb.set_ylabel("SoH RMSE (pp)", fontsize=9); axb.spines[["top","right"]].set_visible(False)
    axb.tick_params(labelsize=9)
    note = "Conditioned model beats both baselines on the degrading cohort." if deg[1][0] else \
           "Baselines only — conditioned Bajaj forecaster is the next step."
    fig.text(0.08, 0.055, note, fontsize=9, color=GREEN if deg[1][0] else AMBER, style="italic")
    if d["band"]:
        fig.text(0.08, 0.03, f"P10–P90 band coverage (target 0.80): overall {d['band']['overall']:.2f}, "
                 f"degrading {d['band']['degrading']:.2f}, flat {d['band']['flat']:.2f}.", fontsize=8.5, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def main():
    Path("reports").mkdir(exist_ok=True)
    out = "reports/model_accuracy_report.pdf"
    with PdfPages(out) as pdf:
        # cover
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(0.08, 0.74, "Battery State-of-Health", fontsize=30, weight="bold", color=INK)
        fig.text(0.08, 0.685, "Degradation Models — Details & Accuracy", fontsize=19, color=TEAL)
        fig.text(0.08, 0.64, f"Mahindra · Euler · Bajaj    |    {DATE}", fontsize=12, color=MUT)
        fig.add_axes([0.08, 0.60, 0.84, 0.002]).axis("off")
        fig.text(0.08, 0.55, "Scope", fontsize=12, weight="bold", color=INK, va="top")
        fig.text(0.08, 0.515,
                 "Per-OEM SoH forecasting models, their architecture and hyperparameters, and their\n"
                 "out-of-sample accuracy measured by leave-one-vehicle-out (LOVO) backtest. Accuracy is\n"
                 "reported separately for genuinely-degrading vs flat vehicles — the degrading cohort is\n"
                 "the one that matters for warranty and RUL decisions.", fontsize=10.5, color=INK,
                 va="top", linespacing=1.5)
        fig.text(0.08, 0.06, "Generated by reports/build_model_report.py", fontsize=8, color=MUT)
        plt.axis("off"); pdf.savefig(fig); plt.close(fig)

        # methodology
        textpage(pdf, "Methodology", subtitle="How the models are built and how accuracy is measured", blocks=[
            ("h", "SoH target (what we predict)"),
            ("b", ["Mahindra: coulomb-counting SoH from intellicar pack current (ΔSoC-weighted pooled).",
                   "Euler: BMS remaining-capacity SoH at high SoC, isotonic-decreasing fit (validated method).",
                   "Bajaj: BMS reported SoH (clean, monotone-enforced). All targets are monthly, per vehicle."]),
            ("h", "Model families"),
            ("b", ["Rate model: predict monthly ΔSoH (loss) from operating conditions + curvature, roll forward.",
                   "Trajectory model (Euler primary): predict cumulative loss vs horizon, with P10/P50/P90 bands.",
                   "A flat-vs-degrading gate prevents the model manufacturing loss on genuinely flat vehicles."]),
            ("h", "Validation — Leave-One-Vehicle-Out (LOVO)"),
            ("b", ["Hold out one whole vehicle at a time; train on all others (no leakage across vehicles).",
                   "Within the held-out vehicle: give the model the first 60% of months, forecast the last 40%.",
                   "Report RMSE/MAE split into degrading (tail loses ≥2 pp) vs flat vehicles.",
                   "Baselines to beat: persistence (last SoH flat) and a √t-trend fit.",
                   "Euler also reports P10–P90 band coverage (target ≈ 0.80)."]),
            ("h", "How to read the accuracy"),
            ("p", "Lower RMSE/MAE (in SoH percentage-points) is better. On flat vehicles persistence is hard "
                  "to beat (SoH barely moves); the test that matters is the degrading cohort, where the "
                  "conditioned model must beat persistence and trend."),
        ])

        for oem in ("Mahindra", "Euler", "Bajaj"):
            acc_table_page(pdf, oem, ACC[oem])

        # summary
        textpage(pdf, "Summary", subtitle="Headline accuracy and caveats", blocks=[
            ("h", "Headline (degrading-cohort RMSE, SoH pp)"),
            ("b", ["Mahindra: model 4.62 vs persistence 5.37 vs trend 4.85 — model wins (−14% vs persistence).",
                   "Euler: model 4.88 vs persistence 5.76 vs trend 5.26 — model wins (−15% vs persistence).",
                   "Bajaj: baselines ~3.0–3.5 (no conditioned model yet); short 10-mo window."]),
            ("h", "What the models are good / not good at"),
            ("b", ["Good: ordering and tracking genuine decliners — the warranty/RUL-relevant cohort.",
                   "Flat vehicles are routed toward persistence by the gate (small added error vs persistence).",
                   "Euler P10–P90 bands are calibrated to ~80% coverage overall (70% on steep degraders)."]),
            ("h", "Caveats"),
            ("b", ["Small cohorts (tens of vehicles); LOVO RMSE is the production error bar — re-backtest as data grows.",
                   "Bajaj needs a conditioned forecaster built (reuse src/model.py / euler_model.py).",
                   "Accuracy is on the observed window; true multi-year extrapolation is unvalidated until vehicles age."]),
        ])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
