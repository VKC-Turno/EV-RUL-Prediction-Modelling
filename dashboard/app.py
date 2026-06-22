"""SoH Dashboard — Streamlit app.

Tabs:
  Forecast          — FIRST LIFE (calculated SoH + model forecast, registration→100 connector,
                      warranty line, 80% EoFL, secondary age-in-years axis) and SECOND LIFE / BESS.
  Customer Insights — per-vehicle battery-care score, behaviour vs fleet, life-extension levers.
  Fleet Behavior    — behaviour segments, behaviour↔degradation drivers, fleet findings.

Data: dashboard/build_dashboard.py (coulomb SoH + XGBoost forecast for Mahindra; BMS-capacity /
reported for Euler). Behaviour analytics: dashboard/insights.py.

Run:  .venv/bin/streamlit run dashboard/app.py
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))     # import sibling modules
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import build_dashboard as bd                                  # importing chdirs to repo root
import insights

st.set_page_config(page_title="SoH Dashboard", layout="wide", page_icon="🔋")

AX = dict(gridcolor="#1c2738", zerolinecolor="#1c2738", color="#8aa0b6", linecolor="#27374e")
METHODS = [("coulomb", "Coulomb counting"), ("bms", "BMS capacity"),
           ("reported", "Mean of min_soh"), ("kalman", "Kalman filter")]
RLAB = {"charging": "charging", "driving": "driving", "thermal": "thermal", "deep_discharge": "deep-discharge"}
STATUS_COL = {"OK": "#2ec16b", "WATCH": "#f2a93b", "AT-RISK": "#ff5b5b"}
NOTE = {"coulomb": "Coulomb-counted SoH. Forecast = condition-aware XGBoost degradation model.",
        "bms": "Euler: BMS remaining-capacity SoH (validated vs coulomb/reported; high-SoC, isotonic). Forecast = degradation trend.",
        "reported": "Euler feed: reported BMS SoH (mean of daily min). Forecast = degradation trend."}


def layout(**kw):
    base = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cdd9e8", size=12), margin=dict(l=52, r=16, t=12, b=40),
                legend=dict(orientation="h", y=1.13, x=0, font=dict(size=10)), height=400)
    base.update(kw)
    return base


@st.cache_data(show_spinner="Computing SoH series + forecasts…")
def load():
    data = {"mahindra": bd.build_mahindra(), "euler": bd.build_euler()}
    return data, bd.build_second_life()


@st.cache_data(show_spinner="Loading feature table…")
def load_features():
    return pd.read_parquet("data/mahindra/features/feature_table.parquet")


@st.cache_data(show_spinner="Analysing fleet behaviour…")
def fleet_insights():
    return insights.fleet_behavior(load_features())


def first_fig(v):
    obs = pd.DataFrame(v["obs"], columns=["ms", "soh"]); obs["t"] = pd.to_datetime(obs["ms"], unit="ms")
    fc = pd.DataFrame(v["fc"], columns=["ms", "soh"]); fc["t"] = pd.to_datetime(fc["ms"], unit="ms")
    col = STATUS_COL.get(v.get("status", "OK"), "#2ec16b")
    reg_t = pd.to_datetime(v.get("reg_anchor", v["obs"][0][0]), unit="ms")
    wt = pd.to_datetime(v["warranty"], unit="ms")
    fig = go.Figure()
    # connector: SoH = 100% at registration -> first observed reading (assumed pre-telemetry fade)
    fig.add_scatter(x=[reg_t, obs["t"].iloc[0]], y=[100.0, obs["soh"].iloc[0]], mode="lines+markers",
                    name="reg → first reading", showlegend=False, line=dict(color="#37e0c8", width=1.2, dash="dot"),
                    marker=dict(color="#37e0c8", size=7, symbol="triangle-down"))
    fig.add_scatter(x=obs["t"], y=obs["soh"], mode="markers+lines", name="Calculated SoH",
                    marker=dict(color="#37e0c8", size=5), line=dict(color="rgba(55,224,200,.5)", width=1.4))
    fig.add_scatter(x=fc["t"], y=fc["soh"], mode="lines", name="Model forecast", line=dict(color=col, width=2.6, dash="dash"))
    fig.add_hline(y=80, line=dict(color="#f2a93b", width=1.4, dash="dot"),
                  annotation_text="80% EoFL", annotation_position="top left", annotation_font_color="#f2a93b")
    fig.add_scatter(x=[wt, wt], y=[0, 100], mode="lines", name=f"{v['wyr']}-yr warranty",
                    line=dict(color="#2ec16b", width=2.2, dash="dashdot"))
    fig.add_annotation(x=wt, y=99, text=f"{v['wyr']}-yr warranty", showarrow=False, textangle=-90,
                       xanchor="right", yanchor="top", font=dict(color="#2ec16b", size=10))
    fig.add_scatter(x=[reg_t, reg_t], y=[0, 100], mode="lines", showlegend=False, line=dict(color="#5a6b82", width=1, dash="dot"))
    fig.add_annotation(x=reg_t, y=2, text="registration", showarrow=False, textangle=-90,
                       xanchor="left", yanchor="bottom", font=dict(color="#8aa0b6", size=9))
    # x-range + secondary 'age (years)' axis anchored at registration
    x0 = min(reg_t, obs["t"].iloc[0]); x1 = max(fc["t"].iloc[-1], wt) + pd.Timedelta(days=20)
    nyr = max(int(math.ceil((x1 - reg_t).days / 365.25)), 1)
    yt = [reg_t + pd.DateOffset(years=k) for k in range(0, nyr + 1)]
    fig.add_scatter(x=yt, y=[50] * len(yt), xaxis="x2", mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip")
    lay = layout(margin=dict(l=52, r=16, t=30, b=54),
                 legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, font=dict(size=10)))
    lay.update(xaxis=dict(range=[x0, x1], **AX),
               xaxis2=dict(range=[x0, x1], overlaying="x", side="top", tickmode="array",
                           tickvals=yt, ticktext=[str(k) for k in range(0, nyr + 1)],
                           title=dict(text="age (years)", font=dict(size=10, color="#8aa0b6")),
                           tickfont=dict(size=9, color="#8aa0b6"), showgrid=False),
               yaxis=dict(range=[0, 100], title="SoH (%)", **AX))
    fig.update_layout(**lay)
    return fig


def second_fig(S):
    fig = go.Figure()
    fig.add_scatter(x=S["xs"], y=S["ys"], mode="lines", line=dict(color="#9b7bff", width=2.6),
                    name=f"EoSL (20% SoH) · ~{S['eosl_cycle']} cycles")
    fig.add_hline(y=20, line=dict(color="#ff5b5b", width=1.4, dash="dot"),
                  annotation_text="20% EoSL", annotation_position="bottom left", annotation_font_color="#ff5b5b")
    if S["eosl_cycle"]:
        fig.add_scatter(x=[S["eosl_cycle"]], y=[20], mode="markers", name="EoSL",
                        marker=dict(color="#ff5b5b", size=10, symbol="x"))
    fig.update_xaxes(title="cycle number (from second-life entry)", **AX)
    fig.update_yaxes(range=[0, 100], title="SoH (%)", **AX)
    fig.update_layout(**layout())
    return fig


def chips(active):
    out = []
    for k, lbl in METHODS:
        if k == active:
            out.append(f'<span style="background:#2ec16b;color:#04140f;padding:4px 11px;border-radius:999px;'
                       f'font-size:12px;font-weight:600">{lbl}</span>')
        else:
            out.append(f'<span style="border:1px solid #1e2a3d;color:#6f8298;padding:4px 11px;border-radius:999px;'
                       f'font-size:12px;text-decoration:line-through">{lbl} · n/a</span>')
    return "&nbsp; ".join(out)


DATA, SECOND = load()

st.markdown("## 🔋 State-of-Health Dashboard")
st.caption("First-life degradation forecast, second-life (BESS) projection, and fleet/customer behaviour")

c1, c2, c3 = st.columns([1, 1.2, 1.8])
oem = c1.selectbox("OEM", list(DATA.keys()), format_func=lambda o: f"{o.capitalize()} ({len(DATA[o])})")
models = ["All"] + sorted(set(v["model"] for v in DATA[oem].values()))
model = c2.selectbox("Model", models)
vehs = [vin for vin, v in DATA[oem].items() if model == "All" or v["model"] == model]
vin = c3.selectbox("Vehicle", vehs, format_func=lambda x: DATA[oem][x]["label"])
v = DATA[oem][vin]

tab_main, tab_cust, tab_fleet = st.tabs(["Forecast", "Customer Insights", "Fleet Behavior"])

with tab_main:
    left, right = st.columns(2)
    with left:
        st.markdown("###### :green[**FIRST LIFE**] — SoH over time + model forecast")
        st.markdown(chips(v["method"]), unsafe_allow_html=True)
        reg_t = pd.to_datetime(v.get("reg_anchor", v["obs"][0][0]), unit="ms")
        st.caption(f"{NOTE.get(v['method'], '')}  ·  Registered "
                   f"{reg_t:%b %Y} {'' if v.get('reg_known') else '(estimated)'}")
        status = v.get("status", "OK")
        m = st.columns(4)
        m[0].metric("Current SoH", f"{v['now']}%")
        m[1].metric("Predicted @ warranty", f"{v['proj_warr']}%")
        m[2].metric("Warranty", f"{v['wyr']} yr")
        m[3].markdown(f"Status<br><span style='color:{STATUS_COL[status]};font-size:1.6rem;font-weight:700'>"
                      f"{status}</span>", unsafe_allow_html=True)
        if status == "AT-RISK":
            fac = ""
            if v.get("risk_factors"):
                fac = "  ·  " + ", ".join(f"{RLAB.get(f[0], f[0])} (z{f[1]})" for f in v["risk_factors"])
            st.warning(f"⚠ Likely cause: {v['risk_reason']}{fac}")
        elif status == "WATCH":
            st.info(f"ⓘ Marginal — predicted {v['proj_warr']}% at warranty, within 2 pp of the 80% EoFL "
                    f"(inside forecast tolerance). Worth monitoring, not yet at-risk.")
        st.plotly_chart(first_fig(v), use_container_width=True)
    with right:
        st.markdown("###### :red[**SECOND LIFE (BESS)**] — SoH vs cycle number")
        st.caption("Repurposed pack (starts at first-life EoL). End-of-second-life at 20% SoH.")
        sc = st.columns(2)
        sc[0].metric("Entry SoH", f"{SECOND['soh0']}%")
        sc[1].metric("EoSL (20%)", f"~{SECOND['eosl_cycle']} cycles")
        st.plotly_chart(second_fig(SECOND), use_container_width=True)

with tab_cust:
    if oem != "mahindra":
        st.info("Behavioural insights are available for Mahindra only (the OEM with the rich feature table).")
    else:
        ci = insights.customer_insights(load_features(), vin)
        if "error" in ci:
            st.info("No behavioural data for this vehicle.")
        else:
            cc = st.columns(4)
            cc[0].metric("Battery-care score", f"{ci['care_score']}/100", ci.get("care_grade", ""))
            cc[1].metric("Current SoH", f"{ci.get('current_soh', v['now'])}%")
            cc[2].metric("Observed fade", f"{ci.get('observed_loss_rate_pct_per_month', '—')} %/mo")
            cc[3].metric("Age", f"{round(ci.get('age_months', 0))} mo")
            st.progress(int(max(0, min(100, ci["care_score"]))))
            for s in ci.get("insights", []):
                st.write("• " + s)
            mdf = pd.DataFrame([{"metric": mm["name"], "pctile": mm["fleet_pctile"], "verdict": mm.get("verdict", "")}
                                for mm in ci.get("metrics", {}).values() if mm.get("fleet_pctile") is not None]).sort_values("pctile")
            if len(mdf):
                fig = go.Figure(go.Bar(x=mdf["pctile"], y=mdf["metric"], orientation="h", text=mdf["verdict"],
                                       marker_color=["#ff5b5b" if p >= 60 else "#2ec16b" for p in mdf["pctile"]]))
                fig.add_vline(x=50, line=dict(color="#8aa0b6", dash="dot"))
                fig.update_xaxes(title="fleet percentile (0 = gentlest, 100 = harshest)", range=[0, 100], **AX)
                fig.update_yaxes(**AX); fig.update_layout(**layout(height=320))
                st.plotly_chart(fig, use_container_width=True)
            if ci.get("levers"):
                st.markdown("**Top life-extension levers**")
                for l in ci["levers"]:
                    st.warning(f"**{l['name']}** ({l.get('fleet_pctile', 0):.0f}th pctile) — {l.get('action', '')}")

with tab_fleet:
    if oem != "mahindra":
        st.info("Fleet behaviour analysis is available for Mahindra only.")
    else:
        fb = fleet_insights()
        st.markdown("##### Behaviour segments")
        seg = pd.DataFrame(fb["segments"])
        scols = st.columns(len(seg))
        for col, (_, s) in zip(scols, seg.iterrows()):
            col.metric(s["label"], f"{s['size']} veh ({s['share_pct']}%)", f"{s['mean_loss_rate_pct_per_month']} %/mo fade")
            col.caption(s["description"])
        figs = go.Figure(go.Bar(x=seg["label"], y=seg["size"], marker_color=seg["mean_loss_rate_pct_per_month"],
                                marker_colorbar=dict(title="loss %/mo"), text=seg["mean_soh_now"]))
        figs.update_yaxes(title="vehicles", **AX); figs.update_xaxes(**AX); figs.update_layout(**layout(height=300))
        st.plotly_chart(figs, use_container_width=True)

        st.markdown("##### What predicts faster fade (behaviour ↔ degradation)")
        dd = pd.DataFrame([d for d in fb["degradation_drivers"] if d.get("spearman") is not None])
        if len(dd):
            figd = go.Figure(go.Bar(x=dd["spearman"], y=dd["name"], orientation="h",
                                    marker_color=["#ff5b5b" if x > 0 else "#37e0c8" for x in dd["spearman"]]))
            figd.update_xaxes(title="Spearman corr vs monthly SoH-loss", **AX); figd.update_yaxes(**AX)
            figd.update_layout(**layout(height=340))
            st.plotly_chart(figd, use_container_width=True)
        st.caption("Correlations are weak — degradation here is mostly calendar/cycle aging, not operating style.")

        st.markdown("##### Fleet findings")
        for f in fb.get("findings", []):
            st.write("• " + f)
        with st.expander("Behaviour distributions"):
            st.dataframe(pd.DataFrame(fb["distributions"]).T)
