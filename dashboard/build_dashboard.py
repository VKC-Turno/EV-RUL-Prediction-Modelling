#!/usr/bin/env python3
"""Build the SoH Dashboard (self-contained dark HTML, Plotly.js).

Two panels per the spec:
  FIRST LIFE  — calculated SoH over time + **model forecast** (NOT a polyfit) to the warranty
                horizon, with the 80% EoFL line. Mahindra uses the condition-aware XGBoost
                degradation model (coulomb-counted SoH); Euler uses reported BMS SoH (mean of
                daily min) + a degradation-trend forecast.
  SECOND LIFE — SoH vs cycle number from the fixed second-life cubic, with the 20% EoSL line.

Run:  .venv/bin/python dashboard/build_dashboard.py   ->  dashboard/index.html
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)            # repo root
sys.path.insert(0, "src")
import model, config
import xgboost as xgb

FEATS, STATE, STRESS = model.FEATS, model.STATE, model.STRESS
ms = lambda ts: int(pd.Timestamp(ts).timestamp() * 1000)
EOFL = 80.0
RISK_MARGIN = 2.0   # only AT-RISK when projected SoH is >2pp below EoFL (forecast tolerance);
                    # 80.0 >= proj >= 78.0 is WATCH (borderline), not at-risk.


def status_of(proj):
    if proj < EOFL - RISK_MARGIN:
        return "AT-RISK"
    if proj < EOFL:
        return "WATCH"
    return "OK"


def soh_at(obs, fc, warr_ms):
    """Interpolate the SoH (observed+forecast trajectory) at the warranty timestamp."""
    pts = obs + fc
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if not warr_ms:
        return ys[-1]
    return float(np.interp(warr_ms, xs, ys))


# ───────────────────────── Mahindra: coulomb SoH + XGBoost forecast ─────────────────────────
def free_run(g, mdl, months):
    g = g.sort_values("month"); last = g.iloc[-1]
    stress = g.iloc[-6:][STRESS].median().to_dict()
    st = {s: float(last[s]) for s in STATE}; soh = float(last["soh"]); out = []
    for _ in range(int(months)):
        isa, dfc = model._curv(st["age_months"], st["soh"])
        x = pd.DataFrame([{**{s: st[s] for s in STATE}, **stress,
                           "inv_sqrt_age": isa, "soh_deficit": dfc}])[FEATS].to_numpy()
        soh = soh - max(mdl.predict(x)[0], 0)
        st.update(soh=soh, age_months=st["age_months"] + 1,
                  cum_ah=st["cum_ah"] + stress.get("ah_throughput", 0))
        out.append(soh)
    return out


def build_mahindra():
    m = pd.read_parquet("data/mahindra/features/feature_table.parquet").sort_values(["vin", "month"])
    tr = model.build_transitions(m)
    xgbm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=8, verbosity=0).fit(
        tr[FEATS].to_numpy(), tr["loss"].to_numpy(), sample_weight=tr["w"].to_numpy())
    vmod = dict(pd.read_csv("data/manifests/mahindra_vin_model.csv").values)
    rdf = pd.read_csv("Mh_Regd_Date.csv"); rdf["reg"] = pd.to_datetime(rdf["vehicle_registration_date"], errors="coerce")
    REG = dict(zip(rdf["vin"], rdf["reg"]))

    # ── per-vehicle risk attribution: z-score recent operating stress vs the fleet, group into
    #    charging / driving / thermal / deep-discharge, name the dominant stressor ───────────
    rec = pd.DataFrame({vin: g.sort_values("month").iloc[-6:][STRESS].median()
                        for vin, g in m.groupby("vin")}).T
    met = pd.DataFrame({
        "charge_cur": rec["cur_chg_mean"], "high_soc": rec["frac_soc_high"],
        "dis_cur": rec["cur_dis_mean"].abs(), "peak_cur": rec["cur_abs_p95"],
        "km": rec["km_month"], "temp": rec["temp_max"], "low_soc": rec["frac_soc_low"]})
    Z = (met - met.mean()) / met.std(ddof=0).replace(0, 1)
    GROUPS = {"charging": ["charge_cur", "high_soc"], "driving": ["dis_cur", "peak_cur", "km"],
              "thermal": ["temp"], "deep_discharge": ["low_soc"]}
    GZ = pd.DataFrame({g: Z[c].mean(axis=1) for g, c in GROUPS.items()})
    RLAB = {"charging": "Aggressive charging (high charge current / high-SoC dwell)",
            "driving": "Hard driving (high discharge current / high mileage)",
            "thermal": "Thermal stress (high battery temperature)",
            "deep_discharge": "Deep discharging (frequent low SoC)"}

    def risk_reason(vin):
        gz = GZ.loc[vin]; top = gz.idxmax(); tz = float(gz.max())
        factors = [[g, round(float(gz[g]), 1)] for g in gz.sort_values(ascending=False).index if gz[g] > 0.5]
        if tz < 0.5:
            return "Calendar & cycle aging — no single operating stressor stands out", factors
        return RLAB[top], factors

    out = {}
    for vin, g in m.groupby("vin"):
        g = g.sort_values("month")
        last_age = float(g["age_months"].iloc[-1]); H = max(int(round(60 - last_age)), 1)
        fc = free_run(g, xgbm, H)
        fcm = [g["month"].iloc[-1] + pd.DateOffset(months=k) for k in range(1, H + 1)]
        mdln = vmod.get(vin, "?"); wyr = config.warranty_for("mahindra", mdln)[0]
        reg = REG.get(vin); has_reg = reg is not None and pd.notna(reg)
        # always have a warranty date: use registration, else reconstruct it from the age at first
        # telemetry so the warranty line draws and "proj @ warranty" is read at the warranty date.
        reg_eff = reg if has_reg else (g["month"].iloc[0] - pd.Timedelta(days=float(g["age_months"].iloc[0]) * 30.4375))
        warr = reg_eff + pd.DateOffset(years=int(wyr))
        obs_l = [[ms(t), round(float(s), 2)] for t, s in zip(g["month"], g["soh"])]
        fc_l = [[ms(t), round(float(s), 2)] for t, s in zip(fcm, fc)]
        warr_ms = ms(warr)
        proj_warr = round(soh_at(obs_l, fc_l, warr_ms), 1)
        rr, rfac = risk_reason(vin)
        out[vin] = dict(
            label=f"{vin[-6:]} · {mdln}", method="coulomb", model=mdln, wyr=wyr,
            obs=obs_l, fc=fc_l, warranty=warr_ms, reg_anchor=ms(reg_eff), reg_known=bool(has_reg),
            now=round(float(g["soh"].iloc[-1]), 1), proj_warr=proj_warr,
            status=status_of(proj_warr), at_risk=bool(proj_warr < EOFL - RISK_MARGIN),
            risk_reason=rr, risk_factors=rfac)
    # well-observed vehicles first, lowest CURRENT SoH first -> default is a genuine degrader,
    # not a short-history extrapolation artifact
    def key(v):
        o = out[v]
        return (0 if len(o["obs"]) >= 8 else 1, o["now"])
    return {v: out[v] for v in sorted(out, key=key)}


# ───────────────────────── Euler SoH: BMS remaining-capacity (validated method) ─────────────
def bms_soh_dense(vin):
    """SoH from BMS remaining-capacity for a densely-imported Euler vehicle, with the fixes from
    the adversarial cross-validation: sanitize garbage rows; full_cap = remCap/(SoC/100) computed
    only at HIGH SoC (>=95) to avoid the negative-intercept SoC bias; physical bounds; isotonic
    (not hard-cummin) monotone fit so a single low-SoC month can't ratchet SoH down permanently.
    Returns a month->SoH Series, or None if no dense parquet / insufficient data."""
    fp = Path(f"data/euler/dense/{vin}.parquet")
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    df["t"] = pd.to_datetime(df["t"]) if "t" in df.columns else pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
    for c in ("batterySoc", "batteryRemainingCapacity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df[(df["batterySoc"].between(95, 100)) & (df["batteryRemainingCapacity"].between(0, 500))].copy()
    if len(d) < 50:
        return None
    d["full_cap"] = d["batteryRemainingCapacity"] / (d["batterySoc"] / 100.0)
    _med = d["full_cap"].median()
    if not np.isfinite(_med) or _med < 40:                     # broken/zero remaining-capacity
        return None
    d = d[d["full_cap"].between(0.6 * _med, 1.4 * _med)]       # adaptive window — handles any pack size (Hiload ≠ 133 Ah)
    d["month"] = d["t"].dt.to_period("M").dt.to_timestamp()
    mon = d.groupby("month").agg(full_cap=("full_cap", "median"), n=("full_cap", "size")).reset_index()
    mon = mon[mon["n"] >= 15]                                   # months with enough high-SoC coverage
    if len(mon) < 6:
        return None
    nominal = float(mon["full_cap"].iloc[:6].quantile(0.90))
    x = np.arange(len(mon))
    try:
        from sklearn.isotonic import IsotonicRegression
        fit = IsotonicRegression(increasing=False).fit_transform(x, mon["full_cap"].to_numpy())
    except Exception:
        fit = mon["full_cap"].rolling(3, min_periods=1, center=True).median().cummin().to_numpy()
    soh = np.clip(100.0 * fit / nominal, None, 100.0)
    return pd.Series(soh, index=mon["month"].to_numpy())


# ───────────────────────── Euler build: BMS-capacity if dense, else reported ─────────────────
def build_euler():
    rdf = pd.read_csv("data/euler/Euler_Regd_Details.csv")
    rdf["reg"] = pd.to_datetime(rdf["regd_date"], format="%d/%m/%y", errors="coerce")
    REG = dict(zip(rdf["vin"], rdf["reg"]))
    REPORTED_NOTE = "Reported-SoH only — operating-stress attribution needs current/voltage (dense pipeline)"
    BMS_NOTE = "BMS remaining-capacity SoH (validated vs coulomb/reported)."
    MODEL_NOTE = "BMS-capacity SoH + condition-aware degradation model (2023+ dense cohort; preliminary, n=6)."

    # condition-aware Euler model, trained on the dense cohort feature table when present
    EFT = EMDL = None
    if Path("data/euler/features/feature_table.parquet").exists():
        try:
            import euler_model as em
            EFT = pd.read_parquet("data/euler/features/feature_table.parquet").sort_values(["vin", "month"])
            EMDL = em.train(em.build_transitions(EFT))
        except Exception:
            EFT = EMDL = None

    feed_vins = {Path(p).stem: p for p in glob.glob("data/euler/feed/*.parquet")}
    eft_vins = set(EFT["vin"]) if EFT is not None else set()
    out = {}
    for vin in sorted(set(feed_vins) | eft_vins):
        if vin in eft_vins:
            # cohort vehicle: validated BMS-capacity SoH + condition-aware model forecast
            import euler_model as em
            eg = EFT[EFT["vin"] == vin].sort_values("month")
            idx = pd.DatetimeIndex(eg["month"]); sm_values = eg["soh"].to_numpy()
            last_age = float(eg["age_months"].iloc[-1]); H = max(int(round(60 - last_age)), 1)
            ffc = np.array(em.free_run(eg, EMDL, H))
            fcm = [idx[-1] + pd.DateOffset(months=k) for k in range(1, H + 1)]
            method, mlabel, note = "bms", "BMS capacity + model", MODEL_NOTE
        else:
            p = feed_vins[vin]
            bms = bms_soh_dense(vin)                             # validated method when dense data exists
            if bms is not None and len(bms) >= 6:
                sm = bms; method, mlabel, note = "bms", "BMS capacity", BMS_NOTE
            else:
                df = pd.read_parquet(p)
                df["t"] = pd.to_datetime(df["t"]) if "t" in df.columns else pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
                df["batterySoh"] = pd.to_numeric(df["batterySoh"], errors="coerce")
                df["odo"] = pd.to_numeric(df.get("odometer"), errors="coerce")
                v = df[(df["batterySoh"] > 0) & (df["batterySoh"] <= 100)].copy()
                if v.empty:
                    continue
                v["day"] = v["t"].dt.floor("D")
                dmin = v.groupby("day")["batterySoh"].min()
                odo = v.groupby("day")["odo"].max()
                mser = dmin.groupby(dmin.index.to_period("M").to_timestamp()).mean().sort_index()
                modo = odo.groupby(odo.index.to_period("M").to_timestamp()).max()
                jump = (mser.diff() > 4) & (modo.reindex(mser.index).diff() > 4000)   # cut a battery replacement
                if jump.any():
                    mser = mser.iloc[:jump.values.argmax()]
                if len(mser) < 4:
                    continue
                sm = mser.rolling(3, min_periods=1, center=True).median()
                method, mlabel, note = "reported", "Mean of min_soh", REPORTED_NOTE
            idx = pd.DatetimeIndex(sm.index); sm_values = sm.values
            age = (idx - idx[0]).days / 365.25
            a, b = np.polyfit(np.sqrt(age), sm_values, 1)[::-1]          # sqrt-time fade trend
            last_age = float(age[-1]); horizon_yr = max(5.0 - last_age, 1.0 / 12)
            fage = np.arange(last_age + 1 / 12, last_age + horizon_yr + 1e-9, 1 / 12)
            ffc = a + b * np.sqrt(fage)
            fcm = [idx[-1] + pd.DateOffset(months=k) for k in range(1, len(fage) + 1)]

        reg = REG.get(vin); has_reg = reg is not None and pd.notna(reg)
        reg_eff = reg if has_reg else idx[0]
        wyr = config.warranty_for("euler", "")[0]
        warr = reg_eff + pd.DateOffset(years=int(wyr))
        obs_l = [[ms(t), round(float(s), 2)] for t, s in zip(idx, sm_values)]
        fc_l = [[ms(t), round(float(s), 2)] for t, s in zip(fcm, ffc)]
        warr_ms = ms(warr)
        proj_warr = round(soh_at(obs_l, fc_l, warr_ms), 1)
        out[vin] = dict(
            label=f"{vin[-6:]} · Euler ({mlabel})", method=method, model="Euler", wyr=wyr,
            obs=obs_l, fc=fc_l, warranty=warr_ms, reg_anchor=ms(reg_eff), reg_known=bool(has_reg),
            now=round(float(sm_values[-1]), 1), proj_warr=proj_warr,
            status=status_of(proj_warr), at_risk=bool(proj_warr < EOFL - RISK_MARGIN),
            risk_reason=note, risk_factors=[])
    return {v: out[v] for v in sorted(out, key=lambda v: (0 if len(out[v]["obs"]) >= 8 else 1, out[v]["now"]))}


# ───────────────────────── Second life: fixed cubic ────────────────────────────────────────
def build_second_life():
    A3, A2, A1, A0 = -9.11155e-11, 5.88935e-07, -0.00921386, 77.6217
    xs = list(range(0, 6700, 50))
    ys = [A3 * x ** 3 + A2 * x ** 2 + A1 * x + A0 for x in xs]
    eosl_cycle = next((x for x in range(0, 9000) if (A3 * x ** 3 + A2 * x ** 2 + A1 * x + A0) <= 20), None)
    return dict(coef=[A3, A2, A1, A0], xs=xs, ys=[round(y, 3) for y in ys],
                eosl=20, eosl_cycle=eosl_cycle, soh0=round(A0, 2))


def main():
    DATA = {"mahindra": build_mahindra(), "euler": build_euler()}
    SECOND = build_second_life()
    payload = json.dumps({"data": DATA, "second": SECOND})
    html = HTML_TEMPLATE.replace("/*__PAYLOAD__*/", payload)
    outp = Path("dashboard/index.html"); outp.write_text(html)
    nveh = {k: len(v) for k, v in DATA.items()}
    print(f"wrote {outp}  | vehicles: {nveh} | second-life EoSL @ ~{SECOND['eosl_cycle']} cycles")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>SoH Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{--bg:#0b0f17;--panel:#111726;--panel2:#0e1421;--ink:#e6edf6;--mut:#8aa0b6;--line:#1e2a3d;
        --teal:#37e0c8;--pink:#ff5d8f;--green:#2ec16b;--purple:#9b7bff;--red:#ff5b5b;--amber:#f2a93b;}
  *{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--ink);
    font-family:Inter,Segoe UI,Roboto,system-ui,sans-serif}
  .wrap{max-width:1320px;margin:0 auto;padding:22px 20px 40px}
  h1{font-size:20px;font-weight:650;margin:0 0 2px} .sub{color:var(--mut);font-size:13px;margin-bottom:16px}
  .controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
  select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:9px;
    padding:8px 11px;font-size:13px;outline:none} label{color:var(--mut);font-size:12px;margin-right:6px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px} @media(max-width:980px){.grid{grid-template-columns:1fr}}
  .panel{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
    border-radius:14px;padding:16px 16px 8px;box-shadow:0 6px 24px rgba(0,0,0,.25)}
  .ptag{font-size:11px;font-weight:750;letter-spacing:.14em} .ptag.first{color:var(--teal)} .ptag.second{color:var(--pink)}
  .ptitle{font-size:14px;font-weight:600;margin:6px 0 2px} .pnote{color:var(--mut);font-size:12px;margin:2px 0 8px}
  .chips{display:flex;gap:8px;margin:8px 0 4px;flex-wrap:wrap}
  .chip{font-size:11.5px;padding:5px 10px;border-radius:999px;border:1px solid var(--line);color:var(--mut);background:#0c1320}
  .chip.on{color:#04140f;background:var(--green);border-color:var(--green);font-weight:650}
  .chip.na{opacity:.55;text-decoration:line-through}
  .stat{display:flex;gap:18px;margin:6px 2px 2px;flex-wrap:wrap}
  .stat div{font-size:12px;color:var(--mut)} .stat b{color:var(--ink);font-size:15px;font-weight:650;display:block}
  .plot{height:330px}
  .pill{display:inline-block;font-size:11px;padding:2px 9px;border-radius:999px;margin-left:8px;font-weight:650}
  .pill.ok{background:rgba(46,193,107,.15);color:var(--green)} .pill.risk{background:rgba(255,91,91,.16);color:var(--red)}
</style></head>
<body><div class="wrap">
  <h1>State-of-Health Dashboard <span id="riskpill"></span></h1>
  <div class="sub">First-life degradation forecast &amp; second-life (BESS) cycle projection</div>
  <div class="controls">
    <div><label>OEM</label><select id="oem"></select></div>
    <div><label>Model</label><select id="model"></select></div>
    <div><label>Vehicle</label><select id="veh"></select></div>
  </div>
  <div class="grid">
    <div class="panel">
      <div class="ptag first">FIRST LIFE</div>
      <div class="ptitle">SoH over time — calculated + model forecast</div>
      <div class="pnote" id="firstnote"></div>
      <div class="chips" id="chips"></div>
      <div class="stat" id="firststat"></div>
      <div id="riskreason" class="pnote"></div>
      <div id="first" class="plot"></div>
    </div>
    <div class="panel">
      <div class="ptag second">SECOND LIFE (BESS)</div>
      <div class="ptitle">SoH vs cycle number — second-life polynomial</div>
      <div class="pnote">Repurposed pack (starts at first-life EoL). End-of-second-life at 20% SoH.</div>
      <div class="stat" id="secondstat"></div>
      <div id="second" class="plot"></div>
    </div>
  </div>
</div>
<script>
const PAYLOAD = /*__PAYLOAD__*/;
const D = PAYLOAD.data, S = PAYLOAD.second;
const AX = {gridcolor:'#1c2738',zerolinecolor:'#1c2738',color:'#8aa0b6',linecolor:'#27374e'};
const BASE = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#cdd9e8',size:11},
  margin:{l:46,r:14,t:10,b:36},showlegend:true,legend:{orientation:'h',y:1.12,x:0,font:{size:10}}};
const METHODS=[['coulomb','Coulomb counting'],['reported','Mean of min_soh'],['kalman','Kalman filter']];
const fmtDate=ms=>{const d=new Date(ms);return d.toLocaleDateString('en-GB',{month:'short',year:'2-digit'});};

function chips(active){
  return METHODS.map(([k,lbl])=>{
    const on = (k===active)||(active==='reported'&&k==='reported')||(active==='coulomb'&&k==='coulomb');
    const cls = on?'chip on':(k==='kalman'?'chip na':'chip na');
    return `<span class="${on?'chip on':'chip na'}">${lbl}${on?'':' · n/a'}</span>`;
  }).join('');
}

function drawFirst(v){
  const obsX=v.obs.map(p=>p[0]),obsY=v.obs.map(p=>p[1]);
  const fcX=v.fc.map(p=>p[0]),fcY=v.fc.map(p=>p[1]);
  const fcCol = v.at_risk?'#ff5b5b':'#2ec16b';
  const traces=[
    {x:obsX,y:obsY,mode:'markers+lines',name:'Calculated SoH',marker:{color:'#37e0c8',size:5},
     line:{color:'rgba(55,224,200,.55)',width:1.4}},
    {x:fcX,y:fcY,mode:'lines',name:'Model forecast',line:{color:fcCol,width:2.4,dash:'dash'}},
  ];
  const xall=obsX.concat(fcX), x0=Math.min(...xall), x1=Math.max(...xall, v.warranty||0);
  const shapes=[{type:'line',x0:x0,x1:x1,y0:80,y1:80,line:{color:'#f2a93b',width:1.4,dash:'dot'}}];
  const ann=[{x:x0,y:80,xanchor:'left',yanchor:'bottom',text:'80% EoFL',showarrow:false,font:{color:'#f2a93b',size:10}}];
  if(v.warranty){shapes.push({type:'line',x0:v.warranty,x1:v.warranty,y0:0,y1:100,line:{color:'#2ec16b',width:1.2,dash:'dashdot'}});
    ann.push({x:v.warranty,y:2,xanchor:'right',yanchor:'bottom',text:v.wyr+'-yr warranty',showarrow:false,font:{color:'#2ec16b',size:10},textangle:-90});}
  if(v.reg){shapes.push({type:'line',x0:v.reg,x1:v.reg,y0:0,y1:100,line:{color:'#5a6b82',width:1}});}
  const lay=Object.assign({},BASE,{shapes,annotations:ann,
    xaxis:Object.assign({},AX,{type:'date'}),
    yaxis:Object.assign({},AX,{title:'SoH (%)',range:[0,100]})});
  Plotly.react('first',traces,lay,{displayModeBar:false,responsive:true});
  // stats
  document.getElementById('firststat').innerHTML=
    `<div>current SoH<b>${v.now.toFixed(1)}%</b></div><div>projected @ warranty<b>${v.proj_warr.toFixed(1)}%</b></div>`+
    `<div>warranty<b>${v.wyr} yr</b></div><div>status<b style="color:${v.at_risk?'#ff5b5b':'#2ec16b'}">${v.at_risk?'AT-RISK':'OK'}</b></div>`;
  document.getElementById('chips').innerHTML=chips(v.method);
  document.getElementById('firstnote').innerHTML = v.method==='reported'
    ? 'Euler feed: reported BMS SoH (mean of daily min). Forecast = degradation trend.'
    : 'Coulomb-counted SoH. Forecast = condition-aware XGBoost degradation model.';
  const RLAB={charging:'charging',driving:'driving',thermal:'thermal',deep_discharge:'deep-discharge'};
  const rr=document.getElementById('riskreason');
  if(v.at_risk){rr.style.display='block';
    const fac=(v.risk_factors&&v.risk_factors.length)?'  <span style="color:#8aa0b6">['+v.risk_factors.map(f=>RLAB[f[0]]+' z'+f[1]).join(', ')+']</span>':'';
    rr.innerHTML='&#9888; <b style="color:#ff5b5b">Likely cause:</b> '+v.risk_reason+fac;
  } else {rr.style.display='none';}
  const rp=document.getElementById('riskpill');
  rp.className='pill '+(v.at_risk?'risk':'ok'); rp.textContent=v.at_risk?'AT-RISK':'WITHIN WARRANTY';
}

function drawSecond(){
  const t={x:S.xs,y:S.ys,mode:'lines',name:`EoSL (20% SoH) · ~${S.eosl_cycle} cycles`,line:{color:'#9b7bff',width:2.6}};
  const shapes=[{type:'line',x0:0,x1:S.xs[S.xs.length-1],y0:20,y1:20,line:{color:'#ff5b5b',width:1.4,dash:'dot'}}];
  const ann=[{x:0,y:20,xanchor:'left',yanchor:'bottom',text:'20% EoSL',showarrow:false,font:{color:'#ff5b5b',size:10}}];
  const traces=[t];
  if(S.eosl_cycle){traces.push({x:[S.eosl_cycle],y:[20],mode:'markers',name:'EoSL',marker:{color:'#ff5b5b',size:9,symbol:'x'}});}
  const lay=Object.assign({},BASE,{shapes,annotations:ann,
    xaxis:Object.assign({},AX,{title:'cycle number (from second-life entry)'}),
    yaxis:Object.assign({},AX,{title:'SoH (%)',range:[0,100]})});
  Plotly.react('second',traces,lay,{displayModeBar:false,responsive:true});
  document.getElementById('secondstat').innerHTML=
    `<div>entry SoH<b>${S.soh0}%</b></div><div>EoSL (20%)<b>~${S.eosl_cycle} cycles</b></div>`;
}

const oemSel=document.getElementById('oem'), modelSel=document.getElementById('model'), vehSel=document.getElementById('veh');
Object.keys(D).forEach(o=>{const op=document.createElement('option');op.value=o;
  op.textContent=o.charAt(0).toUpperCase()+o.slice(1)+` (${Object.keys(D[o]).length})`;oemSel.appendChild(op);});
function fillModels(){modelSel.innerHTML='';
  const models=['All',...Array.from(new Set(Object.values(D[oemSel.value]).map(v=>v.model))).sort()];
  models.forEach(mn=>{const op=document.createElement('option');op.value=mn;op.textContent=mn;modelSel.appendChild(op);});}
function fillVeh(){vehSel.innerHTML='';
  Object.entries(D[oemSel.value]).forEach(([vin,v])=>{
    if(modelSel.value!=='All' && v.model!==modelSel.value) return;
    const op=document.createElement('option');op.value=vin;op.textContent=v.label;vehSel.appendChild(op);});}
function update(){const v=D[oemSel.value][vehSel.value]; if(v) drawFirst(v);}
oemSel.onchange=()=>{fillModels();fillVeh();update();};
modelSel.onchange=()=>{fillVeh();update();};
vehSel.onchange=update;
fillModels(); fillVeh(); drawSecond(); update();
</script></body></html>"""


if __name__ == "__main__":
    main()
