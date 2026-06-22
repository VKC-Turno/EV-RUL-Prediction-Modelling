export const meta = {
  name: 'euler-soh-crossval',
  description: 'Adversarially cross-validate 3 SoH methods (coulomb / BMS-capacity / reported) for Euler 217086',
  phases: [{ title: 'Verify', detail: 'one adversarial reviewer per SoH method, recomputing from the parquet' },
           { title: 'Reconcile', detail: 'compare the three series, pick the dashboard series' }],
}

const VIN = 'MD9EMHDL23A217086'
const PARQUET = `data/euler/dense/${VIN}.parquet`
const CSV = `data/euler/soh/${VIN}_methods.csv`
const SUMMARY = `data/euler/soh/${VIN}_summary.json`

const VERDICT = {
  type: 'object', additionalProperties: false,
  properties: {
    method: { type: 'string' },
    sound: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' } },
    spot_checks: { type: 'array', items: { type: 'string' }, description: 'numbers you INDEPENDENTLY recomputed from the parquet' },
    recommended_fixes: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['method', 'sound', 'issues', 'spot_checks', 'confidence'],
}

phase('Verify')
const methods = [
  { key: 'coulomb', what: 'Coulomb counting: per-session capacity = |∫I·dt| / (|ΔSoC|/100), pooled ΔSoC-weighted per month, then SoH = 100*cap/baseline anchored at registration. Implemented in src/soh.py (coulomb_capacity_monthly + capacity_to_soh).' },
  { key: 'bms_capacity', what: 'BMS remaining-capacity: full_cap = batteryRemainingCapacity/(batterySoc/100); monthly median; SoH = 100*full_cap/nominal where nominal = p90 of first 6 months of full_cap; monotonic envelope.' },
  { key: 'reported', what: 'Reported BMS batterySoh, monthly median, monotonic envelope.' },
]
const verdicts = await parallel(methods.map(m => () => agent(
  `You are an adversarial battery-data reviewer. Independently VERIFY the '${m.key}' SoH method for Euler vehicle ${VIN} — assume it is wrong until your own recomputation says otherwise.

Method under review: ${m.what}

Data: dense parquet ${PARQUET} (columns: t [datetime, 60s logging], vin, batterySoc, batterySoh, batteryCurrent, batteryVoltage, batteryRemainingCapacity, batteryTemperature, vehicleMode, odometer). Pre-computed monthly series: ${CSV}. Summary: ${SUMMARY}.

Use Bash + .venv/bin/python to LOAD the parquet yourself and INDEPENDENTLY recompute spot checks — do NOT trust the CSV:
- coulomb: confirm batteryCurrent is in Amps and the method is sign-agnostic (|∫I·dt|). For 2 sample months, reconstruct continuous sessions (new session when gap>300s), integrate current (trapezoid, /3600 for Ah), divide by |ΔSoC|/100, and check the pooled monthly capacity is a plausible pack size (~100–135 Ah). Verify the ΔSoC>=2 and capacity-bounds filters and the registration anchoring.
- bms_capacity: confirm batteryRemainingCapacity is in Ah; recompute full_cap = remCap/(soc/100) for sample rows; sanity-check nominal (~130 Ah) and the monotonic step.
- reported: confirm batterySoh is coarse/discrete; check monthly median + monotonic.

Return: whether the method is SOUND, concrete issues, the numbers you recomputed (spot_checks with actual values), recommended fixes, and your confidence.`,
  { label: `verify:${m.key}`, phase: 'Verify', schema: VERDICT })))

phase('Reconcile')
const RECON = {
  type: 'object', additionalProperties: false,
  properties: {
    agreement_summary: { type: 'string', description: 'RMSE/correlation between the three series, recomputed' },
    best_method_for_dashboard: { type: 'string', enum: ['coulomb', 'bms_capacity', 'reported'] },
    rationale: { type: 'string' },
    final_series_notes: { type: 'string' },
    required_fixes: { type: 'array', items: { type: 'string' } },
    overall_confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['agreement_summary', 'best_method_for_dashboard', 'rationale', 'overall_confidence'],
}
const recon = await agent(
  `Reconcile the three SoH methods for Euler ${VIN}.
Reviewer verdicts: ${JSON.stringify(verdicts)}

Read ${SUMMARY} and ${CSV}. Independently load the CSV with pandas and compute month-aligned agreement (RMSE and correlation) for coulomb-vs-bms_capacity and coulomb-vs-reported.
Decide which series should drive the dashboard's First Life panel for this vehicle: prefer **coulomb** if it agrees with BMS-capacity within a few pp and is physically plausible and monotone; otherwise justify an alternative. State the cross-method agreement numbers, the choice, any fixes required before integration, and overall confidence.`,
  { label: 'reconcile', phase: 'Reconcile', schema: RECON })

return { verdicts, recon }
