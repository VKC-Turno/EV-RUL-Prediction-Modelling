"""OEM registry — the single source of truth that parameterises every pipeline.

Each pipeline (pipelines/<oem>/pipeline.py) is a thin wrapper that reads its entry here and hands the
values to common.pipeline_factory.get_pipeline(). Onboarding a new OEM = add one entry (audit the feed
first to pick `soh_method`), no new pipeline code.

Mirrors the canonical registry in the research repo: src/config.py (SOH_METHOD / FLEET_WARRANTY) and
src/oem_train.py (CFG). Keep the two in sync when the science changes.
"""
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class OEMConfig:
    oem: str
    soh_method: str          # "coulomb" | "bms_capacity" | "reported"
    model_module: str        # forecaster family: "euler_model" | "model" | "bajaj_model"
    eol_pct: float           # end-of-life SoH threshold (warranty line)
    warr_years: int
    warr_km: int
    has_gate: bool           # is a physically-independent yardstick available -> run the acceptance gate?
    maturity: str            # "mature" | "young" | "placeholder" — drives monitor alarm suppression
    extraction: str          # short note on the extraction path (for docs / step sizing)
    notes: str = ""

    def as_params(self) -> dict:
        return asdict(self)


# ── the five fleets ──────────────────────────────────────────────────────────────────────
REGISTRY = {
    "euler": OEMConfig(
        oem="euler", soh_method="bms_capacity", model_module="euler_model",
        eol_pct=80.0, warr_years=3, warr_km=80_000, has_gate=True, maturity="mature",
        extraction="dense batch",
        notes="BMS remaining-capacity -> recovery-aware clean -> hybrid soh_target. "
              "Independent coulomb full-charge yardstick exists -> ACCEPTANCE GATE active."),
    "mahindra": OEMConfig(
        oem="mahindra", soh_method="coulomb", model_module="model",
        eol_pct=80.0, warr_years=3, warr_km=120_000, has_gate=False, maturity="mature",
        extraction="two-feed: native (~70M tiny files) + intellicar (signed current)",
        notes="coulomb (intellicar) for both-feed vehicles; ~98% native-only scored by the Bayesian "
              "behaviour model (age+km). Yardstick too sparse to gate; soh already isotonic-clean."),
    "bajaj": OEMConfig(
        oem="bajaj", soh_method="reported", model_module="bajaj_model",
        eol_pct=70.0, warr_years=5, warr_km=120_000, has_gate=False, maturity="young",
        extraction="dense native",
        notes="No current/voltage -> reported essBmsSohcEstPercValue (monthly median, non-increasing). "
              "Features from cycles/temp/efficiency. km-limit usually binds before 5y. odo in metres."),
    "piaggio": OEMConfig(
        oem="piaggio", soh_method="coulomb", model_module="model",
        eol_pct=80.0, warr_years=3, warr_km=100_000, has_gate=False, maturity="mature",
        extraction="intellicar (SoH via signed current) + native (features)",
        notes="308k tiny intellicar files -> compaction is the #1 cost lever. Native voltage 100% null; "
              "native supplies thermal/usage features + distance-per-SoC cross-check. Noisy low-util."),
    "montra": OEMConfig(
        oem="montra", soh_method="bms_capacity", model_module="model",
        eol_pct=80.0, warr_years=3, warr_km=100_000, has_gate=False, maturity="placeholder",
        extraction="day-sampled 10-vehicle POC",
        notes="current is UNSIGNED -> no coulomb; resCapacity -> BMS remaining-capacity SoH. New fleet: "
              "flat, 0 decliners -> placeholder model. Has real pack temperature."),
}


def get(oem: str) -> OEMConfig:
    key = oem.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown OEM '{oem}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def all_oems():
    return sorted(REGISTRY)
