#!/usr/bin/env python3
"""Build a clean, verified OEM model-spec sheet -> OEM_Model_Specs.csv (repo root).

Replaces the manually-consolidated, messy multi-header `0. Tech Specs for OEMs.xlsx - Sheet1 (1).csv`.
Values were verified online (2026-06) against official OEM pages first, then reputable aggregators, by
three per-OEM research passes. Conventions:
  - One tidy row per OEM·model·variant.
  - `in_fleet` = "yes" if that variant appears in our modelled SoH cohort (per Vin_Model_Details.csv).
  - Unknown/unpublished fields are the literal string "unverified" (never fabricated).
  - `chemistry` is honest: Bajaj = LFP (aggregator-confirmed); Euler & Mahindra do NOT publish cell
    chemistry anywhere official -> "unverified". Do not seed chemistry into the model from these.
  - Battery-warranty km and years are split out (they differ from vehicle warranty for several models).
Numbers are best-available; see `confidence` and `notes`. Rerun: .venv/bin/python src/build_oem_specs.py
"""
import os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

COLS = ["oem", "model", "variant", "body_type", "pack_voltage_V", "battery_capacity_kWh",
        "nominal_capacity_Ah", "chemistry", "chemistry_confidence", "kerb_weight_kg", "gvw_kg",
        "charge_window_soc", "charger_kW", "charge_time_hr", "rated_range_km", "motor_power_kW",
        "motor_torque_Nm", "max_speed_kmph", "vehicle_warranty_km", "vehicle_warranty_yr",
        "battery_warranty_km", "battery_warranty_yr", "in_fleet", "confidence", "source_urls", "notes"]

U = "unverified"
R = [
    # ───────────────────────── EULER (fleet = 58V HiLoad cargo family + HiCity) ─────────────────────────
    dict(oem="Euler", model="HiLoad", variant="SR", body_type="cargo (DV/PV/FB, 120 CFT)",
         pack_voltage_V=58, battery_capacity_kWh=9.9, nominal_capacity_Ah="~171 (derived)",
         chemistry=U, chemistry_confidence="none (not disclosed)", kerb_weight_kg="744–764", gvw_kg=1495,
         charge_window_soc="10–80%", charger_kW=3.3, charge_time_hr="4.5–5.0", rated_range_km=90,
         motor_power_kW=11.7, motor_torque_Nm=88.55, max_speed_kmph=60, vehicle_warranty_km=U,
         vehicle_warranty_yr=U, battery_warranty_km=U, battery_warranty_yr=U, in_fleet="yes",
         confidence="Med", source_urls="eulermotors.com/en/hiload; evreporter.com",
         notes="Standard-range pack. 58V generation (our fleet), NOT the newer 67V refresh on aggregators. Ah derived (kWh/58V). Battery-warranty sources conflict -> unverified."),
    dict(oem="Euler", model="HiLoad", variant="TR", body_type="cargo (DV 120/DV 170/PV)",
         pack_voltage_V=58, battery_capacity_kWh=11.5, nominal_capacity_Ah="~198 (derived)",
         chemistry=U, chemistry_confidence="none (not disclosed)", kerb_weight_kg="738–758", gvw_kg=1495,
         charge_window_soc="10–80%", charger_kW=3.3, charge_time_hr="4.5–5.0", rated_range_km=120,
         motor_power_kW=11.7, motor_torque_Nm=88.55, max_speed_kmph=60, vehicle_warranty_km=U,
         vehicle_warranty_yr=U, battery_warranty_km=U, battery_warranty_yr=U, in_fleet="yes",
         confidence="Med", source_urls="eulermotors.com/en/hiload; evreporter.com",
         notes="Long-range pack. Largest single Euler cohort (DV 170/DV 120/PV)."),
    dict(oem="Euler", model="HiLoad", variant="XR", body_type="cargo (DV 170/PV, 120/170 CFT)",
         pack_voltage_V=58, battery_capacity_kWh=11.5, nominal_capacity_Ah="~198 (derived)",
         chemistry=U, chemistry_confidence="none (not disclosed)", kerb_weight_kg="738–758", gvw_kg=1495,
         charge_window_soc="10–80%", charger_kW=3.3, charge_time_hr="4.5–5.0", rated_range_km=120,
         motor_power_kW=11.7, motor_torque_Nm=88.55, max_speed_kmph=60, vehicle_warranty_km=U,
         vehicle_warranty_yr=U, battery_warranty_km=U, battery_warranty_yr=U, in_fleet="yes",
         confidence="Low", source_urls="eulermotors.com/en/hiload; Vin_Model_Details.csv",
         notes="'DVXR 13' in fleet: '13' ~ 13 kWh nameplate ≈ 11.5 kWh usable. Not separately spec'd officially."),
    dict(oem="Euler", model="HiCity", variant="TR/Maxx", body_type="passenger auto (D+3)",
         pack_voltage_V=58, battery_capacity_kWh=11.5, nominal_capacity_Ah="~200 (cited)",
         chemistry=U, chemistry_confidence="none (not disclosed)", kerb_weight_kg=U, gvw_kg=900,
         charge_window_soc="10–80%", charger_kW=3.3, charge_time_hr="4.5–5.0", rated_range_km=171,
         motor_power_kW=11.7, motor_torque_Nm=72, max_speed_kmph=60, vehicle_warranty_km=U,
         vehicle_warranty_yr=5, battery_warranty_km=U, battery_warranty_yr=5, in_fleet="yes",
         confidence="Med", source_urls="neo.eulermotors.com/en/hi-city; cmv360.com",
         notes="Passenger; lower torque (72 Nm) than cargo. 5-yr veh+battery stated on NEO page (no km)."),
    dict(oem="Euler", model="NEO HiRange", variant="Maxx", body_type="passenger auto (D+3)",
         pack_voltage_V="67 (sheet); aggregator shows 48", battery_capacity_kWh=13.44,
         nominal_capacity_Ah="~200", chemistry=U, chemistry_confidence="none (not disclosed)",
         kerb_weight_kg=U, gvw_kg=900, charge_window_soc="10–80%", charger_kW=3.3, charge_time_hr=3.25,
         rated_range_km="261 (cert)", motor_power_kW=9, motor_torque_Nm=65, max_speed_kmph=60,
         vehicle_warranty_km=125000, vehicle_warranty_yr=5, battery_warranty_km=150000,
         battery_warranty_yr=6, in_fleet="no", confidence="Med",
         source_urls="trucks.cardekho.com/.../neo-hirange; ackodrive.com",
         notes="Top passenger trim, highest range. Voltage conflict (67 vs 48). 6yr/1.5L-km battery only in old sheet -> verify."),
    # ───────────────────────── MAHINDRA (fleet = Treo Zor cargo) ─────────────────────────
    dict(oem="Mahindra", model="Treo Zor", variant="DV/PU/FB", body_type="cargo",
         pack_voltage_V=48, battery_capacity_kWh=7.37, nominal_capacity_Ah="~153.5 (derived)",
         chemistry=U, chemistry_confidence="none (official says only 'Li-ion')", kerb_weight_kg=417,
         gvw_kg=995, charge_window_soc=U, charger_kW=U, charge_time_hr=3.83, rated_range_km=80,
         motor_power_kW=8, motor_torque_Nm=42, max_speed_kmph=50, vehicle_warranty_km=80000,
         vehicle_warranty_yr=3, battery_warranty_km=150000, battery_warranty_yr=U, in_fleet="yes",
         confidence="High (specs) / Low (chem)",
         source_urls="mahindralastmilemobility.com/treo-zor-dv; trucks.cardekho.com",
         notes="OUR Mahindra cohort. One powertrain across DV/PU/FB. Battery km(150k) > vehicle km(80k); the 3-yr cap is vehicle-only."),
    dict(oem="Mahindra", model="Zor Grand", variant="DV / DV Plus", body_type="cargo (140/170 CFT)",
         pack_voltage_V=48, battery_capacity_kWh=10.24, nominal_capacity_Ah="~213.3 (derived)",
         chemistry=U, chemistry_confidence="none (official says only 'Li-ion')", kerb_weight_kg=U, gvw_kg=998,
         charge_window_soc=U, charger_kW=U, charge_time_hr=4.50, rated_range_km="115 real / 172 cert",
         motor_power_kW=12, motor_torque_Nm=50, max_speed_kmph=50, vehicle_warranty_km=120000,
         vehicle_warranty_yr=5, battery_warranty_km=120000, battery_warranty_yr=5, in_fleet="no",
         confidence="High (specs) / Low (chem)", source_urls="mahindralastmilemobility.com/zor-grand-dv",
         notes="Bigger cargo pack. Battery = vehicle warranty (5yr/120k)."),
    dict(oem="Mahindra", model="Treo Plus", variant="metal-body", body_type="passenger auto (D+3)",
         pack_voltage_V=48, battery_capacity_kWh=10.24, nominal_capacity_Ah="~213.3 (derived)",
         chemistry=U, chemistry_confidence="none (official says only 'Li-ion')", kerb_weight_kg=U, gvw_kg=U,
         charge_window_soc=U, charger_kW=U, charge_time_hr=4.50, rated_range_km="167 ARAI / 150 real",
         motor_power_kW=8, motor_torque_Nm=42, max_speed_kmph=55, vehicle_warranty_km=120000,
         vehicle_warranty_yr=5, battery_warranty_km=120000, battery_warranty_yr=5, in_fleet="no",
         confidence="High (specs) / Low (chem)",
         source_urls="mahindralastmilemobility.com/treo-plus; mahindra.com press release",
         notes="Dominant passenger model in the broader fleet (~10.5k VINs). Current metal-body Treo Plus = 10.24 kWh, NOT 7.37."),
    # ───────────────────────── BAJAJ (fleet = RE E-TEC 9.0 + Maxima XL Cargo) ─────────────────────────
    dict(oem="Bajaj", model="RE E-TEC 9.0", variant="(single)", body_type="passenger auto (D+3)",
         pack_voltage_V=U, battery_capacity_kWh=8.9, nominal_capacity_Ah=U, chemistry="LFP",
         chemistry_confidence="Med (aggregator-stated; official brochure image-only)", kerb_weight_kg=362,
         gvw_kg=708, charge_window_soc="to 80% then 100%", charger_kW=U, charge_time_hr="4.5 (≈3 to 80%)",
         rated_range_km=178, motor_power_kW=4.5, motor_torque_Nm=36, max_speed_kmph=45,
         vehicle_warranty_km=120000, vehicle_warranty_yr=5, battery_warranty_km=120000,
         battery_warranty_yr=5, in_fleet="yes", confidence="High (specs) / Med (chem)",
         source_urls="bajajauto.com RE E-TEC; trucks.cardekho.com",
         notes="OUR dominant Bajaj cohort (passenger). IP67 pack, 2-speed AMT, regen. Veh+battery share 5yr/1.2L-km."),
    dict(oem="Bajaj", model="Maxima XL Cargo", variant="E-TEC 12.0", body_type="cargo",
         pack_voltage_V=48, battery_capacity_kWh=11.8, nominal_capacity_Ah=U, chemistry="LFP",
         chemistry_confidence="Med", kerb_weight_kg=392, gvw_kg=900, charge_window_soc="to 80% then 100%",
         charger_kW=U, charge_time_hr="5.83 (≈4 to 80%)", rated_range_km=183, motor_power_kW=5.5,
         motor_torque_Nm=36, max_speed_kmph=40, vehicle_warranty_km=120000, vehicle_warranty_yr=5,
         battery_warranty_km=120000, battery_warranty_yr=5, in_fleet="yes", confidence="Med-High",
         source_urls="trucks.cardekho.com Maxima XL 12.0; alt-mobility.com",
         notes="Cargo Maxima (~500 kg payload). Fleet 'Maxima E-Tech 12.0/11.0' map here."),
    dict(oem="Bajaj", model="Maxima XL Cargo", variant="E-TEC 9.0", body_type="cargo",
         pack_voltage_V=U, battery_capacity_kWh=8.9, nominal_capacity_Ah=U, chemistry="LFP",
         chemistry_confidence="Med", kerb_weight_kg=U, gvw_kg=U, charge_window_soc="to 80% then 100%",
         charger_kW=U, charge_time_hr="~4.5", rated_range_km=149, motor_power_kW=4.5, motor_torque_Nm=36,
         max_speed_kmph=40, vehicle_warranty_km=120000, vehicle_warranty_yr=5, battery_warranty_km=120000,
         battery_warranty_yr=5, in_fleet="yes", confidence="Low-Med",
         source_urls="bajajauto.com Maxima cargo brochure",
         notes="Smaller-battery cargo Maxima; likely covers the plain 'Maxima XL Cargo' fleet label."),
    dict(oem="Bajaj", model="GoGo", variant="P7012", body_type="passenger auto (D+3)",
         pack_voltage_V=U, battery_capacity_kWh=12.1, nominal_capacity_Ah=U, chemistry="LFP",
         chemistry_confidence="Med", kerb_weight_kg=U, gvw_kg=U, charge_window_soc="to 80% then 100%",
         charger_kW=U, charge_time_hr="4.5 (<4 to 80%)", rated_range_km="259 (cert)", motor_power_kW=5.5,
         motor_torque_Nm=40, max_speed_kmph=50, vehicle_warranty_km=120000, vehicle_warranty_yr=5,
         battery_warranty_km=U, battery_warranty_yr=5, in_fleet="no", confidence="High (official)",
         source_urls="bajajauto.com/three-wheelers/gogo/gogo-p70",
         notes="Newer 12.1 kWh passenger auto; not in our SoH cohort yet. Battery km-cap not published."),
    dict(oem="Bajaj", model="WEGO", variant="C90 (fleet 'GoGo C9012')", body_type="cargo",
         pack_voltage_V=U, battery_capacity_kWh=12.1, nominal_capacity_Ah=U, chemistry="LFP",
         chemistry_confidence="Med", kerb_weight_kg=U, gvw_kg=U, charge_window_soc="to 80% then 100%",
         charger_kW=U, charge_time_hr="4.5", rated_range_km="207 (cert)", motor_power_kW=5.5,
         motor_torque_Nm=36, max_speed_kmph=40, vehicle_warranty_km=120000, vehicle_warranty_yr=5,
         battery_warranty_km=U, battery_warranty_yr=5, in_fleet="no", confidence="High (official)",
         source_urls="bajajauto.com/three-wheelers/wego/wego-c90",
         notes="Cargo sibling of GoGo P-series; charges off 16A socket. Not in our cohort."),
]

df = pd.DataFrame(R)[COLS]
out = "OEM_Model_Specs.csv"
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, {df.shape[1]} cols")
print(f"  in-fleet variants: {(df.in_fleet=='yes').sum()} | reference-only: {(df.in_fleet=='no').sum()}")
print(f"  chemistry verified: {(df.chemistry!='unverified').sum()}/{len(df)} (Bajaj LFP only)")
print("\nper OEM:")
print(df.groupby('oem').agg(rows=('model','size'), in_fleet=('in_fleet', lambda s:(s=='yes').sum())).to_string())
