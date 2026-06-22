# Column Usability for SoH Calculation & Degradation Forecasting

_Battery State-of-Health (SoH) feasibility audit across the telemetry sources in
`s3://oem-data-iot/battery-oem-data/parquet/`. Last updated 2026-06-17._

> **Scope note.** This doc rates columns for two purposes: (a) computing the **SoH target**, and
> (b) serving as **features for a degradation-forecasting model**. A column can be useless for (a)
> but valuable for (b) — e.g. `latitude`/`longitude` (climate proxy), `gearPosition` (driving
> intensity), `batteryTemp` (thermal stress). See the "Forecasting features" section and
> `PROCEDURE.md` §4 for the feature mapping.

## TL;DR

| Source | Has current? | Best SoH method | SoH quality |
|---|---|---|---|
| **intellicar** (battery-data) | ✅ `current` (real, signed, ~10 ms) | **Coulomb counting** (`Q=∫I·dt`) | **Best** — physically grounded |
| **mahindra** (OEM feed) | ❌ | Distance-per-SoC (range retention) | Proxy — confounded by season/driving |
| **euler** (OEM feed) | ❌ (voltage 100% null too) | Reported `batterySoh`, cleaned | Vendor BMS value, coarse |

**Key fact:** only the **intellicar** table carries pack `current`, so it is the *only* source
where true coulomb counting — the gold-standard SoH method — is possible. The per-OEM feeds
(euler, mahindra) lack current entirely and must fall back to proxies or a vendor-reported value.

---

## What SoH needs, and which columns supply it

| SoH method | Required signals | Where available |
|---|---|---|
| **Coulomb counting** `SoH = C_now/C_ref`, `C = ∫I·dt / (ΔSoC/100)` | `current`, `soc`, time | intellicar only |
| **Reported SoH** (trust the BMS) | `batterySoh` | euler only |
| **Distance-per-SoC** (range retention proxy) | `odometer`, `soc`, time | mahindra, intellicar, euler |
| **Energy-per-SoC** (coulomb analog) | *cumulative* energy `kWh`, `soc` | none usable (see `kwh` note) |

---

## Per-source column audit

### intellicar — `intellicar/battery-data/`
84 columns exist, but for **Mahindra rows only 9 are populated** (the rest are filled only for
other OEMs such as Piaggio). Files are dense (~1–3k rows each); sub-second sampling.

| Column | Non-null (Mahindra) | SoH role | Verdict |
|---|---|---|---|
| `current` | 100% (−203…+147 A) | **∫I·dt charge throughput** | ✅ **critical** — enables coulomb counting |
| `soc` | 100% (0–100) | ΔSoC for every method | ✅ critical |
| `eventAt` | 100% | time base for ∫dt | ✅ critical |
| `vin` | 100% | per-vehicle grouping | ✅ critical |
| `batteryVoltage` | 100% (0–57 V) | energy (kWh) cross-check, pack health | ✅ useful |
| `odometer` | 93% | distance-per-SoC cross-check | ✅ useful |
| `dte` (distance-to-empty) | 100% | range-retention cross-check | ⚠️ secondary |
| `make`, `model` | 100% | filter + capacity context (Treo/Zor Grand) | ⚠️ context |
| `requestUUID` | 100% (unique) | none | ❌ drop (overhead) |
| `time` | 100% | redundant with `eventAt` | ❌ drop |
| `engineSpeed`, `headlight` | 100% (constant 0) | none | ❌ drop |
| `batteryTemp`, `chargeCycle`, `controllerTemperature`, `motorTemperature`, `vehicleSpeed`, `tripDistance`, `evMileage`, all `*Fault*`, `gearState`, `driveMode`, `chargerStatus`, … (≈75 cols) | **0% for Mahindra** | — | ❌ empty for Mahindra |

> Note: `chargeCycle` (direct cycle-count aging) and `batteryTemp` ARE populated for other OEMs in
> intellicar — valuable if SoH work expands beyond Mahindra. `current` is a **reserved word** in
> S3 Select SQL → must be quoted: `s."current"`.

### mahindra — `mahindra/vehicle-data/` (OEM feed)
18 columns; ~9 useful. Two interleaved record types (~30% "status" rows carry state/temp/kwh;
~70% "GPS" rows). Tiny files (~5 rows each) → expensive to scan.

| Column | Non-null | SoH role | Verdict |
|---|---|---|---|
| `soc` | 100% | core (has out-of-range garbage to clip) | ✅ keep |
| `odometer` | 100% | distance-per-SoC (has `0` garbage) | ✅ keep |
| `eventAt`, `vin` | 100% | time + id | ✅ keep |
| `distanceToEmpty` | 100% | range-retention cross-check | ✅ keep |
| `state` | 30% | DRIVING/CHARGING/IDLE segmentation | ✅ keep |
| `batteryTemp` | 30% | temp effects (has −50 / 2001 outliers) | ⚠️ keep, clean |
| `kwh` | 25% | **NOT usable** — see note | ⚠️ weak |
| `vehicleModel` | 30% | capacity context | ⚠️ context |
| `requestUUID`, `latitude`, `longitude`, `lastConnected`, `gearPosition`, `licensePlate`, `valid`(const), `color`, `vehicleVariant` | — | none for SoH | ❌ drop |
| **`batterySoh`** | — | **does not exist in this feed** | — |

> **`kwh` is not integrable.** It is a *signed instantaneous* power flow (+charging / −driving),
> holds stale values when parked, and is not cumulative — so it cannot be integrated to charge or
> capacity. The energy-per-SoC method is therefore not viable here.

### euler — `euler/vehicle-data/` (OEM feed)
Has a vendor-reported `batterySoh`, but no current.

| Column | Non-null | SoH role | Verdict |
|---|---|---|---|
| `batterySoh` | populated | **reported SoH** (coarse: ~8 quantized values incl. garbage `0.0`) | ✅ primary (clean it) |
| `batterySoc` | 100% | distance-per-SoC; has large outliers | ✅ keep, clean |
| `odometer` | populated | distance-per-SoC | ✅ keep |
| `batteryTemperature` | populated | temp effects | ⚠️ keep |
| `batteryVoltage` | **0% (all NULL)** | — | ❌ unusable |
| `eventAt`, `vin` | 100% | time + id | ✅ keep |

---

## Beyond SoH: columns as forecasting features

For a degradation **forecasting** model, value columns by whether they capture a known aging driver,
even if they play no role in computing the SoH target:

| Aging driver | Feature | Source column(s) | Note |
|---|---|---|---|
| Cycling / Ah throughput | cumulative Ah, equiv. full cycles | intellicar `current` | strongest driver; only via current |
| C-rate stress | charge/discharge C-rate distribution | intellicar `current` | |
| Depth of discharge | per-discharge ΔSoC | `soc` | |
| SoC dwell | % time at high/low SoC | `soc` | calendar aging |
| Voltage stress | min/max V, time near cutoff | intellicar `batteryVoltage` | |
| **Thermal** | mean/max temp, time hot | mahindra `batteryTemp` | sparse (~30%) but high value |
| **Climate / ambient** | region/season temp proxy | mahindra `latitude`,`longitude` | reliably 100% |
| Usage intensity | km/day, drive vs idle | `odometer`,`gearPosition`,`state` | |
| Calendar age | months since first seen | `eventAt` | |
| Range fade | DTE at given SoC over time | `dte`, `distanceToEmpty` | |

Implication: the cohort download keeps columns that were "drop" for SoH-calc but matter for
forecasting — `latitude`, `longitude`, `gearPosition` (mahindra feed), and captures sparse
`batteryTemp`/`state` when present.

## Recommendations

1. **For the most credible SoH, use intellicar + coulomb counting.** It is the only source with
   `current`. Minimum column set: `vin, eventAt, soc, current` (+ `batteryVoltage`, `odometer`,
   `dte` for cross-checks).
2. **For the OEM feeds, use distance-per-SoC** (`odometer` + `soc`) as a range-retention proxy, and
   enforce a monotonic non-increasing envelope. Treat it as indicative, not calibrated — it is
   confounded by seasonal/driving conditions and tends to under-read degradation.
3. **Never rely on `kwh` (mahindra) for energy counting** — it is not cumulative.
4. **Always apply the monotonic non-increasing rule** to any SoH series (SoH must never increase),
   but smooth (rolling median) *before* the envelope so a single noisy month doesn't create a fake
   permanent cliff.
5. **Cross-validate:** where a vehicle appears in both intellicar and an OEM feed (≈96 Mahindra VINs
   do), compare coulomb-counting SoH against the distance-per-SoC proxy — agreement validates the
   cheaper proxy. (See `notebooks/01_soh_target/mahindra_soh_method_compare.ipynb`.)

## Related artifacts
- `import_intellicar.py` / `import_mahindra.py` / `import_dense.py` — extractors
- `notebooks/01_soh_target/mahindra_soh_coulomb.ipynb` — coulomb counting (intellicar)
- `notebooks/01_soh_target/mahindra_soh_distance_proxy.ipynb` — distance-per-SoC (mahindra feed)
- `notebooks/01_soh_target/mahindra_soh_method_compare.ipynb` — same-vehicle method comparison
