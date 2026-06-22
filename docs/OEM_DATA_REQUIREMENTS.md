# OEM Telemetry Requirements — for SoH prediction & vehicle tracking

What an OEM should ideally stream/log so we can compute State-of-Health accurately, forecast it
(RUL / warranty risk), and track the vehicle. Organized by priority tier, then by category.
Lessons baked in from the Mahindra (intellicar) + Euler feeds are flagged as **⚠ gotchas**.

Legend — what each field unlocks: **[CC]** coulomb-counting SoH · **[CAP]** BMS-capacity SoH ·
**[FEAT]** degradation feature for the model · **[TRK]** tracking · **[META]** static/anchor.

---

## TIER 1 — Mandatory (without these, SoH is guesswork)

### Core battery electrical (continuous, in-drive AND in-charge)
| Field | Unit | Why |
|---|---|---|
| **pack_current** (signed) | A | **[CC]** integrate ∫I·dt. The single most important signal. ⚠ document the **sign convention** (+ = charge or discharge) and keep it consistent; clip impossible spikes (we saw −22000 A sentinels). |
| **pack_voltage** | V | **[CC][FEAT]** voltage stress, OCV checks, power = V·I. |
| **state_of_charge (SoC)** | % | **[CC][CAP][FEAT]** ΔSoC denominator for capacity; SoC dwell/DoD features. ⚠ NULL when unknown, never 0/99999. |
| **timestamp (event time)** | ms, UTC | **[CC][TRK]** integration time base. ⚠ device clock must be monotonic; also send server-receive time. |

### Battery-reported capacity/health (lets us cross-validate, and is the fallback when current is noisy)
| Field | Unit | Why |
|---|---|---|
| **remaining_capacity** | Ah | **[CAP]** full_cap = remaining/(SoC/100) → SoH. ⚠ has a SoC-dependent bias near empty; most reliable at high SoC. |
| **full_charge_capacity** | Ah | **[CAP]** direct SoH = FCC / rated. The cleanest single number if the BMS estimates it well. |
| **reported_SoH** | % | **[CAP]** BMS's own estimate — coarse baseline/sanity check (often quantized & lagged; don't rely on alone). |

### Static metadata (one-time per vehicle/pack — anchors everything)
| Field | Why |
|---|---|
| **VIN / vehicle_id** | **[META][TRK]** join key. |
| **battery_pack_id / serial** | **[META]** ⚠ **detect pack swaps** — SoH resets on replacement; without this a swap looks like miraculous healing. |
| **rated/nominal pack capacity (Ah) & energy (kWh)** | **[META]** the 100% reference for SoH. |
| **registration / commissioning date** | **[META]** anchor SoH = 100% at age 0; gives true calendar age. |
| **cell chemistry (LFP/NMC/…), pack config (S×P), nominal & min/max voltage** | **[META]** fade-curve shape, plausibility bounds, OCV-SoC mapping. |
| **warranty terms (years, km, SoH% threshold)** | **[META]** warranty-risk evaluation. |

---

## TIER 2 — Strongly recommended (the degradation drivers the model needs)

### Thermal
| Field | Why |
|---|---|
| battery pack temperature — **min / max / avg** (per sensor ideally) | **[FEAT]** temperature is a primary aging driver (Arrhenius). |
| ambient / cabin temperature | **[FEAT]** climate exposure, thermal stress separation. |
| coolant temperature, thermal-management state (heat/cool active) | **[FEAT]** thermal load context. |
| controller & motor temperature | **[FEAT]** drivetrain stress (secondary). |

### Cell-level (imbalance is an early, direct health signal)
| Field | Why |
|---|---|
| **min & max cell voltage**, **cell voltage delta / imbalance** | **[FEAT]** rising imbalance = aging/weak cells; strong degradation predictor. |
| min/max cell temperature & their cell IDs | **[FEAT]** hot-spot detection. |
| balancing active flag | **[FEAT]** distinguishes balancing current from real charge. |
| per-cell voltages (if bandwidth allows) | **[FEAT]** best-in-class; weak-cell tracking. |

### Charging behaviour
| Field | Why |
|---|---|
| **charge state** (charging / discharging / idle / fault) | **[CC][FEAT]** ⚠ split sessions by this — don't integrate across a charge↔discharge transition. |
| charge type (AC / DC-fast / slow), charger current & voltage, charge power | **[FEAT]** fast-charge stress. |
| charge start/end SoC, time charging, CC/CV phase flag | **[FEAT]** ⚠ exclude CV/top-balancing (SoC≈100%) from capacity math. |
| **charge cycle count / equivalent full cycles** | **[FEAT]** cycle aging (vs calendar aging). |

### Usage / driving & cycling
| Field | Unit | Why |
|---|---|---|
| **odometer (cumulative)** | km | **[FEAT][TRK]** mileage aging, km-warranty, usage intensity. |
| speed | km/h | **[FEAT][TRK]** driving aggressiveness. |
| distance-to-empty / estimated range | km | **[FEAT]** range-anxiety / deep-discharge behaviour. |
| drive mode / gear, regen active & energy | — | **[FEAT]** duty cycle. |
| throttle position, motor RPM, motor torque | — | **[FEAT]** load/aggressiveness. |
| energy consumed & regenerated per trip (kWh) | kWh | **[FEAT]** efficiency (Wh/km), throughput. |
| trip / drive-cycle / charge-session ID | — | **[FEAT]** clean session boundaries. |

### Location / tracking
| Field | Why |
|---|---|
| **GPS latitude, longitude** (+ fix time) | **[TRK]** position, climate-zone, route. |
| altitude, heading, GPS speed | **[TRK]** terrain/grade load. |
| geofence/zone, network signal | **[TRK]** ops, data-gap diagnosis. |

---

## TIER 3 — Nice-to-have (advanced diagnostics & context)

| Field | Why |
|---|---|
| **internal resistance / DCIR** (BMS estimate) | **[FEAT]** ⭐ direct, physics-based degradation signal — resistance grows as the cell ages. Very valuable if available. |
| fault / DTC codes, BMS protection events (OV/UV/OT/OC) | **[FEAT]** abuse events that accelerate fade. |
| insulation resistance, contactor status | **[FEAT][TRK]** safety/health. |
| power/charge derating flags | **[FEAT]** thermal/SoC limiting. |
| humidity, road/grade, payload/load | **[FEAT]** environmental/load context. |
| OEM-precomputed lifetime aggregates: total Ah throughput, total kWh, equivalent full cycles, time-at-high-SoC, time-at-high-temp, peak/avg C-rate | **[FEAT]** saves us recomputing; great if trustworthy. |

---

## Data-quality & sampling requirements (as important as the fields themselves)

These are the difference between *usable* and *unusable* data — every one bit us on a real feed:

1. **Continuous high-frequency logging during BOTH drive and charge.** Coulomb counting needs an
   unbroken current+SoC series. Target **1–10 s** in active states, ≤60 s minimum. ⚠ A monthly
   snapshot or status-only rows cannot yield SoH.
2. **Session boundaries / state flags**, so a continuous span isn't silently a full charge+discharge
   cycle (this broke naive coulomb counting on Euler — gap-based sessions spanned whole cycles).
3. **Documented, consistent units & sign conventions** for current/power; specify discharge sign.
4. **Missing = NULL, never a sentinel.** ⚠ We saw SoC=79903, current=−22220, temp=1412 °C, capacity
   =101140 — these survive naive filters and corrupt the math. Validate/clip at source.
5. **Synchronized, monotonic timestamps** (device + server). No future timestamps, no resets.
6. **Stable pack_id** to detect battery replacements (SoH discontinuities).
7. **Don't pre-smooth or quantize** the raw signals coarsely — send raw; we'll smooth. Coarse/stuck
   reported-SoH (Euler) carried no usable fade signal.
8. **Per-cell / per-sensor granularity** where possible (min/max/avg at least), not just pack scalars.

---

## Minimum viable set (if forced to pick the absolute few)

For **gold-standard SoH** you need, continuously: `timestamp, pack_current, SoC, pack_voltage`
(+ static `rated_capacity`, `pack_id`, `registration_date`).
If current is unavailable/noisy, `remaining_capacity` (or `full_charge_capacity`) + `SoC` gives a
solid **BMS-capacity SoH**.
For **forecasting** add: `battery_temperature, cell_imbalance, charge_state, odometer, cycle_count`.
For **tracking** add: `GPS lat/long, speed, odometer`.
