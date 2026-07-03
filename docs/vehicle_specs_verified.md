# Turno Fleet — Verified Vehicle Specifications (Warranty-focused)

Re-verified 2026-07-03 against official OEM sites + reputable EV databases. **Warranty is the priority field** (drives warranty-risk calls). Every warranty number below is cited. Warranty is always **whichever comes first** (years OR km).

Key conventions:
- **VEHICLE warranty** and **BATTERY warranty** are captured separately — they frequently differ.
- Blank battery-year/km = **genuinely not published** by the OEM on an authoritative page (not omitted by us). Do not fill it in without a warranty booklet.
- Chemistry: **owner-confirmed LFP fleet-wide** for Euler/Mahindra/Bajaj (official pages disclose only "lithium-ion", so web `chemistry_confidence`=low; the LFP call is the owner's, not the web's).
- Machine-readable version: `docs/OEM_Model_Specs_verified.csv`.

---

## Mahindra (largest fleet OEM: ~11,600 VINs)

Mahindra **splits vehicle and battery warranty and they differ by model** — per-model terms matter.

| Model | Body | Fleet VINs | kWh | Vehicle warranty | Battery warranty | Conf | Notes |
|---|---|---|---|---|---|---|---|
| **Treo Zor** (DV/PU/FB) | cargo | 384 | 7.37 | **3 yr / 80,000 km** | **150,000 km (no yr stated)** | high | OUR Mahindra SoH cohort. Battery-km (150k) > vehicle-km (80k) → the 3-yr time cap is the binding vehicle limit. |
| **Zor Grand** (DV/DV Plus/PU) | cargo | 290 | 10.24 | **5 yr / 120,000 km** (page) *or* 3 yr / 80,000 km (press release) | **5 yr / 120,000 km** (page) *or* 5 yr / 150,000 km (press release) | medium | **CONFLICT between two official Mahindra sources** (see below). The real 5-yr warranty-risk cohort. |
| **Treo Plus** (SFT/HRT metal-body) | passenger | 10,532 | 10.24 | **5 yr / 120,000 km** | **5 yr / 120,000 km** (press release: covers whole vehicle incl. battery+motor) | high | Dominant broader-fleet model. 10.24 kWh, NOT 7.37. |
| **Treo** (base 7.4 kWh) | passenger | 116 | 7.4 | **5 yr / 120,000 km** | not separately stated | medium | Distinct from Treo PLUS. Battery split unclear. |
| **Udo** | passenger | 275 | 11.7 | **6 yr / 150,000 km** | not separately stated | medium | Newer model. Best-in-class vehicle term. |
| **Zeo / Zeo V2** | **4-wheeler SCV** | 15 | 21.3 (V2) | **3 yr / 125,000 km** | **7 yr / 150,000 km** | medium | NOT a 3-wheeler — different vehicle class; flag separately. |

**Zor Grand conflict (important):** the current product page (`zor-grand-dv`) states vehicle **and** battery **5 yr / 120,000 km**; the official launch press release states vehicle **3 yr / 80,000 km** + battery **5 yr / 150,000 km**. We keep 5yr/120k as the headline (matches the live page and the current sheet) but a battery booklet is needed to settle whether battery-km is 120k or 150k and whether the vehicle term is 3yr or 5yr. **Confidence medium, not high.**

Sources: mahindralastmilemobility.com/{treo-zor-dv, zor-grand-dv, treo-plus, treo, udo, mahindra-zeo}; mahindra.com press releases (Treo Plus metal body, Zor Grand launch, Zeo launch); trucks.cardekho.com; autocarpro.in.

---

## Euler (~2,177 VINs) — our 58V cargo cohort

**CRITICAL: warranty is generation-dependent.** Our fleet is the **58V HiLoad (2022-launch) generation**. The HiLoad spec pages on aggregators are now the **newer 67.2 V / 198 km refresh** and must NOT be applied to our cohort.

| Model | Variant | Body | Fleet VINs | kWh | Vehicle warranty | Battery warranty | Conf |
|---|---|---|---|---|---|---|---|
| **HiLoad** | SR (58V) | cargo | 59 | 9.9 | **3 yr / 80,000 km** | **3+2 yr performance (5 yr total); km not published** | medium |
| **HiLoad** | TR (58V) | cargo | 682 | 11.5 | **3 yr / 80,000 km** | 3+2 yr performance (5 yr total); km not published | medium |
| **HiLoad** | XR/DVXR13 (58V) | cargo | 720 | 11.5 | 3 yr / 80,000 km (inherited) | 3+2 yr performance | low |
| **HiCity** | TR/SR (58V) | passenger | 127 | 11.5 | **5 yr** (km not published) | **not separately termed** (onboard charger = 3 yr) | medium |
| **Neo HiRange** | Plus/Maxx | passenger | 69 | 11.56 | **5 yr / 125,000 km** | **6 yr / 150,000 km** | medium |
| Storm / Turbo | (newer gen) | cargo | ~40 | — | — | — | ref-only |

**Corrections vs config/old sheet:**
- HiLoad vehicle warranty is **3 yr / 80,000 km** — the **5 yr / 125,000 km** in `src/config.py` was borrowed from the passenger HiCity/HiRange models. Use 3yr/80k for at-risk (matches prior `euler-warranty-3yr` memory).
- HiCity: the prior "battery 3 yr" is actually the **onboard charger** (3 yr). The **battery term is genuinely unstated** on the official NEO page — do not assume 5yr battery.
- HiLoad battery-km is **never published as a standalone cap** → all HiLoad battery-km values are inferred (low). The newer aggregator generation lists HiLoad as vehicle+battery 3 yr / **100,000 km** (67.2V) — a different generation.

Sources: evreporter.com (HiLoad launch), evupdatemedia.com, rushlane.com, neo.eulermotors.com/en/hi-city, ackodrive.com (HiRange), trucks.cardekho.com/euler/hi-load & /neo-hirange.

---

## Bajaj (~1,858 VINs) — unified 5yr/120k, no battery/vehicle gap

Bajaj's whole current EV 3W lineup (GoGo, WEGO, Maxima XL Cargo, RE E-TEC) carries **one unified 5 yr / 120,000 km** warranty covering the **whole vehicle including the battery**. Battery = vehicle at the headline level.

| Model | Variant | Body | Fleet VINs | kWh | Warranty (veh = batt) | Conf |
|---|---|---|---|---|---|---|
| **GoGo P7012** | passenger | passenger | 707 | 12.1 | **5 yr / 120,000 km** | high |
| **GoGo P5009** | passenger | passenger | 409 | 9.2 | 5 yr / 120,000 km | high |
| **GoGo P5012** | passenger | passenger | 18 | 12.1 | 5 yr / 120,000 km | medium (mapping) |
| **RE E-TEC 9.0** | passenger | passenger | 445 | 8.9 | 5 yr / 120,000 km | high |
| **Maxima XL Cargo E-TEC 12.0** | cargo | cargo | 98 | 11.8 | 5 yr / 120,000 km | high |
| **Maxima XL Cargo E-TEC 9.0** | cargo | cargo | 66 | 8.9 | 5 yr / 120,000 km | medium (spec) |
| **WEGO/GoGo C9012** | cargo | cargo | 76 | 12.1 | 5 yr / 120,000 km | high |
| **WEGO P7012** | passenger | passenger | 39 | 12.1 | 5 yr / 120,000 km | high |

**Warranty-risk caveats:**
1. The 5-yr battery coverage is a **capacity-retention guarantee of ≥70% SoH** — a Bajaj warranty-risk trigger should be **SoH < 70% within 5 yr / 120k km**, not simple failure.
2. cardekho lists a narrower **3 yr / 60,000 km battery-defect** clause for GoGo P5009, distinct from the 5yr/120k capacity headline — aggregator-only, treated low-weight, but a documented ambiguity.

Sources: bajajauto.com/three-wheelers/{gogo/gogo-p70, gogo/gogo-p50, wego/wego-c90, wego/wego-p70, ev-maxima-xl-cargo}, bajajautocredit.com (RE E-TEC, Maxima 9.0), trucks.cardekho.com.

---

## Other in-fleet OEMs (quick warranty pass — outside the 3-OEM SoH cohort)

These are present in `Vin_Model_Details.csv` but were not in the deep 3-OEM findings. Verified in a single pass each — treat as medium/low pending official booklets.

| OEM | Model | Fleet VINs | kWh | Vehicle warranty | Battery warranty | Conf |
|---|---|---|---|---|---|---|
| Piaggio | Ape E-City FX Max | 1,292 | 8.0 | 5 yr / 200,000 km | not sep. confirmed | medium |
| Piaggio | Ape Electrik / E-Xtra FX | ~690 | — | not verified this pass | — | low |
| TVS | King EV Max | 905 | 9.2 | 6 yr / 150,000 km | likely same (not itemized) | medium |
| Montra | Super Auto ePV/EPL 2.0 | 484 | — | 3 yr / 100,000 km (+2yr/50k ext) | covered same term | medium |
| AltiGreen | neEV HD/HDX | 348 | 11.0 | 5 yr / 100,000 km *or* 3yr (conflict) | 150,000 km | low |
| OSM | Rage+ | 107 | 8.5–10.8 | 3 yr / 80,000 km | 3 yr / 80,000 km | medium |
| E-TRIO | Touro Max/MAX+ | 73 | — | 3 yr / 100,000 km | 3 yr / 100,000 km | medium |
| Greaves | Eltra City | 47 | — | 3 yr / 80,000 km | not sep. confirmed (XTRA trim adds 5yr/120k) | medium |

Sources: piaggio-cv.co.in, tvsmotor.com/three-wheelers/king-ev-max, cmv360.com/montra-electric, altigreen.com + cardekho, cardekho.com/osm/rage-plus + evgo.co.in, etrio.in/etrio-touro, cardekho.com/greaves/eltra-city.

---

## Still Unverified / needs an official warranty booklet

1. **Euler HiLoad (58V) battery-km cap** — never published as a standalone number. Battery term is "3+2 yr performance"; the km is inferred from the 80,000 km standard band (low confidence). **Get the Euler warranty booklet.**
2. **Euler HiCity battery-year term** — official page states vehicle 5 yr + charger 3 yr but is **silent on the battery**. Do not assume 5yr battery.
3. **Mahindra Zor Grand** — genuine conflict: product page (5yr/120k veh+batt) vs launch press release (veh 3yr/80k, batt 5yr/150k). **Settle battery-km (120k vs 150k) and vehicle term.**
4. **Mahindra battery-year for Treo Zor, base Treo, Treo Plus, Udo** — no official page states a battery-warranty YEAR (only vehicle year and, for Treo Zor/Udo, a battery km). Treo Plus battery-yr is inferred from the "whole vehicle incl. battery" press-release phrasing (medium).
5. **Bajaj GoGo P5009** — the 3yr/60,000 km battery-DEFECT sub-clause vs the 5yr/120k capacity warranty (aggregator-only) — confirm whether a shorter defect warranty coexists.
6. **AltiGreen neEV** — sources conflict 3yr vs 5yr vehicle term; battery 150,000 km unconfirmed on official page.
7. **Piaggio Ape Electrik / E-Xtra FX line, Montra 5yr/175k prior claim** — Piaggio cargo line not individually verified this pass; the prior memory's "Montra 5yr/175k" is NOT corroborated (current sources say 3yr/100k) — re-verify before using.
8. **Chemistry** — LFP is owner-confirmed, but every OEM's official page discloses only "lithium-ion" (except Bajaj RE E-TEC = "Lithium-ion Phosphate"). Web `chemistry_confidence` stays low for Euler/Mahindra.
