"""Offline unit test for the self-healing catch-up planner (no Spark/Glue). Run:
    .venv/bin/python MLOps/glue/local_catchup_check.py
"""
import sys
from datetime import date

sys.path.insert(0, "MLOps/glue")
from catchup import plan_days

D = date
CASES = [
    # (last_processed, process_date, lookback, max_days) -> (expected_days, advance_to, clamped)
    ("first run",        None,          D(2025,1,5), 1, 14, [D(2025,1,5)],                         D(2025,1,5), False),
    ("normal daily",     D(2025,1,4),   D(2025,1,5), 1, 14, [D(2025,1,5)],                         D(2025,1,5), False),
    ("2 missed days",    D(2025,1,2),   D(2025,1,5), 1, 14, [D(2025,1,3),D(2025,1,4),D(2025,1,5)], D(2025,1,5), False),
    ("reprocess latest", D(2025,1,5),   D(2025,1,5), 1, 14, [D(2025,1,5)],                         None,        False),
    ("reprocess old",    D(2025,1,10),  D(2025,1,5), 1, 14, [D(2025,1,5)],                         None,        False),
    ("rolling lookback", D(2025,1,4),   D(2025,1,5), 3, 14, [D(2025,1,3),D(2025,1,4),D(2025,1,5)], D(2025,1,5), False),
]


def main():
    ok = True
    for name, wm, pd_, lb, mx, exp_days, exp_adv, exp_clamp in CASES:
        r = plan_days(wm, pd_, lb, mx)
        good = (r["days"] == exp_days and r["advance_to"] == exp_adv and r["clamped"] == exp_clamp)
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: days={[str(d) for d in r['days']]} "
              f"advance={r['advance_to']} clamped={r['clamped']}")

    # long-outage clamp: 40-day gap, max 14 -> exactly 14 days ending at target, clamped, oldest reported
    r = plan_days(D(2024,12,1), D(2025,1,9), 1, 14)
    clamp_ok = (len(r["days"]) == 14 and r["days"][-1] == D(2025,1,9)
                and r["days"][0] == D(2024,12,27) and r["clamped"] is True)
    ok &= clamp_ok
    print(f"  [{'ok ' if clamp_ok else 'FAIL'}] long-outage clamp: {len(r['days'])} days "
          f"{r['days'][0]}..{r['days'][-1]} clamped={r['clamped']}")

    print("\n" + ("PASS — catch-up planner correct" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
