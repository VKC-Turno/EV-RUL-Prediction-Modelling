"""Self-healing catch-up planner (pure Python — no Spark/Glue, so it is unit-testable offline).

Given the ingest watermark (last successfully-processed day) and this run's target day, decide which day
partitions to (re)ingest. A failed or skipped day is therefore picked up automatically by the next run —
no day can silently become a permanent gap — because every run catches up from the watermark. All writes
downstream are MERGE-idempotent, so reprocessing an already-done day is a no-op.

Semantics:
  * first run (watermark None): the trailing `lookback_days` window ending at process_date.
  * forward run (process_date > watermark): from watermark+1 (or the lookback window, whichever is earlier)
    up to process_date, clamped to at most `max_days` (so a long outage doesn't try to process months in
    one job — the dropped older days are reported, never silently skipped). Advance the watermark.
  * explicit reprocess (process_date <= watermark): just that one day; do NOT move the watermark.

`lookback_days` (default 1) is a rolling reprocess window for late-arriving data; raise it to reprocess the
last N days every run. `max_days` bounds catch-up after an outage.
"""
from datetime import timedelta


def plan_days(last_processed, process_date, lookback_days=1, max_days=14):
    """Return {days: [date...ascending], advance_to: date|None, clamped: bool}."""
    if last_processed is not None and process_date <= last_processed:
        return {"days": [process_date], "advance_to": None, "clamped": False}   # explicit reprocess
    start_lb = process_date - timedelta(days=lookback_days - 1)
    start = start_lb if last_processed is None else min(last_processed + timedelta(days=1), start_lb)
    floor = process_date - timedelta(days=max_days - 1)
    clamped = start < floor
    if clamped:
        start = floor
    n = (process_date - start).days + 1
    days = [start + timedelta(days=i) for i in range(n)]
    return {"days": days, "advance_to": process_date, "clamped": clamped}
