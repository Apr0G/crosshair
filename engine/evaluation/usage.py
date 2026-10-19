"""
Usage log and weekly summary (P12.3).

    from engine.evaluation import usage
    usage.append_usage({"run_id": rid, "map_version": mv, "tier": "pro", "device": "gpu",
                        "timings": {"parse": 11.2, ..., "total": 48.0}, "warnings": 3,
                        "conservation_residual": 2e-9})
    print(usage.weekly_summary())          # counts, residual > 1e-3, timings over the ceiling

One JSON line per `engine analyze` run in `data/engine/eval/usage.jsonl`. `ts` (UTC,
ISO) is added when missing; the record is redacted (`engine.log.redact_obj`) and NaN
becomes null before writing. `warnings` may be a count or a list (stored as its count).

Weekly summary: runs in the last `days` (7) by `ts`; counts by tier and map version;
every run whose |conservation_residual| > 1e-3 and every run whose total time is over
the hard ceiling of its device (`bench.hard_ceiling_gpu` / `_cpu`; total = `timings
["total"]`, else the sum of the stage timings; unknown device is judged against the
GPU ceiling, the stricter one). Malformed lines are counted and skipped.

`engine analyze` calls `append_usage` after every successful run that writes to the DB
(`analyze.append_usage_record`, to `analyze.usage_log_path()`: next to the engine DB
when `CROSSHAIR_ENGINE_DB` is set, else the default path above); a failed append is
logged, never raised. Monitoring checklist: `docs/ENGINE_OPS.md`.

Filled by: P12.3.
"""
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

FILE_NAME        = "usage.jsonl"
RESIDUAL_MAX     = 1e-3
REQUIRED         = ("run_id", "map_version", "tier", "timings", "warnings", "conservation_residual")
DEFAULT_CEILINGS = {"gpu": 120.0, "cpu": 300.0}       # C3; config bench.hard_ceiling_* wins


def default_path() -> Path:
    from engine import paths
    return paths.EVAL_DIR / FILE_NAME


def _nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj


# ── append ────────────────────────────────────────────────────────────────────

def append_usage(record: dict, path: Path | None = None) -> dict:
    """Append one run record; returns what was written."""
    from engine.log import redact_obj
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"usage record missing {missing}")
    rec = dict(record)
    if isinstance(rec["warnings"], (list, tuple)):
        rec["warnings"] = len(rec["warnings"])
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rec = redact_obj(_nan_to_none(rec))
    path = Path(path) if path is not None else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
    return rec


# ── summary ───────────────────────────────────────────────────────────────────

def read_usage(path: Path) -> tuple[list[dict], int]:
    """(records, malformed line count)."""
    recs, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            recs.append(r)
        else:
            bad += 1
    return recs, bad


def _ts(r: dict) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(r.get("ts")))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def total_time(timings) -> float | None:
    if not isinstance(timings, dict) or not timings:
        return None
    if timings.get("total") is not None:
        return float(timings["total"])
    vals = [float(v) for v in timings.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(vals) if vals else None


def _ceilings(cfg) -> dict[str, float]:
    if cfg is None:
        return dict(DEFAULT_CEILINGS)
    return {"gpu": float(cfg.bench.hard_ceiling_gpu), "cpu": float(cfg.bench.hard_ceiling_cpu)}


def summarize(records: list[dict], *, days: int = 7, now: datetime | None = None, cfg=None) -> dict:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    week = [r for r in records if (_ts(r) is not None and since <= _ts(r) <= now)]
    ceil = _ceilings(cfg)
    by_tier: dict[str, int] = {}
    by_map: dict[str, int] = {}
    residual, slow = [], []
    for r in week:
        by_tier[str(r.get("tier"))] = by_tier.get(str(r.get("tier")), 0) + 1
        by_map[str(r.get("map_version"))] = by_map.get(str(r.get("map_version")), 0) + 1
        res = r.get("conservation_residual")
        if isinstance(res, (int, float)) and abs(res) > RESIDUAL_MAX:
            residual.append({"run_id": r.get("run_id"), "residual": float(res)})
        tot = total_time(r.get("timings"))
        dev = r.get("device") if r.get("device") in ceil else "gpu"
        if tot is not None and tot > ceil[dev]:
            slow.append({"run_id": r.get("run_id"), "total_s": tot, "device": dev, "ceiling_s": ceil[dev]})
    return {"since": since.isoformat(timespec="seconds"), "runs": len(week), "by_tier": by_tier,
            "by_map_version": by_map, "warnings": sum(int(w) for w in (r.get("warnings") for r in week)
                                         if isinstance(w, (int, float)) and not isinstance(w, bool)),
            "residual_over": residual, "over_ceiling": slow}


def weekly_summary(path: Path | None = None, *, days: int = 7, now: datetime | None = None, cfg=None) -> str:
    """Plain-text summary of the last `days` of usage."""
    path = Path(path) if path is not None else default_path()
    if not path.is_file():
        return f"no usage log yet ({path.name})"
    recs, bad = read_usage(path)
    s = summarize(recs, days=days, now=now, cfg=cfg)
    lines = [f"usage since {s['since']}: {s['runs']} runs, {s['warnings']} warnings"
             + (f", {bad} malformed lines skipped" if bad else ""),
             "by tier: " + (", ".join(f"{k} {v}" for k, v in sorted(s["by_tier"].items())) or "none"),
             "by map version: " + (", ".join(f"{k} {v}" for k, v in sorted(s["by_map_version"].items())) or "none"),
             f"residual > {RESIDUAL_MAX:g}: {len(s['residual_over'])}"]
    lines += [f"  {r['run_id']}  residual {r['residual']:.3g}" for r in s["residual_over"]]
    lines.append(f"over the time ceiling: {len(s['over_ceiling'])}")
    lines += [f"  {r['run_id']}  {r['total_s']:.1f} s > {r['ceiling_s']:.0f} s ({r['device']})"
              for r in s["over_ceiling"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Weekly usage summary (P12.3).")
    ap.add_argument("--path", type=Path, default=None, help="usage.jsonl (default data/engine/eval/usage.jsonl)")
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args(argv)
    path = a.path or default_path()
    if not path.is_file():
        print(f"usage: no log at {path.name} yet; `engine analyze` runs append to it once wired in")
        return 2
    from engine import config
    print(weekly_summary(path, days=a.days, cfg=config.load()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
