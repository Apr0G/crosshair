"""
Robustness checks (P11.3): out-of-map gap, patch drift, map-version support.

    from engine.evaluation import robustness
    robustness.out_of_map_gap(metrics_default, metrics_holdout)   # {"gap": 0.07, "warning": "..."}
    robustness.patch_drift(newest_month_reliability)              # warning string or None
    robustness.map_version_support(conn, "de_mirage@2026-05")     # warning string or None

- `out_of_map_gap`: both arguments are metric dicts as `engine.model.calibrate.metrics`
  returns them (`logloss`, `brier`, `auc`, `ece`), the default model and the model
  trained with that map held out, both evaluated on the held-out map's test rounds
  (P11.3.1). The gap is reported, never failed: it documents that a new map needs data.
- `patch_drift`: calibration on the newest month only (P11.3.2). Takes a dict with
  `ece`, or the reliability bins of `calibrate.reliability` (ECE recomputed as
  Σ n_b / N · |mean_pred_b − observed_b|). ECE > 0.03 warns "model predates recent
  patches".
- `map_version_support`: counts non-quarantined `matches` rows of the map version in
  the engine DB. That is every registered match of the version, an upper bound on the
  training matches (the split file decides which are train); fewer than `min_matches`
  (100) warns.

P11.3.3 (FACEIT demo analysed as pro, 64-tick refusal) lives in the CLI and is tested
by `test/engine/test_cli_guards.py`.

Filled by: P11.3.1, P11.3.2 (checks; the training runs they compare need data).
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

PATCH_ECE_MAX     = 0.03          # P11.3.2
MIN_MAP_MATCHES   = 100           # P11.3.1
PATCH_WARNING     = "model predates recent patches"


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# ── P11.3.1 out-of-map ────────────────────────────────────────────────────────

def out_of_map_gap(metrics_default: dict, metrics_holdout: dict) -> dict:
    """Log-loss / Brier gap (holdout − default) on the held-out map; a warning text
    when the held-out model is worse. Missing metrics give None, never 0."""
    out = {}
    for k in ("logloss", "brier", "ece"):
        a, b = _num((metrics_default or {}).get(k)), _num((metrics_holdout or {}).get(k))
        out[f"{k}_default"] = a
        out[f"{k}_holdout"] = b
        out[f"{k}_gap"] = None if a is None or b is None else b - a
    gap = out["logloss_gap"]
    out["gap"] = gap
    out["warning"] = (None if gap is None or gap <= 0 else
                      f"a map unseen in training costs {gap:+.4f} log loss; the engine needs data on a "
                      f"new map version before it trusts itself there")
    return out


# ── P11.3.2 patch drift ───────────────────────────────────────────────────────

def ece_from_reliability(bins: list[dict]) -> float | None:
    """Σ n_b / N · |mean_pred_b − observed_b| over non-empty bins; None when empty."""
    n_total = sum(int(b.get("n") or 0) for b in bins)
    if n_total == 0:
        return None
    return float(sum((b["n"] / n_total) * abs(b["mean_pred"] - b["observed"])
                     for b in bins if b.get("n") and b.get("mean_pred") is not None
                     and b.get("observed") is not None))


def patch_drift(reliability_newest_month, ece_max: float = PATCH_ECE_MAX) -> str | None:
    """Warning when the newest month's ECE exceeds `ece_max`; None otherwise or unknown."""
    if isinstance(reliability_newest_month, dict):
        if "ece" in reliability_newest_month:
            e = _num(reliability_newest_month["ece"])
        else:
            e = ece_from_reliability(reliability_newest_month.get("reliability") or [])
    else:
        e = ece_from_reliability(list(reliability_newest_month or []))
    if e is None or e <= ece_max:
        return None
    return f"{PATCH_WARNING} (newest-month ECE {e:.4f} > {ece_max})"


# ── P11.3.1 map-version support ───────────────────────────────────────────────

def map_version_count(db_conn: sqlite3.Connection, map_version: str) -> int:
    row = db_conn.execute(
        "SELECT COUNT(*) FROM matches WHERE map_version = ? AND (status IS NULL OR status != 'quarantined')",
        (map_version,)).fetchone()
    return int(row[0])


def map_version_support(db_conn: sqlite3.Connection, map_version: str | None,
                        min_matches: int = MIN_MAP_MATCHES) -> str | None:
    """Warning when the map version has fewer than `min_matches` registered matches."""
    if map_version is None:
        return "map version unknown; the engine cannot say how much data backs this map"
    n = map_version_count(db_conn, map_version)
    if n >= min_matches:
        return None
    return f"map version {map_version} has {n} matches (< {min_matches}); results on it are less reliable"


# ── command line ──────────────────────────────────────────────────────────────

def _load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Robustness checks (P11.3).")
    ap.add_argument("--metrics-default", type=Path, default=None, help="JSON metrics of the default model")
    ap.add_argument("--metrics-holdout", type=Path, default=None, help="JSON metrics of the map-held-out model")
    ap.add_argument("--reliability", type=Path, default=None, help="JSON: newest-month {ece} or reliability bins")
    ap.add_argument("--map-version", default=None, help="check DB support for this map version")
    ap.add_argument("--min-matches", type=int, default=MIN_MAP_MATCHES)
    a = ap.parse_args(argv)
    ran = False
    if a.metrics_default and a.metrics_holdout and a.metrics_default.is_file() and a.metrics_holdout.is_file():
        res = out_of_map_gap(_load_json(a.metrics_default), _load_json(a.metrics_holdout))
        print(f"out-of-map log-loss gap: {res['gap']}" + (f"  WARNING: {res['warning']}" if res["warning"] else ""))
        ran = True
    if a.reliability and a.reliability.is_file():
        print(f"patch drift: {patch_drift(_load_json(a.reliability)) or 'ok'}")
        ran = True
    if a.map_version:
        from engine import db
        path = db.db_path()
        if path.is_file():
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                print(f"map support: {map_version_support(conn, a.map_version, a.min_matches) or 'ok'}")
            finally:
                conn.close()
            ran = True
    if not ran:
        print("robustness: needs data: --metrics-default/--metrics-holdout (a model trained with one map held "
              "out, P11.3.1), --reliability (newest-month calibration, P11.3.2) or --map-version with an "
              "engine DB. None exists until models are trained.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
