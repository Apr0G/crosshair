"""
P11.3 robustness checks (out-of-map gap, patch drift, map-version support) and the
P12.3 usage log (append, redaction, weekly summary). Pure functions on planted inputs;
`main` without data exits 2.

    .venv\\Scripts\\python.exe test\\engine\\test_robustness.py
"""
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _harness import check, close, section, summary
from engine import config
from engine.evaluation import robustness as rb
from engine.evaluation import usage as us

# ── out-of-map ────────────────────────────────────────────────────────────────
section("out-of-map gap (P11.3.1)")
g = rb.out_of_map_gap({"logloss": 0.46, "brier": 0.15, "ece": 0.01}, {"logloss": 0.55, "brier": 0.19, "ece": 0.04})
check("gap = holdout − default (0.09)", close(g["gap"], 0.09) and close(g["brier_gap"], 0.04))
check("worse holdout -> warning about new map data", g["warning"] is not None and "map" in g["warning"])
g2 = rb.out_of_map_gap({"logloss": 0.50}, {"logloss": 0.49})
check("holdout not worse -> no warning", g2["warning"] is None and close(g2["gap"], -0.01))
g3 = rb.out_of_map_gap({"logloss": None}, {})
check("missing metrics -> gap None (never 0), no warning", g3["gap"] is None and g3["warning"] is None
      and g3["brier_gap"] is None)
check("NaN metric -> None", rb.out_of_map_gap({"logloss": float("nan")}, {"logloss": 0.5})["gap"] is None)

# ── patch drift ───────────────────────────────────────────────────────────────
section("patch drift (P11.3.2)")
w = rb.patch_drift({"ece": 0.05})
check("ECE 0.05 > 0.03 -> 'model predates recent patches'", w is not None and rb.PATCH_WARNING in w)
check("ECE 0.03 exactly -> no warning", rb.patch_drift({"ece": 0.03}) is None)
check("ECE 0.01 -> no warning", rb.patch_drift({"ece": 0.01}) is None)
bins = [{"lo": 0.0, "hi": 0.5, "n": 100, "mean_pred": 0.30, "observed": 0.40},
        {"lo": 0.5, "hi": 1.0, "n": 300, "mean_pred": 0.70, "observed": 0.70},
        {"lo": 0.9, "hi": 1.0, "n": 0, "mean_pred": None, "observed": None}]
check("ECE from reliability bins = 100/400 · 0.10 = 0.025", close(rb.ece_from_reliability(bins), 0.025))
check("bins with ECE 0.025 -> no warning", rb.patch_drift(bins) is None)
bins[0]["observed"] = 0.50
check("bins with ECE 0.05 -> warning (list and {reliability: ...} forms)",
      rb.patch_drift(bins) is not None and rb.patch_drift({"reliability": bins}) is not None)
check("no data -> None", rb.patch_drift([]) is None and rb.patch_drift({"ece": None}) is None)

# ── map-version support ───────────────────────────────────────────────────────
section("map-version support (P11.3.1)")
conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE matches (match_key TEXT PRIMARY KEY, map_version TEXT, status TEXT)")
rows = ([(f"a{i}", "de_mirage@2026-05", "states") for i in range(105)]
        + [(f"q{i}", "de_mirage@2026-05", "quarantined") for i in range(10)]
        + [(f"b{i}", "de_anubis@unknown", None) for i in range(40)])
conn.executemany("INSERT INTO matches VALUES (?, ?, ?)", rows)
check("count excludes quarantined, keeps null status", rb.map_version_count(conn, "de_mirage@2026-05") == 105
      and rb.map_version_count(conn, "de_anubis@unknown") == 40)
check(">= 100 matches -> no warning", rb.map_version_support(conn, "de_mirage@2026-05") is None)
wa = rb.map_version_support(conn, "de_anubis@unknown")
check("40 matches -> warning naming the count", wa is not None and "40" in wa)
check("custom min_matches", rb.map_version_support(conn, "de_anubis@unknown", min_matches=40) is None)
check("unseen map version -> warning (0 matches)", "0 matches" in rb.map_version_support(conn, "de_dust2@new"))
check("unknown map version -> warning", rb.map_version_support(conn, None) is not None)
conn.close()
check("main() without data exits 2", rb.main([]) == 2)
with tempfile.TemporaryDirectory() as td:
    a, b = Path(td) / "a.json", Path(td) / "b.json"
    a.write_text(json.dumps({"logloss": 0.4}), encoding="utf-8")
    b.write_text(json.dumps({"logloss": 0.5}), encoding="utf-8")
    check("main() with metrics files exits 0", rb.main(["--metrics-default", str(a), "--metrics-holdout", str(b)]) == 0)

# ── usage log (P12.3) ─────────────────────────────────────────────────────────
section("usage log (P12.3)")
cfg = config.load()
now = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)


def rec(i: int, **kw) -> dict:
    r = {"run_id": f"r{i}", "map_version": "de_mirage@unknown", "tier": "pro", "device": "gpu",
         "timings": {"parse": 10.0, "total": 50.0}, "warnings": 1, "conservation_residual": 1e-9,
         "ts": (now - timedelta(days=1)).isoformat()}
    r.update(kw)
    return r


with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "eval" / "usage.jsonl"
    us.append_usage(rec(1), p)
    us.append_usage(rec(2, tier="faceit_10", map_version="de_nuke@unknown", conservation_residual=5e-3,
                        warnings=["a", "b", "c"]), p)
    us.append_usage(rec(3, timings={"parse": 100.0, "attribution": 60.0}), p)            # no total: sum 160 > 120
    us.append_usage(rec(4, device="cpu", timings={"total": 250.0}), p)                   # under the CPU ceiling
    us.append_usage(rec(5, ts=(now - timedelta(days=9)).isoformat(), conservation_residual=1.0), p)  # too old
    us.append_usage(rec(6, conservation_residual=float("nan"),
                        note="https://demos.example.com/x.dem.zst?sig=SECRET"), p)
    written = us.append_usage({k: v for k, v in rec(7).items() if k != "ts"}, Path(td) / "other.jsonl")
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    lines = p.read_text(encoding="utf-8").splitlines()
    check("one JSON line per run (+ the malformed one)", len(lines) == 7)
    check("ts added when missing", "ts" in written)
    check("warnings list stored as its count", json.loads(lines[1])["warnings"] == 3)
    check("NaN residual stored as null", json.loads(lines[5])["conservation_residual"] is None)
    check("URL query redacted", "SECRET" not in p.read_text(encoding="utf-8"))
    try:
        us.append_usage({"run_id": "x"}, p)
        check("record missing required keys refused", False)
    except ValueError:
        check("record missing required keys refused", True)
    s = us.summarize(us.read_usage(p)[0], now=now, cfg=cfg)
    check("last 7 days: 5 runs (r5 is 9 days old)", s["runs"] == 5, f"{s['runs']}")
    check("counts by tier", s["by_tier"] == {"pro": 4, "faceit_10": 1}, f"{s['by_tier']}")
    check("residual > 1e-3 listed (r2 only; r5 is outside the week)",
          [r["run_id"] for r in s["residual_over"]] == ["r2"])
    check("over the ceiling: r3 (160 s summed > 120 GPU); r4 250 s CPU is under 300",
          [r["run_id"] for r in s["over_ceiling"]] == ["r3"])
    txt = us.weekly_summary(p, now=now, cfg=cfg)
    check("summary text names counts, residual run, slow run, malformed line",
          "5 runs" in txt and "r2" in txt and "r3" in txt and "1 malformed" in txt, txt.splitlines()[0])
    check("missing log -> plain message", "no usage log" in us.weekly_summary(Path(td) / "none.jsonl"))
    check("total_time: total wins, else the stage sum, else None",
          us.total_time({"a": 1.0, "total": 9.0}) == 9.0 and us.total_time({"a": 1.0, "b": 2.0}) == 3.0
          and us.total_time({}) is None)
check("usage main() without a log exits 2", us.main(["--path", "no/such/usage.jsonl"]) == 2)

raise SystemExit(summary())
