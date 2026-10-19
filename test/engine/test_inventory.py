"""
Inventory (P1.3, P1.4.2): counts by tier x map x month x status, quarantine by reason,
failed runs, disk usage, the markdown report, and assert_targets shortfalls.

    .venv\\Scripts\\python.exe test\\engine\\test_inventory.py
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from _harness import check, section, skip, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_inv_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import config, db, paths  # noqa: E402
from engine.data import inventory as inv  # noqa: E402


def _rows() -> list[dict]:
    rows = []
    n = 0
    for tier, mp, date, status, k in [("pro", "de_mirage", "2026-08-03", "downloaded", 3),
                                       ("pro", "de_mirage", "2026-09-10", "parsed", 2),
                                       ("pro", "de_nuke", "2026-09-11", "quarantined", 1),
                                       ("faceit_10", "de_nuke", "2026-09-12", "downloaded", 4),
                                       ("faceit_10", None, None, "downloaded", 1)]:
        for _ in range(k):
            n += 1
            rows.append({"match_key": f"m{n}", "tier": tier, "map": mp, "date": date, "status": status,
                         "quarantine_reason": "tick_rate" if status == "quarantined" else None,
                         "demo_sha256": f"{n:064d}"})
    return rows


def targets(**kw) -> SimpleNamespace:
    base = dict(pro_maps=5, pro_per_map=2, pro_per_month=2, faceit_maps=5, faceit_per_map=4)
    return SimpleNamespace(data=SimpleNamespace(targets=SimpleNamespace(**{**base, **kw})))


def main() -> int:
    rows = _rows()
    for r in rows:
        db.register_match(r)
    db.upsert_run("r1", "parse", status="failed")
    db.upsert_run("r2", "parse", status="ok")
    db.upsert_run("r3", "states", status="failed")
    base = TMP / "engine"
    (base / "demos" / "pro").mkdir(parents=True)
    (base / "demos" / "pro" / "a.dem").write_bytes(b"x" * 1500)
    (base / "states" / "v" / "2026-09").mkdir(parents=True)
    (base / "states" / "v" / "2026-09" / "m.parquet").write_bytes(b"y" * 500)

    section("collect")
    data = inv.collect(base)
    c = data["counts"]
    check("counts by tier x map x month x status",
          c[("pro", "de_mirage", "2026-08", "downloaded")] == 3 and c[("pro", "de_mirage", "2026-09", "parsed")] == 2
          and c[("pro", "de_nuke", "2026-09", "quarantined")] == 1 and c[("faceit_10", "de_nuke", "2026-09", "downloaded")] == 4
          and c[("faceit_10", None, None, "downloaded")] == 1 and sum(c.values()) == 11, str(dict(c)))
    check("quarantine by reason", dict(data["quarantine"]) == {"tick_rate": 1})
    check("failed runs by kind", dict(data["failures"]) == {"parse": 1, "states": 1})
    check("disk usage per layout dir", data["disk"]["demos/pro"] == 1500 and data["disk"]["states"] == 500
          and data["disk"]["models"] == 0 and len(data["disk"]) == len(paths.ALL_DIRS), str(data["disk"]))

    section("report")
    text = inv.report(data, ["pro maps: 5 < 6"], today="2026-10-19")
    check("markdown has every section", all(s in text for s in (
        "# Inventory 2026-10-19", "| pro | de_mirage | 2026-08 | downloaded | 3 |",
        "| faceit_10 | (unknown) | (unknown) | downloaded | 1 |", "- tick_rate: 1", "- parse: 1",
        "| demos/pro | 0.00 GB |", "- SHORT pro maps: 5 < 6")), text[:400])
    path = inv.write_report(text, eval_dir=TMP / "eval", today="2026-10-19")
    check("written to eval/inventory_<date>.md", path == TMP / "eval" / "inventory_2026-10-19.md"
          and path.read_text(encoding="utf-8") == text)

    section("assert_targets")
    met = inv.assert_targets(targets(pro_maps=5, pro_per_map=2, pro_per_month=2, faceit_maps=4, faceit_per_map=0),
                             rows, maps=["de_mirage"], last_month="2023-10")
    check("quarantined rows don't count; met targets -> one known month gap only",
          met == ["pro month 2023-10: 0 < 2"], str(met))
    short = inv.assert_targets(targets(pro_maps=6, pro_per_map=3, pro_per_month=3, faceit_maps=10, faceit_per_map=4),
                               rows, maps=["de_mirage", "de_nuke"], last_month="2023-11")
    want = ["pro maps: 5 < 6", "pro de_nuke: 0 < 3", "pro month 2023-10: 0 < 3", "pro month 2023-11: 0 < 3",
            "faceit_10 maps: 5 < 10", "faceit_10 de_mirage: 0 < 4"]
    check("deliberately short table lists every shortfall", short == want, str(short))
    months = inv._months("2023-10", "2026-10")
    check("month span Oct 2023 .. Oct 2026", months[0] == "2023-10" and months[-1] == "2026-10" and len(months) == 37)
    try:
        cfg = config.load()
    except config.ConfigError as e:          # config.yaml is owned elsewhere; don't fail on it here
        skip("real config targets on a tiny table are short", f"config does not load: {e}")
    else:
        real = inv.assert_targets(cfg, rows)
        check("real config targets on a tiny table are short", any(s.startswith("pro maps:") for s in real))
    return summary()


if __name__ == "__main__":
    sys.exit(main())
