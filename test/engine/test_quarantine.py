"""
Quarantine rules and the file move (P1.4.1).

    .venv\\Scripts\\python.exe test\\engine\\test_quarantine.py

Synthetic failure cases: truncated file (unreadable header), 64-tick header, 3 rounds,
4v5, unknown map, implausible duration, missing values. The rules are the parser's
(`awpy3_parser.quarantine_check`); this module checks that `engine.data.quarantine`
judges a missing key and shares the reason vocabulary (2026-10-19 audit, MED-01).
"""
import os
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_quar_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import db  # noqa: E402
from engine.data import quarantine as q  # noqa: E402
from engine.parse import awpy3_parser as ap  # noqa: E402

GOOD = {"tick_rate": 128, "tick_rate_raw": 128.0, "n_rounds": 24, "side_counts": {"ct": 5, "t": 5},
        "map": "de_mirage", "duration_s": 2700.0}


def main() -> int:
    section("rules")
    check("good meta -> None", q.check(GOOD) is None)
    T64 = {"tick_rate": 64, "tick_rate_raw": 64.0}
    cases = [
        (None,                                          "header_unreadable",    "no meta"),
        ({},                                            "header_unreadable",    "empty meta"),
        ({**GOOD, **T64},                               "tick_rate_not_128",    "64-tick header"),
        ({**GOOD, "tick_rate": None, "tick_rate_raw": None}, "tick_rate_not_128", "unknown tick rate"),
        ({**GOOD, "n_rounds": 3},                       "too_few_rounds",       "3 rounds"),
        ({**GOOD, "n_rounds": None},                    "too_few_rounds",       "unknown rounds"),
        ({**GOOD, "side_counts": {"ct": 4, "t": 5}},    "not_5v5",              "4v5"),
        ({**GOOD, "side_counts": {"ct": 5, "t": 6}},    "not_5v5",              "5v6"),
        ({**GOOD, "side_counts": None},                 "not_5v5",              "unknown roster"),
        ({**GOOD, "map": "de_dust"},                    "unknown_map",          "map outside the pool"),
        ({**GOOD, "map": None},                         "unknown_map",          "unknown map"),
        ({**GOOD, "duration_s": 60},                    "duration_implausible", "1-minute match"),
        ({**GOOD, "duration_s": 30 * 3600},             "duration_implausible", "30-hour match"),
        ({**GOOD, "duration_s": None},                  "duration_implausible", "unknown duration"),
    ]
    for meta, want, label in cases:
        got = q.check(meta)
        check(f"{label} -> {want}", got == want, str(got))
    for key in q.REQUIRED_KEYS:
        meta = {k: v for k, v in GOOD.items() if k != key and k != f"{key}_raw"}
        check(f"missing key {key} is judged, not skipped", q.check(meta) is not None, str(q.check(meta)))
    check("16 rounds is enough", q.check({**GOOD, "n_rounds": 16}) is None)
    check("known_maps override", q.check({**GOOD, "map": "de_cache"}, known_maps={"de_cache"}) is None)
    check("every reason is a declared directory name", {c[1] for c in cases} <= set(q.REASONS))
    check("one reason vocabulary with the parser", set(q.REASONS) == {q.UNREADABLE, *ap.QUARANTINE_REASONS})
    check("one map pool with the assets module", tuple(q.KNOWN_MAPS) == tuple(__import__("engine.maps.assets", fromlist=["ACTIVE_POOL"]).ACTIVE_POOL))
    check("same verdicts as the parser on a full meta",
          all(q.check(m) == ap.quarantine_check(m) for m, _, _ in cases if m))

    def truncated_reader(path):
        raise EOFError("unexpected end of demo")
    trunc = TMP / "trunc.dem"
    trunc.write_bytes(b"PBDEMS2\x00")
    check("truncated file -> header_unreadable", q.check_file(trunc, truncated_reader) == "header_unreadable")
    check("check_file passes meta through", q.check_file(trunc, lambda p: {**GOOD, **T64}) == "tick_rate_not_128")

    section("quarantine(): move + status")
    base = TMP / "quarantine"
    demo = TMP / "demos" / "m1.dem"
    demo.parent.mkdir(parents=True)
    demo.write_bytes(b"x" * 2000)
    db.register_match({"match_key": "m1", "demo_sha256": "1" * 64})
    dest = q.quarantine("m1", demo, "tick_rate_not_128", base=base)
    check("file moved under <quarantine>/<reason>/", dest == base / "tick_rate_not_128" / "m1.dem"
          and dest.is_file() and not demo.exists())
    row = db.get_match("m1")
    check("status quarantined with the reason",
          row["status"] == "quarantined" and row["quarantine_reason"] == "tick_rate_not_128")

    demo2 = TMP / "demos" / "ghost.dem"
    demo2.write_bytes(b"y" * 2000)
    try:
        q.quarantine("ghost", demo2, "too_few_rounds", base=base)
        check("unknown match raises", False)
    except KeyError:
        check("unknown match: file moved back, nothing left in quarantine",
              demo2.is_file() and not (base / "too_few_rounds" / "ghost.dem").exists())
    for args, exc, label in [(("m1", demo2, "../evil"), ValueError, "unknown reason"),
                             (("m1", demo2, "tick_rate"), ValueError, "legacy reason name"),
                             (("m1", TMP / "nope.dem", "duration_implausible"), FileNotFoundError, "missing file")]:
        try:
            q.quarantine(*args, base=base)
            check(f"refuses {label}", False)
        except exc:
            check(f"refuses {label}", True)
    clash = TMP / "demos" / "again" / "m1.dem"
    clash.parent.mkdir()
    clash.write_bytes(b"z" * 2000)
    try:
        q.quarantine("m1", clash, "tick_rate_not_128", base=base)
        check("never overwrites a quarantined file", False)
    except FileExistsError:
        check("never overwrites a quarantined file", clash.is_file() and dest.read_bytes() == b"x" * 2000)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
