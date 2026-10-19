"""
Frozen time splits (P1.5): every match in exactly one split, month boundaries,
versions never overwritten, holdout maps, file hash.

    .venv\\Scripts\\python.exe test\\engine\\test_splits.py
"""
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

from engine.data import splits


def _matches() -> list[dict]:
    rows = []
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    for i, mo in enumerate(months):
        for j in range(3):
            tier = "pro" if j == 0 else "faceit_10"
            rows.append({"match_key": f"{tier}-{mo}-{j}", "date": f"{mo}-{10 + j}T12:00:00",
                         "map": "de_nuke" if j == 2 else "de_mirage", "tier": tier})
    rows.append({"match_key": "nodate", "date": None, "map": "de_mirage"})
    rows.append({"match_key": "baddate", "date": "2026-13-01", "map": "de_mirage"})
    return rows


def main() -> int:
    out = Path(tempfile.mkdtemp(prefix="ch_splits_test_"))
    rows = _matches()

    section("assignment")
    path = splits.make_splits(rows, out_dir=out, note="first")
    s = splits.load(path)
    check("written as splits_v1.json", path.name == "splits_v1.json" and s["version"] == 1 and s["supersedes"] is None)
    check("months: train = all but last four, val = two before last two, test = last two",
          s["months"] == {"train": ["2026-01", "2026-02"], "val": ["2026-03", "2026-04"], "test": ["2026-05", "2026-06"]},
          str(s["months"]))
    sets = {k: set(v) for k, v in s["splits"].items()}
    all_keys = [k for v in s["splits"].values() for k in v]
    check("no match in two splits", len(all_keys) == len(set(all_keys)))
    dated = {r["match_key"] for r in rows if splits.month_of(r["date"])}
    check("every dated match in exactly one split", set(all_keys) == dated)
    check("undated / invalid dates listed, in no split", s["undated"] == ["baddate", "nodate"])
    by_key = {r["match_key"]: splits.month_of(r["date"]) for r in rows}
    check("month boundaries respected", all(by_key[k] in s["months"][name]
                                            for name in ("train", "val", "test") for k in sets[name]))
    check("pro and FACEIT share the boundaries", "pro-2026-05-0" in sets["test"] and "faceit_10-2026-05-1" in sets["test"]
          and "pro-2026-04-0" in sets["val"] and "faceit_10-2026-02-1" in sets["train"])
    check("counts recorded", s["counts"] == {"train": 6, "val": 6, "test": 6, "holdout": 0, "undated": 2}, str(s["counts"]))

    section("versions are never overwritten")
    h1 = splits.file_hash(path)
    p2 = splits.make_splits(rows, out_dir=out, holdout_maps=("de_nuke",), note="out-of-map")
    check("second call writes v2", p2.name == "splits_v2.json" and splits.load(p2)["supersedes"] == 1)
    check("v1 untouched", splits.file_hash(path) == h1)
    s2 = splits.load(p2)
    check("holdout map removed from train/val/test", s2["splits"]["holdout"] == sorted(
        r["match_key"] for r in rows if r["map"] == "de_nuke")
          and not any(k.endswith("-2") for n in ("train", "val", "test") for k in s2["splits"][n]))
    check("holdout recorded", s2["holdout_maps"] == ["de_nuke"] and s2["counts"]["holdout"] == 6)
    check("file hash is a sha256 hex", len(h1) == 64 and int(h1, 16) >= 0)

    section("refusals")
    (out / "splits_v3.json").write_text("{}", encoding="utf-8")
    p4 = splits.make_splits(rows, out_dir=out)
    check("next version follows the highest existing one", p4.name == "splits_v4.json"
          and (out / "splits_v3.json").read_text(encoding="utf-8") == "{}")
    for bad, why in [(rows + [rows[0]], "duplicate match_key"),
                     ([r for r in rows if r["date"] and r["date"].startswith(("2026-01", "2026-02", "2026-03"))], "too few months")]:
        try:
            splits.make_splits(bad, out_dir=out)
            check(f"refuses {why}", False)
        except ValueError:
            check(f"refuses {why}", True)
    check("custom month counts", splits.assign(rows, n_test_months=1, n_val_months=1)["months"]["train"]
          == ["2026-01", "2026-02", "2026-03", "2026-04"])
    return summary()


if __name__ == "__main__":
    sys.exit(main())
