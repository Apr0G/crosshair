"""
Frozen time splits (P1.5): train = all months except the last four, val = the two
before the last two, test = the last two; written once to `splits_v<N>.json`, never
edited, only superseded.

    from engine.data import splits
    path = splits.make_splits(db.list_matches())               # datasets/splits_v1.json
    path = splits.make_splits(rows, holdout_maps=("de_nuke",), note="out-of-map check")
    s    = splits.load(path); splits.file_hash(path)           # hash goes into manifests

    .venv\\Scripts\\python.exe -m engine.data.splits --db [--holdout-maps de_nuke] [--note ...]
    .venv\\Scripts\\python.exe -m engine.data.splits --catalog data\\engine\\catalog.parquet [--out DIR]

Matches are assigned by the month of their `date` (`yyyy-mm`), so pro and FACEIT share
the same month boundaries. A match with no date is listed under `undated` and is in no
split. Holdout maps (P1.5.2) are removed from train/val/test into `holdout`.
`main()` reads the rows from engine.db (`--db`) or from a catalog file (`--catalog`,
parquet or a JSON list); a catalog row without `date` / `map` uses its `month` /
`map_version` (LOW-35, 2026-10-19).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from engine import paths

_MONTH_RE   = re.compile(r"^(\d{4})-(\d{2})")
_VERSION_RE = re.compile(r"^splits_v(\d+)\.json$")
SPLIT_NAMES = ("train", "val", "test")


def month_of(date) -> str | None:
    """`yyyy-mm` from an ISO date string; None when it has no usable month."""
    if date is None:
        return None
    m = _MONTH_RE.match(str(date))
    if not m or not 1 <= int(m.group(2)) <= 12:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def existing_versions(out_dir: Path) -> list[int]:
    if not Path(out_dir).is_dir():
        return []
    return sorted(int(m.group(1)) for p in Path(out_dir).iterdir()
                  if (m := _VERSION_RE.match(p.name)))


def assign(matches: Iterable[dict], n_test_months: int = 2, n_val_months: int = 2,
           holdout_maps: Iterable[str] = ()) -> dict:
    """Pure split assignment; see module docstring. Returns the dict that is written."""
    if n_test_months < 1 or n_val_months < 1:
        raise ValueError("n_test_months and n_val_months must be >= 1")
    holdout = set(holdout_maps)
    rows    = list(matches)
    dupes   = sorted(k for k, n in Counter(r["match_key"] for r in rows).items() if n > 1)   # O(n), LOW-35
    if dupes:
        raise ValueError(f"duplicate match_key in input: {dupes[:5]}")

    dated   = [(r, month_of(r.get("date"))) for r in rows]
    months  = sorted({m for r, m in dated if m is not None and r.get("map") not in holdout})
    if len(months) < n_test_months + n_val_months + 1:
        raise ValueError(f"need at least {n_test_months + n_val_months + 1} months with data "
                         f"for train/val/test, have {len(months)}: {months}")
    test_m  = months[-n_test_months:]
    val_m   = months[-(n_test_months + n_val_months):-n_test_months]
    train_m = months[:-(n_test_months + n_val_months)]
    which   = {**{m: "train" for m in train_m}, **{m: "val" for m in val_m},
               **{m: "test" for m in test_m}}

    out = {name: [] for name in (*SPLIT_NAMES, "holdout", "undated")}
    for r, m in dated:
        if r.get("map") in holdout:
            out["holdout"].append(r["match_key"])
        elif m is None:
            out["undated"].append(r["match_key"])
        else:
            out[which[m]].append(r["match_key"])
    for v in out.values():
        v.sort()
    return {
        "months":       {"train": train_m, "val": val_m, "test": test_m},
        "splits":       {k: out[k] for k in (*SPLIT_NAMES, "holdout")},
        "undated":      out["undated"],
        "counts":       {k: len(v) for k, v in out.items()},
        "n_test_months": n_test_months,
        "n_val_months":  n_val_months,
        "holdout_maps":  sorted(holdout),
    }


def make_splits(matches: Iterable[dict], n_test_months: int = 2, n_val_months: int = 2,
                holdout_maps: Iterable[str] = (), *, out_dir: Path | None = None,
                note: str | None = None) -> Path:
    """Assign and write `splits_v<N+1>.json` (N = highest existing version). Written
    with exclusive create, so an existing version is never overwritten."""
    out_dir = Path(out_dir) if out_dir is not None else Path(paths.DATASETS_DIR)
    body    = assign(matches, n_test_months, n_val_months, holdout_maps)
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = existing_versions(out_dir)
    version  = (versions[-1] if versions else 0) + 1
    doc = {"version": version,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "note": note,
           "supersedes": versions[-1] if versions else None,
           **body}
    path = out_dir / f"splits_v{version}.json"
    with open(path, "x", encoding="utf-8") as fh:          # "x": refuse to overwrite
        json.dump(doc, fh, indent=2, sort_keys=True)
    return path


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    """sha256 of the splits file, recorded in manifests (P1.5.1)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def read_rows(catalog: Path) -> list[dict]:
    """Rows from a catalog file (parquet or JSON list); `date` falls back to `month`,
    `map` to the map name of `map_version`."""
    p = Path(catalog)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(p).to_pylist()
    else:
        rows = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        r = dict(r)
        if r.get("date") is None:
            r["date"] = r.get("month")
        if r.get("map") is None and r.get("map_version"):
            r["map"] = str(r["map_version"]).split("@")[0]
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the next frozen splits_v<N>.json (P1.5).")
    ap.add_argument("--catalog", default=None, help="catalog.parquet or a JSON list of match rows")
    ap.add_argument("--db", action="store_true", help="read the matches table of engine.db instead")
    ap.add_argument("--out", default=None, help="output directory (default data/engine/datasets)")
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--val-months", type=int, default=2)
    ap.add_argument("--holdout-maps", nargs="*", default=())
    ap.add_argument("--note", default=None)
    a = ap.parse_args(argv)
    if bool(a.catalog) == bool(a.db):
        print("splits: give exactly one of --catalog PATH or --db", file=sys.stderr)
        return 2
    try:
        if a.db:
            from engine import db
            rows = db.list_matches()
        else:
            rows = read_rows(Path(a.catalog))
        path = make_splits(rows, a.test_months, a.val_months, a.holdout_maps,
                           out_dir=Path(a.out) if a.out else None, note=a.note)
    except (FileNotFoundError, ValueError) as exc:
        print(f"splits: {exc}", file=sys.stderr)
        return 2
    doc = load(path)
    print(f"{path}  months {doc['months']}  counts {doc['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
