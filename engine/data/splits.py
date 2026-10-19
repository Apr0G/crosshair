"""
Frozen time splits (P1.5): train = all months except the last four, val = the two
before the last two, test = the last two; written once to `splits_v<N>.json`, never
edited, only superseded.

    from engine.data import splits
    path = splits.make_splits(db.list_matches())               # datasets/splits_v1.json
    path = splits.make_splits(rows, holdout_maps=("de_nuke",), note="out-of-map check")
    s    = splits.load(path); splits.file_hash(path)           # hash goes into manifests

Matches are assigned by the month of their `date` (`yyyy-mm`), so pro and FACEIT share
the same month boundaries. A match with no date is listed under `undated` and is in no
split. Holdout maps (P1.5.2) are removed from train/val/test into `holdout`.
"""
import hashlib
import json
import re
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
    keys    = [r["match_key"] for r in rows]
    dupes   = sorted({k for k in keys if keys.count(k) > 1})
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
