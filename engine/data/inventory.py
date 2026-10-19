"""
`engine inventory` (P1.3): match counts by tier x map x month x status, disk use,
failures, quarantine counts; `--assert-targets` against `config.data.targets`.

    from engine.data import inventory
    inv  = inventory.collect()                    # reads engine.db + walks data/engine
    text = inventory.report(inv)                  # markdown
    inventory.write_report(text)                  # data/engine/eval/inventory_<date>.md
    short = inventory.assert_targets(cfg, inv["rows"])   # [] when every target is met

Targets (P1.1.6, P1.2.3) count non-quarantined matches only. Tier names: `pro`,
`faceit_10` (engine.ingest.TIER_DIRS). The per-map targets apply to the active pool
(`quarantine.KNOWN_MAPS` unless `maps` is given), so a map with zero demos is a
shortfall; the per-month pro target covers every month from October 2023 through
`last_month` (default: the current month).
"""
from collections import Counter
from datetime import date as _date
from pathlib import Path
from typing import Iterable

from engine import db, paths
from engine.data.quarantine import KNOWN_MAPS
from engine.data.splits import month_of

PRO_TIER      = "pro"
FACEIT_TIER   = "faceit_10"
FIRST_MONTH   = "2023-10"
UNKNOWN_LABEL = "(unknown)"          # display only; the DB keeps NULL


# ── collection ────────────────────────────────────────────────────────────────

def counts(rows: Iterable[dict]) -> Counter:
    """Counter keyed by (tier, map, month, status); unknown parts are None."""
    return Counter((r.get("tier"), r.get("map"), month_of(r.get("date")), r.get("status"))
                   for r in rows)


def quarantine_counts(rows: Iterable[dict]) -> Counter:
    return Counter(r.get("quarantine_reason") for r in rows if r.get("status") == "quarantined")


def failure_counts(runs: Iterable[dict]) -> Counter:
    """Runs whose status is `failed`, by kind."""
    return Counter(r.get("kind") for r in runs if r.get("status") == "failed")


def disk_usage(base: Path | None = None) -> dict[str, int]:
    """Bytes under each layout directory (relative name -> bytes); missing dirs -> 0."""
    root = Path(base) if base is not None else paths.ENGINE_DIR
    out: dict[str, int] = {}
    for d in paths.ALL_DIRS:
        rel  = d.relative_to(paths.ENGINE_DIR)
        full = root / rel
        out[rel.as_posix()] = sum(f.stat().st_size for f in full.rglob("*") if f.is_file()) \
            if full.is_dir() else 0
    return out


def collect(base: Path | None = None) -> dict:
    rows = db.list_matches()
    runs = db.list_runs()
    return {"rows": rows, "counts": counts(rows), "quarantine": quarantine_counts(rows),
            "failures": failure_counts(runs), "disk": disk_usage(base)}


# ── targets ───────────────────────────────────────────────────────────────────

def _months(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    out = []
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def assert_targets(cfg, rows: Iterable[dict] | None = None, *, maps: Iterable[str] | None = None,
                   last_month: str | None = None) -> list[str]:
    """Shortfalls against `cfg.data.targets` as human-readable lines; [] = all met."""
    rows   = list(db.list_matches() if rows is None else rows)
    good   = [r for r in rows if r.get("status") != "quarantined"]
    t      = cfg.data.targets
    pool   = sorted(maps if maps is not None else KNOWN_MAPS)
    last   = last_month or _date.today().strftime("%Y-%m")
    pro    = [r for r in good if r.get("tier") == PRO_TIER]
    faceit = [r for r in good if r.get("tier") == FACEIT_TIER]
    short: list[str] = []

    if len(pro) < t.pro_maps:
        short.append(f"pro maps: {len(pro)} < {t.pro_maps}")
    by_map = Counter(r.get("map") for r in pro)
    short += [f"pro {m}: {by_map[m]} < {t.pro_per_map}" for m in pool if by_map[m] < t.pro_per_map]
    by_month = Counter(month_of(r.get("date")) for r in pro)
    short += [f"pro month {mo}: {by_month[mo]} < {t.pro_per_month}"
              for mo in _months(FIRST_MONTH, last) if by_month[mo] < t.pro_per_month]

    if len(faceit) < t.faceit_maps:
        short.append(f"faceit_10 maps: {len(faceit)} < {t.faceit_maps}")
    by_map = Counter(r.get("map") for r in faceit)
    short += [f"faceit_10 {m}: {by_map[m]} < {t.faceit_per_map}"
              for m in pool if by_map[m] < t.faceit_per_map]
    return short


# ── report ────────────────────────────────────────────────────────────────────

def _label(v) -> str:
    return UNKNOWN_LABEL if v is None else str(v)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def report(inv: dict, shortfalls: list[str] | None = None, *, today: str | None = None) -> str:
    today = today or _date.today().isoformat()
    lines = [f"# Inventory {today}", "", f"Matches: {len(inv['rows'])}", "",
             "## Matches by tier × map × month × status", "",
             "| tier | map | month | status | n |", "|---|---|---|---|---|"]
    for (tier, mp, mo, st), n in sorted(inv["counts"].items(), key=lambda kv: tuple(map(_label, kv[0]))):
        lines.append(f"| {_label(tier)} | {_label(mp)} | {_label(mo)} | {_label(st)} | {n} |")
    lines += ["", "## Quarantine by reason", ""]
    lines += [f"- {_label(k)}: {n}" for k, n in sorted(inv["quarantine"].items(), key=lambda kv: _label(kv[0]))] \
        or ["- none"]
    lines += ["", "## Failed runs by kind", ""]
    lines += [f"- {_label(k)}: {n}" for k, n in sorted(inv["failures"].items(), key=lambda kv: _label(kv[0]))] \
        or ["- none"]
    lines += ["", "## Disk usage", "", "| dir | size |", "|---|---|"]
    lines += [f"| {d} | {_gb(n)} |" for d, n in sorted(inv["disk"].items())]
    if shortfalls is not None:
        lines += ["", "## Targets", ""]
        lines += [f"- SHORT {s}" for s in shortfalls] or ["- all targets met"]
    return "\n".join(lines) + "\n"


def write_report(text: str, *, eval_dir: Path | None = None, today: str | None = None) -> Path:
    out = Path(eval_dir) if eval_dir is not None else Path(paths.EVAL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"inventory_{today or _date.today().isoformat()}.md"
    tmp  = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def run(cfg=None, *, check_targets: bool = False, base: Path | None = None) -> int:
    """What `engine inventory [--assert-targets]` does: print + write the report.
    Returns 1 when `check_targets` and a target is missed (a reminder, not a gate)."""
    inv   = collect(base)
    short = None
    if check_targets:
        if cfg is None:
            from engine import config
            cfg = config.load()
        short = assert_targets(cfg, inv["rows"])
    text = report(inv, short)
    print(text)
    print(f"written: {write_report(text)}")
    return 1 if short else 0
