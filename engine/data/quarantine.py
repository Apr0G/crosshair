"""
Quarantine and validation (P1.4.1): pure rules on a demo's smoke-parse meta, plus the
move of a failing demo under `data/engine/demos/quarantine/<reason>/`.

    from engine.data import quarantine
    reason = quarantine.check(meta)                     # None = good
    if reason:
        quarantine.quarantine(match_key, demo_path, reason)

Meta keys read (all from `awpy3_parser.header_info(path, with_roster=True)` or the
full parse meta): `tick_rate` / `tick_rate_raw`, `n_rounds`, `side_counts`, `map`,
`duration_s`. A key that is missing or None cannot be verified and fails its rule: an
unverifiable demo is quarantined, never assumed good.

The rules themselves live in ONE place, `awpy3_parser.quarantine_check` (thresholds at
the top of that module); this module only adds the strict "every key must be present"
contract, the `header_unreadable` reason and the file move. The 2026-10-19 audit
(HIGH-01, MED-01) found a second, divergent rule table here that nothing called.
"""
import shutil
from pathlib import Path
from typing import Callable, Iterable

from engine import db, paths
from engine.maps.assets import ACTIVE_POOL
from engine.parse import awpy3_parser

# The competitive pool whose mesh + nav are cached (docs/ENGINE_ENV.md). One list:
# `engine.maps.assets.ACTIVE_POOL`; pass `known_maps` to override.
KNOWN_MAPS = ACTIVE_POOL

# Every key the full check judges; an absent key is judged as None (fails).
REQUIRED_KEYS = ("tick_rate", "n_rounds", "side_counts", "map", "duration_s")

# Reason strings double as directory names under DEMOS_QUARANTINE_DIR. Same vocabulary
# as the parser (plus `header_unreadable` for a header awpy cannot read).
UNREADABLE = "header_unreadable"
REASONS    = (UNREADABLE, *awpy3_parser.QUARANTINE_REASONS)


def check(meta: dict | None, known_maps: Iterable[str] | None = None) -> str | None:
    """The first failing rule's reason, or None when the demo passes every rule.
    Unlike `awpy3_parser.quarantine_check`, a key missing from `meta` is judged (as
    unverifiable) rather than skipped: this is the full check, not the header-only one."""
    if not meta:
        return UNREADABLE
    full = {k: None for k in REQUIRED_KEYS}
    full.update(meta)
    pool = tuple(known_maps) if known_maps is not None else KNOWN_MAPS
    return awpy3_parser.quarantine_check(full, known_maps=pool)


def check_file(demo_path: Path, reader: Callable[[Path], dict],
               known_maps: Iterable[str] | None = None) -> str | None:
    """Run `reader` (the header + roster read) on the file; any exception ->
    `header_unreadable`."""
    try:
        meta = reader(Path(demo_path))
    except Exception:
        return UNREADABLE
    return check(meta, known_maps)


def quarantine(match_key: str, demo_path: Path, reason: str, *,
               base: Path | None = None) -> Path:
    """Move the demo to `<quarantine>/<reason>/<file name>` and set the match's status
    to `quarantined` with the reason. Never overwrites; if the status update fails the
    file is moved back. Returns the new path."""
    if reason not in REASONS:
        raise ValueError(f"unknown quarantine reason {reason!r}; expected one of {REASONS}")
    src  = Path(demo_path)
    if not src.is_file():
        raise FileNotFoundError(f"demo not found: {src}")
    root = Path(base) if base is not None else Path(paths.DEMOS_QUARANTINE_DIR)
    dest = root / reason / src.name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; not overwriting")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    try:
        db.set_status(match_key, "quarantined", reason)
    except BaseException:
        shutil.move(str(dest), str(src))
        raise
    return dest
