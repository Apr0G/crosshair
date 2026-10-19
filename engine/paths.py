"""
On-disk layout of the engine (P0.3). Everything lives under `data/engine/`, which is
gitignored with the rest of `data/`.

    from engine import paths
    paths.ensure()                     # create every directory
    paths.check_space()                # SpaceStatus.OK | WARN | REFUSE

Stdlib only: this module must import on a bare interpreter.
"""
import shutil
from enum import Enum
from pathlib import Path

# ── layout ────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
ENGINE_DIR = DATA_DIR / "engine"

# Relative to ENGINE_DIR. P0.3.1 lists the first twelve; golden (P0.7.1),
# quarantine (P1.4.1) and windows (P5.1.1) are named later in the plan.
_LAYOUT = {
    "DEMOS_PRO":        "demos/pro",
    "DEMOS_FACEIT":     "demos/faceit",
    "DEMOS_GOLDEN":     "demos/golden",
    "DEMOS_QUARANTINE": "demos/quarantine",
    "PARSED":           "parsed",
    "STATES":           "states",
    "ACTIONS":          "actions",
    "WINDOWS":          "windows",
    "DATASETS":         "datasets",
    "MODELS":           "models",
    "EVAL":             "eval",
    "REPORTS":          "reports",
    "BENCH":            "bench",
    "LOGS":             "logs",
    "CACHE_MAPS":       "cache/maps",
}

DEMOS_PRO_DIR        = ENGINE_DIR / _LAYOUT["DEMOS_PRO"]
DEMOS_FACEIT_DIR     = ENGINE_DIR / _LAYOUT["DEMOS_FACEIT"]
DEMOS_GOLDEN_DIR     = ENGINE_DIR / _LAYOUT["DEMOS_GOLDEN"]
DEMOS_QUARANTINE_DIR = ENGINE_DIR / _LAYOUT["DEMOS_QUARANTINE"]
PARSED_DIR           = ENGINE_DIR / _LAYOUT["PARSED"]
STATES_DIR           = ENGINE_DIR / _LAYOUT["STATES"]
ACTIONS_DIR          = ENGINE_DIR / _LAYOUT["ACTIONS"]
WINDOWS_DIR          = ENGINE_DIR / _LAYOUT["WINDOWS"]
DATASETS_DIR         = ENGINE_DIR / _LAYOUT["DATASETS"]
MODELS_DIR           = ENGINE_DIR / _LAYOUT["MODELS"]
EVAL_DIR             = ENGINE_DIR / _LAYOUT["EVAL"]
REPORTS_DIR          = ENGINE_DIR / _LAYOUT["REPORTS"]
BENCH_DIR            = ENGINE_DIR / _LAYOUT["BENCH"]
LOGS_DIR             = ENGINE_DIR / _LAYOUT["LOGS"]
CACHE_MAPS_DIR       = ENGINE_DIR / _LAYOUT["CACHE_MAPS"]

ALL_DIRS: tuple[Path, ...] = tuple(ENGINE_DIR / rel for rel in _LAYOUT.values())

DB_PATH = ENGINE_DIR / "engine.db"          # Amendment 1: the engine's own database


def ensure(base: Path | None = None) -> list[Path]:
    """Create every layout directory (idempotent) and return them.
    `base` replaces ENGINE_DIR, so tests can build the layout in a temp dir."""
    root = Path(base) if base is not None else ENGINE_DIR
    made = [root / rel for rel in _LAYOUT.values()]
    for d in made:
        d.mkdir(parents=True, exist_ok=True)
    return made


# ── free space ────────────────────────────────────────────────────────────────

class SpaceStatus(Enum):
    OK     = "ok"        # enough room for anything
    WARN   = "warn"      # below the warning line: say so, keep going
    REFUSE = "refuse"    # below the floor: batch parsing must not start


def _existing_ancestor(path: Path) -> Path:
    """disk_usage needs a path that exists; walk up until one does."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_space_gb(path: Path | None = None) -> float:
    """Free space in GB (10**9 bytes) on the volume holding `path` (default ENGINE_DIR)."""
    usage = shutil.disk_usage(_existing_ancestor(path if path is not None else ENGINE_DIR))
    return usage.free / 1e9


def check_space(min_warn_gb: float = 100, min_refuse_gb: float = 20,
                path: Path | None = None) -> SpaceStatus:
    """P0.3.2: WARN below `min_warn_gb`, REFUSE below `min_refuse_gb`."""
    if not 0 <= min_refuse_gb <= min_warn_gb:
        raise ValueError(f"need 0 <= min_refuse_gb <= min_warn_gb, got {min_refuse_gb}, {min_warn_gb}")
    free = free_space_gb(path)
    if free < min_refuse_gb:
        return SpaceStatus.REFUSE
    if free < min_warn_gb:
        return SpaceStatus.WARN
    return SpaceStatus.OK
