"""
Parquet layout (P5.1): where per-match artifacts live and the catalog over them.

    data/engine/<kind>/<map_version>/<yyyy-mm>/<match_key>.parquet   kind: states|actions|windows
    data/engine/<kind>/<map_version>/<yyyy-mm>/manifest.json          files + schema_version + config_hash
    data/engine/catalog.parquet                                       one row per match

    from engine.data import layout
    p = layout.states_path("de_mirage@unknown", "2026-09", "1-abc")
    layout.write_dir_manifest(p.parent, schema_version=1, config_hash=h)
    layout.find("states", map_version="de_mirage@unknown", month="2026-09")
    layout.rebuild_catalog(tiers={"1-abc": "faceit_10"})

`base` everywhere replaces `paths.ENGINE_DIR` (tests use a temp dir). pyarrow is
imported lazily, only by the catalog functions.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from engine import paths

KINDS           = ("states", "actions", "windows")
DIR_MANIFEST    = "manifest.json"
CATALOG_NAME    = "catalog.parquet"
LAYOUT_VERSION  = 1
CATALOG_COLUMNS = ("match_key", "tier", "map_version", "month", "n_rounds", "n_states",
                   "n_actions", "states_path", "actions_path", "schema_version", "built_at")

_PART_RE  = re.compile(r"^[A-Za-z0-9._@-]+$")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _root(base: Path | None) -> Path:
    return Path(base) if base is not None else paths.ENGINE_DIR


def _part(name: str, value: str) -> str:
    if not isinstance(value, str) or not _PART_RE.match(value) or value in (".", ".."):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


# ── paths ─────────────────────────────────────────────────────────────────────

def artifact_path(kind: str, map_version: str, month: str, match_key: str,
                  base: Path | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if not isinstance(month, str) or not _MONTH_RE.match(month):
        raise ValueError(f"month must be yyyy-mm, got {month!r}")
    return (_root(base) / kind / _part("map_version", map_version) / month
            / f"{_part('match_key', match_key)}.parquet")


def states_path(map_version: str, month: str, match_key: str, base: Path | None = None) -> Path:
    return artifact_path("states", map_version, month, match_key, base)


def actions_path(map_version: str, month: str, match_key: str, base: Path | None = None) -> Path:
    return artifact_path("actions", map_version, month, match_key, base)


def windows_path(map_version: str, month: str, match_key: str, base: Path | None = None) -> Path:
    return artifact_path("windows", map_version, month, match_key, base)


def find(kind: str, map_version: str | None = None, month: str | None = None,
         base: Path | None = None) -> list[Path]:
    """Every artifact file of `kind`, optionally narrowed to a map version and month."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    mv = _part("map_version", map_version) if map_version is not None else "*"
    mo = month if month is not None else "*"
    if month is not None and not _MONTH_RE.match(month):
        raise ValueError(f"month must be yyyy-mm, got {month!r}")
    return sorted((_root(base) / kind).glob(f"{mv}/{mo}/*.parquet"))


# ── per-directory manifest ────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def write_dir_manifest(directory: Path, *, schema_version: int, config_hash: str | None) -> dict:
    """(Re)write `<directory>/manifest.json` listing the parquet files present now,
    with the builder's schema_version and config_hash. Atomic replace. A file whose
    size and mtime_ns equal the previous manifest's entry keeps its recorded sha256
    instead of being re-hashed (one write per match no longer re-hashes the whole
    month directory, LOW-34)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    try:
        prev = {f["name"]: f for f in (read_dir_manifest(d) or {}).get("files", []) if isinstance(f, dict)}
    except (ValueError, OSError):
        prev = {}
    files = []
    for f in sorted(d.glob("*.parquet")):
        st  = f.stat()
        old = prev.get(f.name) or {}
        same = old.get("bytes") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns and old.get("sha256")
        files.append({"name": f.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                      "sha256": old["sha256"] if same else _sha256(f)})
    m = {
        "layout_version": LAYOUT_VERSION,
        "schema_version": schema_version,
        "config_hash":    config_hash,
        "updated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files":          files,
    }
    tmp = d / (DIR_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(d / DIR_MANIFEST)
    return m


def read_dir_manifest(directory: Path) -> dict | None:
    p = Path(directory) / DIR_MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ── catalog ───────────────────────────────────────────────────────────────────

def _parquet_counts(path: Path) -> tuple[int | None, int | None]:
    """(rows, distinct round_num) of a parquet file; round count None without the column."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    meta = pq.read_metadata(path)
    n_rounds = None
    if "round_num" in pq.read_schema(path).names:
        col = pq.read_table(path, columns=["round_num"]).column("round_num")
        n_rounds = len(pc.unique(col.drop_null()))
    return meta.num_rows, n_rounds


def catalog_rows(base: Path | None = None, tiers: dict[str, str] | None = None) -> list[dict]:
    """One row per match that has a states file; actions joined when present."""
    root    = _root(base)
    tiers   = tiers or {}
    actions = {(p.parent.parent.name, p.parent.name, p.stem): p for p in find("actions", base=base)}
    rows = []
    for sp in find("states", base=base):
        mv, mo, mk = sp.parent.parent.name, sp.parent.name, sp.stem
        n_states, n_rounds = _parquet_counts(sp)
        ap = actions.get((mv, mo, mk))
        n_actions = _parquet_counts(ap)[0] if ap is not None else None
        man = read_dir_manifest(sp.parent) or {}
        rows.append({
            "match_key":      mk,
            "tier":           tiers.get(mk),
            "map_version":    mv,
            "month":          mo,
            "n_rounds":       n_rounds,
            "n_states":       n_states,
            "n_actions":      n_actions,
            "states_path":    sp.relative_to(root).as_posix(),
            "actions_path":   ap.relative_to(root).as_posix() if ap is not None else None,
            "schema_version": man.get("schema_version"),
            "built_at":       datetime.fromtimestamp(sp.stat().st_mtime, timezone.utc)
                                      .isoformat(timespec="seconds"),
        })
    return rows


def rebuild_catalog(base: Path | None = None, tiers: dict[str, str] | None = None) -> Path:
    """Rebuild `<base>/catalog.parquet` from the files on disk. `tiers` maps match_key
    to tier; when None it is read from engine.db (`matches.tier`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if tiers is None:
        from engine import db
        tiers = {r["match_key"]: r["tier"] for r in db.list_matches()}
    rows   = catalog_rows(base, tiers)
    schema = pa.schema([
        ("match_key", pa.string()), ("tier", pa.string()), ("map_version", pa.string()),
        ("month", pa.string()), ("n_rounds", pa.int64()), ("n_states", pa.int64()),
        ("n_actions", pa.int64()), ("states_path", pa.string()), ("actions_path", pa.string()),
        ("schema_version", pa.int64()), ("built_at", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    out   = _root(base) / CATALOG_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp   = out.with_name(out.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    return out


def read_catalog(base: Path | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(_root(base) / CATALOG_NAME).to_pylist()
