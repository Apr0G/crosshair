"""
Run manifests (P0.5): every artifact directory under `data/engine/` carries a
`manifest.json` saying exactly what produced it.

    from engine import manifest
    manifest.write(run_dir, "parse", inputs=[demo], outputs=["kills.parquet"],
                   extra={"match_key": mk}, timings={"parse": 11.2})
    m = manifest.read(run_dir)          # refuses unknown schema versions

Recorded: schema_version, kind, created_at (UTC), git sha + dirty flag, config_hash,
the awpy-data pin from config, map asset hashes (when the caller has them), package
versions for the Appendix E list (missing -> null), sha256 + size of every input file,
outputs, per-stage timings, hostname, platform, Python version, GPU name.

GPU name: read from torch only when the run has already imported torch (any model run
has), or when `probe_gpu=True`. A parse-only run never pays for importing torch just to
fill one field, and a broken torch install cannot break manifest writing. Unknown or
unavailable values are null, never 0 or "?".
"""
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from engine import config as engine_config
from engine import paths
from engine.log import redact_obj

SCHEMA_VERSION        = 1
KNOWN_SCHEMA_VERSIONS = frozenset({1})
MANIFEST_NAME         = "manifest.json"

# Appendix E distribution names (pip names, not import names).
PACKAGES = (
    "torch", "awpy", "polars", "pyarrow", "numpy", "pandas", "lightgbm", "scikit-learn",
    "faiss-cpu", "matplotlib", "pyyaml", "jsonschema", "zstandard", "requests",
)

REQUIRED_KEYS = (
    "schema_version", "kind", "created_at", "git_sha", "git_dirty", "config_hash",
    "awpy_data_release", "map_assets", "packages", "inputs", "outputs", "timings",
    "hostname", "platform", "python", "gpu", "extra",
)


class ManifestSchemaError(ValueError):
    """The manifest's schema_version is missing or unknown to this build."""


# ── collectors ────────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD sha, dirty flag); (None, None) when git or the repo is unavailable."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git("status", "--porcelain")
    return sha.strip(), (None if status is None else bool(status.strip()))


def package_versions(names: Iterable[str] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_name(probe: bool = False) -> str | None:
    """GPU name via torch, or None. Imports torch only if already loaded or `probe`."""
    if "torch" not in sys.modules and not probe:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:          # a missing or half-installed torch must not break manifests
        return None
    return None


def _describe_input(p: Path) -> dict:
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f"manifest input is not a file: {p}")
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def _describe_output(run_dir: Path, p) -> dict:
    full = Path(p) if Path(p).is_absolute() else run_dir / p
    try:
        shown = str(full.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        shown = str(full)
    return {"path": shown, "bytes": full.stat().st_size if full.is_file() else None}


# ── public API ────────────────────────────────────────────────────────────────

def build(run_dir: Path, kind: str, inputs: Iterable = (), outputs: Iterable = (),
          extra: dict | None = None, *, cfg: engine_config.Config | None = None,
          timings: dict[str, float] | None = None, map_assets: dict[str, str] | None = None,
          probe_gpu: bool = False) -> dict:
    """The manifest dict, without writing it."""
    run_dir = Path(run_dir)
    cfg     = cfg if cfg is not None else engine_config.load()
    sha, dirty = git_state()
    return {
        "schema_version":    SCHEMA_VERSION,
        "kind":              kind,
        "created_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha":           sha,
        "git_dirty":         dirty,
        "config_hash":       engine_config.config_hash(cfg),
        "awpy_data_release": cfg.data.awpy_data_release,
        "map_assets":        dict(map_assets) if map_assets else None,
        "packages":          package_versions(),
        "inputs":            [_describe_input(p) for p in inputs],
        "outputs":           [_describe_output(run_dir, p) for p in outputs],
        "timings":           {k: float(v) for k, v in (timings or {}).items()},
        "hostname":          socket.gethostname(),
        "platform":          platform.platform(),
        "python":            platform.python_version(),
        "gpu":               gpu_name(probe_gpu),
        # extra is caller-supplied and may carry a demo URL: redact before writing.
        "extra":             redact_obj(extra or {}),
    }


def write(run_dir: Path, kind: str, inputs: Iterable = (), outputs: Iterable = (),
          extra: dict | None = None, **kwargs) -> dict:
    """Build the manifest and write `<run_dir>/manifest.json`; return the dict.
    Keyword arguments: cfg, timings, map_assets, probe_gpu (see build)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    m   = build(run_dir, kind, inputs, outputs, extra, **kwargs)
    tmp = run_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(run_dir / MANIFEST_NAME)          # never leave a half-written manifest
    return m


def read(run_dir: Path) -> dict:
    """Load `<run_dir>/manifest.json`; raise ManifestSchemaError on an unknown or
    missing schema_version (P0.5.2: readers refuse artifacts they do not understand)."""
    path = Path(run_dir) / MANIFEST_NAME
    m = json.loads(path.read_text(encoding="utf-8"))
    version = m.get("schema_version") if isinstance(m, dict) else None
    if isinstance(version, bool) or version not in KNOWN_SCHEMA_VERSIONS:
        raise ManifestSchemaError(
            f"{path}: schema_version {version!r} is not known to this build "
            f"(known: {sorted(KNOWN_SCHEMA_VERSIONS)}); refusing to read it")
    return m
