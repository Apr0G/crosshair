"""
Multiprocess parse of a directory (P2.6.1): one demo -> `data/engine/parsed/<match_key>/`
with one parquet per table (plus `meta.parquet`) and `manifest.json`; one retry;
failures (exception type + redacted message) appended to `parsed/_failures.jsonl`;
already-parsed matches skipped unless `--force`.

    .venv\\Scripts\\python.exe -m engine.parse.worker <demo_dir> [--out DIR] [--procs N]
                                                     [--force] [--smoke]

`match_key` is the demo filename stem. Output is validated with
`RawTables.from_parsed(...).validate()` before anything is written. A match counts as parsed when its
`manifest.json` exists; the manifest is written last, so a crash mid-write leaves a
match that is simply parsed again. `--smoke` runs only `header_info` +
`quarantine_check` (P1.4.1) per demo and writes nothing.

The quarantine rules (P1.4.1) run on the header BEFORE every parse, not only under
`--smoke`: a demo that fails one gets status `quarantined` with the reason and nothing
is written (2026-10-19 audit, HIGH-01). Moving the file under `demos/quarantine/` and
setting the `matches` row is done by `engine.cli parse`, which owns the DB; this module
only reports.

The parser and the header reader are injectable (`run(..., parser=f, header_fn=h)`,
f(path) -> (tables, meta), h(path, with_roster=True) -> meta) so tests can use fakes;
both must be module-level functions when `procs > 1` (pickled).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from engine import manifest, paths
from engine.log import redact_text
from engine.parse import awpy3_parser
from engine.parse.schema import RawTables

FAILURES_NAME = "_failures.jsonl"
DEMO_GLOB     = "*.dem"
MAX_ATTEMPTS  = 2                 # first try + one retry

Parser   = Callable[[Path], tuple[dict, dict]]
HeaderFn = Callable[..., dict]


# ── helpers ───────────────────────────────────────────────────────────────────

def match_key(path: Path) -> str:
    return Path(path).stem


def discover(demo_dir: Path) -> list[Path]:
    return sorted(p for p in Path(demo_dir).glob(DEMO_GLOB) if p.is_file())


def default_procs() -> int:
    return max(1, (os.cpu_count() or 3) - 2)


def is_parsed(out_dir: Path, key: str) -> bool:
    return (Path(out_dir) / key / manifest.MANIFEST_NAME).is_file()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _map_asset_hashes(map_name) -> tuple[dict | None, str | None]:
    """Asset hashes for the manifest; a map without cached assets is recorded, not fatal."""
    if not map_name:
        return None, "meta has no map"
    try:
        from engine.maps import assets
        return assets.asset_hashes(map_name), None
    except Exception as exc:                      # recorded in the manifest, never hidden
        return None, redact_text(f"{type(exc).__name__}: {exc}")


def _write(path: Path, out: Path, tables: dict, meta: dict, timings: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    old = out / manifest.MANIFEST_NAME
    if old.exists():
        old.unlink()                              # --force: invalid until the new manifest lands
    names = []
    t0 = time.perf_counter()
    for name, df in tables.items():
        df.write_parquet(out / f"{name}.parquet")
        names.append(f"{name}.parquet")
    awpy3_parser.meta_frame(meta).write_parquet(out / "meta.parquet")
    names.append("meta.parquet")
    timings["write"] = time.perf_counter() - t0
    hashes, asset_err = _map_asset_hashes(meta.get("map"))
    extra = {"match_key": match_key(path), "meta": meta}
    if asset_err:
        extra["map_assets_error"] = asset_err
    manifest.write(out, "parse", inputs=[path], outputs=names, extra=extra,
                   timings=timings, map_assets=hashes)


# ── one demo ──────────────────────────────────────────────────────────────────

def quarantine_reason(path: Path, header_fn: HeaderFn | None = None) -> str | None:
    """P1.4.1 on the header + roster: the failing rule, `header_unreadable` when the
    header cannot be read, None when the demo passes."""
    header_fn = header_fn or awpy3_parser.header_info
    try:
        info = header_fn(path, with_roster=True)
    except Exception:
        return "header_unreadable"
    return awpy3_parser.quarantine_check(info)


def parse_one(path: Path, out_dir: Path, parser: Parser | None = None, force: bool = False,
              header_fn: HeaderFn | None = None) -> dict:
    """Parse and write one demo; never raises. Returns a result dict with
    status parsed | skipped | quarantined | failed. Quarantined: the header failed a
    P1.4.1 rule (`reason`), nothing was parsed or written."""
    path, out_dir = Path(path), Path(out_dir)
    key = match_key(path)
    if not force and is_parsed(out_dir, key):
        return {"match_key": key, "status": "skipped"}
    reason = quarantine_reason(path, header_fn)
    if reason is not None:
        return {"match_key": key, "status": "quarantined", "reason": reason, "path": str(path)}
    parser = parser or awpy3_parser.parse
    last: BaseException | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            t0 = time.perf_counter()
            tables, meta = parser(path)
            raw = RawTables.from_parsed(tables, meta)
            raw.validate(strict=True)            # SchemaError -> retried, then logged
            timings = {"parse": time.perf_counter() - t0}
            _write(path, out_dir / key, raw.frames(), meta, timings)
            return {"match_key": key, "status": "parsed", "attempts": attempt,
                    "seconds": round(sum(timings.values()), 3)}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:              # retried once, then reported; a pyo3 PanicException
            last = exc                            # is a BaseException and must not escape (MED-11)
    return {"match_key": key, "status": "failed", "attempts": MAX_ATTEMPTS, "path": str(path),
            "exc_type": type(last).__name__, "error": redact_text(str(last))}


def smoke_one(path: Path, header_fn: HeaderFn | None = None) -> dict:
    """P1.4.1 smoke test: header + roster readable and the quarantine rules pass."""
    header_fn = header_fn or awpy3_parser.header_info
    key = match_key(path)
    try:
        info = header_fn(path, with_roster=True)
    except Exception as exc:
        return {"match_key": key, "status": "quarantine", "reason": "header_unreadable",
                "exc_type": type(exc).__name__, "error": redact_text(str(exc)), "path": str(path)}
    reason = awpy3_parser.quarantine_check(info)
    brief = {k: info.get(k) for k in ("map", "tick_rate", "n_rounds", "side_counts", "duration_s")}
    return {"match_key": key, "status": "ok" if reason is None else "quarantine",
            "reason": reason, "info": brief, "path": str(path)}


def _append_failures(out_dir: Path, results: list[dict]) -> None:
    failed = [r for r in results if r["status"] == "failed"]
    if not failed:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / FAILURES_NAME, "a", encoding="utf-8") as fh:
        for r in failed:
            rec = {"ts": _now(), **{k: r.get(k) for k in ("match_key", "path", "exc_type", "error", "attempts")}}
            fh.write(redact_text(json.dumps(rec, ensure_ascii=False)) + "\n")


# ── directory ─────────────────────────────────────────────────────────────────

def run(demo_dir: Path, *, out_dir: Path | None = None, procs: int | None = None,
        force: bool = False, smoke: bool = False, parser: Parser | None = None,
        header_fn: HeaderFn | None = None) -> dict:
    """Parse (or smoke-test) every `*.dem` in `demo_dir`. Returns
    {"results": [...], "counts": {status: n}}."""
    out_dir = Path(out_dir) if out_dir is not None else paths.PARSED_DIR
    demos = discover(demo_dir)
    procs = max(1, procs if procs is not None else default_procs())
    if smoke:
        jobs, args = smoke_one, [(p, header_fn) for p in demos]
    else:
        jobs, args = parse_one, [(p, out_dir, parser, force, header_fn) for p in demos]
    if procs == 1 or len(demos) <= 1:
        results = [jobs(*a) for a in args]
    else:
        # submit + as_completed: a worker that dies hard (Rust abort) breaks the pool and
        # fails every pending future; each becomes a `failed` result naming its demo, so
        # `_failures.jsonl` is still written (MED-11, 2026-10-19).
        results = []
        with ProcessPoolExecutor(max_workers=min(procs, len(demos))) as pool:
            futures = {pool.submit(jobs, *a): a[0] for a in args}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    results.append(fut.result())
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    results.append({"match_key": match_key(p), "status": "failed", "attempts": 1,
                                    "path": str(p), "exc_type": type(exc).__name__,
                                    "error": redact_text(str(exc) or "worker process died")})
        results.sort(key=lambda r: r["match_key"])
    if not smoke:
        _append_failures(out_dir, results)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"results": results, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse a directory of demos (P2.6.1).")
    ap.add_argument("demo_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help=f"output root (default {paths.PARSED_DIR})")
    ap.add_argument("--procs", type=int, default=None, help="worker processes (default cpu_count - 2)")
    ap.add_argument("--force", action="store_true", help="re-parse matches that are already parsed")
    ap.add_argument("--smoke", action="store_true", help="header + quarantine check only; writes nothing")
    a = ap.parse_args(argv)
    if not a.demo_dir.is_dir():
        print(f"not a directory: {a.demo_dir}", file=sys.stderr)
        return 2
    out = run(a.demo_dir, out_dir=a.out, procs=a.procs, force=a.force, smoke=a.smoke)
    for r in out["results"]:
        print(json.dumps(r, ensure_ascii=False, default=str))
    print(f"done: {out['counts']}")
    return 1 if out["counts"].get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
