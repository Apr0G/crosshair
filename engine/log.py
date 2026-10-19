"""
Structured run logging with redaction (P0.2.4).

    from engine import log
    lg = log.get_logger(kind="parse")          # data/engine/logs/<run_id>.jsonl + console
    lg.add_secret(signed_url)                  # optional: scrub this exact string too
    lg.info("downloaded", url=signed_url, bytes=n)
    lg.error("download failed", exc=err)
    lg.close()

Every record, file and console alike, passes through `redact_text` before it is
written. That covers: full URLs (query string dropped via
`engine.sources.scraper.redact_url`, the one implementation of the rule), any
`.dem` / `.dem.zst` / `.dem.gz` path followed by `?query` (urllib3 and requests log
`path?query` without the scheme and host, so URL matching alone misses them), and
`Bearer <token>` headers. Secrets registered with `add_secret` are replaced first.
"""
import json
import re
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from engine import paths
from engine.sources.scraper import redact_url

# ── redaction ─────────────────────────────────────────────────────────────────

REDACTED        = "<redacted>"
_URL_RE         = re.compile(r"https?://[^\s'\"<>]+")
_DEMO_QUERY_RE  = re.compile(r"(\.dem(?:\.zst|\.gz)?)\?(?!<redacted>)[^\s'\"<>]*", re.IGNORECASE)
_BEARER_RE      = re.compile(r"(Bearer\s+)[^\s'\"<>]+", re.IGNORECASE)


def _redact_match(m: re.Match) -> str:
    # Already-redacted text ("...path?<redacted>") must not be re-processed: the regex
    # stops at "<", and redact_url would then drop the dangling "?".
    if m.string.startswith(REDACTED, m.end()):
        return m.group(0)
    return redact_url(m.group(0))


def redact_text(text, *urls: str) -> str:
    """Return `text` with every credential-bearing part removed.

    Each url in `urls` is replaced by its redacted form wherever it appears; then any
    remaining http(s) URL loses its query string, any `.dem[.zst|.gz]?query` loses
    the query, and `Bearer <token>` loses the token."""
    if text is None:
        return ""
    s = str(text)
    for u in urls:
        if u:
            s = s.replace(str(u), redact_url(str(u)))
    s = _URL_RE.sub(_redact_match, s)
    s = _DEMO_QUERY_RE.sub(lambda m: f"{m.group(1)}?{REDACTED}", s)
    s = _BEARER_RE.sub(lambda m: f"{m.group(1)}{REDACTED}", s)
    return s


def redact_obj(obj, *urls: str):
    """Redact every string inside a JSON-like structure; other leaves become str."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {redact_text(k, *urls): redact_obj(v, *urls) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [redact_obj(v, *urls) for v in obj]
    return redact_text(obj, *urls)


# ── run logger ────────────────────────────────────────────────────────────────

LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def new_run_id(kind: str = "run") -> str:
    """`<UTC timestamp>-<kind>-<6 hex>`: sortable by time, unique per run."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{kind}-{uuid.uuid4().hex[:6]}"


class RunLogger:
    """JSON-lines file `<log_dir>/<run_id>.jsonl` plus a console line per record."""

    def __init__(self, run_id: str | None = None, *, kind: str = "run",
                 log_dir: Path | None = None, console: bool = True,
                 level: str = "info", stream: IO[str] | None = None):
        self.run_id   = run_id or new_run_id(kind)
        self.path     = Path(log_dir if log_dir is not None else paths.LOGS_DIR) / f"{self.run_id}.jsonl"
        self.console  = console
        self.level    = LEVELS[level]
        self._stream  = stream
        self._secrets: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def add_secret(self, *urls: str) -> None:
        """Register exact strings (signed URLs) to scrub from every later record."""
        self._secrets.extend(u for u in urls if u)

    def log(self, level: str, msg: str, exc: BaseException | None = None, **fields) -> dict:
        record = {
            "ts":     datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "level":  level,
            "msg":    msg,
        }
        if fields:
            record["fields"] = fields
        if exc is not None:
            record["exc_type"]  = type(exc).__name__
            record["exc"]       = str(exc)
            record["traceback"] = "".join(traceback.format_exception(exc))
        record = redact_obj(record, *self._secrets)
        if LEVELS[level] >= self.level:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
            if self.console:
                self._echo(record)
        return record

    def _echo(self, record: dict) -> None:
        extra = " ".join(f"{k}={v}" for k, v in record.get("fields", {}).items())
        line  = f"[{record['level'].upper():<7}] {record['msg']}"
        if extra:
            line += f"  {extra}"
        if "exc_type" in record:
            line += f"  ({record['exc_type']}: {record['exc']})"
        print(line, file=self._stream or sys.stderr, flush=True)

    def debug(self, msg: str, **fields) -> dict:
        return self.log("debug", msg, **fields)

    def info(self, msg: str, **fields) -> dict:
        return self.log("info", msg, **fields)

    def warning(self, msg: str, **fields) -> dict:
        return self.log("warning", msg, **fields)

    def error(self, msg: str, exc: BaseException | None = None, **fields) -> dict:
        return self.log("error", msg, exc=exc, **fields)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def get_logger(run_id: str | None = None, **kwargs) -> RunLogger:
    """Convenience constructor; see RunLogger for keyword arguments."""
    return RunLogger(run_id, **kwargs)
