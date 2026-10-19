"""
Shared bits for the bench scripts (P0.6.1). Stdlib only.

    from _bench import timed, write
    stages = {}
    with timed("parse", stages):
        ...
    write(path, stages, extra={"demo": name})

`timed` adds the elapsed wall-clock seconds to `stages[stage]` (accumulating if the
same stage is timed twice). Without an explicit dict it uses the module-level STAGES.
"""
import json
import platform
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STAGES: dict[str, float] = {}


@contextmanager
def timed(stage: str, stages: dict[str, float] | None = None):
    target = STAGES if stages is None else stages
    t0 = time.perf_counter()
    try:
        yield
    finally:
        target[stage] = target.get(stage, 0.0) + (time.perf_counter() - t0)


def write(path: Path, stages: dict[str, float], extra: dict | None = None) -> Path:
    """Write a bench result JSON: stages, their total, host info, `extra`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname":   socket.gethostname(),
        "platform":   platform.platform(),
        "python":     platform.python_version(),
        "stages":     {k: round(float(v), 4) for k, v in stages.items()},
        "total":      round(float(sum(stages.values())), 4),
        "extra":      extra or {},
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return path
