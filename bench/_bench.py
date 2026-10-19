"""
Shared bits for the bench scripts (P0.6.1). Stdlib only.

    from _bench import finish, result_name, timed, write
    stages = {}
    with timed("parse", stages):
        ...
    write(paths.BENCH_DIR / result_name("gpu", "cuda"), stages, extra={"demo": name})
    finish(main())          # last line of a script that used the GPU

`timed` adds the elapsed wall-clock seconds to `stages[stage]` (accumulating if the
same stage is timed twice). Without an explicit dict it uses the module-level STAGES.
`result_name` stamps date, time and tags (device, stub) into the file name so runs on
the same day never overwrite each other; `finish` is the HIP-teardown exit of
test/engine/_fake_states.finish (LOW-38, 2026-10-19).
"""
import json
import os
import platform
import socket
import sys
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


def result_name(kind: str, *tags: str) -> str:
    """`<kind>_<yyyy-mm-dd>_<hhmmss>[_<tag>...].json` (local time)."""
    parts = [kind, datetime.now().strftime("%Y-%m-%d_%H%M%S"), *(str(t) for t in tags if t)]
    return "_".join(parts) + ".json"


def finish(code: int) -> None:
    """End a bench with `code`. After GPU use, flush and leave through os._exit: with
    torch 2.9.1 + ROCm 7.2.1 on Windows the interpreter sometimes hangs in the HIP
    runtime's teardown after everything is written (same workaround as
    test/engine/_fake_states.finish). Without torch / a GPU it is sys.exit."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)


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
