"""
The pinned golden demo is on disk and matches its recorded hash (P0.7.1).

    .venv\\Scripts\\python.exe test\\engine\\test_golden_present.py

Exit 2 when no golden demo is pinned yet (demo_sha256 null) or the file under
`data/engine/demos/golden/` is missing; 0 on hash match; 1 on mismatch.
GOLDEN.json may name the file in `demo_file`; otherwise the directory must hold
exactly one `.dem`.
"""
import json
import sys
from pathlib import Path

from _harness import check, section, summary

from engine import paths
from engine.manifest import sha256_file

GOLDEN_JSON = Path(__file__).resolve().parent / "golden" / "GOLDEN.json"


def locate(meta: dict) -> Path | None:
    if meta.get("demo_file"):
        p = paths.DEMOS_GOLDEN_DIR / meta["demo_file"]
        return p if p.is_file() else None
    dems = sorted(paths.DEMOS_GOLDEN_DIR.glob("*.dem")) if paths.DEMOS_GOLDEN_DIR.is_dir() else []
    return dems[0] if len(dems) == 1 else None


def main() -> int:
    section("golden demo")
    meta = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    expected = meta.get("demo_sha256")
    if not expected:
        print(f"  no golden demo pinned yet (demo_sha256 is null in {GOLDEN_JSON.name}); "
              f"choose one per P0.7.1. Note: {meta.get('note', '')}")
        return 2
    demo = locate(meta)
    if demo is None:
        print(f"  golden demo missing: expected it under {paths.DEMOS_GOLDEN_DIR} "
              f"(demo_file={meta.get('demo_file')!r}); copy it there to run L2 tests")
        return 2
    actual = sha256_file(demo)
    check("golden demo sha256 matches GOLDEN.json", actual == expected,
          f"{demo.name}: {actual[:12]} vs {expected[:12]}")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
