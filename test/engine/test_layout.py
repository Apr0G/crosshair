"""
Parquet layout (P5.1): paths by map version and month, per-directory manifests, and a
catalog that equals the files on disk.

    .venv\\Scripts\\python.exe test\\engine\\test_layout.py

Writes a synthetic match in a temp dir; data/ is never touched.
"""
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

from engine.data import layout

MV = "de_mirage@unknown"


def _write(path: Path, rounds: list[int]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"round_num": rounds, "x": [float(i) for i in range(len(rounds))]}), path)


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="ch_layout_test_"))

    section("paths")
    sp = layout.states_path(MV, "2026-09", "1-abc", base)
    check("states/<map_version>/<yyyy-mm>/<match_key>.parquet",
          sp == base / "states" / MV / "2026-09" / "1-abc.parquet", str(sp))
    check("actions and windows mirror it",
          layout.actions_path(MV, "2026-09", "1-abc", base) == base / "actions" / MV / "2026-09" / "1-abc.parquet"
          and layout.windows_path(MV, "2026-09", "1-abc", base).parts[-4] == "windows")
    for args, why in [((MV, "2026-13", "k"), "bad month"), ((MV, "2026-9", "k"), "unpadded month"),
                      (("../x", "2026-09", "k"), "path in map_version"), ((MV, "2026-09", "a/b"), "path in match_key"),
                      ((MV, "2026-09", ".."), "dot-dot key")]:
        try:
            layout.states_path(*args, base=base)
            check(f"refuses {why}", False)
        except ValueError:
            check(f"refuses {why}", True)

    section("write a synthetic match, find it by map and month")
    _write(sp, [1, 1, 2, 2, 3])
    _write(layout.actions_path(MV, "2026-09", "1-abc", base), [1, 2])
    other = layout.states_path("de_nuke@unknown", "2026-08", "2-def", base)
    _write(other, [1, 2])
    check("found by map + month", layout.find("states", MV, "2026-09", base) == [sp])
    check("found by map only", layout.find("states", MV, base=base) == [sp])
    check("all states files", layout.find("states", base=base) == sorted([sp, other]))
    check("nothing for another month", layout.find("states", MV, "2026-07", base) == [])

    m = layout.write_dir_manifest(sp.parent, schema_version=3, config_hash="abc123")
    got = layout.read_dir_manifest(sp.parent)
    check("dir manifest lists files + schema_version + config_hash",
          got == m and got["schema_version"] == 3 and got["config_hash"] == "abc123"
          and [f["name"] for f in got["files"]] == ["1-abc.parquet"] and len(got["files"][0]["sha256"]) == 64)
    check("no manifest -> None", layout.read_dir_manifest(other.parent) is None)

    section("catalog")
    out  = layout.rebuild_catalog(base, tiers={"1-abc": "faceit_10"})
    rows = {r["match_key"]: r for r in layout.read_catalog(base)}
    check("catalog.parquet under the base", out == base / "catalog.parquet" and out.is_file())
    check("columns exactly P5.1.2", tuple(rows["1-abc"]) == layout.CATALOG_COLUMNS, str(tuple(rows["1-abc"])))
    r = rows["1-abc"]
    check("row values match the files", r["tier"] == "faceit_10" and r["map_version"] == MV and r["month"] == "2026-09"
          and r["n_states"] == 5 and r["n_rounds"] == 3 and r["n_actions"] == 2 and r["schema_version"] == 3
          and r["states_path"] == f"states/{MV}/2026-09/1-abc.parquet"
          and r["actions_path"] == f"actions/{MV}/2026-09/1-abc.parquet" and r["built_at"], str(r))
    o = rows["2-def"]
    check("unknowns are None (tier, actions, schema)", o["tier"] is None and o["n_actions"] is None
          and o["actions_path"] is None and o["schema_version"] is None, str(o))
    check("catalog rows equal states files on disk", set(rows) == {p.stem for p in layout.find("states", base=base)})
    other.unlink()
    layout.rebuild_catalog(base, tiers={})
    check("rebuild reflects a deleted file", [r["match_key"] for r in layout.read_catalog(base)] == ["1-abc"])
    return summary()


if __name__ == "__main__":
    sys.exit(main())
