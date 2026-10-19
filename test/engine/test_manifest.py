"""
Run manifests: round-trip, required keys, input hashing, schema refusal (P0.5).

    .venv\\Scripts\\python.exe test\\engine\\test_manifest.py

Works in a temp dir. Does not import torch (gpu is read only if already imported).
"""
import json
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

from engine import config, manifest

SECRET = "SUPERSECRETSIGNATURE123"


def test_round_trip(tmp: Path) -> None:
    section("write / read")
    demo = tmp / "in" / "match.dem"
    demo.parent.mkdir()
    demo.write_bytes(b"PBDEMS2\x00" + b"\x01" * 1000)
    run = tmp / "run1"
    (run).mkdir()
    (run / "kills.parquet").write_bytes(b"x" * 10)
    url = f"https://demos.faceit-cdn.net/cs2/a.dem.zst?X-Amz-Signature={SECRET}"
    m = manifest.write(run, "parse", inputs=[demo], outputs=["kills.parquet", "missing.parquet"],
                       extra={"match_key": "m1", "demo_url": url}, timings={"parse": 1.5})
    back = manifest.read(run)
    check("manifest.json written", (run / "manifest.json").is_file())
    check("round-trip equal", back == m)
    missing = [k for k in manifest.REQUIRED_KEYS if k not in back]
    check("all required keys present", not missing, f"missing {missing}")
    check("schema_version recorded", back["schema_version"] == manifest.SCHEMA_VERSION)
    check("config_hash matches config", back["config_hash"] == config.config_hash(config.load()))
    check("input sha256 recorded", back["inputs"][0]["sha256"] == manifest.sha256_file(demo))
    check("input size recorded", back["inputs"][0]["bytes"] == demo.stat().st_size)
    check("output relative to run dir", back["outputs"][0] == {"path": "kills.parquet", "bytes": 10})
    check("absent output -> bytes null", back["outputs"][1]["bytes"] is None)
    check("timings recorded", back["timings"] == {"parse": 1.5})
    check("every Appendix E package listed", set(manifest.PACKAGES) <= set(back["packages"]))
    check("installed package has a version", back["packages"]["numpy"] is not None)
    check("git_dirty is bool or null", back["git_dirty"] in (True, False, None))
    check("gpu null when torch not imported", "torch" in sys.modules or back["gpu"] is None)
    check("awpy pin from config", back["awpy_data_release"] == config.load().data.awpy_data_release)
    check("signed URL in extra redacted", SECRET not in (run / "manifest.json").read_text(encoding="utf-8"))
    check("no temp file left", not (run / "manifest.json.tmp").exists())

    section("input hashing")
    before = back["inputs"][0]["sha256"]
    demo.write_bytes(b"PBDEMS2\x00" + b"\x02" * 1000)
    after = manifest.write(run, "parse", inputs=[demo])["inputs"][0]["sha256"]
    check("hash changes when an input file changes", before != after)
    try:
        manifest.write(run, "parse", inputs=[tmp / "nope.dem"])
        check("missing input refused", False)
    except FileNotFoundError:
        check("missing input refused", True)


def test_unknown_schema_refused(tmp: Path) -> None:
    section("unknown_schema_refused")
    for label, value in (("future version", 999), ("missing version", None), ("bool version", True)):
        run = tmp / f"bad_{label.replace(' ', '_')}"
        run.mkdir()
        doc = {"kind": "parse"} if value is None else {"schema_version": value, "kind": "parse"}
        (run / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
        try:
            manifest.read(run)
            check(f"{label} refused", False)
        except manifest.ManifestSchemaError as e:
            check(f"{label} refused", "schema_version" in str(e), str(e)[:80])
    try:
        manifest.read(tmp / "no_such_run")
        check("missing manifest raises FileNotFoundError", False)
    except FileNotFoundError:
        check("missing manifest raises FileNotFoundError", True)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        test_round_trip(Path(d))
        test_unknown_schema_refused(Path(d))
    return summary()


if __name__ == "__main__":
    sys.exit(main())
