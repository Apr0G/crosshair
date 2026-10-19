"""
No signed URL ever reaches a log file or the console (P0.2.4, CLAUDE.md redaction rule).

    .venv\\Scripts\\python.exe test\\engine\\test_log_redaction.py

Stdlib + engine only; writes into a temp dir, never into data/.
"""
import io
import json
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

from engine import log

SECRET    = "SUPERSECRETSIGNATURE123"
TOKEN     = "APIKEYTOKEN987"
HOST      = "https://demos-europe-central-1.faceit-cdn.net"
DEMO_PATH = "/cs2/1-0a1b2c3d-e4f5-6789-abcd-ef0123456789-1-1.dem.zst"
SIGNED    = f"{HOST}{DEMO_PATH}?X-Amz-Signature={SECRET}&X-Amz-Expires=3600"
# urllib3 / requests style: path?query with no scheme or host
URLLIB3   = f'{HOST}:443 "GET {DEMO_PATH}?X-Amz-Signature={SECRET} HTTP/1.1" 403 0'
REQUESTS  = (f"HTTPSConnectionPool(host='demos.faceit-cdn.net', port=443): Max retries "
             f"exceeded with url: {DEMO_PATH}?sig={SECRET} (Caused by ...)")


def test_redact_text() -> None:
    section("redact_text")
    out = log.redact_text(f"downloading {SIGNED} now", SIGNED)
    check("given URL loses its query", SECRET not in out, out)
    check("path is kept for debugging", DEMO_PATH in out)
    check("marked as redacted", "?<redacted>" in out)
    check("URL found without being passed", SECRET not in log.redact_text(f"x {SIGNED} y"))
    check("urllib3 path?query redacted", SECRET not in log.redact_text(URLLIB3), log.redact_text(URLLIB3))
    check("requests 'with url: path?query' redacted", SECRET not in log.redact_text(REQUESTS))
    check(".dem.gz / .dem variants redacted",
          all(SECRET not in log.redact_text(f"/a/b{ext}?s={SECRET}") for ext in (".dem", ".dem.gz", ".DEM")))
    check("Bearer token redacted", TOKEN not in log.redact_text(f"Authorization: Bearer {TOKEN}"))
    upper = log.redact_text(f"x HTTPS://demos.faceit-cdn.net{DEMO_PATH}?sig={SECRET} y")
    check("upper-case scheme redacted (LOW-02/LOW-74)", SECRET not in upper and "?<redacted>" in upper, upper)
    check("mixed-case Http:// redacted", SECRET not in log.redact_text(f"Http://h/a/b.json?token={SECRET}"))
    for text in (f'"GET /api/v4/matches/1-abc?token={SECRET} HTTP/1.1" 403',
                 f"Max retries exceeded with url: /download/v2/x.zip?X-Amz-Signature={SECRET} (Caused by",
                 f"/resolve/main/2026-08/a.parquet?download=true&sig={SECRET}"):
        out = log.redact_text(text)
        check(f"scheme-less non-.dem path?query redacted: {text.split('?')[0][-28:]!r}",
              SECRET not in out and "?<redacted>" in out, out)
    check("non-.dem path keeps its path", log.redact_text(f"/a/b.zip?s={SECRET}") == "/a/b.zip?<redacted>")
    check("question mark in prose untouched", log.redact_text("is it /tmp/x? yes") == "is it /tmp/x? yes")
    check("already-redacted text is stable",
          log.redact_text(log.redact_text(SIGNED)) == log.redact_text(SIGNED), log.redact_text(SIGNED))
    check("plain URL without query unchanged",
          log.redact_text(f"see {HOST}{DEMO_PATH} ok") == f"see {HOST}{DEMO_PATH} ok")
    check("None -> empty string", log.redact_text(None) == "")


def test_log_file(tmp: Path) -> None:
    section("log file and console")
    console = io.StringIO()
    lg = log.get_logger(kind="test", log_dir=tmp, stream=console)
    lg.add_secret(SIGNED)
    lg.info(f"fetching {SIGNED}")
    lg.info("fetch", url=SIGNED, nested={"urls": [SIGNED], "msg": URLLIB3})
    lg.warning(URLLIB3)
    lg.warning(REQUESTS)
    try:
        raise RuntimeError(f"403 for url: {SIGNED}")
    except RuntimeError as e:
        lg.error("download failed", exc=e)
    lg.close()

    text = lg.path.read_text(encoding="utf-8")
    lines = text.splitlines()
    check("log file named <run_id>.jsonl", lg.path.name == f"{lg.run_id}.jsonl")
    check("one JSON record per call", len(lines) == 5, f"{len(lines)} lines")
    check("every line is valid JSON", all(json.loads(l) for l in lines))
    check("secret never in the log file", SECRET not in text)
    check("secret never on the console", SECRET not in console.getvalue())
    check("records carry run_id and level",
          all(json.loads(l)["run_id"] == lg.run_id and "level" in json.loads(l) for l in lines))
    check("exception type recorded", json.loads(lines[-1]).get("exc_type") == "RuntimeError")


def main() -> int:
    test_redact_text()
    with tempfile.TemporaryDirectory() as d:
        test_log_file(Path(d))
    return summary()


if __name__ == "__main__":
    sys.exit(main())
