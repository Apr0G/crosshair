"""
Hugging Face archive source (P1.1.3) against a local HTTP server serving a fake tree
listing (two pages, Link-header pagination) and the files it lists.

    .venv\\Scripts\\python.exe test\\engine\\test_hf_archive.py

No network (the real-archive integration check is out of scope here). Checks: listing
filter + LFS sha256, map/month classification, wave selection and packing, sha256
verification (match, mismatch -> no file), the free-space stop before a wave, and the
source interface through ingest.run.
"""
import contextlib
import hashlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_hf_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import db, ingest, paths  # noqa: E402
from engine.sources import hf_archive as hf  # noqa: E402

REPO = "someone/pro-demos"
paths.DEMOS_PRO_DIR = TMP / "demos" / "pro"


def body(tag: str, n: int = 3000) -> bytes:
    return b"PBDEMS2\x00" + tag.encode() + b"\x00" * n


FILES = {
    "2026-08/de_mirage/a.dem":   body("a"),
    "2026-08/de_mirage/b.dem":   body("b"),
    "2026-08/de_nuke/c.dem":     body("c"),
    "2026-09/de_mirage/d.dem":   body("d"),
    "2026-09/de_nuke/bad.dem":   body("bad"),          # listing carries a wrong sha
    "misc/9216090725219723660_0.dem": body("opaque"),  # no map, no month
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def entry(path: str, data: bytes, lfs: bool = True) -> dict:
    e = {"type": "file", "path": path, "size": len(data), "oid": "1" * 40}
    if lfs:
        e["lfs"] = {"oid": "0" * 64 if path.endswith("bad.dem") else sha(data), "size": len(data), "pointerSize": 133}
    return e


class Handler(BaseHTTPRequestHandler):
    port = 0
    hits: list = []

    def _send(self, status, ctype, data, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.hits.append(self.path)
        tree = f"/api/datasets/{REPO}/tree/main"
        if self.path == f"{tree}?recursive=true":
            items = [entry(p, d) for p, d in list(FILES.items())[:3]]
            items += [{"type": "directory", "path": "2026-08", "size": 0, "oid": "2" * 40},
                      {"type": "file", "path": "README.md", "size": 10, "oid": "3" * 40}]
            nxt = f"<http://127.0.0.1:{self.port}{tree}?recursive=true&cursor=abc>; rel=\"next\""
            return self._send(200, "application/json", json.dumps(items).encode(), [("Link", nxt)])
        if self.path == f"{tree}?recursive=true&cursor=abc":
            items = [entry(p, d) for p, d in list(FILES.items())[3:]]
            return self._send(200, "application/json", json.dumps(items).encode())
        prefix = f"/datasets/{REPO}/resolve/main/"
        if self.path.startswith(prefix) and self.path[len(prefix):] in FILES:
            return self._send(200, "application/octet-stream", FILES[self.path[len(prefix):]])
        self._send(404, "text/plain", b"nope")

    def log_message(self, *a):
        pass


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **kw)
    return rc, buf.getvalue()


def test_pure() -> None:
    section("classification and waves (pure)")
    check("map + month from path", hf.classify_path("2026-08/de_mirage/a.dem") == ("de_mirage", "2026-08"))
    check("opaque numeric ids give no month", hf.classify_path("complete/9216090725219723660_0.dem") == (None, None))
    files = [hf.HfFile(f"p{i}", s, None, m, mo) for i, (s, m, mo) in enumerate(
        [(10, "de_mirage", "2026-08"), (10, "de_mirage", "2026-08"), (10, "de_nuke", "2026-08"),
         (25, "de_mirage", "2026-09"), (5, None, None)])]
    waves = hf.plan_waves(files, wave_bytes=20)
    check("packed into waves <= budget (oversize file alone)",
          [[f.path for f in w] for w in waves] == [["p4"], ["p0", "p1"], ["p2"], ["p3"]],
          str([[f.path for f in w] for w in waves]))
    sel = hf.plan_waves(files, maps={"de_mirage"}, months={"2026-08"}, wave_bytes=100)
    check("map + month selection", [[f.path for f in w] for w in sel] == [["p0", "p1"]])
    check("unknown map never selected by a map filter",
          all(f.map is not None for w in hf.plan_waves(files, maps={"de_mirage", "de_nuke"}) for f in w))
    check("match ids are safe", hf.match_id_for("2026-08/de_mirage/a b.dem.zst") == "hf-2026-08-de_mirage-a-b")
    check("resolve URL", hf.resolve_url("o/r", "x/y z.dem", base="http://h")
          == "http://h/datasets/o/r/resolve/main/x/y%20z.dem")


def test_server() -> None:
    port = _free_port()
    Handler.port = port
    srv  = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    api, base = f"http://127.0.0.1:{port}/api", f"http://127.0.0.1:{port}"
    try:
        section("listing")
        files = hf.list_files(REPO, api_base=api)
        check("both pages read, only demo files kept", sorted(f.path for f in files) == sorted(FILES),
              str([f.path for f in files]))
        a = next(f for f in files if f.path.endswith("a.dem"))
        check("sha256 from lfs.oid, map/month classified",
              a.sha256 == sha(FILES[a.path]) and a.map == "de_mirage" and a.month == "2026-08", str(a))
        try:
            hf.list_files("nobody/none", api_base=api)
            check("missing repo raises", False)
        except IOError:
            check("missing repo raises", True)

        section("download in waves with sha256 verification")
        out = TMP / "dl"
        got, log = _quiet(hf.download_waves, files, out, REPO, maps={"de_mirage"}, months={"2026-08"},
                          wave_bytes=4000, base=base)
        check("selected map-month downloaded, one file per wave",
              sorted(p.name for p in got) == ["hf-2026-08-de_mirage-a.dem", "hf-2026-08-de_mirage-b.dem"]
              and log.count("wave ") == 2, log[-200:])
        check("bytes verified intact", (out / "hf-2026-08-de_mirage-a.dem").read_bytes() == FILES["2026-08/de_mirage/a.dem"])
        n_before = len(Handler.hits)
        _quiet(hf.download_waves, files, out, REPO, maps={"de_mirage"}, months={"2026-08"}, base=base)
        check("existing verified files are not downloaded again", len(Handler.hits) == n_before)
        try:
            _quiet(hf.download_waves, files, out, REPO, maps={"de_nuke"}, months={"2026-09"}, base=base)
            check("sha256 mismatch raises", False)
        except IOError as e:
            check("sha256 mismatch raises and leaves no file",
                  "mismatch" in str(e) and not (out / "hf-2026-09-de_nuke-bad.dem").exists(), str(e))

        saved = paths.check_space
        paths.check_space = lambda *a, **k: paths.SpaceStatus.REFUSE
        try:
            got, log = _quiet(hf.download_waves, files, TMP / "full", REPO, base=base)
            check("free space below the floor: stops before the first wave", got == [] and "stopping" in log)
        finally:
            paths.check_space = saved

        section("source interface through ingest.run")
        hdr = lambda p: {"map": "de_mirage", "tick_rate": 128}  # noqa: E731
        n_failed, log = _quiet(ingest.run, hf, tier="pro", header_reader=hdr, repo=REPO,
                               maps=["de_mirage"], api_base=api, base=base)
        keys = sorted(r["match_key"] for r in db.list_matches())
        check("three mirage files registered", n_failed == 0 and keys ==
              ["hf-2026-08-de_mirage-a", "hf-2026-08-de_mirage-b", "hf-2026-09-de_mirage-d"], f"{keys} {log[-150:]}")
        check("source recorded as hf_archive", db.get_match(keys[0])["source"] == "hf_archive")
        n_failed, log = _quiet(ingest.run, hf, tier="pro", header_reader=hdr, repo=REPO,
                               maps=["de_nuke"], months=["2026-09"], api_base=api, base=base)
        check("a sha mismatch counts as a failed match", n_failed == 1 and "mismatch" in log, log[-150:])
        _, log = _quiet(ingest.run, hf, tier="pro", header_reader=hdr, repo=REPO,
                        maps=["de_mirage"], api_base=api, base=base)
        check("processed files skipped on the next run", "of 0 attempted" in log)
        try:
            list(hf.iter_unprocessed_demos())
            check("no default repo", False)
        except ValueError:
            check("no default repo", True)
    finally:
        srv.shutdown()


def main() -> int:
    test_pure()
    test_server()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
