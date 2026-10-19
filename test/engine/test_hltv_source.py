"""
HLTV source (P1.1.2) against a local HTTP server: match page -> demo archive -> one
`.dem` per map with the `<event>_<date>_<team1>-<team2>_<map>.dem` name.

    .venv\\Scripts\\python.exe test\\engine\\test_hltv_source.py

No network. Checks: page parsing, naming, extraction, a corrupt zip (bad CRC and
truncated) and a .rar leave no partial files, redaction of signed URLs in errors,
>= 10 s spacing between requests, and the source interface through ingest.run.
"""
import contextlib
import io
import os
import socket
import sys
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="ch_hltv_test_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from engine import db, ingest, paths  # noqa: E402
from engine.sources import hltv  # noqa: E402

SECRET = "SUPERSECRETSIGNATURE123"
UNIX_MS = 1726000000000
DATE = datetime.fromtimestamp(UNIX_MS / 1000, timezone.utc).strftime("%Y-%m-%d")
paths.DEMOS_PRO_DIR = TMP / "demos" / "pro"


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


DEM1 = b"PBDEMS2\x00mirage" + b"\x01" * 3000
DEM2 = b"PBDEMS2\x00inferno" + b"\x02" * 3000
GOOD = _zip({"navi-vs-faze-m1-mirage.dem": DEM1, "navi-vs-faze-m2-de_inferno.dem": DEM2,
             "readme.txt": b"hi"})
# flip bytes inside the SECOND member's data: the first extracts, the second fails CRC
_off = GOOD.index(b"\x02" * 100)
BAD_CRC = GOOD[:_off] + b"\x03" * 100 + GOOD[_off + 100:]
TRUNCATED = GOOD[: len(GOOD) // 2]
RAR = b"Rar!\x1a\x07\x00" + b"\x00" * 2000


def page(demo_link: str, event: str = "IEM Cologne 2026") -> bytes:
    return f"""<html><body>
<div class="timeAndEvent"><div class="date" data-unix="{UNIX_MS}">10th of September 2024</div>
<div class="event text-ellipsis"><a href="/events/1/x" title="{event}">{event}</a></div></div>
<div class="team1-gradient"><div class="teamName">Natus Vincere</div></div>
<div class="team2-gradient"><div class="teamName">FaZe</div></div>
<a class="stream-box" data-demo-link="{demo_link}"><span>GOTV Demo</span></a>
</body></html>""".encode()


class Handler(BaseHTTPRequestHandler):
    routes: dict = {}
    hits: list = []

    def do_GET(self):
        self.hits.append(self.path)
        spec = self.routes.get(self.path.split("?")[0])
        if spec is None:
            self.send_response(404); self.end_headers(); return
        ctype, body = spec
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **kw)
    return rc, buf.getvalue()


def _blob(e: BaseException) -> str:
    out = f"{e!r} {e}"
    if e.__cause__ is not None:
        out += f" {e.__cause__!r}"
    return out


def test_pure() -> None:
    section("pure helpers")
    info = hltv.parse_match_page(page("/download/demo/1").decode())
    check("page parsed", info == {"event": "IEM Cologne 2026", "date": DATE, "team1": "Natus Vincere",
                                  "team2": "FaZe", "demo_path": "/download/demo/1"}, str(info))
    check("missing fields are None", hltv.parse_match_page("<html></html>") ==
          {"event": None, "date": None, "team1": None, "team2": None, "demo_path": None})
    check("slug never contains the '_' separator", hltv.slug("Team_Liquid & Co.") == "team-liquid-co")
    check("map from member name", hltv.map_from_member("a-vs-b-m1-mirage.dem") == "de_mirage"
          and hltv.map_from_member("x/de_nuke.dem") == "de_nuke")
    check("id -> URL", hltv.match_url("2370000", "http://h") == ("2370000", "http://h/matches/2370000/match"))
    try:
        hltv.match_url(f"https://www.hltv.org/news/1?sig={SECRET}")
        check("non-match URL refused", False)
    except ValueError as e:
        check("non-match URL refused, redacted", SECRET not in str(e), str(e))

    t, sleeps = [100.0], []
    th = hltv.Throttle(10.0, clock=lambda: t[0], sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)))
    th.wait()
    t[0] += 3.0
    th.wait()
    t[0] += 12.0
    th.wait()
    check("throttle: no wait first, then tops up to 10 s, none after a long gap",
          len(sleeps) == 1 and abs(sleeps[0] - 7.0) < 1e-9, str(sleeps))
    check("module default spacing is >= 10 s", hltv._THROTTLE.min_interval_s >= 10.0)


def test_server() -> None:
    Handler.routes = {
        "/matches/1/good":   ("text/html", page(f"/download/demo/1?sig={SECRET}")),
        "/matches/2/crc":    ("text/html", page(f"/download/demo/2?sig={SECRET}")),
        "/matches/3/trunc":  ("text/html", page(f"/download/demo/3?sig={SECRET}")),
        "/matches/4/rar":    ("text/html", page(f"/download/demo/4?sig={SECRET}")),
        "/matches/5/gone":   ("text/html", page(f"/download/demo/404?sig={SECRET}")),
        "/matches/6/nodemo": ("text/html", b"<html>no demo</html>"),
        "/matches/7/run":    ("text/html", page("/download/demo/7", event="BLAST Fall")),
        "/download/demo/1":  ("application/octet-stream", GOOD),
        "/download/demo/2":  ("application/octet-stream", BAD_CRC),
        "/download/demo/3":  ("application/octet-stream", TRUNCATED),
        "/download/demo/4":  ("application/octet-stream", RAR),
        "/download/demo/7":  ("application/octet-stream", GOOD),
    }
    port = _free_port()
    srv  = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    fast = hltv.Throttle(0)
    try:
        section("download + unpack")
        out_dir = TMP / "out"
        (info, files), out = _quiet(hltv.download_match, f"{base}/matches/1/good", out_dir, throttle=fast)
        names = sorted(p.name for p in files)
        want  = sorted([f"hltv-1_iem-cologne-2026_{DATE}_natus-vincere-faze_de_mirage.dem",
                        f"hltv-1_iem-cologne-2026_{DATE}_natus-vincere-faze_de_inferno.dem"])
        check("one .dem per map, P1.1.2 names", names == want, str(names))
        check("contents extracted intact",
              (out_dir / want[1]).read_bytes() == DEM1 and (out_dir / want[0]).read_bytes() == DEM2)
        check("non-.dem members ignored, no staging left",
              sorted(p.name for p in out_dir.iterdir()) == want, str(list(out_dir.iterdir())))
        check("signed archive URL never printed", SECRET not in out)
        try:
            _quiet(hltv.download_match, f"{base}/matches/1/good", out_dir, throttle=fast)
            check("re-download never overwrites", False)
        except FileExistsError:
            check("re-download never overwrites", sorted(p.name for p in out_dir.iterdir()) == want)

        section("failures leave nothing behind and are redacted")
        for route, exc, label in [("/matches/2/crc", IOError, "bad CRC in the second member"),
                                  ("/matches/3/trunc", IOError, "truncated zip"),
                                  ("/matches/4/rar", hltv.UnsupportedArchive, ".rar archive"),
                                  ("/matches/5/gone", IOError, "404 on the archive"),
                                  ("/matches/6/nodemo", LookupError, "page without a demo link")]:
            d = TMP / f"fail{route.split('/')[2]}"
            try:
                _, out = _quiet(hltv.download_match, f"{base}{route}?sig={SECRET}", d, throttle=fast)
                check(f"{label} raises", False, "no exception")
            except exc as e:
                left = sorted(p.name for p in d.rglob("*")) if d.exists() else []
                check(f"{label} raises and leaves no file", not left, str(left))
                check(f"{label}: signed URL absent from the error", SECRET not in _blob(e), _blob(e)[:100])
        try:
            hltv.fetch_page(f"http://127.0.0.1:{_free_port()}/matches/9/x?sig={SECRET}", fast)
            check("connection failure raises", False)
        except IOError as e:
            check("connection failure is redacted", SECRET not in _blob(e), _blob(e)[:100])

        section("source interface through ingest.run")
        hdr = lambda p: {"map": "de_mirage" if "mirage" in p.name else "de_inferno", "tick_rate": 128}  # noqa: E731
        n_failed, out = _quiet(ingest.run, hltv, tier="pro", header_reader=hdr,
                               matches=[f"{base}/matches/7/run"], base_url=base, throttle=fast)
        keys = sorted(r["match_key"] for r in db.list_matches())
        want = sorted([f"hltv-7_blast-fall_{DATE}_natus-vincere-faze_de_mirage",
                       f"hltv-7_blast-fall_{DATE}_natus-vincere-faze_de_inferno"])
        check("both maps registered under their names", n_failed == 0 and keys == want, f"{keys} {out[-200:]}")
        row = db.get_match(want[0])
        check("event/date from the page reach the row, source=hltv, tier=pro",
              row["event"] == "BLAST Fall" and row["date"] == DATE and row["source"] == "hltv"
              and row["tier"] == "pro", str(row))
        check("files under demos/pro", all((paths.DEMOS_PRO_DIR / f"{k}.dem").is_file() for k in want))
        before = len(Handler.hits)
        n_failed, out = _quiet(ingest.run, hltv, tier="pro", header_reader=hdr,
                               matches=[f"{base}/matches/7/run"], base_url=base, throttle=fast)
        check("second run skips the seen match without downloading the archive",
              "of 0 attempted" in out and all("/download/" not in h for h in Handler.hits[before:]), out[-120:])
        try:
            hltv.download_demo((TMP / "x.dem").as_uri(), TMP / "y.dem")
            check("foreign file URI refused", False)
        except ValueError:
            check("foreign file URI refused", True)
    finally:
        srv.shutdown()


def main() -> int:
    test_pure()
    test_server()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
