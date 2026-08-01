"""
Tests for scraper.download_demo against a local HTTP server.

Run from repo root:
    .venv/bin/python test/test_download.py

No credentials, no network, no real demo. Serves crafted responses from localhost
so the validation added during the security pass is actually exercised:
magic-byte format detection, truncated-transfer detection, HTML/error-page
rejection, size caps, .part cleanup, and credential redaction in error messages.

These paths run for the first time on a real scrape, so they get tested here
rather than discovered at hour three of an overnight run.
"""
import gzip
import io
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_results: list[tuple[bool, str]] = []

SECRET = "SUPERSECRETSIGNATURE123"
DEMO_MAGIC = b"PBDEMS2\x00"          # CS2 demo header
BODY_RAW = DEMO_MAGIC + b"\x00" * 4000


def check(desc: str, ok: bool, detail: str = ""):
    _results.append((bool(ok), desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))


# ── a tiny server that can misbehave on demand ────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):
        path = self.path.split("?")[0]
        spec = self.routes.get(path)
        if spec is None:
            self.send_response(404); self.end_headers(); return
        status, ctype, body, declared_len, truncate_to = spec
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(declared_len if declared_len is not None else len(body)))
        self.end_headers()
        self.wfile.write(body[:truncate_to] if truncate_to is not None else body)

    def log_message(self, *a):
        pass


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def main() -> int:
    import scraper

    zst_body = None
    try:
        import zstandard as zstd
        zst_body = zstd.ZstdCompressor().compress(BODY_RAW)
    except Exception:
        pass
    gz_body = gzip.compress(BODY_RAW)

    Handler.routes = {
        "/ok.dem":        (200, "application/octet-stream", BODY_RAW, None, None),
        "/ok.dem.gz":     (200, "application/octet-stream", gz_body, None, None),
        "/truncated.dem": (200, "application/octet-stream", BODY_RAW, len(BODY_RAW), 500),
        "/page.dem":      (200, "text/html; charset=utf-8", b"<html>Access denied</html>", None, None),
        "/tiny.dem":      (200, "application/octet-stream", b"xx", None, None),
        "/boom.dem":      (500, "text/plain", b"server error", None, None),
    }
    if zst_body is not None:
        Handler.routes["/ok.dem.zst"] = (200, "application/octet-stream", zst_body, None, None)

    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    tmp = Path("/tmp/ch_dl_test"); tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*"):
        f.unlink()

    def dl(route: str, name: str, query: str = ""):
        return scraper.download_demo(f"{base}{route}{query}", tmp / name)

    try:
        print("── happy paths: format detected from MAGIC BYTES, not the URL ───────────")
        try:
            p = dl("/ok.dem", "a.dem")
            check("uncompressed demo downloads", p.exists() and p.read_bytes() == BODY_RAW,
                  f"{p.stat().st_size} bytes")
        except Exception as e:
            check("uncompressed demo downloads", False, repr(e))

        try:
            p = dl("/ok.dem.gz", "b.dem")
            check("gzip demo is decompressed", p.exists() and p.read_bytes() == BODY_RAW,
                  f"{p.stat().st_size} bytes")
        except Exception as e:
            check("gzip demo is decompressed", False, repr(e))

        if zst_body is not None:
            try:
                p = dl("/ok.dem.zst", "c.dem")
                check("zstd demo is decompressed", p.exists() and p.read_bytes() == BODY_RAW,
                      f"{p.stat().st_size} bytes")
            except Exception as e:
                check("zstd demo is decompressed", False, repr(e))

            # The old code chose the branch by substring-matching the WHOLE url,
            # including the signed query string. A ".zst" inside a token picked the
            # wrong branch; magic bytes must win.
            try:
                p = dl("/ok.dem.gz", "d.dem", query="?sig=abc.zst.def")
                check("a '.zst' in the query does not misroute a gzip body",
                      p.exists() and p.read_bytes() == BODY_RAW)
            except Exception as e:
                check("a '.zst' in the query does not misroute a gzip body", False, repr(e))

        print("\n── rejections ──────────────────────────────────────────────────────────")
        for route, name, desc in [
            ("/truncated.dem", "e.dem", "a truncated transfer is rejected, not silently kept"),
            ("/page.dem",      "f.dem", "an HTML error page is rejected"),
            ("/tiny.dem",      "g.dem", "an implausibly small body is rejected"),
            ("/boom.dem",      "h.dem", "a 5xx is rejected"),
            ("/missing.dem",   "i.dem", "a 404 is rejected"),
        ]:
            try:
                dl(route, name)
                check(desc, False, "no exception raised")
            except Exception:
                check(desc, True)

        print("\n── no partial file survives a failure ──────────────────────────────────")
        leftovers = sorted(p.name for p in tmp.glob("*.part"))
        check("no .part files left behind", not leftovers, str(leftovers))
        for name in ("e.dem", "f.dem", "g.dem", "h.dem", "i.dem"):
            f = tmp / name
            if f.exists():
                check(f"failed download left no {name}", False, f"{f.stat().st_size} bytes")
        check("failed downloads left no output files",
              not any((tmp / n).exists() for n in ("e.dem", "f.dem", "g.dem", "h.dem", "i.dem")))

        print("\n── credentials never reach an error message ────────────────────────────")
        for route, name, label in [("/boom.dem", "j.dem", "5xx"), ("/missing.dem", "k.dem", "404")]:
            try:
                dl(route, name, query=f"?sig={SECRET}")
                check(f"{label} raises", False)
            except Exception as e:
                blob = f"{e!r} {e}"
                cause = getattr(e, "__cause__", None)
                if cause:
                    blob += f" {cause!r} {cause}"
                check(f"signed query absent from the {label} error", SECRET not in blob,
                      blob[:110])

        # Connection-level failure: nothing listening. urllib3 embeds "path?query".
        try:
            scraper.download_demo(f"http://127.0.0.1:{_free_port()}/x.dem?sig={SECRET}", tmp / "l.dem")
            check("connection failure raises", False)
        except Exception as e:
            blob = f"{e!r} {e}"
            cause = getattr(e, "__cause__", None)
            if cause:
                blob += f" {cause!r} {cause}"
            check("signed query absent from a CONNECTION error", SECRET not in blob, blob[:110])

        print("\n── size cap is enforced on written bytes ───────────────────────────────")
        original = scraper.MAX_DEMO_BYTES
        try:
            scraper.MAX_DEMO_BYTES = 100          # smaller than the 4000-byte body
            try:
                dl("/ok.dem", "m.dem")
                check("oversize download is aborted", False, "no exception")
            except Exception:
                check("oversize download is aborted", True)
        finally:
            scraper.MAX_DEMO_BYTES = original

    finally:
        srv.shutdown()

    passed = sum(1 for ok, _ in _results if ok)
    failed = [d for ok, d in _results if not ok]
    print(f"\n{'─'*60}\n  {passed}/{len(_results)} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
