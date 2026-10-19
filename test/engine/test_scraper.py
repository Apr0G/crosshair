"""Manual smoke check for demo-URL discovery.

Run from the repo root:
    .venv/bin/python test/test_scraper.py [match_id]

NOTE: this opens a real Chrome window against your live profile and waits for you
to press ENTER. Everything is inside main() deliberately — at module level, merely
importing this file (any test runner, any IDE test panel) would launch that browser.
"""
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_MATCH = "1-ecb64f76-ee62-4b86-bcd9-9968b36fec62"
DEMO_EXTS = (".dem", ".dem.zst", ".dem.gz")


def main() -> int:
    from scraper import get_demo_urls_from_page

    match_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCH

    print(f"testing: {match_id}")
    urls = get_demo_urls_from_page(match_id)

    print(f"\nfound {len(urls)} demo(s):")
    for u in urls:
        # Path only — the query string carries the signature.
        p = urlsplit(u)
        print(f"  {p.scheme}://{p.netloc}{p.path}")

    if not urls:
        print("\nFAILED: no demo URLs intercepted (the selectors may have broken)")
        return 1

    bad = [u for u in urls if not urlsplit(u).path.lower().endswith(DEMO_EXTS)]
    if bad:
        print(f"\nFAILED: {len(bad)} intercepted URL(s) do not look like demo files")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
