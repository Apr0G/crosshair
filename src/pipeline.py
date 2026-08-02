"""
End-to-end ingestion pipeline.

  1. Discover high-ELO Faceit players via the public API
  2. Visit each match room with Playwright, intercept the demo download URL
  3. Download + decompress the .dem
  4. Parse with awpy, extract events + state snapshots
  5. Store in SQLite, delete the demo

Invoked by `main.py scrape` or run directly with `python src/pipeline.py`.
"""
import argparse
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import db
import scraper
import faceit_api
import extract as extractor
import feature_extractor
import state_sampler

# FACEIT match ids look like "1-<uuid>", optionally suffixed "_m1" for BO series.
_MATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def redact_url(url: str) -> str:
    """Drop the query string. FACEIT signed URLs carry their credential there, and
    `requests` puts the full URL into every HTTPError message."""
    if not url:
        return ""
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return "<unparseable url>"
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return f"{clean}?<redacted>" if parts.query else clean


# urllib3 embeds only "path?query" in connection-level errors, not the full URL, so a
# literal replace of the known URL never matches. Strip any query string that shows up.
_QUERY_RE = re.compile(r"\?[^\s)'\"]+")


def redact_text(text: str, *urls: str) -> str:
    """Scrub credential-bearing URLs out of an arbitrary message.

    Two passes, because exceptions carry the URL in more than one shape:
      1. literal replacement of URLs we were handed
      2. a blanket strip of any remaining `?...` query string
    """
    out = str(text)
    for u in urls:
        if u:
            out = out.replace(str(u), redact_url(u))
    return _QUERY_RE.sub("?<redacted>", out)


def process_one(match_id: str, demo_url: str, source=scraper) -> int:
    if not _MATCH_ID_RE.match(match_id or ""):
        raise ValueError(f"refusing unsafe match_id: {match_id!r}")
    if db.is_processed(match_id):
        print(f"[{match_id}] already in DB, skipping.")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / f"{match_id}.dem"
        # pathlib's `/` discards the left operand for an absolute right operand.
        if not demo_path.resolve().is_relative_to(Path(tmpdir).resolve()):
            raise ValueError(f"match_id escapes the temp dir: {match_id!r}")

        print(f"[{match_id}] downloading ...")
        source.download_demo(demo_url, demo_path)

        print(f"[{match_id}] parsing demo ...")
        tables = extractor.extract(str(demo_path))

        map_name    = tables.get("map_name", "unknown")
        vis_checker = feature_extractor.build_vis_checker(map_name)

        print(f"[{match_id}] extracting events ...")
        events = feature_extractor.extract_events(tables, match_id, map_name, vis_checker=vis_checker)

        print(f"[{match_id}] sampling round states ...")
        states = state_sampler.sample_round_states(tables, match_id, map_name, vis_checker=vis_checker)
        # Extra states at each instantaneous action's boundaries, so impact
        # attribution can value the action's own jump rather than the whole second.
        boundary = state_sampler.sample_boundary_states(
            tables, match_id, events, map_name, vis_checker=vis_checker)
        if boundary:
            print(f"[{match_id}] + {len(boundary)} boundary states")
            states = states + boundary

        print(f"[{match_id}] storing {len(events)} events + {len(states)} states ...")
        # One transaction: events, states and the processed marker land together or
        # not at all, so a crash can't leave rows behind that the next run re-inserts.
        # Never persist a signed URL — the Playwright path yields one, and its
        # credential lives in the query string.
        db.store_match(match_id, events, states, map_name, redact_url(demo_url))

        print(f"[{match_id}] done.")
        return len(events)


def run(min_elo: int = 2500, region: str = "EU", max_players: int = 10, matches_per_player: int = 20, start_offset: int = 0, source: str = "playwright"):
    db.init_db()

    src = faceit_api if source == "api" else scraper
    print(f"source: {source}")

    total_events = 0
    i = 0
    n_ok = 0
    failed: list[tuple[str, str]] = []
    for match_id, demo_url in src.iter_unprocessed_demos(
        min_elo=min_elo,
        region=region,
        max_players=max_players,
        matches_per_player=matches_per_player,
        start_offset=start_offset,
    ):
        i += 1
        print(f"\n── match {i}: {match_id} ──")
        try:
            n = process_one(match_id, demo_url, source=src)
            total_events += n
            n_ok += 1
        except Exception as e:
            # `e` can embed the signed URL (requests puts it in every HTTPError).
            msg = redact_text(e, demo_url)
            print(f"[{match_id}] error: {msg}")
            failed.append((match_id, msg))

    if hasattr(src, "close_browser"):
        src.close_browser()

    if i == 0:
        print("no new demos found.")

    s = db.stats()
    print(f"\ndone: {n_ok} ok, {len(failed)} failed, of {i} attempted (+{total_events} events)")
    if failed:
        print("failed matches:")
        for mid, msg in failed:
            print(f"  {mid}: {msg}")
    print(f"db: {s['matches_processed']} matches | {s['total_events']} events | {s['total_states']} states")
    return len(failed)


def loop(min_elo: int = 2500, region: str = "EU", interval: int = 3600,
         source: str = "playwright", max_consecutive_failures: int = 5):
    consecutive = 0
    while True:
        try:
            run(min_elo=min_elo, region=region, source=source)
            consecutive = 0
        except Exception:
            consecutive += 1
            # str(e) throws away the traceback, and a bare KeyError prints as just
            # 'nickname' — useless for diagnosing an unattended run.
            # format_exc() can carry a signed URL from anywhere in the stack.
            print(f"run failed ({consecutive}/{max_consecutive_failures}):\n"
                  f"{redact_text(traceback.format_exc())}")
            if consecutive >= max_consecutive_failures:
                print("too many consecutive failures, exiting.")
                return 1
        print(f"\nsleeping {interval}s ...")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-elo",            type=int, default=2500)
    parser.add_argument("--region",             default="EU")
    parser.add_argument("--max-players",        type=int, default=10)
    parser.add_argument("--matches-per-player", type=int, default=20)
    parser.add_argument("--loop",               action="store_true")
    parser.add_argument("--interval",           type=int, default=3600)
    parser.add_argument("--start-offset",       type=int, default=0)
    parser.add_argument("--source",             choices=["playwright", "api"], default="playwright")
    args = parser.parse_args()

    if args.loop:
        sys.exit(loop(args.min_elo, args.region, args.interval, source=args.source) or 0)
    else:
        n_failed = run(args.min_elo, args.region, args.max_players,
                       args.matches_per_player, args.start_offset, source=args.source)
        sys.exit(1 if n_failed else 0)
