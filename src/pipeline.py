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
import tempfile
import time
from pathlib import Path

import db
import scraper
import extract as extractor
import feature_extractor
import state_sampler


def process_one(match_id: str, demo_url: str) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / f"{match_id}.dem"

        print(f"[{match_id}] downloading ...")
        scraper.download_demo(demo_url, demo_path)

        print(f"[{match_id}] parsing demo ...")
        tables = extractor.extract(str(demo_path))

        map_name    = tables.get("map_name", "unknown")
        vis_checker = feature_extractor.build_vis_checker(map_name)

        print(f"[{match_id}] extracting events ...")
        events = feature_extractor.extract_events(tables, match_id, map_name, vis_checker=vis_checker)

        print(f"[{match_id}] sampling round states ...")
        states = state_sampler.sample_round_states(tables, match_id, map_name, vis_checker=vis_checker)

        print(f"[{match_id}] storing {len(events)} events + {len(states)} states ...")
        db.insert_events(events)
        db.insert_round_states(states)
        db.mark_processed(match_id, map_name, demo_url)

        print(f"[{match_id}] done.")
        return len(events)


def run(min_elo: int = 2500, region: str = "EU", max_players: int = 10, matches_per_player: int = 20, start_offset: int = 0):
    db.init_db()

    total_events = 0
    i = 0
    for match_id, demo_url in scraper.iter_unprocessed_demos(
        min_elo=min_elo,
        region=region,
        max_players=max_players,
        matches_per_player=matches_per_player,
        start_offset=start_offset,
    ):
        i += 1
        print(f"\n── match {i}: {match_id} ──")
        try:
            n = process_one(match_id, demo_url)
            total_events += n
        except Exception as e:
            print(f"[{match_id}] error: {e}")

    if i == 0:
        print("no new demos found.")

    s = db.stats()
    print(f"\ndone: {i} matches, +{total_events} events")
    print(f"db: {s['matches_processed']} matches | {s['total_events']} events | {s['total_states']} states")


def loop(min_elo: int = 2500, region: str = "EU", interval: int = 3600):
    while True:
        try:
            run(min_elo=min_elo, region=region)
        except Exception as e:
            print(f"run failed: {e}")
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
    args = parser.parse_args()

    if args.loop:
        loop(args.min_elo, args.region, args.interval)
    else:
        run(args.min_elo, args.region, args.max_players, args.matches_per_player, args.start_offset)
