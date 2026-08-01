import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import extract as extractor
import feature_extractor
import state_sampler
import db


def main():
    if len(sys.argv) < 2:
        print("usage: .venv/bin/python test/test_local_demo.py path/to/match.dem [--write-db]")
        print("       (inspection only by default; --write-db appends to the REAL data/crosshair.db)")
        sys.exit(1)

    demo_path = sys.argv[1]
    write_db  = "--write-db" in sys.argv[2:]

    print("=== extracting tables ===")
    tables = extractor.extract(demo_path)
    for name, val in tables.items():
        if hasattr(val, "shape"):
            print(f"  {name}: {val.shape[0]} rows x {val.shape[1]} cols")
        else:
            print(f"  {name}: {val}")

    map_name    = tables.get("map_name", "unknown")
    vis_checker = feature_extractor.build_vis_checker(map_name)
    print(f"\nmap: {map_name}")

    # ONE id, used for the rows, the dedup check and the marker. Deriving the marker
    # from the argv string while writing rows under a different literal id means the
    # guard protects an id no row carries — and `demos/x.dem` vs `./demos/x.dem`
    # produce different markers for the same file.
    match_id = f"local_{Path(demo_path).stem}"

    print("\n=== extracting events ===")
    events = feature_extractor.extract_events(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)

    counts = Counter(e["event_type"] for e in events)
    print(f"total events: {len(events)}")
    for etype, n in counts.items():
        print(f"  {etype}: {n}")

    print("\n=== sampling round states ===")
    states = state_sampler.sample_round_states(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)
    print(f"total states: {len(states)}")

    failures = []
    if not events:
        failures.append("no events extracted")
    if not states:
        failures.append("no round states sampled")
    if map_name == "unknown":
        failures.append("map_name did not resolve")

    if write_db:
        print("\n=== storing in db ===")
        db.init_db()
        if db.is_processed(match_id):
            print("  already in db, skipping.")
        else:
            db.insert_events(events)
            db.insert_round_states(states)
            db.mark_processed(match_id, map_name, demo_path)
            s = db.stats()
            print(f"  stored. db: {s['matches_processed']} matches | {s['total_events']} events | {s['total_states']} states")
    else:
        print(f"\n=== db write skipped (pass --write-db to store as '{match_id}') ===")

    print("\n=== sample event per type ===")
    seen = set()
    for e in events:
        if e["event_type"] not in seen:
            seen.add(e["event_type"])
            print(f"\n--- {e['event_type']} ---")
            print(json.dumps(e, indent=2, default=str))
        if len(seen) == 4:
            break

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: events, states and map_name all present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
