"""
Inspection script for one local .dem — NOT a stored ingest.

    .venv/bin/python test/test_local_demo.py path/to/match.dem

Parses the demo, runs the extractor and the sampler, and prints what came out:
table shapes, event counts by type, state counts and one sample event per type.
Nothing is written to the database; to store a demo use `main.py demo <path>`,
which goes through the real seam (pipeline.ingest_demo).

Exit 1 when the demo yields no events, no states, or no map name.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import extract as extractor
import feature_extractor
import state_sampler


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: .venv/bin/python test/test_local_demo.py path/to/match.dem")
        return 1

    demo_path = sys.argv[1]
    match_id  = f"local_{Path(demo_path).stem}"

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

    print("\n=== extracting events ===")
    events = feature_extractor.extract_events(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)
    print(f"total events: {len(events)}")
    for etype, n in Counter(e["event_type"] for e in events).items():
        print(f"  {etype}: {n}")

    print("\n=== sampling round states ===")
    states   = state_sampler.sample_round_states(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)
    boundary = state_sampler.sample_boundary_states(tables, match_id, events, map_name, vis_checker=vis_checker)
    print(f"total states: {len(states)} grid + {len(boundary)} boundary")

    print("\n=== sample event per type ===")
    seen = set()
    for e in events:
        if e["event_type"] not in seen:
            seen.add(e["event_type"])
            print(f"\n--- {e['event_type']} ---")
            print(json.dumps(e, indent=2, default=str))

    failures = []
    if not events:
        failures.append("no events extracted")
    if not states:
        failures.append("no round states sampled")
    if map_name == "unknown":
        failures.append("map_name did not resolve")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nOK: events, states and map_name all present. (not stored — use `main.py demo {demo_path}`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
