"""
Crosshair CLI — single entry point for the whole pipeline.

Usage:
    python src/main.py status                     # DB stats
    python src/main.py scrape [opts]              # scrape Faceit + process
    python src/main.py demo path/to/match.dem     # process one local .dem
    python src/main.py train [--eval]             # train WP model on DB
    python src/main.py score [--match-id ID]      # fill p_before/p_after/impact
    python src/main.py top [--by impact] [-n 10]  # top-impact moments
    python src/main.py player <name> [-n 20]      # top moments for a player
    python src/main.py round <match_id> <round>   # round-by-round play log
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "crosshair.db"

sys.path.insert(0, str(Path(__file__).parent))


# ── helpers ───────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        sys.exit(f"No database at {DB_PATH}. Run `scrape` or `demo` first.")
    return sqlite3.connect(DB_PATH)


def _short(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[:n] + "…"


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_status(args):
    con = _conn()
    n_matches = con.execute("SELECT COUNT(*) FROM processed_matches").fetchone()[0]
    n_events  = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_states  = con.execute("SELECT COUNT(*) FROM round_states").fetchone()[0]
    # `impact` is absent from a DB built by db._init_schema — degrade instead of
    # dying with a raw OperationalError before printing anything.
    try:
        n_scored = con.execute("SELECT COUNT(*) FROM events WHERE impact IS NOT NULL").fetchone()[0]
        scored_note = f"{n_scored:,} scored, {100 * n_scored // max(n_events, 1)}%"
    except sqlite3.OperationalError:
        scored_note = "impact column missing — schema needs migrating"

    print(f"DB        : {DB_PATH}  ({DB_PATH.stat().st_size / 1e9:.2f} GB)")
    print(f"Matches   : {n_matches:,}")
    print(f"Events    : {n_events:,}  ({scored_note})")
    print(f"States    : {n_states:,}")

    print("\nBy event type:")
    for et, n in con.execute(
        "SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY COUNT(*) DESC"
    ):
        print(f"  {et:<14} {n:>10,}")

    print("\nBy map:")
    for m, n in con.execute(
        "SELECT map, COUNT(*) FROM processed_matches GROUP BY map ORDER BY COUNT(*) DESC"
    ):
        print(f"  {m or 'unknown':<14} {n:>4} matches")

    model = ROOT / "data" / "win_prob.lgb"
    print(f"\nWP model  : {'✓ present' if model.exists() else '✗ NOT TRAINED'}")
    con.close()
    return 0


def cmd_scrape(args):
    import pipeline
    n_failed = pipeline.run(
        min_elo            = args.min_elo,
        region             = args.region,
        max_players        = args.max_players,
        matches_per_player = args.matches_per_player,
        start_offset       = args.start_offset,
        source             = args.source,
    )
    return 1 if n_failed else 0


def cmd_demo(args):
    import db
    import extract as extractor
    import feature_extractor
    import state_sampler

    match_id = args.match_id or f"local_{Path(args.path).stem}"
    db.init_db()
    # Check BEFORE parsing — extract() is minutes of CPU, and the bare id alone
    # misses the BO-series form the scrape path also checks.
    if db.is_processed(match_id) or db.is_processed(f"{match_id}_m1"):
        print(f"[{match_id}] already in DB, skipping.")
        return 0

    print(f"=== processing {args.path} ===")
    tables = extractor.extract(args.path)
    map_name = tables.get("map_name", "unknown")
    vis_checker = feature_extractor.build_vis_checker(map_name)

    events = feature_extractor.extract_events(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)
    states = state_sampler.sample_round_states(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)
    boundary = state_sampler.sample_boundary_states(tables, match_id, events, map_name, vis_checker=vis_checker)
    if boundary:
        print(f"[{match_id}] + {len(boundary)} boundary states")
        states = states + boundary

    db.store_match(match_id, events, states, map_name, args.path)
    print(f"[{match_id}] stored {len(events):,} events, {len(states):,} states.")
    return 0


def cmd_train(args):
    import win_probability
    win_probability.train(eval=args.eval)
    return 0


def cmd_score(args):
    import score_impact
    argv = []
    if args.match_id:            argv += ["--match-id", args.match_id]
    if args.limit is not None:   argv += ["--limit", str(args.limit)]
    if getattr(args, "attribution", None): argv += ["--attribution", args.attribution]
    return score_impact.main(argv)


def cmd_top(args):
    con = _conn()
    by  = args.by
    where  = ["impact IS NOT NULL"]
    params: list = []
    if args.event_type:
        where.append("event_type = ?")
        params.append(args.event_type)
    if args.side:
        where.append("player_side = ?")
        params.append(args.side)

    # `order` is chosen from a literal dict, never from user text.
    order = {
        "impact":     "ABS(impact) DESC",
        "positive":   "impact DESC",
        "negative":   "impact ASC",
    }.get(by, "ABS(impact) DESC")

    print(f"Top {args.n} events by {by}:\n")
    print(f"  {'id':>9} {'match':<24} {'rn':>3} {'t':>6} {'side':<4} "
          f"{'type':<11} {'pb':>6} {'pa':>6} {'imp':>7}")
    print(f"  {'-'*9} {'-'*24} {'-'*3} {'-'*6} {'-'*4} {'-'*11} {'-'*6} {'-'*6} {'-'*7}")
    for r in con.execute(f"""
        SELECT id, match_id, round_num, time_into_round, player_side, event_type,
               p_before, p_after, impact
        FROM events WHERE {' AND '.join(where)}
        ORDER BY {order} LIMIT ?
    """, (*params, args.n)):
        # round_num / time_into_round / player_side are all nullable.
        rn = f"{r[2]:>3}" if r[2] is not None else "  -"
        t  = f"{r[3]:>6.1f}" if r[3] is not None else "     -"
        print(f"  {r[0]:>9} {_short(r[1], 24):<24} {rn} {t} "
              f"{(r[4] or '-'):<4} {(r[5] or '-'):<11} {r[6]:>6.3f} {r[7]:>6.3f} {r[8]:>+7.3f}")
    con.close()
    return 0


def _like_literal(s: str) -> str:
    """Escape LIKE metacharacters. Nicknames commonly contain _ and %, which would
    otherwise act as wildcards and silently match a different player."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def cmd_player(args):
    """List scored engagements that MENTION a player.

    NOTE: the events schema has no actor column, so these rows cannot be narrowed to
    the ones where `name` is the actor — a player also appears in another player's
    `situation.enemies_spotted` and as a victim. The totals below are therefore over
    every event mentioning the name, on both sides. Treat them as a search result,
    not as a per-player statistic.
    """
    con = _conn()
    name = args.name
    pattern = f'%"{_like_literal(name)}"%'

    print(f"Events mentioning '{name}' (not filtered to actor — see --help):\n")

    rows = con.execute("""
        SELECT id, match_id, round_num, time_into_round, player_side, event_type,
               json_extract(action, '$.role') role,
               json_extract(outcome, '$.result') result,
               json_extract(action, '$.weapon') weapon,
               impact
        FROM events
        WHERE event_type = 'engagement'
          AND impact IS NOT NULL
          AND (situation LIKE ? ESCAPE '\\' OR action LIKE ? ESCAPE '\\')
        ORDER BY ABS(impact) DESC
        LIMIT ?
    """, (pattern, pattern, args.n)).fetchall()

    if not rows:
        print(f"No scored engagements mention '{name}'.")
        con.close()
        return 0

    print(f"  events shown : {len(rows):,}  (top {args.n} by |impact|)")
    print(f"\nTop {args.n} moments by |impact|:")
    print(f"  {'rn':>3} {'t':>6} {'side':<4} {'role':<8} {'result':<6} {'wpn':<10} {'imp':>7}")
    for r in rows:
        rn = f"{r[2]:>3}" if r[2] is not None else "  -"
        t  = f"{r[3]:>6.1f}" if r[3] is not None else "     -"
        print(f"  {rn} {t} {(r[4] or '-'):<4} {(r[6] or '-'):<8} {(r[7] or '-'):<6} "
              f"{(r[8] or '-'):<10} {r[9]:>+7.3f}")
    con.close()
    return 0


def cmd_round(args):
    """Show a play-by-play of one round with WP at each step."""
    con = _conn()
    rows = con.execute("""
        SELECT id, time_into_round, player_side, event_type,
               json_extract(action, '$.role') role,
               json_extract(outcome, '$.result') result,
               json_extract(action, '$.weapon') weapon,
               p_before, p_after, impact
        FROM events
        WHERE match_id = ? AND round_num = ?
        ORDER BY time_into_round, id
    """, (args.match_id, args.round)).fetchall()

    if not rows:
        print(f"No events for match {args.match_id} round {args.round}.")
        con.close()
        return 1

    # Get round winner
    winner = con.execute("""
        SELECT round_won_ct FROM round_states
        WHERE match_id = ? AND round_num = ? LIMIT 1
    """, (args.match_id, args.round)).fetchone()
    winner_side = "CT" if (winner and winner[0]) else "T"

    print(f"\nMatch {args.match_id}  round {args.round}  ({winner_side} won)\n")
    print(f"  {'t':>6} {'side':<4} {'type':<11} {'role':<8} {'result':<6} {'wpn':<10} {'WP':>6} {'imp':>7}")
    print(f"  {'-'*6} {'-'*4} {'-'*11} {'-'*8} {'-'*6} {'-'*10} {'-'*6} {'-'*7}")
    for r in rows:
        if args.engagements_only and r[3] != "engagement":
            continue
        imp = f"{r[9]:+.3f}" if r[9] is not None else "  -  "
        wp  = f"{r[8]:.3f}" if r[8] is not None else "  -  "
        t   = f"{r[1]:>6.1f}" if r[1] is not None else "     -"
        print(f"  {t} {(r[2] or '-'):<4} {(r[3] or '-'):<11} {(r[4] or '-'):<8} "
              f"{(r[5] or '-'):<6} {(r[6] or '-'):<10} {wp:>6} {imp:>7}")
    con.close()
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap  = argparse.ArgumentParser(prog="crosshair", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("status", help="DB stats")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("scrape", help="Scrape Faceit and process demos")
    sp.add_argument("--min-elo",            type=int, default=2500)
    sp.add_argument("--region",             default="EU")
    sp.add_argument("--max-players",        type=int, default=10)
    sp.add_argument("--matches-per-player", type=int, default=20)
    sp.add_argument("--start-offset",       type=int, default=0)
    sp.add_argument("--source",             choices=["playwright", "api"], default="playwright",
                    help="demo source: playwright (default) or api (FACEIT Downloads API)")
    sp.set_defaults(fn=cmd_scrape)

    sp = sub.add_parser("demo", help="Process one local .dem file")
    sp.add_argument("path")
    sp.add_argument("--match-id", help="Override the auto-generated match_id")
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("train", help="Train WP model on round_states in DB")
    sp.add_argument("--eval", action="store_true", help="Print AUC and feature importance")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("score", help="Compute p_before/p_after/impact for all events")
    sp.add_argument("--match-id", help="Score only one match")
    sp.add_argument("--limit",    type=int, help="Score first N matches")
    sp.add_argument("--attribution", choices=["grid", "event"], default="grid",
                    help="grid = legacy 1 Hz bracketing; event = per-action boundaries")
    sp.set_defaults(fn=cmd_score)

    sp = sub.add_parser("top", help="Top-impact moments in the DB")
    sp.add_argument("-n",    type=int, default=10)
    sp.add_argument("--by",  choices=["impact", "positive", "negative"], default="impact",
                    help="impact = abs(impact); positive/negative = signed")
    sp.add_argument("--event-type", help="Filter to one event_type (engagement, bomb, …)")
    sp.add_argument("--side",       choices=["ct", "t"])
    sp.set_defaults(fn=cmd_top)

    sp = sub.add_parser("player", help="Aggregate impact for one player by name")
    sp.add_argument("name")
    sp.add_argument("-n", type=int, default=20)
    sp.set_defaults(fn=cmd_player)

    sp = sub.add_parser("round", help="Play-by-play of one round")
    sp.add_argument("match_id")
    sp.add_argument("round", type=int)
    sp.add_argument("--engagements-only", action="store_true")
    sp.set_defaults(fn=cmd_round)

    args = ap.parse_args()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
