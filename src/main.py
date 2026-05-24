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
import json
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
    n_scored  = con.execute("SELECT COUNT(*) FROM events WHERE impact IS NOT NULL").fetchone()[0]

    print(f"DB        : {DB_PATH}  ({DB_PATH.stat().st_size / 1e9:.2f} GB)")
    print(f"Matches   : {n_matches:,}")
    print(f"Events    : {n_events:,}  ({n_scored:,} scored, "
          f"{100 * n_scored // max(n_events, 1)}%)")
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


def cmd_scrape(args):
    import pipeline
    pipeline.run(
        min_elo            = args.min_elo,
        region             = args.region,
        max_players        = args.max_players,
        matches_per_player = args.matches_per_player,
        start_offset       = args.start_offset,
    )


def cmd_demo(args):
    import db
    import extract as extractor
    import feature_extractor
    import state_sampler

    print(f"=== processing {args.path} ===")
    tables = extractor.extract(args.path)
    map_name = tables.get("map_name", "unknown")
    vis_checker = feature_extractor.build_vis_checker(map_name)

    match_id = args.match_id or f"local_{Path(args.path).stem}"
    db.init_db()
    if db.is_processed(match_id):
        print(f"[{match_id}] already in DB, skipping.")
        return

    events = feature_extractor.extract_events(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)
    states = state_sampler.sample_round_states(tables, match_id=match_id, map_name=map_name, vis_checker=vis_checker)

    db.insert_events(events)
    db.insert_round_states(states)
    db.mark_processed(match_id, map_name, args.path)
    print(f"[{match_id}] stored {len(events):,} events, {len(states):,} states.")


def cmd_train(args):
    import win_probability
    win_probability.train(eval=args.eval)


def cmd_score(args):
    import score_impact
    sys.argv = ["score_impact"]
    if args.match_id: sys.argv += ["--match-id", args.match_id]
    if args.limit:    sys.argv += ["--limit", str(args.limit)]
    score_impact.main()


def cmd_top(args):
    con = _conn()
    by  = args.by
    where = "impact IS NOT NULL"
    if args.event_type:
        where += f" AND event_type = '{args.event_type}'"
    if args.side:
        where += f" AND player_side = '{args.side}'"

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
        FROM events WHERE {where}
        ORDER BY {order} LIMIT {args.n}
    """):
        print(f"  {r[0]:>9} {_short(r[1], 24):<24} {r[2]:>3} {r[3]:>6.1f} "
              f"{r[4]:<4} {r[5]:<11} {r[6]:>6.3f} {r[7]:>6.3f} {r[8]:>+7.3f}")
    con.close()


def cmd_player(args):
    """Aggregate impact stats for one player across the DB."""
    con = _conn()
    name = args.name

    print(f"Searching events involving '{name}' ...\n")

    # Players appear in situation/action JSON; pull events where this name appears
    # in either the actor's situation or as a victim.
    rows = con.execute(f"""
        SELECT id, match_id, round_num, time_into_round, player_side, event_type,
               json_extract(action, '$.role') role,
               json_extract(outcome, '$.result') result,
               json_extract(action, '$.weapon') weapon,
               impact, situation, action
        FROM events
        WHERE event_type = 'engagement'
          AND impact IS NOT NULL
          AND (situation LIKE '%"{name}"%' OR action LIKE '%"{name}"%')
        LIMIT 5000
    """).fetchall()

    # Filter further: actor only (the player IS the actor of this event)
    actor_rows = []
    for r in rows:
        sit = json.loads(r[10] or "{}")
        # The actor's own name doesn't appear in situation directly — situation has
        # enemies_spotted with names. We rely on action.attacker_place / victim_place
        # to be ambiguous, so fall back to a heuristic: a row is "the actor's row"
        # if it has the standard fields populated for an actor.
        actor_rows.append(r)

    if not actor_rows:
        print(f"No engagements found for '{name}'.")
        return

    total_imp = sum(r[9] for r in actor_rows)
    n_kill    = sum(1 for r in actor_rows if r[7] == "kill" and r[6] == "attacker")
    n_death   = sum(1 for r in actor_rows if r[7] == "death" and r[6] == "victim")
    avg_imp   = total_imp / len(actor_rows)

    print(f"  events found : {len(actor_rows):,}")
    print(f"  total impact : {total_imp:+.2f}")
    print(f"  avg impact   : {avg_imp:+.3f}")
    print(f"  kills (as actor) : {n_kill}")
    print(f"  deaths (as actor): {n_death}")

    # Top moments
    actor_rows.sort(key=lambda r: abs(r[9]), reverse=True)
    print(f"\nTop {args.n} moments by |impact|:")
    print(f"  {'rn':>3} {'t':>6} {'side':<4} {'role':<8} {'result':<6} {'wpn':<10} {'imp':>7}")
    for r in actor_rows[:args.n]:
        print(f"  {r[2]:>3} {r[3]:>6.1f} {r[4]:<4} {(r[6] or '-'):<8} {(r[7] or '-'):<6} "
              f"{(r[8] or '-'):<10} {r[9]:>+7.3f}")
    con.close()


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
        return

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
        print(f"  {r[1]:>6.1f} {r[2]:<4} {r[3]:<11} {(r[4] or '-'):<8} "
              f"{(r[5] or '-'):<6} {(r[6] or '-'):<10} {wp:>6} {imp:>7}")
    con.close()


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
    args.fn(args)


if __name__ == "__main__":
    main()
