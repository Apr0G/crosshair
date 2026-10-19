"""
Action invariants (P4.7.3) on every synthetic story at 64 and 128 tick, checked against
the raw tables (not the extractor's own lookups):
- every action validates; no None type; every A.5 type has an extractor;
- the actor was alive at tick - 1. Grenades are judged at their throw tick (they may go
  off after the thrower died) and so are kills/damage by grenade weapons; pseudo-actions
  (pop/fade/start/end) and team_buy are exempt (the victim-side / no-actor entries);
- every tick (tick, window, state ticks) inside its round;
- windows ordered and non-overlapping;
- each kill row -> exactly one kill action; each smoke -> one smoke action, one pop,
  one fade; pseudo-actions point at an existing parent with the same actor;
- unique ids, enablers point at existing actions, extraction is idempotent.

    .venv\\Scripts\\python.exe test\\engine\\test_actions_invariants.py
"""
import sys
from collections import Counter

from _harness import check, section, summary
from _synthetic import PLAYERS, STORIES, make_raw_tables

from engine import config
from engine.actions import extractor, windows
from engine.actions.types import PSEUDO_TYPES, TYPE_KIND, missing_extractors
from engine.state import sampler
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash, weapon_class

CFG = config.load()
GEO = FlatGeometry()
SID_OF = {player_id_hash(s): s for s, _ in PLAYERS.values()}
GRENADE_TYPES = ("flash", "he", "smoke", "molotov", "decoy")

section("registry")
check("every non-pseudo A.5 type has an extractor", missing_extractors() == [], str(missing_extractors()))

for tr in (128, 64):
    for story in STORIES:
        section(f"{story} @ {tr}")
        raw = make_raw_tables(story, tr)
        snaps, acts, ws = extractor.run(raw, CFG, match_key="m1", geometry=GEO)
        alive = {(s, t): a for s, t, a in zip(raw.ticks["steamid"].to_list(), raw.ticks["tick"].to_list(),
                                               raw.ticks["is_alive"].to_list())}
        rounds = {r["round_num"]: (r["start_tick"], sampler.terminal_tick(r)) for r in raw.rounds.to_dicts()}
        ids = {a.action_id: a for a in acts}

        check("every action validates", all(a.validate() == [] for a in acts),
              str([(a.type, a.validate()) for a in acts if a.validate()][:2]))
        check("no None / unknown type", all(a.type is not None and a.type in TYPE_KIND for a in acts))
        check("unique ids", len(ids) == len(acts))

        bad = []
        for a in acts:
            if a.type in PSEUDO_TYPES or a.actor_id_hash is None:
                continue
            sid = SID_OF[a.actor_id_hash]
            t = a.tick if a.tick is not None else a.window_start_tick
            if a.type in GRENADE_TYPES or (a.type in ("kill", "damage") and a.params.get("weapon_class") == "grenade"):
                g = raw.grenades.filter(raw.grenades["thrower_steamid"] == sid).to_dicts()
                t = max((x["throw_tick"] for x in g if x["throw_tick"] <= t), default=t) + 1
            if not alive.get((sid, t - 1), False):
                bad.append((a.type, t))
        check("actor alive at tick - 1 (grenades: at the throw)", not bad, str(bad[:3]))

        def inside(a) -> bool:
            lo, hi = rounds[a.round_num]
            ts = [a.tick, a.window_start_tick, a.window_end_tick, a.state_pre_tick, a.state_post_tick]
            return all(lo <= t <= hi for t in ts if t is not None)
        check("every tick inside its round", all(inside(a) for a in acts),
              str([(a.type, a.tick, a.window_start_tick) for a in acts if not inside(a)][:3]))
        check("windows ordered", ws == sorted(ws, key=lambda w: (w.round_num, w.start_tick, w.end_tick)))
        check("windows never overlap", not windows.overlaps(ws))
        check("window members exist and are instant", all(m in ids and ids[m].kind == "instant"
                                                           for w in ws for m in w.member_action_ids))

        krows = Counter((r["tick"], player_id_hash(r["attacker_steamid"]), player_id_hash(r["victim_steamid"]))
                        for r in raw.kills.to_dicts())
        kacts = Counter((a.tick, a.actor_id_hash, a.params["victim_id"]) for a in acts if a.type in ("kill", "team_kill"))
        check("each kill row -> exactly one kill action", krows == kacts)
        n_sm = raw.smokes.height
        cnt = Counter(a.type for a in acts)
        check("each smoke -> one smoke, one pop, one fade",
              (cnt["smoke"], cnt["smoke_pop"], cnt["smoke_fade"]) == (n_sm, n_sm, n_sm))
        check("pseudo-actions point at a parent with the same actor", all(
            a.params.get("parent_action_id") in ids and ids[a.params["parent_action_id"]].actor_id_hash == a.actor_id_hash
            for a in acts if a.type in PSEUDO_TYPES))
        check("enablers point at existing actions", all(e["action_id"] is None or e["action_id"] in ids
                                                        for a in acts for e in a.enablers))
        a2, w2 = extractor.extract(raw, snaps, CFG, match_key="m1", geometry=GEO)
        check("idempotent: extract twice on the same snapshots -> identical", a2 == acts and w2 == ws)
        check("weapon classes known", all(a.params.get("weapon_class") in (None, weapon_class(a.params.get("weapon")))
                                          for a in acts if a.type == "kill"))

sys.exit(summary())
