"""
Movement actions (P4.2.5, P4.3.3-P4.3.5) from the real builder's grid snapshots with
FlatGeometry, plus the pure span helpers for the negative cases the stories lack
(callout jitter, a player who moves every 3 s).

    .venv\\Scripts\\python.exe test\\engine\\test_actions_movement.py
"""
import sys

from _harness import check, section, summary
from _synthetic import PLAYERS, STORIES, make_raw_tables, scale_tick

from engine import config
from engine.actions import extractor, movement
from engine.state.geometry import FlatGeometry
from engine.state.players import player_id_hash

CFG = config.load()
GEO = FlatGeometry()
PID = {n: player_id_hash(s) for n, (s, _) in PLAYERS.items()}
NAME = {v: k for k, v in PID.items()}
RUNS = {}


def run(story: str, tr: int = 128):
    if (story, tr) not in RUNS:
        RUNS[(story, tr)] = extractor.run(make_raw_tables(story, tr), CFG, match_key="m1", geometry=GEO)
    return RUNS[(story, tr)]


def of(acts, type_, rnd=1):
    return [a for a in acts if a.type == type_ and a.round_num == rnd]


# ── peek ──
section("peek (base): B walks out of the smoke toward C and D")
for tr in (128, 64):
    _, acts, _ = run("base", tr)
    pk = of(acts, "peek")
    check(f"@{tr}: two peeks, both by B", len(pk) == 2 and all(p.actor_id_hash == PID["B"] for p in pk),
          str([(NAME[p.actor_id_hash], p.tick) for p in pk]))
    check(f"@{tr}: targets C and D", sorted(p.params["target_id"] for p in pk) == sorted([PID["C"], PID["D"]]))
    check(f"@{tr}: at the first grid sample where sight begins (1640 at 128)",
          all(p.tick == scale_tick(1640, tr) for p in pk), str([p.tick for p in pk]))
    check(f"@{tr}: A-D sight returns when the smoke fades (no movement) -> no peek by or at A/D",
          not [p for p in pk if PID["A"] in (p.actor_id_hash, p.params["target_id"])])
    check(f"@{tr}: params", all(set(p.params) == {"from_callout", "target_id", "target_callout",
                                                  "target_spotted_me_first"} for p in pk))
    check(f"@{tr}: no spotted mask between the samples -> target_spotted_me_first None",
          all(p.params["target_spotted_me_first"] is None for p in pk))
    check(f"@{tr}: ids unique (salted by target)", len({p.action_id for p in pk}) == 2)
_, acts, _ = run("base")
check("from_callout is B's callout at the previous grid sample", of(acts, "peek")[0].params["from_callout"]
      == GEO.callout((1576 - 1000) * 2000 / 1024, 0, 0))

# ── rotation ──
section("rotation (base): B from its start cell to D's side")
for tr in (128, 64):
    _, acts, _ = run("base", tr)
    ro = of(acts, "rotation")
    check(f"@{tr}: exactly one rotation in round 1, by B", len(ro) == 1 and ro[0].actor_id_hash == PID["B"],
          str([(NAME[r.actor_id_hash], r.window_start_tick) for r in ro]))
    r = ro[0]
    check(f"@{tr}: window [last sample in the start cell, first sample in the end cell]",
          (r.window_start_tick, r.window_end_tick) == (scale_tick(1256, tr), scale_tick(1832, tr)),
          f"{r.window_start_tick}, {r.window_end_tick}")
    check(f"@{tr}: from/to callouts", (r.params["from_callout"], r.params["to_callout"])
          == (GEO.callout(0, 0, 0), GEO.callout(2000, 0, 0)))
    check(f"@{tr}: nav_dist 1536 (three 512 cells)", r.params["nav_dist_units"] == 1536.0)
    check(f"@{tr}: toward_bomb (D carries it at x=3000)", r.params["toward_bomb"] is True)
    check(f"@{tr}: duration_s = window / tick rate", r.params["duration_s"] == round(
        (r.window_end_tick - r.window_start_tick) / tr, 4))
check("D's 400 u walk after the plant is too short for a rotation",
      not [r for r in of(acts, "rotation") if r.actor_id_hash == PID["D"]])
_, acts, _ = run("double_kill_same_tick")
check("B dies 1.3 s after arriving: new callout not held 2 s -> no rotation", not of(acts, "rotation"))

samples = [(0, "a"), (64, "a"), (128, "a"), (192, "a"), (256, "a"), (320, "b"), (384, "b"),
           (448, "a"), (512, "a"), (576, "a"), (640, "a"), (704, "a")]
check("jitter across a border and back -> none", movement.rotation_spans(samples, 256, lambda a, b: True) == [])
far_walk = samples[:5] + [(320, "b"), (384, "b"), (448, "b"), (512, "b"), (576, "b")]
check("held >= 2 s in a far callout -> one", len(movement.rotation_spans(far_walk, 256, lambda a, b: True)) == 1)
check("held but near (< rotation_min_units) -> none", movement.rotation_spans(far_walk, 256, lambda a, b: False) == [])

# ── hold ──
section("hold")
for tr in (128, 64):
    _, acts, _ = run("base", tr)
    ha = [h for h in of(acts, "hold") if h.actor_id_hash == PID["A"]]
    check(f"@{tr}: A stands still -> one hold from freeze_end", len(ha) == 1 and ha[0].window_start_tick == scale_tick(1000, tr))
    check(f"@{tr}: hold >= 7.8 s", ha[0].params["duration_s"] >= 7.8, str(ha[0].params["duration_s"]))
    check(f"@{tr}: callout and facing_callout (500 u along +x)",
          ha[0].params["callout"] == GEO.callout(0, 300, 0) and ha[0].params["facing_callout"] == GEO.callout(500, 300, 0))
    hc = [h for h in of(acts, "hold") if h.actor_id_hash == PID["C"]]
    check(f"@{tr}: C's hold ends before C dies at 2000", len(hc) == 1 and hc[0].window_end_tick < scale_tick(2000, tr))
    hb = [h for h in of(acts, "hold") if h.actor_id_hash == PID["B"]]
    check(f"@{tr}: B holds only after arriving", all(h.window_start_tick >= scale_tick(2024, tr) for h in hb) and hb)
mover = []
for k in range(40):        # 1 s still, 2 s walking, repeat: never 5 s still
    t = k * 64
    mover.append((t, "a", 0.0 if (k % 6) < 2 else 250.0))
check("player who moves every 3 s -> no hold", movement.hold_runs(mover, 20.0, 5 * 128) == [])
still = [(k * 64, "a", 0.0) for k in range(12)]
check("still 5.5 s -> one hold", movement.hold_runs(still, 20.0, 5 * 128) == [(0, 11)])
check("still but switching callout -> none",
      movement.hold_runs([(t, "a" if t < 320 else "b", v) for t, _, v in still], 20.0, 5 * 128) == [])

# ── save ──
section("save / retreat")
for tr in (128, 64):
    _, acts, _ = run("retreat", tr)
    sv = of(acts, "save")
    check(f"@{tr}: retreat story: one save, by C", len(sv) == 1 and sv[0].actor_id_hash == PID["C"],
          str([(NAME[s.actor_id_hash], s.window_start_tick) for s in sv]))
    s = sv[0]
    check(f"@{tr}: window inside the run 3000..3640 (+ one grid step)",
          scale_tick(3000, tr) <= s.window_start_tick < s.window_end_tick <= scale_tick(3640, tr) + tr // 2,
          f"{s.window_start_tick}, {s.window_end_tick}")
    check(f"@{tr}: params", (s.params["from_callout"], s.params["to_callout"], s.params["teammates_alive"])
          == (GEO.callout(3000, 300, 0), GEO.callout(3000, 1550, 0), 1))
for story in STORIES:
    if story != "retreat":
        _, acts, _ = run(story)
        check(f"{story}: no save", not [a for a in acts if a.type == "save"])
dist = [100.0, 100.0, 500.0, 1000.0, 1000.0]
ticks = [0, 128, 256, 384, 512]
check("retreat with a shot fired -> none",
      movement.retreat_spans(ticks, dist, ["bomb"] * 5, 640, 800, lambda i, j: True) == [])
check("retreat without a shot -> trimmed span", movement.retreat_spans(ticks, dist, ["bomb"] * 5, 640, 800,
                                                                       lambda i, j: False) == [(1, 3)])

# ── alive ──
section("actors alive throughout")
for story in STORIES:
    snaps, acts, _ = run(story)
    by = {(s.meta.round_num, s.meta.tick): s for s in snaps}
    ok = True
    for a in acts:
        if a.type not in ("peek", "rotation", "hold", "save"):
            continue
        ticks = [a.tick] if a.tick is not None else [a.window_start_tick, a.window_end_tick]
        for t in ticks:
            s = by.get((a.round_num, t))
            i = s.index_of(a.actor_id_hash) if s is not None else None
            ok &= s is not None and i is not None and s.players[i].alive == 1
    check(f"{story}: movement actor alive at every sample its action uses", ok)

sys.exit(summary())
