"""
Synthetic RawTables fixtures (P3.9.1): every story at 64 and 128 tick matches
STORY_FACTS, has a tick row for every tick of every round for all 4 players, and moves
the way the story says (B at 250 u/s, C's retreat, the flash, the drop).

    .venv\\Scripts\\python.exe test\\engine\\test_synthetic.py
"""
import sys

import polars as pl

from _harness import check, close, section, summary
from _synthetic import PLAYERS, STORIES, STORY_FACTS, make_raw_tables, scale_tick

from engine import config
from engine.actions import extractor
from engine.state.geometry import FlatGeometry

ID = {p: sid for p, (sid, _) in PLAYERS.items()}

for tr in (64, 128):
    T = lambda t128: scale_tick(t128, tr)  # noqa: E731
    for story in STORIES:
        section(f"{story} @ {tr} tick")
        f   = STORY_FACTS[story]
        raw = make_raw_tables(story, tr)
        check("validates", raw.validate() == [])
        check("meta", raw.meta["map"] == "de_test" and raw.meta["tick_rate"] == tr
              and len(raw.meta["demo_sha256"]) == 64)

        # ── counts ──
        counts = {"rounds": raw.rounds.height, "kills": raw.kills.height, "smokes": raw.smokes.height,
                  "infernos": raw.infernos.height, "grenades": raw.grenades.height,
                  "flashes": raw.flashes.height, "bomb_events": raw.bomb.height,
                  "damages": raw.damages.height, "items": raw.items.height,
                  "weapon_fire": raw.weapon_fire.height}
        for k, v in counts.items():
            check(f"{k} = {f[k]}", v == f[k], f"got {v}")
        for rnd in (1, 2):
            n = raw.kills.filter(pl.col("round_num") == rnd).height
            check(f"round {rnd} kills = {f['kills_round'][rnd]}", n == f["kills_round"][rnd])
            evs = raw.bomb.filter(pl.col("round_num") == rnd).sort("tick")["event"].to_list()
            check(f"round {rnd} bomb events", evs == f["bomb_round"][rnd], str(evs))
        got_kills = [(t, a, v) for t, a, v in raw.kills.select("tick", "attacker_steamid", "victim_steamid").rows()]
        want_kills = [(T(t), ID[a], ID[v]) for t, a, v in f["kill_events"]]
        check("kill events (tick, attacker, victim)", sorted(got_kills) == sorted(want_kills), str(got_kills))

        rounds = raw.rounds.sort("round_num").rows(named=True)
        check("winners", [r["winner_side"] for r in rounds] == [f["winner"][1], f["winner"][2]])
        check("end reasons", [r["end_reason"] for r in rounds] == [f["end_reason"][1], f["end_reason"][2]])
        check("round 1 end / official end / round 2 start",
              (rounds[0]["end_tick"], rounds[0]["official_end_tick"], rounds[1]["start_tick"])
              == (T(f["r1_end"]), T(f["r1_official_end"]), T(f["r2_start"])))
        check("round 1 freeze_end 1000", rounds[0]["freeze_end_tick"] == T(1000))
        check("round 2 scores follow round 1",
              (rounds[1]["ct_score_before"], rounds[1]["t_score_before"])
              == (int(f["winner"][1] == "ct"), int(f["winner"][1] == "t")))

        # ── tick table ──
        complete = True
        for r in rounds:
            want = r["official_end_tick"] - r["start_tick"] + 1
            sub = raw.ticks.filter(pl.col("round_num") == r["round_num"])
            per = sub.group_by("steamid").agg(pl.col("tick").n_unique().alias("n"),
                                              pl.col("tick").min().alias("lo"),
                                              pl.col("tick").max().alias("hi"))
            complete &= (per.height == 4 and (per["n"] == want).all()
                         and (per["lo"] == r["start_tick"]).all() and (per["hi"] == r["official_end_tick"]).all())
        check("every tick of every round for all 4 players", complete)
        check("no duplicate (tick, steamid)", raw.ticks.select("tick", "steamid").is_duplicated().sum() == 0)

        # ── dead players ──
        dead_ok = True
        for t, _, v in f["kill_events"]:
            rows = raw.ticks.filter((pl.col("steamid") == ID[v]) & (pl.col("round_num") == 1)
                                    & (pl.col("tick") >= T(t)))
            before = raw.ticks.filter((pl.col("steamid") == ID[v]) & (pl.col("tick") == T(t) - 1))
            dead_ok &= (not rows["is_alive"].any() and (rows["health"] == 0).all()
                        and rows["active_weapon"].null_count() == rows.height
                        and (rows["equipment_value"] == 0).all() and bool(before["is_alive"][0]))
        check("victims dead from the kill tick on (health 0, no weapon)", dead_ok)
        r1_alive_end = raw.ticks.filter((pl.col("tick") == T(f["r1_end"])) & pl.col("is_alive"))
        check("alive count at round 1 end", r1_alive_end.height == 4 - f["kills_round"][1])

        # ── movement ──
        b = raw.ticks.filter((pl.col("steamid") == ID["B"]) & (pl.col("tick") > T(1000))
                             & (pl.col("tick") <= T(1990)))
        speed = ((b["vx"] ** 2 + b["vy"] ** 2).sqrt())
        check("B runs at ~250 u/s (1000..1990)", close(float(speed.mean()), 250.0, 2.0)
              and close(float(speed.min()), 250.0, 3.0), f"mean {float(speed.mean()):.1f}")
        bpos = raw.ticks.filter((pl.col("steamid") == ID["B"]) & (pl.col("tick") == T(1000) + tr))
        check("B covered 250 units in the first second", close(float(bpos["x"][0]), 250.0, 3.0))
        a = raw.ticks.filter((pl.col("steamid") == ID["A"]) & (pl.col("round_num") == 1)
                             & (pl.col("tick") <= T(2000)))
        check("A stands still and looks +x", (a["vx"] == 0).all() and (a["yaw"] == 0).all())
        check("HE damage C 30 / D 20 at 1800",
              sorted(raw.damages.filter(pl.col("weapon") == "hegrenade")["dmg_health"].to_list()) == [20, 30])
        sm = raw.smokes.row(0, named=True)
        check("smoke 1500 .. 1500+18 s at (1000,0,0)",
              (sm["start_tick"], sm["end_tick"], sm["x"]) == (T(1500), T(1500) + 18 * tr, 1000.0))
        c4 = raw.ticks.filter((pl.col("steamid") == ID["D"]) & (pl.col("round_num") == 1))
        check("D carries the c4 until plant_complete",
              bool(c4.filter(pl.col("tick") == T(3000) - 1)["has_c4"][0])
              and not c4.filter(pl.col("tick") == T(3000))["has_c4"][0])
        check("D is_planting during the plant",
              bool(c4.filter(pl.col("tick") == T(2950))["is_planting"][0])
              and not c4.filter(pl.col("tick") == T(3000))["is_planting"][0])
        ks = raw.ticks.filter((pl.col("tick") == rounds[0]["official_end_tick"]))
        kills_by = {sid: n for sid, n in ks.select("steamid", "kills_round").rows()}
        check("kills_round tallies kills", all(
            kills_by[ID[p]] == sum(a_ == p for _, a_, _ in f["kill_events"]) for p in PLAYERS))

        # ── story specifics ──
        if story == "flash":
            bl = {sid: d for sid, d in raw.flashes.select("player_steamid", "blind_duration").rows()}
            check("flash blinds as STORY_FACTS['blinded'] says (D 2.5 s, C 1.0 s)",
                  bl == {ID[p]: d for p, d in f["blinded"].items()})
            check("C blind at the kill (victim_blind)", raw.kills.filter(
                pl.col("victim_steamid") == ID["C"])["victim_blind"].to_list() == [True])
            d = raw.ticks.filter((pl.col("steamid") == ID["D"]) & (pl.col("tick") == T(1900) + tr))
            check("D has 1.5 s of blind left one second later", close(float(d["flash_duration"][0]), 1.5, 1e-4))
        else:
            check("nobody blind", not raw.kills["victim_blind"].any() and raw.ticks["flash_duration"].max() == 0)
        if story == "double_kill_same_tick":
            check("two kills on tick 2000", raw.kills.filter(pl.col("tick") == T(2000)).height == 2)
        if story == "trade":
            tk = raw.kills.filter(pl.col("is_trade"))
            check("trade kills as STORY_FACTS['trade_kills'] says (B kills C at 2100)",
                  tk.select("tick", "attacker_steamid", "victim_steamid").rows()
                  == [(T(t), ID[a_], ID[v]) for t, a_, v in f["trade_kills"]])
            check("bomb explodes at round end",
                  raw.bomb.filter(pl.col("event") == "explode")["tick"].to_list() == [T(3000) + 40 * tr])
        else:
            check("no trade flags", not raw.kills["is_trade"].any())
        if story == "retreat":
            rt = f["retreat"]
            c = raw.ticks.filter(pl.col("steamid") == ID[rt["player"]])
            y0 = float(c.filter(pl.col("tick") == T(rt["from_tick"]))["y"][0])
            y1 = float(c.filter(pl.col("tick") == T(rt["to_tick"]))["y"][0])
            check("C runs STORY_FACTS['retreat'] units (1250) away from the bomb (+y)",
                  close(y1 - y0, rt["units"], 1.0))
            check("C survives round 1", bool(c.filter(pl.col("tick") == T(f["r1_end"]))["is_alive"][0]))
            bt = raw.ticks.filter((pl.col("steamid") == ID["B"]) & (pl.col("tick") == T(3950)))
            check("B defusing with kit at 3950", bool(bt["is_defusing"][0]) and bool(bt["has_defuser"][0]))
        if story == "drop":
            ev = raw.items.select("tick", "steamid", "event", "item").rows()
            check("items as STORY_FACTS['item_events'] says (A drops m4a1 at 1100, B picks it up at 1150)",
                  ev == [(T(t), ID[p], e, i) for t, p, e, i in f["item_events"]])
            bi = raw.ticks.filter((pl.col("steamid") == ID["B"]))
            check("B holds the m4a1 only after the pickup",
                  "weapon_m4a1" not in bi.filter(pl.col("tick") == T(1150) - 1)["inventory"][0].to_list()
                  and "weapon_m4a1" in bi.filter(pl.col("tick") == T(1150))["inventory"][0].to_list())
        else:
            check("no item events", raw.items.height == 0)

section("spotted_by=False: the parser's all-null spotted_by (LOW-63)")
for tr in (64, 128):
    full, nul = make_raw_tables("base", tr), make_raw_tables("base", tr, spotted_by=False)
    check(f"@{tr}: default fills spotted_by", full.ticks["spotted_by"].null_count() == 0)
    check(f"@{tr}: spotted_by=False -> null on every row, strict validate passes",
          nul.ticks["spotted_by"].null_count() == nul.ticks.height and nul.validate(strict=True) == [])
    check(f"@{tr}: every other column identical", nul.ticks.drop("spotted_by").equals(full.ticks.drop("spotted_by")))
snaps = extractor.run(make_raw_tables("base", spotted_by=False), config.load(), match_key="m1",
                      geometry=FlatGeometry())[0]
check("the builder reads an all-null spotted_by as unknown (spotted_by_count None), not 0",
      {p.spotted_by_count for s in snaps for p in s.players if p.alive == 1} == {None})

sys.exit(summary())
