"""
awpy 3 parser -> Appendix A.1 tables, on a fake awpy-shaped demo (P2.5.1, P1.4.1).

    .venv\\Scripts\\python.exe test\\engine\\test_awpy3_parser.py [path\\to\\demo.dem]

No demo is needed: `FakeDemo` carries hand-built awpy 3.1.0-shaped frames for two
rounds and four players (sides swap at "halftime" after round 1). With a real demo
path, the same contract checks also run on a real parse; without one that section is
skipped (the fake-demo checks still decide the exit code).
"""
import sys
import tempfile
from pathlib import Path

import polars as pl

from _harness import check, section, skip, summary

from engine.parse import awpy3_parser as ap

# ── fake demo ─────────────────────────────────────────────────────────────────

TR = 128
A, B, C, D = 76561198000000001, 76561198000000002, 76561198000000003, 76561198000000004
NAMES = {A: "alpha", B: "bravo", C: "charlie", D: "delta"}
CTS, TS = "counter-terrorist", "terrorist"


def _side_of(sid: int, rnd: int) -> str:
    first_ct = sid in (A, B)
    return CTS if (first_ct == (rnd == 1)) else TS


def _round_of(tick: int) -> int:
    return 1 if tick < 700 else 2


class FakeEvents(dict):
    """dict with awpy's `names`."""
    @property
    def names(self) -> list[str]:
        return sorted(self)


class FakeDemo:
    header = {"map_name": "de_mirage", "server_name": "FACEIT.com server", "client_name": "GOTV",
              "build_num": 10524, "demo_version_name": "valve_demo_2", "game_directory": "csgo",
              "playback_ticks": TR * 2400, "playback_time": 2400.0, "playback_frames": 150000}
    tick_rate = 128.0

    def __init__(self):
        self.snapshot_calls: list[tuple] = []
        self.rounds = pl.DataFrame({
            "round_num": [1, 2], "start_tick": [100, 700], "freeze_end_tick": [200, 800],
            "end_tick": [600, 1200], "official_end_tick": [700, None], "winner": [3, 2],
            "winner_side": [CTS, TS], "reason": [7, 1], "reason_name": ["bomb_defused", "target_bombed"],
            "is_knife_round": [False, False],
        }, schema_overrides={"round_num": pl.Int32, "start_tick": pl.Int32, "freeze_end_tick": pl.Int32,
                             "end_tick": pl.Int32, "official_end_tick": pl.Int32})
        kill_rows = [  # tick, attacker, victim, assister, assistedflash
            (50, A, C, None, False),          # warmup: outside every round window -> dropped
            (400, A, C, B, True),
            (420, D, B, None, False),
            (1000, A, B, None, False),        # round 2: A and B are both T -> team kill
        ]
        self.kills = pl.DataFrame({
            "tick": [r[0] for r in kill_rows],
            "attacker_steamid": [r[1] for r in kill_rows], "victim_steamid": [r[2] for r in kill_rows],
            "assister_steamid": [r[3] for r in kill_rows],
            "attacker_side": [_side_of(r[1], _round_of(r[0])) for r in kill_rows],
            "victim_side": [_side_of(r[2], _round_of(r[0])) for r in kill_rows],
            "assistedflash": [r[4] for r in kill_rows],
            "attacker_x": [0.0, 0.0, 10.0, 0.0], "attacker_y": [0.0] * 4, "attacker_z": [0.0] * 4,
            "victim_x": [300.0, 300.0, 10.0, 0.0], "victim_y": [400.0, 400.0, 0.0, 50.0], "victim_z": [0.0] * 4,
            "weapon": ["ak47", "ak47", "awp", "deagle"], "headshot": [True, True, False, False],
            "noscope": [False, False, True, False], "penetrated": [False, False, True, False],
            "thrusmoke": [False, False, False, True], "hitgroup": [1, 1, 2, 3],
            "is_trade": [False, False, True, False],
        }, schema_overrides={"attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64,
                             "assister_steamid": pl.UInt64, "tick": pl.Int32})
        self.damages = pl.DataFrame({
            "tick": [390, 400, 420, 1000],
            "attacker_steamid": [A, A, D, A], "victim_steamid": [C, C, B, B],
            "attacker_side": [CTS, CTS, TS, TS], "victim_side": [TS, TS, CTS, TS],
            "weapon": ["ak47", "ak47", "awp", "deagle"], "dmg_health": [27, 73, 100, 100],
            "dmg_armor": [3, 0, 0, 0], "hitgroup": [2, 1, 2, 3],
            "hitgroup_name": ["chest", "head", "chest", "stomach"],
        }, schema_overrides={"attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64})
        self.bomb = pl.DataFrame({
            "tick": [210, 300, 305, 310, 350, 550, 900, 950],
            "event": ["pickup", "start_plant", "interrupt_plant", "start_plant", "finish_plant", "defuse",
                      "start_plant", "finish_plant"],
            "steamid": [C, C, C, C, C, A, A, A],
            "bombsite": [None, None, None, None, "A", "A", None, "B"],
            "x": [1.0] * 8, "y": [2.0] * 8, "z": [3.0] * 8,
        }, schema_overrides={"steamid": pl.UInt64})
        # Trajectories: smoke (slot 10, serial 1), flash (11), molotov (12) in round 1;
        # an HE reusing slot 10 with serial 2 in round 2.
        traj = []
        for eid, ser, sid, typ, t0, t1 in [(10, 1, B, "smoke", 250, 300), (11, 1, D, "flashbang", 380, 395),
                                            (12, 1, C, "molotov", 320, 330), (10, 2, A, "he", 900, 920)]:
            for t in range(t0, t1 + 1):
                traj.append({"tick": t, "thrower_steamid": sid, "thrower_name": NAMES[sid], "thrower_side": "x",
                             "type": typ, "entity_id": eid, "entity_serial": ser,
                             "x": float(t), "y": float(eid), "z": 5.0})
        self.grenades = pl.DataFrame(traj, schema_overrides={"thrower_steamid": pl.UInt64})
        self.smokes = pl.DataFrame({"start_tick": [301], "end_tick": [600], "thrower_steamid": [B],
                                    "thrower_name": ["bravo"], "thrower_side": [CTS], "entity_id": [10],
                                    "entity_serial": [1], "x": [-1.0], "y": [-2.0], "z": [-3.0]},
                                   schema_overrides={"thrower_steamid": pl.UInt64})
        self.fires = pl.DataFrame({"start_tick": [331], "end_tick": [450], "thrower_steamid": [C],
                                   "thrower_name": ["charlie"], "thrower_side": [TS], "entity_id": [40],
                                   "entity_serial": [7], "x": [9.0], "y": [8.0], "z": [7.0],
                                   "type": ["molotov"]}, schema_overrides={"thrower_steamid": pl.UInt64})
        self.blinds = pl.DataFrame({"tick": [395, 395], "attacker_steamid": [D, D],
                                    "victim_steamid": [A, B], "duration": [2.0, 1.0]},
                                   schema_overrides={"attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64})
        self.shots = pl.DataFrame({"tick": [399, 400, 1000], "steamid": [A, A, A],
                                   "weapon": ["ak47", "ak47", "deagle"]},
                                  schema_overrides={"steamid": pl.UInt64})
        self.item_events = pl.DataFrame({"tick": [205, 260, 270], "action": ["purchase", "drop", "pickup"],
                                         "steamid": [A, A, B], "item": ["ak47", "deagle", "deagle"]},
                                        schema_overrides={"steamid": pl.UInt64})
        self.players = pl.DataFrame({"steamid": [A, B, C, D], "name": [NAMES[s] for s in (A, B, C, D)],
                                     "side": [TS, TS, CTS, CTS], "team_clan_name": ["x", "x", "y", "y"]},
                                    schema_overrides={"steamid": pl.UInt64})
        self.events = FakeEvents({"player_footstep": pl.DataFrame(
            {"tick": [500, 501], "user_steamid": [str(A), str(C)], "user_name": ["alpha", "charlie"]})})

    def snapshots(self, *, start_tick=None, end_tick=None, **kw):
        self.snapshot_calls.append((start_tick, end_tick))
        rows = []
        for t in range(start_tick, end_tick + 1):
            rnd = _round_of(t)
            for sid in (A, B, C, D):
                dead = (sid == C and rnd == 1 and t >= 400) or (sid == B and rnd == 1 and t >= 420)
                defusing = sid == A and rnd == 1 and (450 <= t < 460 or 500 <= t <= 549)
                rows.append({
                    "tick": t, "steamid": sid, "name": NAMES[sid], "side": _side_of(sid, rnd),
                    "x": float(t), "y": 1.0, "z": 2.0, "velocity_x": 1.0, "velocity_y": 0.0,
                    "velocity_z": 0.0, "velocity": 1.0, "pitch": 0.0, "yaw": 90.0,
                    "health": 0 if dead else 100, "armor": 100, "has_helmet": True,
                    "has_defuser": sid == A, "has_bomb": sid == C and rnd == 1 and t < 350,
                    "active_weapon": "ak47", "primary_weapon": "ak47", "secondary_weapon": "glock",
                    "fire_grenades": 0, "smoke_grenades": 0, "he_grenades": 0, "flashbangs": 0,
                    "decoy_grenades": 0, "inventory": ["ak47", "glock"], "equipment_value": 3700,
                    "equipment_value_round_start": 3700, "money": 800, "is_crouched": False,
                    "is_walking": False, "is_jumping": False, "is_in_bomb_zone": False,
                    "is_scoped": False, "is_reloading": None, "is_silent_reloading": None,
                    "is_defusing": defusing, "is_blinded": (sid == A and 395 <= t <= 410) or (sid == B and 395 <= t <= 420),
                    "flash_duration": 2.0 if sid == A else 0.0,
                })
        return pl.DataFrame(rows, schema_overrides={"steamid": pl.UInt64, "tick": pl.Int32,
                                                    "is_reloading": pl.Boolean})


# ── checks ────────────────────────────────────────────────────────────────────

def check_contract(tables: dict, meta: dict, label: str) -> None:
    section(f"A.1 contract ({label})")
    check("exactly the A.1 tables", set(tables) == set(ap.TABLES), f"{sorted(tables)}")
    for name, schema in ap.A1_SCHEMAS.items():
        df = tables.get(name)
        if df is None:
            check(f"{name}: present", False)
            continue
        got = dict(df.schema)
        extra = [c for c in got if c not in schema]
        check(f"{name}: columns in A.1 order (+ optional extras only)",
              list(got)[:len(schema)] == list(schema) and set(extra) <= set(ap.A1_OPTIONAL.get(name, {})),
              f"{[c for c in schema if c not in got]} missing, {extra} extra")
        bad = {c: (str(got[c]), str(dt)) for c, dt in schema.items() if c in got and got[c] != dt}
        check(f"{name}: dtypes", not bad, f"{bad}")
    for k in ap.META_SCHEMA:
        check(f"meta has {k}", k in meta)
    check("meta n_rounds and players present", "n_rounds" in meta and "players" in meta)
    check("meta frame matches A.1", dict(ap.meta_frame(meta).schema) == ap.META_SCHEMA)
    bomb_ev = set(tables["bomb"]["event"].to_list())
    check("bomb events within the A.1 vocabulary", bomb_ev <= set(ap.BOMB_EVENTS), f"{bomb_ev}")
    gtypes = set(tables["grenades"]["type"].drop_nulls().to_list())
    check("grenade types within the A.1 vocabulary", gtypes <= set(ap.GRENADE_TYPES), f"{gtypes}")
    check("item events within the A.1 vocabulary", set(tables["items"]["event"].to_list()) <= set(ap.ITEM_EVENTS))
    sides = set(tables["ticks"]["side"].drop_nulls().to_list())
    check("ticks side is ct/t", sides <= {"ct", "t"}, f"{sides}")


def check_rawtables(tables: dict, meta: dict, label: str) -> None:
    try:
        from engine.parse.schema import RawTables
        raw = RawTables.from_parsed(tables, meta)
    except (ImportError, AttributeError, NotImplementedError) as exc:
        skip(f"RawTables.from_parsed accepts the {label} output", f"schema.py not ready: {exc}")
        return
    problems = raw.validate(strict=False)
    check(f"RawTables.from_parsed + validate() clean ({label})", not problems, "; ".join(problems[:5]))


def test_fake(tmp: Path) -> None:
    demo_file = tmp / "fake.dem"
    demo_file.write_bytes(b"PBDEMS2\x00" + b"\x00" * 64)
    demo = FakeDemo()
    tables, meta = ap.parse(demo_file, demo=demo)
    check_contract(tables, meta, "fake demo")
    check_rawtables(tables, meta, "fake demo")

    section("meta")
    check("map from header", meta["map"] == "de_mirage")
    check("tick_rate from playback ticks / time", meta["tick_rate"] == 128 and isinstance(meta["tick_rate"], int))
    check("game_version is the build number", meta["game_version"] == "10524")
    check("server from header", meta["server"] == "FACEIT.com server")
    check("demo_sha256 of the file", meta["demo_sha256"] == ap.sha256_file(demo_file))
    check("n_rounds", meta["n_rounds"] == 2)
    check("roster has 4 players with ct/t sides",
          len(meta["players"]) == 4 and {p["side"] for p in meta["players"]} == {"ct", "t"})
    check("side_counts from ticks", meta["side_counts"] == {"ct": 2, "t": 2}, f"{meta['side_counts']}")
    check("dropped rows counted (interrupt_plant, purchase)",
          meta["dropped_rows"] == {"bomb": 1, "items": 1}, f"{meta['dropped_rows']}")
    check("spotted_by listed as unavailable", "ticks.spotted_by" in meta["unavailable"])

    section("rounds")
    r = tables["rounds"]
    check("winner_side normalised", r["winner_side"].to_list() == ["ct", "t"])
    check("end_reason from reason_name", r["end_reason"].to_list() == ["bomb_defused", "target_bombed"])
    check("official_end_tick null for the last round", r["official_end_tick"].to_list() == [700, None])
    check("bomb_planted", r["bomb_planted"].to_list() == [True, True])
    check("scores before each round follow the team across the side swap",
          r["ct_score_before"].to_list() == [0, 0] and r["t_score_before"].to_list() == [0, 1],
          f"ct {r['ct_score_before'].to_list()} t {r['t_score_before'].to_list()}")

    section("ticks")
    t = tables["ticks"]
    check("one snapshot call for the whole match", demo.snapshot_calls == [(100, 1200)], f"{demo.snapshot_calls}")
    check("every tick of every round for every player", t.height == (1200 - 100 + 1) * 4, f"{t.height}")
    check("boundary tick goes to the later round",
          t.filter(pl.col("tick") == 700)["round_num"].unique().to_list() == [2])
    one = lambda sid, tick, col: t.filter((pl.col("steamid") == sid) & (pl.col("tick") == tick))[col].item()  # noqa: E731
    check("is_alive from health", one(C, 399, "is_alive") is True and one(C, 400, "is_alive") is False)
    check("has_c4 from has_bomb", one(C, 300, "has_c4") is True and one(A, 300, "has_c4") is False)
    check("is_planting during [start, interrupt)", one(C, 302, "is_planting") is True and one(C, 306, "is_planting") is False)
    check("is_planting during the second plant", one(C, 320, "is_planting") is True and one(C, 350, "is_planting") is False)
    check("kills_round counts enemy kills", one(A, 399, "kills_round") == 0 and one(A, 400, "kills_round") == 1)
    check("kills_round resets per round", one(A, 800, "kills_round") == 0)
    check("team kill not counted", one(A, 1100, "kills_round") == 0)
    check("damage_round sums enemy damage", one(A, 395, "damage_round") == 27 and one(A, 600, "damage_round") == 100)
    check("team damage not counted", one(A, 1100, "damage_round") == 0)
    check("is_ducking from is_crouched", t["is_ducking"].null_count() == 0)
    check("is_reloading stays null when awpy has no field", t["is_reloading"].null_count() == t.height)
    check("inventory is a list of strings", one(A, 300, "inventory").to_list() == ["ak47", "glock"])
    check("flash_duration is the remaining blind time",
          one(A, 394, "flash_duration") == 0.0 and abs(one(A, 395 + 64, "flash_duration") - 1.5) < 1e-6
          and one(A, 395 + 300, "flash_duration") == 0.0)
    check("dead player keeps rows: health 0, no weapon, empty inventory, equipment 0",
          (one(C, 450, "health"), one(C, 450, "active_weapon"), one(C, 450, "inventory").to_list(),
           one(C, 450, "equipment_value")) == (0, None, [], 0))
    check("dead player keeps the last alive position", one(C, 450, "x") == 399.0)
    check("never-null tick columns", all(t[c].null_count() == 0 for c in ("tick", "round_num", "steamid", "side", "is_alive")))

    section("kills / damages")
    k = tables["kills"]
    check("warmup kill dropped", k.height == 3 and 50 not in k["tick"].to_list())
    first = k.row(0, named=True)
    check("flash assist moves to flash_assister_steamid",
          first["flash_assister_steamid"] == B and first["assister_steamid"] is None)
    check("distance in units from positions", abs(first["distance"] - 500.0) < 1e-3)
    check("victim_blind from the snapshot", first["victim_blind"] is False and k.row(1, named=True)["victim_blind"] is True)
    check("attacker_blind from the snapshot", first["attacker_blind"] is True)
    check("team_kill", k["team_kill"].to_list() == [False, False, True])
    check("optional is_trade kept when awpy has it", k["is_trade"].to_list() == [False, True, False])
    check("penetrated as i16", k["penetrated"].to_list() == [0, 1, 0])
    check("through_smoke from thrusmoke", k["through_smoke"].to_list() == [False, False, True])
    check("hitgroup name used", tables["damages"]["hitgroup"].to_list()[0] == "chest")

    section("utility")
    g = tables["grenades"].sort("throw_tick")
    check("grenades collapsed to one row per entity", g.height == 4, f"{g.height}")
    check("slot reuse with a new serial is a separate grenade", g["entity_id"].n_unique() == 4)
    smoke = g.filter(pl.col("type") == "smoke").row(0, named=True)
    check("smoke throw / detonate / expire",
          (smoke["throw_tick"], smoke["detonate_tick"], smoke["expire_tick"]) == (250, 301, 600), f"{smoke}")
    check("smoke detonate position from the cloud", (smoke["x"], smoke["y"], smoke["z"]) == (-1.0, -2.0, -3.0))
    molo = g.filter(pl.col("type") == "molotov").row(0, named=True)
    check("molotov expire from the fire", (molo["detonate_tick"], molo["expire_tick"]) == (331, 450))
    flash = g.filter(pl.col("type") == "flash").row(0, named=True)
    check("flash detonates at its last tick, no expiry", flash["detonate_tick"] == 395 and flash["expire_tick"] is None)
    he = g.filter(pl.col("type") == "he").row(0, named=True)
    check("he detonate position = last trajectory point", he["x"] == 920.0)
    s = tables["smokes"]
    check("smokes share the grenade entity id", s["entity_id"].to_list() == [smoke["entity_id"]])
    inf = tables["infernos"]
    check("infernos share the grenade entity id", inf["entity_id"].to_list() == [molo["entity_id"]])
    check("extent_radius null (awpy has no flame extents)", inf["extent_radius"].null_count() == inf.height)
    f = tables["flashes"]
    check("one flashes row per blinded player", f.height == 2)
    check("flashes linked to the flash grenade", set(f["entity_id"].to_list()) == {flash["entity_id"]})
    check("blind_duration", sorted(f["blind_duration"].to_list()) == [1.0, 2.0])

    section("bomb")
    b = tables["bomb"]
    evs = b["event"].to_list()
    check("awpy names normalised", evs.count("plant_start") == 3 and evs.count("plant_complete") == 2
          and "defuse_complete" in evs and "pickup" in evs, f"{evs}")
    check("defuse_start from is_defusing rising edges", evs.count("defuse_start") == 2)
    abort = b.filter(pl.col("event") == "defuse_abort")
    check("defuse_abort only for the unfinished defuse", abort["tick"].to_list() == [460], f"{abort['tick'].to_list()}")
    check("explode from target_bombed at round end",
          b.filter(pl.col("event") == "explode")["tick"].to_list() == [1200])
    check("explode carries the plant site", b.filter(pl.col("event") == "explode")["site"].to_list() == ["B"])
    ds = b.filter(pl.col("event") == "defuse_start")
    check("defuse_start has_kit from the snapshot", ds["has_kit"].to_list() == [True, True])
    check("bomb round_num", b.filter(pl.col("event") == "explode")["round_num"].to_list() == [2])

    section("weapon_fire / items / footsteps")
    check("weapon_fire from shots", tables["weapon_fire"].height == 3)
    it = tables["items"]
    check("items keep drop / pickup only", it["event"].to_list() == ["drop", "pickup"])
    fs = tables["footsteps"]
    check("footsteps from player_footstep with positions", fs.height == 2 and fs["x"].to_list() == [500.0, 501.0])
    tables2, _ = ap.parse(demo_file, demo=FakeDemo(), footsteps=False)
    check("footsteps=False gives the empty A.1 frame",
          tables2["footsteps"].height == 0 and dict(tables2["footsteps"].schema) == ap.A1_SCHEMAS["footsteps"])

    section("empty demo")
    class Bare:
        header = {"map_name": "de_nuke"}
        rounds = None
        def snapshots(self, **kw):
            raise AssertionError("no rounds: snapshots must not be called")
    tb, mb = ap.parse(tmp / "missing.dem", demo=Bare())
    check_contract(tb, mb, "bare demo")
    check("bare: tick_rate unknown is None, not 64", mb["tick_rate"] is None)
    check("bare: demo_sha256 None without a file", mb["demo_sha256"] is None)


def test_quarantine(tmp: Path) -> None:
    section("quarantine_check")
    good = {"map": "de_mirage", "tick_rate": 128, "tick_rate_raw": 127.99, "n_rounds": 21,
            "side_counts": {"ct": 5, "t": 5}, "duration_s": 2700.0}
    check("good meta passes", ap.quarantine_check(good) is None)
    cases = {
        "tick_rate_not_128":    {"tick_rate": 64, "tick_rate_raw": 64.0},
        "too_few_rounds":       {"n_rounds": 3},
        "not_5v5":              {"side_counts": {"ct": 5, "t": 4}},
        "unknown_map":          {"map": "de_dust"},
        "duration_implausible": {"duration_s": 120.0},
    }
    for reason, patch in cases.items():
        check(f"returns {reason}", ap.quarantine_check({**good, **patch}) == reason)
    check("unknown tick rate quarantined", ap.quarantine_check({**good, "tick_rate": None, "tick_rate_raw": None})
          == "tick_rate_not_128")
    check("R8 (MED-04): a 0.0 raw rate is a value and fails; NaN / inf are unknown and fall back to the rounded rate",
          ap.quarantine_check({**good, "tick_rate_raw": 0.0}) == "tick_rate_not_128"
          and ap.quarantine_check({**good, "tick_rate_raw": float("nan")}) is None
          and ap.quarantine_check({**good, "tick_rate_raw": float("inf")}) is None
          and ap.quarantine_check({**good, "tick_rate": None, "tick_rate_raw": float("nan")}) == "tick_rate_not_128")
    header_only = {k: good[k] for k in ("map", "tick_rate", "tick_rate_raw", "duration_s")}
    check("header-only meta: absent keys not judged", ap.quarantine_check(header_only) is None)

    section("header_info")
    info = ap.header_info(tmp / "x.dem", demo=FakeDemo())
    check("header only: map / tick_rate / duration", (info["map"], info["tick_rate"], info["duration_s"])
          == ("de_mirage", 128, 2400.0))
    check("header only: no roster keys", "n_rounds" not in info and "side_counts" not in info)
    full = ap.header_info(tmp / "x.dem", demo=FakeDemo(), with_roster=True)
    check("with_roster: n_rounds and side_counts", full["n_rounds"] == 2 and full["side_counts"] == {"ct": 2, "t": 2})
    check("fake demo (2v2, 2 rounds) is quarantined", ap.quarantine_check(full) == "too_few_rounds")
    try:
        ap.header_info(tmp / "does_not_exist.dem")
        check("missing path raises FileNotFoundError", False)
    except FileNotFoundError as exc:
        check("missing path raises FileNotFoundError", "does_not_exist.dem" in str(exc))


def test_real(path: Path) -> None:
    section(f"real demo {path.name}")
    import time
    t0 = time.perf_counter()
    tables, meta = ap.parse(path)
    print(f"  parsed in {time.perf_counter() - t0:.1f}s: "
          + ", ".join(f"{k}={v.height}" for k, v in tables.items()))
    check_contract(tables, meta, "real demo")
    check_rawtables(tables, meta, "real demo")
    print(f"  unavailable: {meta['unavailable']}")
    print(f"  quarantine: {ap.quarantine_check(meta)}")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_fake(tmp)
        test_quarantine(tmp)
    if len(sys.argv) > 1:
        test_real(Path(sys.argv[1]))
    else:
        skip("real-demo contract checks", "no demo path given (pass one to run them)")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
