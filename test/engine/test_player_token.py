"""
Player tokens (P3.2) on the synthetic stories: alive / dead / blind / scoped / bomb /
defuser cases, weapon vocabulary, spotted timing, heard_enemy, kills/damage so far,
player_id hash stability. Tokens come from the real causal view (P3.9.2).

    .venv\\Scripts\\python.exe test\\engine\\test_player_token.py
"""
import hashlib
import logging
import sys

import polars as pl

from _harness import check, close, section, summary
from _synthetic import PLAYERS, STORIES, make_raw_tables

from engine.state import players, utility
from engine.state.builder import RoundView
from engine.state.geometry import FlatGeometry
from engine.state.schema import DEAD_NULL_FIELDS, PLAYER_FIELDS, PlayerToken

GEO = FlatGeometry()
TR  = 128


def tok(raw, rnd, tick, who, view=None) -> PlayerToken:
    v  = view or RoundView.build(raw, rnd)
    sl = v.until(tick)
    util, _, _ = utility.tokens(sl)
    sid, side = PLAYERS[who]
    return players.token(sl, sid, side, GEO, "pro", util)


class _Catch(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


base = make_raw_tables("base", TR)
v1   = RoundView.build(base, 1)

# ── fields ──
section("fields")
a = tok(base, 1, 1512, "A", v1)
check("token is a PlayerToken with the 41 A.2 fields", isinstance(a, PlayerToken) and len(PLAYER_FIELDS) == 41)
check("tier passes through", a.tier == "pro")

# ── alive ──
section("alive")
check("A alive at 1512, position (0,300,0)", a.alive == 1 and (a.x, a.y, a.z) == (0.0, 300.0, 0.0))
check("A health/armor 100, helmet, defuser, no c4",
      (a.health, a.armor, a.has_helmet, a.has_defuser, a.has_c4) == (100.0, 100.0, 1, 1, 0))
check("A weapon class rifle (weapon_ak47)", a.weapon_class == "rifle")
check("A grenades after throwing the smoke: 0 smoke, 1 HE",
      (a.n_smoke, a.n_flash, a.n_he, a.n_molly, a.n_decoy) == (0, 0, 1, 0, 0))
check("A not blind -> flash_duration_remaining 0.0 (legitimate zero)", a.flash_duration_remaining == 0.0)
check("A money / equipment", a.money == 650.0 and a.equipment_value == 4600.0)
check("nav_area / callout / grid_cell from geometry",
      a.nav_area == GEO.find_area((0.0, 300.0, 0.0)) and a.callout == f"cell_{a.nav_area}"
      and a.grid_cell == a.nav_area)
b = tok(base, 1, 1512, "B", v1)
check("B speed 250 u/s while walking, A speed 0", close(b.speed, 250.0, 1e-3) and a.speed == 0.0)
check("B inside A's smoke at 1512 (passing (1000,0)), A not", b.in_smoke == 1 and a.in_smoke == 0)
check("on_fire 0 with no fires", a.on_fire == 0 and b.on_fire == 0)
check("flags from per-tick props",
      (a.is_scoped, a.is_ducking, a.is_walking, a.is_reloading, a.is_defusing, a.is_planting) == (0,) * 6)

# ── dead ──
section("dead")
c = tok(base, 1, 2000, "C", v1)
check("C dead at 2000 (killed that tick)", c.alive == 0)
check("dead: health 0, armor 0", c.health == 0.0 and c.armor == 0.0)
check("dead: every DEAD_NULL field None", all(getattr(c, f) is None for f in DEAD_NULL_FIELDS),
      str([f for f in DEAD_NULL_FIELDS if getattr(c, f) is not None]))
check("dead: keeps last position (3000,300)", (c.x, c.y) == (3000.0, 300.0))
check("dead: flash 0.0, in_smoke / on_fire None",
      c.flash_duration_remaining == 0.0 and c.in_smoke is None and c.on_fire is None)
d = tok(base, 1, 4100, "D", v1)
check("D dead at 4100 keeps the post-plant position (3400,-300)", d.alive == 0 and (d.x, d.y) == (3400.0, -300.0))

# ── blind ──
section("blind (flash story)")
fl = make_raw_tables("flash", TR)
vf = RoundView.build(fl, 1)
check("D blind 2.5 s at the flash tick", close(tok(fl, 1, 1900, "D", vf).flash_duration_remaining, 2.5))
check("D 2.0 s left half a second later", close(tok(fl, 1, 1964, "D", vf).flash_duration_remaining, 2.0))
check("C 0.5 s left at 1964", close(tok(fl, 1, 1964, "C", vf).flash_duration_remaining, 0.5))
check("D 0.0 after it wore off", tok(fl, 1, 1900 + 3 * TR, "D", vf).flash_duration_remaining == 0.0)
check("not blind before the flash", tok(fl, 1, 1899, "D", vf).flash_duration_remaining == 0.0)
check("A (thrower, not blinded) 0.0", tok(fl, 1, 1900, "A", vf).flash_duration_remaining == 0.0)

# ── scoped / unknown weapon (edited copy of base) ──
section("scoped, unknown weapon")
A_ID = PLAYERS["A"][0]
f = base.frames()
late_a = (pl.col("steamid") == A_ID) & (pl.col("tick") >= 1600)
f["ticks"] = base.ticks.with_columns(
    pl.when(late_a).then(True).otherwise(pl.col("is_scoped")).alias("is_scoped"),
    pl.when(late_a).then(pl.lit("weapon_mystery_gun")).otherwise(pl.col("active_weapon")).alias("active_weapon"))
edited = type(base)(**f, meta=dict(base.meta))
ve = RoundView.build(edited, 1)
h = _Catch()
logging.getLogger("engine.state.players").addHandler(h)
x1, x2 = tok(edited, 1, 1600, "A", ve), tok(edited, 1, 1700, "A", ve)
logging.getLogger("engine.state.players").removeHandler(h)
check("is_scoped 1 when the prop is set", x1.is_scoped == 1 and tok(edited, 1, 1599, "A", ve).is_scoped == 0)
check("unknown weapon name -> 'unknown' class", x1.weapon_class == "unknown" and x2.weapon_class == "unknown")
check("unmapped name logged exactly once", sum("weapon_mystery_gun" in m for m in h.lines) == 1, str(h.lines))
check("null weapon name -> None", players.weapon_class(None) is None)

# ── bomb carrier / planting / defuser ──
section("bomb and defuser")
check("D carries the bomb before the plant", tok(base, 1, 2500, "D", v1).has_c4 == 1)
dp = tok(base, 1, 2950, "D", v1)
check("D planting at 2950: is_planting 1, weapon class c4", dp.is_planting == 1 and dp.weapon_class == "c4")
check("D no bomb after plant_complete", tok(base, 1, 3001, "D", v1).has_c4 == 0)
check("CTs have defusers, Ts do not",
      tok(base, 1, 1512, "B", v1).has_defuser == 1 and tok(base, 1, 1512, "C", v1).has_defuser == 0)
rt = make_raw_tables("retreat", TR)
vr = RoundView.build(rt, 1)
check("retreat: B is_defusing during the defuse",
      tok(rt, 1, 3950, "B", vr).is_defusing == 1 and tok(rt, 1, 3899, "B", vr).is_defusing == 0)

# ── spotted ──
section("spotted timing")
check("None before the first spot (A at 1935)", tok(base, 1, 1935, "A", v1).seconds_since_last_seen_by_enemy is None)
check("0.0 when first spotted (1936)", tok(base, 1, 1936, "A", v1).seconds_since_last_seen_by_enemy == 0.0)
check("0.0 on the last spotted tick (1999)", tok(base, 1, 1999, "A", v1).seconds_since_last_seen_by_enemy == 0.0)
check("1.0 one second after the last spot",
      close(tok(base, 1, 1999 + TR, "A", v1).seconds_since_last_seen_by_enemy, 1.0))
check("spotted_by_count 1 while C watches A, 0 after C dies",
      tok(base, 1, 1950, "A", v1).spotted_by_count == 1 and tok(base, 1, 2001, "A", v1).spotted_by_count == 0)
vals = [tok(base, 1, t, w, v1).seconds_since_last_seen_by_enemy for t in range(1000, 4360, 97) for w in "ABCD"]
check("never negative", all(v is None or v >= 0 for v in vals))

# ── last shot / heard ──
section("last shot, heard_enemy")
check("A's last shot: smoke throw at 1400 -> 100/128 s at 1500",
      close(tok(base, 1, 1500, "A", v1).seconds_since_last_shot, 100 / TR))
check("None before any shot (B at 1500)", tok(base, 1, 1500, "B", v1).seconds_since_last_shot is None)
check("C hears B running at 1800 (B within 1500 u)", tok(base, 1, 1800, "C", v1).heard_enemy == 1)
check("C hears nothing at 1100 (B far away)", tok(base, 1, 1100, "C", v1).heard_enemy == 0)
check("D hears B shortly after B stops (2100)", tok(base, 1, 2100, "D", v1).heard_enemy == 1)
check("D hears nothing 2 s after B's last step (2400)", tok(base, 1, 2400, "D", v1).heard_enemy == 0)
check("heard_enemy None when dead", tok(base, 1, 2100, "C", v1).heard_enemy is None)
check("a shorter window forgets sooner",
      players.token(v1.until(2100), PLAYERS["D"][0], "t", GEO, "pro", (), heard_window_s=0.25).heard_enemy == 0)

# ── kills / damage so far ──
section("kills_round, damage_round")
mism = []
for story in STORIES:
    raw = make_raw_tables(story, TR)
    for rnd in (1, 2):
        v = RoundView.build(raw, rnd)
        rows = raw.ticks.filter(pl.col("round_num") == rnd)
        for t in rows["tick"].unique().sort().to_list()[::150]:
            for who, (sid, _) in PLAYERS.items():
                k = tok(raw, rnd, t, who, v)
                r = rows.filter((pl.col("tick") == t) & (pl.col("steamid") == sid))
                if (k.kills_round, k.damage_round) != (float(r["kills_round"][0]), float(r["damage_round"][0])):
                    mism.append((story, rnd, t, who))
check("kills_round / damage_round match the parser's running totals on every story", not mism, str(mism[:5]))
a2 = tok(base, 1, 2001, "A", v1)
check("A after 2000: 1 kill, damage = HE 30 + 20 + lethal 70", (a2.kills_round, a2.damage_round) == (1.0, 120.0))

# ── weapon vocabulary ──
section("weapon vocabulary")
names = set()
for story in STORIES:
    raw = make_raw_tables(story, TR)
    names |= set(raw.ticks["active_weapon"].drop_nulls().to_list())
    names |= {i for inv in raw.ticks["inventory"].to_list() for i in inv}
    for tb in (raw.kills, raw.damages, raw.weapon_fire):
        names |= set(tb["weapon"].to_list())
    names |= set(raw.items["item"].to_list())
unmapped = sorted(n for n in names if players.weapon_class(n) == "unknown")
check(f"every weapon name in _synthetic.py maps ({len(names)} names)", not unmapped, str(unmapped))
pairs = [("weapon_ak47", "rifle"), ("ak47", "rifle"), ("AK-47", "rifle"), ("weapon_awp", "sniper"),
         ("mp9", "smg"), ("weapon_nova", "shotgun"), ("negev", "lmg"), ("weapon_glock", "pistol"),
         ("usp_silencer", "pistol"), ("weapon_knife_karambit", "knife"), ("weapon_bayonet", "knife"),
         ("hegrenade", "grenade"), ("weapon_incgrenade", "grenade"), ("inferno", "grenade"),
         ("weapon_taser", "zeus"), ("weapon_c4", "c4"), ("planted_c4", "c4")]
bad = [(n, players.weapon_class(n)) for n, want in pairs if players.weapon_class(n) != want]
check("both name forms (weapon_x and x) and display names map", not bad, str(bad))
check("class set is the fixed vocabulary", set(players.WEAPON_CLASSES) == {
    "rifle", "sniper", "smg", "shotgun", "lmg", "pistol", "knife", "grenade", "zeus", "c4", "unknown"})
check("grenade counts: molotov + incgrenade -> n_molly, duplicates counted",
      players.grenade_counts(["weapon_molotov", "weapon_incgrenade", "weapon_flashbang", "weapon_flashbang"])
      == {"n_smoke": 0, "n_flash": 2, "n_he": 0, "n_molly": 2, "n_decoy": 0})
check("unknown inventory -> None", players.grenade_counts(None) is None)

# ── identity ──
section("player_id hash")
h1   = players.player_id_hash(76561198000000001)
want = int.from_bytes(hashlib.sha1(b"76561198000000001").digest()[:8], "big") & ((1 << 63) - 1)
check("sha1 truncated to 63 bits, stable", h1 == want == players.player_id_hash(76561198000000001))
check("in [0, 2^63) and never the pad id", 0 <= h1 < 2 ** 63 and h1 != players.PAD_PLAYER_ID)
check("token player_id = hash(steamid)", a.player_id == players.player_id_hash(A_ID))

# ── helpers ──
section("null helpers")
check("_nan: NaN -> None, 0 kept", players._nan(float("nan")) is None and players._nan(0) == 0)
check("_first_valid skips NaN and None, keeps 0", players._first_valid(None, float("nan"), 0, 5) == 0)

sys.exit(summary())
