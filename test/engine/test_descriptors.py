"""
P9.1.1 action descriptors: every Appendix A.5 type yields a descriptor with exactly the
documented fields; unknown callout -> UNK; bucket edges exact; vocab round trip.

    .venv\\Scripts\\python.exe test\\engine\\test_descriptors.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, section, summary
from _cf_states import action, make_snapshot
from engine.actions.types import TYPE_KIND, Action
from engine.state.schema import PAIR_CHANNELS
from engine.value import descriptors as dsc

PARAMS = {
    "kill":            {"weapon_class": "sniper", "attacker_callout": "Mid", "victim_callout": "A Site", "victim_id": 202},
    "team_kill":       {"weapon_class": "pistol", "attacker_callout": "Mid", "victim_callout": "Mid", "victim_id": 102},
    "damage":          {"weapon_class": "rifle", "damage_total": 40, "victim_id": 203},
    "flash":           {"blinded_enemies": [{"id": 201, "duration": 2.0}, {"id": 202, "duration": 1.0}],
                        "land_callout": "A Site", "throw_callout": "CT"},
    "he":              {"enemies_damaged": [{"id": 201, "dmg": 30}], "land_callout": "B Site"},
    "peek":            {"from_callout": "Mid", "target_id": 201, "target_callout": "Long"},
    "plant_start":     {"site": "A", "planter_callout": "A Site"},
    "plant_complete":  {"site": "B", "planter_callout": "B Site"},
    "defuse_start":    {"kit": True},
    "defuse_complete": {"kit": True},
    "defuse_abort":    {"kit": False},
    "bomb_drop":       {"callout": "Mid"},
    "bomb_pickup":     {"callout": "Mid"},
    "drop":            {"weapon_class": "rifle", "receiver_id": 102},
    "smoke":           {"land_callout": "Mid", "throw_callout": "T Spawn"},
    "molotov":         {"land_callout": "Palace", "enemies_damaged": [{"id": 101, "dmg": 8}], "enemies_displaced": []},
    "decoy":           {"land_callout": "B Site"},
    "rotation":        {"from_callout": "A Site", "to_callout": "B Site"},
    "hold":            {"callout": "Connector", "facing_callout": "Mid"},
    "save":            {"from_callout": "A Site", "to_callout": "CT"},
    "buy":             {"loadout_class": "full", "primary_class": "rifle"},
    "team_buy":        {"team_class": "force", "total_equip": 18000},
}
CT_TYPES = ("defuse_start", "defuse_complete", "defuse_abort")


def make(type_: str, **over) -> Action:
    kind = TYPE_KIND[type_]
    actor = 101 if type_ in CT_TYPES or type_ not in ("plant_start", "plant_complete", "bomb_drop", "bomb_pickup") else 201
    a = action(type_, actor, kind="durative" if kind == "durative" else "instant", **{**PARAMS.get(type_, {}), **over})
    if kind == "round":
        a.kind = "round"
    if type_ == "team_buy":
        a.actor_id_hash = None
    return a


def state_for(type_: str):
    planted = type_ in CT_TYPES
    return make_snapshot(5, 5, planted=planted, plant_tick=4000 if planted else 5000)


# ── every type ────────────────────────────────────────────────────────────────

section("every Appendix A.5 type yields a descriptor")
descs = {}
for t in TYPE_KIND:
    try:
        d = dsc.encode(make(t), state_for(t))
        descs[t] = d
        ok = tuple(d) == dsc.FIELDS and d["type"] == t
    except Exception as e:                                   # noqa: BLE001 - reported below
        ok, d = False, repr(e)
    check(f"{t}: fields exactly P9.1.1", ok, "" if ok else str(d))
check("all types covered", set(descs) == set(TYPE_KIND))
check("state_pre None: state fields None, no crash",
      all(dsc.encode(make(t), None)[f] is None for t in TYPE_KIND for f in ("timing_bucket", "alive_diff_bucket")))
try:
    bad = make("kill")
    bad.type = "teabag"
    dsc.encode(bad, state_for("kill"))
    check("unknown type raises", False)
except ValueError:
    check("unknown type raises", True)

section("field values per type")
check("kill: weapon_class from params", descs["kill"]["weapon_class"] == "sniper")
check("kill: from/to = attacker/victim callout", (descs["kill"]["from_callout"], descs["kill"]["to_callout"]) == ("Mid", "A Site"))
check("peek: weapon_class falls back to the actor's (rifle)", descs["peek"]["weapon_class"] == "rifle")
check("smoke: to_callout = land_callout", descs["smoke"]["to_callout"] == "Mid")
check("rotation: from/to callouts", (descs["rotation"]["from_callout"], descs["rotation"]["to_callout"]) == ("A Site", "B Site"))
check("hold: callout / facing_callout", (descs["hold"]["from_callout"], descs["hold"]["to_callout"]) == ("Connector", "Mid"))
check("he: from_callout falls back to the actor's state callout",
      descs["he"]["from_callout"] == state_for("he").players[0].callout)
check("damage: to_callout = victim's state callout",
      descs["damage"]["to_callout"] == state_for("damage").players[state_for("damage").index_of(203)].callout)
check("flash: n_enemies_affected = 2", descs["flash"]["n_enemies_affected"] == 2)
check("he / molotov: n_enemies_affected = 1", descs["he"]["n_enemies_affected"] == 1 and descs["molotov"]["n_enemies_affected"] == 1)
many = [{"id": i, "duration": 1.0} for i in range(7)]
check("flash: n_enemies_affected clipped to 5",
      dsc.encode(make("flash", blinded_enemies=many), state_for("flash"))["n_enemies_affected"] == 5)
check("kill: n_enemies_affected None", descs["kill"]["n_enemies_affected"] is None)
check("buy: loadout_class; team_buy: team_class", descs["buy"]["loadout_class"] == "full" and descs["team_buy"]["loadout_class"] == "force")
check("buy: weapon_class = primary_class", descs["buy"]["weapon_class"] == "rifle")
check("team_buy: no actor -> no callout / weapon", descs["team_buy"]["from_callout"] is None and descs["team_buy"]["weapon_class"] is None)
check("plant: site from params", descs["plant_start"]["site"] == "A" and descs["plant_complete"]["site"] == "B")
check("defuse: site from ctx.bomb_site", all(descs[t]["site"] == "A" for t in CT_TYPES))
check("kill: site None", descs["kill"]["site"] is None)

section("has_los_to_enemy (peek / hold only)")
s = make_snapshot(5, 5)
check("peek with LOS to an enemy -> 1", dsc.encode(make("peek"), s)["has_los_to_enemy"] == 1)
s0 = make_snapshot(5, 5)
k = PAIR_CHANNELS.index("los")
s0.pair[:, :, k] = np.where(np.isnan(s0.pair[:, :, k]), np.nan, 0.0)
check("peek, no LOS anywhere -> 0", dsc.encode(make("peek"), s0)["has_los_to_enemy"] == 0)
check("hold uses it too", dsc.encode(make("hold"), s)["has_los_to_enemy"] == 1)
check("dead actor -> None", dsc.encode(make("peek"), make_snapshot(5, 5, dead=(101,)))["has_los_to_enemy"] is None)
check("kill -> None (not a peek/hold)", dsc.encode(make("kill"), s)["has_los_to_enemy"] is None)

# ── buckets ───────────────────────────────────────────────────────────────────

section("timing_bucket edges (pre-plant half-open, post-plant early <= 20 s)")


def at(into=None, since=None):
    sn = make_snapshot(5, 5, planted=since is not None, plant_tick=5000)
    if into is not None:
        sn.ctx.time_into_round_s = into
    if since is not None:
        sn.ctx.time_since_plant_s = since
    return dsc.timing_bucket(sn)


for into, want in ((0.0, "0-15"), (14.999, "0-15"), (15.0, "15-45"), (44.999, "15-45"),
                   (45.0, "45-115"), (114.9, "45-115"), (130.0, "45-115")):
    check(f"into {into} s -> {want}", at(into=into) == want, str(at(into=into)))
for since, want in ((0.0, "pp_early"), (20.0, "pp_early"), (20.001, "pp_late"), (39.0, "pp_late")):
    check(f"since plant {since} s -> {want}", at(since=since) == want, str(at(since=since)))
check("every bucket value is documented", set(dsc.TIMING_BUCKETS) == {"0-15", "15-45", "45-115", "pp_early", "pp_late"})

section("alive_diff_bucket (actor side minus enemy, clipped to ±3)")
for (act, ali_t, side, want) in ((5, 1, "ct", 3), (5, 1, "t", -3), (3, 3, "ct", 0), (4, 2, "ct", 2),
                                 (2, 5, "ct", -3), (1, 4, "t", 3), (5, 2, "ct", 3)):
    got = dsc.alive_diff_bucket(side, make_snapshot(act, ali_t))
    check(f"{act}v{ali_t} as {side} -> {want}", got == want, str(got))
check("unknown side -> None", dsc.alive_diff_bucket(None, make_snapshot()) is None)

# ── vocab ─────────────────────────────────────────────────────────────────────

section("vocabulary: UNK / PAD, unknown callout, round trip")
vocab = dsc.DescriptorVocab.fit(descs.values())
check("UNK = 0, PAD = 1", (dsc.UNK, dsc.PAD) == (0, 1))
check("real tokens start at 2", all(min(m.values()) >= 2 for m in vocab.maps.values() if m))
check("sizes = tokens + 2", all(vocab.sizes()[f] == len(vocab.maps[f]) + 2 for f in dsc.FIELDS))
ids = dsc.to_ids(descs["rotation"], vocab)
check("to_ids: int64[len(FIELDS)]", ids.dtype == np.int64 and ids.shape == (len(dsc.FIELDS),))
check("known callout -> known id", ids[dsc.FIELDS.index("from_callout")] >= 2)
odd = dsc.encode(make("rotation", from_callout="Nowhere Land"), state_for("rotation"))
check("unknown callout -> UNK", dsc.to_ids(odd, vocab)[dsc.FIELDS.index("from_callout")] == dsc.UNK)
check("None value -> UNK", dsc.to_ids(descs["kill"], vocab)[dsc.FIELDS.index("site")] == dsc.UNK)
with tempfile.TemporaryDirectory() as tmp:
    p = vocab.save(Path(tmp) / "desc_vocab.json")
    v2 = dsc.DescriptorVocab.load(p)
    check("save / load round trip equal", v2 == vocab)
    check("ids identical after reload", all(np.array_equal(dsc.to_ids(d, vocab), dsc.to_ids(d, v2)) for d in descs.values()))

section("frame columns and key")
types = list(TYPE_KIND)
fc = dsc.frame_columns([make(t) for t in types], [state_for(t) for t in types])
check("frame_columns: one row per action, desc_* columns",
      fc.height == len(types) and all(dsc.COLUMN_PREFIX + f in fc.columns for f in dsc.FIELDS))
via_frame = dsc.ids_from_frame(fc, vocab)
check("ids_from_frame == to_ids per row", all(np.array_equal(via_frame[i], dsc.to_ids(descs[t], vocab)) for i, t in enumerate(types)))
check("vocab fit from frame == fit from dicts", dsc.DescriptorVocab.fit_frame(fc) == vocab)
check("key = type|subtype|location", dsc.key(descs["rotation"]) == "rotation||B Site")
check("key: falls back to from_callout, then site",
      dsc.key(descs["plant_start"]) == "plant_start||A Site" and dsc.key({**descs["buy"], "site": None}) == "buy||")
check("key: molotov subtype included",
      dsc.key(dsc.encode(Action(**{**make("molotov").to_dict(), "subtype": "incendiary", "action_id": None}),
                         state_for("molotov"))) == "molotov|incendiary|Palace")

sys.exit(summary())
