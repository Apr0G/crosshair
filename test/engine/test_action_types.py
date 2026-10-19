"""
Action contract (P4.1.1, P4.1.2): deterministic ids, kind/type/timing consistency,
JSON round trip in both flatten modes, and the extractor registry.

    .venv\\Scripts\\python.exe test\\engine\\test_action_types.py
"""
import hashlib
import json
import sys

from _harness import check, section, summary

from engine.actions.types import (KINDS, PSEUDO_TYPES, REGISTRY, TYPE_KIND, Action,
                                  assert_unique_ids, make_action_id, missing_extractors, register)

A5_TYPES = {"kill", "team_kill", "damage", "flash", "he", "peek", "plant_start", "plant_complete",
            "defuse_start", "defuse_complete", "defuse_abort", "bomb_drop", "bomb_pickup", "smoke",
            "molotov", "decoy", "rotation", "hold", "save", "buy", "team_buy", "drop",
            "smoke_pop", "smoke_fade", "fire_start", "fire_end", "decoy_start", "decoy_end"}


def kill(**kw) -> Action:
    base = dict(match_key="m1", round_num=1, actor_id_hash=123, actor_side="ct", kind="instant",
                type="kill", tick=2000, params={"weapon": "ak47", "headshot": True, "victim_id": 456},
                state_pre_tick=1999, state_post_tick=2001)
    return Action(**{**base, **kw})


# ── ids ──
section("deterministic action_id")
a, b = kill(), kill()
want = hashlib.sha1(b"m1|1|kill|123|2000").hexdigest()[:16]
check("id = sha1(match|round|type|actor|tick)[:16]", a.action_id == want, a.action_id)
check("same inputs -> same id", a.action_id == b.action_id)
check("different tick -> different id", kill(tick=2001).action_id != a.action_id)
check("different actor -> different id", kill(actor_id_hash=124).action_id != a.action_id)
check("params do not change the id", kill(params={}).action_id == a.action_id)
check("salt disambiguates", make_action_id("m1", 1, "damage", 123, 1800, salt="456")
      != make_action_id("m1", 1, "damage", 123, 1800, salt="789"))
check("explicit action_id kept", kill(action_id="0123456789abcdef").action_id == "0123456789abcdef")
smoke = Action(match_key="m1", round_num=1, actor_id_hash=123, actor_side="ct", kind="durative",
               type="smoke", window_start_tick=1500, window_end_tick=3804)
check("durative id uses window_start_tick", smoke.action_id == make_action_id("m1", 1, "smoke", 123, 1500))
try:
    assert_unique_ids([a, b])
    check("duplicate ids detected", False)
except ValueError:
    check("duplicate ids detected", True)

# ── vocabulary ──
section("TYPE_KIND covers Appendix A.5")
check("KINDS", KINDS == ("instant", "durative", "round"))
check("every A.5 type (with pseudo types) present", set(TYPE_KIND) == A5_TYPES, str(set(TYPE_KIND) ^ A5_TYPES))
check("every kind valid", set(TYPE_KIND.values()) <= set(KINDS))
check("pseudo types are instant", all(TYPE_KIND[t] == "instant" for t in PSEUDO_TYPES))
check("spot checks", (TYPE_KIND["smoke"], TYPE_KIND["buy"], TYPE_KIND["drop"], TYPE_KIND["damage"])
      == ("durative", "round", "instant", "instant"))

# ── validate ──
section("validate()")
check("valid kill", a.validate() == [], str(a.validate()))
check("valid smoke", smoke.validate() == [], str(smoke.validate()))
buy = Action(match_key="m1", round_num=1, actor_id_hash=123, actor_side="t", kind="round", type="buy",
             tick=1000, params={"loadout_class": "full"})
check("valid buy", buy.validate() == [])
team = Action(match_key="m1", round_num=1, actor_id_hash=None, actor_side="t", kind="round",
              type="team_buy", tick=1000)
check("team_buy may have no actor", team.validate() == [])
check("kill without actor rejected", kill(actor_id_hash=None).validate() != [])
check("invalid kind rejected", any("kind" in p for p in kill(kind="sometimes").validate()))
check("kind/type mismatch rejected", any("is instant" in p for p in kill(kind="durative",
      window_start_tick=1, window_end_tick=2, tick=None).validate()))
check("unknown type rejected", any("A.5" in p for p in kill(type="teabag").validate()))
check("instant without tick rejected", kill(tick=None).validate() != [])
check("instant with window rejected", kill(window_start_tick=2000, window_end_tick=2100).validate() != [])
dmg = kill(type="damage", window_start_tick=2000, window_end_tick=2200)
check("damage may carry window metadata starting at tick", dmg.validate() == [], str(dmg.validate()))
check("damage window not at tick rejected", kill(type="damage", window_start_tick=1990,
                                                 window_end_tick=2200).validate() != [])
check("durative start > end rejected", any("> window_end_tick" in p for p in Action(
    match_key="m1", round_num=1, actor_id_hash=1, actor_side="ct", kind="durative", type="hold",
    window_start_tick=3000, window_end_tick=2000).validate()))
check("durative without window rejected", Action(match_key="m1", round_num=1, actor_id_hash=1,
      actor_side="ct", kind="durative", type="hold", tick=3000).validate() != [])
check("round without tick rejected", Action(match_key="m1", round_num=1, actor_id_hash=1,
      actor_side="ct", kind="round", type="buy").validate() != [])
check("bad side rejected", kill(actor_side="?").validate() != [])
check("pre > post rejected", kill(state_pre_tick=2001, state_post_tick=1999).validate() != [])
check("non-JSON params rejected", kill(params={"x": object()}).validate() != [])
check("NaN params rejected", kill(params={"x": float("nan")}).validate() != [])
check("malformed enabler rejected", kill(enablers=[{"kind": "flash"}]).validate() != [])

# ── serialization ──
section("JSON round trip")
full = kill(enablers=[{"action_id": "aaaaaaaaaaaaaaaa", "kind": "flash"}],
            params={"weapon": "ak47", "blinded": [{"id": 1, "duration": 2.5}], "callout": None})
for flatten in (False, True):
    d = full.to_dict(flatten=flatten)
    text = json.dumps(d)
    back = Action.from_dict(json.loads(text))
    check(f"flatten={flatten}: JSON-safe and round trips", back == full)
d = full.to_dict(flatten=True)
check("flatten=True: params/enablers are strings",
      isinstance(d["params"], str) and isinstance(d["enablers"], str))
check("flatten=True: deterministic text", d["params"] == kill(
    enablers=[{"action_id": "aaaaaaaaaaaaaaaa", "kind": "flash"}],
    params={"callout": None, "blinded": [{"id": 1, "duration": 2.5}], "weapon": "ak47"}).to_dict(True)["params"])
check("to_dict copies params (no aliasing)", full.to_dict()["params"] is not full.params)

# ── registry ──
section("registry")
before = dict(REGISTRY)


@register("decoy")
def _extract_decoys(raw, states, ctx):
    return []


check("decorator stores the callable", REGISTRY["decoy"] is _extract_decoys)
check("decorator returns the function unchanged", _extract_decoys(None, None, None) == [])
check("re-registering the same function is allowed", register("decoy")(_extract_decoys) is _extract_decoys)


def _other(raw, states, ctx):
    return []


for name, fn in (("second extractor for a type rejected", lambda: register("decoy")(_other)),
                 ("unknown type rejected", lambda: register("teabag")),
                 ("pseudo type rejected", lambda: register("smoke_pop"))):
    try:
        fn()
        check(name, False)
    except ValueError:
        check(name, True)
check("missing_extractors lists unregistered non-pseudo types",
      "decoy" not in missing_extractors()
      and ("kill" in missing_extractors()) == ("kill" not in REGISTRY))
check("missing_extractors never lists pseudo types", not set(missing_extractors()) & set(PSEUDO_TYPES))
REGISTRY.clear()
REGISTRY.update(before)

sys.exit(summary())
