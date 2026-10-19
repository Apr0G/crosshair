"""
Enabler splits (P8.5) with stubbed hazard values: known q -> known shares; the trade and
other bands clamp; the 0.5 total cap rescales proportionally; shares + the actor's
remainder = φ_e exactly; the `min_hazard` rule (q = 0.01) and NaN heads use the band
floors; failed removals use the floor; `info` goes to the clock; fixed mode (P8.5.3).
Then `remove_i` states and head selection on hand-built snapshots, and the split inside
a full round with a stub hazard head.

    .venv\\Scripts\\python.exe test\\engine\\test_enabler_split.py
"""
import math
import sys
from dataclasses import replace

import numpy as np

from _harness import check, close, section, summary
from _cf_states import action, make_snapshot
from _attr_states import make_round, round_hazard, round_win
from _stubs import StubPredictor

from engine import config
from engine.attribution import engine
from engine.attribution import enablers as en

CFG   = config.load()
ACFG  = CFG.attribution
FIXED = replace(ACFG, enabler_mode="fixed")


def E(kind, aid=None):
    return {"kind": kind, "action_id": aid or f"{kind[:4]:0<16}"[:16]}


def shares(sp) -> list[float]:
    return [s.share for s in sp.shares]


# ── known q -> known shares ───────────────────────────────────────────────────

section("hazard mode: known q -> known shares")
sp = en.split([E("flash"), E("smoke_cover")], 0.4, {0: 0.3, 1: 0.38}, ACFG)
check("raw_i = (q_pre − q_rm) / q_pre", close(sp.shares[0].raw, 0.25, 1e-12) and close(sp.shares[1].raw, 0.05, 1e-12))
check("shares inside the band are the raw values", close(shares(sp)[0], 0.25, 1e-12) and close(shares(sp)[1], 0.05, 1e-12))
check("total = Σ shares, actor share = 1 − total", close(sp.total, 0.30, 1e-12) and close(sp.actor_share, 0.70, 1e-12))
for phi_e in (0.12, -0.0731, 0.0):
    actor, parts = en.distribute(sp, phi_e)
    check(f"φ_e = {phi_e:+.4f}: actor remainder + enabler parts == φ_e exactly",
          actor + math.fsum(a for _, a in parts) == phi_e and close(actor, 0.7 * phi_e, 1e-15))
sp = en.split([E("flash")], 0.4, {0: 0.5}, ACFG)
check("removal raising q -> raw 0 -> band floor (0 for other kinds)", sp.shares[0].raw == 0.0 and shares(sp) == [0.0])

section("clamps")
lo, hi = ACFG.trade_share_band
sp = en.split([E("trade_setup")], 0.4, {0: 0.396}, ACFG)
check(f"trade raw 0.01 -> clamped up to {lo}", close(shares(sp)[0], lo, 1e-12))
sp = en.split([E("trade_setup")], 0.4, {0: 0.04}, ACFG)
check(f"trade raw 0.9 -> clamped down to {hi}", close(shares(sp)[0], hi, 1e-12))
sp = en.split([E("he_damage")], 0.4, {0: 0.04}, replace(ACFG, enabler_total_cap=1.0))
check(f"other raw 0.9 -> clamped to {ACFG.other_share_band[1]} (cap lifted to isolate the clamp)",
      close(shares(sp)[0], ACFG.other_share_band[1], 1e-12) and not sp.capped)
sp = en.split([E("he_damage")], 0.4, {0: 0.04}, ACFG)
check("... and with the shipped cap the clamped 0.6 is then capped to 0.5", close(shares(sp)[0], 0.5, 1e-12) and sp.capped)

section("total cap")
sp = en.split([E("flash"), E("smoke_cover")], 0.4, {0: 0.16, 1: 0.28}, ACFG)  # raw 0.6, 0.3
check("Σ 0.9 > cap -> capped", sp.capped and close(sp.total, ACFG.enabler_total_cap, 1e-12))
check("rescaled proportionally (2:1 kept)", close(shares(sp)[0], 1 / 3, 1e-12) and close(shares(sp)[1], 1 / 6, 1e-12))
actor, parts = en.distribute(sp, 0.2)
check("capped: actor keeps (1 − cap)·φ_e and the parts add back to φ_e",
      close(actor, 0.1, 1e-15) and actor + math.fsum(a for _, a in parts) == 0.2)

section("min_hazard (P8.5.2), NaN head, failed removal, info")
sp = en.split([E("flash"), E("trade_setup")], 0.01, {0: 0.0, 1: 0.0}, ACFG)
check("q = 0.01 < min_hazard -> band floors for every enabler",
      shares(sp) == [ACFG.other_share_band[0], lo] and "min_hazard" in sp.flags)
check("q = 0.01: logged per enabler too", all("min_hazard" in s.flags for s in sp.shares))
sp = en.split([E("flash")], float("nan"), {0: 0.1}, ACFG)
check("NaN head -> floor, flagged hazard_nan", shares(sp) == [0.0] and "hazard_nan" in sp.flags)
sp = en.split([E("trade_setup")], 0.4, {0: None}, ACFG)
check("failed removal -> floor, flagged remove_failed", shares(sp) == [lo] and sp.shares[0].flags == ["remove_failed:trade_setup"])
sp = en.split([{"kind": "info", "action_id": None}], 0.4, {}, ACFG)
check("info -> other floor, routed to the clock", shares(sp) == [ACFG.other_share_band[0]] and sp.shares[0].to_clock)
check("no enablers -> actor keeps everything", en.split([], 0.4, {}, ACFG).actor_share == 1.0)

section("fixed mode (P8.5.3)")
fs = ACFG.fixed_shares
sp = en.split([E("flash")], None, {}, FIXED)
check("fixed flash share", close(shares(sp)[0], fs.flash, 1e-15) and sp.mode == "fixed")
sp = en.split([E("flash"), E("smoke_cover"), E("trade_setup"), E("he_damage"), E("fire_pressure")], None, {}, FIXED)
want = [fs.flash, fs.smoke_cover, fs.trade_setup, fs.he_damage, fs.fire_pressure]
scale = ACFG.enabler_total_cap / sum(want)
check("fixed: all five kinds, Σ 0.7 capped to 0.5 proportionally",
      sp.capped and all(close(x, w * scale, 1e-12) for x, w in zip(shares(sp), want)) and close(sp.total, 0.5, 1e-12))
check("fixed: q values ignored (min_hazard does not apply)", "min_hazard" not in en.split([E("flash")], 0.001, {}, FIXED).flags)
check("fixed: info gets 0", shares(en.split([{"kind": "info", "action_id": None}], None, {}, FIXED)) == [0.0])

# ── remove_i and heads ────────────────────────────────────────────────────────

section("remove_i states (P8.5.1) and head selection")
s_pre = make_snapshot(tick=4999, blind={202: 2.0}, smokes=[(750, 200, 0, 103)], fires=[(1400, 300, 0, 104)],
                      health={202: 60.0})
fl = action("flash", 104, tick=4900, blinded_enemies=[{"id": 202, "duration": 2.5}], blinded_teammates=[])
s = en.removal_state("flash", fl, s_pre)
check("flash -> unflash: the victim's blind is gone", s.players[s.index_of(202)].flash_duration_remaining == 0.0)
sm = action("smoke", 103, tick=4500, kind="durative")
s = en.removal_state("smoke_cover", sm, s_pre)
check("smoke_cover -> remove_utility: the smoke token is gone", not any(u.kind == "smoke" for u in s.utility))
mo = action("molotov", 104, tick=4500, kind="durative")
s = en.removal_state("fire_pressure", mo, s_pre)
check("fire_pressure -> remove_utility: the fire token is gone", not any(u.kind == "fire" for u in s.utility))
dmg = action("damage", 101, tick=4800, victim_id=202, damage_total=40.0, armor_damage=0.0)
ref = make_snapshot(tick=4799)
s = en.removal_state("he_damage", dmg, s_pre, ref)
check("he_damage (damage action) -> undo_damage: health back up", s.players[s.index_of(202)].health == 100.0)
he = action("he", 101, tick=4800, enemies_damaged=[{"id": 202, "dmg": 40.0}], teammates_damaged=[])
s = en.removal_state("he_damage", he, s_pre, ref)
check("he_damage (he action) -> undo_he: health back up", s.players[s.index_of(202)].health == 100.0)
pre_dead = make_snapshot(tick=4999, dead={102}, kills={203: 1}, damage={203: 100})
tk = action("kill", 203, tick=4700, victim_id=102, victim_health_before=100.0)
s = en.removal_state("trade_setup", tk, pre_dead, make_snapshot(tick=4699))
check("trade_setup -> undo_kill of the teammate's death: teammate alive", s.players[s.index_of(102)].alive == 1)
try:
    en.removal_state("info", None, s_pre)
    info_ok = False
except ValueError:
    info_ok = True
check("info has no intervention (ValueError)", info_ok)

out = {"hazard": np.arange(10, dtype=float)[None, :] / 100, "plant15": np.array([0.33]), "defuse15": np.array([0.44])}
kill = action("kill", 101, victim_id=204)
check("kill -> death hazard at the victim's slot", close(en.head_value(out, 0, kill, s_pre), s_pre.index_of(204) / 100, 1e-15))
check("damage -> death hazard of the victim",
      close(en.head_value(out, 0, action("damage", 101, victim_id=203, damage_total=1.0), s_pre), 0.07, 1e-15))
check("plant_complete -> plant15", en.head_value(out, 0, action("plant_complete", 201), s_pre) == 0.33)
check("defuse_complete -> defuse15", en.head_value(out, 0, action("defuse_complete", 101), s_pre) == 0.44)
out["plant15"] = np.array([np.nan])
check("NaN head -> None", en.head_value(out, 0, action("plant_complete", 201), s_pre) is None)

# ── inside a round ────────────────────────────────────────────────────────────

section("split inside the story round (stub hazard head)")
snaps, actions, windows, a = make_round()
k3, fl3 = a["kill_101_204"], a["flash"]
for label, cfg in (("hazard", CFG), ("fixed", replace(CFG, attribution=FIXED))):
    ra = engine.attribute_round(snaps, actions, windows, StubPredictor(win=round_win, hazard=round_hazard), cfg)
    comps = [c for sa in ra.actions for c in sa.components if c.get("event_id") == k3.action_id]
    k3c = next(c for c in next(sa for sa in ra.actions if sa.action_id == k3.action_id).components
               if c["source"] == "window")
    phi_e = k3c["phi_event"]
    info = phi_e * next((c["share"] for c in comps if c["kind"] == "info"), 0.0)
    total = k3c["phi"] + math.fsum(c["phi"] for c in comps) + info
    check(f"[{label}] actor remainder + enabler parts (+ info to clock) == φ_e", abs(total - phi_e) <= 1e-15,
          f"{total:+.6f} vs {phi_e:+.6f}")
    fl_share = next(c["share"] for sa in ra.actions for c in sa.components
                    if sa.action_id == fl3.action_id and c["source"] == "enabler")
    tr = [sa for sa in ra.actions if sa.type == "trade_setup"]
    if label == "hazard":
        q_pre = round_hazard(next(s for s in snaps if s.meta.tick == 5299))[snaps[0].index_of(204)]
        check("[hazard] flash share = (q − q_unflashed)/q from the stub head",
              close(fl_share, (q_pre - 0.05 * 4) / q_pre, 1e-12), f"{fl_share:.4f}")
        check("[hazard] trade_setup raw 0 (reviving 102 raises 204's hazard) -> trade floor",
              len(tr) == 1 and tr[0].actor_id_hash == 102 and close(tr[0].V_raw, lo * phi_e, 1e-15))
    else:
        check("[fixed] flash share = fixed 0.20", close(fl_share, fs.flash, 1e-15))
        check("[fixed] trade beneficiary 102 gets 0.20 · φ_e", close(tr[0].V_raw, fs.trade_setup * phi_e, 1e-15))
    check(f"[{label}] conservation ≤ 1e-9", abs(ra.conservation_residual) <= 1e-9, f"{ra.conservation_residual:.2e}")

sys.exit(summary())
