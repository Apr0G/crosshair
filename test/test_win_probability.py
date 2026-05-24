"""
Tests for win_probability.py

Run from repo root:
    python test/test_win_probability.py

Requires a trained model at data/win_prob.lgb.
Train first with:  python src/win_probability.py --train
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from win_probability import predict, predict_batch

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

_results: list[tuple[bool, str]] = []


def check(desc: str, condition: bool, detail: str = ""):
    ok = bool(condition)
    tag = PASS if ok else FAIL
    msg = f"  {tag}  {desc}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((ok, desc))


def between(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


# ── base state (5v5, mid-round, no plant, balanced) ──────────────────────────

def _state(**overrides) -> dict:
    base = dict(
        time_into_round_s  = 45.0,
        time_remaining_s   = 70.0,
        post_plant         = 0,
        alive_ct           = 5,
        alive_t            = 5,
        total_hp_ct        = 500,
        total_hp_t         = 500,
        total_armor_ct     = 500,
        total_armor_t      = 500,
        helmets_ct         = 5,
        helmets_t          = 5,
        has_defuser        = 0,
        equip_value_ct     = 20000,
        equip_value_t      = 20000,
        smokes_ct          = 2,
        smokes_t           = 2,
        flashes_ct         = 1,
        flashes_t          = 1,
        he_ct              = 1,
        he_t               = 1,
        molotovs_ct        = 0,
        molotovs_t         = 0,
        active_smokes      = 0,
        active_infernos    = 0,
        min_dist_ct_to_bomb= None,
        min_dist_t_to_bomb = None,
        ct_spotted_count   = 0,
        t_spotted_count    = 0,
        ct_heard_enemy     = 0,
        t_heard_enemy      = 0,
        ct_spread          = 500.0,
        t_spread           = 500.0,
        map                = "de_mirage",
    )
    base.update(overrides)
    return base


# ── tests ─────────────────────────────────────────────────────────────────────

print("\n── Output range ─────────────────────────────────────────────────────────")

p = predict(_state())
check("predict() returns float",    isinstance(p, float))
check("output is in [0, 1]",        between(p, 0.0, 1.0), f"{p:.4f}")
check("balanced state near 0.5",    between(p, 0.35, 0.65), f"{p:.4f}")


print("\n── Player advantage ─────────────────────────────────────────────────────")

p_5v1 = predict(_state(alive_ct=5, alive_t=1, total_hp_ct=500, total_hp_t=80))
p_1v5 = predict(_state(alive_ct=1, alive_t=5, total_hp_ct=80,  total_hp_t=500))
check("5v1 CT > 0.5",               p_5v1 > 0.5, f"{p_5v1:.4f}")
check("1v5 CT < 0.5",               p_1v5 < 0.5, f"{p_1v5:.4f}")
check("5v1 CT > 1v5 CT",            p_5v1 > p_1v5)

p_5v3 = predict(_state(alive_ct=5, alive_t=3))
p_3v5 = predict(_state(alive_ct=3, alive_t=5))
check("5v3 CT > 3v5 CT",            p_5v3 > p_3v5)


print("\n── HP advantage ─────────────────────────────────────────────────────────")

p_high_ct = predict(_state(total_hp_ct=500, total_hp_t=100))
p_high_t  = predict(_state(total_hp_ct=100, total_hp_t=500))
check("CT high HP > CT low HP",     p_high_ct > p_high_t)


print("\n── Post-plant ───────────────────────────────────────────────────────────")

p_post = predict(_state(
    post_plant=1,
    min_dist_ct_to_bomb=500.0,
    min_dist_t_to_bomb=200.0,
    time_remaining_s=20.0,
))
check("post-plant state produces valid output", between(p_post, 0.0, 1.0), f"{p_post:.4f}")

# T should have advantage post-plant with bomb close and little time
p_t_close = predict(_state(
    post_plant=1,
    min_dist_ct_to_bomb=1500.0,
    min_dist_t_to_bomb=50.0,
    has_defuser=0,
    time_remaining_s=10.0,
    alive_ct=1,
    alive_t=3,
    total_hp_ct=80,
    total_hp_t=300,
))
check("T dominant post-plant < 0.5 for CT",    p_t_close < 0.5, f"{p_t_close:.4f}")

# CT with defuser close should be better than without
p_defuser     = predict(_state(post_plant=1, min_dist_ct_to_bomb=100.0, min_dist_t_to_bomb=800.0, has_defuser=1, time_remaining_s=25.0))
p_no_defuser  = predict(_state(post_plant=1, min_dist_ct_to_bomb=100.0, min_dist_t_to_bomb=800.0, has_defuser=0, time_remaining_s=25.0))
check("defuser > no defuser (post-plant, CT close)", p_defuser > p_no_defuser)


print("\n── Economy advantage ────────────────────────────────────────────────────")

p_rich_ct = predict(_state(equip_value_ct=40000, equip_value_t=4000))
p_rich_t  = predict(_state(equip_value_ct=4000,  equip_value_t=40000))
check("CT rich eco > T rich eco",   p_rich_ct > p_rich_t)


print("\n── Edge cases ───────────────────────────────────────────────────────────")

# 1v1 — should be close to 0.5
p_1v1 = predict(_state(alive_ct=1, alive_t=1, total_hp_ct=100, total_hp_t=100))
check("1v1 balanced near 0.5",      between(p_1v1, 0.3, 0.7), f"{p_1v1:.4f}")

# All zeros / min values
p_zero = predict(_state(
    alive_ct=0, alive_t=5,
    total_hp_ct=0, total_hp_t=500,
    equip_value_ct=0, equip_value_t=20000,
))
check("0 CT alive < 0.1",           p_zero < 0.1, f"{p_zero:.4f}")

# Unknown map falls back gracefully
p_unknown_map = predict(_state(map="de_unknown"))
check("unknown map doesn't crash",  between(p_unknown_map, 0.0, 1.0), f"{p_unknown_map:.4f}")

# Null distances pre-plant
p_null_dist = predict(_state(min_dist_ct_to_bomb=None, min_dist_t_to_bomb=None))
check("null bomb distances OK",     between(p_null_dist, 0.0, 1.0), f"{p_null_dist:.4f}")

# All maps produce valid output
maps = ["de_mirage", "de_dust2", "de_inferno", "de_nuke", "de_ancient", "de_anubis", "de_vertigo"]
map_preds = {m: predict(_state(map=m)) for m in maps}
check("all maps in [0,1]",          all(between(v, 0.0, 1.0) for v in map_preds.values()))


print("\n── predict_batch ────────────────────────────────────────────────────────")

states = [_state(alive_ct=5, alive_t=i) for i in range(1, 6)]
batch  = predict_batch(states)
check("batch returns correct length",   len(batch) == 5)
check("batch all in [0,1]",             all(between(float(v), 0.0, 1.0) for v in batch))
check("batch monotone: more T = lower CT WP", all(batch[i] > batch[i+1] for i in range(len(batch)-1)),
      " ".join(f"{v:.3f}" for v in batch))


print("\n── Monotonicity ─────────────────────────────────────────────────────────")

# Increasing CT alive count should increase CT WP
ct_alive_preds = [predict(_state(alive_ct=n, alive_t=3, total_hp_ct=n*100, total_hp_t=300))
                  for n in range(1, 6)]
check("CT alive 1→5 is monotone increasing",
      all(ct_alive_preds[i] < ct_alive_preds[i+1] for i in range(len(ct_alive_preds)-1)),
      " ".join(f"{v:.3f}" for v in ct_alive_preds))

# More time remaining pre-plant should slightly favour T (more time to execute)
# — not strictly monotone but we just check the endpoints
p_low_time  = predict(_state(time_remaining_s=5.0,  post_plant=0))
p_high_time = predict(_state(time_remaining_s=100.0, post_plant=0))
check("output valid at low time remaining",  between(p_low_time,  0.0, 1.0), f"{p_low_time:.4f}")
check("output valid at high time remaining", between(p_high_time, 0.0, 1.0), f"{p_high_time:.4f}")


# ── summary ───────────────────────────────────────────────────────────────────

passed = sum(1 for ok, _ in _results if ok)
total  = len(_results)
failed = [(desc) for ok, desc in _results if not ok]

print(f"\n{'─'*60}")
print(f"  {passed}/{total} passed")
if failed:
    print("  Failed:")
    for d in failed:
        print(f"    - {d}")
print()
sys.exit(0 if not failed else 1)
