"""
Hand-built, valid `Snapshot` rounds for the model tests (P5.3, P6).

The real state builder (engine/state/builder.py) is tested on `_synthetic.py` through
the builder itself (P3.9.2). The model tests only need plausible, schema-valid
snapshots in volume, so they are made here directly from the schema dataclasses:

    from _fake_states import make_round, make_corpus, test_device
    snaps = make_round(40, seed=3, ct_wins=True)         # one round, every snapshot validates
    corpus = make_corpus(32, seed=0)                     # 32 rounds over 8 matches, 2 tiers, 2 maps

A round: 5 v 5, t_s from 0 (freeze end) in mostly 0.5 s grid steps with irregular
boundary samples in between; deaths spread over the round (the losing side loses more
players), so alive counts only decrease; optional plant (and defuse in CT-won planted
rounds); smokes / fires / decoys with lifetimes; a symmetric NaN-diagonal pair table;
labels attached the way engine/state/builder.py attaches them (`round_won_ct`,
`dies_within_5s` x10, `plant_within_15s`, `defuse_within_15s`,
`time_to_next_engagement_s`, `round_end_reason`, `is_terminal`).
"""
import math
import sys
import zlib

import numpy as np

from _harness import ROOT  # importing _harness puts the repo root on sys.path
from engine.state.schema import (N_PLAYERS, PAIR_CHANNELS, ContextToken, PlayerToken, Snapshot,
                                 SnapshotMeta, UtilityToken)

REPO_ROOT = ROOT
WEAPONS = ("rifle", "sniper", "smg", "pistol", "shotgun")
SITES   = ("A", "B")
H_DIE, H_PLANT, H_DEFUSE, TTE_CAP = 5.0, 15.0, 15.0, 30.0


def test_device() -> str:
    """"cuda" when available and `--cpu` is not on the command line, else "cpu"."""
    if "--cpu" in sys.argv:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _window(e, t, h):
    """Builder semantics: None if the event already happened, 0 if it never does."""
    if e is None:
        return 0
    if e < t:
        return None
    return int(e < t + h)


def _times(n: int, rng, tick_rate: int) -> tuple[list[float], list[tuple[str, ...]]]:
    times, kinds, t = [0.0], [("grid",)], 0.0
    grid_next = 0.5
    while len(times) < n:
        if rng.random() < 0.2 and grid_next - t > 3.0 / tick_rate:
            # a boundary pair around an "action" between two grid ticks
            mid = t + (grid_next - t) * rng.uniform(0.3, 0.7)
            for off, kind in ((-1, "boundary_pre"), (1, "boundary_post")):
                if len(times) < n:
                    times.append(round((mid + off / tick_rate) * tick_rate) / tick_rate)
                    kinds.append((kind,))
            t = times[-1]
            continue
        times.append(grid_next)
        kinds.append(("grid",))
        t, grid_next = grid_next, grid_next + 0.5
    kinds[-1] = tuple(sorted(set(kinds[-1]) | {"terminal"}))
    return times, kinds


def make_round(n_snapshots: int, seed: int, ct_wins: bool, tick_rate: int = 128, *,
               match_key: str = "fake-0", round_num: int = 1, tier: str = "pro",
               map_version: str = "de_fake@1", check: bool = True) -> list[Snapshot]:
    """One plausible round of `n_snapshots` valid snapshots (see module docstring)."""
    if n_snapshots < 2:
        raise ValueError("a round needs at least 2 snapshots")
    rng   = np.random.default_rng(seed)
    times, kinds = _times(n_snapshots, rng, tick_rate)
    T_end = times[-1]
    sides = ["ct"] * 5 + ["t"] * 5
    base  = 10_000 + (zlib.crc32(match_key.encode()) % 40) * 10
    ids   = [base + i for i in range(N_PLAYERS)]

    # deaths: the losing side loses 3-5, the winner 0-3; times inside the round
    lose_side = "t" if ct_wins else "ct"
    death_t: list[float | None] = [None] * N_PLAYERS
    for side in ("ct", "t"):
        k = int(rng.integers(3, 6)) if side == lose_side else int(rng.integers(0, 4))
        slots = [i for i in range(N_PLAYERS) if sides[i] == side]
        for i in rng.choice(slots, size=k, replace=False):
            death_t[int(i)] = float(rng.uniform(0.15, 0.97) * T_end)
    killer: list[int | None] = [None] * N_PLAYERS
    for i, d in enumerate(death_t):
        if d is not None:
            enemies = [j for j in range(N_PLAYERS) if sides[j] != sides[i]
                       and (death_t[j] is None or death_t[j] > d)]
            killer[i] = int(rng.choice(enemies)) if enemies else None
    engagements = sorted(d for d in death_t if d is not None)

    carrier = 5
    plant_t = None
    if (not ct_wins and rng.random() < 0.7) or (ct_wins and rng.random() < 0.3):
        cand = 0.55 * T_end
        if death_t[carrier] is None or death_t[carrier] > cand:
            plant_t = round(cand * tick_rate) / tick_rate
    site     = SITES[int(rng.integers(0, 2))] if plant_t is not None else None
    defuse_t = T_end if (ct_wins and plant_t is not None) else None
    defuser  = next((i for i in range(5) if death_t[i] is None), None) if defuse_t is not None else None
    if defuser is None:
        defuse_t = None

    start = [np.array([rng.uniform(-200, 200), rng.uniform(-400, 400), 0.0]) if s == "ct"
             else np.array([3000 + rng.uniform(-200, 200), rng.uniform(-400, 400), 0.0]) for s in sides]
    vel   = [rng.normal(0, 120, 3) * np.array([1, 1, 0]) for _ in range(N_PLAYERS)]
    vel   = [v * min(1.0, 250.0 / max(float(np.linalg.norm(v)), 1e-9)) for v in vel]   # <= 250 u/s (run speed)
    weap  = [WEAPONS[int(rng.integers(0, len(WEAPONS)))] for _ in range(N_PLAYERS)]
    equip = [float(rng.choice([800.0, 2900.0, 4700.0, 5700.0])) for _ in range(N_PLAYERS)]
    money = [float(rng.integers(0, 60)) * 100.0 for _ in range(N_PLAYERS)]
    first_seen = [float(rng.uniform(0, T_end)) if rng.random() < 0.8 else None for _ in range(N_PLAYERS)]
    first_shot = [float(rng.uniform(0, T_end)) if rng.random() < 0.7 else None for _ in range(N_PLAYERS)]
    no_reload_prop = rng.random() < 0.3       # some "parsers" lack is_reloading

    # utility: (kind, start, life, pos, owner)
    utils = []
    for _ in range(int(rng.integers(0, 7))):
        kind  = str(rng.choice(["smoke", "smoke", "fire", "decoy"]))
        life  = {"smoke": 18.0, "fire": 7.0, "decoy": 15.0}[kind]
        owner = int(rng.integers(0, N_PLAYERS))
        utils.append((kind, float(rng.uniform(0, T_end)), life,
                      (float(rng.uniform(0, 3000)), float(rng.uniform(-500, 500)), 0.0), owner))
    score_ct, score_t = int(rng.integers(0, 12)), int(rng.integers(0, 12))
    los_seed = rng.integers(0, 2, size=(N_PLAYERS, N_PLAYERS))

    snaps: list[Snapshot] = []
    for si, t in enumerate(times):
        alive = [d is None or t < d for d in death_t]
        planted = plant_t is not None and t >= plant_t
        players = []
        pos_now = []
        for i in range(N_PLAYERS):
            stop  = t if alive[i] else death_t[i]
            pos   = start[i] + vel[i] * min(stop, 25.0) / 10.0
            pos_now.append(pos)
            kills = float(sum(1 for j in range(N_PLAYERS) if killer[j] == i and death_t[j] <= t))
            common = dict(side=sides[i], alive=int(alive[i]), x=float(pos[0]), y=float(pos[1]),
                          z=float(pos[2]), nav_area=int(abs(pos[0]) // 300) * 10 + int(abs(pos[1]) // 300),
                          callout=f"area{int(pos[0] // 800)}_{int(pos[1] // 800)}",
                          grid_cell=int((pos[0] + 4000) // 512) * 32 + int((pos[1] + 4000) // 512),
                          seconds_since_last_seen_by_enemy=(None if first_seen[i] is None or first_seen[i] > t
                                                            else float(min(t - first_seen[i], 7.0))),
                          seconds_since_last_shot=(None if first_shot[i] is None or first_shot[i] > t
                                                   else float(min(t - first_shot[i], 9.0))),
                          kills_round=kills, damage_round=100.0 * kills,
                          player_id=ids[i], tier=tier)
            if alive[i]:
                blind = float(rng.uniform(0.2, 2.5)) if rng.random() < 0.05 else 0.0
                players.append(PlayerToken(
                    **common, yaw=float(rng.uniform(-180, 180)), pitch=float(rng.uniform(-30, 30)),
                    speed=float(np.linalg.norm(vel[i])) if t < 25.0 else 0.0,
                    health=float(max(1, 100 - int(rng.integers(0, 40)) * (si % 3 == 0))),
                    armor=100.0 if equip[i] > 1000 else 0.0, has_helmet=int(equip[i] > 1000),
                    has_defuser=int(sides[i] == "ct" and equip[i] > 4000),
                    has_c4=int(i == carrier and not planted),
                    weapon_class=weap[i], n_smoke=int(i % 2), n_flash=int(i % 3 == 0), n_he=0,
                    n_molly=int(i == 2), n_decoy=0,
                    is_scoped=int(weap[i] == "sniper" and rng.random() < 0.5),
                    is_ducking=int(rng.random() < 0.1), is_walking=int(rng.random() < 0.2),
                    is_reloading=None if no_reload_prop else int(rng.random() < 0.05),
                    is_defusing=int(defuser == i and defuse_t is not None and defuse_t - 5 <= t),
                    is_planting=int(i == carrier and plant_t is not None and plant_t - 3 <= t < plant_t),
                    flash_duration_remaining=blind, in_smoke=0, on_fire=0,
                    money=money[i], equipment_value=equip[i],
                    spotted_by_count=int(rng.integers(0, 3)), heard_enemy=int(rng.random() < 0.3)))
            else:
                players.append(PlayerToken(
                    **common, yaw=None, pitch=None, speed=None, health=0.0, armor=0.0,
                    has_helmet=None, has_defuser=None, has_c4=None, weapon_class=None,
                    n_smoke=None, n_flash=None, n_he=None, n_molly=None, n_decoy=None,
                    is_scoped=None, is_ducking=None, is_walking=None, is_reloading=None,
                    is_defusing=None, is_planting=None, flash_duration_remaining=0.0,
                    in_smoke=None, on_fire=None, money=None, equipment_value=None,
                    spotted_by_count=None, heard_enemy=None))

        active = [u for u in utils if u[1] <= t < u[1] + u[2]][-16:]
        utility = [UtilityToken(kind=k, x=p[0], y=p[1], z=p[2], time_left_s=float(st + life - t),
                                owner_side=sides[o], owner_id=ids[o],
                                radius_units=144.0 if k == "smoke" else (None if k == "decoy" else 120.0))
                   for k, st, life, p, o in active]

        n_ct = sum(alive[:5])
        n_t  = sum(alive[5:])
        defusing = defuser is not None and defuse_t - 5 <= t and alive[defuser]
        ctx = ContextToken(
            map_version=map_version, tier=tier,
            time_remaining_s=float(max(0.0, 40.0 - (t - plant_t))) if planted else float(max(0.0, 115.0 - t)),
            time_into_round_s=float(t), bomb_planted=int(planted), bomb_site=site if planted else None,
            time_since_plant_s=float(t - plant_t) if planted else None,
            defuse_in_progress=int(bool(defusing)), defuse_with_kit=int(players[defuser].has_defuser or 0) if defusing else None,
            bomb_carrier_alive=None if planted else int(alive[carrier]),
            alive_ct=n_ct, alive_t=n_t,
            equip_ct=float(sum(equip[i] for i in range(5) if alive[i])),
            equip_t=float(sum(equip[i] for i in range(5, 10) if alive[i])),
            # every roster account, dead players included (the builder's definition, HIGH-05)
            money_ct=float(sum(money[i] for i in range(5))),
            money_t=float(sum(money[i] for i in range(5, 10))),
            loss_streak_ct=round_num % 3, loss_streak_t=0,
            score_ct=score_ct, score_t=score_t, round_num=round_num, half=1 if round_num <= 12 else 2,
            is_overtime=0, n_active_smokes=sum(u.kind == "smoke" for u in utility),
            n_active_fires=sum(u.kind == "fire" for u in utility))

        pair = np.full((N_PLAYERS, N_PLAYERS, len(PAIR_CHANNELS)), np.nan, dtype=np.float32)
        for i in range(N_PLAYERS):
            for j in range(i + 1, N_PLAYERS):
                enemy = sides[i] != sides[j]
                both  = alive[i] and alive[j]
                d     = float(math.dist(pos_now[i], pos_now[j]))
                # los: alive enemy pairs only; smoke_between: alive pairs only, NaN otherwise
                # (as engine/state/pairwise.py writes them; LOW-64)
                vals  = (float((los_seed[i, j] + si // 4) % 2) if enemy and both else np.nan,
                         d, d * 1.15,
                         float(los_seed[j, i] == 1 and bool(active)) if both else np.nan,
                         float(not enemy))
                pair[i, j] = pair[j, i] = vals

        tte = next((e - t for e in engagements if e >= t), None)
        labels = {
            "round_won_ct":              int(ct_wins),
            "round_end_reason":          "ct_win_elimination" if ct_wins else "t_win_elimination",
            "dies_within_5s":            [_window(death_t[i], t, H_DIE) for i in range(N_PLAYERS)],
            "plant_within_15s":          _window(plant_t, t, H_PLANT),
            "defuse_within_15s":         _window(defuse_t, t, H_DEFUSE),
            "time_to_next_engagement_s": float(TTE_CAP if tte is None else min(TTE_CAP, tte)),
            "is_terminal":               int("terminal" in kinds[si]),
        }
        meta = SnapshotMeta(match_key=match_key, round_num=round_num,
                            tick=int(round(1000 + t * tick_rate)), t_s=float(t), kinds=kinds[si])
        s = Snapshot(players=players, utility=utility, ctx=ctx, pair=pair, meta=meta, labels=labels)
        if check:
            problems = s.validate()
            if problems:
                raise AssertionError(f"fake snapshot {match_key}#{round_num} t={t}: {problems}")
        snaps.append(s)
    return snaps


def make_corpus(n_rounds: int, seed: int, *, n_snapshots: int = 30, rounds_per_match: int = 4,
                tiers: tuple[str, ...] = ("pro", "faceit_10"),
                maps: tuple[str, ...] = ("de_fake@1", "de_other@2"), tick_rate: int = 128,
                prefix: str = "fake") -> list[Snapshot]:
    """`n_rounds` rounds grouped into matches of `rounds_per_match`; tier and map
    alternate by match; the winner is random (seeded)."""
    rng = np.random.default_rng(seed)
    out: list[Snapshot] = []
    for r in range(n_rounds):
        m = r // rounds_per_match
        out += make_round(n_snapshots, seed=int(rng.integers(0, 2**31)), ct_wins=bool(rng.random() < 0.5),
                          tick_rate=tick_rate, match_key=f"{prefix}-{seed}-{m}",
                          round_num=r % rounds_per_match + 1, tier=tiers[m % len(tiers)],
                          map_version=maps[m % len(maps)])
    return out


# ── shared model-test helpers ─────────────────────────────────────────────────

def fake_arrays(n_rounds: int = 16, seed: int = 0, **kw):
    """(snapshots, vocab, world-unit arrays, norm stats) for a fake corpus."""
    from engine.data import dataset
    snaps = make_corpus(n_rounds, seed, **kw)
    vocab = dataset.Vocab().fit(snaps)
    arrs  = dataset.tensorize(snaps, vocab)
    return snaps, vocab, arrs, dataset.NormStats.fit([arrs])


def histories(snaps: list) -> list[list]:
    """The factual past (same round, earlier) of every snapshot, as Predictor expects."""
    seen: dict = {}
    out = []
    for s in snaps:
        k = (s.meta.match_key, s.meta.round_num)
        out.append(list(seen.get(k, [])))
        seen.setdefault(k, []).append(s)
    return out


def make_model_dir(path, vocab, norm, seed: int = 0, **flags):
    """An untrained (seeded) WPModel saved as a model directory Predictor can load."""
    import torch
    from engine import config
    from engine.model import checkpoints
    from engine.model.wp_model import WPModel
    cfg = config.load()
    torch.manual_seed(seed)
    model = WPModel(cfg, vocab.sizes(), **flags)
    model.set_norm(norm)
    checkpoints.write_model_files(path, vocab, norm, cfg)
    checkpoints.save(path / "best.pt", model)
    return model.eval()


def finish(code: int) -> None:
    """End a test with `code`. When the test used the GPU, flush and leave through
    os._exit: on this machine (torch 2.9.1 + ROCm 7.2.1 on Windows) the interpreter
    sometimes hangs forever in the HIP runtime's teardown after every check has
    passed and atexit has run (seen in test_wp_model). Skipping the C++ static
    destructors avoids that; nothing is lost because results are already printed."""
    import os
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)
