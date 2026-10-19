"""
Round assembly (P8.6) on the `_attr_states` story with stubbed W: the conservation
identity `Σ V_raw + clock_ct − clock_t = W(terminal) − W(freeze_end)` to 1e-9 (exact
windows) and 1e-6 (a forced sampled window); two kills on one tick get opposite-sign
direct shares that sum to the window ΔW; the decoy gets 0; `V = V_direct + V_enabler +
V_movement` per action; determinism (two runs -> identical values and checksums);
`to_frame` schema; predictor calls ≤ windows + 2 (call 1: breakpoints + enabler
removals, call 2: all drift intervals, then one per window); `attribute_match`.

    .venv\\Scripts\\python.exe test\\engine\\test_attribution_round.py
"""
import hashlib
import math
import sys
from dataclasses import replace

import polars as pl

from _harness import check, close, section, summary
from _cf_states import RuleStub
from _attr_states import FREEZE_END, TERMINAL, make_round, round_hazard, round_win, snapshot_at
from _stubs import StubPredictor

from engine import config
from engine.attribution import engine
from engine.attribution.types import RoundAttribution, ScoredAction

CFG = config.load()
SAMPLED = replace(CFG, attribution=replace(CFG.attribution, shapley_exact_max=2))


def row(ra, aid) -> ScoredAction:
    return next(sa for sa in ra.actions if sa.action_id == aid)


def window_phi(ra, aid, wid) -> float:
    return math.fsum(c.get("phi_event", c["phi"]) for c in row(ra, aid).components
                     if c.get("window_id") == wid and c["source"] in ("window", "pop", "fade"))


def checksum(ra) -> str:
    h = hashlib.sha256()
    for sa in ra.actions:
        h.update(f"{sa.action_id}|{sa.V!r}|{sa.V_direct!r}|{sa.V_enabler!r}|{sa.V_movement!r}".encode())
    h.update(f"{ra.clock_ct!r}|{ra.clock_t!r}".encode())
    return h.hexdigest()


snaps, actions, windows, a = make_round()

# ── conservation ──────────────────────────────────────────────────────────────

section("conservation identity")
runs = {}
for label, pred, cfg, tol in (("RuleStub/exact", RuleStub(round_win), CFG, 1e-9),
                              ("Stub+hazard/exact", StubPredictor(win=round_win, hazard=round_hazard), CFG, 1e-9),
                              ("Stub+hazard/sampled", StubPredictor(win=round_win, hazard=round_hazard), SAMPLED, 1e-6)):
    ra = engine.attribute_round(snaps, actions, windows, pred, cfg)
    runs[label] = ra
    lhs = math.fsum([sa.V_raw for sa in ra.actions] + [ra.clock_ct, -ra.clock_t])
    check(f"[{label}] Σ V_raw + clock_ct − clock_t = W(terminal) − W(freeze_end) (tol {tol:g})",
          abs(lhs - (ra.w_end - ra.w_start)) <= tol and abs(ra.conservation_residual) <= tol,
          f"residual {ra.conservation_residual:.2e}")
    check(f"[{label}] stats agree (conservation_ok, tolerance)",
          ra.stats["conservation_ok"] and ra.stats["conservation_tol"] == tol)
    check(f"[{label}] clock buckets non-negative, per-round clock identity",
          ra.clock_ct >= 0 and ra.clock_t >= 0 and ra.stats["clock_identity_ok"])
    check(f"[{label}] telescoping exact", abs(ra.stats["telescoping_residual"]) <= 1e-12)
ra = runs["Stub+hazard/exact"]
check("terminal W pinned to the round label (CT win -> 1.0)", ra.w_end == 1.0 and ra.stats["terminal_pinned"])
w_fe = round_win(snapshot_at(FREEZE_END))            # built independently of `snaps` (LOW-69)
check("w_start == W(freeze_end snapshot), computed independently, for every run",
      all(r.w_start == w_fe for r in runs.values()), f"{[r.w_start for r in runs.values()]} vs {w_fe}")
shuffled = snaps[1::2] + snaps[0::2]
ra_sh = engine.attribute_round(shuffled, actions, windows, StubPredictor(win=round_win, hazard=round_hazard), CFG)
check("w_start is the freeze_end snapshot even when the input list does not start with it",
      ra_sh.w_start == w_fe and shuffled[0].meta.tick != FREEZE_END)
check("Σ V_raw + clock_ct − clock_t == 1.0 − W(freeze_end) with both ends pinned independently",
      abs(math.fsum([sa.V_raw for sa in ra.actions] + [ra.clock_ct, -ra.clock_t]) - (1.0 - w_fe)) <= 1e-9)
check("sampled run used sampled Shapley for the 3-member window", runs["Stub+hazard/sampled"].stats["sampled_windows"] == ["w5000"])
check("no failed windows", not ra.stats["failed_windows"])
ra_np = engine.attribute_round(snaps, actions, windows, StubPredictor(win=round_win, hazard=round_hazard), CFG,
                               pin_terminal=False)
check("pin_terminal=False keeps the model's terminal W and still conserves",
      ra_np.w_end == round_win(snaps[-1]) == round_win(snapshot_at(TERMINAL))
      and abs(ra_np.conservation_residual) <= 1e-9)

# ── two kills on one tick, decoy ──────────────────────────────────────────────

section("two kills on one tick, decoy null")
k1, k2, dec = a["kill_101_202"], a["kill_203_102"], a["decoy"]
for label in ("Stub+hazard/exact", "Stub+hazard/sampled"):
    ra = runs[label]
    f1, f2 = window_phi(ra, k1.action_id, "w5000"), window_phi(ra, k2.action_id, "w5000")
    dW = round_win(snapshot_at(5001)) - round_win(snapshot_at(4999))
    fd = window_phi(ra, dec.action_id, "w5000")
    check(f"[{label}] both killers non-zero, opposite-sign (CT-positive) direct shares",
          f1 > 0 > f2, f"φ(101→202)={f1:+.4f} φ(203→102)={f2:+.4f}")
    check(f"[{label}] they sum (with the null decoy) to the window ΔW", abs(f1 + f2 + fd - dW) <= 1e-12)
    check(f"[{label}] decoy φ == 0 and decoy V == 0", fd == 0.0 and row(ra, dec.action_id).V == 0.0)
    check(f"[{label}] both kills read +0.1 for their own side (actor-signed V)",
          close(row(ra, k1.action_id).V, 0.1, 1e-12) and close(row(ra, k2.action_id).V, 0.1, 1e-12))

# ── V identity and rows ───────────────────────────────────────────────────────

section("V = V_direct + V_enabler + V_movement; rows")
ra = runs["Stub+hazard/exact"]
check("V identity holds for every row", all(sa.V == sa.V_direct + sa.V_enabler + sa.V_movement for sa in ra.actions))
check("V_raw is actor-signed V back to CT-positive",
      all(sa.V_raw == (sa.V if sa.actor_side == "ct" else -sa.V) for sa in ra.actions))
types = sorted(sa.type for sa in ra.actions)
check("one row per non-pseudo action + synthetic positioning (204) and trade_setup (102)",
      types == sorted(["rotation", "decoy", "smoke", "kill", "kill", "flash", "kill", "positioning", "trade_setup"]), str(types))
check("pseudo-actions have no own rows (folded into parents)",
      not any(sa.type in ("smoke_pop", "decoy_start") for sa in ra.actions))
check("smoke pop credited to the smoke action (+0.05)", close(row(ra, a["smoke"].action_id).V_direct, 0.05, 1e-12))
fl = row(ra, a["flash"].action_id)
check("flash: direct (its own window) and enabler (the kill it set up) both > 0", fl.V_direct > 0 and fl.V_enabler > 0)
rot = row(ra, a["rotation"].action_id)
check("rotation: movement only, method loo/exact", rot.V_movement > 0 and rot.V_direct == 0 and rot.method in ("loo", "exact"))
check("every row has method / n_coalition when it got value",
      all((sa.method is not None and sa.n_coalition) for sa in ra.actions if sa.components))
check("RoundAttribution fields", isinstance(ra, RoundAttribution) and ra.match_key == "cf_m1" and ra.round_num == 7)

# ── determinism ───────────────────────────────────────────────────────────────

section("determinism (P8.6.4)")
for cfg, label in ((CFG, "exact"), (SAMPLED, "sampled")):
    r1 = engine.attribute_round(snaps, actions, windows, StubPredictor(win=round_win, hazard=round_hazard), cfg)
    s2, a2, w2, _ = make_round()
    r2 = engine.attribute_round(list(reversed(s2)), list(reversed(a2)), list(reversed(w2)),
                                StubPredictor(win=round_win, hazard=round_hazard), cfg)
    check(f"[{label}] two runs (inputs rebuilt and reversed) -> identical checksums", checksum(r1) == checksum(r2))
    check(f"[{label}] identical V bit for bit", [sa.V for sa in r1.actions] == [sa.V for sa in r2.actions])

# ── predictor calls ───────────────────────────────────────────────────────────

section("predictor calls")
stub = StubPredictor(win=round_win, hazard=round_hazard)
ra = engine.attribute_round(snaps, actions, windows, stub, CFG)
nw = len(windows)
check(f"calls = 1 (breakpoints + enabler removals) + 1 (all drift) + {nw} (one per window)",
      stub.calls == ra.stats["predictor_calls"] == nw + 2, f"calls={stub.calls}")
check("calls ≤ windows + drift batches (2)", stub.calls <= nw + 2)
check("coalition evaluations counted", ra.stats["coalition_evals"] > 0 and stub.n_snapshots == ra.stats["n_snapshots_predicted"])
fx = engine.attribute_round(snaps, actions, windows, StubPredictor(win=round_win), replace(CFG, attribution=replace(
    CFG.attribution, enabler_mode="fixed")))
check("fixed mode builds no enabler removal states", fx.stats["n_enabler_states"] == 0)

# ── to_frame ──────────────────────────────────────────────────────────────────

section("to_frame (actions_valued.parquet)")
df = engine.to_frame([ra, runs["RuleStub/exact"]])
check("schema is FRAME_SCHEMA", dict(df.schema) == engine.FRAME_SCHEMA, str(dict(df.schema)))
check("one row per ScoredAction", df.height == len(ra.actions) + len(runs["RuleStub/exact"].actions))
check("h_pre left for P9 (all null)", df["h_pre"].null_count() == df.height)
check("required columns present",
      {"V", "V_direct", "V_enabler", "V_movement", "method", "n_coalition", "residual", "components"} <= set(df.columns))
check("components are JSON", all(isinstance(x, str) and x.startswith("[") for x in df["components"]))
check("V column equals the records",
      df.filter(pl.col("action_id") == a["flash"].action_id)["V"][0] == row(ra, a["flash"].action_id).V)
check("empty input -> empty frame with the schema", engine.to_frame([]).schema == pl.Schema(engine.FRAME_SCHEMA))

# ── attribute_match ───────────────────────────────────────────────────────────

section("attribute_match")
ras = engine.attribute_match(snaps, actions, windows, StubPredictor(win=round_win, hazard=round_hazard), CFG)
check("one RoundAttribution per round, same values as attribute_round",
      len(ras) == 1 and checksum(ras[0]) == checksum(ra))

sys.exit(summary())
