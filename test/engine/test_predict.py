"""
Inference engine (P6.4.1): the Amendment 1 contract (keys and shapes as StubPredictor);
cached == uncached; batch of 1 == batch of 512 element-wise; edit_hash differs for
different edits (and ignores labels); calibration.json is applied; windows use only
the factual past within temporal_window_s; dead players' hazard is exactly 0.

    .venv\\Scripts\\python.exe test\\engine\\test_predict.py [--cpu]
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from _harness import check, require, section, skip, summary
require("torch", "the Predictor needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _fake_states import fake_arrays, finish, histories, make_model_dir, test_device  # noqa: E402
from _stubs import StubPredictor

from engine.model import calibrate  # noqa: E402
from engine.model.predict import Predictor, cache_key, edit_hash  # noqa: E402

DEV = test_device()
print(f"device: {DEV}")
TOL = 1e-5
BF16_P99_LOGIT = 0.1
snaps, vocab, arrs, norm = fake_arrays(18, seed=51, n_snapshots=30)
hist = histories(snaps)
N512 = 512
sub, sub_h = snaps[:N512], hist[:N512]

with tempfile.TemporaryDirectory() as d:
    mdir = Path(d) / "wp_test"
    make_model_dir(mdir, vocab, norm, seed=3)

    section("contract")
    pred = Predictor(mdir, DEV)
    out = pred.predict(sub, sub_h)
    stub = StubPredictor(win=lambda s, h: 0.5)
    check("same keys as StubPredictor", set(out) == set(stub.predict(sub[:2], sub_h[:2])))
    check("shapes win (N,), hazard (N,10), plant15 / defuse15 (N,)",
          out["win"].shape == (N512,) and out["hazard"].shape == (N512, 10)
          and out["plant15"].shape == (N512,) and out["defuse15"].shape == (N512,))
    check("win finite and different states give different win (LOW-70)",
          bool(np.isfinite(out["win"]).all()) and float(np.ptp(out["win"])) > 1e-4,
          f"range {float(np.ptp(out['win'])):.2e}")
    dead = np.array([[p.alive == 0 for p in s.players] for s in sub])
    check("dead players' hazard exactly 0", bool((out["hazard"][dead] == 0).all()) and dead.any())
    check("alive players' hazard > 0", bool((out["hazard"][~dead] > 0).all()))
    try:
        pred.predict(sub[:2], sub_h[:1])
        check("length mismatch raises", False)
    except ValueError:
        check("length mismatch raises", True)
    check("empty input", pred.predict([], [])["win"].shape == (0,))

    section("cache")
    nocache = Predictor(mdir, DEV, cache=False)
    out_nc = nocache.predict(sub, sub_h)
    dmax = max(float(np.abs(out[k] - out_nc[k]).max()) for k in out)
    check("cached == uncached", dmax <= TOL, f"max |diff| {dmax:.2e}")
    hits0 = pred.hits
    again = pred.predict(sub, sub_h)
    check("second call served from the cache", pred.hits > hits0 and pred.misses == len({cache_key(s) for s in sub}))
    check("second call identical", all(np.array_equal(out[k], again[k]) for k in out))
    i = 200
    edited = sub[i].copy()
    j = next(k for k, p in enumerate(edited.players) if p.alive)
    edited.players[j].health = 1.0
    e_c  = pred.predict([edited], [sub_h[i]])
    e_nc = nocache.predict([edited], [sub_h[i]])
    check("edited present with cached history == uncached", abs(e_c["win"][0] - e_nc["win"][0]) <= TOL)
    check("the edit changes the prediction", abs(e_c["win"][0] - out["win"][i]) > 1e-7)

    section("batch-size invariance")
    single = [Predictor(mdir, DEV, cache=False).predict([s], [h]) for s, h in zip(sub[:64], sub_h[:64])]
    one = {k: np.concatenate([o[k] for o in single]) for k in out}
    dmax = max(float(np.abs(one[k] - out_nc[k][:64]).max()) for k in out)
    check("batch of 1 == batch of 512 element-wise (first 64 checked one by one)", dmax <= TOL, f"max |diff| {dmax:.2e}")
    p_small = Predictor(mdir, DEV, cache=False, batch_size=7)
    o7 = p_small.predict(sub, sub_h)
    dmax = max(float(np.abs(o7[k] - out_nc[k]).max()) for k in out)
    check("internal batch size 7 == 512", dmax <= TOL, f"max |diff| {dmax:.2e}")

    section("edit_hash")
    s0 = sub[100]
    check("identical copy -> same hash", edit_hash(s0) == edit_hash(s0.copy()))
    e1 = s0.copy()
    e1.players[0].health = (e1.players[0].health or 0) + 1
    e2 = s0.copy()
    e2.pair[0, 1, 1] = e2.pair[1, 0, 1] = 1.0
    e3 = s0.copy()
    e3.utility = e3.utility[:-1] if e3.utility else []
    e4 = s0.copy()
    e4.labels = {"round_won_ct": 1 - s0.labels["round_won_ct"]}
    hashes = [edit_hash(x) for x in (s0, e1, e2)] + ([edit_hash(e3)] if s0.utility else [])
    check("different edits -> different hashes", len(set(hashes)) == len(hashes))
    check("labels do not enter the hash", edit_hash(e4) == edit_hash(s0))
    check("cache key = (match, round, tick, edit_hash)",
          cache_key(s0) == (s0.meta.match_key, s0.meta.round_num, s0.meta.tick, edit_hash(s0)))

    section("windows")
    k = 250
    w = pred.window(sub[k], sub_h[k] + sub[k + 1:k + 5])
    check("future snapshots excluded from the window", all(x.meta.t_s < sub[k].meta.t_s for x in w[:-1]) and w[-1] is sub[k])
    check("window limited to temporal_window_s", all(sub[k].meta.t_s - x.meta.t_s <= pred.window_s + 1e-6 for x in w))
    check("window in time order", all(a.meta.t_s < b.meta.t_s for a, b in zip(w, w[1:])))
    fut = pred.predict([sub[k]], [sub_h[k] + list(sub[k + 1:k + 5])])
    check("passing future snapshots does not change the output", abs(fut["win"][0] - out["win"][k]) <= TOL)
    emb = pred.encode(sub[:5])
    check("encode -> (N, d_pool)", emb.shape == (5, pred.model.arch["d_pool"]))

    section("calibration applied")
    logit = np.log(out_nc["win"]) - np.log1p(-out_nc["win"])
    (mdir / "calibration.json").write_text(json.dumps({"T": 2.0, "mode": "single", "groups": {}}))
    cal = Predictor(mdir, DEV, cache=False).predict(sub, sub_h)
    check("single temperature: win = sigmoid(logit / T)", np.allclose(cal["win"], 1 / (1 + np.exp(-logit / 2.0)), atol=1e-6))
    check("hazard / plant15 untouched by calibration", np.allclose(cal["hazard"], out_nc["hazard"], atol=TOL))
    mv0 = sub[0].ctx.map_version
    ph = calibrate.phase_of([s.ctx.bomb_planted for s in sub], [s.ctx.time_remaining_s for s in sub],
                            [s.ctx.time_into_round_s for s in sub])
    g0 = f"{mv0}|{ph[0]}"
    (mdir / "calibration.json").write_text(json.dumps({"T": 1.0, "mode": "group", "groups": {g0: 4.0}}))
    calg = Predictor(mdir, DEV, cache=False).predict(sub, sub_h)
    in_g = np.array([s.ctx.map_version == mv0 for s in sub]) & (ph == ph[0])
    check("group temperatures applied to their group only",
          np.allclose(calg["win"][in_g], 1 / (1 + np.exp(-logit[in_g] / 4.0)), atol=1e-6)
          and np.allclose(calg["win"][~in_g], out_nc["win"][~in_g], atol=1e-6))
    # LOW-29: group keys go through the vocab in the Predictor as in the fit, so an unseen
    # map version is '<unk>' and a null one '<null>' (before: the raw string / 'None')
    odd = []
    for x, mv in zip(sub[:40], ["de_never_seen_v9"] * 20 + [None] * 20):
        y = x.copy()
        y.ctx.map_version = mv
        odd.append(y)
    odd_h = sub_h[:40]
    (mdir / "calibration.json").write_text(json.dumps({"T": 1.0, "mode": "single", "groups": {}}))
    p1 = Predictor(mdir, DEV, cache=False).predict(odd, odd_h)["win"]
    lo = np.log(p1) - np.log1p(-p1)
    groups = {f"<unk>|{p}": 3.0 for p in calibrate.PHASES} | {f"<null>|{p}": 5.0 for p in calibrate.PHASES}
    (mdir / "calibration.json").write_text(json.dumps({"T": 1.0, "mode": "group", "groups": groups}))
    pg = Predictor(mdir, DEV, cache=False).predict(odd, odd_h)["win"]
    T_exp = np.array([3.0] * 20 + [5.0] * 20)
    check("group T keyed '<unk>' / '<null>' applies to unseen / null map versions (LOW-29)",
          np.allclose(pg, 1 / (1 + np.exp(-lo / T_exp)), atol=1e-6))
    check("Predictor keys == fit keys (map_versions_through_vocab vs map_versions_from_arrays)",
          list(calibrate.map_versions_through_vocab([s.ctx.map_version for s in sub[:50]], vocab))
          == list(calibrate.map_versions_from_arrays({k: v[:50] for k, v in arrs.items()}, vocab)))
    (mdir / "calibration.json").write_text(json.dumps({"T": 1.0, "mode": "group", "groups": {g0: 4.0}}))

    section("bf16 option")
    check("bf16 is ignored on CPU", Predictor(mdir, "cpu", bf16=True).bf16 is False)
    if DEV == "cuda":
        ob = Predictor(mdir, DEV, cache=False, bf16=True).predict(sub, sub_h)
        # LOW-79: p99 of |logit diff| (same calibration.json as calg), not the max of a
        # probability diff, which swings with GPU load (unpinned ROCm nondeterminism)
        lg = lambda p: np.log(p) - np.log1p(-p)   # noqa: E731
        d99 = float(np.quantile(np.abs(lg(ob["win"]) - lg(calg["win"])), 0.99))
        check(f"bf16 Predictor p99 |logit diff| vs fp32 < {BF16_P99_LOGIT}", d99 < BF16_P99_LOGIT,
              f"p99 {d99:.2e}")
    else:
        skip("bf16 Predictor tolerance vs fp32", "bf16 runs on the GPU only")

    section("no-aux-heads model")
    adir = Path(d) / "wp_noaux"
    make_model_dir(adir, vocab, norm, seed=3, aux_heads=False)
    oa = Predictor(adir, DEV).predict(sub[:10], sub_h[:10])
    check("missing heads come back NaN, win still predicted",
          np.isnan(oa["hazard"]).all() and np.isnan(oa["plant15"]).all() and np.isfinite(oa["win"]).all())

finish(summary())
