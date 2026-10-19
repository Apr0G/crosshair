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

from _harness import check, section, summary
from _fake_states import fake_arrays, finish, histories, make_model_dir, test_device
from _stubs import StubPredictor

from engine.model import calibrate
from engine.model.predict import Predictor, cache_key, edit_hash

DEV = test_device()
print(f"device: {DEV}")
TOL = 1e-5
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
    check("probabilities in [0, 1]", all(((v >= 0) & (v <= 1)).all() for v in out.values()))
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

    section("bf16 option")
    if DEV == "cuda":
        ob = Predictor(mdir, DEV, cache=False, bf16=True).predict(sub, sub_h)
        dmax = float(np.abs(ob["win"] - calg["win"]).max())     # same calibration.json as calg
        check("bf16 Predictor within 1e-2 of fp32 on win", dmax < 1e-2, f"max |diff| {dmax:.2e}")
    else:
        check("bf16 is ignored on CPU", Predictor(mdir, DEV, bf16=True).bf16 is False)

    section("no-aux-heads model")
    adir = Path(d) / "wp_noaux"
    make_model_dir(adir, vocab, norm, seed=3, aux_heads=False)
    oa = Predictor(adir, DEV).predict(sub[:10], sub_h[:10])
    check("missing heads come back NaN, win still predicted",
          np.isnan(oa["hazard"]).all() and np.isnan(oa["plant15"]).all() and np.isfinite(oa["win"]).all())

finish(summary())
