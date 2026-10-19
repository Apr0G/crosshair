"""
WPModel assembly (P6.2.5): forward on the fake corpus with temporal windows; gradient
reaches every parameter (no dead branches); deterministic under a fixed seed; fp32 and
bf16 logits agree within 1e-2 (GPU only, skipped on CPU); ablation switches remove
exactly their modules' parameters; parameter count printed (estimate 2-4 M).

    .venv\\Scripts\\python.exe test\\engine\\test_wp_model.py [--cpu]
"""

import numpy as np

from _harness import check, require, section, skip, summary
torch = require("torch", "needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _fake_states import fake_arrays, finish, test_device  # noqa: E402

from engine import config  # noqa: E402
from engine.data.dataset import CTX_LAYOUT, PLAYER_LAYOUT, UTIL_LAYOUT  # noqa: E402
from engine.model import batching  # noqa: E402
from engine.model.train import compute_loss  # noqa: E402
from engine.model.wp_model import WPModel  # noqa: E402

DEV = test_device()
print(f"device: {DEV}")
cfg = config.load()
snaps, vocab, arrs, norm = fake_arrays(16, seed=11)
wi = batching.WindowIndex(arrs["round_key"], arrs["t_s"], cfg.model.temporal_window_s)
targets = np.random.default_rng(0).choice(len(snaps), 48, replace=False)
nb = batching.gather(arrs, [wi.window(i) for i in targets])


def build(seed=0, **flags) -> WPModel:
    torch.manual_seed(seed)
    m = WPModel(cfg, vocab.sizes(), **flags)
    m.set_norm(norm)
    return m.to(DEV)


model = build()
print(f"default model parameters: {model.n_params:,} (vocab sizes {vocab.sizes()})")

section("forward")
b = batching.to_device(nb, DEV)
model.eval()
with torch.no_grad():
    out = model(b)
check("win [B]", tuple(out["win"].shape) == (48,))
check("hazard [B,10]", tuple(out["hazard"].shape) == (48, 10))
check("plant15 / defuse15 / tte [B]", all(tuple(out[k].shape) == (48,) for k in ("plant15", "defuse15", "tte")))
check("finite win / plant / defuse / tte", all(bool(torch.isfinite(out[k]).all()) for k in ("win", "plant15", "defuse15", "tte")))
with torch.no_grad():
    h = model.encode(b)
check("encode -> h_t [U, d_pool]", tuple(h.shape) == (nb["player_num"].shape[0], cfg.model.d_pool))
with torch.no_grad():
    plain = model({k: v for k, v in b.items() if not k.startswith("win_")})
check("forward without windows: one output per snapshot", tuple(plain["win"].shape) == (nb["player_num"].shape[0],))
check("parameter count in the plan's 2-4 M estimate", 2_000_000 <= model.n_params <= 4_000_000, f"{model.n_params:,}")

section("gradient reaches every parameter")
# make every nullable field missing at least once so every missing vector is exercised
nb2 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in nb.items()}
for block, lay in (("ctx_num", CTX_LAYOUT), ("util_num", UTIL_LAYOUT), ("player_num", PLAYER_LAYOUT)):
    pres = [lay.col(f"present_{s}") for s in lay.presence_sources]
    nb2[block][0, ..., pres] = 0.0
nb2["util_mask"][0, 0] = True
model.train()
b2 = batching.to_device(nb2, DEV)
out2 = model(b2)
loss, parts = compute_loss(out2, b2, cfg.model.loss_weights, cfg.model.terminal_weight)
model.zero_grad()
loss.backward()
dead_params = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
check("loss finite with every part", bool(torch.isfinite(loss)) and set(parts) == {"win", "hazard", "plant", "defuse", "tte"},
      str({k: round(v, 3) for k, v in parts.items()}))
check("every parameter has a non-zero gradient", not dead_params, ", ".join(dead_params[:8]))

section("determinism under a fixed seed")
m1, m2 = build(seed=5).eval(), build(seed=5).eval()
same_init = all(torch.equal(a, c) for a, c in zip(m1.state_dict().values(), m2.state_dict().values()))
check("same seed -> identical parameters", same_init)
with torch.no_grad():
    o1, o2 = m1(b), m2(b)
check("same seed -> identical outputs", all(torch.equal(o1[k], o2[k]) for k in ("win", "plant15", "tte")))
check("different seed -> different parameters",
      not torch.equal(build(seed=6).heads.win.net[0].weight, m1.heads.win.net[0].weight))

section("fp32 vs bf16")
if DEV == "cuda" and torch.cuda.is_bf16_supported():
    with torch.no_grad():
        f32 = m1(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf = m1(b)
    worst = 0.0
    for k in ("win", "plant15", "defuse15", "tte"):
        worst = max(worst, float((f32[k].float() - bf[k].float()).abs().max()))
    alive = f32["alive"]
    worst = max(worst, float((f32["hazard"][alive].float() - bf["hazard"][alive].float()).abs().max()))
    check("fp32 and bf16 logits agree within 1e-2", worst < 1e-2, f"max |diff| {worst:.2e}")
else:
    skip("fp32 and bf16 logits agree within 1e-2", "needs a CUDA/ROCm device with bf16")

section("ablation switches")
full = build()


def count(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


nt, npw, na = build(temporal=False), build(pairwise=False), build(aux_heads=False)
aux_mods = (full.heads.plant, full.heads.defuse, full.heads.tte, full.heads.z_to_player, full.heads.hazard)
check("--no-temporal removes exactly the temporal encoder", nt.n_params == full.n_params - count(full.temporal),
      f"{full.n_params:,} -> {nt.n_params:,}")
check("--no-pairwise removes exactly the pair-bias layer", npw.n_params == full.n_params - count(full.set_enc.pair_bias),
      f"{full.n_params:,} -> {npw.n_params:,}")
check("--no-aux-heads removes exactly the aux heads", na.n_params == full.n_params - sum(count(m) for m in aux_mods),
      f"{full.n_params:,} -> {na.n_params:,}")
with torch.no_grad():
    o_nt, o_na = nt.eval()(b), na.eval()(b)
check("no-temporal forward works", tuple(o_nt["win"].shape) == (48,))
check("no-aux-heads outputs win only (+alive)", set(o_na) == {"win", "alive"})
check("flags recorded", nt.flags == {"temporal": False, "pairwise": True, "aux_heads": True})

finish(summary())
