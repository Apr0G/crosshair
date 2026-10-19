"""
Heads (P6.2.4): shapes; dead players' hazard logits masked to -inf -> p = 0 exactly;
the raw hazard stays finite for the loss; `aux=False` builds only the win head.

    .venv\\Scripts\\python.exe test\\engine\\test_heads.py [--cpu]
"""

import numpy as np

from _harness import check, require, section, summary
torch = require("torch", "needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _fake_states import fake_arrays, finish, test_device  # noqa: E402

from engine import config  # noqa: E402
from engine.data.dataset import PLAYER_LAYOUT, SNAPSHOT_KEYS  # noqa: E402
from engine.model import batching  # noqa: E402
from engine.model.heads import Heads  # noqa: E402
from engine.model.wp_model import WPModel  # noqa: E402

DEV = test_device()
print(f"device: {DEV}")
torch.manual_seed(0)

section("Heads module")
heads = Heads(256, 128).to(DEV).eval()
z  = torch.randn(6, 256, device=DEV)
po = torch.randn(6, 10, 128, device=DEV)
alive = torch.rand(6, 10, device=DEV) > 0.4
with torch.no_grad():
    o = heads(z, po, alive)
check("keys", set(o) == {"win", "hazard", "hazard_raw", "plant15", "defuse15", "tte"})
check("win / plant15 / defuse15 / tte shape [B]", all(tuple(o[k].shape) == (6,) for k in ("win", "plant15", "defuse15", "tte")))
check("hazard shape [B,10]", tuple(o["hazard"].shape) == (6, 10))
check("dead hazard logits are -inf", bool(torch.isneginf(o["hazard"][~alive]).all()))
check("dead hazard probability is exactly 0", bool((torch.sigmoid(o["hazard"][~alive]) == 0).all()))
check("alive hazard logits finite", bool(torch.isfinite(o["hazard"][alive]).all()))
check("hazard_raw finite everywhere", bool(torch.isfinite(o["hazard_raw"]).all()))
win_only = Heads(256, 128, aux=False).to(DEV)
with torch.no_grad():
    o2 = win_only(z, po, alive)
check("aux=False: win only", set(o2) == {"win"})

section("through WPModel")
cfg = config.load()
snaps, vocab, arrs, norm = fake_arrays(4, seed=7)
model = WPModel(cfg, vocab.sizes()).to(DEV).eval()
model.set_norm(norm)
a = {k: v.copy() for k, v in arrs.items()}
a["player_mask"][3, 2] = False                       # an absent token
with torch.no_grad():
    out = model(batching.to_device({k: a[k] for k in SNAPSHOT_KEYS}, DEV))
dead = (a["player_num"][..., PLAYER_LAYOUT.col("alive")] < 0.5) | ~a["player_mask"]
p = torch.sigmoid(out["hazard"]).cpu().numpy()
check("fixture has dead players", bool(dead.any()))
check("dead and absent players: hazard p = 0", bool((p[dead] == 0).all()))
check("alive players: hazard p > 0 and differs between players / states (LOW-70)",
      bool((p[~dead] > 0).all()) and float(np.ptp(p[~dead])) > 1e-4, f"range {float(np.ptp(p[~dead])):.2e}")
check("alive mask returned for the loss", np.array_equal(out["alive"].cpu().numpy(), ~dead))

finish(summary())
