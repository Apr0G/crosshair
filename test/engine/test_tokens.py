"""
Token embeddings (P6.2.1): output shapes; a None feature changes the output
differently from 0 (presence bit + learned missing vector); normalization buffers are
used; a dead player's token depends only on side / tier / alive.

    .venv\\Scripts\\python.exe test\\engine\\test_tokens.py [--cpu]
"""

import numpy as np

from _harness import check, require, section, summary
torch = require("torch", "needs the platform torch build (docs/ENGINE_ENV.md)")   # LOW-61
from _fake_states import fake_arrays, finish, test_device  # noqa: E402

from engine.data import dataset  # noqa: E402
from engine.data.dataset import PLAYER_LAYOUT, SNAPSHOT_KEYS, UTIL_LAYOUT  # noqa: E402
from engine.model import batching  # noqa: E402
from engine.model.tokens import NumericEmbed, TokenEmbedder  # noqa: E402

DEV = test_device()
print(f"device: {DEV}")
torch.manual_seed(0)
snaps, vocab, arrs, norm = fake_arrays(4, seed=1)
emb = TokenEmbedder(vocab.sizes(), 128).to(DEV).eval()
emb.set_norm(norm)


def batch_of(snapshots):
    a = dataset.tensorize(snapshots, vocab)
    return batching.to_device({k: a[k] for k in SNAPSHOT_KEYS}, DEV)


@torch.no_grad()
def tokens(snapshots):
    return emb(batch_of(snapshots))


section("shapes")
ctx, pl, ut = tokens(snaps[:7])
check("ctx [B,1,128]", tuple(ctx.shape) == (7, 1, 128))
check("players [B,10,128]", tuple(pl.shape) == (7, 10, 128))
check("utility [B,16,128]", tuple(ut.shape) == (7, 16, 128))
check("all finite", bool(torch.isfinite(ctx).all() and torch.isfinite(pl).all() and torch.isfinite(ut).all()))

section("None differs from 0")
s = next(x for x in snaps if any(p.alive and p.seconds_since_last_shot is not None for p in x.players))
j = next(k for k, p in enumerate(s.players) if p.alive and p.seconds_since_last_shot is not None)
a_none, a_zero = s.copy(), s.copy()
a_none.players[j].seconds_since_last_shot = None
a_zero.players[j].seconds_since_last_shot = 0.0
_, p_none, _ = tokens([a_none])
_, p_zero, _ = tokens([a_zero])
_, p_orig, _ = tokens([s])
d_nz = float((p_none[0, j] - p_zero[0, j]).abs().max())
check("player token: None != 0", d_nz > 1e-4, f"max |diff| {d_nz:.3g}")
check("player token: 0 != original value", float((p_zero[0, j] - p_orig[0, j]).abs().max()) > 1e-5)
others = [k for k in range(10) if k != j]
check("other players' tokens unaffected", float((p_none[0, others] - p_zero[0, others]).abs().max()) < 1e-6)

c_none, c_zero = s.copy(), s.copy()
c_none.ctx.equip_ct = None          # validate() would object; the embedder must still separate them
c_zero.ctx.equip_ct = 0.0
cn, _, _ = tokens([c_none])
cz, _, _ = tokens([c_zero])
check("ctx token: None != 0", float((cn - cz).abs().max()) > 1e-4)

ne = NumericEmbed(UTIL_LAYOUT, 16)
x0 = torch.zeros(1, len(UTIL_LAYOUT))
x1 = x0.clone()
x1[0, UTIL_LAYOUT.col("present_time_left_s")] = 1.0     # time_left 0.0, known
with torch.no_grad():
    nn_out, kn_out = ne(x0), ne(x1)
check("NumericEmbed: missing (presence 0) != known 0", float((nn_out - kn_out).abs().max()) > 1e-4)
with torch.no_grad():
    ne.missing.zero_()
    diff_lin = float((ne(x0) - ne(x1)).abs().max())
check("presence bit alone separates them (missing vector is a second, independent path)", diff_lin > 1e-5)

section("normalization buffers")
xs = torch.zeros(1, len(UTIL_LAYOUT))
xs[0, UTIL_LAYOUT.col("x")] = 1000.0
ne2 = NumericEmbed(UTIL_LAYOUT, 8)
with torch.no_grad():
    before = ne2(xs).clone()
    mean, std = np.zeros(len(UTIL_LAYOUT)), np.ones(len(UTIL_LAYOUT))
    mean[UTIL_LAYOUT.col("x")], std[UTIL_LAYOUT.col("x")] = 1000.0, 500.0
    ne2.set_norm(mean, std)
    after = ne2(xs)
check("set_norm z-scores continuous columns", float((before - after).abs().max()) > 1e-3)
check("norm buffers are in the state dict", "player_num.mean" in emb.state_dict() and "ctx_num.std" in emb.state_dict())

section("dead players")
sd = next(x for x in snaps if any(not p.alive for p in x.players))
k = next(i for i, p in enumerate(sd.players) if not p.alive)
e2 = sd.copy()
e2.players[k].x = 12345.0
e2.players[k].callout = "somewhere_else"
e2.players[k].kills_round = 3.0
e2.players[k].player_id = 77
_, pd0, _ = tokens([sd])
_, pd1, _ = tokens([e2])
check("dead player's token ignores position / callout / kills / id", float((pd0[0, k] - pd1[0, k]).abs().max()) < 1e-6)
e3 = sd.copy()
e3.players[k].tier = "faceit_10" if sd.players[k].tier == "pro" else "pro"
_, pd2, _ = tokens([e3])
check("dead player's token keeps tier", float((pd0[0, k] - pd2[0, k]).abs().max()) > 1e-6)
alive_col = PLAYER_LAYOUT.col("alive")
check("alive column index is first", alive_col == 0)

finish(summary())
