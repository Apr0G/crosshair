"""
GPU benchmark (P0.1.4): a 3-layer pre-norm transformer encoder on batch 512 x
sequence 24 x width 128, 200 training iterations (forward + backward + AdamW step) in
fp32 and in bf16 autocast. Attention is pinned to the math SDPA backend (the fused
ROCm kernels are experimental on this GPU and stay off).

    .venv\\Scripts\\python.exe bench\\bench_gpu.py [--iters 200] [--cpu-iters 10]

Writes `data/engine/bench/gpu_<date>_<hhmmss>_<cuda|cpu>.json` (the device that was
timed; a second run the same day never overwrites the first). Without a GPU it times
the CPU only and says so. `--cpu-iters 0` skips the CPU reference.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch                                         # noqa: E402
import torch.nn as nn                                # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel   # noqa: E402

from _bench import finish, result_name, write        # noqa: E402
from engine import paths                             # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

B, S, D, LAYERS, HEADS = 512, 24, 128, 3, 4


def make_model(device: str) -> nn.Module:
    layer = nn.TransformerEncoderLayer(D, HEADS, dim_feedforward=4 * D, batch_first=True,
                                       norm_first=True, dropout=0.0)
    return nn.Sequential(nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False),
                         nn.Linear(D, 1)).to(device)


def run(device: str, iters: int, bf16: bool) -> dict:
    torch.manual_seed(0)
    model = make_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(B, S, D, device=device)
    y = torch.rand(B, S, 1, device=device)

    def step() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(5):                            # warm-up (kernel selection / JIT)
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            loss = step()
        if device == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
    res = {"device": device, "dtype": "bf16" if bf16 else "fp32", "iters": iters, "seconds": round(secs, 4),
           "ms_per_iter": round(1000 * secs / iters, 3), "samples_per_s": round(B * iters / secs, 1),
           "final_loss": round(float(loss), 5), "finite": bool(torch.isfinite(loss))}
    print(f"{device:4s} {res['dtype']}: {res['ms_per_iter']:8.2f} ms/iter  {res['samples_per_s']:10.1f} samples/s  "
          f"loss {res['final_loss']:.4f}")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cpu-iters", type=int, default=10)
    a = ap.parse_args(argv)
    results, stages = [], {}
    gpu = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if gpu else None
    print(f"torch {torch.__version__}, GPU: {name}, model {LAYERS}x transformer, batch {B} x seq {S} x d {D}")
    if gpu:
        for bf16 in (False, True):
            if bf16 and not torch.cuda.is_bf16_supported():
                print("bf16 not supported on this device; skipped")
                continue
            r = run("cuda", a.iters, bf16)
            results.append(r)
            stages[f"cuda_{r['dtype']}"] = r["seconds"]
    else:
        print("no GPU: timing the CPU only")
    if a.cpu_iters > 0:
        r = run("cpu", a.cpu_iters, False)
        results.append(r)
        stages["cpu_fp32"] = r["seconds"]
    out = write(paths.BENCH_DIR / result_name("gpu", "cuda" if gpu else "cpu"), stages,
                extra={"torch": torch.__version__, "gpu": name, "shape": [B, S, D], "layers": LAYERS,
                       "heads": HEADS, "sdpa_backend": "math", "results": results})
    print(f"-> {out}")
    return 0 if all(r["finite"] for r in results) else 1


if __name__ == "__main__":
    finish(main())          # os._exit after GPU use (HIP teardown hang), see _bench.finish
