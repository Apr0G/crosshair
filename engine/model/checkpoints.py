"""
Checkpoint save/load with a manifest per model directory (P6.3.4).

A model directory (`data/engine/models/wp_v<N>/`) holds:

    best.pt  last.pt  [final.pt]     checkpoints (final = after the pro fine-tune, P6.3.2)
    vocab.json  norm.json            the dataset vocab and train normalization stats
    config.yaml                      the engine config the run used
    curves.csv  manifest.json        per-epoch curves; run manifest (engine.manifest)
    [calibration.json]               written by calibrate.py (P6.5)

A checkpoint is a plain dict (loads with `torch.load(weights_only=True)`):
`model_args` (arch, vocab sizes, ablation flags, layout hash), `model` (state dict),
optional `optimizer` / `scheduler`, and `state` (epoch, step, best metric, ...).
Writes are atomic (temp file then replace), so a crash never leaves half a checkpoint.
"""
import hashlib
import json
from pathlib import Path

import torch

from engine.data.dataset import NormStats, Vocab, layout_hash
from engine.model.wp_model import WPModel

LOAD_ORDER = ("final.pt", "best.pt", "last.pt")


def model_args(model: WPModel) -> dict:
    return {"arch": dict(model.arch), "vocab_sizes": dict(model.vocab_sizes),
            "flags": dict(model.flags), "layout_hash": layout_hash()}


def save(path: Path, model: WPModel, *, optimizer=None, scheduler=None, state: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_args": model_args(model),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "state": dict(state or {})}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    return path


def load(path: Path, device="cpu") -> dict:
    return torch.load(Path(path), map_location=device, weights_only=True)


def build_model(ckpt: dict, device="cpu") -> WPModel:
    a = ckpt["model_args"]
    if a.get("layout_hash") != layout_hash():
        raise ValueError(f"checkpoint layout {a.get('layout_hash')} != code layout {layout_hash()}")
    model = WPModel(a["arch"], a["vocab_sizes"], **a["flags"])
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def state_hash(state_dict: dict) -> str:
    """sha1 over every tensor of a state dict, for "which checkpoint was loaded" checks."""
    h = hashlib.sha1()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_model_files(run_dir: Path, vocab: Vocab, norm: NormStats, cfg=None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(run_dir / "vocab.json")
    norm.save(run_dir / "norm.json")
    if cfg is not None:
        import yaml
        from engine import config
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config.as_dict(cfg), sort_keys=True), encoding="utf-8")


def find_checkpoint(run_dir: Path, name: str | None = None) -> Path:
    run_dir = Path(run_dir)
    if name is not None:
        return run_dir / name
    for n in LOAD_ORDER:
        if (run_dir / n).is_file():
            return run_dir / n
    raise FileNotFoundError(f"no checkpoint ({', '.join(LOAD_ORDER)}) in {run_dir}")


def load_model_dir(run_dir: Path, device="cpu", name: str | None = None):
    """(model in eval mode, vocab, norm, calibration dict or None, checkpoint path)."""
    run_dir = Path(run_dir)
    path    = find_checkpoint(run_dir, name)
    model   = build_model(load(path, device), device).eval()
    vocab   = Vocab.load(run_dir / "vocab.json")
    norm    = NormStats.load(run_dir / "norm.json")
    cal_p   = run_dir / "calibration.json"
    cal     = json.loads(cal_p.read_text(encoding="utf-8")) if cal_p.is_file() else None
    if cal is not None and cal.get("state_hash"):          # fitted for THIS checkpoint? (MED-20)
        have = state_hash(model.state_dict())
        if have != cal["state_hash"]:
            raise ValueError(f"{run_dir.name}/calibration.json was fitted for checkpoint "
                             f"{cal.get('checkpoint')} ({cal['state_hash'][:8]}), but {path.name} ({have[:8]}) "
                             f"was loaded; re-run calibrate for this checkpoint")
    return model, vocab, norm, cal, path


def next_version_dir(parent: Path, prefix: str) -> Path:
    parent = Path(parent)
    have = [int(p.name[len(prefix):]) for p in parent.glob(f"{prefix}*") if p.name[len(prefix):].isdigit()]
    return parent / f"{prefix}{max(have, default=0) + 1}"
