"""
Data QA report per split (P5.4): null %, ranges, histograms, outlier rules; thresholds
from `config.qa`; `--assert-qa` exits 1 on violation.

    from engine.data import qa
    res = qa.report("val", vocab=vocab)              # reads datasets/val/, writes eval/qa_val_<date>.md
    res = qa.report("synthetic", arrays=arrs)        # from in-memory arrays
    qa.violations(res, cfg)                          # list of threshold breaches (empty = ok)
    qa.assert_thresholds(res, cfg)                   # raises QAError on any breach

Reported, from the world-unit shard arrays (engine/data/dataset.py):
- per feature, overall and per map version: null % (presence bit 0 among present
  tokens; NaN for the pairwise table, off-diagonal), min / max / mean of known values,
  a 10-bin histogram;
- per tier: rounds and snapshots (actions by type only when the caller passes counts:
  the shards do not carry actions);
- UNK counts per categorical column (a val-only callout lands here, P5.3.2);
- outlier rules with counts and example rows: speed > `qa.outlier_max_speed` (alive
  players), health > `qa.outlier_max_health`, negative time left (round clock or a
  utility's time_left_s), more than `qa.outlier_max_alive` alive on a side.

Thresholds: `qa.max_null_pct.<family>` (player / utility / context / pairwise; null =
not checked) is the most any single feature of that family may be null;
`qa.max_outliers` (null = not checked) caps the total outlier count.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

from engine import paths
from engine.data.dataset import (CTX_LAYOUT, FEATURE_LAYOUT, PAIR_CHANNELS, PLAYER_LAYOUT, UNK,
                                 UTIL_LAYOUT, read_shards)

FAMILIES = {"player": ("player_num", PLAYER_LAYOUT, "player_mask"),
            "utility": ("util_num", UTIL_LAYOUT, "util_mask"),
            "context": ("ctx_num", CTX_LAYOUT, None)}
N_HIST = 10
N_EXAMPLES = 5


class QAError(AssertionError):
    """QA thresholds violated; `.violations` lists them."""

    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations))
        self.violations = violations


# ── stats ─────────────────────────────────────────────────────────────────────

def _stats(values: np.ndarray, n_total: int) -> dict:
    v = values[np.isfinite(values)]
    if n_total == 0:
        return {"n": 0, "null_pct": None, "min": None, "max": None, "mean": None, "hist": None}
    out = {"n": int(n_total), "null_pct": 100.0 * (1.0 - v.size / n_total)}
    if v.size:
        counts, edges = np.histogram(v, bins=N_HIST)
        out.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                   hist={"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]})
    else:
        out.update(min=None, max=None, mean=None, hist=None)
    return out


def feature_stats(arrays: dict, rows: np.ndarray | None = None) -> dict[str, dict]:
    """`<family>.<feature>` -> stats, over `rows` (all snapshots when None)."""
    sel = (lambda a: a) if rows is None else (lambda a: a[rows])
    out: dict[str, dict] = {}
    for fam, (block, lay, mask_key) in FAMILIES.items():
        x = sel(arrays[block])
        tok = sel(arrays[mask_key]) if mask_key else np.ones(x.shape[:1], bool)
        for c, (name, _, _) in enumerate(lay.values):
            v = x[..., c][tok].astype(np.float64)
            p = lay.presence_of[c]
            if p >= 0:
                v = np.where(x[..., p][tok] > 0.5, v, np.nan)
            out[f"{fam}.{name}"] = _stats(v, int(tok.sum()))
    pair = sel(arrays["pair"])
    off = ~np.eye(pair.shape[1], dtype=bool) if pair.ndim == 4 else None
    for k, ch in enumerate(PAIR_CHANNELS):
        v = pair[..., k][:, off] if len(pair) else np.zeros(0)
        out[f"pairwise.{ch}"] = _stats(v.reshape(-1).astype(np.float64), int(v.size))
    return out


def _col(arrays, block, lay, name):
    return arrays[block][..., lay.col(name)]


def outliers(arrays: dict, cfg) -> dict[str, dict]:
    q = cfg.qa
    pn, pm = arrays["player_num"], arrays["player_mask"]
    alive = (_col(arrays, "player_num", PLAYER_LAYOUT, "alive") > 0.5) & pm
    speed_ok = pn[..., PLAYER_LAYOUT.col("present_speed")] > 0.5
    rules = {
        "speed": (alive & speed_ok & (_col(arrays, "player_num", PLAYER_LAYOUT, "speed") > q.outlier_max_speed)).any(1),
        "health": (pm & (_col(arrays, "player_num", PLAYER_LAYOUT, "health") > q.outlier_max_health)).any(1),
        "negative_time_left": (_col(arrays, "ctx_num", CTX_LAYOUT, "time_remaining_s") < 0)
                              | (arrays["util_mask"]
                                 & (arrays["util_num"][..., UTIL_LAYOUT.col("present_time_left_s")] > 0.5)
                                 & (_col(arrays, "util_num", UTIL_LAYOUT, "time_left_s") < 0)).any(1),
        "alive_per_side": ((_col(arrays, "ctx_num", CTX_LAYOUT, "alive_ct") > q.outlier_max_alive)
                           | (_col(arrays, "ctx_num", CTX_LAYOUT, "alive_t") > q.outlier_max_alive)
                           | (alive[:, :5].sum(1) > q.outlier_max_alive)
                           | (alive[:, 5:].sum(1) > q.outlier_max_alive)),
    }
    out = {}
    for name, hit in rules.items():
        idx = np.flatnonzero(hit)
        out[name] = {"count": int(idx.size),
                     "examples": [{"round_key": str(arrays["round_key"][i]), "t_s": float(arrays["t_s"][i])}
                                  for i in idx[:N_EXAMPLES]]}
    return out


def unk_counts(arrays: dict) -> dict[str, int]:
    out = {}
    for block in ("player_cat", "util_cat", "ctx_cat"):
        mask = arrays.get({"player_cat": "player_mask", "util_cat": "util_mask"}.get(block, ""))
        for c, name in enumerate(FEATURE_LAYOUT[block]):
            v = arrays[block][..., c]
            hit = (v == UNK) if mask is None else ((v == UNK) & mask)
            out[f"{block}.{name}"] = int(hit.sum())
    return out


# ── report ────────────────────────────────────────────────────────────────────

def report(split: str, *, arrays: dict | None = None, base: Path | None = None, vocab=None, cfg=None,
           out_dir: Path | None = None, actions: dict | None = None, write: bool = True,
           date_str: str | None = None) -> dict:
    """Compute the QA result for a split and (by default) write the markdown report
    `<out_dir or eval>/qa_<split>_<date>.md`. `actions` = {tier: {type: count}}."""
    if cfg is None:
        from engine import config
        cfg = config.load()
    arrs = arrays if arrays is not None else read_shards(split, base)
    n = len(arrs["t_s"])
    tiers = {}
    for t in sorted(set(arrs["tier"].tolist())):
        m = arrs["tier"] == t
        tiers[t] = {"snapshots": int(m.sum()), "rounds": int(len(np.unique(arrs["round_key"][m]))),
                    "actions": (actions or {}).get(t)}
    mv_col = FEATURE_LAYOUT["ctx_cat"].index("map_version")
    inv = vocab.inverse("map_version") if vocab is not None else None
    per_map = {}
    for code in sorted(set(arrs["ctx_cat"][:, mv_col].tolist())):
        name = inv[code] if inv is not None and code < len(inv) else f"map#{code}"
        per_map[name] = feature_stats(arrs, np.flatnonzero(arrs["ctx_cat"][:, mv_col] == code))
    res = {"split": split, "n_snapshots": int(n), "n_rounds": int(len(np.unique(arrs["round_key"]))),
           "per_tier": tiers, "features": feature_stats(arrs), "per_map": per_map,
           "outliers": outliers(arrs, cfg), "unk": unk_counts(arrs), "path": None}
    if write:
        d = Path(out_dir) if out_dir is not None else paths.EVAL_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"qa_{split}_{date_str or date.today().isoformat()}.md"
        path.write_text(markdown(res, cfg), encoding="utf-8")
        res["path"] = str(path)
    return res


def violations(res: dict, cfg) -> list[str]:
    out = []
    lim = cfg.qa.max_null_pct
    for key, st in res["features"].items():
        fam = key.split(".", 1)[0]
        cap = getattr(lim, fam)
        if cap is not None and st["null_pct"] is not None and st["null_pct"] > cap:
            out.append(f"{key}: null {st['null_pct']:.2f}% > {cap}% ({fam} limit)")
    total = sum(o["count"] for o in res["outliers"].values())
    if cfg.qa.max_outliers is not None and total > cfg.qa.max_outliers:
        detail = ", ".join(f"{k}={v['count']}" for k, v in res["outliers"].items() if v["count"])
        out.append(f"outliers: {total} > {cfg.qa.max_outliers} ({detail})")
    return out


def assert_thresholds(res: dict, cfg) -> None:
    bad = violations(res, cfg)
    if bad:
        raise QAError(bad)


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def markdown(res: dict, cfg) -> str:
    L = [f"# QA report: {res['split']}", "",
         f"{res['n_snapshots']} snapshots, {res['n_rounds']} rounds.", "", "## Per tier", "",
         "| tier | rounds | snapshots | actions by type |", "|---|---:|---:|---|"]
    for t, d in res["per_tier"].items():
        L.append(f"| {t} | {d['rounds']} | {d['snapshots']} | {d['actions'] if d['actions'] else 'n/a (not in shards)'} |")
    L += ["", "## Outliers", "", "| rule | count | examples |", "|---|---:|---|"]
    for k, o in res["outliers"].items():
        ex = "; ".join(f"{e['round_key']} t={e['t_s']:.2f}" for e in o["examples"])
        L.append(f"| {k} | {o['count']} | {ex} |")
    L += ["", "## UNK categories (unseen in train)", "", "| column | UNK count |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in res["unk"].items()]
    bad = violations(res, cfg)
    L += ["", "## Thresholds", "", *(f"- VIOLATION: {b}" for b in bad)] if bad else ["", "## Thresholds", "", "- none violated (null thresholds may be unset)"]
    L += ["", "## Features (all maps)", "", "| feature | n | null % | min | max | mean | histogram (10 bins) |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for k, st in res["features"].items():
        h = " ".join(str(c) for c in st["hist"]["counts"]) if st["hist"] else ""
        L.append(f"| {k} | {st['n']} | {_f(st['null_pct'])} | {_f(st['min'])} | {_f(st['max'])} | {_f(st['mean'])} | {h} |")
    for mv, feats in res["per_map"].items():
        L += ["", f"## Features: {mv}", "", "| feature | null % | min | max | mean |", "|---|---:|---:|---:|---:|"]
        for k, st in feats.items():
            L.append(f"| {k} | {_f(st['null_pct'])} | {_f(st['min'])} | {_f(st['max'])} | {_f(st['mean'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    from engine import config
    from engine.data.dataset import Vocab
    ap = argparse.ArgumentParser(description="QA report for a dataset split (P5.4).")
    ap.add_argument("--split", required=True)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--assert-qa", action="store_true")
    a = ap.parse_args(argv)
    cfg = config.load()
    res = report(a.split, vocab=Vocab.load(a.vocab) if a.vocab else None, cfg=cfg)
    print(f"QA report -> {res['path']}")
    bad = violations(res, cfg)
    for b in bad:
        print(f"QA violation: {b}")
    return 1 if (a.assert_qa and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
