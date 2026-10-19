"""
Analyst review loop (P11.2): pick rounds that cover the review categories, write one
case directory per round, keep Faig's judgements in `eval/review_log.md`, and turn
every "disagree" into a numbered item.

    from engine.evaluation import review
    picks = review.select_review_rounds(frame, n=20, seed=1337)       # [{match_key, round_num, categories}]
    dirs  = review.write_review_set(frame, paths.EVAL_DIR)            # eval/review_round_<i>/case.json
    review.init_review_log(paths.EVAL_DIR / "review_log.md", cases)   # template, never overwrites
    items = review.parse_review_log(paths.EVAL_DIR / "review_log.md") # one item per "disagree"

`frame` is an `actions_scored` frame, ideally passed through `anchors.enrich` (tick,
alive counts, team_class, round labels). Categories (P11.2.1) and what marks them:

    opening_kill           a `kill` in the round
    trade                  a synthetic `trade_setup` row, or a kill with a trade_setup enabler
    flash_assisted_kill    a kill with a flash enabler
    smoke_covered_plant    a plant_complete with a smoke_cover enabler
    clutch                 an action by a player alone (1) against >= 2 alive enemies
    save                   a `save` action
    double_kill_same_tick  two kills on one tick
    clock_decided          round_end_reason is a time-out (CLOCK_REASONS)
    eco_win                the winning side's team_buy is pistol/eco and the loser's is not
                           (pistol rounds 1 and 13 excluded)
    failed_defuse          a `defuse_abort`

Enabler kinds come from the `enabler_kinds` column (enriched) and from the attribution
`components` of enabler rows (`source = "enabler"`, `kind`, `event_id` = the enabled
kill / plant). Categories that need a missing column are simply never detected.

Selection is a seeded greedy set cover (most uncovered categories first, ties broken
by the seed), then filled to `n` with other rounds, rounds with any category first.
`exclude` skips rounds already reviewed (P11.2.4: 20 new rounds per cycle).

Review log format (one section per round; parse is strict so typos are caught):

    <!-- case index=1 match_key=m1 round_num=7 -->
    | rank | action | action_id | V | D | X | panel |
    | 1 | kill by ct p_1a2b3c4d ... | 0123456789abcdef | agree | disagree | unsure | todo |
    Comment: one sentence.

Cells take agree / disagree / unsure (the first word counts, so `disagree: too big`
works); `todo` or an empty cell = not reviewed yet.

Filled by: P11.2.1 - P11.2.3 (the tooling; the reviews are Faig's).
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import polars as pl

# ── constants ─────────────────────────────────────────────────────────────────

CATEGORIES = ("opening_kill", "trade", "flash_assisted_kill", "smoke_covered_plant", "clutch", "save",
              "double_kill_same_tick", "clock_decided", "eco_win", "failed_defuse")
CLOCK_REASONS   = ("target_saved", "time_ran_out", "targetsaved", "round_time_expired")
ECO_CLASSES     = ("pistol", "eco")
PISTOL_ROUNDS   = (1, 13)                # MR12 halves (ENGINE_PROGRESS: halves assume MR12)
KILL_TYPES      = ("kill", "team_kill")
ASPECTS         = ("V", "D", "X", "panel")
JUDGEMENTS      = ("agree", "disagree", "unsure", "todo")
TOP_K           = 3
DEFAULT_N       = 20
DEFAULT_SEED    = 1337
CASE_NAME       = "case.json"
CASE_VERSION    = 1
TARGET_MAX_DISAGREE = 2                  # P11.2.4: <= 2 disagreements out of 60 judgements
_CASE_RE = re.compile(r"<!--\s*case\s+index=(\d+)\s+match_key=(\S+)\s+round_num=(-?\d+)\s*-->")
CASE_FIELDS = ("action_id", "type", "actor_id_hash", "actor_side", "tick", "V", "V_direct", "V_enabler",
               "V_movement", "Q", "B", "D", "R", "X", "verdict_decision", "verdict_execution", "panel_n")


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clean(v):
    """JSON-safe: NaN -> None."""
    return None if isinstance(v, float) and math.isnan(v) else v


# ── category detection ────────────────────────────────────────────────────────

def enabler_kinds_by_event(frame: pl.DataFrame) -> dict[str, set[str]]:
    """action_id -> enabler kinds that helped it (enriched column + components)."""
    out: dict[str, set[str]] = {}
    if "enabler_kinds" in frame.columns:
        for aid, kinds in zip(frame["action_id"].to_list(), frame["enabler_kinds"].to_list()):
            if kinds:
                out.setdefault(aid, set()).update(kinds)
    if "components" in frame.columns:
        for comps in frame["components"].to_list():
            if not comps:
                continue
            try:
                items = json.loads(comps)
            except (TypeError, ValueError):
                continue
            for c in items if isinstance(items, list) else []:
                if isinstance(c, dict) and c.get("source") == "enabler" and c.get("event_id") and c.get("kind"):
                    out.setdefault(c["event_id"], set()).add(c["kind"])
    return out


def _alive(row: dict) -> tuple[float | None, float | None]:
    ct, t = _num(row.get("alive_ct_pre")), _num(row.get("alive_t_pre"))
    if row.get("actor_side") == "ct":
        return ct, t
    if row.get("actor_side") == "t":
        return t, ct
    return None, None


def _is_clutch(row: dict) -> bool:
    mine, theirs = _alive(row)
    return mine is not None and theirs is not None and mine == 1 and theirs >= 2


def round_categories(frame: pl.DataFrame) -> dict[tuple[str, int], set[str]]:
    """(match_key, round_num) -> categories present in that round."""
    enab = enabler_kinds_by_event(frame)
    rounds: dict[tuple[str, int], list[dict]] = {}
    for r in frame.iter_rows(named=True):
        rounds.setdefault((r["match_key"], r["round_num"]), []).append(r)
    out = {}
    for key, rows in rounds.items():
        cats = set()
        kills = [r for r in rows if r["type"] in KILL_TYPES]
        if any(r["type"] == "kill" for r in rows):
            cats.add("opening_kill")
        if any(r["type"] == "trade_setup" for r in rows) or any(
                "trade_setup" in enab.get(r["action_id"], ()) for r in kills):
            cats.add("trade")
        if any("flash" in enab.get(r["action_id"], ()) for r in kills):
            cats.add("flash_assisted_kill")
        if any(r["type"] == "plant_complete" and "smoke_cover" in enab.get(r["action_id"], ()) for r in rows):
            cats.add("smoke_covered_plant")
        if any(_is_clutch(r) for r in rows):
            cats.add("clutch")
        if any(r["type"] == "save" for r in rows):
            cats.add("save")
        ticks = [r.get("tick") for r in kills if r.get("tick") is not None]
        if len(ticks) != len(set(ticks)):
            cats.add("double_kill_same_tick")
        reason = next((r.get("round_end_reason") for r in rows if r.get("round_end_reason")), None)
        if reason is not None and str(reason).lower() in CLOCK_REASONS:
            cats.add("clock_decided")
        if _eco_win(key[1], rows):
            cats.add("eco_win")
        if any(r["type"] == "defuse_abort" for r in rows):
            cats.add("failed_defuse")
        out[key] = cats
    return out


def _eco_win(round_num: int, rows: list[dict]) -> bool:
    if round_num in PISTOL_ROUNDS:
        return False
    won_ct = next((r.get("round_won_ct") for r in rows if r.get("round_won_ct") is not None), None)
    if won_ct is None:
        return False
    winner, loser = ("ct", "t") if int(won_ct) == 1 else ("t", "ct")
    cls = {r.get("actor_side"): r.get("team_class") for r in rows if r["type"] == "team_buy"}
    if cls.get(winner) not in ECO_CLASSES:
        return False
    return cls.get(loser) is None or cls.get(loser) not in ECO_CLASSES


# ── P11.2.1 selection ─────────────────────────────────────────────────────────

def select_review_rounds(frame: pl.DataFrame, n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                         exclude=()) -> list[dict]:
    """`n` rounds covering every category present (greedy set cover), seeded."""
    rng = random.Random(seed)
    cats = round_categories(frame)
    skip = {(str(m), int(r)) for m, r in exclude}
    pool = sorted(k for k in cats if (str(k[0]), int(k[1])) not in skip)
    chosen: list[tuple] = []
    uncovered = set().union(*(cats[k] for k in pool)) if pool else set()
    while uncovered and len(chosen) < n:
        scores = {k: len(cats[k] & uncovered) for k in pool if k not in chosen}
        best = max(scores.values(), default=0)
        if best == 0:
            break
        pick = rng.choice(sorted(k for k, s in scores.items() if s == best))
        chosen.append(pick)
        uncovered -= cats[pick]
    rest = [k for k in pool if k not in chosen]
    rng.shuffle(rest)
    rest.sort(key=lambda k: 0 if cats[k] else 1)            # stable: shuffled order within groups
    chosen += rest[:max(0, n - len(chosen))]
    return [{"match_key": k[0], "round_num": int(k[1]), "categories": sorted(cats[k])} for k in chosen]


def covered(selection: list[dict]) -> set[str]:
    return set().union(*(set(s["categories"]) for s in selection)) if selection else set()


def missing_categories(selection: list[dict]) -> list[str]:
    """Categories no selected round covers (absent from the data, or n too small)."""
    got = covered(selection)
    return [c for c in CATEGORIES if c not in got]


# ── case directories ──────────────────────────────────────────────────────────

def top_actions(frame: pl.DataFrame, match_key: str, round_num: int, k: int = TOP_K) -> list[dict]:
    """The round's k actions with the largest |V| (ties: action_id)."""
    rows = [r for r in frame.filter((pl.col("match_key") == match_key) & (pl.col("round_num") == round_num))
            .iter_rows(named=True)]
    rows.sort(key=lambda r: (-abs(_num(r.get("V")) or 0.0), r["action_id"]))
    return [{"rank": i + 1, **{f: _clean(r.get(f)) for f in CASE_FIELDS}} for i, r in enumerate(rows[:k])]


def write_review_set(frame: pl.DataFrame, out_root: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED, *,
                     selection: list[dict] | None = None, exclude=(), overwrite: bool = False) -> list[Path]:
    """`out_root/review_round_<i>/case.json` for i = 1..n. Refuses to overwrite an
    existing case unless `overwrite` (a case may already carry a reviewed report)."""
    out_root = Path(out_root)
    sel = selection if selection is not None else select_review_rounds(frame, n, seed, exclude)
    dirs = []
    for i, s in enumerate(sel, start=1):
        d = out_root / f"review_round_{i}"
        if (d / CASE_NAME).exists() and not overwrite:
            raise FileExistsError(f"{d.name}/{CASE_NAME} exists; pass overwrite=True or use a new out_root")
        dirs.append(d)
    for i, (s, d) in enumerate(zip(sel, dirs), start=1):
        d.mkdir(parents=True, exist_ok=True)
        n_actions = frame.filter((pl.col("match_key") == s["match_key"])
                                 & (pl.col("round_num") == s["round_num"])).height
        case = {"schema_version": CASE_VERSION, "index": i, "match_key": s["match_key"],
                "round_num": s["round_num"], "categories": s["categories"], "n_actions": n_actions,
                "top_actions": top_actions(frame, s["match_key"], s["round_num"]),
                "report": "put this round's report.md and charts here (engine analyze on the match)"}
        (d / CASE_NAME).write_text(json.dumps(case, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return dirs


def read_cases(out_root: Path) -> list[dict]:
    """Every `review_round_<i>/case.json` under `out_root`, by index."""
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in Path(out_root).glob(f"review_round_*/{CASE_NAME}")]
    return sorted(cases, key=lambda c: c["index"])


def exclusions_from(roots) -> list[tuple[str, int]]:
    """(match_key, round_num) of every case under the given roots: what a later cycle
    must not select again (P11.2.4: 20 NEW rounds per cycle; MED-29, 2026-10-19)."""
    out: list[tuple[str, int]] = []
    for root in roots:
        out += [(str(c["match_key"]), int(c["round_num"])) for c in read_cases(Path(root))]
    return out


# ── P11.2.2 review log ────────────────────────────────────────────────────────

def _fmt(v) -> str:
    v = _num(v)
    return "–" if v is None else f"{v:+.3f}"


def _actor(h) -> str:
    return "team" if h is None else f"p_{int(h) & 0xFFFFFFFF:08x}"


def _describe(a: dict) -> str:
    text = (f"{a.get('type')} by {a.get('actor_side')} {_actor(a.get('actor_id_hash'))} "
            f"(V {_fmt(a.get('V'))}, D {_fmt(a.get('D'))}, X {_fmt(a.get('X'))}, panel n {a.get('panel_n')})")
    return text.replace("|", "/")


def render_review_log(cases: list[dict], cycle: int = 1) -> str:
    lines = [f"# Review log — cycle {cycle}", "",
             "For each round read `review_round_<i>/report.md` and the charts, then replace every `todo` "
             "with agree, disagree or unsure for the action's V, D, X and its panel, and write one sentence "
             "after `Comment:` (P11.2.2). Every disagree becomes a numbered item (P11.2.3). Review the "
             "callout images in the same session. Target: <= 2 disagreements in a cycle, minimum two cycles "
             "(P11.2.4).", "", "<!-- review-log v1 -->"]
    for c in cases:
        lines += ["", f"## Round {c['index']} — {c['match_key']} round {c['round_num']}",
                  f"<!-- case index={c['index']} match_key={c['match_key']} round_num={c['round_num']} -->",
                  f"Categories: {', '.join(c.get('categories') or []) or 'none'}", "",
                  "| rank | action | action_id | V | D | X | panel |",
                  "|---:|---|---|---|---|---|---|"]
        for a in c.get("top_actions") or []:
            lines.append(f"| {a['rank']} | {_describe(a)} | {a['action_id']} | todo | todo | todo | todo |")
        lines += ["", "Comment: "]
    return "\n".join(lines) + "\n"


def init_review_log(path: Path, cases: list[dict], cycle: int = 1, overwrite: bool = False) -> Path:
    """Write the review-log template. Never overwrites Faig's judgements unless asked."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path.name} exists (it may hold judgements); pass overwrite=True or a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review_log(cases, cycle), encoding="utf-8")
    return path


def review_judgements(path: Path) -> list[dict]:
    """Every action row of the log: case, rank, action_id, {aspect: judgement}, comment."""
    rows, case, comments = [], None, {}
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        m = _CASE_RE.search(line)
        if m:
            case = {"round_index": int(m.group(1)), "match_key": m.group(2), "round_num": int(m.group(3))}
            continue
        if case is None:
            continue
        s = line.strip()
        if s.startswith("Comment:"):
            comments[case["round_index"]] = s[len("Comment:"):].strip()
            continue
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 3 + len(ASPECTS) or not cells[0].isdigit():
            continue
        judged = {}
        for aspect, cell in zip(ASPECTS, cells[3:]):
            word = re.match(r"[a-z]*", cell.lower()).group(0) if cell else "todo"
            if word not in JUDGEMENTS:
                raise ValueError(f"{Path(path).name} line {ln}: {aspect} = {cell!r}; use agree, disagree or unsure")
            judged[aspect] = word
        rows.append({**case, "rank": int(cells[0]), "action": cells[1], "action_id": cells[2],
                     "judgements": judged, "line": ln})
    for r in rows:
        r["comment"] = comments.get(r["round_index"]) or None
    return rows


def parse_review_log(path: Path) -> list[dict]:
    """Every "disagree" as a numbered item (P11.2.3), in log order."""
    items = []
    for r in review_judgements(path):
        for aspect in ASPECTS:
            if r["judgements"][aspect] == "disagree":
                items.append({"item": len(items) + 1, "round_index": r["round_index"], "match_key": r["match_key"],
                              "round_num": r["round_num"], "rank": r["rank"], "action_id": r["action_id"],
                              "aspect": aspect, "comment": r["comment"], "line": r["line"]})
    return items


def cycle_summary(path: Path) -> dict:
    """Counts per judgement and whether the P11.2.4 target (<= 2 disagreements, all
    judged) is met. A judgement is one action row; it disagrees if any aspect does."""
    rows = review_judgements(path)
    flat = [j for r in rows for j in r["judgements"].values()]
    n_dis_rows = sum(1 for r in rows if "disagree" in r["judgements"].values())
    todo = flat.count("todo")
    return {"rounds": len({r["round_index"] for r in rows}), "judgements": len(rows),
            "cells": len(flat), **{k: flat.count(k) for k in JUDGEMENTS},
            "disagree_rows": n_dis_rows,
            "target_met": bool(rows) and todo == 0 and n_dis_rows <= TARGET_MAX_DISAGREE}


# ── command line ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    from engine import paths
    ap = argparse.ArgumentParser(description="Review loop tooling (P11.2).")
    ap.add_argument("--scored", type=Path, default=None, help="actions_scored parquet to pick rounds from")
    ap.add_argument("--actions", type=Path, default=None, help="matching P4 actions parquet (context columns)")
    ap.add_argument("--states", type=Path, default=None, help="matching states parquet (alive counts, labels)")
    ap.add_argument("--out-root", type=Path, default=paths.EVAL_DIR, help="where review_round_<i>/ go")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cycle", type=int, default=1)
    ap.add_argument("--exclude-from", type=Path, action="append", default=None,
                    help="root(s) of earlier cycles whose rounds must not be selected again "
                         "(default for --cycle > 1: --out-root itself)")
    ap.add_argument("--parse", type=Path, default=None, help="print the disagree items of a review log")
    a = ap.parse_args(argv)
    if a.parse is not None:
        if not a.parse.is_file():
            print(f"review: {a.parse.name} not found")
            return 2
        for it in parse_review_log(a.parse):
            print(f"#{it['item']}  round {it['round_index']} ({it['match_key']} r{it['round_num']})  "
                  f"action {it['rank']} {it['action_id']}  {it['aspect']}: {it['comment'] or ''}")
        print(json.dumps(cycle_summary(a.parse), sort_keys=True))
        return 0
    if a.scored is None or not a.scored.is_file():
        print("review: needs scored pro rounds (--scored actions_scored.parquet, ideally with --actions and "
              "--states). None exists until a model is trained and matches are analysed.")
        return 2
    from engine.evaluation.anchors import enrich
    frame = enrich(pl.read_parquet(a.scored),
                   pl.read_parquet(a.actions) if a.actions and a.actions.is_file() else None,
                   pl.read_parquet(a.states) if a.states and a.states.is_file() else None)
    roots = a.exclude_from if a.exclude_from is not None else ([a.out_root] if a.cycle > 1 else [])
    exclude = exclusions_from(roots)
    sel = select_review_rounds(frame, a.n, a.seed, exclude=exclude)
    try:
        dirs = write_review_set(frame, a.out_root, selection=sel)
        cases = [json.loads((d / CASE_NAME).read_text(encoding="utf-8")) for d in dirs]
        log = init_review_log(a.out_root / "review_log.md", cases, a.cycle)
    except FileExistsError as exc:
        print(f"review: {exc}")
        return 1
    print(f"{len(dirs)} review rounds, log {log}; categories missing: {missing_categories(sel) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
