"""
Model registry (P12.1): `data/engine/models/<kind>_v<N>/manifest.json` lists what
exists; a row in the adoption table of `docs/ENGINE_MODELS.md` is the only thing that
makes a model adopted.

    from engine.evaluation import registry
    registry.list_models()                    # [{name, kind, created_at, adopted, eval, ...}]
    registry.is_adopted("wp_v3")              # True only with an "adopted" row
    registry.require_adopted("wp_v3")         # raises NotAdopted otherwise
    registry.adopt("wp_v3", {"logloss": .46, "brier": .15, "ece": .012, "baseline_logloss": .49},
                   "P6.7.2: beats LightGBM by 0.03 log loss, ECE 0.012")

Adoption table columns (ADOPTION_HEADER): date, model, kind, log loss, Brier, ECE,
baseline log loss, hazard AUC, decision rule satisfied, status. Status is one of
STATUSES; to retire a model, edit the status cell of its row in place (never add a
second row for the same model). `engine.analyze.adopted_models` reads the same table
through `adoption_rows` (model cell + status cell), so the two readers cannot
disagree. `adopt` still refuses rule text that names another model version or carries
a status word, so a row reads unambiguously to a human too (2026-10-19 audit, HIGH-02).

P6.7.2 is enforced for `wp` models: log loss at least LOGLOSS_MARGIN (0.01) below the
LightGBM baseline and ECE at most `calib.ece_target` (0.02). Other kinds record their
numbers; the decision rule text says why they were adopted.

Filled by: P12.1.
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

from engine import paths

# ── constants ─────────────────────────────────────────────────────────────────

DOC_PATH        = paths.ROOT / "docs" / "ENGINE_MODELS.md"
ADOPTION_HEADER = ("date", "model", "kind", "log loss", "Brier", "ECE", "baseline log loss", "hazard AUC",
                   "decision rule satisfied", "status")
STATUSES        = ("adopted", "retired", "superseded", "revoked", "rejected")
LOGLOSS_MARGIN  = 0.01            # P6.7.2: beat the baseline by >= 0.01 log loss
ECE_MAX         = 0.02            # P6.7.2 / calib.ece_target when no config is given
NAME_RE         = re.compile(r"^[a-z]+_v\d+$")
TOKEN_RE        = re.compile(r"\b[a-z]+_v\d+\b")
_NOT_ADOPTED_RE = re.compile(r"reject|retire|supersede|revoke", re.IGNORECASE)
NUMBER_KEYS     = ("logloss", "brier", "ece", "baseline_logloss", "hazard_auc")


class NotAdopted(RuntimeError):
    """The model has no "adopted" row in docs/ENGINE_MODELS.md."""


def kind_of(name: str) -> str:
    return name.rsplit("_v", 1)[0]


# ── adoption table ────────────────────────────────────────────────────────────

def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_bounds(lines: list[str]) -> tuple[int, int] | None:
    """(header line index, index after the last table row) of the adoption table."""
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and tuple(_cells(line)) == ADOPTION_HEADER:
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            return i, j
    return None


def adoption_rows(doc: Path | None = None) -> list[dict]:
    """Rows of the adoption table as dicts keyed by ADOPTION_HEADER."""
    doc = Path(doc) if doc is not None else DOC_PATH
    if not doc.is_file():
        return []
    lines = doc.read_text(encoding="utf-8").splitlines()
    b = _table_bounds(lines)
    if b is None:
        return []
    rows = []
    for line in lines[b[0] + 1:b[1]]:
        cells = _cells(line)
        if len(cells) != len(ADOPTION_HEADER) or set("".join(cells)) <= set("-: "):
            continue
        rows.append(dict(zip(ADOPTION_HEADER, cells)))
    return rows


def is_adopted(name: str, doc: Path | None = None) -> bool:
    return any(r["model"] == name and r["status"].lower() == "adopted" for r in adoption_rows(doc))


def require_adopted(name: str, allow_unadopted: bool = False, doc: Path | None = None) -> None:
    if not allow_unadopted and not is_adopted(name, doc):
        raise NotAdopted(f"model {name} is not adopted in docs/ENGINE_MODELS.md; "
                         f"adopt it (P6.7.2) or pass --allow-unadopted")


def rule_problems(name: str, numbers: dict, ece_max: float = ECE_MAX) -> list[str]:
    """Why `numbers` fail P6.7.2 for a wp model ([] = passes; other kinds always [])."""
    if kind_of(name) != "wp":
        return []
    need = [k for k in ("logloss", "brier", "ece", "baseline_logloss") if numbers.get(k) is None]
    if need:
        return [f"missing {need}"]
    out = []
    margin = float(numbers["baseline_logloss"]) - float(numbers["logloss"])
    if margin < LOGLOSS_MARGIN:
        out.append(f"log loss beats the baseline by {margin:.4f} < {LOGLOSS_MARGIN}")
    if float(numbers["ece"]) > ece_max:
        out.append(f"ECE {float(numbers['ece']):.4f} > {ece_max}")
    return out


def _cell(v) -> str:
    return "–" if v is None else f"{float(v):.4f}"


def adopt(name: str, numbers: dict, rule: str, *, doc: Path | None = None, when: str | None = None,
          models_dir: Path | None = None, require_dir: bool = True, cfg=None) -> str:
    """Append an "adopted" row; returns the row. Refuses a bad name, a missing model
    directory, a second row for the same model, rule text that `engine.analyze` would
    read as not adopted, and wp numbers that fail P6.7.2."""
    doc = Path(doc) if doc is not None else DOC_PATH
    if not NAME_RE.match(name or ""):
        raise ValueError(f"model name {name!r} must look like <kind>_v<N>")
    unknown = sorted(set(numbers) - set(NUMBER_KEYS))
    if unknown:
        raise ValueError(f"unknown number keys {unknown}; expected {NUMBER_KEYS}")
    if require_dir:
        mdir = Path(models_dir) if models_dir is not None else paths.MODELS_DIR
        if not (mdir / name / "manifest.json").is_file():
            raise ValueError(f"no manifest for {name} under the models directory")
    rule = (rule or "").strip()
    if not rule or "|" in rule or "\n" in rule:
        raise ValueError("rule must be one non-empty line without '|'")
    if _NOT_ADOPTED_RE.search(rule):
        raise ValueError("rule text must not contain reject / retire / supersede / revoke "
                         "(status words belong in the status cell)")
    others = sorted(set(TOKEN_RE.findall(rule)) - {name})
    if others:
        raise ValueError(f"rule text must not name another model version {others}; "
                         f"only the model cell says which model a row adopts")
    if any(r["model"] == name for r in adoption_rows(doc)):
        raise ValueError(f"{name} already has a row; change its status in place instead")
    ece_max = float(cfg.calib.ece_target) if cfg is not None else ECE_MAX
    bad = rule_problems(name, numbers, ece_max)
    if bad:
        raise ValueError(f"{name} does not satisfy P6.7.2: {'; '.join(bad)}")
    if not doc.is_file():
        raise FileNotFoundError(f"{doc.name} not found")
    lines = doc.read_text(encoding="utf-8").splitlines()
    b = _table_bounds(lines)
    if b is None:
        raise ValueError(f"{doc.name} has no adoption table (header {' | '.join(ADOPTION_HEADER)})")
    row = "| " + " | ".join([when or date.today().isoformat(), name, kind_of(name),
                             *(_cell(numbers.get(k)) for k in NUMBER_KEYS), rule, "adopted"]) + " |"
    lines.insert(b[1], row)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")     # repo file: keep LF
    return row


# ── model directories ─────────────────────────────────────────────────────────

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_models(models_dir: Path | None = None, doc: Path | None = None) -> list[dict]:
    """Every `<models_dir>/<name>/manifest.json` with its eval numbers and adoption."""
    from engine import manifest
    mdir = Path(models_dir) if models_dir is not None else paths.MODELS_DIR
    if not mdir.is_dir():
        return []
    adopted = {r["model"] for r in adoption_rows(doc) if r["status"].lower() == "adopted"}
    out = []
    for d in sorted(p for p in mdir.iterdir() if (p / manifest.MANIFEST_NAME).is_file()):
        rec = {"name": d.name, "kind": kind_of(d.name), "adopted": d.name in adopted,
               "created_at": None, "config_hash": None, "git_sha": None, "error": None, "eval": {}}
        try:
            m = manifest.read(d)
            rec.update(created_at=m.get("created_at"), config_hash=m.get("config_hash"), git_sha=m.get("git_sha"))
            extra = m.get("extra") or {}
            for k in ("eval", "metrics"):
                if isinstance(extra.get(k), dict):
                    rec["eval"].update(extra[k])
        except (manifest.ManifestSchemaError, ValueError, OSError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        cal = _read_json(d / "calibration.json")
        if isinstance(cal, dict):
            rec["eval"].update({k: cal.get(k) for k in ("T", "mode", "ece_val_single", "ece_val_group")})
        ev = _read_json(d / "eval.json")
        if isinstance(ev, dict):
            rec["eval"].update(ev)
        out.append(rec)
    return out


# ── command line ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Model registry (P12.1): list models or adopt one.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="model directories, eval numbers, adoption")
    p = sub.add_parser("adopt", help="append an adopted row to docs/ENGINE_MODELS.md")
    p.add_argument("name")
    p.add_argument("--rule", required=True, help="the decision rule satisfied, one line")
    for k in NUMBER_KEYS:
        p.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "adopt":
        from engine import config
        nums = {k: getattr(a, k) for k in NUMBER_KEYS if getattr(a, k) is not None}
        try:
            print(adopt(a.name, nums, a.rule, cfg=config.load()))
        except (ValueError, FileNotFoundError) as exc:
            print(f"refused: {exc}")
            return 1
        return 0
    models = list_models()
    if not models:
        print("no model directories yet (train one with `engine.cli train-wp`)")
        return 2
    for m in models:
        print(f"{m['name']}  adopted={'yes' if m['adopted'] else 'no'}  created={m['created_at']}  "
              f"eval={json.dumps(m['eval'], sort_keys=True, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
