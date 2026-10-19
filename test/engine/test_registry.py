"""
P12.1 model registry: the adoption table in docs/ENGINE_MODELS.md parses (and has no
rows yet), `adopt` appends a row and enforces P6.7.2 for wp models, an unadopted
model is refused, retiring in place works, `list_models` reads manifests and eval
numbers, and the loose reader in `engine.analyze` agrees on every case.

    .venv\\Scripts\\python.exe test\\engine\\test_registry.py
"""
import json
import re
import shutil
import tempfile
from pathlib import Path

from _harness import ROOT, check, section, skip, summary
from engine.evaluation import registry as rg

DOC = ROOT / "docs" / "ENGINE_MODELS.md"

try:
    from engine.analyze import adopted_models          # owned by the P10 agent; compared when importable
except Exception as exc:                               # an unfinished analyze must not fail this test
    adopted_models = None
    _why = f"{type(exc).__name__}: {exc}"


def agrees(doc: Path, names: list[str]) -> None:
    if adopted_models is None:
        skip("engine.analyze.adopted_models agrees", _why)
        return
    loose = adopted_models(doc)
    check(f"engine.analyze agrees on {names}", all((n in loose) == rg.is_adopted(n, doc) for n in names),
          f"loose={sorted(loose)}")


def model_dir(root: Path, name: str, cal: dict | None = None, extra: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"schema_version": 1, "kind": "train_wp", "created_at": "2026-10-19",
                                                 "config_hash": "abc", "git_sha": None, "extra": extra or {}}),
                                     encoding="utf-8")
    if cal is not None:
        (d / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    return d


GOOD = {"logloss": 0.460, "brier": 0.150, "ece": 0.012, "baseline_logloss": 0.490, "hazard_auc": 0.81}

# ── the real doc ──────────────────────────────────────────────────────────────
section("docs/ENGINE_MODELS.md")
text = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
check("doc exists with the adoption table header", rg._table_bounds(text.splitlines()) is not None)
check("no adoption rows yet", rg.adoption_rows(DOC) == [])
check("no model-version token anywhere (engine.analyze would adopt it)",
      re.findall(r"\b[a-z]+_v\d+\b", text) == [], f"{re.findall(r'[a-z]+_v[0-9]+', text)}")
check("P6.7.2 thresholds written down", "0.01" in text and "0.02" in text and "0.80" in text and "0.75" in text)

# ── adopt / refuse ────────────────────────────────────────────────────────────
section("adopt and refuse")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    doc = td / "ENGINE_MODELS.md"
    shutil.copyfile(DOC, doc)
    models = td / "models"
    model_dir(models, "wp_v7", cal={"T": 1.3, "mode": "single", "ece_val_single": 0.011, "ece_val_group": 0.012},
              extra={"eval": {"logloss_test": 0.46}})
    model_dir(models, "wp_v8")
    model_dir(models, "q_v2")

    check("unadopted model: is_adopted False", not rg.is_adopted("wp_v7", doc))
    try:
        rg.require_adopted("wp_v7", doc=doc)
        check("unadopted model refused (NotAdopted)", False)
    except rg.NotAdopted as exc:
        check("unadopted model refused (NotAdopted)", "allow-unadopted" in str(exc))
    rg.require_adopted("wp_v7", allow_unadopted=True, doc=doc)
    check("--allow-unadopted lets it through", True)
    agrees(doc, ["wp_v7"])

    row = rg.adopt("wp_v7", GOOD, "P6.7.2: beats LightGBM by 0.03, ECE 0.012", doc=doc, when="2026-11-01",
                   models_dir=models)
    rows = rg.adoption_rows(doc)
    check("adopt appends one row", len(rows) == 1 and rows[0]["model"] == "wp_v7" and rows[0]["status"] == "adopted")
    check("row carries date, kind and 4-decimal numbers",
          rows[0]["date"] == "2026-11-01" and rows[0]["kind"] == "wp" and rows[0]["log loss"] == "0.4600"
          and rows[0]["ECE"] == "0.0120" and rows[0]["hazard AUC"] == "0.8100", row)
    check("is_adopted True after adopt", rg.is_adopted("wp_v7", doc))
    check("doc keeps LF line endings", b"\r\n" not in doc.read_bytes())
    check("row inserted inside the table, prose after it kept",
          doc.read_text(encoding="utf-8").rstrip().endswith("adopted.") and rg._table_bounds(
              doc.read_text(encoding="utf-8").splitlines()) is not None)
    agrees(doc, ["wp_v7", "wp_v8"])

    def refused(desc: str, fn) -> None:
        before = doc.read_text(encoding="utf-8")
        try:
            fn()
            check(desc, False, "not refused")
        except (ValueError, FileNotFoundError):
            check(desc, doc.read_text(encoding="utf-8") == before)

    refused("second row for the same model refused", lambda: rg.adopt("wp_v7", GOOD, "again", doc=doc,
                                                                      models_dir=models))
    refused("wp beating the baseline by only 0.005 refused (P6.7.2)",
            lambda: rg.adopt("wp_v8", {**GOOD, "logloss": 0.485}, "close", doc=doc, models_dir=models))
    refused("wp with ECE 0.03 refused (P6.7.2)",
            lambda: rg.adopt("wp_v8", {**GOOD, "ece": 0.03}, "ece", doc=doc, models_dir=models))
    refused("wp without a baseline number refused",
            lambda: rg.adopt("wp_v8", {"logloss": 0.4, "brier": 0.1, "ece": 0.01}, "x", doc=doc, models_dir=models))
    refused("rule text with 'rejected' refused (analyze would skip the row)",
            lambda: rg.adopt("wp_v8", GOOD, "not rejected by anyone", doc=doc, models_dir=models))
    refused("rule text with '|' refused", lambda: rg.adopt("wp_v8", GOOD, "a | b", doc=doc, models_dir=models))
    refused("model without a manifest refused", lambda: rg.adopt("wp_v9", GOOD, "ok", doc=doc, models_dir=models))
    refused("bad name refused", lambda: rg.adopt("WP-7", GOOD, "ok", doc=doc, models_dir=models))
    refused("unknown number key refused", lambda: rg.adopt("wp_v8", {**GOOD, "auc": 0.9}, "ok", doc=doc,
                                                           models_dir=models))
    check("cfg ece_target is honoured", rg.rule_problems("wp_v8", {**GOOD, "ece": 0.025}, ece_max=0.03) == [])

    rg.adopt("q_v2", {"logloss": None}, "C2 identity 7e-18, panel coverage 97%", doc=doc, models_dir=models)
    check("q model adopted with its rule text (no numeric rule)", rg.is_adopted("q_v2", doc))
    check("second adopt appended after the first row",
          [r["model"] for r in rg.adoption_rows(doc)] == ["wp_v7", "q_v2"])

    section("retire in place")
    doc.write_text(doc.read_text(encoding="utf-8").replace("| adopted |\n", "| retired |\n", 1), encoding="utf-8")
    check("status edited to retired -> not adopted", not rg.is_adopted("wp_v7", doc) and rg.is_adopted("q_v2", doc))
    agrees(doc, ["wp_v7", "q_v2"])
    try:
        rg.require_adopted("wp_v7", doc=doc)
        check("retired model refused", False)
    except rg.NotAdopted:
        check("retired model refused", True)

    section("list_models")
    lm = {m["name"]: m for m in rg.list_models(models, doc)}
    check("three model directories listed", sorted(lm) == ["q_v2", "wp_v7", "wp_v8"])
    check("adoption flags", lm["q_v2"]["adopted"] and not lm["wp_v7"]["adopted"] and not lm["wp_v8"]["adopted"])
    check("eval numbers from calibration.json and manifest extra",
          lm["wp_v7"]["eval"].get("T") == 1.3 and lm["wp_v7"]["eval"].get("logloss_test") == 0.46
          and lm["wp_v8"]["eval"] == {})
    check("created_at / kind read", lm["wp_v7"]["created_at"] == "2026-10-19" and lm["q_v2"]["kind"] == "q")
    (models / "wp_v8" / "manifest.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    check("unknown manifest schema -> error recorded, still listed",
          rg.list_models(models, doc)[2]["error"] is not None)
    check("missing models dir -> []", rg.list_models(td / "nope", doc) == [])

check("missing doc -> no rows, nothing adopted",
      rg.adoption_rows(ROOT / "docs" / "no_such_file.md") == [] and not rg.is_adopted("wp_v1", ROOT / "x.md"))

raise SystemExit(summary())
