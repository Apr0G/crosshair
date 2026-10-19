"""
P10.2.1: engine/report/schema.json is a valid draft 2020-12 schema; a synthetic report
validates, also with every optional field absent; a report missing a required field,
with a wrong version, or with a bad enum fails; write_json refuses an invalid report
and turns NaN into null. (Golden-free: the golden report joins once the demo exists.)

    .venv\\Scripts\\python.exe test\\engine\\test_report_schema.py
"""
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="crosshair_schema_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

import jsonschema                                 # noqa: E402

from _cf_states import RuleStub, good_win         # noqa: E402
from _synthetic import make_raw_tables            # noqa: E402
from engine import analyze, config                # noqa: E402
from engine.report import writer                  # noqa: E402
from engine.state.geometry import FlatGeometry    # noqa: E402

SCHEMA = writer.schema()

section("schema itself")
try:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    ok = True
except jsonschema.SchemaError:
    ok = False
check("schema.json is a valid draft 2020-12 schema", ok)
check("schema pins report_schema_version 1", SCHEMA["properties"]["report_schema_version"] == {"const": 1})
check("writer.REPORT_SCHEMA_VERSION == 1", writer.REPORT_SCHEMA_VERSION == 1)

res = analyze.analyze_tables(make_raw_tables("flash"), match_key="syn_flash", tier="pro", geometry=FlatGeometry(),
                             predictor=RuleStub(good_win), cfg=config.load(), out_dir=TMP / "out", write_db=False)
rep = json.loads((TMP / "out" / "report.json").read_text(encoding="utf-8"))

section("synthetic report")
check("validates", writer.validate(rep) == [], "; ".join(writer.validate(rep)[:3]))


def resolve(s: dict) -> dict:
    while "$ref" in s:
        s = SCHEMA["$defs"][s["$ref"].rsplit("/", 1)[1]]
    return s


def minimal(obj, s: dict):
    """`obj` with every property the schema does not require removed, recursively."""
    s = resolve(s)
    if "oneOf" in s:
        if obj is None:
            return None
        s = next(resolve(b) for b in s["oneOf"] if resolve(b).get("type") == "object" or "properties" in resolve(b))
    if isinstance(obj, dict) and "properties" in s:
        req = set(s.get("required", []))
        return {k: minimal(v, s["properties"].get(k, {})) for k, v in obj.items() if k in req}
    if isinstance(obj, list) and "items" in s:
        return [minimal(v, s["items"]) for v in obj]
    return obj


mini = minimal(rep, SCHEMA)
check("every optional field absent: still validates", writer.validate(mini) == [], "; ".join(writer.validate(mini)[:3]))
check("the minimal report really dropped optional fields",
      "source_tier" not in mini["meta"] and "where" not in mini["rounds"][0]["actions"][0]
      and "stats" not in mini["rounds"][0])

section("invalid reports fail")


def broken(fn) -> list[str]:
    r = copy.deepcopy(rep)
    fn(r)
    return writer.validate(r)


cases = {
    "missing meta.config_hash":  lambda r: r["meta"].pop("config_hash"),
    "missing rounds[].wp_curve": lambda r: r["rounds"][0].pop("wp_curve"),
    "missing action V":          lambda r: r["rounds"][0]["actions"][0].pop("V"),
    "missing action Q (null is required, absence is not)": lambda r: r["rounds"][0]["actions"][0].pop("Q"),
    "missing players":           lambda r: r.pop("players"),
    "wrong schema version":      lambda r: r.__setitem__("report_schema_version", 2),
    "bad tier":                  lambda r: r["meta"].__setitem__("tier", "silver"),
    "bad verdict":               lambda r: r["rounds"][0]["actions"][0].__setitem__("verdict_decision", "great"),
    "raw steamid as actor":      lambda r: r["rounds"][0]["actions"][0].__setitem__("actor", "76561198000000000"),
    "W above 1":                 lambda r: r["rounds"][0]["wp_curve"][0].__setitem__("W", 1.5),
    "unknown top-level key":     lambda r: r.__setitem__("extra", 1),
}
for name, fn in cases.items():
    check(f"{name} -> invalid", broken(fn) != [])

section("write_json")
bad = copy.deepcopy(rep)
bad["meta"].pop("run_id")
try:
    writer.write_json(bad, TMP / "bad.json")
    refused = False
except writer.ReportSchemaError:
    refused = True
check("write_json refuses an invalid report", refused and not (TMP / "bad.json").exists())
nan = copy.deepcopy(rep)
nan["rounds"][0]["actions"][0]["Q"] = math.nan
writer.write_json(nan, TMP / "nan.json")
back = json.loads((TMP / "nan.json").read_text(encoding="utf-8"))
check("NaN written as null", back["rounds"][0]["actions"][0]["Q"] is None)

sys.exit(summary())
