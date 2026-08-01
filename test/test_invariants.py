"""
Schema and invariant checks — stdlib only, no numpy/pandas/lightgbm, no trained model.

Run from repo root:
    .venv/bin/python test/test_invariants.py

These are the cheap mechanical checks for the rules in CLAUDE.md that previously had
no test at all. They read source text rather than importing the modules, so they work
even when the venv has no third-party packages installed.

Covers:
  P8  feature parity   — FEATURES ⊆ round_states columns ⊆ insert_round_states params
  P9  schema integrity — every column a writer/reader references exists in _init_schema
  P4  source interface — scraper and faceit_api expose the same two functions
  P1  dedup            — both id forms (<mid> and <mid>_m1) are checked in both sources
"""
import ast
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
DB   = ROOT / "data" / "crosshair.db"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"

_results: list[tuple[bool, str]] = []


def check(desc: str, ok: bool, detail: str = ""):
    _results.append((bool(ok), desc))
    print(f"  {PASS if ok else FAIL}  {desc}" + (f"  ({detail})" if detail else ""))


def skip(desc: str, why: str):
    print(f"  {SKIP}  {desc}  (skipped — {why})")


# ── source parsing ────────────────────────────────────────────────────────────

def _src(name: str) -> str:
    return (SRC / name).read_text()


def create_table_columns(sql_text: str, table: str) -> set[str]:
    """Column names from a CREATE TABLE ... ( ... ) block."""
    m = re.search(rf"CREATE TABLE IF NOT EXISTS\s+{table}\s*\((.*?)\);", sql_text, re.S)
    if not m:
        return set()
    cols = set()
    for line in m.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "--")):
            continue
        cols.add(line.split()[0])
    return cols


def insert_columns(sql_text: str, table: str) -> set[str]:
    """Column names from an INSERT INTO <table> (...) VALUES block."""
    m = re.search(rf"INSERT INTO\s+{table}\s*\((.*?)\)\s*VALUES", sql_text, re.S)
    if not m:
        return set()
    return {c.strip() for c in m.group(1).replace("\n", " ").split(",") if c.strip()}


def literal_list(src_text: str, name: str) -> list:
    """Value of a module-level list literal, without importing the module."""
    tree = ast.parse(src_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    return []


def function_names(src_text: str) -> set[str]:
    return {n.name for n in ast.parse(src_text).body if isinstance(n, ast.FunctionDef)}


# ── checks ────────────────────────────────────────────────────────────────────

def main() -> int:
    db_src   = _src("db.py")
    wp_src   = _src("win_probability.py")
    ss_src   = _src("state_sampler.py")

    schema_states = create_table_columns(db_src, "round_states")
    schema_events = create_table_columns(db_src, "events")
    ins_states    = insert_columns(db_src, "round_states")
    ins_events    = insert_columns(db_src, "events")
    features      = set(literal_list(wp_src, "FEATURES"))

    print("\n── P9: _init_schema covers what the writers insert ───────────────────────")
    check("round_states: every inserted column exists in the schema",
          ins_states <= schema_states,
          f"missing from schema: {sorted(ins_states - schema_states)}")
    check("events: every inserted column exists in the schema",
          ins_events <= schema_events,
          f"missing from schema: {sorted(ins_events - schema_events)}")

    print("\n── P9: _init_schema covers what the readers/updaters reference ───────────")
    # score_impact UPDATEs these; main.py status/top/round SELECT them.
    scored = {"p_before", "p_after", "impact"}
    check("events schema declares p_before / p_after / impact",
          scored <= schema_events,
          f"missing: {sorted(scored - schema_events)}")

    print("\n── P8: feature parity ───────────────────────────────────────────────────")
    check("every FEATURE exists as a round_states column",
          features <= schema_states,
          f"missing: {sorted(features - schema_states)}")
    check("every FEATURE is actually written by insert_round_states",
          features <= ins_states,
          f"missing: {sorted(features - ins_states)}")

    # The sampler builds its dict literal inline; check the keys it emits.
    emitted = set(re.findall(r'"([a-z0-9_]+)":\s', ss_src.split("states.append({")[-1]))
    if emitted:
        check("every inserted round_states column is emitted by the sampler",
              (ins_states - {"map"}) <= (emitted | {"map"}),
              f"never emitted: {sorted(ins_states - emitted - {'map'})}")
    else:
        skip("sampler emits every inserted column", "could not locate the state dict")

    print("\n── P4 / P1: the two ingestion sources agree ─────────────────────────────")
    sc_src = _src("scraper.py")
    api_src = _src("faceit_api.py")
    required = {"iter_unprocessed_demos", "download_demo"}
    check("scraper exposes the source interface",   required <= function_names(sc_src),
          f"missing: {sorted(required - function_names(sc_src))}")
    check("faceit_api exposes the source interface", required <= function_names(api_src),
          f"missing: {sorted(required - function_names(api_src))}")
    for label, text in (("scraper", sc_src), ("faceit_api", api_src)):
        check(f"{label} dedups the per-map id form",
              "is_processed(map_match_id)" in text)

    print("\n── P10: demo-URL host allowlist ─────────────────────────────────────────")
    # Pure-stdlib check of the SSRF gate. A bare endswith("faceit.com") also accepts
    # "evilfaceit.com"; the leading dot is what makes it a domain boundary.
    ns: dict = {}
    tree = ast.parse(sc_src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("_DEMO_HOSTS", "_DEMO_PATH_EXTS")
                for t in node.targets):
            exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_is_demo_url":
            ns["urlsplit"] = urlsplit
            exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
    is_demo = ns.get("_is_demo_url")
    if not is_demo:
        skip("demo-URL host allowlist", "could not extract _is_demo_url")
    else:
        cases = [
            ("https://demos.faceit-cdn.net/cs2/x.dem.zst", True),
            ("https://www.faceit.com/d/x.dem",             True),
            ("https://faceit.com/d/x.dem",                 True),
            ("https://evilfaceit.com/x.dem",               False),
            ("https://notfaceit.com/x.dem",                False),
            ("https://xfaceit-cdn.net/x.dem",              False),
            ("https://faceit.com.evil.io/x.dem",           False),
            ("https://evil.io/x.dem?a=faceit.com",         False),
            ("https://demos.faceit-cdn.net/x.exe",         False),
            ("file:///etc/passwd.dem",                     False),
        ]
        wrong = [u for u, want in cases if is_demo(u) != want]
        check("host allowlist rejects look-alike domains and bad schemes",
              not wrong, f"wrong verdict for: {wrong}")

    print("\n── P10: credential redaction ────────────────────────────────────────────")
    # urllib3 puts "path?query" — not the full URL — into connection-level errors, so a
    # literal replace of the known URL never matches. Every print/exception path that
    # can carry a signed FACEIT URL must survive these.
    pl_src = _src("pipeline.py")
    ns2: dict = {"urlsplit": urlsplit, "urlunsplit": urlunsplit, "re": re}
    tree2 = ast.parse(pl_src)
    for node in tree2.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_QUERY_RE" for t in node.targets):
            exec(compile(ast.Module([node], []), "<x>", "exec"), ns2)
        if isinstance(node, ast.FunctionDef) and node.name in ("redact_url", "redact_text"):
            exec(compile(ast.Module([node], []), "<x>", "exec"), ns2)
    rt, ru = ns2.get("redact_text"), ns2.get("redact_url")
    if not (rt and ru):
        skip("credential redaction", "could not extract redact_text/redact_url")
    else:
        SECRET = "SUPERSECRETSIGNATURE123"
        signed = f"https://demos-eu.faceit-cdn.net/cs2/m.dem.zst?sig={SECRET}&exp=9"
        urllib3_msg = ("HTTPSConnectionPool(host='demos-eu.faceit-cdn.net', port=443): "
                       f"Max retries exceeded with url: /cs2/m.dem.zst?sig={SECRET}&exp=9")
        leaks = []
        if SECRET in ru(signed):                       leaks.append("redact_url")
        if SECRET in rt(urllib3_msg, signed):          leaks.append("redact_text+signed")
        # The API path only ever hands pipeline the UNSIGNED resource url.
        if SECRET in rt(urllib3_msg, "https://open.faceit.com/d/v2/abc"):
            leaks.append("redact_text+unsigned")
        if SECRET in rt(urllib3_msg):                  leaks.append("redact_text+nourl")
        check("no signed-URL query survives redaction", not leaks, f"leaked via: {leaks}")

    print("\n── Live database ────────────────────────────────────────────────────────")
    if not DB.exists():
        skip("live DB matches the current schema", "no data/crosshair.db")
    else:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            live_states = {r[1] for r in con.execute("PRAGMA table_info(round_states)")}
            live_events = {r[1] for r in con.execute("PRAGMA table_info(events)")}
        finally:
            con.close()
        check("live round_states has every column insert_round_states writes",
              ins_states <= live_states,
              f"missing: {sorted(ins_states - live_states)}")
        check("live events has every column score_impact writes",
              scored <= live_events,
              f"missing: {sorted(scored - live_events)}")
        check("live round_states has every FEATURE",
              features <= live_states,
              f"missing: {sorted(features - live_states)}")

    passed = sum(1 for ok, _ in _results if ok)
    failed = [d for ok, d in _results if not ok]
    print(f"\n{'─'*60}")
    print(f"  {passed}/{len(_results)} passed")
    if failed:
        print("  Failed:")
        for d in failed:
            print(f"    - {d}")
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
