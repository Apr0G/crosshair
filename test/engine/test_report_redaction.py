"""
P10.2.4: report.json / report.md never contain a file-system path outside the output
directory, a URL, or (without --show-ids) a SteamID.

    .venv\\Scripts\\python.exe test\\engine\\test_report_redaction.py

A run is seeded with warnings that carry a signed URL, an absolute path outside the
output directory, one inside it, and a SteamID64; the written files must hold none of
them (the inside path becomes relative, the SteamID its p_<hex> label).
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from _harness import check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="crosshair_redact_"))
os.environ["CROSSHAIR_ENGINE_DB"] = str(TMP / "engine.db")

from _cf_states import RuleStub, good_win          # noqa: E402
from _synthetic import PLAYERS, make_raw_tables    # noqa: E402
from engine import analyze, config                 # noqa: E402
from engine.report import writer                   # noqa: E402
from engine.state.geometry import FlatGeometry     # noqa: E402
from engine.state.players import player_id_hash    # noqa: E402

STEAMID = "76561198012345678"
URL     = "https://demos.example.com/m1.dem.zst?X-Amz-Signature=SECRETSIG"
OUTSIDE = str(Path.home() / "Documents" / "private" / "demo.dem")
SPACED  = str(Path.home() / "Some User" / "Program Files" / "model dir" / "x.pt")   # MED-22
CFG     = config.load()


def run(name: str, show_ids: bool):
    out = TMP / name
    seeded = [
        {"code": "seed_url", "message": f"fetched from {URL}"},
        {"code": "seed_path", "message": f"read {OUTSIDE} and /home/faig/secret/file.txt"},
        {"code": "seed_inside", "message": f"wrote {out / 'charts' / 'x.png'}"},
        {"code": "seed_steamid", "message": f"player {STEAMID} disconnected"},
        {"code": "seed_spaced", "message": f"loaded {SPACED} (fallback)"},
    ]
    analyze.analyze_tables(make_raw_tables("trade"), match_key="syn_trade", tier="pro", geometry=FlatGeometry(),
                           predictor=RuleStub(good_win), cfg=CFG, out_dir=out, warnings=seeded,
                           show_ids=show_ids, write_db=False)
    return out, (out / "report.json").read_text(encoding="utf-8"), (out / "report.md").read_text(encoding="utf-8")


section("default (ids hashed)")
out, js, md = run("default", False)
rep = json.loads(js)
check("find_leaks: none in report.json", writer.find_leaks(rep, out) == [], str(writer.find_leaks(rep, out)[:3]))
import socket  # noqa: E402
for f in sorted(p for p in out.rglob("*") if p.suffix in (".json", ".md", ".txt")):
    text = f.read_text(encoding="utf-8")
    obj = json.loads(text) if f.suffix == ".json" else text
    leaks = writer.find_leaks(obj, out)
    check(f"{f.relative_to(out)}: find_leaks empty, no hostname (MED-07)",
          leaks == [] and socket.gethostname() not in text, str(leaks[:3]))
for text, name in ((js, "report.json"), (md, "report.md")):
    check(f"{name}: no URL", "http" not in text and "SECRETSIG" not in text)
    check(f"{name}: no outside path", OUTSIDE not in text and "Documents" not in text and "/home/faig" not in text)
    check(f"{name}: no out-dir absolute path", str(out) not in text and str(TMP) not in text)
    check(f"{name}: no SteamID64", STEAMID not in text and not re.search(r"7656119\d{10}", text))
    check(f"{name}: no repo path", str(Path(__file__).resolve().parents[2]) not in text)
msgs = {w["code"]: w["message"] for w in rep["warnings"]}
spaced_left = [k for k, w in (("user", "Some User"), ("program", "Program Files"), ("model", "model dir")) if w in js]
check("spaced path under home redacted whole (MED-22)", not spaced_left
      and msgs["seed_spaced"].startswith("loaded <path>"),
      f"segments left: {spaced_left}, starts with 'loaded <path>': {msgs['seed_spaced'].startswith('loaded <path>')}")
leaks = writer.find_leaks({"m": f"see {SPACED} now", "u": URL, "s": STEAMID})
check("find_leaks sees the spaced path, the URL and the id, and never echoes them (MED-23)",
      len(leaks) == 3 and all(("Some User" not in l and "SECRETSIG" not in l and STEAMID not in l) for l in leaks),
      str(leaks))
check("URL replaced by <url>", "<url>" in msgs["seed_url"])
check("outside paths replaced by <path>", msgs["seed_path"].count("<path>") == 2)
check("inside path made relative", "charts/x.png" in msgs["seed_inside"] and "<path>" not in msgs["seed_inside"])
check("SteamID replaced by its player label", writer.steamid_label(STEAMID) in msgs["seed_steamid"])
hashes = {player_id_hash(sid) for sid, _ in PLAYERS.values()}
check("actors are p_<hex> labels", all(re.fullmatch(r"p_[0-9a-f]{10}", a["actor"])
                                       for r in rep["rounds"] for a in r["actions"] if a.get("actor")))
check("engine id hashes never appear in the report", not any(str(h) in js for h in hashes))
check("no steamid / name fields by default", '"steamid"' not in js and '"name"' not in js)
check("chart file names use labels", all(p.name.startswith(("round_", "team_", "player_p_"))
                                         for p in (out / "charts").glob("*.png")))

section("--show-ids")
out2, js2, md2 = run("ids", True)
rep2 = json.loads(js2)
acts = [a for r in rep2["rounds"] for a in r["actions"] if a.get("actor")]
check("steamid fields present", acts and all(a.get("steamid") in {str(s) for s, _ in PLAYERS.values()} for a in acts))
check("players carry steamid", all(p.get("steamid") for p in rep2["players"]))
check("find_leaks with show_ids: no URL / path", writer.find_leaks(rep2, out2, show_ids=True) == [])
check("URL still redacted with show_ids", "http" not in js2)
check("a SteamID64 in free text is kept with show_ids", STEAMID in js2)
check("report still validates", writer.validate(rep2) == [])
# LOW-37: a name with '|' or a newline stays inside its cell of the markdown player table
rep3 = json.loads(js2)
rep3["players"][0]["name"] = "evil|name\nnext"
md3 = writer.markdown(rep3).splitlines()
head = next(l for l in md3 if l.startswith("| player |"))
row = [l for l in md3 if "evil" in l]
ncols = lambda line: len(re.findall(r"(?<!\\)\|", line))   # noqa: E731
check("player table: '|' escaped and newline folded in a --show-ids name (LOW-37)",
      len(row) == 1 and "evil\\|name next" in row[0] and ncols(row[0]) == ncols(head),
      f"rows {len(row)}, cols {[ncols(r) for r in row]} vs {ncols(head)}")

section("scrub_text units")
check("url", writer.scrub_text("see http://x.y/z?a=1 now") == "see <url> now")
check("windows path outside", writer.scrub_text(r"C:\secret\a.dem", TMP) == "<path>")
check("windows path inside", writer.scrub_text(str(TMP / "sub" / "f.png"), TMP) == "sub/f.png")
check("posix path", writer.scrub_text("at /var/log/x.txt", TMP) == "at <path>")
check("plain text untouched", writer.scrub_text("smoke to CT (41%), ct/t, n/a") == "smoke to CT (41%), ct/t, n/a")
check("steamid hashed", writer.scrub_text(STEAMID) == writer.steamid_label(STEAMID))
check("steamid kept with show_ids", writer.scrub_text(STEAMID, show_ids=True) == STEAMID)

sys.exit(summary())
