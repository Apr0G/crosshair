"""
Writes report.json (schema-validated), report.md, charts/ with redaction (P10.2).

    from engine.report import writer
    rep = writer.redact(report, out_dir, show_ids=False)     # P10.2.4
    writer.validate(rep)                                     # [] or the problems
    writer.write_json(rep, out_dir / "report.json")          # validates, then writes
    writer.write_markdown(rep, out_dir / "report.md")

report.json is written canonically (sorted keys, 2-space indent, no NaN), so two runs
on the same input give the same bytes except `meta.run_id`, `meta.created_at` and
`meta.timings` (`DYNAMIC_META`).

Redaction (P10.2.4), applied to every string in the report: http(s) URLs become
`<url>`; an absolute file-system path under the output directory becomes relative to
it, any other absolute path becomes `<path>`; without `show_ids` a SteamID64 inside a
string becomes its `p_<hex>` player id. Player ids are hashed by construction
(`player_label`); `find_leaks` is the check the tests run.

Filled by: P10.2.1, P10.2.2, P10.2.4.
"""
import hashlib
import json
import math
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from engine import paths
from engine.report import cards

# ── constants ─────────────────────────────────────────────────────────────────

REPORT_SCHEMA_VERSION = 1
SCHEMA_PATH  = Path(__file__).resolve().parent / "schema.json"
DYNAMIC_META = ("run_id", "created_at", "timings")
TOP_CARDS    = 5

URL_TOKEN  = "<url>"
PATH_TOKEN = "<path>"
_URL_RE      = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"(?:\\\\[^\s'\"<>|]+|[A-Za-z]:[\\/][^\s'\"<>|]*)")
_POSIX_RE    = re.compile(r"(?:(?<=^)|(?<=[\s'\"(=]))/(?:[^\s/'\"<>]+/)+[^\s'\"<>]*")
_STEAMID_RE  = re.compile(r"(?<!\d)7656119\d{10}(?!\d)")


# ── ids ───────────────────────────────────────────────────────────────────────

def player_label(actor_id_hash) -> str | None:
    """`p_<10 hex>`: sha256 of the engine's player id hash. The id hash itself is a
    sha1 of the SteamID (P3.2.6); hashing again keeps the integer out of the report.
    Pseudonymous, not anonymous: anyone with the SteamID can recompute it."""
    if actor_id_hash is None:
        return None
    return "p_" + hashlib.sha256(f"crosshair-report|{int(actor_id_hash)}".encode()).hexdigest()[:10]


def steamid_label(steamid) -> str:
    """The player label of a raw SteamID64 (same mapping as the engine's id hash)."""
    from engine.state.players import player_id_hash
    return player_label(player_id_hash(int(steamid)))


# ── redaction (P10.2.4) ───────────────────────────────────────────────────────

def _scrub_path(m: re.Match, out_dir: Path | None) -> str:
    text = m.group(0)
    if out_dir is not None:
        try:
            rel = Path(text).resolve().relative_to(Path(out_dir).resolve())
            return rel.as_posix()
        except (ValueError, OSError):
            pass
    return PATH_TOKEN


def _rel_or_keep(m: re.Match, out_dir: Path | None) -> str:
    """Paths inside out_dir become relative; anything else is left for the later passes."""
    text = m.group(0)
    if out_dir is not None:
        try:
            return Path(text).resolve().relative_to(Path(out_dir).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return text


def known_roots() -> tuple[str, ...]:
    """Directory prefixes that identify this machine or user: home, cwd, the repo, its
    data dir and the temp dir, in both separator spellings, longest first. A path under
    one of these is redacted whole, spaces included (MED-22, 2026-10-19): the old
    whitespace-free regex stopped at the first space of a user name or folder and left
    the rest of the path behind."""
    roots: set[str] = set()
    cands = [Path.home(), Path.cwd(), paths.ROOT, paths.DATA_DIR, Path(tempfile.gettempdir())]
    for c in cands:
        for p in {c, c.resolve()}:
            s = str(p).rstrip("\\/")
            if len(s) > 3:                       # never a bare drive
                roots.add(s)
                roots.add(s.replace("\\", "/"))
                roots.add(s.replace("/", "\\"))
    return tuple(sorted(roots, key=len, reverse=True))


# The rest of the path after a known root: directory segments may contain spaces (each
# is followed by another separator); the final segment stops at whitespace.
_ROOT_TAIL = ("(?:[\\\\/][^\\\\/\\s'\"<>|()]+(?:[ ][^\\\\/\\s'\"<>|()]+)*(?=[\\\\/]))*"   # dirs, spaces inside
              "(?:[\\\\/][^\\s'\"<>|()]*)?")                                            # final segment


def _roots_re() -> re.Pattern:
    alts = "|".join(re.escape(r) for r in known_roots())
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{alts}){_ROOT_TAIL}", re.IGNORECASE if os.name == "nt" else 0)


def _scrub_roots(s: str, out_dir: Path | None) -> str:
    def one(m: re.Match) -> str:
        text = m.group(0).rstrip(" .,;:")
        tail = m.group(0)[len(text):]
        return _scrub_path(_Match(text), out_dir) + tail
    return _roots_re().sub(one, s)


class _Match:
    """A minimal stand-in so `_scrub_path` can be reused on a trimmed string."""
    def __init__(self, text: str):
        self._t = text
    def group(self, _i=0) -> str:
        return self._t


def scrub_text(s: str, out_dir: Path | None = None, show_ids: bool = False) -> str:
    s = _URL_RE.sub(URL_TOKEN, s)
    if out_dir is not None:                      # exact out_dir paths first: they become relative
        s = _WIN_PATH_RE.sub(lambda m: _rel_or_keep(m, out_dir), s)
        s = _POSIX_RE.sub(lambda m: _rel_or_keep(m, out_dir), s)
    s = _scrub_roots(s, out_dir)                 # known roots, spaces and all
    s = _WIN_PATH_RE.sub(lambda m: _scrub_path(m, out_dir), s)
    s = _POSIX_RE.sub(lambda m: _scrub_path(m, out_dir), s)
    if not show_ids:
        s = _STEAMID_RE.sub(lambda m: steamid_label(m.group(0)), s)
    return s


def redact(obj, out_dir: Path | None = None, show_ids: bool = False, _key: str | None = None):
    """A copy of a JSON-like structure with every string scrubbed (see module doc).
    With `show_ids`, the `steamid` fields the caller added are kept as they are."""
    if isinstance(obj, dict):
        return {k: redact(v, out_dir, show_ids, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v, out_dir, show_ids, _key) for v in obj]
    if isinstance(obj, str):
        if show_ids and _key == "steamid":
            return obj
        return scrub_text(obj, out_dir, show_ids)
    return obj


def _strings(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, None
            yield from _strings(v, k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v, key)
    elif isinstance(obj, str):
        yield obj, key
    elif isinstance(obj, int) and not isinstance(obj, bool):
        yield str(obj), key


def find_leaks(obj, out_dir: Path | None = None, show_ids: bool = False) -> list[str]:
    """Every string (and integer) that breaks P10.2.4: a URL, an absolute path outside
    `out_dir` (a known root with spaces included), or (without `show_ids`) a SteamID64.
    Messages name the key and the kind only, never the matched text (R9; MED-23)."""
    out = []
    rr = _roots_re()
    for s, key in _strings(obj):
        if _URL_RE.search(s):
            out.append(f"url in {key}")
            s = _URL_RE.sub(URL_TOKEN, s)             # a URL's own path part is not a second leak
        if any(_scrub_path(_Match(m.group(0).rstrip(" .,;:")), out_dir) == PATH_TOKEN for m in rr.finditer(s)):
            out.append(f"path (known root) in {key}")
        elif any(_scrub_path(m, out_dir) == PATH_TOKEN for rx in (_WIN_PATH_RE, _POSIX_RE) for m in rx.finditer(s)):
            out.append(f"path in {key}")
        if not show_ids and _STEAMID_RE.search(s):
            out.append(f"steamid in {key}")
    return out


# ── schema ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(report: dict) -> list[str]:
    """Problems against schema.json (JSON Schema draft 2020-12); [] when valid."""
    import jsonschema
    v = jsonschema.Draft202012Validator(schema())
    errs = sorted(v.iter_errors(report), key=lambda e: list(map(str, e.absolute_path)))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errs]


class ReportSchemaError(ValueError):
    """report.json does not validate against engine/report/schema.json."""


# ── JSON ──────────────────────────────────────────────────────────────────────

def _clean(obj):
    """NaN / inf -> None (unknown), numpy scalars -> Python, tuples -> lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and type(obj).__module__ == "numpy":
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dumps(report: dict) -> str:
    return json.dumps(_clean(report), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(report: dict, path: Path) -> Path:
    """Validate, then write canonically. Raises ReportSchemaError listing problems."""
    rep = _clean(report)
    problems = validate(rep)
    if problems:
        raise ReportSchemaError("report.json invalid:\n  " + "\n  ".join(problems[:20]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(rep), encoding="utf-8", newline="\n")
    return path


def strip_dynamic(report: dict) -> dict:
    """The report without DYNAMIC_META (for reproducibility comparisons)."""
    rep = json.loads(dumps(report))
    for k in DYNAMIC_META:
        rep["meta"].pop(k, None)
    return rep


# ── Markdown (P10.2.2) ────────────────────────────────────────────────────────

def _f(v, digits: int = 3) -> str:
    return cards.fmt_signed(v, digits)


def _header(meta: dict) -> list[str]:
    teams = meta.get("teams") or {}
    vs = f"{teams.get('ct_start') or 'CT'} vs {teams.get('t_start') or 'T'}"
    lines = [f"# {meta['match_key']}: {vs}", "",
             f"- map: {meta.get('map') or 'n/a'} ({meta.get('map_version') or 'n/a'}), "
             f"tier: {meta['tier']}, date: {meta.get('date') or 'n/a'}",
             "- models: " + ", ".join(f"{k} {v or 'none'}" for k, v in sorted(meta["model_versions"].items())),
             f"- value stage: {meta['value_stage']}; config {meta['config_hash'][:12]}; run {meta['run_id']}",
             f"- created {meta['created_at']}; timings: "
             + ", ".join(f"{k} {v:.2f}s" for k, v in meta["timings"].items())]
    return lines


def _round_section(rnd: dict, chart: str | None) -> list[str]:
    lines = [f"## Round {rnd['round_num']}", "",
             f"Winner: {(rnd.get('winner_side') or 'n/a').upper()} ({rnd.get('end_reason') or 'n/a'}); "
             f"clock CT {_f(rnd['clock_ct'])}, clock T {_f(rnd['clock_t'])}; "
             f"conservation residual {rnd['conservation_residual']:.1e}", ""]
    if chart:
        lines += [f"![round {rnd['round_num']} win probability]({chart})", ""]
    top = sorted(rnd["actions"], key=lambda a: (-abs(a["V"]), a["action_id"]))[:TOP_CARDS]
    for i, a in enumerate(top, 1):
        lines.append(f"### {i}. {cards.type_words(a['type'])} (V {_f(a['V'])})")
        lines.append("")
        lines += cards.render(cards.card(a))
        lines.append("")
    return lines


def _player_table(players: list[dict]) -> list[str]:
    lines = ["## Players", "", "| player | sides | actions | V total | top type (V) | D~ (kill) | X~ (kill) |",
             "|---|---|---:|---:|---|---:|---:|"]
    for p in players:
        best = max(p["by_type"], key=lambda r: (abs(r["V"]), r["type"]), default=None)
        kill = next((c for c in p.get("by_class") or [] if c["class"] == "kill"), {})
        who = p.get("name") or p["player"]
        lines.append(f"| {who} | {'/'.join(s.upper() for s in p.get('sides') or [])} | {p['n']} | "
                     f"{_f(p['V_total'])} | {(best['type'] + ' ' + _f(best['V'])) if best else 'n/a'} | "
                     f"{_f(kill.get('D_tilde'))} | {_f(kill.get('X_tilde'))} |")
    return lines + [""]


def _team_table(teams: list[dict]) -> list[str]:
    lines = ["## Teams (by side)", "", "| side | V of actions | clock | buys valued |", "|---|---:|---:|---:|"]
    for t in teams:
        lines.append(f"| {t['side'].upper()} | {_f(t['V_total'])} | {_f(t['clock_total'])} | "
                     f"{len(t.get('buys') or [])} |")
    return lines + [""]


def markdown(report: dict, charts: list[str] | None = None) -> str:
    charts = charts or []
    by_round = {c.split("round_", 1)[1].split("_wp", 1)[0]: c for c in charts if "/round_" in c}
    lines = _header(report["meta"]) + [""]
    for rnd in report["rounds"]:
        lines += _round_section(rnd, by_round.get(str(rnd["round_num"])))
    lines += _player_table(report["players"])
    lines += _team_table(report["teams"])
    if "charts/team_clock.png" in charts:
        lines += ["![team clock](charts/team_clock.png)", ""]
    lines += ["## Warnings", ""]
    if report["warnings"]:
        groups: dict = {}                          # (code, message) -> rounds; report.json keeps every one
        for w in report["warnings"]:
            groups.setdefault((w["code"], w["message"]), []).append(w.get("round_num"))
        for (code, msg), rnds in groups.items():
            rs = sorted({r for r in rnds if r is not None})
            where = f" (rounds {', '.join(map(str, rs))})" if rs else ""
            times = f" x{len(rnds)}" if len(rnds) > 1 else ""
            lines.append(f"- `{code}`{times}{where}: {msg}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_markdown(report: dict, path: Path, charts: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown(report, charts), encoding="utf-8", newline="\n")
    return path
