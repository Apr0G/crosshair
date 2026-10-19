"""
Action cards for report.md (P10.2.2): who, what, where, when, V, Q, B, D, X, verdicts,
enabler lines ("flash by p_1a2b3c4d5e: 31% of this kill") and the panel's top-5 lines
("smoke to CT (41%), mean V +0.021").

    from engine.report import cards
    c = cards.card(action_row)            # action_row: one report.json action dict
    lines = cards.render(c)               # markdown lines

Card data comes only from the report.json action dict, so a card never shows a number
the JSON does not hold. Unknown values print as "n/a", never 0.

Filled by: P10.2.2.
"""

# ── constants ─────────────────────────────────────────────────────────────────

TOP_PANEL_LINES = 5
NA              = "n/a"

_TYPE_WORDS = {
    "kill": "kill", "team_kill": "team kill", "damage": "damage", "flash": "flash",
    "he": "HE grenade", "smoke": "smoke", "molotov": "molotov", "decoy": "decoy",
    "peek": "peek", "rotation": "rotation", "hold": "hold", "save": "save",
    "plant_start": "plant start", "plant_complete": "plant", "defuse_start": "defuse start",
    "defuse_complete": "defuse", "defuse_abort": "defuse abort", "bomb_drop": "bomb drop",
    "bomb_pickup": "bomb pickup", "buy": "buy", "team_buy": "team buy", "drop": "weapon drop",
    "positioning": "positioning (no action of its own)", "trade_setup": "trade setup (death that set up a trade)",
}

_ENABLER_WORDS = {
    "flash": "flash", "smoke_cover": "smoke cover", "trade_setup": "trade setup",
    "he_damage": "HE damage", "fire_pressure": "fire pressure", "info": "info",
}


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_signed(v, digits: int = 3) -> str:
    """+0.021 / -0.004; n/a for None."""
    if v is None:
        return NA
    return f"{round(float(v), digits) + 0.0:+.{digits}f}"          # + 0.0: no "-0.000"


def fmt_pct(v) -> str:
    if v is None:
        return NA
    return f"{100.0 * float(v):.0f}%"


def type_words(t: str | None) -> str:
    return _TYPE_WORDS.get(t or "", t or NA)


def panel_line(entry: dict) -> str:
    """`type|subtype|location` -> "smoke to CT (41%), mean V +0.021"."""
    parts = (entry.get("key") or "").split("|")
    t, sub, loc = (parts + ["", "", ""])[:3]
    what = type_words(t) + (f" ({sub})" if sub else "")
    where = f" to {loc}" if loc else ""
    return f"{what}{where} ({fmt_pct(entry.get('freq'))}), mean V {fmt_signed(entry.get('mean_V'))}"


def enabler_line(e: dict, event_type: str | None) -> str:
    """"flash by p_1a2b3c4d5e: 31% of this kill"."""
    kind = _ENABLER_WORDS.get(e.get("kind") or "", e.get("kind") or "enabler")
    by = f" by {e['actor']}" if e.get("actor") else ""
    return f"{kind}{by}: {fmt_pct(e.get('share'))} of this {type_words(event_type)} (V {fmt_signed(e.get('V'))})"


# ── card ──────────────────────────────────────────────────────────────────────

def card(a: dict) -> dict:
    """Card data for one report.json action dict."""
    when = None
    if a.get("t_s") is not None:
        when = f"{float(a['t_s']):.1f} s after freeze end"
    elif a.get("tick") is not None:
        when = f"tick {a['tick']}"
    panel = a.get("panel") or None
    return {
        "action_id":   a.get("action_id"),
        "who":         a.get("name") or a.get("actor") or NA,
        "side":        (a.get("actor_side") or NA).upper(),
        "what":        type_words(a.get("type")) + (f" ({a['subtype']})" if a.get("subtype") else ""),
        "where":       a.get("where") or NA,
        "when":        when or NA,
        "V":           a.get("V"),
        "V_direct":    a.get("V_direct"),
        "V_enabler":   a.get("V_enabler"),
        "V_movement":  a.get("V_movement"),
        "Q":           a.get("Q"),
        "B":           a.get("B"),
        "D":           a.get("D"),
        "X":           a.get("X"),
        "verdicts":    (a.get("verdict_decision"), a.get("verdict_execution")),
        "enablers":    [enabler_line(e, a.get("type")) for e in a.get("enablers") or []],
        "panel":       [panel_line(e) for e in (panel or {}).get("top", [])[:TOP_PANEL_LINES]],
        "panel_n":     (panel or {}).get("n_used"),
        "flags":       list(a.get("flags") or []),
        "method":      a.get("method"),
    }


def render(c: dict) -> list[str]:
    """Markdown lines of one card."""
    dec, exe = c["verdicts"]
    lines = [
        f"**{c['what']}** by {c['who']} ({c['side']}) at {c['where']}, {c['when']}",
        "",
        f"- V {fmt_signed(c['V'])} (direct {fmt_signed(c['V_direct'])}, enabler {fmt_signed(c['V_enabler'])}, "
        f"movement {fmt_signed(c['V_movement'])}; method {c['method'] or NA})",
        f"- Q {fmt_signed(c['Q'])}, B {fmt_signed(c['B'])}, D {fmt_signed(c['D'])}, X {fmt_signed(c['X'])}; "
        f"decision {dec or NA}, execution {exe or NA}",
    ]
    for e in c["enablers"]:
        lines.append(f"- {e}")
    if c["panel"]:
        lines.append(f"- what pros did here (panel of {c['panel_n']}):")
        lines += [f"  - {p}" for p in c["panel"]]
    if c["flags"]:
        lines.append(f"- flags: {', '.join(c['flags'])}")
    return lines
