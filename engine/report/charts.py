"""
matplotlib charts (P10.2.3), Agg backend, no seaborn:

    charts/round_<n>_wp.png          CT win probability over the round, action markers
                                     coloured by side and sized by |V|
    charts/player_<id>_summary.png   D~ vs X~ per type class with 90% intervals when the
                                     value stage ran; otherwise V per action type
    charts/team_clock.png            team clock per round (CT up, T down)

    from engine.report import charts
    files = charts.write_all(report, out_dir / "charts")      # relative names

Every chart reads only report.json data (`rounds[].wp_curve` + `actions`,
`players[].by_class` / `by_type`, `rounds[].clock_ct` / `clock_t`), so the numbers
behind a picture are always in the JSON. Pixel goldens are not used (fonts differ by
machine); tests check the files exist with non-zero size.

Colours: CT = categorical slot 1 (blue), T = slot 2 (orange) of the dataviz reference
palette (validated pair); text stays in ink colours, grid recessive.

Filled by: P10.2.3.
"""
from pathlib import Path

# ── style ─────────────────────────────────────────────────────────────────────

SIDE_COLOUR  = {"ct": "#2a78d6", "t": "#eb6834"}
INK          = "#0b0b0b"
INK_2        = "#52514e"
GRID         = "#e4e3df"
SURFACE      = "#fcfcfb"
DPI          = 100
SIZE_WP      = (9.0, 3.6)
SIZE_PLAYER  = (6.0, 4.2)
SIZE_CLOCK   = (8.0, 3.4)
MARKER_MIN   = 30.0          # pt^2, >= 8 px markers
MARKER_SCALE = 3000.0        # pt^2 per unit |V|
_PNG_META    = {"Software": None}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(GRID)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata=_PNG_META)
    _plt().close(fig)
    return path


# ── charts ────────────────────────────────────────────────────────────────────

def round_wp(rnd: dict, path: Path) -> Path:
    """W (CT win probability) against seconds after freeze end, with the round's
    actions marked at their time (synthetic rows without a time are skipped)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=SIZE_WP)
    curve = [p for p in rnd["wp_curve"] if p.get("t_s") is not None]
    xs = [p["t_s"] for p in curve]
    ys = [p["W"] for p in curve]
    ax.step(xs, ys, where="post", color=INK_2, linewidth=2, label="CT win probability")
    for side in ("ct", "t"):
        pts = [a for a in rnd["actions"] if a.get("actor_side") == side and a.get("t_s") is not None]
        if not pts:
            continue
        ax.scatter([a["t_s"] for a in pts], [_w_at(curve, a["t_s"]) for a in pts],
                   s=[MARKER_MIN + MARKER_SCALE * abs(a["V"]) for a in pts], color=SIDE_COLOUR[side],
                   edgecolors=SURFACE, linewidths=2, zorder=3, label=f"{side.upper()} action (size = |V|)")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color=GRID, linewidth=1)
    _style(ax, f"Round {rnd['round_num']}: CT win probability", "seconds after freeze end", "W (CT)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)
    return _save(fig, path)


def _w_at(curve: list[dict], t: float) -> float:
    """W of the last curve point at or before t (step interpolation)."""
    w = curve[0]["W"] if curve else 0.5
    for p in curve:
        if p["t_s"] <= t:
            w = p["W"]
        else:
            break
    return w


def player_summary(player: dict, path: Path) -> Path:
    """D~ vs X~ per type class with 90% intervals, or V per type when the value stage
    was skipped (no class has both numbers)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=SIZE_PLAYER)
    pts = [c for c in player.get("by_class") or []
           if c.get("D_tilde") is not None and c.get("X_tilde") is not None]
    colour = SIDE_COLOUR.get((player.get("sides") or ["ct"])[0], SIDE_COLOUR["ct"])
    if pts:
        for c in pts:
            xerr = [[c["D_tilde"] - c["D_lo"]], [c["D_hi"] - c["D_tilde"]]] if c.get("D_lo") is not None else None
            yerr = [[c["X_tilde"] - c["X_lo"]], [c["X_hi"] - c["X_tilde"]]] if c.get("X_lo") is not None else None
            ax.errorbar([c["D_tilde"]], [c["X_tilde"]], xerr=xerr, yerr=yerr, fmt="o", color=colour,
                        ecolor=INK_2, elinewidth=1, markersize=8)
            ax.annotate(f"{c['class']} (n={c['n']})", (c["D_tilde"], c["X_tilde"]), fontsize=8,
                        color=INK_2, xytext=(5, 5), textcoords="offset points")
        ax.axhline(0, color=GRID, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=1)
        _style(ax, f"{player['player']}: decision vs execution (shrunk, 90% interval)",
               "D~ (decision)", "X~ (execution)")
    else:
        rows = sorted(player.get("by_type") or [], key=lambda r: r["type"])
        names = [r["type"] for r in rows] or ["(none)"]
        vals = [r["V"] for r in rows] or [0.0]
        ax.barh(names, vals, color=colour, height=0.6)
        ax.axvline(0, color=INK_2, linewidth=1)
        _style(ax, f"{player['player']}: realized V by action type", "sum of V", "")
    return _save(fig, path)


def team_clock(rounds: list[dict], path: Path) -> Path:
    """Clock bucket per round: CT's clock above zero, T's below (both are CT-positive
    win-probability terms; T's is drawn negated so a T gain points down)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=SIZE_CLOCK)
    xs = [r["round_num"] for r in rounds]
    ax.bar(xs, [r["clock_ct"] for r in rounds], color=SIDE_COLOUR["ct"], width=0.6, label="CT clock")
    ax.bar(xs, [-r["clock_t"] for r in rounds], color=SIDE_COLOUR["t"], width=0.6, label="T clock (negated)")
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(xs)
    _style(ax, "Team clock per round", "round", "win probability")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)
    return _save(fig, path)


# ── all ───────────────────────────────────────────────────────────────────────

def write_all(report: dict, charts_dir: Path) -> list[str]:
    """Every chart of the report; returns names relative to the charts dir's parent
    (e.g. `charts/round_1_wp.png`), in a fixed order."""
    charts_dir = Path(charts_dir)
    written = []
    for rnd in report["rounds"]:
        written.append(round_wp(rnd, charts_dir / f"round_{rnd['round_num']}_wp.png"))
    for p in report["players"]:
        written.append(player_summary(p, charts_dir / f"player_{p['player']}_summary.png"))
    written.append(team_clock(report["rounds"], charts_dir / "team_clock.png"))
    return [f"{charts_dir.name}/{w.name}" for w in written]
