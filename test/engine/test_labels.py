"""
Labels and terminal states (P3.7): attached in a separate pass, y_dies_within_5s window
(C killed at 2000 -> 1 on (2000 - 5*tr, 2000], 0 before, None after), plant / defuse
within 15 s, time to next engagement capped at 30, round winner / reason, exactly one
terminal snapshot per round.

    .venv\\Scripts\\python.exe test\\engine\\test_labels.py
"""
import sys

from _harness import check, close, section, summary
from _synthetic import PLAYERS, STORIES, STORY_FACTS, make_raw_tables

from engine import config
from engine.state import builder, players
from engine.state.geometry import FlatGeometry
from engine.state.schema import LABEL_PREFIX

CFG = config.load()
GEO = FlatGeometry()


def snaps_at(raw, rnd, ticks):
    v = builder.RoundView.build(raw, rnd)
    out = [builder.snapshot_at(v.until(t), GEO, CFG, match_key="l", tier="pro") for t in ticks]
    return builder.attach_labels(out, raw, CFG)


def slot(snap, who):
    return snap.index_of(players.player_id_hash(PLAYERS[who][0]))


def main() -> int:
    for tr in (64, 128):
        T = lambda t128: int(round(t128 * tr / 128))  # noqa: E731
        raw = make_raw_tables("base", tr)
        k = T(2000)

        # ── dies within 5 s ──
        section(f"y_dies_within_5s @ {tr} tick")
        ticks = [k - 5 * tr - 1, k - 5 * tr, k - 5 * tr + 1, k - tr, k - 1, k, k + 1]
        ss = snaps_at(raw, 1, ticks)
        c = [s.labels["dies_within_5s"][slot(s, "C")] for s in ss]
        check("C: 0 at 2000-5tr-1 and at 2000-5tr (open end)", c[0] == 0 and c[1] == 0, str(c))
        check("C: 1 on (2000-5tr, 2000]", c[2:6] == [1, 1, 1, 1], str(c))
        check("C: None once already dead (2001)", c[6] is None)
        check("A survives the round: 0 everywhere", all(s.labels["dies_within_5s"][slot(s, "A")] == 0 for s in ss))
        check("10 entries, empty slots None", all(len(s.labels["dies_within_5s"]) == 10 for s in ss)
              and all(s.labels["dies_within_5s"][i] is None for s in ss for i in (2, 3, 4, 7, 8, 9)))
        full = builder.build(raw, GEO, CFG, match_key="l", tier="pro")
        r1 = [s for s in full if s.meta.round_num == 1]
        wrong = [s.meta.tick for s in r1 if s.meta.tick <= k
                 and s.labels["dies_within_5s"][slot(s, "C")] != int(k - 5 * tr < s.meta.tick <= k)]
        check("every built snapshot of round 1 agrees with the window", not wrong, str(wrong[:5]))

        # ── plant / defuse ──
        section(f"plant / defuse @ {tr} tick")
        p = T(3000)
        pl_ = snaps_at(raw, 1, [p - 15 * tr, p - 15 * tr + 1, p, p + 1])
        check("plant_within_15s: 0 at p-15s, 1 on (p-15s, p], None after",
              [s.labels["plant_within_15s"] for s in pl_] == [0, 1, 1, None])
        check("defuse_within_15s 0 when the round has no defuse",
              all(s.labels["defuse_within_15s"] == 0 for s in pl_))
        rt = make_raw_tables("retreat", tr)
        dfz = T(4200)
        dd = snaps_at(rt, 1, [dfz - 15 * tr, dfz - 15 * tr + 1, dfz, dfz + 1])
        check("retreat: defuse_within_15s 0, 1, 1, None", [s.labels["defuse_within_15s"] for s in dd] == [0, 1, 1, None])

        # ── engagements ──
        section(f"time to next engagement @ {tr} tick")
        e = snaps_at(raw, 1, [T(1000), T(1800) - tr, T(1800), k + 1, T(4000) + 1])
        tte = [s.labels["time_to_next_engagement_s"] for s in e]
        check("capped at 30 s far from any damage", tte[0] == min(30.0, (T(1800) - T(1000)) / tr))
        check("1.0 s before the HE damage, 0.0 on it", close(tte[1], 1.0) and tte[2] == 0.0)
        check("after 2000: next is the 4000 kill", close(tte[3], (T(4000) - k - 1) / tr))
        check("after the last engagement: 30.0 (censored)", tte[4] == 30.0)

        # ── outcome, terminal ──
        section(f"outcome and terminal @ {tr} tick")
        for story in STORIES:
            sraw = make_raw_tables(story, tr)
            snaps = builder.build(sraw, GEO, CFG, match_key="l", tier="pro")
            f = STORY_FACTS[story]
            ok_win = all(s.labels["round_won_ct"] == int(f["winner"][s.meta.round_num] == "ct")
                         and s.labels["round_end_reason"] == f["end_reason"][s.meta.round_num] for s in snaps)
            terms = {r: [s for s in snaps if s.meta.round_num == r and s.labels["is_terminal"] == 1] for r in (1, 2)}
            last = {r: max(s.meta.tick for s in snaps if s.meta.round_num == r) for r in (1, 2)}
            check(f"{story}: round_won_ct / end_reason on every snapshot", ok_win)
            check(f"{story}: exactly one terminal per round, at the last tick",
                  all(len(terms[r]) == 1 and terms[r][0].meta.tick == last[r] for r in (1, 2)))

    # ── separation from features ──
    section("labels are a separate pass")
    raw = make_raw_tables("base", 128)
    unl = builder.build(raw, GEO, CFG, match_key="l", tier="pro", labels=False)
    lab = builder.attach_labels([s.copy() for s in unl], raw, CFG)
    check("build(labels=False) leaves labels None", all(s.labels is None for s in unl))
    check("attaching labels changes nothing else", all(a.players == b.players and a.ctx == b.ctx for a, b in zip(unl, lab)))
    row = lab[0].to_row()
    check("label columns are y_* in rows", {LABEL_PREFIX + k for k in lab[0].labels} <= set(row)
          and "y_dies_within_5s" in row and "y_is_terminal" in row)

    return summary()


if __name__ == "__main__":
    sys.exit(main())
