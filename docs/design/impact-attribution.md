# Impact attribution — design proposal

**Status:** proposal, awaiting approval
**Supersedes:** the sampling-grid bracketing in `score_impact._impact_for_event`
**Audit refs:** HIGH-05, CRIT-04

---

## 1. The problem

`score_impact` values an action by bracketing it between the two nearest ~1 Hz
`round_states` samples and taking the win-probability difference across that whole
second. Measured on the previous corpus:

- **49%** of scored non-zero events shared an `|impact|` value with another event in
  the same round — they were handed the same delta.
- **115,022** brackets contained both a CT and a T actor, so a kill and the trade that
  answered it received equal-and-opposite *full-window* deltas. The opening kill of a
  traded exchange is credited by whichever side happened to net ahead over that second.
- All 157,074 `buy` events carry a hardcoded `time_into_round = 0.0`, which is exactly
  the first sample's timestamp, so every buy in a round collapses onto one bracket.

The root error: **the model is evaluated on a fixed time grid, but actions do not
happen on that grid.** Everything that occurred inside a sampling interval is
attributed, in full, to every event inside it.

## 2. What the field actually does

This is a solved problem shape in sports analytics, and CS has a direct precedent.

| Framework | Domain | Core idea |
|---|---|---|
| **WPA** (win probability added) | baseball, NFL | Value an event by the change in win probability it caused. The origin of this whole family. |
| **VAEP** — Decroos et al., KDD 2019 | soccer | Value *every* action, not just goals, by the change in scoring/conceding probability over the next few actions. The canonical "value individual actions in a team sport" paper. |
| **EPV** — Cervone et al. | basketball | Treat the game as a continuous-time stochastic process; value = change in expected possession value at the moment of the action. |
| **Xenopoulos et al., IEEE BigData 2020** | **CS:GO** | Win-probability model over game state; player action value = WP delta. Same author as `awpy`, the parser this project already uses. |

The consistent lesson across all four: **evaluate the value function immediately before
and immediately after each action** — never on a sampling grid. Crosshair already has
the right value function (the LightGBM WP model); it evaluates it in the wrong places.

*(Citations from memory — worth verifying exact venues before we lean on them in
public writing.)*

## 3. Proposal

### 3.1 Layer 0 — state at an arbitrary tick

The enabling change. Today `state_sampler` can only produce states on the 1 Hz grid.
Refactor so a state vector can be built for **any** tick:

```
build_state(tick, ctx) -> dict          # same 41 fields the sampler emits
sample_round_states(...)                # unchanged public behaviour; now calls build_state
```

Two-tier construction, for cost reasons (see §5):

- **Exact tier** — alive counts, HP, armour, helmets, equipment, utility held,
  positions, spread, distance-to-bomb, post-plant, clocks. Cheap, read straight from
  the tick table, computed at the exact tick.
- **Cached tier** — visibility (`*_spotted_count`) and sound (`*_heard_enemy`). Expensive
  raycasting and window scans. Reuse the most recent value from a **backwards-bounded**
  cache, which is exactly the semantics CRIT-02 established (never look forward).

Training rows keep coming from the 1 Hz grid, so the model is unchanged.

### 3.2 Layer 1 — classify actions by how they resolve

Not every event is instantaneous, and this is where a naive "before/after" design breaks.
A rotation is a player moving over a second; a smoke's value is realised over 18 seconds.
Taking a 2-tick delta around those yields ~0 and makes 77% of the corpus meaningless.

| Class | Events | Valuation |
|---|---|---|
| **Instantaneous** | kill, damage, bomb plant, defuse, explode, flash detonate, HE detonate | Jump across the action: `WP(t⁺) − WP(t⁻)` |
| **Durative** | smoke (pop→expire), molotov (burn window), rotation (movement window) | **Residual** drift over the action's window: total Δ over the window minus the instantaneous jumps inside it |
| **Round-level** | buy | Not an in-round action. Valued once per team per round: `WP(first live tick) − WP(economic baseline for that buy)` |

The durative rule is the important one. It says: *a smoke gets credit for the win
probability that accrued while it was up and that no discrete event explains.* That is
the honest answer to "was that smoke useful?" — and it is exactly what the current
design cannot express.

### 3.3 Layer 2 — simultaneity via Shapley

Two kills on the same tick have no natural order, and picking one arbitrarily biases the
result. For a set of `k` events at the same tick, compute each event's marginal
contribution averaged over all `k!` orderings — the Shapley value.

This is tractable because the state vector is composed of counts and sums, so applying
an event to a state is a cheap arithmetic edit (`alive_t −= 1`, `total_hp_t −= 87`, …).
No re-parsing. In practice `k ≤ 3` almost always; cap exact enumeration at `k ≤ 4`
(24 orderings) and Monte-Carlo sample above that.

### 3.4 The guarantee

For every round, the decomposition must satisfy:

```
Σ impact(instantaneous) + Σ impact(durative) + Σ impact(buy) + unattributed_drift
    ==  WP(round end) − WP(round start)
```

**This is the property that makes the design verifiable rather than merely plausible.**
It becomes a test asserting the identity holds to floating-point tolerance for every
round in the corpus. If credit is ever double-counted, invented, or lost, that test
fails. `unattributed_drift` is reported explicitly and never silently folded into a
player's number — if it is large, the model is missing something, and we want to see it.

## 4. What changes

| Component | Change |
|---|---|
| `state_sampler.py` | Extract `build_state(tick, ctx)`; `sample_round_states` becomes a caller. Additive. |
| `score_impact.py` | Replace `_impact_for_event` bracketing with the classifier + evaluator. |
| `db.py` | New columns on `events`: `impact_kind` (instantaneous/durative/round), `attribution_note`. New table `round_attribution` (per round: start WP, end WP, drift, residual check). **Additive migration, `user_version` 2.** |
| `feature_extractor.py` | Emit a real `time_into_round` for buys instead of the hardcoded `0.0`, plus each durative action's window bounds. |
| `test/` | The conservation test; a Shapley-symmetry test; a regression test that no two events in a round share an identical non-zero impact by construction. |

## 5. Cost

~20k events/match × 2 boundaries = ~40k WP evaluations. LightGBM scored 20,506 events
in 0.2 s, so prediction is free. The real cost is state construction: the cached tier
keeps the expensive visibility/sound work on the existing 1 Hz cadence, so the added
work is the cheap tier only — vector arithmetic over already-loaded tick rows.

Expected: **low seconds per match**, against ~100–140 s currently spent in feature
extraction. Not the bottleneck.

## 6. Deliberately out of scope

- **Match-level win probability.** A round win at 14–14 is worth more than at 15–0.
  Correct, valuable, and a much larger change — it needs score and economy context in
  the state vector. Phase 3.
- **Decision quality.** Impact measures *outcome*, not whether the decision was right.
  A won 50/50 duel scores +0.3 even if taking it was reckless. The "Stockfish" layer
  needs `E[Δ WP | situation, action]` so impact decomposes into decision, execution and
  luck. This is the natural next thing after attribution is correct, and it is where the
  coaching product actually lives — but it depends on this layer being right first.

## 7. Rollout

1. `build_state(tick)` + unit tests, `sample_round_states` behaviour unchanged
2. Action classifier + windows in `feature_extractor`
3. New evaluator in `score_impact`, behind `--attribution=grid|event` so the old path
   stays runnable for comparison
4. Conservation test; compare both attributions on the same matches
5. Migration + flip the default

Steps 1–2 are additive and safe to land before the corpus re-scrape. Steps 3–5 want the
new corpus, since re-scoring is cheap but re-ingesting is not.
