# states.parquet schema changelog

Every change to `engine/state/schema.py::SCHEMA_VERSION` gets an entry here (P3.8.1).
`test/engine/test_states_schema.py` fails if the current version has no `## v<N>` entry.
A version bump means every existing `states.parquet` must be rebuilt (`Snapshot.from_row`
refuses rows of another version).

## v1 — 2026-10-19

First version (P3, Amendment 1 contracts).

- One row per snapshot: `schema_version`, `match_key`, `round_num`, `tick`, `t_s`
  (seconds since freeze_end), `kinds` (list of `grid` / `boundary_pre` /
  `boundary_post` / `terminal`).
- `players`: list of 10 structs, Appendix A.2 fields in order; CT first then T, each
  sorted by SteamID, padded with empty slots (`alive = 0`, `player_id = -1`) when a side
  has fewer than 5 players.
- `utility`: list of up to 16 structs (A.3), oldest first, newest kept on overflow.
- `ctx_*`: the A.4 context fields, flat.
- `pair`: 500 float32 (10 x 10 x 5, channels `los, dist_units, nav_dist_units,
  smoke_between, same_team`), NaN for unknown.
- Labels (dropped at inference by the `y_` prefix): `y_round_won_ct`,
  `y_round_end_reason`, `y_dies_within_5s` (list of 10), `y_plant_within_15s`,
  `y_defuse_within_15s`, `y_time_to_next_engagement_s` (capped at 30),
  `y_is_terminal`. Event windows are `t <= event < t + horizon`; None once the event
  already happened (or for an empty slot).
- Positions are raw Hammer units, angles raw degrees (normalization happens in the
  model).
- Amendment, same day (2026-10-19, audit HIGH-04; no `states.parquet` had been built
  outside temp dirs, so no version bump): `player.on_fire` may be None while alive.
  It is 1 inside a fire of known extent, 0 when every active fire is known and misses,
  None when a fire of unknown extent is burning (awpy 3.1 emits `extent_radius` null,
  so on real demos it is None for the life of every molotov). Dataset layout unchanged:
  the field was already nullable and carries a `present_on_fire` bit.
