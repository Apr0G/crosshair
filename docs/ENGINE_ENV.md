# Engine environment

How the Crosshair engine v2 environment is set up on the Windows PC (P0.1, Amendment 1).

## Interpreter and venv

- One venv at the repo root: `.venv` (Amendment 1 replaced the planned `.venv-engine`).
- Interpreter: `.venv\Scripts\python.exe`, Python 3.12.10 (64-bit, Windows).
- The old 3.11 venv was renamed `.venv-old311`. The engine never uses it, and Faig can
  delete it.
- Always call the venv interpreter explicitly. A bare `python` on PATH may be a
  different install.

## Installing dependencies

`requirements-engine.txt` is the authoritative dependency list. `pyproject.toml` reads
its dependencies from that file.

```
.venv\Scripts\python.exe -m pip install -r requirements-engine.txt
```

After any deliberate upgrade, re-freeze the exact versions into
`requirements-engine.lock`. `test/engine/test_env.py` compares installed versions
against the lock once it exists.

torch is **not** in `requirements-engine.txt`. Install it separately from AMD's
ROCm-for-Windows wheels for Python 3.12, following AMD's "PyTorch on Windows" release
notes (P0.1.3). On other platforms, install the matching torch build before running
`pip install -r`. The torch version, device, and GPU benchmark results go in the
section below, filled in by P0.1.

## torch / GPU

(filled in by P0.1: chosen path (native Windows ROCm, WSL2, or CPU), torch version,
device name, `bench/bench_gpu.py` numbers)

## Checks

```
.venv\Scripts\python.exe test\engine\test_env.py            # packages import; torch SKIP
.venv\Scripts\python.exe test\engine\test_env.py --torch    # also imports torch (P0.1)
.venv\Scripts\python.exe test\run_all.py                    # whole engine suite
.venv\Scripts\python.exe -m pyflakes engine test bench
```

## Parser notes

(filled in by P2.1.1: the exact awpy 3 calls for rounds, kills, damages, grenades,
bomb events, per-tick player props, footsteps, tick rate, map name and game version,
plus any parser differences table)

### awpy 3.1.0 calls used (P2.1.1, checked against the installed package 2026-10-19)

No demo exists on this machine yet, so every call below is checked against awpy
3.1.0's stubs/docstrings (`_awpy.pyi`, `awpy.schema`) and the dataset reference at
awpy.readthedocs.io, and exercised only through the fake demo in
`test/engine/test_awpy3_parser.py`. Re-verify on the golden demo:
`test\engine\test_awpy3_parser.py <demo>` runs the same contract checks on a real parse.

| Need | Call |
|---|---|
| open / header | `awpy.Demo(path)`; `demo.header` (`map_name`, `server_name`, `build_num`, `playback_ticks`, `playback_time`, ...) |
| preload | `demo.load("rounds","kills","damages","bomb","grenades","fires","smokes","shots","players","blinds","item_events")` (shares decode passes) |
| tick rate | `playback_ticks / playback_time` from the header. **Not** `demo.tick_rate`: it silently returns 64.0 when playback timing is missing; we emit None and quarantine instead |
| map / game version / server | `header["map_name"]`, `str(header["build_num"])`, `header["server_name"]` |
| rounds | `demo.rounds` |
| kills / damages | `demo.kills`, `demo.damages` |
| grenades (throw) | `demo.grenades` (one row per tick per live projectile), collapsed per entity |
| grenades (detonate / expire) | `demo.smokes`, `demo.fires` (one row per cloud / fire with start/end tick), `demo.blinds` |
| bomb | `demo.bomb` + per-tick `is_defusing` + `rounds.reason_name` |
| per-tick player props | `demo.snapshots(start_tick=first_round_start, end_tick=last_round_end)`, one call, every tick |
| shots / items / roster | `demo.shots`, `demo.item_events`, `demo.players` |
| footsteps | `demo.events["player_footstep"]` if present (see below) |

### awpy -> Appendix A.1 mapping (`engine/parse/awpy3_parser.py`)

Sides are normalised everywhere to `ct` / `t` (awpy gives `counter-terrorist` /
`terrorist` strings or team numbers 3 / 2). Rows of every event table outside all
round windows `[start_tick, official_end_tick or end_tick]` (warmup, post-round gap)
are dropped; on a shared boundary tick the later round wins.

| A.1 column(s) | awpy source / derivation |
|---|---|
| `meta.map`, `game_version`, `server` | header `map_name`, `build_num` (as str), `server_name` |
| `meta.tick_rate` | `round(playback_ticks / playback_time)`; None if the header lacks them |
| `meta.demo_sha256` | sha256 of the file (None when parsing a fake without a file) |
| `rounds.*_tick`, `round_num` | `rounds`, same names; `official_end_tick` is null for the last round (awpy); a null `start_tick` (never-null in schema.py) falls back to the previous round's end, then `freeze_end_tick` |
| `rounds.winner_side`, `end_reason` | `winner_side` (fallback: `winner` team number), `reason_name` |
| `rounds.bomb_planted` | any `plant_complete` in the round |
| `rounds.ct/t_score_before` | derived: teams tracked by roster overlap with round 1's CT side (handles the halftime swap); knife rounds (`is_knife_round`) score nothing; null when there are no ticks |
| `ticks.vx/vy/vz` | `velocity_x/y/z` |
| `ticks.has_c4`, `is_ducking` | `has_bomb`, `is_crouched` |
| `ticks` rows | `snapshots` rows with a ct/t side (spectator / caster rows dropped) |
| `ticks.is_alive` | `health > 0` (null health -> False) |
| dead players (schema.py contract) | rows kept: health 0, x/y/z = last alive position, `active_weapon` null, `inventory` [], `equipment_value` 0, `flash_duration` 0. Whether awpy emits rows for dead players at all is unverified without a demo |
| `ticks.flash_duration` | derived REMAINING blind seconds (schema.py contract): latest `blinds` row of that player, `duration - elapsed`, floored at 0. awpy's `flash_duration` (onset duration) is not used; null if the demo has no blinds frame |
| `ticks.is_planting` | derived: between a player's `start_plant` and their next `finish_plant` / `interrupt_plant` (or round end) |
| `ticks.kills_round`, `damage_round` | derived: running count of enemy kills / sum of enemy `dmg_health` within the round (team kills, team damage and self-damage excluded) |
| `ticks.spotted_by` | **null** (see below) |
| `kills.assister_steamid`, `flash_assister_steamid` | `assister_steamid`, split by `assistedflash` when that column exists; otherwise flash_assister is **null** |
| `kills.through_smoke`, `penetrated` | `thrusmoke`; `penetrated` (bool in awpy) as i16 0/1 |
| `kills.attacker_blind`, `victim_blind` | event `attackerblind` if present, else the snapshot `is_blinded` of that player at the kill tick |
| `kills.distance` | derived: 3D distance between attacker and victim positions in Hammer units (CS2's own `distance` key is metres and not always present) |
| `kills.is_trade` (optional, schema.py `OPTIONAL_COLUMNS`) | awpy `is_trade`, kept when present |
| `kills.team_kill` | `attacker_side == victim_side` and attacker != victim; False when there is no attacker (world damage) |
| `damages.hitgroup` | `hitgroup_name` (fallback: `hitgroup` code as str) |
| `grenades.entity_id` (and the linked `smokes` / `infernos` / `flashes.entity_id`) | `(entity_serial * 65536 + entity_id) * 16 + segment` (awpy: slot + generation identify an entity); a slot reused without a serial, or a trajectory with a > 8-tick hole, is split into segments that each get their own id (MED-12) |
| `grenades.type` | `flashbang` -> `flash`; molotov vs incendiary from the linked fire's `type` when it has one |
| `grenades.throw_tick` | first trajectory tick |
| `grenades.detonate_tick`, `x,y,z` | smoke / molotov: linked cloud / fire `start_tick` and position; flash / HE: last trajectory tick and point; decoy: first tick at its final position (it came to rest) and that point |
| `grenades.expire_tick` | smoke / molotov: linked `end_tick`; flash / HE: **null** (instant); decoy: last trajectory tick (the entity is removed when it stops); the state builder gives a decoy with no end its nominal 15 s (MED-08) |
| linking | smokes / fires / blinds -> the thrower's latest grenade of the matching type thrown at or before the event, within 20 s |
| `smokes`, `infernos` | `demo.smokes`, `demo.fires`; `infernos.extent_radius` **null** (awpy has one row per fire, no flame extents) |
| `flashes` | `demo.blinds`: `tick` -> `detonate_tick`, `victim_steamid` -> `player_steamid`, `duration` -> `blind_duration` |
| `bomb.event` | `start_plant` -> `plant_start`, `finish_plant` -> `plant_complete`, `defuse` -> `defuse_complete`, `pickup`, `drop`; `interrupt_plant` **dropped** (no A.1 name) |
| `bomb` `defuse_start` / `defuse_abort` | derived from `is_defusing` edges: rising edge = start (`has_kit` from `has_defuser`); falling edge with no `defuse_complete` within 4 ticks = abort, at the first non-defusing tick |
| `bomb` `explode` | round with `reason_name` `target_bombed`: tick = round `end_tick`, site / position of that round's plant |
| `bomb.site` | `bombsite` |
| `weapon_fire` | `demo.shots` (`tick`, `steamid`, `weapon`) |
| `items` | `demo.item_events` with `action` in drop / pickup; `purchase` rows **dropped** (no A.1 name) |
| `footsteps` | `events["player_footstep"]` with a resolved steamid column (`user_steamid`, `userid_steamid`, `player_steamid` or `steamid`), position from the same tick's snapshot; otherwise an empty A.1 frame. The column name is unverified on a real demo |

Dropped-row counts are reported in `meta["dropped_rows"]`; every all-null column of a
non-empty table is listed in `meta["unavailable"]`. Adding `plant_abort` / `purchase`
to the A.1 vocabularies would be an additive schema change (ask first).

### Fields that could not be obtained (emitted as typed nulls)

- `ticks.spotted_by`: awpy 3.1.0 has no spotted prop in `SNAPSHOT_PROPERTIES`. The raw
  `m_bSpottedByMask` could be requested with `demo.ticks(props=[...])`, but (a) it is a
  2 x uint32 array and awpy's `ticks` only infers Int64/Float64/Boolean/Utf8 columns,
  and (b) its bits index controller entity slots, while `ticks(players_only=True)`
  returns `steamid` but not the slot, so bit -> steamid cannot be mapped reliably.
  Neither could be tried without a demo. Revisit on the golden demo; until then the
  state builder computes visibility itself (P3.5, VisibilityChecker).
- `ticks.is_reloading`: null whenever awpy's own field is absent (awpy documents this).
- `infernos.extent_radius`: no per-flame data in `demo.fires`. Consequences:
  `player.on_fire` is None (unknown, never 0) while a fire burns, and the
  `fire_pressure` enabler links through inferno damage rows (victim burning), not
  through the extent.
- `grenades.expire_tick` for flash, HE (instant by definition) and decoy (not exposed).
- `rounds.official_end_tick` for the last round (not in the demo).
- `kills.flash_assister_steamid` unless awpy's kills frame carries `assistedflash`
  (the raw `player_death` key exists; the documented frame does not list it).
- `footsteps`: empty unless the event frame has a resolved steamid column.

Caveat: awpy gives bots steamid 0, so several bots share one id. FACEIT / pro demos have
none; the quarantine side count ignores steamid 0 (a bot-filled side fails `not_5v5`).

### Quarantine (P1.4.1) from `header_info` / `quarantine_check`

`header_info(path)` reads the header only; `with_roster=True` (used by `worker` before
every parse, by `--smoke` and by `analyze`) also reads `rounds` and `players`. Reasons, in order: `tick_rate_not_128` (|rate - 128|
> 0.5, or unknown), `too_few_rounds` (< 16), `not_5v5` (distinct non-bot players per
side != 5), `unknown_map` (not in `engine.maps.assets.ACTIVE_POOL`),
`duration_implausible` (playback time outside 15 min to 5 h). `config.yaml` has no
quarantine keys (Appendix D lists none), so the thresholds are constants at the top of
`awpy3_parser.py`. The worker adds `header_unreadable` when the header cannot be read.
This is the only rule table: `engine.data.quarantine.check` delegates to it (judging a
missing key as a failure) and `quarantine.quarantine()` moves the file and sets the row,
accepting exactly these reason strings.

### Map assets

`engine/maps/assets.py` reads mesh + nav straight from awpy's cache
(`~/.awpy/<release>/geometry|navs/`) for the release pinned in config and never
downloads (a missing map or release raises `MapAssetError`). Nav distances use
`NavMesh.find_path(weight="distance")` summed over 3D centroid distances (identical to
our own Dijkstra over `neighbors()`, checked in `test_map_assets.py`); connections can
be one-way, so distances are not always symmetric.

## GPU / torch (verified 2026-10-19)

- Machine: Ryzen 7 7800X3D, Radeon RX 9060 XT 16 GB (gfx1200), 32 GB DDR5, Windows 11 Pro,
  AMD Software 26.8.1 (driver 32.0.31041.1004; AMD requires ≥ 26.2.2 for this build).
- Installed into `.venv` from AMD's wheels (exact URLs with sha256 in
  `requirements-engine.lock`): `rocm-sdk-core/devel/libraries-custom 7.2.1`, `rocm 7.2.1`,
  `torch 2.9.1+rocm7.2.1` (cp312). torchvision/torchaudio not installed (not needed).
- `torch.cuda.is_available()` → `True`, device `AMD Radeon RX 9060 XT`, HIP 7.2.53211.
- Training smoke test (3-layer transformer, d=128, batch 512×24 tokens, AdamW, forward+backward):
  fp32 22.5 it/s, bf16 autocast 20.8 it/s, no NaN, 487 MB peak. AMD's docs say "no ML
  training support" on Windows; empirically autograd and optimizer steps work. Treat
  as unsupported-but-working; keep checkpoints every epoch (P6.3.1) and the WSL2 fallback.
- Warnings seen: flash/mem-efficient SDPA on this GPU is experimental; enable with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` only after a correctness check.
- Chosen path: **native Windows ROCm PyTorch**. WSL2 and CPU fallbacks not needed so far.

## Map assets (verified 2026-10-19)

awpy-data release **2000933** pinned in `engine/config.yaml`; mesh + nav for de_mirage,
de_inferno, de_nuke, de_ancient, de_anubis, de_dust2, de_overpass, de_train, de_vertigo
cached under `C:\Users\AproG\.awpy\2000933` (`VisibilityChecker(map, version=2000933)`,
`NavMesh(map, version=2000933)`; 66k–548k triangles, 2.0k–3.9k nav areas per map).
