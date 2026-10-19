"""
SQLite persistence layer.

Three tables:
  processed_matches — dedup index of match_ids that have been parsed
  events            — every player action (kill, util throw, buy, rotation, bomb)
                      with WP-before / WP-after / impact filled by score_impact.py
  round_states      — team-state snapshots: the ~1 Hz grid the WP model trains on,
                      plus `boundary` rows around instantaneous actions for attribution

Every write goes through store_match(), which is ONE transaction per match.
Importing this module needs only the stdlib; numpy is loaded lazily when a row is
encoded, so `main.py status` and the schema tests run on a bare interpreter.
"""
import json
import sqlite3
from pathlib import Path

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
DB_PATH    = DATA_DIR / "crosshair.db"
MODEL_PATH = DATA_DIR / "win_prob.lgb"


class _Encoder(json.JSONEncoder):
    """numpy scalars/arrays leak into the event blobs; json can't serialise them."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_Encoder)


# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 2

# Ordered, additive-only migrations applied to any DB whose user_version is behind.
# CREATE TABLE IF NOT EXISTS can never add a column to an existing table, so new
# columns MUST be added here as well as in _init_schema.
_MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        "ALTER TABLE events ADD COLUMN p_before REAL",
        "ALTER TABLE events ADD COLUMN p_after  REAL",
        "ALTER TABLE events ADD COLUMN impact   REAL",
        "ALTER TABLE round_states ADD COLUMN active_smokes_xy   TEXT",
        "ALTER TABLE round_states ADD COLUMN active_infernos_xy TEXT",
        "ALTER TABLE round_states ADD COLUMN site_smoked        INTEGER",
        "ALTER TABLE round_states ADD COLUMN site_on_fire       INTEGER",
    ]),
    (2, [
        # Impact attribution needs the state immediately before and after an action.
        # Those extra rows live in round_states alongside the 1 Hz grid, so they must
        # be distinguishable — training uses the grid only, or it over-samples around
        # kills. NULL means "grid": every pre-existing row predates boundary states.
        "ALTER TABLE round_states ADD COLUMN state_kind TEXT",
        "CREATE INDEX IF NOT EXISTS idx_states_kind ON round_states(state_kind)",
    ]),
]

STATE_GRID     = "grid"       # the ~1 Hz training sample
STATE_BOUNDARY = "boundary"   # tick±1 around an instantaneous action


def _init_schema(c: sqlite3.Connection):
    c.executescript("""
        CREATE TABLE IF NOT EXISTS processed_matches (
            match_id     TEXT PRIMARY KEY,
            map          TEXT,
            demo_url     TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        TEXT NOT NULL,
            map             TEXT,
            round_num       INTEGER,
            player_side     TEXT,
            event_type      TEXT,
            time_into_round REAL,
            situation       TEXT,
            action          TEXT,
            outcome         TEXT,
            round_won       INTEGER,
            p_before        REAL,
            p_after         REAL,
            impact          REAL
        );

        CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
        CREATE INDEX IF NOT EXISTS idx_events_type  ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_won   ON events(round_won);

        CREATE TABLE IF NOT EXISTS round_states (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id             TEXT NOT NULL,
            map                  TEXT,
            round_num            INTEGER,
            tick                 INTEGER,
            time_into_round_s    REAL,
            time_remaining_s     REAL,
            post_plant           INTEGER,
            alive_ct             INTEGER,
            alive_t              INTEGER,
            total_hp_ct          REAL,
            total_hp_t           REAL,
            total_armor_ct       REAL,
            total_armor_t        REAL,
            helmets_ct           INTEGER,
            helmets_t            INTEGER,
            ct_spread            REAL,
            t_spread             REAL,
            has_defuser          INTEGER,
            equip_value_ct       REAL,
            equip_value_t        REAL,
            smokes_ct            INTEGER,
            smokes_t             INTEGER,
            flashes_ct           INTEGER,
            flashes_t            INTEGER,
            he_ct                INTEGER,
            he_t                 INTEGER,
            molotovs_ct          INTEGER,
            molotovs_t           INTEGER,
            active_smokes        INTEGER,
            active_infernos      INTEGER,
            active_smokes_xy     TEXT,
            active_infernos_xy   TEXT,
            site_smoked          INTEGER,
            site_on_fire         INTEGER,
            min_dist_ct_to_bomb  REAL,
            min_dist_t_to_bomb   REAL,
            ct_spotted_count     INTEGER,
            t_spotted_count      INTEGER,
            ct_heard_enemy       INTEGER,
            t_heard_enemy        INTEGER,
            round_won_ct         INTEGER,
            state_kind           TEXT
        );

        -- Composite: score_impact filters by match_id and orders by
        -- (round_num, time_into_round_s); a match_id-only index forces a temp b-tree.
        CREATE INDEX IF NOT EXISTS idx_states_match_order
            ON round_states(match_id, round_num, time_into_round_s);
        CREATE INDEX IF NOT EXISTS idx_states_kind ON round_states(state_kind);
    """)


def _migrate(c: sqlite3.Connection):
    """Apply any additive migrations this DB has not seen. Safe to run every open."""
    version = c.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return
    for target, statements in _MIGRATIONS:
        if target <= version:
            continue
        for stmt in statements:
            try:
                c.execute(stmt)
            except sqlite3.OperationalError as e:
                # Column already present (added out of band before this ran).
                if "duplicate column" not in str(e).lower():
                    raise
        c.execute(f"PRAGMA user_version = {target}")
        print(f"[db] migrated schema → v{target}")
    c.commit()


# ── connections ───────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Open the DB, creating the schema on a fresh file and migrating an old one."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    # busy_timeout MUST be set before anything that takes a lock. journal_mode=WAL
    # needs an exclusive lock, so with the timeout set afterwards it failed instantly
    # the moment two ingest workers opened the DB at once ("database is locked").
    c.execute("PRAGMA busy_timeout=30000")
    # WAL is persistent, so only pay for the exclusive lock when it isn't already on.
    # It lets `score`/`train` read while a scrape writes; without it a reader blocks
    # the writer, whose insert then fails mid-match and leaves rows with no marker.
    try:
        if (c.execute("PRAGMA journal_mode").fetchone()[0] or "").lower() != "wal":
            c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # Another process is mid-switch; it will land in WAL either way.
        pass
    # Key off the tables, not the file: a bare sqlite3.connect elsewhere creates an
    # empty file, after which a file-existence check would skip init forever.
    have_tables = c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='processed_matches'"
    ).fetchone() is not None
    if not have_tables:
        _init_schema(c)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        c.commit()
    else:
        _migrate(c)
    return c


def init_db():
    """Create or migrate the schema up front, so parallel workers don't race on it."""
    _conn().close()


# ── reads ─────────────────────────────────────────────────────────────────────

def is_processed(match_id: str) -> bool:
    c = _conn()
    try:
        row = c.execute(
            "SELECT 1 FROM processed_matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        return row is not None
    finally:
        c.close()


def stats() -> dict:
    c = _conn()
    try:
        return {
            "matches_processed": c.execute("SELECT COUNT(*) FROM processed_matches").fetchone()[0],
            "total_events":      c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "total_states":      c.execute("SELECT COUNT(*) FROM round_states").fetchone()[0],
            "events_by_type":    dict(c.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
            ).fetchall()),
        }
    finally:
        c.close()


# ── writes ────────────────────────────────────────────────────────────────────

_EVENTS_SQL = """
        INSERT INTO events
            (match_id, map, round_num, player_side, event_type,
             time_into_round, situation, action, outcome, round_won)
        VALUES
            (:match_id, :map, :round_num, :player_side, :event_type,
             :time_into_round, :situation, :action, :outcome, :round_won)
    """

_STATES_SQL = """
        INSERT INTO round_states
            (match_id, map, round_num, tick, time_into_round_s, time_remaining_s,
             post_plant, alive_ct, alive_t, total_hp_ct, total_hp_t,
             total_armor_ct, total_armor_t, helmets_ct, helmets_t, ct_spread, t_spread,
             has_defuser, equip_value_ct, equip_value_t,
             smokes_ct, smokes_t, flashes_ct, flashes_t,
             he_ct, he_t, molotovs_ct, molotovs_t, active_smokes, active_infernos,
             active_smokes_xy, active_infernos_xy,
             site_smoked, site_on_fire, min_dist_ct_to_bomb, min_dist_t_to_bomb,
             ct_spotted_count, t_spotted_count, ct_heard_enemy, t_heard_enemy, round_won_ct,
             state_kind)
        VALUES
            (:match_id, :map, :round_num, :tick, :time_into_round_s, :time_remaining_s,
             :post_plant, :alive_ct, :alive_t, :total_hp_ct, :total_hp_t,
             :total_armor_ct, :total_armor_t, :helmets_ct, :helmets_t, :ct_spread, :t_spread,
             :has_defuser, :equip_value_ct, :equip_value_t,
             :smokes_ct, :smokes_t, :flashes_ct, :flashes_t,
             :he_ct, :he_t, :molotovs_ct, :molotovs_t, :active_smokes, :active_infernos,
             :active_smokes_xy, :active_infernos_xy,
             :site_smoked, :site_on_fire, :min_dist_ct_to_bomb, :min_dist_t_to_bomb,
             :ct_spotted_count, :t_spotted_count, :ct_heard_enemy, :t_heard_enemy, :round_won_ct,
             :state_kind)
    """


def _event_params(e: dict) -> dict:
    return {**e,
            "situation": _dumps(e.get("situation", {})),
            "action":    _dumps(e.get("action",    {})),
            "outcome":   _dumps(e.get("outcome",   {}))}


def _state_params(s: dict) -> dict:
    return {**s,
            "state_kind":         s.get("state_kind", STATE_GRID),
            "active_smokes_xy":   _dumps(s.get("active_smokes_xy",   [])),
            "active_infernos_xy": _dumps(s.get("active_infernos_xy", []))}


def store_match(match_id: str, events: list[dict], states: list[dict],
                map_name: str = None, demo_url: str = None) -> None:
    """Write one match's events, states and processed marker in ONE transaction.

    Replace rather than append, so re-processing a match is idempotent even if a
    previous attempt died partway through. Dedup is check-then-act and the row
    tables have no uniqueness constraint, so this is the only thing standing between
    a crash and a double-inserted match.
    """
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        c.execute("DELETE FROM events       WHERE match_id = ?", (match_id,))
        c.execute("DELETE FROM round_states WHERE match_id = ?", (match_id,))
        if events:
            c.executemany(_EVENTS_SQL, [_event_params(e) for e in events])
        if states:
            c.executemany(_STATES_SQL, [_state_params(s) for s in states])
        c.execute(
            "INSERT OR REPLACE INTO processed_matches (match_id, map, demo_url) VALUES (?, ?, ?)",
            (match_id, map_name, demo_url),
        )
        c.commit()
        print(f"  stored {len(events)} events + {len(states)} states (one transaction)", flush=True)
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()
