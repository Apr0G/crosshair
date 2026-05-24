"""
SQLite persistence layer.

Three tables:
  processed_matches — dedup index of match_ids that have been parsed
  events            — every player action (kill, util throw, buy, rotation, bomb)
                      with WP-before / WP-after / impact filled by score_impact.py
  round_states      — periodic team-state snapshots (~1 Hz) used to train the WP model
"""
import sqlite3
import json
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "crosshair.db"


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_Encoder)


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    needs_init = not DB_PATH.exists()
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    if needs_init:
        _init_schema(c)
    return c


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
            round_won       INTEGER
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
            round_won_ct         INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_states_match ON round_states(match_id);
        CREATE INDEX IF NOT EXISTS idx_states_won   ON round_states(round_won_ct);
    """)


def init_db():
    c = _conn()
    try:
        _init_schema(c)
    finally:
        c.close()


def is_processed(match_id: str) -> bool:
    c = _conn()
    try:
        row = c.execute(
            "SELECT 1 FROM processed_matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        return row is not None
    finally:
        c.close()


def mark_processed(match_id: str, map_name: str = None, demo_url: str = None):
    c = _conn()
    try:
        c.execute(
            "INSERT OR IGNORE INTO processed_matches (match_id, map, demo_url) VALUES (?, ?, ?)",
            (match_id, map_name, demo_url)
        )
        c.commit()
    finally:
        c.close()


def insert_events(events: list[dict], batch_size: int = 500):
    if not events:
        return
    sql = """
        INSERT INTO events
            (match_id, map, round_num, player_side, event_type,
             time_into_round, situation, action, outcome, round_won)
        VALUES
            (:match_id, :map, :round_num, :player_side, :event_type,
             :time_into_round, :situation, :action, :outcome, :round_won)
    """
    c = _conn()
    try:
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            c.executemany(sql, [
                {**e,
                 "situation": _dumps(e.get("situation", {})),
                 "action":    _dumps(e.get("action",    {})),
                 "outcome":   _dumps(e.get("outcome",   {}))}
                for e in batch
            ])
            c.commit()
            print(f"  stored {min(i + batch_size, len(events))}/{len(events)} events", flush=True)
    finally:
        c.close()


def insert_round_states(states: list[dict], batch_size: int = 500):
    if not states:
        return
    sql = """
        INSERT INTO round_states
            (match_id, map, round_num, tick, time_into_round_s, time_remaining_s,
             post_plant, alive_ct, alive_t, total_hp_ct, total_hp_t,
             total_armor_ct, total_armor_t, helmets_ct, helmets_t, ct_spread, t_spread,
             has_defuser, equip_value_ct, equip_value_t,
             smokes_ct, smokes_t, flashes_ct, flashes_t,
             he_ct, he_t, molotovs_ct, molotovs_t, active_smokes, active_infernos,
             active_smokes_xy, active_infernos_xy,
             site_smoked, site_on_fire, min_dist_ct_to_bomb, min_dist_t_to_bomb,
             ct_spotted_count, t_spotted_count, ct_heard_enemy, t_heard_enemy, round_won_ct)
        VALUES
            (:match_id, :map, :round_num, :tick, :time_into_round_s, :time_remaining_s,
             :post_plant, :alive_ct, :alive_t, :total_hp_ct, :total_hp_t,
             :total_armor_ct, :total_armor_t, :helmets_ct, :helmets_t, :ct_spread, :t_spread,
             :has_defuser, :equip_value_ct, :equip_value_t,
             :smokes_ct, :smokes_t, :flashes_ct, :flashes_t,
             :he_ct, :he_t, :molotovs_ct, :molotovs_t, :active_smokes, :active_infernos,
             :active_smokes_xy, :active_infernos_xy,
             :site_smoked, :site_on_fire, :min_dist_ct_to_bomb, :min_dist_t_to_bomb,
             :ct_spotted_count, :t_spotted_count, :ct_heard_enemy, :t_heard_enemy, :round_won_ct)
    """
    c = _conn()
    try:
        for i in range(0, len(states), batch_size):
            batch = [
                {**s,
                 "active_smokes_xy":   _dumps(s.get("active_smokes_xy",   [])),
                 "active_infernos_xy": _dumps(s.get("active_infernos_xy", []))}
                for s in states[i:i + batch_size]
            ]
            c.executemany(sql, batch)
            c.commit()
            print(f"  stored {min(i + batch_size, len(states))}/{len(states)} round states", flush=True)
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
