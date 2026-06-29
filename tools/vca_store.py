"""DuckDB store for VCA counts + eval history (roadmap M0).

Single responsibility: persistence. The live counter (vehicle_counter.py) is
NOT modified here — it can call insert_crossing() in a later milestone.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "counts.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crossings (
    ts            TIMESTAMP,
    track_id      INTEGER,
    class         VARCHAR,
    direction     VARCHAR,
    lane          VARCHAR,
    camera_id     VARCHAR,
    model_version VARCHAR,
    source_file   VARCHAR
);
CREATE TABLE IF NOT EXISTS eval_runs (
    ts            TIMESTAMP,
    kind          VARCHAR,
    model_version VARCHAR,
    metric        VARCHAR,
    value         DOUBLE,
    payload       VARCHAR
);
"""


def connect(db_path=DEFAULT_DB):
    """Open (creating if needed) the DuckDB store and ensure the schema."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_SCHEMA)
    return con


def insert_crossing(con, ts, track_id, cls, direction, lane,
                    camera_id=None, model_version=None, source_file=None):
    """Record one lane-crossing event."""
    con.execute(
        "INSERT INTO crossings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [ts, track_id, cls, direction, lane, camera_id, model_version, source_file],
    )


def insert_eval_run(con, ts, kind, model_version, metric, value, payload):
    """Record one eval result (kind = 'counting' | 'val')."""
    con.execute(
        "INSERT INTO eval_runs VALUES (?, ?, ?, ?, ?, ?)",
        [ts, kind, model_version, metric, value, payload],
    )


def counts_by_class(con, camera_id=None):
    """Return [(class, direction, n)] grouped, optionally filtered by camera."""
    q = "SELECT class, direction, COUNT(*) AS n FROM crossings"
    params = []
    if camera_id is not None:
        q += " WHERE camera_id = ?"
        params.append(camera_id)
    q += " GROUP BY class, direction ORDER BY class, direction"
    return con.execute(q, params).fetchall()
