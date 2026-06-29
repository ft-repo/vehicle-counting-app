from pathlib import Path

import tools.vca_store as store


def test_insert_and_count_by_class(tmp_path):
    db = tmp_path / "t.duckdb"
    con = store.connect(db)

    store.insert_crossing(con, "2026-04-22 14:02:07", 6, "bike", "in", "Lane 3")
    store.insert_crossing(con, "2026-04-22 14:02:08", 8, "bike", "in", "Lane 3")
    store.insert_crossing(con, "2026-04-22 14:02:09", 9, "car", "out", "Lane 2")

    rows = store.counts_by_class(con)
    assert ("bike", "in", 2) in rows
    assert ("car", "out", 1) in rows


def test_schema_is_idempotent(tmp_path):
    db = tmp_path / "t.duckdb"
    store.connect(db).close()
    con = store.connect(db)  # second connect must not error or wipe data
    store.insert_crossing(con, "2026-04-22 14:02:07", 6, "bike", "in", "Lane 3")
    assert store.counts_by_class(con) == [("bike", "in", 1)]


def test_camera_filter(tmp_path):
    con = store.connect(tmp_path / "t.duckdb")
    store.insert_crossing(con, "2026-04-22 14:02:07", 6, "bike", "in", "Lane 3", camera_id="A")
    store.insert_crossing(con, "2026-04-22 14:02:08", 7, "car", "in", "Lane 3", camera_id="B")
    assert store.counts_by_class(con, camera_id="A") == [("bike", "in", 1)]


def test_insert_eval_run(tmp_path):
    con = store.connect(tmp_path / "t.duckdb")
    store.insert_eval_run(con, "2026-06-22 10:37:00", "counting", "run6",
                          "overall_abs_count_error_rate", 0.1171875, '{"total_counted": 127}')
    val = con.execute("SELECT kind, model_version, metric, value FROM eval_runs").fetchall()
    assert val == [("counting", "run6", "overall_abs_count_error_rate", 0.1171875)]
