import json
import pytest
import tools.vca_store as store
import tools.migrate_to_duckdb as mig


def _fixture_csv(tmp_path):
    p = tmp_path / "vehicle_counts.csv"
    p.write_text(
        "timestamp,track_id,class,direction,lane\n"
        "2026-04-22 14:02:07,6,bike,in,Lane 3\n"
        "2026-04-22 14:02:07,8,pickup,out,Lane 4\n"
    )
    return p


def test_migrate_crossings_inserts_rows(tmp_path):
    con = store.connect(tmp_path / "t.duckdb")
    n = mig.migrate_crossings(con, _fixture_csv(tmp_path), source_file="vehicle_counts.csv")
    assert n == 2
    assert ("bike", "in", 1) in store.counts_by_class(con)
    assert ("pickup", "out", 1) in store.counts_by_class(con)


def test_migrate_crossings_is_idempotent(tmp_path):
    con = store.connect(tmp_path / "t.duckdb")
    csv = _fixture_csv(tmp_path)
    mig.migrate_crossings(con, csv, source_file="vehicle_counts.csv")
    mig.migrate_crossings(con, csv, source_file="vehicle_counts.csv")  # twice
    total = con.execute("SELECT COUNT(*) FROM crossings").fetchone()[0]
    assert total == 2  # not 4


def test_migrate_eval_json_counting(tmp_path):
    """Test migrate_eval_json with counting results."""
    con = store.connect(tmp_path / "t.duckdb")
    json_path = tmp_path / "counting.json"
    json_path.write_text(json.dumps({"overall_abs_count_error_rate": 0.5, "total_counted": 10}))

    mig.migrate_eval_json(con, json_path, "counting", "run6")

    rows = con.execute("SELECT kind, model_version, metric, value FROM eval_runs").fetchall()
    assert len(rows) == 1
    kind, model_version, metric, value = rows[0]
    assert kind == "counting"
    assert model_version == "run6"
    assert metric == "overall_abs_count_error_rate"
    assert value == 0.5


def test_migrate_eval_json_idempotent(tmp_path):
    """Test that migrate_eval_json is idempotent."""
    con = store.connect(tmp_path / "t.duckdb")
    json_path = tmp_path / "counting.json"
    json_path.write_text(json.dumps({"overall_abs_count_error_rate": 0.5, "total_counted": 10}))

    mig.migrate_eval_json(con, json_path, "counting", "run6")
    mig.migrate_eval_json(con, json_path, "counting", "run6")  # twice

    total = con.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    assert total == 1  # not 2


def test_migrate_eval_json_missing_metric_raises(tmp_path):
    """Test that missing metric key raises KeyError."""
    con = store.connect(tmp_path / "t.duckdb")
    json_path = tmp_path / "counting.json"
    json_path.write_text(json.dumps({"total_counted": 10}))  # missing metric

    with pytest.raises(KeyError):
        mig.migrate_eval_json(con, json_path, "counting", "run6")
