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
