"""One-shot, idempotent migration of historical VCA outputs into DuckDB (M0).

Usage:
    python tools/migrate_to_duckdb.py            # migrate the standard files
    python tools/migrate_to_duckdb.py --db data/counts.duckdb
"""
import argparse
import csv
import json
from pathlib import Path

import tools.vca_store as store

ROOT = Path(__file__).resolve().parent.parent


def migrate_crossings(con, csv_path, source_file=None):
    """Load a vehicle_counts.csv into crossings. Idempotent per source_file."""
    csv_path = Path(csv_path)
    label = source_file or csv_path.name
    con.execute("DELETE FROM crossings WHERE source_file = ?", [label])
    n = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            store.insert_crossing(
                con,
                row["timestamp"],
                int(row["track_id"]),
                row["class"],
                row["direction"],
                row["lane"],
                source_file=label,
            )
            n += 1
    return n


def migrate_eval_json(con, json_path, kind, model_version):
    """Load a counting/val results JSON into eval_runs. Idempotent per (kind, model)."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    con.execute("DELETE FROM eval_runs WHERE kind = ? AND model_version = ?",
                [kind, model_version])
    if kind == "counting":
        metric, value = "overall_abs_count_error_rate", data.get("overall_abs_count_error_rate")
        ts = "1970-01-01 00:00:00"  # counting JSON has no timestamp; set on re-export
    elif kind == "val":
        metric, value = "overall_map50", data.get("overall_map50")
        ts = data.get("timestamp", "1970-01-01 00:00:00").replace("T", " ").replace("Z", "")
    else:
        raise ValueError(f"Unknown eval kind: {kind!r}")
    if value is None:
        raise KeyError(f"{json_path}: missing metric '{metric}' for kind '{kind}'")
    store.insert_eval_run(con, ts, kind, model_version, metric, value, json.dumps(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(store.DEFAULT_DB))
    args = ap.parse_args()
    con = store.connect(args.db)

    total = 0
    for rel in ("vehicle_counts.csv", "logs/vehicle_counts.csv"):
        p = ROOT / rel
        if p.exists():
            total += migrate_crossings(con, p, source_file=rel)
            print(f"  crossings: {rel}")

    for rel, kind, model in (
        ("counting_eval_results.json", "counting", "run6"),
        ("val_results.json", "val", "run4"),
    ):
        p = ROOT / rel
        if p.exists():
            migrate_eval_json(con, p, kind, model)
            print(f"  eval: {rel} ({kind}/{model})")

    print(f"Done. {total} crossings in store.")
    print("By class:", store.counts_by_class(con))


if __name__ == "__main__":
    main()
