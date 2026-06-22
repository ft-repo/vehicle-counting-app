# archive/

Retired scripts kept for reference only. **Do not run these.**

- `merge_dataset.py` — RETIRED 2026-05-23. Stale 11-class person-first order.
  Superseded by `model_compare/build_split.py` (canonical, leak-safe, flat-export aware).
- `export_dataset.py` — RETIRED 2026-05-23. Same reason. Use `build_split.py`.
- `compare.py` — dead bench tool. Hardcoded stale `counting_app/old_win_code/` paths; unreferenced.
- `dashboard.py` — dead bench tool. Same stale paths; unreferenced.

The split pipeline has exactly one supported entry point: `model_compare/build_split.py`.
