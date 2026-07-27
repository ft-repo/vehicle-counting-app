#!/usr/bin/env python3
"""
cluster_cameras.py — local, offline viewpoint clustering for the camera-install audit.

PURPOSE
    Group 1000+ installed cameras into a small number of *viewpoint archetypes*
    so run7 training can sample a representative subset instead of auditing every
    camera one by one. Two cameras with the same height / tilt / lens / road
    orientation are the SAME problem to the detector regardless of location.

PRIVACY (read this)
    - Runs 100% locally. No network calls. Nothing is sent anywhere.
    - Columns that look identifying (ip, gps/lat/lng, address, location, site,
      customer, name, serial, mac, url, host, phone) are NEVER used as features
      and their VALUES are NEVER printed. They are quarantined on load.
    - To the terminal this script prints ONLY aggregates (cluster sizes, mean
      tilt, top lens types, etc.) — safe to paste back into a chat.
    - The per-camera sample list (which cameras to actually label) is written to
      a LOCAL csv on your machine and NOT printed. Keep that file local.

USAGE
    conda activate <your-env>
    cd ~/Desktop/vehicle-counting-app/camera_audit
    # put the 4 downloaded .xlsx (or .csv) files in this folder, then:

    # 1) See what columns exist and how each is classified (no clustering yet):
    python cluster_cameras.py --mode discover

    # 2) Once discovery confirms viewpoint columns exist, cluster:
    python cluster_cameras.py --mode cluster --k 8
    #    (omit --k to auto-pick k by silhouette score)

DEPS
    pandas, openpyxl, scikit-learn, numpy  (all already in the ultralytics env)
"""

from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Column classification by header keywords. Normalised (lowercased, non-alnum
# stripped) substring match. Tune these lists after `--mode discover` shows you
# the real headers — that's the one manual step.
# ----------------------------------------------------------------------------

# NEVER used as a feature, NEVER printed. Identifying / sensitive.
SENSITIVE_KEYS = [
    "ip", "lat", "lng", "lon", "long", "gps", "coord", "location", "loc",
    "address", "addr", "site", "customer", "client", "owner", "name",
    "serial", "sn", "mac", "url", "link", "host", "hostname", "phone",
    "tel", "contact", "province", "district", "amphoe", "tambon", "zip",
    "postal", "street", "road name", "rtsp", "password", "passwd", "user",
]

# Numeric viewpoint features — standardised then clustered.
NUMERIC_VIEWPOINT_KEYS = {
    "height":      ["height", "mount height", "pole", "elevation", "high", "meter", "metre"],
    "tilt":        ["tilt", "pitch", "angle", "downtilt", "depression"],
    "pan":         ["pan", "azimuth", "bearing", "direction deg"],
    "fov":         ["fov", "field of view", "focal", "zoom", "mm"],
}

# Categorical viewpoint features — one-hot encoded then clustered.
CATEGORICAL_VIEWPOINT_KEYS = {
    "model":       ["model", "camera model", "device", "type", "cam type", "product"],
    "lens":        ["lens", "varifocal", "fixed lens", "optics"],
    "orientation": ["orientation", "facing", "view", "lane dir", "road dir",
                    "approach", "flow"],
    "resolution":  ["resolution", "res", "mp", "megapixel", "1080", "4k", "pixels"],
    "mounting":    ["mount", "gantry", "overhead", "roadside", "median", "position"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _matches(header: str, keys: list[str]) -> bool:
    h = _norm(header)
    return any(_norm(k) in h for k in keys)


def classify_column(header: str) -> tuple[str, str | None]:
    """Return (kind, feature_name). kind in {sensitive, numeric, categorical, unknown}."""
    if _matches(header, SENSITIVE_KEYS):
        return "sensitive", None
    for feat, keys in NUMERIC_VIEWPOINT_KEYS.items():
        if _matches(header, keys):
            return "numeric", feat
    for feat, keys in CATEGORICAL_VIEWPOINT_KEYS.items():
        if _matches(header, keys):
            return "categorical", feat
    return "unknown", None


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_all(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "*.xlsx"))) + \
            sorted(glob.glob(str(folder / "*.csv")))
    if not files:
        sys.exit(f"[FATAL] no .xlsx/.csv files in {folder}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f)
        except Exception as e:
            print(f"  [WARN] could not read {Path(f).name}: {e}")
            continue
        df["__source_file"] = Path(f).name
        frames.append(df)
        print(f"  loaded {Path(f).name}: {len(df)} rows, {len(df.columns)} cols")
    if not frames:
        sys.exit("[FATAL] nothing readable.")
    # union of columns; rows stacked
    return pd.concat(frames, ignore_index=True, sort=False)


# ----------------------------------------------------------------------------
# Discover
# ----------------------------------------------------------------------------

def discover(df: pd.DataFrame) -> None:
    print("\n=== COLUMN CLASSIFICATION ===")
    buckets = {"numeric": [], "categorical": [], "sensitive": [], "unknown": []}
    feat_map = {}
    for col in df.columns:
        if col == "__source_file":
            continue
        kind, feat = classify_column(col)
        buckets[kind].append((col, feat))
        if feat:
            feat_map.setdefault(feat, []).append(col)

    for kind in ("numeric", "categorical", "unknown", "sensitive"):
        items = buckets[kind]
        print(f"\n  [{kind.upper()}]  ({len(items)})")
        for col, feat in items:
            tag = f" -> viewpoint:{feat}" if feat else ""
            # NOTE: we print only the HEADER, never the values.
            print(f"    - {col}{tag}")

    n_feat = len(buckets["numeric"]) + len(buckets["categorical"])
    print("\n=== VERDICT ===")
    if n_feat == 0:
        print("  ✗ No viewpoint columns detected. Clustering by angle is NOT")
        print("    possible with these sheets — they likely hold only location/")
        print("    identity data. Options: (a) add viewpoint fields, or (b) infer")
        print("    angle from sample frames instead of the spreadsheet.")
    else:
        print(f"  ✓ {n_feat} viewpoint column(s) found across "
              f"{len(feat_map)} feature group(s): {', '.join(feat_map)}")
        print("    Check the UNKNOWN list above — if a real viewpoint column")
        print("    landed there, add its header keyword to the *_VIEWPOINT_KEYS")
        print("    dict at the top of this script, then re-run discover.")
        print("    When it looks right:  python cluster_cameras.py --mode cluster")


# ----------------------------------------------------------------------------
# Cluster
# ----------------------------------------------------------------------------

def build_features(df: pd.DataFrame):
    """Return (X, feature_labels, numeric_cols, categorical_cols) — sensitive cols excluded."""
    num_cols, cat_cols = [], []
    for col in df.columns:
        if col == "__source_file":
            continue
        kind, _ = classify_column(col)
        if kind == "numeric":
            num_cols.append(col)
        elif kind == "categorical":
            cat_cols.append(col)

    parts, labels = [], []

    for col in num_cols:
        v = pd.to_numeric(df[col], errors="coerce")
        if v.notna().sum() == 0:
            continue
        v = v.fillna(v.median())
        std = v.std() or 1.0
        parts.append(((v - v.mean()) / std).to_numpy().reshape(-1, 1))
        labels.append(f"num:{col}")

    for col in cat_cols:
        s = df[col].astype("string").fillna("NA")
        # cap cardinality so a stray free-text column can't explode the matrix
        top = s.value_counts().head(12).index
        s = s.where(s.isin(top), other="OTHER")
        dummies = pd.get_dummies(s, prefix=_norm(col))
        parts.append(dummies.to_numpy())
        labels.extend(dummies.columns.tolist())

    if not parts:
        sys.exit("[FATAL] no usable viewpoint features — run --mode discover first.")
    X = np.hstack(parts).astype(float)
    return X, labels, num_cols, cat_cols


def pick_k(X, kmin=3, kmax=12):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_s = kmin, -1.0
    kmax = min(kmax, len(X) - 1)
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(X, km.labels_)
        print(f"    k={k}: silhouette={s:.3f}")
        if s > best_s:
            best_k, best_s = k, s
    print(f"  -> auto-selected k={best_k} (silhouette={best_s:.3f})")
    return best_k


def cluster(df: pd.DataFrame, k: int | None, folder: Path,
            per_cluster_sample: int) -> None:
    from sklearn.cluster import KMeans

    X, labels, num_cols, cat_cols = build_features(df)
    print(f"\n  feature matrix: {X.shape[0]} cameras x {X.shape[1]} dims")
    print(f"  numeric viewpoint cols: {num_cols or '(none)'}")
    print(f"  categorical viewpoint cols: {cat_cols or '(none)'}")

    if k is None:
        print("\n=== auto-selecting k by silhouette ===")
        k = pick_k(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    df = df.copy()
    df["__cluster"] = km.labels_

    print(f"\n=== {k} VIEWPOINT ARCHETYPES (aggregates only) ===")
    for c in range(k):
        sub = df[df["__cluster"] == c]
        print(f"\n  Cluster {c}: {len(sub)} cameras "
              f"({100*len(sub)/len(df):.1f}%)")
        for col in num_cols:
            v = pd.to_numeric(sub[col], errors="coerce")
            if v.notna().sum():
                print(f"    {col}: mean={v.mean():.2f}  "
                      f"[{v.min():.1f}–{v.max():.1f}]")
        for col in cat_cols:
            vc = sub[col].astype("string").value_counts().head(3)
            top = ", ".join(f"{idx}={n}" for idx, n in vc.items())
            print(f"    {col}: {top}")

    # ---- per-cluster sample list -> LOCAL FILE ONLY (identifiers stay local) ----
    sample = (df.groupby("__cluster", group_keys=False)
                .apply(lambda g: g.sample(min(per_cluster_sample, len(g)),
                                          random_state=0)))
    out = folder / "sampled_cameras.csv"
    sample.to_csv(out, index=False)
    print(f"\n=== SAMPLING PLAN ===")
    print(f"  Wrote {len(sample)} cameras "
          f"({per_cluster_sample}/cluster) -> {out}")
    print( "  ^ this file has full identifying columns. It stays on your Mac.")
    print( "    Label THESE cameras for run7 (weight the confused pairs:")
    print( "    car/pickup, truck/bus, taxi/car). Hold out a few clusters'")
    print( "    cameras entirely as an unseen-viewpoint validation set.")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=".",
                    help="folder with the downloaded .xlsx/.csv (default: cwd)")
    ap.add_argument("--mode", choices=["discover", "cluster"], default="discover")
    ap.add_argument("--k", type=int, default=None,
                    help="number of clusters (omit to auto-pick by silhouette)")
    ap.add_argument("--sample", type=int, default=25,
                    help="cameras to sample per cluster for labeling (default 25)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    print(f"folder: {folder}")
    df = load_all(folder)
    print(f"total: {len(df)} camera rows\n")

    if args.mode == "discover":
        discover(df)
    else:
        cluster(df, args.k, folder, args.sample)


if __name__ == "__main__":
    main()
