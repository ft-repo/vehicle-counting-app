#!/usr/bin/env python3
"""
discover_angles.py — Path B: discover how many distinct camera VIEWPOINTS exist,
from the images themselves, with no metadata and no human checking one by one.

WHAT IT DOES
    1. Reads a LOCAL list of counting-camera stream URLs.
    2. Grabs ONE frame from each camera (concurrent, with timeouts/retries).
    3. Embeds each frame into a vector that describes what the camera SEES
       (scene layout / viewpoint), not just object content.
    4. Clusters the vectors. The number of clusters ≈ the number of distinct
       viewpoints. Auto-picks the cluster count by silhouette score.
    5. Saves a representative thumbnail per cluster so you can eyeball the
       archetypes — instead of eyeballing 1000 cameras.

PRIVACY (read this)
    - Runs 100% locally. Connects to cameras FROM your Mac. The only outbound
      traffic is to your own cameras (and, ONCE, a public model-weights download
      if torchvision is used — that is public weights coming IN, not your data
      going out; use --embedder classic to stay fully offline).
    - Camera URLs/IPs are read from a local file, are NEVER printed, and NEVER
      enter any chat. Cameras are referred to by index + short hash only.
    - Frames, thumbnails, and the camera↔cluster map are written to local files.
    - Terminal output is aggregates only (counts, cluster sizes) — safe to paste.

CAMERA LIST FILE  (any of these, in this folder)
      - cameras.xlsx : Google Sheet downloaded as Excel — reads ALL tabs at once
                       and sweeps every cell for rtsp/http URLs. Best for the
                       multi-tab (per-site / per-year) install workbook.
      - cameras.csv  : one tab exported as CSV; also cell-swept for URLs.
      - cameras.txt  : one URL per line (blank lines / # comments ignored).
    If the sheet has only bare IPs (no rtsp/http links), the script stops and
    tells you — we then build the stream URL from the IP + your URL pattern.
    Example URL:  rtsp://user:pass@10.0.0.12/axis-media/media.amp

USAGE
    conda activate <your-env>
    cd ~/Desktop/vehicle-counting-app/camera_audit

    # smoke test on the first 20 cameras:
    python discover_angles.py --cameras cameras.txt --limit 20

    # full run, auto-pick number of viewpoints:
    python discover_angles.py --cameras cameras.txt

    # fully offline (no weights download), force cluster count:
    python discover_angles.py --cameras cameras.txt --embedder classic --k 12

DEPS
    opencv-python, numpy, scikit-learn  (required)
    torch + torchvision                 (optional, for --embedder torch/auto)
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# RTSP robustness: TCP transport + socket timeout (microseconds). Must be set
# BEFORE cv2 touches FFMPEG. 8s open/read timeout.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;8000000|max_delay;5000000",
)

import numpy as np  # noqa: E402
import cv2          # noqa: E402


# ----------------------------------------------------------------------------
# Camera list (local, never printed)
# ----------------------------------------------------------------------------

URL_COL_HINTS = ("url", "rtsp", "source", "stream", "address", "uri", "link")

# Tabs across years have different layouts, so rather than guess a column we
# sweep every cell for things that look like a stream URL or a bare IP.
URL_RE = re.compile(r"\b(?:rtsp|rtsps|http|https)://[^\s,;'\"]+", re.I)
IP_RE  = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _scan_cells(values) -> tuple[list[str], list[str]]:
    """Return (urls, bare_ips) found anywhere in an iterable of cell strings."""
    urls, ips = [], []
    for v in values:
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        m = URL_RE.findall(s)
        if m:
            urls.extend(m)
        elif IP_RE.search(s) and "." in s:
            # a cell that's basically just an IP (not part of a URL we caught)
            for ip in IP_RE.findall(s):
                ips.append(ip)
    return urls, ips


def read_camera_list(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"[FATAL] camera list not found: {path}\n"
                 f"        Download the Google Sheet as .xlsx (all tabs) into\n"
                 f"        this folder, or make a cameras.txt (one URL per line).")

    raw_urls: list[str] = []
    all_ips: list[str] = []
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls", ".ods"):
        import pandas as pd
        engine = "odf" if suffix == ".ods" else None
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
        print(f"  workbook has {len(sheets)} sheet(s): "
              f"{', '.join(list(sheets)[:8])}{' ...' if len(sheets) > 8 else ''}")
        for name, df in sheets.items():
            cells = df.astype(str).to_numpy().ravel().tolist()
            u, ip = _scan_cells(cells)
            raw_urls += u
            all_ips += ip
    elif suffix == ".csv":
        with open(path, newline="") as f:
            cells = [c for row in csv.reader(f) for c in row]
        raw_urls, all_ips = _scan_cells(cells)
    else:  # .txt — one entry per line
        for line in path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                raw_urls.append(s)

    # de-dup, preserve order
    seen, urls = set(), []
    for u in raw_urls:
        if u not in seen:
            seen.add(u)
            urls.append(u)

    if not urls:
        uniq_ips = sorted(set(all_ips))
        if uniq_ips:
            sys.exit(
                f"[FATAL] found {len(uniq_ips)} bare IP(s) but NO full stream URLs.\n"
                f"        The sheet has IPs, not rtsp/http stream links, so frames\n"
                f"        can't be grabbed directly. Tell me your camera URL pattern\n"
                f"        (from the counting app config) and I'll add IP->URL build.")
        sys.exit("[FATAL] no camera URLs or IPs found in the file.")
    print(f"  parsed {len(urls)} unique stream URL(s)"
          + (f"; also saw {len(set(all_ips))} bare IP(s) ignored" if all_ips else ""))
    return urls


def cam_id(url: str) -> str:
    """Stable non-identifying handle for a camera (no IP leaks to logs)."""
    return hashlib.sha1(url.encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------
# Frame grab
# ----------------------------------------------------------------------------

def grab_frame(url: str, warmup: int = 8, retries: int = 2):
    """Open stream, discard a few frames to let it stabilise, return one BGR frame."""
    for _ in range(retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            try:  # not all builds expose these props
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
            except Exception:
                pass
            if not cap.isOpened():
                continue
            frame = None
            for _ in range(warmup):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is not None:
                return frame
        finally:
            cap.release()
    return None


def capture_all(urls: list[str], out_dir: Path, workers: int, timeout: int):
    """Returns dict {idx: (url, frame_path)} for successful grabs."""
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    ok: dict[int, tuple[str, Path]] = {}
    fails = 0

    def task(i_url):
        i, url = i_url
        frame = grab_frame(url)
        if frame is None:
            return i, url, None
        p = frame_dir / f"cam_{i:04d}_{cam_id(url)}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return i, url, p

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, iu) for iu in enumerate(urls)]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                i, url, p = fut.result(timeout=timeout)
            except Exception:
                fails += 1
                continue
            if p is None:
                fails += 1
            else:
                ok[i] = (url, p)
            if n % 25 == 0 or n == len(urls):
                print(f"  captured {len(ok)}/{n} (failed {fails}) ...")
    print(f"  DONE capturing: {len(ok)} ok, {fails} unreachable, "
          f"of {len(urls)} cameras")
    return ok


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------

def classic_descriptor(path: Path) -> np.ndarray:
    """Offline, dependency-free viewpoint descriptor: spatial gradient-orientation
    histogram + coarse intensity map. Captures scene LAYOUT (where the road is,
    how steep the view is) rather than fine content."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi)  # 0..2pi
    feats = []
    bins = 8
    for r in range(4):           # 4x4 spatial grid of orientation histograms
        for c in range(4):
            m = mag[r*32:(r+1)*32, c*32:(c+1)*32]
            a = ang[r*32:(r+1)*32, c*32:(c+1)*32]
            h, _ = np.histogram(a, bins=bins, range=(0, 2*np.pi), weights=m)
            feats.append(h)
    coarse = cv2.resize(img, (8, 8)).astype(np.float32).flatten()  # layout/intensity
    v = np.concatenate(feats + [coarse]).astype(np.float32)
    n = np.linalg.norm(v) or 1.0
    return v / n


def torch_embedder():
    """Return a fn(path)->vector using a pretrained MobileNetV3 backbone."""
    import torch
    import torchvision
    from torchvision import transforms
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier = torch.nn.Identity()
    model.eval()
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def embed(path: Path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = tf(img).unsqueeze(0)
        v = model(x).squeeze(0).numpy().astype(np.float32)
        n = np.linalg.norm(v) or 1.0
        return v / n

    return embed


def build_embeddings(ok: dict, mode: str):
    embed = None
    if mode in ("auto", "torch"):
        try:
            embed = torch_embedder()
            print("  embedder: torchvision MobileNetV3 (pretrained)")
        except Exception as e:
            if mode == "torch":
                sys.exit(f"[FATAL] torch embedder unavailable: {e}")
            print(f"  torch unavailable ({e}); falling back to classic descriptor")
    if embed is None:
        embed = classic_descriptor
        print("  embedder: classic gradient-layout descriptor (offline)")

    idxs, vecs = [], []
    for i, (_url, p) in sorted(ok.items()):
        v = embed(p)
        if v is not None:
            idxs.append(i)
            vecs.append(v)
    if not vecs:
        sys.exit("[FATAL] no frames could be embedded.")
    return idxs, np.vstack(vecs)


# ----------------------------------------------------------------------------
# Cluster
# ----------------------------------------------------------------------------

def pick_k(X, kmin, kmax):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    kmax = min(kmax, len(X) - 1)
    best_k, best_s = kmin, -1.0
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(X, km.labels_)
        print(f"    k={k}: silhouette={s:.3f}")
        if s > best_s:
            best_k, best_s = k, s
    print(f"  -> {best_k} viewpoints best explains the data "
          f"(silhouette={best_s:.3f})")
    return best_k


def cluster_and_report(idxs, X, ok, out_dir, k, kmax, sample_per_cluster):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    n = len(idxs)
    if n < 3:
        print(f"\n  only {n} frame(s) captured — too few to cluster into "
              f"viewpoints. Capture more cameras first (raise --limit / fix "
              f"reachability), then re-run.")
        return
    if k is None:
        print("\n=== choosing number of viewpoints (silhouette) ===")
        k = pick_k(Xs, 3, kmax)
    k = max(2, min(k, n))  # never ask for more clusters than frames

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    labels = km.labels_

    rep_dir = out_dir / "viewpoint_reps"
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {k} DISTINCT VIEWPOINTS FOUND ===")
    rows = []
    for c in range(k):
        members = [j for j, lab in enumerate(labels) if lab == c]
        if not members:
            continue
        # representative = frame closest to cluster centroid
        sub = Xs[members]
        center = km.cluster_centers_[c]
        rep_local = members[int(np.argmin(((sub - center) ** 2).sum(1)))]
        rep_cam_idx = idxs[rep_local]
        _url, rep_path = ok[rep_cam_idx]
        rep_out = rep_dir / f"viewpoint_{c:02d}_n{len(members)}.jpg"
        img = cv2.imread(str(rep_path))
        if img is not None:
            cv2.imwrite(str(rep_out), img)
        print(f"  Viewpoint {c:02d}: {len(members)} cameras "
              f"({100*len(members)/len(idxs):.1f}%)  -> rep: {rep_out.name}")
        for j in members:
            rows.append((idxs[j], c))

    # camera -> cluster map, LOCAL FILE ONLY (carries the real URL, stays local)
    assign = out_dir / "viewpoint_assignments.csv"
    with open(assign, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera_index", "camera_id", "viewpoint_cluster", "stream_url"])
        for cam_idx, c in sorted(rows):
            url, _p = ok[cam_idx]
            w.writerow([cam_idx, cam_id(url), c, url])

    print(f"\n=== OUTPUT ===")
    print(f"  representative thumbnails : {rep_dir}")
    print(f"  camera->viewpoint map     : {assign}  (LOCAL — has URLs, keep it local)")
    print( "\n  Next: open viewpoint_reps/ and eyeball the archetypes. For run7,")
    print( "  sample cameras from EACH viewpoint to label (weight the confused")
    print( "  pairs: car/pickup, truck/bus, taxi/car), and HOLD OUT one or two")
    print( "  whole viewpoints as an unseen-angle validation set.")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default="cameras.txt",
                    help="local file of stream URLs (txt or csv)")
    ap.add_argument("--out", default="angle_discovery",
                    help="local output folder")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process first N cameras (0 = all; use for a smoke test)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent stream grabs")
    ap.add_argument("--timeout", type=int, default=30,
                    help="per-camera hard timeout (s)")
    ap.add_argument("--embedder", choices=["auto", "torch", "classic"],
                    default="auto")
    ap.add_argument("--k", type=int, default=None,
                    help="force number of viewpoints (omit to auto-pick)")
    ap.add_argument("--max-k", type=int, default=20,
                    help="upper bound when auto-picking k")
    ap.add_argument("--sample", type=int, default=25,
                    help="(reserved) cameras to sample per viewpoint for labeling")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    urls = read_camera_list(Path(args.cameras).expanduser().resolve())
    if args.limit:
        urls = urls[:args.limit]
    print(f"cameras to probe: {len(urls)}  (URLs kept local, never printed)")

    print("\n=== [1/3] grabbing one frame per camera ===")
    ok = capture_all(urls, out_dir, args.workers, args.timeout)
    if not ok:
        sys.exit("[FATAL] no cameras reachable — check the URL list / network.")

    print("\n=== [2/3] embedding frames ===")
    idxs, X = build_embeddings(ok, args.embedder)
    print(f"  embedded {len(idxs)} frames into {X.shape[1]}-dim vectors")

    print("\n=== [3/3] clustering into viewpoints ===")
    cluster_and_report(idxs, X, ok, out_dir, args.k, args.max_k, args.sample)


if __name__ == "__main__":
    main()
