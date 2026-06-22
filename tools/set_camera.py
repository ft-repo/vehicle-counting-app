#!/usr/bin/env python3
"""
Set the camera source in config/scene_config.json from just a camera IP.

The enixma HLS proxy serves each camera at:
    https://<gateway>.enixma.net/live/<CAMERA-IP>.stream/playlist.m3u8

The <gateway> subdomain is per-SITE (not per-IP). Known gateways are listed
below; the script tries each with the given IP and uses the first one whose
stream actually opens, then writes the source + native resolution into the
scene config.

Usage:
    python tools/set_camera.py 10.207.200.109
    python tools/set_camera.py 10.207.200.109 --gateway 69mst-ktpsv1
    python tools/set_camera.py 10.207.200.109 --dry-run

A new site? Find its gateway once (the camera dashboard's URL column shows the
prefix), confirm DNS resolves, then add it to KNOWN_GATEWAYS below.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "scene_config.json"

# Per-site HLS proxy gateways. Add new sites here as you encounter them.
KNOWN_GATEWAYS = [
    "69mst-ktpsv1",   # 69MST-KTP (Nonthri Rd area)
    "68ftd-pksv1",    # 68FTD
]


def hls_url(gateway: str, ip: str) -> str:
    return f"https://{gateway}.enixma.net/live/{ip}.stream/playlist.m3u8"


def probe(url: str):
    """Return (width, height) if the stream opens, else None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-timeout", "15000000",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", url],
            capture_output=True, text=True, timeout=40,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        streams = json.loads(out.stdout).get("streams", [])
        if streams:
            return int(streams[0]["width"]), int(streams[0]["height"])
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="Set camera source from a camera IP.")
    ap.add_argument("ip", help="Camera LAN IP, e.g. 10.207.200.109")
    ap.add_argument("--gateway", help="Force a specific gateway subdomain")
    ap.add_argument("--dry-run", action="store_true", help="Print result, don't write config")
    args = ap.parse_args()

    gateways = [args.gateway] if args.gateway else KNOWN_GATEWAYS

    found = None
    for gw in gateways:
        url = hls_url(gw, args.ip)
        print(f"[probe] {url}")
        res = probe(url)
        if res:
            found = (url, res)
            print(f"[ok]    feed opens — {res[0]}x{res[1]}")
            break
        print("[miss]  no stream")

    if not found:
        print(f"\nNo known gateway serves {args.ip}.")
        print("Find this site's gateway (camera dashboard URL column), add it to "
              "KNOWN_GATEWAYS, or pass --gateway <subdomain>.")
        raise SystemExit(1)

    url, (w, h) = found
    if args.dry_run:
        print(f"\n[dry-run] would set source={url}  width={w} height={h}")
        return

    cfg = json.loads(CONFIG.read_text())
    cfg.setdefault("camera", {})
    cfg["camera"]["source"] = url
    cfg["camera"]["width"] = w
    cfg["camera"]["height"] = h
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\n[done]  {CONFIG} updated.")
    print("        Reminder: ROI/lanes are scene-specific — redraw with "
          "`python run_camera.py` ([E] edit, [S] save) if the view changed.")


if __name__ == "__main__":
    main()
