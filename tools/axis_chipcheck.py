"""Axis on-camera ML feasibility check (roadmap M0.5).

Queries a camera's VAPIX Properties and classifies whether the on-device
(DLPU) deploy path is viable. ARTPEC-8/9 -> on-camera; else -> site needs a
DLPU-capable Axis camera (ARTPEC-8/9).
(See memory: vca-edge-deployment-target — lucius-fox verdict 2026-06-29.)

Privacy: host/creds are runtime-only. Do NOT commit them. Output prints the
chip + verdict, never stores the IP.

Usage:
    python tools/axis_chipcheck.py --host 10.207.200.109 --user root
    # password via prompt or AXIS_PASS env var
"""
import argparse
import os
import re

import requests
from requests.auth import HTTPDigestAuth

_ARTPEC_RE = re.compile(r"artpec[\s\-]?(\d+)", re.IGNORECASE)


def detect_artpec(vapix_text):
    """Return the ARTPEC generation (int) found in a VAPIX dump, else None."""
    m = _ARTPEC_RE.search(vapix_text or "")
    return int(m.group(1)) if m else None


def classify(artpec_gen):
    """Map an ARTPEC generation to an on-camera viability verdict."""
    viable = artpec_gen is not None and artpec_gen >= 8
    if viable:
        return {
            "artpec_gen": artpec_gen,
            "on_camera_viable": True,
            "recommendation": "on-camera",
            "reason": f"ARTPEC-{artpec_gen} has a DLPU — INT8 TFLite via ACAP/Larod is viable "
                      f"(still needs the YOLO26n head graph-cut + INT8 accuracy pass).",
        }
    return {
        "artpec_gen": artpec_gen,
        "on_camera_viable": False,
        "recommendation": "needs-dlpu-axis",
        "reason": ("No DLPU-capable ARTPEC-8/9 detected — a DLPU-capable Axis camera "
                   "(ARTPEC-8/9) is required at this site. If the model is too large for "
                   "the DLPU, adapt the model (graph-cut before the detection head + "
                   "per-tensor INT8, or use a smaller/leaner model variant). "
                   "Non-Axis or companion-box hardware is not the fallback."),
    }


def query_vapix(host, user, password, timeout=5):
    """Fetch the System Properties param group over VAPIX (digest auth)."""
    url = f"http://{host}/axis-cgi/param.cgi"
    params = {"action": "list", "group": "Properties.System"}
    try:
        resp = requests.get(url, params=params, auth=HTTPDigestAuth(user, password),
                            timeout=timeout)
        resp.raise_for_status()
        return {"reachable": True, "text": resp.text, "error": None}
    except requests.RequestException as e:
        return {"reachable": False, "text": None, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="camera IP/hostname (LOCAL, not committed)")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default=os.environ.get("AXIS_PASS"))
    args = ap.parse_args()
    if not args.password:
        import getpass
        args.password = getpass.getpass("Axis password: ")

    res = query_vapix(args.host, args.user, args.password)
    if not res["reachable"]:
        print(f"UNREACHABLE: {res['error']}")
        print("Cannot determine chip over VAPIX. If the camera is only reachable on-site / via "
              "the gateway, run this there. Until confirmed, assume the site needs a "
              "DLPU-capable Axis camera (ARTPEC-8/9).")
        return

    gen = detect_artpec(res["text"])
    verdict = classify(gen)
    print(f"ARTPEC generation : {gen}")
    print(f"On-camera viable  : {verdict['on_camera_viable']}")
    print(f"Recommendation    : {verdict['recommendation']}")
    print(f"Reason            : {verdict['reason']}")


if __name__ == "__main__":
    main()
