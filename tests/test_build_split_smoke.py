"""
Offline smoke test for model_compare/build_split.py — runs WITHOUT the SSD or DGX.

Generates a tiny synthetic FLAT corpus (1x1 PNGs — build_split never decodes
pixels, it symlinks images and reads labels), runs build_split against it, and
asserts the split is produced with the canonical 14-class schema and the right
image count. Uses new_data/dataset as scratch and cleans it up.

Run:
    python tests/test_build_split_smoke.py
"""
import base64
import shutil
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "sample_corpus"      # generated, git-ignored
OUT    = ROOT / "new_data" / "dataset"         # build_split output (scratch)

# 1x1 transparent PNG — content is irrelevant; build_split only symlinks images.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def make_corpus() -> int:
    img_dir, lbl_dir = CORPUS / "images", CORPUS / "labels"
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    canon = [c for c in (ROOT / "models" / "traffic14.names").read_text().splitlines() if c.strip()]
    (CORPUS / "classes.txt").write_text("\n".join(canon) + "\n")

    # A few images per non-dropped class so val/test populate. ids: 0=car 1=bike 7=person
    plan = {0: 5, 1: 4, 7: 3}
    n = 0
    for cid, count in plan.items():
        for i in range(count):
            stem = f"img_{cid}_{i:02d}"
            (img_dir / f"{stem}.png").write_bytes(PNG)
            (lbl_dir / f"{stem}.txt").write_text(f"{cid} 0.5 0.5 0.2 0.2\n")
            n += 1
    return n


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    n = make_corpus()

    r = subprocess.run(
        [sys.executable, str(ROOT / "model_compare" / "build_split.py"), "--source", str(CORPUS)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("[SMOKE] build_split failed")

    yaml = (OUT / "data.yaml").read_text()
    assert "nc: 14" in yaml, f"expected nc:14, got:\n{yaml}"
    total = sum(len(list((OUT / "images" / s).glob("*"))) for s in ("train", "val", "test"))
    assert total == n, f"expected {n} images split, got {total}"

    print(f"[SMOKE] OK — {n} images → train/val/test, nc:14 canonical, flat ingest")
    shutil.rmtree(OUT)        # the real split builds on the DGX, not here
    shutil.rmtree(CORPUS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
