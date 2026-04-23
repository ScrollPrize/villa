"""Snapshot the latest UNet checkpoint from a training output directory.

Idempotent: if the destination already exists, refuses to overwrite a different
file (different mtime / size). Used to freeze a teacher UNet for the
dinoguided-labels pipeline so the student run is reproducible across restarts.
"""
import argparse
import re
import shutil
from pathlib import Path


CKPT_RE = re.compile(r"ckpt_(\d+)\.pth$")


def find_latest(src_dir: Path) -> Path:
    candidates = []
    for p in src_dir.glob("ckpt_*.pth"):
        m = CKPT_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"no ckpt_*.pth found in {src_dir}")
    candidates.sort()
    return candidates[-1][1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", required=True, type=Path,
                   help="training output dir, e.g. /ephemeral/3d_ink_checkpoints/ps256_bcedice")
    p.add_argument("--dst-dir", required=True, type=Path,
                   help="destination dir, e.g. /ephemeral/dinov2_ckpts")
    p.add_argument("--label", default="teacher_unet",
                   help="filename stem (suffixed with _ckpt_<step>.pth)")
    args = p.parse_args()

    src = find_latest(args.src_dir)
    m = CKPT_RE.search(src.name)
    step = int(m.group(1))
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    dst = args.dst_dir / f"{args.label}_ckpt_{step:06d}.pth"

    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            print(f"already snapshotted: {dst} (size match, no-op)")
            return
        raise FileExistsError(
            f"{dst} exists with different size; refusing to overwrite. "
            f"Delete it manually if you intend to re-snapshot."
        )

    print(f"copy {src} -> {dst} ({src.stat().st_size/1e9:.2f} GB)")
    shutil.copy2(src, dst)
    print(f"done. snapshot step={step}")


if __name__ == "__main__":
    main()
