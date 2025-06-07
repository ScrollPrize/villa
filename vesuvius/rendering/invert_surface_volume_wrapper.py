#!/usr/bin/env python3
"""
Run inverse-PPM on many scroll-segments in parallel without unbounded RAM growth.

Memory-safe strategy
--------------------
* We spawn a ``multiprocessing.Pool`` whose workers handle **only one task each**
  (``maxtasksperchild=1``).  After finishing a segment the child process exits,
  releasing all of its memory back to the OS.
* The parent process collects status tuples, writes a summary, and appends every
  *new* success to a ``completed_segments.txt`` file so the next launch can skip
  work that was already done.

File format of ``completed_segments.txt`` (one per line)
--------------------------------------------------------
::

    s1/AB12
    s1/XY07
    s5/CD99
"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from functools import partial
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("segment-runner")

# ──────────────────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────────────────
SCROLL_DIMS = {
    "s1": (14376, 7888, 8096),
    "s5": (21000, 6700, 9100),
}

# URL root per scroll
BASE_URLS = {
    "s1": "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths",
    "s5": "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths",
}

# ──────────────────────────────────────────────────────────────────────────
# Helper: download an OBJ (plain or “…_flat.obj”)
# ──────────────────────────────────────────────────────────────────────────
def _download_obj(scroll_id: str, segment_id: str, out_dir: str) -> str:
    base_url = BASE_URLS[scroll_id]
    os.makedirs(out_dir, exist_ok=True)

    for suffix in ("", "_flat"):
        name = f"{segment_id}{suffix}.obj"
        url = f"{base_url}/{segment_id}/{name}"
        dst = os.path.join(out_dir, name)
        try:
            urllib.request.urlretrieve(url, dst)
            logger.debug("[%s/%s] downloaded %s", scroll_id, segment_id, name)
            return dst
        except Exception as exc:
            logger.debug("[%s/%s] could not download %s: %s",
                         scroll_id, segment_id, name, exc)
    raise RuntimeError("no OBJ variant found on the server")

# ──────────────────────────────────────────────────────────────────────────
# Core worker (heavy)
# ──────────────────────────────────────────────────────────────────────────
def process_segment(
    scroll_id: str,
    segment_id: str,
    layer_paths: List[str],
    dims: Tuple[int, int, int],
    *,
    invppm_script: str,
    step: float,
    max_side_triangle: int,
    batch_triangles: int,
    output_root: str,
) -> str:
    """
    Download OBJ (+mask), stage PNG layers, run inverse-PPM.

    Returns
    -------
    str
        Path of the output folder that now contains the Zarr datasets.
    """
    logger.info("[%s/%s] PID %d starting", scroll_id, segment_id, os.getpid())

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) OBJ
        obj_path = _download_obj(scroll_id, segment_id, tmpdir)

        # 2) mask (optional)
        obj_base = os.path.splitext(os.path.basename(obj_path))[0]  # ABCD or ABCD_flat
        mask_name = f"{obj_base}_mask.png"
        mask_url = f"{BASE_URLS[scroll_id]}/{segment_id}/{mask_name}"
        mask_path = os.path.join(tmpdir, mask_name)
        try:
            urllib.request.urlretrieve(mask_url, mask_path)
            logger.debug("[%s/%s] mask downloaded", scroll_id, segment_id)
        except Exception:
            mask_path = ""  # Proceed without mask
            logger.debug("[%s/%s] mask not found (optional)", scroll_id, segment_id)

        # 3) stage images as 00.png, 01.png, …
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for src in layer_paths:
            layer_idx = int(os.path.splitext(os.path.basename(src))[0].split("_")[-1])
            shutil.copyfile(src, os.path.join(images_dir, f"{layer_idx:02d}.png"))

        # 4) prepare output folder
        seg_out = os.path.join(output_root, f"{scroll_id}_{segment_id}")
        if os.path.isdir(seg_out):
            raise RuntimeError(f"output folder {seg_out!s} already exists")
        os.makedirs(seg_out, exist_ok=True)

        # 5) invoke inverse-PPM script
        cmd = [
            "python3", invppm_script,
            "--obj", obj_path,
            "--images", images_dir,
            "--output", seg_out,
            "--dims", *map(str, dims),
            "--step", str(step),
            "--max_side_triangle", str(max_side_triangle),
            "--batch_triangles", str(batch_triangles),
        ]
        logger.debug("[%s/%s] running: %s", scroll_id, segment_id, " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret:
            raise RuntimeError(f"inverse-PPM exited with code {ret}")

        logger.info("[%s/%s] finished OK", scroll_id, segment_id)
        return seg_out

# ──────────────────────────────────────────────────────────────────────────
# Pool initialiser + safe wrapper
# ──────────────────────────────────────────────────────────────────────────
_GLOBAL_WORKER = None  # will be set in pool workers


def _init_pool(worker_partial):
    global _GLOBAL_WORKER
    _GLOBAL_WORKER = worker_partial


def _safe_worker(args_tuple):
    """Run the heavy worker and capture any exception."""
    scroll_id, segment_id, *_ = args_tuple
    try:
        out_path = _GLOBAL_WORKER(*args_tuple)
        return scroll_id, segment_id, True, out_path
    except Exception as exc:
        return scroll_id, segment_id, False, str(exc)

# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Parallel inverse-PPM segment runner (memory-safe).")
    ap.add_argument("-i", "--input_folder", required=True,
                    help="Folder with PNGs named <scroll>_<segment>_<layer>.png")
    ap.add_argument("-o", "--output_folder", required=True,
                    help="Where each <scroll>_<segment> result tree will be written")
    ap.add_argument("-s", "--invppm_script", required=True,
                    help="Path to the heavy inverse-PPM Python script")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--max_side_triangle", type=int, default=10)
    ap.add_argument("--batch_triangles", type=int, default=4000)
    ap.add_argument("-w", "--workers", type=int, default=os.cpu_count(),
                    help="Parallel workers (default: CPU count)")
    ap.add_argument("--completed_file",
                    help="Text file that stores finished segment IDs "
                         "(default: <output_folder>/completed_segments.txt)")
    args = ap.parse_args()

    completed_file = args.completed_file or os.path.join(
        args.output_folder, "completed_segments.txt")
    completed_ids = set()
    if os.path.isfile(completed_file):
        with open(completed_file, "r", encoding="utf-8") as fh:
            completed_ids = {ln.strip() for ln in fh if ln.strip()}
        logger.info("loaded %d completed IDs from %s", len(completed_ids), completed_file)

    # ---------- gather tasks -------------------------------------------------
    patt = re.compile(r'(?P<scroll>[^_]+)_(?P<segment>[^_]+)_(?P<layer>\d+)\.png')
    task_dict = {}  # {scroll: {segment: [png, png, …]}}
    for pth in glob.glob(os.path.join(args.input_folder, "*.png")):
        m = patt.match(os.path.basename(pth))
        if not m:
            logger.warning("ignoring file with wrong name: %s", pth)
            continue
        s_id = m["scroll"]
        g_id = m["segment"]
        task_dict.setdefault(s_id, {}).setdefault(g_id, []).append(pth)

    tasks = []
    for scroll_id, segs in task_dict.items():
        if scroll_id not in SCROLL_DIMS:
            logger.warning("no dims for scroll %s — skipped", scroll_id)
            continue
        dims = SCROLL_DIMS[scroll_id]
        for seg_id, layer_list in segs.items():
            seg_key = f"{scroll_id}/{seg_id}"
            if seg_key in completed_ids:
                logger.info("skip %s (already done)", seg_key)
                continue
            layer_list.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
            tasks.append((scroll_id, seg_id, layer_list, dims))

    if not tasks:
        logger.info("nothing to do — all segments completed")
        return

    os.makedirs(args.output_folder, exist_ok=True)

    # ---------- prepare worker partial & start pool --------------------------
    worker_partial = partial(
        process_segment,
        invppm_script=args.invppm_script,
        step=args.step,
        max_side_triangle=args.max_side_triangle,
        batch_triangles=args.batch_triangles,
        output_root=args.output_folder,
    )

    results = []
    with mp.Pool(processes=args.workers,
                 maxtasksperchild=1,
                 initializer=_init_pool,
                 initargs=(worker_partial,)) as pool:
        for scroll_id, seg_id, ok, payload in pool.imap_unordered(_safe_worker, tasks):
            if ok:
                logger.info("[%s/%s] DONE → %s", scroll_id, seg_id, payload)
            else:
                logger.error("[%s/%s] FAILED → %s", scroll_id, seg_id, payload)
            results.append((scroll_id, seg_id, ok, payload))

    # ---------- summary & bookkeeping ---------------------------------------
    succeeded = [f"{s}/{g}" for s, g, ok, _ in results if ok]
    failed    = [f"{s}/{g}" for s, g, ok, _ in results if not ok]
    logger.info("summary: %d succeeded, %d failed", len(succeeded), len(failed))

    if succeeded:
        os.makedirs(os.path.dirname(completed_file), exist_ok=True)
        with open(completed_file, "a", encoding="utf-8") as fh:
            for seg_key in succeeded:
                if seg_key not in completed_ids:  # avoid dupes
                    fh.write(seg_key + "\n")
        logger.info("appended %d new IDs to %s", len(succeeded), completed_file)

    if failed:
        logger.info("failed segments:")
        for seg_key in failed:
            logger.info("  • %s", seg_key)


if __name__ == "__main__":
    mp.freeze_support()
    main()
