"""Fetch a thin band of slices at full resolution over SFTP: a sparse mirror of the acquisitions' RAW_DATA.

Each helix frame sees a slice on one detector row, which moves down the detector as the stage rises (about
2300 rows a turn on Paris4). A projection file stores every frame as chunks of 128 rows, each its own HTJ2K
codestream, so a band of slices needs only the one or two chunks of each frame that hold its rows. This tool
fetches the masters whole, the HDF5 metadata of each projection file and those chunks whole into a file of
the source's size at the source's relative path (the rest of the file stays a hole), and dark and flat files
whole. The output is laid out like RAW_DATA and is read as one: a chunk that was not fetched holds zeros and
fails to decode ("not a blosc2 frame") rather than reading as data, and frame files outside the band are absent.

Run ``python -m vc_tomography.fetch_band --help``. Rerun the same command to resume: a file is done when its
``<file>.band.json`` record exists with this plan's settings.

Where frame i stands at stage height z_i, the band's middle is on detector row
(rows - 1) / 2 + row_sign * (z_i - center_mm) / pixel, and its rows are [middle - slices / 2 - margin,
middle + slices / 2 + margin). row_sign is the calibration's sense of the rows against the axis stage (+1 if a
larger stage position puts the sample on higher rows; Paris4's tiles: -1). The margin covers what the record's
heights do not hold: a tile's height in the calibration (tens of rows on Paris4) and the pitch's scale.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time

from .download_projections import _atomic_json, _connect, _drop_connection, _fetch, _json_bytes, _remote_path

FORMAT = 1


def band_rows(height_mm, center_mm, pixel_um, rows, slices, margin, row_sign):
    """Detector rows [lo, hi) of the band in a frame at stage height `height_mm`, clipped to the detector
    (lo >= hi: the band is off the detector)."""
    middle = (rows - 1) / 2 + row_sign * (height_mm - center_mm) * 1000.0 / pixel_um
    lo = math.floor(middle - slices / 2 - margin)
    hi = math.ceil(middle + slices / 2 + margin) + 1
    return max(0, lo), min(rows, hi)


def chunk_selection(heights_mm, first_frame, center_mm, pixel_um, rows, slices, margin, row_sign):
    """{frame index in the file: [row_lo, row_hi)} for the frames of a file whose band is on the detector;
    heights_mm[k] is the height of the file's frame first_frame + k."""
    out = {}
    for k, z in enumerate(heights_mm):
        lo, hi = band_rows(z, center_mm, pixel_um, rows, slices, margin, row_sign)
        if lo < hi:
            out[first_frame + k] = (lo, hi)
    return out


def _settings(args):
    return dict(format_version=FORMAT, tiles=args.tiles, tile_folder=args.tile_folder, center_mm=args.center_mm,
                slices=args.slices, margin=args.margin, row_sign=args.row_sign, remote=args.remote,
                source_root=str(args.source_root), dark_flat_files=args.dark_flat_files)


def build_plan(args, log):
    """Masters fetched whole now (the plan needs their records); one job per frame file."""
    from . import _core
    settings = _settings(args)
    plan_path = args.output / "band_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["settings"] != settings:
            raise ValueError("Output already has a different band plan; use a new output folder or the original settings")
        log.info("Resuming band plan: %d files", len(plan["jobs"]))
        return plan
    link = dict(spec=args.remote, path=args.remote.partition(":")[2], source_root=str(args.source_root))
    jobs, tiles = [], []
    for number, tile in enumerate(args.tiles, 1):
        label = args.tile_folder.format(tile=tile)
        relative = Path(label) / (label + "_0001")
        master_rel = relative / (label + "_0001.h5")
        master = args.output / master_rel
        if not master.exists():
            master.parent.mkdir(parents=True, exist_ok=True)
            tmp = master.with_name(master.name + ".partial")
            tmp.unlink(missing_ok=True)
            _fetch(_connect(args.remote), _remote_path(link, args.source_root / master_rel)[0], str(tmp), None, None)
            os.replace(tmp, master)
        # context 0: frames within one detector height of the band, a superset of those that see it
        p = _core.plan_projection_subset(str(master), args.center_mm, args.slices, 0.0)
        n_projection = n_chunk_frames = 0
        for series in p["series"]:
            projection = series["role"] == "projection"
            pieces = series["pieces"] if projection or not args.dark_flat_files else series["pieces"][:args.dark_flat_files]
            for piece in pieces:
                rel = Path(piece["file"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe source path: {rel}")
                job = dict(tile=tile, role=series["role"], entry=series["entry"], file=str(relative / rel),
                           dataset=piece["dataset"], rows=p["rows"])
                if projection:
                    offset = piece["first"] - series["first"]
                    heights = series["height_mm"][offset:offset + piece["count"]]
                    selection = chunk_selection(heights, piece["source_first"], args.center_mm, p["pixel_um"],
                                                p["rows"], args.slices, args.margin, args.row_sign)
                    if not selection:
                        continue
                    job["frames"] = {str(k): list(v) for k, v in selection.items()}
                    n_projection += 1
                    n_chunk_frames += len(selection)
                job["id"] = hashlib.sha256(_json_bytes(job)).hexdigest()
                jobs.append(job)
        tiles.append(dict(tile=tile, master=str(master_rel), pixel_um=p["pixel_um"], rows=p["rows"], cols=p["cols"],
                          pitch_mm=p["pitch_mm"], projection_files=n_projection, band_frames=n_chunk_frames))
        log.info("Planned tile %d/%d %s: %d projection files, %d frames see the band", number, len(args.tiles), tile,
                 n_projection, n_chunk_frames)
    jobs.sort(key=lambda j: j["role"] != "projection")  # projections first (stable: tile and file order kept)
    plan = dict(settings=settings, tiles=tiles, jobs=jobs,
                note="A sparse mirror of RAW_DATA: projection files hold only the band's chunks (the rest is a hole "
                     "and fails to decode); dark and flat files are whole.")
    _atomic_json(plan_path, plan)
    return plan


def _record(path):
    return path.with_name(path.name + ".band.json")


def _done(output, job):
    record = _record(output / job["file"])
    if not record.exists():
        return False
    return json.loads(record.read_text()).get("job") == job["id"] and (output / job["file"]).exists()


def _run_job(job, output, remote, source_root, retries):
    """One file, in a worker process (its own SFTP connection, kept between files)."""
    path = Path(output) / job["file"]
    if _done(Path(output), job):
        return dict(id=job["id"], bytes=0, seconds=0.0, resumed=True)
    link = dict(spec=remote, path=remote.partition(":")[2], source_root=source_root)
    remote_path, origin = _remote_path(link, str(Path(source_root) / job["file"]))
    select = None
    if job["role"] == "projection":
        frames = {int(k): v for k, v in job["frames"].items()}

        def select(offset, shape):
            band = frames.get(offset[0])
            return band is not None and offset[1] < band[1] and offset[1] + shape[1] > band[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    for attempt in range(retries + 1):
        tmp.unlink(missing_ok=True)
        start = time.monotonic()
        try:
            fetched, transferred = _fetch(_connect(remote), remote_path, str(tmp), job["dataset"] if select else None,
                                          None, select)
            break
        except Exception:
            _drop_connection()
            if attempt == retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    seconds = time.monotonic() - start
    if select is not None and not fetched[0]:
        raise ValueError(f"{job['file']}: no chunk of the source holds the band's rows")
    os.replace(tmp, path)
    _atomic_json(_record(path), dict(job=job["id"], source=origin, role=job["role"], whole=select is None,
                                     fetched=None if select is None else dict(offsets=fetched[0], lengths=fetched[1]),
                                     bytes_transferred=transferred, seconds=seconds))
    return dict(id=job["id"], bytes=transferred, seconds=seconds, resumed=False)


def run(plan, args, log):
    todo = [j for j in plan["jobs"] if not _done(args.output, j)]
    if args.max_files:
        todo = todo[:args.max_files]
    log.info("%d of %d files to fetch with %d workers", len(todo), len(plan["jobs"]), args.workers)
    start, total, done, last = time.monotonic(), 0, 0, 0.0
    by_role = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_run_job, j, str(args.output), args.remote, str(args.source_root), args.retries) for j in todo]
        roles = {f: j["role"] for f, j in zip(futures, todo)}
        for f in as_completed(futures):
            r = f.result()
            done += 1
            total += r["bytes"]
            role = by_role.setdefault(roles[f], [0, 0, 0.0])
            role[0] += 1
            role[1] += r["bytes"]
            role[2] += r["seconds"]
            now = time.monotonic()
            if now - last >= args.log_interval or done == len(todo):
                elapsed = now - start
                log.info("%d/%d files, %.2f GB, %.1f MB/s over %.0f s", done, len(todo), total / 1e9,
                         total / 1e6 / max(elapsed, 1e-9), elapsed)
                last = now
    for role, (n, b, s) in sorted(by_role.items()):
        log.info("%s: %d files, %.2f GB, %.1f MB/s per worker", role, n, b / 1e9, b / 1e6 / max(s, 1e-9))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="the sparse RAW_DATA mirror")
    parser.add_argument("--remote", required=True, help="rclone SFTP remote:path of RAW_DATA (e.g. esrf:/data/.../RAW_DATA)")
    parser.add_argument("--source-root", type=Path, help="the path --remote stands for (default: its path)")
    parser.add_argument("--tiles", nargs="+", required=True)
    parser.add_argument("--tile-folder", required=True, help="a tile's acquisition folder, {tile} standing for its name")
    parser.add_argument("--center-mm", type=float, required=True, help="stage height at which the band is on the middle row")
    parser.add_argument("--slices", type=int, default=16, help="band thickness in original-resolution rows")
    parser.add_argument("--margin", type=int, default=32, help="rows added on either side of the band")
    parser.add_argument("--row-sign", type=int, choices=[-1, 1], required=True,
                        help="sense of the rows against the axis stage (calibration; Paris4: -1)")
    parser.add_argument("--dark-flat-files", type=int, default=0, help="files of each dark/flat series to fetch; 0: all")
    parser.add_argument("--workers", type=int, default=16, help="concurrent files, each its own SFTP connection")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=0, help="stop after this many pending files; 0: all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-interval", type=float, default=10)
    args = parser.parse_args(argv)
    if "{tile}" not in args.tile_folder:
        parser.error("--tile-folder must contain {tile}")
    if args.slices < 1 or args.margin < 0 or args.workers < 1 or args.retries < 0 or args.dark_flat_files < 0 or args.max_files < 0:
        parser.error("counts must be positive (margin, retries, files nonnegative)")
    if shutil.which("rclone") is None:
        parser.error("the SFTP login comes from the rclone config; rclone is not on PATH")
    args.source_root = args.source_root or Path(args.remote.partition(":")[2])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("vc_tomography.fetch_band")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(args.output / "fetch_band.log")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
    import fcntl
    with (args.output / ".fetch_band.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another fetch is using this output folder")
        log.info("Band: %d slices at %.4f mm, margin %d rows, row sign %+d", args.slices, args.center_mm, args.margin, args.row_sign)
        plan = build_plan(args, log)
        if args.plan_only:
            return 0
        return run(plan, args, log)


if __name__ == "__main__":
    raise SystemExit(main())
