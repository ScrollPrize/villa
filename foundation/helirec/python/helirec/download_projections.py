"""Download a physical Paris4 slab as uncompressed, reduced detector frames.

Run with ``python -m helirec.download_projections --help``. Image reads, HTJ2K
resolution selection, slab planning and HDF5 frame writes use the C++ core.
Python schedules files, preserves metadata and handles logging and restart.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import sys
import time

import h5py
import numpy as np

from . import _core

BASE = Path("/mnt/raid_nvme/paris4_mosaic_recon")
SOURCE = Path("/home/sean/mnt/esrf-bm18/keep_it_for_replaying_20260311/RAW_DATA")
TILES = ["T0-01"] + [f"T1-{i:02}" for i in range(1, 7)] + [f"T2-{i:02}" for i in range(1, 13)]
FORMAT = 1


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path, value):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        f.write(_json_bytes(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _copy_metadata(src, dst, log):
    """Copy only the master (never follow its virtual frame datasets)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    digest = hashlib.sha256()
    if not dst.exists():
        size, done, last = src.stat().st_size, 0, time.monotonic()
        with src.open("rb") as inp, tmp.open("wb") as out:
            while block := inp.read(4 << 20):
                out.write(block)
                done += len(block)
                if time.monotonic() - last >= 5:
                    log.info("Metadata %s: %.1f / %.1f MiB", src.name, done / 2**20, size / 2**20)
                    last = time.monotonic()
            out.flush()
            os.fsync(out.fileno())
        if done != size:
            raise OSError(f"Metadata changed or was truncated: {src}")
        os.replace(tmp, dst)
    with dst.open("rb") as f:
        while block := f.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


def _source(job):
    for path in job["sources"]:
        if Path(path).is_file():
            return path
    raise FileNotFoundError(f"No source file for {job['tile']} {job['relative_file']}: {job['sources']}")


def build_plan(args, log):
    settings = dict(format_version=FORMAT, tiles=args.tiles, slices=args.slices, reduce=args.reduce,
                    center_mm=args.center_mm, context_turns=args.context_turns,
                    source_root=str(args.source_root), local_root=str(args.local_root))
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["settings"] != settings:
            raise ValueError("Output already has a different plan; use a new output directory or the original settings")
        log.info("Resuming existing plan: %d files", len(plan["jobs"]))
        return plan
    jobs, tiles = [], []
    factor = 1 << args.reduce
    for number, tile in enumerate(args.tiles, 1):
        log.info("Planning tile %d/%d: %s", number, len(args.tiles), tile)
        label = f"1.129um_mosaic_PHerc_Paris4_ROI-1_{tile}"
        relative = Path(label) / (label + "_0001")
        local, remote = args.local_root / relative, args.source_root / relative
        master_name = label + "_0001.h5"
        source_master = local / master_name if (local / master_name).is_file() else remote / master_name
        master_copy = args.output / "metadata" / tile / master_name
        master_sha = _copy_metadata(source_master, master_copy, log)
        p = _core.plan_projection_subset(str(master_copy), args.center_mm, args.slices, args.context_turns)
        if p["rows"] % factor or p["cols"] % factor:
            raise ValueError(f"{tile}: detector shape is not divisible by {factor}")
        projection = next(s for s in p["series"] if s["role"] == "projection")
        sample = next((p for p in projection["pieces"] if (local / p["file"]).is_file()), projection["pieces"][0])
        candidates = [str(local / sample["file"]), str(remote / sample["file"])]
        probe = _core.RawReader([_source(dict(sources=candidates, tile=tile, relative_file=sample["file"]))],
                                sample["dataset"], p["rows"], p["cols"], decoder="openjph")
        storage = probe.storage()
        if not storage["compressed"] or storage["codec"] != "JPEG 2000 (HTJ2K)" or storage["reversible"]:
            raise ValueError(f"{tile}: this downloader expects the irreversible HTJ2K projection format; got {storage}")
        chunk_rows = storage["chunk_rows"]
        if chunk_rows % factor or chunk_rows // factor < 4:
            raise ValueError(f"{tile}: unsupported wavelet chunk height {chunk_rows}")
        summary = {k: v for k, v in p.items() if k not in ("series", "master")}
        summary.update(tile=tile, source_master=str(source_master), source_master_sha256=master_sha,
                       stored_shape=[p["rows"], p["cols"]], output_shape=[p["rows"] // factor, p["cols"] // factor],
                       projection_first=projection["first"], projection_count=projection["count"], chunk_rows=chunk_rows)
        tiles.append(summary)
        for series in p["series"]:
            for piece in series["pieces"]:
                rel = Path(piece["file"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe source path: {rel}")
                offset = piece["first"] - series["first"]
                metadata = {key: values[offset:offset + piece["count"]] for key in ("angle_deg", "height_mm", "epoch")
                            if (values := series[key])}
                dtype = "uint16" if series["role"] == "projection" else "float32"
                job = dict(tile=tile, relative_file=piece["file"], sources=[str(local / rel), str(remote / rel)],
                           output=str(Path(tile) / rel), dataset=piece["dataset"], entry=series["entry"], role=series["role"],
                           first=piece["first"], source_first=piece["source_first"], count=piece["count"],
                           rows=p["rows"], cols=p["cols"], reduce=args.reduce, chunk_rows=chunk_rows,
                           dtype=dtype, source_master_sha256=master_sha, metadata=metadata)
                job["bytes"] = job["count"] * (job["rows"] // factor) * (job["cols"] // factor) * np.dtype(dtype).itemsize
                job["id"] = hashlib.sha256(_json_bytes(job)).hexdigest()
                jobs.append(job)
        log.info("%s: projection frames [%d, %d), %.6f mm slab, %.4f mm context each side",
                 tile, projection["first"], projection["first"] + projection["count"], p["thickness_mm"], p["context_mm"])
    # Projections first; this makes even an interrupted run useful for inspection.
    jobs.sort(key=lambda j: (j["role"] != "projection", args.tiles.index(j["tile"]), j["entry"], j["first"]))
    if len({j["output"] for j in jobs}) != len(jobs):
        raise ValueError("Source mappings reuse an output file; refusing overlapping export jobs")
    plan = dict(settings=settings, tiles=tiles, jobs=jobs, total_bytes=sum(j["bytes"] for j in jobs),
                note="Metadata masters are original records, not runnable reduced BLISS masters. Use the completed tile index.h5 and explicit frame indices.")
    _atomic_json(plan_path, plan)
    return plan


def _complete(path, job):
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            d = f[job["dataset"]]
            shape = (job["count"], job["rows"] >> job["reduce"], job["cols"] >> job["reduce"])
            valid = (f.attrs.get("complete", False) and f.attrs.get("job_sha256") == job["id"]
                     and d.shape == shape and d.dtype == np.dtype(job["dtype"])
                     and d.id.get_create_plist().get_nfilters() == 0)
            if valid:
                return True
    except (OSError, KeyError):
        pass
    raise ValueError(f"Existing output is invalid or belongs to another plan; refusing to overwrite {path}")


def _download_once(job, output, threads, batch, events):
    from .calibration import at_resolution
    path = Path(output) / job["output"]
    if _complete(path, job):
        return dict(bytes_requested=0, resumed=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)  # only this downloader's unpublished temporary file
    source = _source(job)
    projection = job["role"] == "projection"
    reader = _core.RawReader([source], job["dataset"], job["rows"], job["cols"],
                             reduce=job["reduce"] if projection else 0, decoder="openjph" if projection else "auto")
    storage = reader.storage()
    if projection and (not storage["compressed"] or storage["reversible"] or storage["chunk_rows"] != job["chunk_rows"]):
        raise ValueError(f"Projection storage differs from the planned calibration grid: {source}")
    if job["source_first"] + job["count"] > storage["frames"]:
        raise ValueError(f"Source has fewer frames than the master records: {source}")
    rows, cols = job["rows"] >> job["reduce"], job["cols"] >> job["reduce"]
    writer = _core.UncompressedFrameWriter(str(tmp), job["dataset"], job["count"], rows, cols, job["dtype"])
    digest, start, last = hashlib.sha256(), time.monotonic(), 0.0
    try:
        for first in range(0, job["count"], batch if projection else 1):
            n = min(batch if projection else 1, job["count"] - first)
            frames = np.arange(job["source_first"] + first, job["source_first"] + first + n, dtype=np.int32)
            data = reader.read_frames(np.zeros(n, np.int32), frames, n_threads=threads)
            if not projection:
                data = np.ascontiguousarray(at_resolution(data, job["reduce"], job["chunk_rows"]))
            writer.write(first, data)
            digest.update(memoryview(data).cast("B"))
            now = time.monotonic()
            if now - last >= 2 or first + n == job["count"]:
                events.put(("progress", job["id"], first + n, reader.bytes_read))
                last = now
    finally:
        writer.close()
    with h5py.File(tmp, "r+") as f:
        f.attrs.update(complete=True, job_sha256=job["id"], data_sha256=digest.hexdigest(), source=source,
                       source_master_sha256=job["source_master_sha256"], original_frame_first=job["first"],
                       source_frame_first=job["source_first"], scale=2.0 ** -job["reduce"], role=job["role"],
                       decoder="OpenJPH 0.32.0" if projection else "helirec.calibration.at_resolution (9/7 low-pass)",
                       bytes_requested=reader.bytes_read)
        meta = f.create_group("helirec_metadata")
        meta.create_dataset("original_frame_index", data=np.arange(job["first"], job["first"] + job["count"], dtype=np.int64))
        for key, value in job["metadata"].items():
            meta.create_dataset(key, data=np.asarray(value, dtype=np.float64))
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return dict(bytes_requested=reader.bytes_read, seconds=time.monotonic() - start, resumed=False)


def _download(job, output, threads, batch, retries, events):
    for attempt in range(retries + 1):
        try:
            return _download_once(job, output, threads, batch, events)
        except Exception as e:
            if attempt == retries:
                raise
            events.put(("retry", job["id"], attempt + 1, str(e)))
            time.sleep(min(30, 2 ** attempt))


def _index_tile(output, tile, jobs):
    """Publish a compact VDS only once every source it names is complete."""
    path = output / tile / "index.h5"
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    with h5py.File(tmp, "w") as f:
        for entry in dict.fromkeys(j["entry"] for j in jobs):
            pieces = [j for j in jobs if j["entry"] == entry]
            pieces.sort(key=lambda j: j["first"])
            first = pieces[0]
            rows, cols = first["rows"] >> first["reduce"], first["cols"] >> first["reduce"]
            layout = h5py.VirtualLayout((sum(j["count"] for j in pieces), rows, cols), dtype=first["dtype"])
            at = 0
            for j in pieces:
                if not _complete(output / j["output"], j):
                    raise ValueError("Cannot publish index with missing frames")
                source = h5py.VirtualSource(str(Path(j["output"]).relative_to(tile)), j["dataset"], (j["count"], rows, cols))
                layout[at:at+j["count"]] = source
                at += j["count"]
            g = f.create_group(entry)
            g.attrs.update(role=first["role"], scale=2.0 ** -first["reduce"])
            g.create_virtual_dataset("data", layout)
            g.create_dataset("original_frame_index", data=np.concatenate([np.arange(j["first"], j["first"] + j["count"], dtype=np.int64) for j in pieces]))
            for key in first["metadata"]:
                g.create_dataset(key, data=np.concatenate([j["metadata"][key] for j in pieces]))
        f.attrs["complete"] = True
    os.replace(tmp, path)


def _duration(seconds):
    if not np.isfinite(seconds):
        return "--"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def run_download(plan, args, log):
    jobs = plan["jobs"]
    done, todo = [], []
    for k, job in enumerate(jobs):
        (done if _complete(args.output / job["output"], job) else todo).append(job)
        if k and k % 500 == 0:
            log.info("Checking existing outputs: %d/%d files", k, len(jobs))
    completed = sum(j["bytes"] for j in done)
    initial = completed
    remaining = plan["total_bytes"] - completed
    free = shutil.disk_usage(args.output).free
    if free < remaining + (5 << 30):
        raise OSError(f"Need {remaining / 2**30:.1f} GiB plus 5 GiB free; available {free / 2**30:.1f} GiB")
    total_done, failed = len(done), []
    log.info("Plan: %d tiles, %d files, %.2f GiB output; already complete %d files (%.2f GiB)",
             len(plan["tiles"]), len(jobs), plan["total_bytes"] / 2**30, total_done, completed / 2**30)
    if args.max_files:
        todo = todo[:args.max_files]
    context = mp.get_context("spawn")
    start, last_log, read_bytes = time.monotonic(), 0.0, 0
    status = {}
    with context.Manager() as manager, ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        events = manager.Queue()
        active = {}
        iterator = iter(todo)
        def submit():
            job = next(iterator, None)
            if job is not None:
                active[pool.submit(_download, job, str(args.output), args.threads, args.batch_frames, args.retries, events)] = job
                status[job["id"]] = (0, 0)
                log.info("START %s %s (%s, %d frames)", job["tile"], job["relative_file"], job["role"], job["count"])
        for _ in range(args.workers):
            submit()
        while active:
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            while True:
                try:
                    kind, jid, a, b = events.get_nowait()
                except queue.Empty:
                    break
                if jid not in status:
                    continue
                if kind == "progress":
                    status[jid] = (a, b)
                else:
                    log.warning("RETRY %s attempt %d: %s", jid[:12], a, b)
                    status[jid] = (0, 0)
            for future in finished:
                job = active.pop(future)
                status.pop(job["id"], None)
                try:
                    result = future.result()
                except Exception as e:
                    failed.append(job["output"])
                    log.error("FAILED %s: %s", job["output"], e)
                else:
                    total_done += 1
                    completed += job["bytes"]
                    read_bytes += result["bytes_requested"]
                    log.info("DONE %s %s — %d/%d files", job["tile"], job["relative_file"], total_done, len(jobs))
                submit()
            now = time.monotonic()
            if now - last_log >= args.log_interval or not active:
                in_flight = sum(j["bytes"] * status[j["id"]][0] / j["count"] for j in active.values())
                held = completed + in_flight
                rate = (held - initial) / max(now - start, 1e-6)
                eta = (plan["total_bytes"] - held) / rate if rate else float("inf")
                current = ", ".join(f"{j['tile']}/{j['relative_file']} {status[j['id']][0]}/{j['count']} frames" for j in active.values())
                log.info("PROGRESS %.2f%% | %d/%d files | %.2f/%.2f GiB | %.1f MiB/s output | elapsed %s | ETA %s | %s",
                         100 * held / plan["total_bytes"], total_done, len(jobs), held / 2**30, plan["total_bytes"] / 2**30,
                         rate / 2**20, _duration(now - start), _duration(eta), current or "idle")
                _atomic_json(args.output / "progress.json", dict(complete_files=total_done, total_files=len(jobs),
                             output_bytes=completed, planned_bytes=plan["total_bytes"], requested_bytes_this_run=read_bytes,
                             elapsed_seconds=now-start, failed=failed, finished=False))
                last_log = now
    for tile in args.tiles:
        pieces = [j for j in jobs if j["tile"] == tile]
        if all(_complete(args.output / j["output"], j) for j in pieces):
            _index_tile(args.output, tile, pieces)
            log.info("Tile %s complete: %s/index.h5", tile, tile)
    _atomic_json(args.output / "progress.json", dict(complete_files=total_done, total_files=len(jobs),
                 output_bytes=completed, planned_bytes=plan["total_bytes"], requested_bytes_this_run=read_bytes,
                 elapsed_seconds=time.monotonic()-start, failed=failed, finished=total_done == len(jobs)))
    log.info("%s: %d/%d files complete, %d failed. Re-run the same command to resume.",
             "FINISHED" if total_done == len(jobs) else "STOPPED", total_done, len(jobs), len(failed))
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=BASE / "projections/scale_0.125")
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--local-root", type=Path, default=BASE / "RAW_DATA")
    parser.add_argument("--slices", type=int, default=256, help="slab thickness in original-resolution Z slices")
    parser.add_argument("--reduce", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--center-mm", type=float, default=-256.0385, help="recorded motor-Z centre of the existing Paris4 ROI")
    parser.add_argument("--context-turns", type=float, default=1.5)
    parser.add_argument("--tiles", nargs="+", choices=TILES, default=TILES)
    parser.add_argument("--workers", type=int, default=4, help="concurrent files, separate processes")
    parser.add_argument("--threads", type=int, default=2, help="C++ decoding threads per file")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-interval", type=float, default=5)
    parser.add_argument("--plan-only", action="store_true", help="copy metadata and report the plan without reading projection pixels")
    parser.add_argument("--max-files", type=int, default=0, help="stop after this many pending files; 0 means all")
    args = parser.parse_args(argv)
    if min(args.workers, args.threads, args.batch_frames, args.slices) < 1 or args.retries < 0 or args.max_files < 0 or args.log_interval <= 0:
        parser.error("counts and log interval must be positive; retries/max-files must be nonnegative")
    if len(set(args.tiles)) != len(args.tiles):
        parser.error("duplicate tiles")
    if not _core.htj2k_float_frames_supported():
        parser.error("this requires helirec built with OpenJPH (HELIREC_OPENJPH=ON)")
    args.output = args.output.resolve()
    args.local_root = args.local_root.resolve()
    args.source_root = args.source_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl  # supported on the project's Ubuntu and macOS platforms
    with (args.output / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another downloader is using this output directory")
        log = logging.getLogger("helirec.download_projections")
        log.handlers.clear()
        log.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(args.output / "download.log")):
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(handler)
        try:
            log.info("Starting slab: %d original Z slices (%.6f mm at 1.129 µm), detector scale %.3f",
                     args.slices, args.slices * 0.001129, 2.0 ** -args.reduce)
            plan = build_plan(args, log)
            log.info("Plan ready: %d tiles, %d files, %.2f GB (%.2f GiB) uncompressed payload",
                     len(plan["tiles"]), len(plan["jobs"]), plan["total_bytes"] / 1e9, plan["total_bytes"] / 2**30)
            if args.plan_only:
                return 0
            return run_download(plan, args, log)
        except KeyboardInterrupt:
            log.warning("Interrupted; completed files are safe. Re-run the same command to resume.")
            return 130
        except Exception:
            log.exception("Download stopped; completed files are safe.")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
