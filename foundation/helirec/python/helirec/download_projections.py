"""Download a physical BLISS slab as reduced detector frames: uncompressed HDF5 files in the
output folder, or (``--zarr STORE``) straight into a level of a projection store, compressed.

Run with ``python -m helirec.download_projections --help``. Image reads, HTJ2K
resolution selection, slab planning and the HDF5 and Zarr frame writes use the C++ core.
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
import random
import shutil
import subprocess
import sys
import tempfile
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


def _copy_metadata(src, dst, log, fetch=None):
    """Copy only the master (never follow its virtual frame datasets). `fetch(path)`, when given, writes the
    whole master to the new file `path` instead of reading `src` (--remote)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    digest = hashlib.sha256()
    if not dst.exists() and fetch is not None:
        tmp.unlink(missing_ok=True)
        fetch(str(tmp))
        os.replace(tmp, dst)
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


def _levels(job):
    return job.get("reductions", [job["reduce"]])


def _level_job(job, reduce):
    level = {k: v for k, v in job.items() if k != "reductions"}
    level["reduce"] = reduce
    level["bytes"] = job["count"] * (job["rows"] >> reduce) * (job["cols"] >> reduce) * np.dtype(job["dtype"]).itemsize
    return level


def build_plan(args, log):
    settings = dict(format_version=FORMAT, tiles=args.tiles, slices=args.slices, reduce=args.reduce,
                    center_mm=args.center_mm, context_turns=args.context_turns,
                    source_root=str(args.source_root), local_root=str(args.local_root))
    reductions = getattr(args, "reductions", [args.reduce])
    if len(reductions) > 1:
        settings["reductions"] = reductions
    if getattr(args, "scans", None):
        settings["scans"] = args.scans
    if getattr(args, "zarr", None):
        settings["zarr"] = str(args.zarr)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["settings"] != settings:
            raise ValueError("Output already has a different plan; use a new output directory or the original settings")
        log.info("Resuming existing plan: %d files", len(plan["jobs"]))
        return plan
    jobs, tiles = [], []
    factor = 1 << max(reductions)
    link = _remote(args)
    for number, tile in enumerate(args.tiles, 1):
        log.info("Planning tile %d/%d: %s", number, len(args.tiles), tile)
        label = tile if getattr(args, "scans", None) else f"1.129um_mosaic_PHerc_Paris4_ROI-1_{tile}"
        relative = Path(label) / (label + "_0001")
        local, remote = args.local_root / relative, args.source_root / relative
        master_name = label + "_0001.h5"
        source_master = local / master_name if (local / master_name).is_file() else remote / master_name
        master_copy = args.output / "metadata" / tile / master_name
        # --remote: what is not local comes over SFTP, the plan included (no mount of --source-root is read)
        over_sftp = link is not None and not (local / master_name).is_file()
        master_sha = _copy_metadata(source_master, master_copy, log, fetch=(lambda path: _fetch(
            _connect(link["spec"]), _remote_path(link, source_master)[0], path, None, None)) if over_sftp else None)
        p = _core.plan_projection_subset(str(master_copy), args.center_mm, args.slices, args.context_turns)
        if p["rows"] % factor or p["cols"] % factor:
            raise ValueError(f"{tile}: detector shape is not divisible by {factor}")
        projection = next(s for s in p["series"] if s["role"] == "projection")
        sample = next((p for p in projection["pieces"] if (local / p["file"]).is_file()), projection["pieces"][0])
        candidates = [str(local / sample["file"]), str(remote / sample["file"])]
        with tempfile.TemporaryDirectory(prefix="plan-", dir=args.output) as tmp:
            if link is not None and not Path(candidates[0]).is_file():  # the chunk headers are enough
                candidates = [str(Path(tmp) / "sample.h5")]
                _fetch(_connect(link["spec"]), _remote_path(link, remote / sample["file"])[0], candidates[0],
                       sample["dataset"], _HEADS[3][0])
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
                       stored_shape=[p["rows"], p["cols"]], output_shape=[p["rows"] >> args.reduce, p["cols"] >> args.reduce],
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
                if len(reductions) > 1:
                    job["reductions"] = reductions
                job["bytes"] = sum(_level_job(job, r)["bytes"] for r in reductions)
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


def _scan_group(store, job):
    return str(Path(store) / job["tile"] / Path(job["relative_file"]).parent.name)


def _declare_store(plan, args, log):
    """The scan groups of the plan in the store, with this plan's level recorded as stored. The frames must be the
    ones the store already holds for a scan (the C++ side refuses others). Done here, once, before the workers."""
    tiles = {t["tile"]: t for t in plan["tiles"]}
    scans = {}
    for job in plan["jobs"]:
        scans.setdefault(_scan_group(args.zarr, job), []).append(job)
    for group, jobs in scans.items():
        jobs = sorted(jobs, key=lambda j: j["first"])
        first, end = jobs[0]["first"], jobs[-1]["first"] + jobs[-1]["count"]
        values = {}
        for job in jobs:
            for key, v in job["metadata"].items():
                values.setdefault(key, [None] * (end - first))[job["first"] - first:job["first"] - first + job["count"]] = v
        tile = tiles[jobs[0]["tile"]]
        spec = dict(tile=jobs[0]["tile"], scan=Path(group).name, role=jobs[0]["role"], rows=jobs[0]["rows"], cols=jobs[0]["cols"],
                    frame_first=first, frames=end - first,
                    tile_attributes={k: v for k, v in tile.items() if k not in ("source_master", "output_shape", "slices")},
                    files=[dict(file=Path(j["relative_file"]).name, source_first=j["source_first"], first=j["first"], count=j["count"]) for j in jobs],
                    frame_values=values)
        for reduce in getattr(args, "reductions", [args.reduce]):
            _core.declare_projection_store_scan(str(args.zarr), json.dumps(spec), reduce, args.remote or str(args.source_root), jobs[0]["dtype"])
    log.info("Store %s: %d scan groups declare levels %s", args.zarr, len(scans), getattr(args, "reductions", [args.reduce]))


def _complete(path, job, store=None):
    if store is not None:  # a file is one write of its shard
        return all(_core.ProjectionStoreWriter(_scan_group(store, job), r).holds(job["first"], job["count"]) for r in _levels(job))
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


class _StoreFrames:
    """The frames of one file, held until they are all there and then written as one shard of the store."""

    def __init__(self, store, job, rows, cols):
        self.writer, self.first = _core.ProjectionStoreWriter(_scan_group(store, job), job["reduce"]), job["first"]
        self.data = np.empty((job["count"], rows, cols), job["dtype"])

    def write(self, first, data):
        self.data[first:first + len(data)] = data

    def close(self):
        pass


class _LevelEvents:
    """Report a level's progress against the total output bytes of its source-file job."""
    def __init__(self, events, job, before, size):
        self.events, self.job, self.before, self.size = events, job, before, size

    def put(self, event):
        kind, jid, count, read = event
        if kind == "progress":
            count = (self.before * self.job["count"] + self.size * count) // self.job["bytes"]
        self.events.put((kind, jid, count, read))


def _download_once(job, output, threads, batch, events, local=None, fetched=None, origin=None, store=None):
    if len(_levels(job)) == 1:
        return _download_level(job, output, threads, batch, events, local, fetched, origin, store)
    if store is None:
        raise ValueError("multiple reductions require --zarr")
    pending = [j for r in _levels(job) if not _complete(Path(output) / job["output"], j := _level_job(job, r), store)]
    if not pending:
        return dict(bytes_requested=0, resumed=True)
    source = local if local is not None else _source(job)
    before = job["bytes"] - sum(j["bytes"] for j in pending)
    requested, start = 0, time.monotonic()
    for level in pending:
        progress = _LevelEvents(events, job, before, level["bytes"])
        result = _download_level(level, output, threads, batch, progress, source, fetched, origin, store)
        requested += result["bytes_requested"]
        before += level["bytes"]
    return dict(bytes_requested=requested, seconds=time.monotonic() - start, resumed=False)


def _download_level(job, output, threads, batch, events, local=None, fetched=None, origin=None, store=None):
    """local: a (partial) local copy of the source to read instead; fetched: its byte ranges that hold data;
    store: the projection store the frames go into instead of an HDF5 file."""
    from .calibration import at_resolution
    path = Path(output) / job["output"]
    if _complete(path, job, store):
        return dict(bytes_requested=0, resumed=True)
    tmp = path.with_name(path.name + ".partial")
    if store is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.unlink(missing_ok=True)  # only this downloader's unpublished temporary file
    source = local if local is not None else _source(job)
    projection = job["role"] == "projection"
    reader = _core.RawReader([source], job["dataset"], job["rows"], job["cols"],
                             reduce=job["reduce"] if projection else 0, decoder="openjph" if projection else "auto")
    if fetched is not None:
        reader.set_fetched(0, *fetched)
    storage = reader.storage()
    if projection and (not storage["compressed"] or storage["reversible"] or storage["chunk_rows"] != job["chunk_rows"]):
        raise ValueError(f"Projection storage differs from the planned calibration grid: {source}")
    if job["source_first"] + job["count"] > storage["frames"]:
        raise ValueError(f"Source has fewer frames than the master records: {source}")
    rows, cols = job["rows"] >> job["reduce"], job["cols"] >> job["reduce"]
    writer = (_core.UncompressedFrameWriter(str(tmp), job["dataset"], job["count"], rows, cols, job["dtype"]) if store is None
              else _StoreFrames(store, job, rows, cols))
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
    if store is not None:  # the whole shard at once, read back and compared
        writer.writer.write(writer.first, writer.data)
        return dict(bytes_requested=reader.bytes_read, seconds=time.monotonic() - start, resumed=False)
    with h5py.File(tmp, "r+") as f:
        f.attrs.update(complete=True, job_sha256=job["id"], data_sha256=digest.hexdigest(), source=origin or source,
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


# --remote: one SSH connection per worker process, kept between files. Each new login costs seconds, and the
# ESRF server resets logins that start together. At its ~120 ms round trip, file reads through a mount are
# latency bound; paramiko's readv instead keeps all of a file's chunk requests in flight on one open handle.
_CONNECTION = None
# Bytes of each chunk's start to fetch, by `reduce`, tried in turn; None: the whole file. A Paris4 chunk (162 KB)
# decodes from its first 16.7, 24.8 and 57.7 KB at reduce 3, 2 and 1 (measured on five files of four tiles).
_HEADS = {3: (20 << 10, 64 << 10, None), 2: (28 << 10, 64 << 10, None), 1: (64 << 10, None)}
_TAIL = 1 << 10  # the decoder reads the codestream's end marker too


def _rclone_remote(spec):
    name, _, path = spec.partition(":")
    config = json.loads(subprocess.run(["rclone", "config", "dump"], capture_output=True, check=True).stdout)
    if config.get(name, {}).get("type") != "sftp":
        raise ValueError(f"--remote: {name!r} is not an SFTP remote in the rclone config")
    return config[name], path


def _remote(args):
    """--remote as the workers take it: the rclone spec, its path, and the --source-root it stands for."""
    return (dict(spec=args.remote, path=_rclone_remote(args.remote)[1], source_root=str(args.source_root))
            if args.remote else None)


def _remote_path(remote, source):
    """A file under --source-root: its path on the SFTP server and its origin as recorded in outputs."""
    relative = Path(source).relative_to(remote["source_root"]).as_posix()
    return remote["path"].rstrip("/") + "/" + relative, remote["spec"].rstrip("/") + "/" + relative


def _connect(spec):
    global _CONNECTION
    if _CONNECTION is not None and _CONNECTION[0].get_transport() is not None and _CONNECTION[0].get_transport().is_active():
        return _CONNECTION[1]
    import paramiko
    cfg, _ = _rclone_remote(spec)
    client = paramiko.SSHClient()
    client.load_host_keys(os.path.expanduser(cfg.get("known_hosts_file") or "~/.ssh/known_hosts"))
    kwargs = dict(hostname=cfg["host"], port=int(cfg.get("port") or 22), username=cfg.get("user") or None,
                  timeout=30, banner_timeout=60, auth_timeout=60)
    if cfg.get("key_file"):
        kwargs["key_filename"] = os.path.expanduser(cfg["key_file"])
    if cfg.get("pass"):
        kwargs["password"] = subprocess.run(["rclone", "reveal", cfg["pass"]], capture_output=True, check=True,
                                            text=True).stdout.strip()
    # paramiko prints a traceback for every refused login from its transport thread; failures surface here instead
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)
    import fcntl
    lock_path = Path(tempfile.gettempdir()) / f"helirec-sftp-login-{hashlib.sha256(spec.encode()).hexdigest()[:12]}.lock"
    for attempt in range(8):
        # The server resets logins that start together: the workers log in one at a time.
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                client.connect(**kwargs)
                break
            except paramiko.AuthenticationException:
                raise
            except (paramiko.SSHException, OSError, EOFError):
                client.close()
                if attempt == 7:
                    raise
            finally:
                time.sleep(0.5)  # before the next worker's login
        time.sleep(min(30, 2 ** attempt) * random.uniform(0.5, 1.5))
    _CONNECTION = (client, client.open_sftp())
    return _CONNECTION[1]


def _drop_connection():
    global _CONNECTION
    if _CONNECTION is not None:
        _CONNECTION[0].close()
        _CONNECTION = None


def _fetch(sftp, remote_path, local_path, dataset, head):
    """Copy the HDF5 metadata and the first `head` + last _TAIL bytes of each chunk of `dataset` into a sparse
    local file (None: the whole file). Returns the chunk ranges fetched (None if whole) and bytes transferred."""
    import io
    block = 1 << 16
    with sftp.open(remote_path, "rb") as f:
        size = f.stat().st_size
        out = os.open(local_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.ftruncate(out, size)
            if head is None:
                pieces = [(o, min(1 << 20, size - o)) for o in range(0, size, 1 << 20)]
                for (o, _), data in zip(pieces, f.readv(pieces)):
                    os.pwrite(out, data, o)
                return None, size
            blocks = {}

            class Metadata(io.RawIOBase):  # what libhdf5 reads to find the chunks, in 64 KiB blocks
                pos = 0
                def readable(self): return True
                def seekable(self): return True
                def tell(self): return self.pos
                def seek(self, o, whence=0):
                    self.pos = o if whence == 0 else self.pos + o if whence == 1 else size + o
                    return self.pos
                def readinto(self, b):
                    n, got = min(len(b), max(0, size - self.pos)), bytearray()
                    while len(got) < n:
                        k, o = divmod(self.pos + len(got), block)
                        if k not in blocks:
                            n_k = min(block, size - k * block)  # readv: its pieces in flight together
                            blocks[k] = b"".join(f.readv([(k * block, n_k)]))
                        got += blocks[k][o:o + n - len(got)]
                    b[:n] = got
                    self.pos += n
                    return n

            with h5py.File(Metadata(), "r") as h:
                d = h[dataset].id
                chunks = [d.get_chunk_info(i) for i in range(d.get_num_chunks())]
            for k, data in blocks.items():
                os.pwrite(out, data, k * block)
            ranges = []
            for c in chunks:
                ranges.append((c.byte_offset, min(head, c.size)))
                if c.size > head:
                    tail = min(_TAIL, c.size - head)
                    ranges.append((c.byte_offset + c.size - tail, tail))
            for (o, _), data in zip(ranges, f.readv(ranges)):
                os.pwrite(out, data, o)
            return ([o for o, _ in ranges], [n for _, n in ranges]), sum(map(len, blocks.values())) + sum(n for _, n in ranges)
        finally:
            os.close(out)


def _download_remote(job, output, threads, batch, events, remote, store=None):
    if _complete(Path(output) / job["output"], job, store):
        return dict(bytes_requested=0, resumed=True)
    remote_path, origin = _remote_path(remote, job["sources"][-1])
    work = Path(output) / ".remote"
    work.mkdir(parents=True, exist_ok=True)
    pending = [r for r in _levels(job) if not _complete(Path(output) / job["output"], _level_job(job, r), store)]
    transferred_total = 0
    for head in (_HEADS[min(pending)] if job["role"] == "projection" else (None,)):
        with tempfile.TemporaryDirectory(prefix="source-", dir=work) as tmp:
            local = str(Path(tmp) / "source.h5")
            fetched, transferred = _fetch(_connect(remote["spec"]), remote_path, local, job["dataset"], head)
            transferred_total += transferred
            try:
                result = _download_once(job, output, threads, batch, events, local=local, fetched=fetched, origin=origin, store=store)
            except Exception as e:
                if head is None:
                    raise
                # bytes beyond the fetched ones (or unread metadata): fetch more of each chunk
                events.put(("retry", job["id"], 0, f"{head} bytes per chunk were not enough ({e}); fetching more"))
                continue
            result["bytes_transferred"] = transferred_total
            return result


def _download(job, output, threads, batch, retries, events, remote=None, store=None):
    for attempt in range(retries + 1):
        try:
            if remote is not None and not Path(job["sources"][0]).is_file():
                return _download_remote(job, output, threads, batch, events, remote, store)
            return _download_once(job, output, threads, batch, events, store=store)
        except Exception as e:
            if attempt == retries:
                raise
            if remote is not None:
                _drop_connection()
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
    store = str(args.zarr) if getattr(args, "zarr", None) else None
    if store:
        _declare_store(plan, args, log)
    done, todo, missing_bytes = [], [], {}
    for k, job in enumerate(jobs):
        missing_bytes[job["id"]] = sum(level["bytes"] for r in _levels(job)
            if not _complete(args.output / job["output"], level := _level_job(job, r), store))
        (todo if missing_bytes[job["id"]] else done).append(job)
        if k and k % 500 == 0:
            log.info("Checking existing outputs: %d/%d files", k, len(jobs))
    completed = plan["total_bytes"] - sum(missing_bytes.values())
    initial = completed
    remaining = plan["total_bytes"] - completed
    if args.max_files:  # only what this run writes
        remaining = sum(missing_bytes[j["id"]] for j in todo[:args.max_files])
    free = shutil.disk_usage(args.zarr if store else args.output).free
    # in the store the frames are compressed: 1.3x on Paris4 frames (bitshuffle + lz4), darks and flats less
    needed = remaining / 1.25 if store else remaining
    if free < needed + (5 << 30):
        raise OSError(f"Need {needed / 2**30:.1f} GiB plus 5 GiB free; available {free / 2**30:.1f} GiB")
    total_done, failed = len(done), []
    log.info("Plan: %d tiles, %d files, %.2f GiB output; already complete %d files (%.2f GiB)",
             len(plan["tiles"]), len(jobs), plan["total_bytes"] / 2**30, total_done, completed / 2**30)
    if args.max_files:
        todo = todo[:args.max_files]
    context = mp.get_context("spawn")
    start, last_log, read_bytes = time.monotonic(), 0.0, 0
    status = {}
    remote = _remote(args)
    with context.Manager() as manager, ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        events = manager.Queue()
        active = {}
        iterator = iter(todo)
        def submit():
            job = next(iterator, None)
            if job is not None:
                active[pool.submit(_download, job, str(args.output), args.threads, args.batch_frames, args.retries, events, remote, store)] = job
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
                    completed += missing_bytes[job["id"]]
                    read_bytes += result["bytes_requested"]
                    log.info("DONE %s %s — %d/%d files", job["tile"], job["relative_file"], total_done, len(jobs))
                submit()
            now = time.monotonic()
            if now - last_log >= args.log_interval or not active:
                in_flight = sum(max(0, j["bytes"] * status[j["id"]][0] / j["count"]
                                   - (j["bytes"] - missing_bytes[j["id"]])) for j in active.values())
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
    for tile in ([] if store else args.tiles):  # (the store needs no index: its scan groups are whole arrays)
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
    parser.add_argument("--reduce", type=int, nargs="+", choices=[1, 2, 3], default=[3],
                        help="resolution levels: 1=half, 2=quarter, 3=eighth; with --zarr, multiple levels share each source transfer")
    parser.add_argument("--center-mm", type=float, default=-256.0385, help="recorded motor-Z centre of the existing Paris4 ROI")
    parser.add_argument("--context-turns", type=float, default=1.5)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tiles", nargs="+", choices=TILES, default=TILES)
    selection.add_argument("--scans", nargs="+", help="acquisition folder names under the source root, instead of mosaic tiles; "
                           "each must contain NAME_0001/NAME_0001.h5. Names also identify groups in the output store.")
    parser.add_argument("--workers", type=int, default=4, help="concurrent files, separate processes")
    parser.add_argument("--threads", type=int, default=2, help="C++ decoding threads per file")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--remote", help="rclone SFTP remote:path matching --source-root (e.g. esrf:/data/.../RAW_DATA): "
                        "read sources missing locally over SFTP, fetching only the bytes the reduced decode reads; "
                        "host, user and password come from the rclone config. Nothing under --source-root is then read: it only names the files.")
    parser.add_argument("--zarr", type=Path, help="a projection store (helirec-projections-zarr): write the frames into its level "
                        "--reduce, compressed, instead of HDF5 files; --output then holds only the plan, the masters and the logs. "
                        "A scan already in the store must cover the same frames (same --center-mm, --slices, --context-turns).")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-interval", type=float, default=5)
    parser.add_argument("--plan-only", action="store_true", help="copy metadata and report the plan without reading projection pixels")
    parser.add_argument("--max-files", type=int, default=0, help="stop after this many pending files; 0 means all")
    args = parser.parse_args(argv)
    if len(set(args.reduce)) != len(args.reduce):
        parser.error("duplicate reduction levels")
    args.reductions = sorted(args.reduce)
    args.reduce = args.reductions[0]
    if len(args.reductions) > 1 and not args.zarr:
        parser.error("multiple reduction levels require --zarr")
    if args.scans:
        if any(name in (".", "..") or "/" in name or "\\" in name for name in args.scans):
            parser.error("--scans requires folder names, not paths")
        args.tiles = args.scans
    if args.remote:
        try:
            import paramiko  # noqa: F401
        except ImportError:
            parser.error("--remote requires paramiko")
        if shutil.which("rclone") is None:
            parser.error("--remote reads its connection from the rclone config; rclone is not on PATH")
    if min(args.workers, args.threads, args.batch_frames, args.slices) < 1 or args.retries < 0 or args.max_files < 0 or args.log_interval <= 0:
        parser.error("counts and log interval must be positive; retries/max-files must be nonnegative")
    if len(set(args.tiles)) != len(args.tiles):
        parser.error("duplicate tiles")
    if not _core.htj2k_float_frames_supported():
        parser.error("this requires helirec built with OpenJPH (HELIREC_OPENJPH=ON)")
    args.output = args.output.resolve()
    args.local_root = args.local_root.resolve()
    args.source_root = args.source_root.resolve()
    if args.zarr:
        args.zarr = args.zarr.resolve()
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
            log.info("Starting slab: %d original Z slices, detector scales %s; physical thickness comes from each scan's pixel size",
                     args.slices, [2.0 ** -r for r in args.reductions])
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
