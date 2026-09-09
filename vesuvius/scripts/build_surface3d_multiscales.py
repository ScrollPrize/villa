"""Build six volume-cartographer-compatible anisotropic OME-Zarr levels after native inference finishes; never uploads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from numcodecs import Blosc
import zarr

from download_surface_volumes import write_json
from surface3d_validity import verified_depth_validity
from vesuvius.label_zarr import create_v2_array, downsample_mean

LEVELS = 6
FACTORS = (1, 2, 2)


def native_ready(item, root):
    path = Path(item["output"]) / ".zattrs"
    if not path.exists() or not json.loads(path.read_text()).get("complete"):
        return False
    with sqlite3.connect(root / "jobs.sqlite", timeout=60) as connection:
        total, done = connection.execute(
            "SELECT COUNT(*),SUM(status='done') FROM jobs WHERE dataset=?", (item["id"],)).fetchone()
    return bool(total and total == done)


def file_inventory(path):
    """Track every native file's identity/size/mtime to detect accidental rewrites."""
    records = []
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            stat = entry.stat()
            records.append((str(entry.relative_to(path)), stat.st_size, stat.st_mtime_ns, stat.st_ino))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def verify_level_inventory(path, expected, *, required):
    """Reject altered/missing completed chunks instead of accepting Zarr fill zeros."""
    if expected is None:
        if required:
            raise ValueError(f"Pyramid level {path.name} has no completed inventory")
        return
    if not (path / ".zarray").is_file() or file_inventory(path) != expected:
        raise ValueError(f"Pyramid level {path.name} inventory mismatch")


def active_children(path, chunks, factors=FACTORS):
    """Map present slash-separated v2 chunks to XY-pooled target chunks (same chunks)."""
    metadata = json.loads((path / ".zarray").read_text())
    if metadata.get("dimension_separator", ".") != "/" or metadata.get("fill_value") != 0:
        raise ValueError("Sparse pooling requires slash-separated, zero-filled arrays")
    if tuple(metadata["chunks"]) != tuple(chunks):
        raise ValueError("Pyramid levels must retain the same storage chunk dimensions")
    result = set()
    if len(factors) != len(chunks):
        raise ValueError("Pooling factors must match array dimensions")
    for entry in path.glob("/".join(["*"] * len(chunks))):
        parts = entry.relative_to(path).parts
        if entry.is_file() and all(p.isdigit() for p in parts):
            result.add(tuple(int(p) // f for p, f in zip(parts, factors)))
    return sorted(result)


def make_metadata(attrs):
    result = deepcopy(attrs["multiscales"])
    for scale in result:
        base = next(d for d in scale["datasets"] if d["path"] == "0")
        datasets = []
        for level in range(LEVELS):
            dataset = deepcopy(base)
            dataset["path"] = str(level)
            transforms = dataset["coordinateTransformations"]
            transform = next(t for t in transforms if t["type"] == "scale")
            transform["scale"] = [v * (1 if axis == 0 else 2**level)
                                  for axis, v in enumerate(transform["scale"])]
            datasets.append(dataset)
        scale["datasets"] = datasets
        scale.pop("type", None)
        scale["metadata"] = {"downsampling_method": "mean"}
    return result


def populate_chunk(source, target, index, factors=FACTORS):
    starts = tuple(i*c for i,c in zip(index, target.chunks))
    stops = tuple(min(s+c,n) for s,c,n in zip(starts,target.chunks,target.shape))
    target_slice = tuple(slice(a,b) for a,b in zip(starts,stops))
    source_slice = tuple(slice(a*f,min(b*f,n)) for a,b,f,n in zip(starts,stops,factors,source.shape))
    block = source[source_slice]
    if block.ndim == 2:
        result = downsample_mean(block[None], (1,*factors), rounding="half_up")[0]
    else:
        result = downsample_mean(block, factors, rounding="half_up")
    target[target_slice] = result
    # Verify persisted, decompressed values rather than merely successful writes.
    if not np.array_equal(target[target_slice], result):
        raise RuntimeError(f"Pyramid readback failed at chunk {index}")
    return result.size


def build_one(item, root, workers=16, progress=lambda **kw: None):
    path = Path(item["output"])
    attrs = json.loads((path / ".zattrs").read_text())
    native = zarr.open_array(str(path / "0"), mode="r")
    if native.dtype != np.uint8 or list(native.shape) != item["shape"]:
        raise ValueError("Native array differs from the plan")
    before = file_inventory(path / "0")
    signature = {"native_inventory_sha256": before, "checkpoint_sha256": attrs["checkpoint_sha256"],
                 "levels": LEVELS, "factors_zyx": list(FACTORS), "method": "vc-mean-half-up-uint8-v2"}
    signature.update(verified_depth_validity(path, attrs, item["shape"]))
    require_inventories = True
    receipt_path = root / (item["id"] + "-multiscales-complete.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        matches = all(receipt.get(k) == v for k,v in signature.items())
        old_five = (receipt.get("levels") == 5 and receipt.get("method") == "mean-rint-uint8-v1"
                    and all(receipt.get(k) == signature[k] for k in ("native_inventory_sha256", "checkpoint_sha256")))
        if not matches and not old_five:
            raise ValueError("Existing pyramid was generated from different native data/settings")
        if matches and attrs.get("multiscales_complete") is True:
            inventories = receipt.get("level_inventory_sha256")
            for level in range(1, LEVELS):
                verify_level_inventory(path / str(level),
                    inventories.get(str(level)) if inventories is not None else None,
                    required=require_inventories or inventories is not None)
            return receipt
    started = time.time()
    kwargs = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
    group = zarr.open_group(str(path), mode="a", **kwargs)
    render_attrs = json.loads((Path(item["input"]) / ".zattrs").read_text())
    metadata = make_metadata(render_attrs)
    metadata[0]["name"] = item["id"] + "-3d-ink"
    native_metadata = deepcopy(metadata)
    native_metadata[0]["datasets"] = native_metadata[0]["datasets"][:1]
    # Hide lower levels during rebuild; native values and completion remain intact.
    group.attrs.update({"multiscales_complete": False, "pyramid_levels": LEVELS,
                        "multiscales": native_metadata})
    shape = native.shape
    timings = []
    for level in range(1, LEVELS):
        shape = tuple((n+f-1)//f for n,f in zip(shape,FACTORS))
        marker = root / f"{item['id']}-pyramid-vc6-level-{level}.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["signature"] != signature or list(group[str(level)].shape) != list(shape):
                raise ValueError("Pyramid resume marker mismatch")
            verify_level_inventory(path / str(level), saved.get("output_inventory_sha256"),
                                   required=require_inventories)
            timings.append(saved)
            continue
        # An interrupted level is rebuilt; already verified levels are reused.
        target = create_v2_array(group, str(level), shape=shape, chunks=native.chunks,
                                 dtype=np.uint8, compressor=Blosc(cname="lz4",clevel=1,shuffle=1), fill_value=0)
        target.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
        source = zarr.open_array(str(path / str(level-1)), mode="r")
        indices = active_children(path / str(level-1), native.chunks)
        level_start = time.time()
        voxels = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Queue only chunk indices; large blocks are loaded inside the bounded workers.
            for count, size in enumerate(pool.map(lambda index: populate_chunk(source,target,index), indices),1):
                voxels += size
                if count % 64 == 0 or count == len(indices):
                    progress(sample=item["id"],level=level,chunks_done=count,chunks_total=len(indices),
                             elapsed_seconds=time.time()-level_start)
        record = {"signature": signature, "level": level, "shape": list(shape),
                  "chunks": len(indices), "verified_voxels": voxels, "seconds": time.time()-level_start,
                  "output_inventory_sha256": file_inventory(path / str(level))}
        write_json(marker, record)
        timings.append(record)
    if file_inventory(path / "0") != before:
        raise RuntimeError("Native level-zero files changed during pyramid generation")
    if any(signature.get(k) != v for k, v in verified_depth_validity(path, attrs, item["shape"]).items()):
        raise RuntimeError("Depth validity changed during pyramid generation")
    group.attrs.update({"multiscales": metadata, "multiscales_complete": True,
                        "pyramid_levels": LEVELS, "pyramid_factors_zyx": list(FACTORS),
                        "pyramid_downsampling": "volume-cartographer recursive 2x2 XY mean, partial edges, half-up uint8 rounding",
                        "pyramid_method": signature["method"],
                        "num_slices": native.shape[0], "slice_step": render_attrs.get("slice_step", 1.0),
                        "canvas_size": [native.shape[2], native.shape[1]], "chunk_size": list(native.chunks),
                        "note_axes_order": "ZYX (slice, row, col)",
                        "native_inventory_sha256": before})
    receipt = {**signature, "sample": item["id"], "complete": True, "seconds": time.time()-started,
               "timings": timings, "verified": "every written chunk read back exactly; native file inventory unchanged"}
    # Keep old non-localization receipts compatible when resuming a legacy level
    # marker that predates inventories. New localization outputs require all five.
    if all("output_inventory_sha256" in record for record in timings):
        receipt["level_inventory_sha256"] = {
            str(record["level"]): record["output_inventory_sha256"] for record in timings}
    write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    plan = json.loads(args.plan.read_text())
    root = Path(plan["root"])
    lock = (root / "multiscales.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"complete": {}, "phase": "waiting_for_native_inference", "levels": LEVELS}
    def progress(**values):
        state.update(values, updated_unix=time.time())
        write_json(root / "multiscales-status.json", state)
        print(json.dumps({k:v for k,v in state.items() if k != "complete"}), flush=True)
    try:
        while len(state["complete"]) < len(plan["datasets"]):
            for item in plan["datasets"]:
                if item["id"] in state["complete"] or not native_ready(item,root):
                    continue
                progress(phase="building", sample=item["id"])
                receipt = build_one(item,root,args.workers,progress)
                state["complete"][item["id"]] = receipt
                progress(phase="verified",sample=item["id"])
                identity = root / "wandb-run.json"
                if identity.exists():
                    try:
                        import wandb
                        run_id = json.loads(identity.read_text())["id"]
                        run = wandb.Api(timeout=30).run(f"{plan['wandb_entity']}/{plan['wandb_project']}/{run_id}")
                        for key in ("levels", "seconds", "verified", "complete", "factors_zyx"):
                            run.summary[f"multiscales/{item['id']}/{key}"] = receipt[key]
                        run.summary.update()
                    except Exception as error:
                        print(json.dumps({"wandb_warning": type(error).__name__}),flush=True)
            if len(state["complete"]) < len(plan["datasets"]):
                progress(phase="waiting_for_native_inference",sample=None)
                time.sleep(30)
        progress(phase="complete",sample=None)
        write_json(root / "multiscales-complete.json",state)
    except BaseException as error:
        progress(phase="failed",error=repr(error))
        raise


if __name__ == "__main__":
    main()
