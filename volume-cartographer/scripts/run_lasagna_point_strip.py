#!/usr/bin/env python3
"""Submit Lasagna new-point-strip jobs from a VC3D point-collections JSON file.

This is a small client for an already-running Lasagna fit service. It mirrors
the VC3D request shape closely enough for new_point_strip.json, but sends one
request per point collection and uses the middle point in the ordered strip as
the seed.
"""

import argparse
import copy
import io
import json
import math
import sys
import tarfile
import shutil
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


API_VERSION_HEADER = "X-Fit-Service-API-Version"
API_VERSION = "2"
DEFAULT_SERVICE_URL = "http://127.0.0.1:9999"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "lasagna" / "configs" / "new_point_strip.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_progress(path):
    path = Path(path)
    if not path.is_file():
        return {"version": 1, "completed": {}}

    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"progress file must contain a JSON object: {path}")

    completed = data.get("completed")
    if not isinstance(completed, dict):
        completed = {}
    data["completed"] = completed
    data.setdefault("version", 1)
    return data


def write_progress(path, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def record_completed_collection(progress, *, collection_info, job_id, output_name, placed):
    progress.setdefault("completed", {})[collection_info["id"]] = {
        "collection_id": collection_info["id"],
        "collection_name": collection_info["name"],
        "center_point_id": collection_info["center_point_id"],
        "point_count": collection_info["point_count"],
        "job_id": job_id,
        "requested_output_name": output_name,
        "extracted": list(placed),
        "completed_at": utc_timestamp(),
    }


def record_failed_collection(progress, *, collection_info, job_id, output_name, error):
    progress.setdefault("failed", {})[collection_info["id"]] = {
        "collection_id": collection_info["id"],
        "collection_name": collection_info["name"],
        "center_point_id": collection_info["center_point_id"],
        "point_count": collection_info["point_count"],
        "job_id": job_id,
        "requested_output_name": output_name,
        "error": str(error),
        "failed_at": utc_timestamp(),
    }



def has_tifxyz_channels(path):
    return path.is_dir() and all((path / name).is_file() for name in ("x.tif", "y.tif", "z.tif"))


def write_wrapped_shell_copy(src, dst):
    import tifffile
    import numpy as np

    dst.mkdir(parents=True, exist_ok=True)
    for name in ("x.tif", "y.tif", "z.tif"):
        arr = tifffile.imread(src / name)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"cannot wrap {src / name}: expected 2D array with W>=2, got {arr.shape}")
        wrapped = np.concatenate([arr, arr[:, :1]], axis=1)
        tifffile.imwrite(dst / name, wrapped)
    meta_src = src / "meta.json"
    if meta_src.is_file():
        shutil.copy2(meta_src, dst / "meta.json")


def first_last_columns_match(path):
    import tifffile
    import numpy as np

    arrays = [tifffile.imread(path / name) for name in ("x.tif", "y.tif", "z.tif")]
    return all(np.allclose(arr[:, 0], arr[:, -1], atol=1.0e-3, rtol=1.0e-5) for arr in arrays)



def bbox_distance2(point, bbox):
    if not isinstance(bbox, list) or len(bbox) != 2:
        return float("inf")
    lo, hi = bbox
    if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) >= 3 and len(hi) >= 3):
        return float("inf")
    total = 0.0
    for i in range(3):
        v = float(point[i])
        a = float(lo[i])
        b = float(hi[i])
        if v < a:
            total += (a - v) ** 2
        elif v > b:
            total += (v - b) ** 2
    return total


def shell_candidates(root):
    native = [p for p in sorted(root.glob("shell_*.tifxyz")) if has_tifxyz_channels(p)]
    if native:
        return native, True
    return [p for p in sorted(root.iterdir()) if has_tifxyz_channels(p)], False


def select_shells(candidates, seed_base, max_shells):
    if max_shells is None or max_shells <= 0 or len(candidates) <= max_shells:
        return candidates
    ranked = []
    for path in candidates:
        meta = {}
        meta_path = path / "meta.json"
        if meta_path.is_file():
            try:
                meta = load_json(meta_path)
            except Exception:
                meta = {}
        ranked.append((bbox_distance2(seed_base, meta.get("bbox")), path.name, path))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max_shells]]


def prepare_shell_dir(init_shell_dir, work_root, *, collection_id, seed_base, max_shells):
    root = Path(init_shell_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"init shell dir does not exist: {root}")

    candidates, native = shell_candidates(root)
    if not candidates:
        raise ValueError(f"init shell dir contains no tifxyz shell directories: {root}")

    selected = select_shells(candidates, seed_base, max_shells)
    view_root = Path(work_root) / f"collection_{collection_id}" / "init_shells"
    if view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for idx, src in enumerate(selected):
        dst = view_root / f"shell_{idx:04d}.tifxyz"
        if native or first_last_columns_match(src):
            dst.symlink_to(src, target_is_directory=True)
        else:
            write_wrapped_shell_copy(src, dst)
            copied += 1

    print(
        f"collection {collection_id}: using {len(selected)}/{len(candidates)} init shell(s) "
        f"near seed=({seed_base[0]:.1f}, {seed_base[1]:.1f}, {seed_base[2]:.1f}) at {view_root}"
    )
    if copied:
        print(f"collection {collection_id}: wrapped-copy shells={copied}")
    return str(view_root)


def request_import_scale(manifest, volume_shape_zyx):
    base = manifest.get("base_shape_zyx")
    if not volume_shape_zyx or not isinstance(base, list) or len(base) != 3:
        return 1.0
    ratios = [float(b) / float(s) for b, s in zip(base, volume_shape_zyx)]
    return sum(ratios) / 3.0


def absolutized_manifest(source_path):
    source_path = Path(source_path).expanduser().resolve()
    manifest = load_json(source_path)
    source_parent = source_path.parent

    def absolutize(value):
        if not isinstance(value, str) or not value:
            return value
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (source_parent / p).resolve())

    if "umbilicus_json" in manifest:
        manifest["umbilicus_json"] = absolutize(manifest["umbilicus_json"])
    groups = manifest.get("groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            if "zarr" in group:
                group["zarr"] = absolutize(group["zarr"])
            if "zarr_path" in group:
                group["zarr_path"] = absolutize(group["zarr_path"])
    return manifest


def prepare_data_input_manifest(data_input, init_shell_dir, work_root, *, collection_info, volume_shape_zyx, max_shells):
    if init_shell_dir is None:
        return str(data_input)

    manifest = absolutized_manifest(data_input)
    scale = request_import_scale(manifest, volume_shape_zyx)
    seed_base = [float(v) * scale for v in collection_info["seed"]]
    manifest["init_shell_dir"] = prepare_shell_dir(
        init_shell_dir,
        work_root,
        collection_id=collection_info["id"],
        seed_base=seed_base,
        max_shells=max_shells,
    )

    out_dir = Path(work_root) / f"collection_{collection_info['id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "point_strip.lasagna.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"collection {collection_info['id']}: using Lasagna manifest {out}")
    return str(out)

def numeric_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def point_position(point, *, context):
    pos = point.get("p")
    if not isinstance(pos, list) or len(pos) < 3:
        raise ValueError(f"{context} has no valid p=[x,y,z]")
    try:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has non-numeric p={pos!r}") from exc


def sanitize_points(points_obj, *, missing_wind_a):
    if not isinstance(points_obj, dict):
        raise ValueError("collection points must be an object keyed by point id")

    out = {}
    for point_id in sorted(points_obj, key=numeric_key):
        point = copy.deepcopy(points_obj[point_id])
        if not isinstance(point, dict):
            continue
        point_position(point, context=f"point {point_id}")
        wind_a = point.get("wind_a")
        if wind_a is None:
            if missing_wind_a == "skip":
                continue
            if missing_wind_a == "error":
                raise ValueError(f"point {point_id} has null/missing wind_a")
            point["wind_a"] = 0.0
        else:
            try:
                wind_a = float(wind_a)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"point {point_id} has non-numeric wind_a={wind_a!r}") from exc
            if not math.isfinite(wind_a):
                if missing_wind_a == "skip":
                    continue
                if missing_wind_a == "error":
                    raise ValueError(f"point {point_id} has non-finite wind_a={wind_a!r}")
                wind_a = 0.0
            point["wind_a"] = wind_a
        out[str(point_id)] = point

    return out


def collections_from_file(path, *, missing_wind_a):
    data = load_json(path)
    collections = data.get("collections")
    if not isinstance(collections, dict):
        raise ValueError("point-collections JSON must contain a collections object")

    parsed = []
    for collection_id in sorted(collections, key=numeric_key):
        collection = collections[collection_id]
        if not isinstance(collection, dict):
            continue
        raw_points = collection.get("points")
        points = sanitize_points(raw_points, missing_wind_a=missing_wind_a)
        ordered_ids = sorted(points, key=numeric_key)
        if not ordered_ids:
            print(f"warning: skipping collection {collection_id}: no usable points", file=sys.stderr)
            continue

        center_id = ordered_ids[len(ordered_ids) // 2]
        seed = point_position(points[center_id], context=f"collection {collection_id} center point {center_id}")

        single_collection = copy.deepcopy(collection)
        single_collection["points"] = points
        single_collection.setdefault("metadata", {})
        single_collection.setdefault("name", f"collection_{collection_id}")

        parsed.append({
            "id": str(collection_id),
            "name": str(single_collection.get("name") or f"collection_{collection_id}"),
            "center_point_id": str(center_id),
            "seed": seed,
            "corr_points": {"collections": {str(collection_id): single_collection}},
            "point_count": len(points),
        })

    if not parsed:
        raise ValueError("no collections found")
    return parsed


def request_json(service_url, path, body=None, *, timeout=30):
    url = urljoin(service_url.rstrip("/") + "/", path.lstrip("/"))
    data = None
    method = "GET"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        method = "POST"
    req = Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            API_VERSION_HEADER: API_VERSION,
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    return json.loads(payload.decode("utf-8"))


def request_bytes(service_url, path, *, timeout=300):
    url = urljoin(service_url.rstrip("/") + "/", path.lstrip("/"))
    req = Request(url, method="GET", headers={API_VERSION_HEADER: API_VERSION})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def safe_extract_tar_gz(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    placed = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        base = output_dir.resolve()
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"refusing to extract path outside output dir: {member.name}")
        tar.extractall(output_dir)
        for member in tar.getmembers():
            name = member.name.split("/", 1)[0]
            if name and name not in placed:
                placed.append(name)
    return placed


def build_request(*, base_config, data_input, width, height, windings, collection_info, output_name, source, volume_shape_zyx=None):
    config = copy.deepcopy(base_config)
    args = dict(config.get("args") or {})
    args["seed"] = collection_info["seed"]
    args["model-w"] = float(width)
    args["model-w-unit"] = "voxels"
    args["model-h"] = float(height)
    args["windings"] = int(windings)
    config["args"] = args
    config["corr_points"] = collection_info["corr_points"]

    job_spec = {
        "config": config,
        "linked_surfaces": [],
    }
    request = {
        "source": source,
        "data_input": str(data_input),
        "single_segment": True,
        "copy_model": True,
        "config_name": "new_point_strip.json",
        "output_name": output_name,
        "config": config,
        "job_spec": job_spec,
    }
    if volume_shape_zyx is not None:
        shape = [int(v) for v in volume_shape_zyx]
        request["volume_shape_zyx"] = shape
        job_spec["volume_shape_zyx"] = shape
    return request


def wait_for_job(service_url, job_id, *, poll_interval, timeout):
    started = time.monotonic()
    last_line = ""
    while True:
        status = request_json(service_url, f"/jobs/{job_id}", timeout=30)
        state = status.get("state")
        stage = status.get("stage") or ""
        stage_name = status.get("stage_name") or ""
        step = status.get("step", 0)
        total = status.get("total_steps", 0)
        line = f"job {job_id}: {state}"
        if stage or stage_name:
            line += f" {stage_name or stage}"
        if total:
            line += f" {step}/{total}"
        if line != last_line:
            print(line, flush=True)
            last_line = line

        if state == "finished":
            return status
        if state in {"error", "cancelled"}:
            raise RuntimeError(f"job {job_id} {state}: {status.get('error') or ''}")
        if timeout is not None and time.monotonic() - started > timeout:
            raise TimeoutError(f"timed out waiting for job {job_id}")
        time.sleep(poll_interval)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run Lasagna new_point_strip.json for each collection in a VC3D point-collections JSON file."
    )
    ap.add_argument("point_collections_json", type=Path)
    ap.add_argument("--data-input", required=True,
                    help="Lasagna .lasagna.json/OME-Zarr input path passed as request data_input.")
    ap.add_argument("--init-shell-dir", type=Path,
                    help="Override the .lasagna.json init_shell_dir. Creates a per-collection shell_*.tifxyz subset view.")
    ap.add_argument("--max-init-shells", type=int, default=8,
                    help="Maximum init shells to expose per collection. Use 0 to expose all. Default: 8.")
    ap.add_argument("--work-dir", type=Path,
                    help="Persistent work directory for per-job manifests and shell views. Default: <output-dir>/.lasagna_point_strip_work")
    ap.add_argument("--progress-file", type=Path,
                    help="JSON file recording completed collections for resume. Default: <output-dir>/.lasagna_point_strip_progress.json")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where downloaded tifxyz results are extracted.")
    ap.add_argument("--width", type=float, required=True,
                    help="Requested model width in voxels.")
    ap.add_argument("--height", type=float, required=True,
                    help="Requested model height in voxels.")
    ap.add_argument("--windings", type=int, default=1)
    ap.add_argument("--volume-shape-zyx", type=int, nargs=3, metavar=("Z", "Y", "X"),
                    help="VC3D/input point coordinate volume shape. Enables Lasagna import/export coordinate scaling.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=None,
                    help="Per-job timeout in seconds. Default: wait indefinitely.")
    ap.add_argument("--missing-wind-a", choices=("zero", "skip", "error"), default="zero",
                    help="How to handle points whose wind_a is null/missing. Default: zero.")
    ap.add_argument("--source", default="point-strip-script")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned jobs without submitting them.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print every planned collection before submitting.")
    args = ap.parse_args(argv)

    base_config = load_json(args.config)
    collections = collections_from_file(args.point_collections_json, missing_wind_a=args.missing_wind_a)

    work_dir = (args.work_dir or (args.output_dir / ".lasagna_point_strip_work")).expanduser().resolve()
    progress_file = (args.progress_file or (args.output_dir / ".lasagna_point_strip_progress.json")).expanduser().resolve()
    if args.init_shell_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"using persistent work dir: {work_dir}")

    progress = load_progress(progress_file)
    progress["point_collections_json"] = str(args.point_collections_json.expanduser().resolve())
    progress["output_dir"] = str(args.output_dir.expanduser().resolve())
    progress["updated_at"] = utc_timestamp()
    completed = progress.setdefault("completed", {})
    progress.setdefault("failed", {})
    completed_ids = {str(collection_id) for collection_id in completed}

    print(f"loaded {len(collections)} collection(s) from {args.point_collections_json}")
    if completed_ids:
        resumable = sum(1 for info in collections if info["id"] in completed_ids)
        print(f"progress file: {progress_file} ({resumable} completed collection(s) will be skipped)")
    else:
        print(f"progress file: {progress_file}")
    preview = collections if args.verbose else collections[:20]
    for info in preview:
        print(
            f"collection {info['id']} ({info['name']}): "
            f"{info['point_count']} points, center point {info['center_point_id']}, "
            f"seed=({info['seed'][0]:.3f}, {info['seed'][1]:.3f}, {info['seed'][2]:.3f})"
        )
    if not args.verbose and len(collections) > len(preview):
        print(f"... {len(collections) - len(preview)} more collection(s); use --verbose to list all")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_progress(progress_file, progress)

    request_json(args.service_url, "/health", timeout=10)

    results = []
    skipped_results = []
    failed_results = []
    for info in collections:
        if info["id"] in completed_ids:
            previous = completed[info["id"]]
            skipped_results.append((info["id"], previous))
            print(f"skipping collection {info['id']}: already completed in {progress_file}")
            continue

        output_name = f"{uuid.uuid4()}.tifxyz"
        job_id = None
        try:
            body = build_request(
                base_config=base_config,
                data_input=prepare_data_input_manifest(
                    args.data_input,
                    args.init_shell_dir,
                    work_dir,
                    collection_info=info,
                    volume_shape_zyx=args.volume_shape_zyx,
                    max_shells=args.max_init_shells,
                ),
                width=args.width,
                height=args.height,
                windings=args.windings,
                collection_info=info,
                output_name=output_name,
                source=args.source,
                volume_shape_zyx=args.volume_shape_zyx,
            )
            print(f"submitting collection {info['id']} as {output_name}")
            response = request_json(args.service_url, "/optimize", body, timeout=30)
            job_id = response.get("job_id")
            if not job_id:
                raise RuntimeError(f"optimize response did not include job_id: {response}")

            wait_for_job(
                args.service_url,
                job_id,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
            archive = request_bytes(args.service_url, f"/jobs/{job_id}/results")
            placed = safe_extract_tar_gz(archive, args.output_dir)
        except Exception as exc:
            print(f"warning: collection {info['id']} failed; continuing with next collection: {exc}", file=sys.stderr)
            record_failed_collection(
                progress,
                collection_info=info,
                job_id=job_id,
                output_name=output_name,
                error=exc,
            )
            progress["updated_at"] = utc_timestamp()
            write_progress(progress_file, progress)
            failed_results.append((info["id"], job_id, output_name, str(exc)))
            continue

        record_completed_collection(
            progress,
            collection_info=info,
            job_id=job_id,
            output_name=output_name,
            placed=placed,
        )
        progress.setdefault("failed", {}).pop(info["id"], None)
        progress["updated_at"] = utc_timestamp()
        write_progress(progress_file, progress)
        completed_ids.add(info["id"])
        results.append((info["id"], job_id, output_name, placed))
        print(f"downloaded job {job_id}: {', '.join(placed) if placed else output_name}")
        print(f"recorded progress for collection {info['id']} in {progress_file}")

    if skipped_results:
        print("skipped completed:")
        for collection_id, previous in skipped_results:
            placed = previous.get("extracted") or []
            placed_text = ", ".join(placed) if placed else previous.get("requested_output_name", "")
            print(f"  collection {collection_id}: job {previous.get('job_id')}, extracted {placed_text}")

    print("completed this run:")
    for collection_id, job_id, output_name, placed in results:
        placed_text = ", ".join(placed) if placed else output_name
        print(f"  collection {collection_id}: job {job_id}, requested {output_name}, extracted {placed_text}")
    if failed_results:
        print("failed this run:")
        for collection_id, job_id, output_name, error in failed_results:
            job_text = job_id if job_id else "not submitted"
            print(f"  collection {collection_id}: job {job_text}, requested {output_name}, error {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
