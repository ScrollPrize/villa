"""Extract and load sparse, confidence-weighted fiber direction samples.

The extractor reads only Zarr chunks intersecting a requested z range, across
the full XY extent. Within each fixed prediction-space cell it retains the
highest-presence voxel, avoiding a full-volume download and most duplicate
samples across a fiber's thickness.

With ``--cache`` the artifact goes where a fit finds it by the fiber-inference
dataset's identity (lasagna_source.fiber_directions_path), extracted over the
prediction's whole height unless ``--z-roi`` narrows it; ``--adopt`` places an
existing artifact there instead of extracting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import unquote, urljoin, urlparse
import xml.etree.ElementTree as ET

import numpy as np


FORMAT_VERSION = 2


def _get_json(session, url):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _chunk_url(array_url, index, metadata):
    separator = metadata.get("dimension_separator", ".")
    return f"{array_url.rstrip('/')}/{separator.join(map(str, index))}"


def _read_chunk(session, array_url, metadata, index):
    import numcodecs
    import requests

    url = _chunk_url(array_url, index, metadata)
    for attempt in range(6):
        try:
            response = session.get(url, timeout=60)
            if response.status_code < 500:
                break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 5:
                raise
        if attempt == 5:
            response.raise_for_status()
        time.sleep(min(2 ** attempt, 8))
    shape = np.asarray(metadata["shape"], dtype=np.int64)
    chunks = np.asarray(metadata["chunks"], dtype=np.int64)
    begin = np.asarray(index, dtype=np.int64) * chunks
    chunk_shape = tuple(np.minimum(chunks, shape - begin))
    if response.status_code == 404:
        return np.full(chunk_shape, metadata.get("fill_value") or 0,
                       dtype=np.dtype(metadata["dtype"]))
    response.raise_for_status()
    compressor = metadata.get("compressor")
    decoded = (numcodecs.get_codec(compressor).decode(response.content)
               if compressor is not None else response.content)
    dtype = np.dtype(metadata["dtype"])
    values = np.frombuffer(decoded, dtype=dtype)
    edge_size = int(np.prod(chunk_shape))
    full_size = int(np.prod(chunks))
    order = metadata.get("order", "C")
    if values.size == edge_size:
        return values.reshape(chunk_shape, order=order)
    if values.size == full_size:
        # Some Zarr v2 writers store boundary chunks at the nominal full
        # chunk shape. Crop that representation to the logical array edge.
        full = values.reshape(tuple(chunks), order=order)
        return full[tuple(slice(0, size) for size in chunk_shape)]
    raise ValueError(
        f"decoded chunk {index} has {values.size} values; expected "
        f"{edge_size} (logical edge) or {full_size} (full stored chunk)")


def _list_s3_chunk_indices(session, array_url, metadata, first_chunk, last_chunk):
    """Return existing chunk indices for an anonymous virtual-hosted S3 URL."""
    parsed = urlparse(array_url)
    if not parsed.hostname or ".s3." not in parsed.hostname:
        return None
    separator = metadata.get("dimension_separator", ".")
    array_key = unquote(parsed.path.lstrip("/").rstrip("/"))
    result = set()
    for cz in range(int(first_chunk[0]), int(last_chunk[0]) + 1):
        prefix = f"{array_key}/{cz}{separator}"
        continuation = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if continuation:
                params["continuation-token"] = continuation
            response = session.get(
                f"{parsed.scheme}://{parsed.netloc}/", params=params, timeout=60)
            if response.status_code in (401, 403):
                return None
            response.raise_for_status()
            root = ET.fromstring(response.content)
            keys = [node.text for node in root.iter()
                    if node.tag.rsplit("}", 1)[-1] == "Key"]
            for key in keys:
                suffix = key[len(array_key) + 1:]
                try:
                    index = tuple(int(value) for value in suffix.split(separator))
                except (TypeError, ValueError):
                    continue
                if (len(index) == 3
                        and all(first_chunk[axis] <= index[axis] <= last_chunk[axis]
                                for axis in range(3))):
                    result.add(index)
            truncated = next((node.text == "true" for node in root.iter()
                              if node.tag.rsplit("}", 1)[-1] == "IsTruncated"), False)
            if not truncated:
                break
            continuation = next((node.text for node in root.iter()
                                 if node.tag.rsplit("}", 1)[-1]
                                 == "NextContinuationToken"), None)
            if not continuation:
                raise ValueError("truncated S3 listing omitted continuation token")
    return result


def _parse_z_roi(value):
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "z ROI must be BEGIN,END integers") from error
    if len(result) != 2 or result[0] < 0 or result[1] <= result[0]:
        raise argparse.ArgumentTypeError(
            "z ROI must be BEGIN,END with 0 <= BEGIN < END")
    return result


def _cell_argmax(presence, global_begin, cell_size, threshold, valid_begin, valid_end):
    local = np.indices(presence.shape, dtype=np.int64).reshape(3, -1).T
    global_zyx = local + global_begin
    values = presence.reshape(-1)
    valid = ((values >= threshold)
             & (global_zyx >= valid_begin).all(axis=1)
             & (global_zyx < valid_end).all(axis=1))
    if not valid.any():
        return np.empty(0, dtype=np.int64), global_zyx[:0]
    flat_indices = np.flatnonzero(valid)
    coords = global_zyx[valid]
    cells = coords // cell_size
    # Presence descending, then flat index ascending gives deterministic ties.
    order = np.lexsort((flat_indices, -values[valid].astype(np.int64),
                        cells[:, 2], cells[:, 1], cells[:, 0]))
    ordered_cells = cells[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1)
    selected = order[first]
    return flat_indices[selected], coords[selected]


def extract(manifest_url, z_roi, output=None, *, cache=None, output_scale=1.0,
            threshold=160, cell_size=2, overwrite=False, workers=1):
    """Extract into ``output``, or with ``cache`` into the dataset's cached
    artifact; a ``z_roi`` of None is the prediction's whole height."""
    import requests
    from lasagna_source import fiber_directions_path, url_identity

    identity = url_identity(manifest_url)
    with requests.Session() as session:
        manifest = _get_json(session, manifest_url)
        output = Path(fiber_directions_path(cache, identity, output_scale)
                      if cache is not None else output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output} (use --overwrite)")
        groups = manifest["groups"]
        urls = {name: urljoin(manifest_url, groups[name]["zarr"])
                for name in ("presence", "nx", "ny")}
        metadata = {name: _get_json(session, f"{url.rstrip('/')}/.zarray")
                    for name, url in urls.items()}
        reference = metadata["presence"]
        for name in ("nx", "ny"):
            if (metadata[name]["shape"] != reference["shape"]
                    or metadata[name]["chunks"] != reference["chunks"]):
                raise ValueError("presence/nx/ny Zarr shapes or chunks differ")
        chunks = np.asarray(reference["chunks"], dtype=np.int64)
        if np.any(chunks % cell_size):
            raise ValueError(
                f"cell size {cell_size} must divide source chunk shape {tuple(chunks)}")

        scales = []
        for name in ("presence", "nx", "ny"):
            scales.append(float(manifest["source_to_base"])
                          * 2.0 ** int(groups[name]["scaledown"]))
        if not np.allclose(scales, scales[0]):
            raise ValueError(f"presence/nx/ny scales differ: {scales}")
        prediction_to_base_scale = scales[0]
        prediction_to_output_scale = prediction_to_base_scale / output_scale
        shape = np.asarray(reference["shape"], dtype=np.int64)
        if z_roi is None:
            z_roi = (0, int(np.ceil(shape[0] * prediction_to_output_scale)))
        z_begin, z_end = z_roi
        begin = np.asarray((
            np.floor(z_begin / prediction_to_output_scale), 0, 0),
            dtype=np.int64)
        end = np.asarray((
            np.ceil(z_end / prediction_to_output_scale), shape[1], shape[2]),
            dtype=np.int64)
        begin = np.maximum(begin, 0)
        end = np.minimum(end, shape)
        if end[0] <= begin[0]:
            raise ValueError(
                f"z ROI {z_roi} does not intersect prediction z extent "
                f"[0, {shape[0] * prediction_to_output_scale:g})")
        extraction_identity = {
            "artifact_type": "fiber_direction_samples",
            "format_version": FORMAT_VERSION,
            "manifest_url": manifest_url,
            "dataset_identity": identity,
            "z_roi": list(z_roi),
            "output_scale_base_voxels": float(output_scale),
            "prediction_to_output_scale": prediction_to_output_scale,
            "presence_threshold": int(threshold),
            "dedup_cell_size_prediction_voxels": int(cell_size),
        }
        first_chunk = begin // chunks
        last_chunk = (end - 1) // chunks
        available = _list_s3_chunk_indices(
            session, urls["presence"], reference, first_chunk, last_chunk)
        if available is not None:
            print(f"S3 listing found {len(available):,} stored presence chunks "
                  f"in the requested z ROI", flush=True)
        work = []
        for cz in range(first_chunk[0], last_chunk[0] + 1):
            for cy in range(first_chunk[1], last_chunk[1] + 1):
                for cx in range(first_chunk[2], last_chunk[2] + 1):
                    index = (int(cz), int(cy), int(cx))
                    if available is not None and index not in available:
                        continue
                    work.append(index)

        def process(index):
            # requests.Session is not documented as thread-safe. Each worker
            # invocation therefore owns its short-lived connection context.
            with requests.Session() as worker_session:
                presence = _read_chunk(
                    worker_session, urls["presence"], metadata["presence"], index)
                chunk_begin = np.asarray(index, dtype=np.int64) * chunks
                selected, prediction_zyx = _cell_argmax(
                    presence, chunk_begin, cell_size, threshold, begin, end)
                position_zyx = (prediction_zyx.astype(np.float32)
                                * np.float32(prediction_to_output_scale))
                if len(selected):
                    nx = _read_chunk(worker_session, urls["nx"], metadata["nx"], index)
                    ny = _read_chunk(worker_session, urls["ny"], metadata["ny"], index)
                    nx_selected = nx.reshape(-1)[selected].astype(np.uint8)
                    ny_selected = ny.reshape(-1)[selected].astype(np.uint8)
                else:
                    nx_selected = ny_selected = np.empty(0, dtype=np.uint8)
                return (position_zyx, nx_selected, ny_selected,
                        presence.reshape(-1)[selected].astype(np.uint8))

        positions, nxs, nys, presences = [], [], [], []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for completed, arrays in enumerate(executor.map(process, work), 1):
                position, nx, ny, presence = arrays
                positions.append(position)
                nxs.append(nx)
                nys.append(ny)
                presences.append(presence)
                if completed % 100 == 0 or completed == len(work):
                    print(f"downloaded {completed:,}/{len(work):,} chunks", flush=True)

    metadata_out = {
        **extraction_identity,
        "sample_count": int(sum(map(len, positions))),
    }
    arrays = {
        "position_zyx": np.concatenate(positions),
        "nx": np.concatenate(nxs),
        "ny": np.concatenate(nys),
        "presence": np.concatenate(presences),
        "metadata_json": np.asarray(json.dumps(metadata_out)),
    }
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    print(f"wrote {metadata_out['sample_count']:,} samples from "
          f"{len(work):,} chunks to {output}")


def adopt(artifact, cache, *, overwrite=False):
    """Place an existing artifact where a fit finds it by its fiber-inference
    dataset's identity (its manifest_url's) and scale: a hard link when the
    cache is on the same filesystem, else a copy. Returns the cached path."""
    import os
    import shutil

    from lasagna_source import fiber_directions_path, url_identity

    artifact = Path(artifact)
    metadata = _artifact_metadata(artifact)
    identity = url_identity(metadata["manifest_url"])
    target = Path(fiber_directions_path(cache, identity,
                                        metadata["output_scale_base_voxels"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists (use --overwrite)")
        target.unlink()
    try:
        os.link(artifact, target)
    except OSError:
        shutil.copy2(artifact, target)
    print(f"adopted {artifact} as {target}")
    return target


def _artifact_metadata(artifact):
    with np.load(artifact) as packed:
        metadata = json.loads(str(packed["metadata_json"]))
    if (metadata.get("artifact_type") != "fiber_direction_samples"
            or metadata.get("format_version") != FORMAT_VERSION):
        raise ValueError(f"unsupported fiber-direction artifact in {artifact}")
    return metadata


def load_fiber_direction_samples(path, z_begin, z_end):
    """Load one packed artifact, filtering in the fitter's coordinate frame.
    The fit's window must lie inside the z range it was extracted over."""
    if not path:
        return None
    artifact = Path(path)
    with np.load(artifact) as packed:
        metadata = json.loads(str(packed["metadata_json"]))
        if (metadata.get("artifact_type") != "fiber_direction_samples"
                or metadata.get("format_version") != FORMAT_VERSION):
            raise ValueError(f"unsupported fiber-direction artifact in {artifact}")
        extracted = metadata.get("z_roi")
        if extracted is not None and not (extracted[0] <= z_begin and z_end <= extracted[1]):
            raise ValueError(
                f"{artifact} holds fiber directions for z [{extracted[0]}, "
                f"{extracted[1]}), not the fit's [{z_begin}, {z_end})")
        position = packed["position_zyx"]
        keep = (position[:, 0] >= z_begin) & (position[:, 0] < z_end)
        if not keep.any():
            return None
        whole = bool(keep.all())
        select = slice(None) if whole else keep
        result = {
            "position_zyx": position[select].astype(np.float32, copy=False),
            "nx": packed["nx"][select].astype(np.uint8, copy=False),
            "ny": packed["ny"][select].astype(np.uint8, copy=False),
            "presence": packed["presence"][select].astype(np.uint8, copy=False),
            "metadata": metadata,
        }
    if not len(result["position_zyx"]):
        return None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?",
                        help="HTTP(S) Lasagna fiber manifest")
    parser.add_argument("output", type=Path, nargs="?",
                        help="output packed .npz artifact (or --cache)")
    parser.add_argument("--cache", type=Path, default=None, metavar="DIR",
                        help="write the artifact into a fit cache (fit_spiral.py "
                             "and spiral_service.py --cache), keyed by the "
                             "dataset's identity and --output-scale")
    parser.add_argument("--adopt", type=Path, default=None, metavar="NPZ",
                        help="place an existing artifact into --cache instead "
                             "of extracting")
    parser.add_argument("--z-roi", type=_parse_z_roi, default=None,
                        help="output-coordinate half-open z range BEGIN,END "
                             "(default: the prediction's whole height)")
    parser.add_argument(
        "--output-scale", type=float, default=1.0,
        help="base /0 voxels per output/fitter voxel (use 4 for pyramid /2)")
    parser.add_argument("--presence-threshold", type=int, default=160)
    parser.add_argument("--cell-size", type=int, default=2,
                        help="deduplication cell edge in prediction voxels")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel chunk downloads (default: 1)")
    args = parser.parse_args()
    if args.adopt is not None:
        if args.cache is None or args.manifest or args.output:
            parser.error("--adopt takes only --cache (and --overwrite)")
        adopt(args.adopt, args.cache, overwrite=args.overwrite)
        return
    if not args.manifest or (args.output is None) == (args.cache is None):
        parser.error("give the manifest and either an output path or --cache")
    if not 0 <= args.presence_threshold <= 255:
        parser.error("--presence-threshold must be in [0, 255]")
    if args.cell_size <= 0:
        parser.error("--cell-size must be positive")
    if args.output_scale <= 0:
        parser.error("--output-scale must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    extract(args.manifest, args.z_roi, args.output, cache=args.cache,
            output_scale=args.output_scale,
            threshold=args.presence_threshold, cell_size=args.cell_size,
            overwrite=args.overwrite, workers=args.workers)


if __name__ == "__main__":
    main()
