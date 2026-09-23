"""Extract and load sparse, confidence-weighted fiber direction samples.

The extractor reads only Zarr chunks intersecting a requested z range, across
the full XY extent. Within each fixed prediction-space cell it retains the
highest-presence voxel, avoiding a full-volume download and most duplicate
samples across a fiber's thickness.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import unquote, urljoin, urlparse
import xml.etree.ElementTree as ET
import zipfile

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
    if response.status_code == 404:
        return _empty_chunk(metadata, index)
    response.raise_for_status()
    return _decode_chunk(response.content, metadata, index)


def _empty_chunk(metadata, index):
    shape = np.asarray(metadata["shape"], dtype=np.int64)
    chunks = np.asarray(metadata["chunks"], dtype=np.int64)
    begin = np.asarray(index, dtype=np.int64) * chunks
    chunk_shape = tuple(np.minimum(chunks, shape - begin))
    return np.full(chunk_shape, metadata.get("fill_value") or 0,
                   dtype=np.dtype(metadata["dtype"]))


def _decode_chunk(raw, metadata, index):
    import numcodecs

    shape = np.asarray(metadata["shape"], dtype=np.int64)
    chunks = np.asarray(metadata["chunks"], dtype=np.int64)
    begin = np.asarray(index, dtype=np.int64) * chunks
    chunk_shape = tuple(np.minimum(chunks, shape - begin))
    compressor = metadata.get("compressor")
    decoded = (numcodecs.get_codec(compressor).decode(raw)
               if compressor is not None else raw)
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


def extract(manifest_url, z_roi, output, *, output_scale=1.0,
            threshold=160, cell_size=2, overwrite=False, workers=1):
    import requests

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output} (use --overwrite)")
    with requests.Session() as session:
        manifest = _get_json(session, manifest_url)
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


def _local_arrays(directory, group):
    directory = Path(directory)
    hits = sorted(directory.glob('*_presence.ome.zarr'))
    if len(hits) != 1:
        raise ValueError(f'{directory}: expected exactly one *_presence.ome.zarr store')
    prefix = hits[0].name.removesuffix('_presence.ome.zarr')
    paths = {name: directory / f'{prefix}_{name}.ome.zarr' / str(group)
             for name in ('presence', 'nx', 'ny')}
    metadata = {}
    scales = []
    for name, path in paths.items():
        metadata[name] = json.loads((path / '.zarray').read_text())
        attrs = json.loads((path.parent / '.zattrs').read_text())
        datasets = attrs['multiscales'][0]['datasets']
        dataset = next(item for item in datasets if item['path'] == str(group))
        scale = next(transform['scale'] for transform in dataset['coordinateTransformations']
                     if transform['type'] == 'scale')
        if len(scale) != 3 or not np.allclose(scale, scale[0]):
            raise ValueError(f'{path}: expected an isotropic three-dimensional scale')
        scales.append(float(scale[0]))
    reference = metadata['presence']
    if (not np.allclose(scales, scales[0]) or scales[0] <= 0
            or any(metadata[name]['shape'] != reference['shape']
                   or metadata[name]['chunks'] != reference['chunks']
                   or np.dtype(metadata[name]['dtype']) != np.uint8
                   for name in paths)):
        raise ValueError('local presence/nx/ny arrays must share a uint8 grid and scale')
    return paths, metadata, scales[0]


def _local_chunk_indices(array_dir, metadata, first_chunk, last_chunk):
    separator = metadata.get('dimension_separator', '.')
    indices = []
    if separator == '/':
        for cz in range(int(first_chunk[0]), int(last_chunk[0]) + 1):
            z_dir = array_dir / str(cz)
            if not z_dir.is_dir():
                continue
            for y_dir in z_dir.iterdir():
                if not y_dir.is_dir() or not y_dir.name.isdecimal():
                    continue
                cy = int(y_dir.name)
                if not first_chunk[1] <= cy <= last_chunk[1]:
                    continue
                for file in y_dir.iterdir():
                    if file.is_file() and file.name.isdecimal():
                        cx = int(file.name)
                        if first_chunk[2] <= cx <= last_chunk[2]:
                            indices.append((cz, cy, cx))
    elif separator == '.':
        for file in array_dir.iterdir():
            if not file.is_file():
                continue
            try:
                index = tuple(int(v) for v in file.name.split('.'))
            except ValueError:
                continue
            if len(index) == 3 and all(first_chunk[a] <= index[a] <= last_chunk[a]
                                       for a in range(3)):
                indices.append(index)
    else:
        raise ValueError(f'unsupported Zarr dimension separator {separator!r}')
    return sorted(indices)


def _read_local_chunk(array_dir, metadata, index):
    separator = metadata.get('dimension_separator', '.')
    path = array_dir / separator.join(map(str, index))
    return (_decode_chunk(path.read_bytes(), metadata, index) if path.is_file()
            else _empty_chunk(metadata, index))


def _write_streamed_npz(output, directory, count, metadata):
    temporary = directory / 'fiber_directions.npz'
    schema = {'position_zyx': (np.float32, (count, 3)),
              'nx': (np.uint8, (count,)), 'ny': (np.uint8, (count,)),
              'presence': (np.uint8, (count,))}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as archive:
        for name, (dtype, shape) in schema.items():
            with archive.open(f'{name}.npy', 'w', force_zip64=True) as target:
                np.lib.format.write_array_header_2_0(target, {
                    'descr': np.lib.format.dtype_to_descr(np.dtype(dtype)),
                    'fortran_order': False, 'shape': shape})
                with (directory / f'{name}.bin').open('rb') as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        with archive.open('metadata_json.npy', 'w', force_zip64=True) as target:
            np.lib.format.write_array(target, np.asarray(json.dumps(metadata)),
                                      allow_pickle=False)
    os.replace(temporary, output)


def extract_local(directory, z_roi, output, *, group='3', output_scale=4.0,
                  threshold=160, cell_size=2, overwrite=False, workers=4):
    """Extract a sparse fitted-coordinate NPZ from local Lasagna Zarr stores."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f'output already exists: {output} (use --overwrite)')
    paths, metadata, base_scale = _local_arrays(directory, group)
    chunks = np.asarray(metadata['presence']['chunks'], dtype=np.int64)
    if np.any(chunks % cell_size):
        raise ValueError(f'cell size {cell_size} must divide source chunks {tuple(chunks)}')
    prediction_to_output_scale = base_scale / output_scale
    shape = np.asarray(metadata['presence']['shape'], dtype=np.int64)
    begin = np.asarray((np.floor(z_roi[0] / prediction_to_output_scale), 0, 0),
                       dtype=np.int64)
    end = np.asarray((np.ceil(z_roi[1] / prediction_to_output_scale), shape[1], shape[2]),
                     dtype=np.int64)
    begin = np.maximum(begin, 0)
    end = np.minimum(end, shape)
    if end[0] <= begin[0]:
        raise ValueError('requested z ROI does not intersect the fiber prediction volume')
    work = _local_chunk_indices(paths['presence'], metadata['presence'],
                                begin // chunks, (end - 1) // chunks)
    if not work:
        raise ValueError('no stored fiber presence chunks intersect the requested z ROI')
    print(f'found {len(work):,} local presence chunks at pyramid group {group}', flush=True)

    def process(index):
        presence = _read_local_chunk(paths['presence'], metadata['presence'], index)
        chunk_begin = np.asarray(index) * chunks
        local_begin = max(0, int(begin[0] - chunk_begin[0]))
        local_end = min(len(presence), int(end[0] - chunk_begin[0]))
        partial = presence[local_begin:local_end]
        selected, prediction_zyx = _cell_argmax(
            partial, chunk_begin + [local_begin, 0, 0], cell_size, threshold,
            begin, end)
        position = (prediction_zyx.astype(np.float32)
                    * np.float32(prediction_to_output_scale))
        if len(selected):
            selected = selected + local_begin * presence.shape[1] * presence.shape[2]
            nx = _read_local_chunk(paths['nx'], metadata['nx'], index).reshape(-1)[selected]
            ny = _read_local_chunk(paths['ny'], metadata['ny'], index).reshape(-1)[selected]
            confidence = presence.reshape(-1)[selected]
        else:
            nx = ny = confidence = np.empty(0, dtype=np.uint8)
        return position, nx, ny, confidence

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as temp:
        temporary = Path(temp)
        files = {name: (temporary / f'{name}.bin').open('wb')
                 for name in ('position_zyx', 'nx', 'ny', 'presence')}
        count = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for completed, arrays in enumerate(
                        executor.map(process, work, buffersize=workers * 2), 1):
                    count += len(arrays[0])
                    for stream, values in zip(files.values(), arrays):
                        stream.write(values.tobytes())
                    if completed % 100 == 0 or completed == len(work):
                        print(f'processed {completed:,}/{len(work):,} chunks; '
                              f'{count:,} samples; {time.monotonic()-started:.0f}s',
                              flush=True)
        finally:
            for stream in files.values():
                stream.close()
        if count == 0:
            raise ValueError('no fiber direction samples met the presence threshold in this z ROI')
        metadata_out = {
            'artifact_type': 'fiber_direction_samples', 'format_version': FORMAT_VERSION,
            'source_directory': str(Path(directory).resolve()), 'source_group': str(group),
            'z_roi': list(z_roi), 'output_scale_base_voxels': float(output_scale),
            'prediction_to_output_scale': prediction_to_output_scale,
            'presence_threshold': int(threshold),
            'dedup_cell_size_prediction_voxels': int(cell_size), 'sample_count': count,
        }
        _write_streamed_npz(output, temporary, count, metadata_out)
    print(f'wrote {count:,} fiber direction samples to {output}', flush=True)


def load_fiber_direction_samples(path, z_begin, z_end):
    """Load one packed artifact, filtering in the fitter's coordinate frame."""
    if not path:
        return None
    artifact = Path(path)
    with np.load(artifact) as packed:
        metadata = json.loads(str(packed["metadata_json"]))
        if (metadata.get("artifact_type") != "fiber_direction_samples"
                or metadata.get("format_version") != FORMAT_VERSION):
            raise ValueError(f"unsupported fiber-direction artifact in {artifact}")
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
    parser.add_argument("manifest", help="HTTP(S) fiber manifest or local fiber_zarrs directory")
    parser.add_argument("output", type=Path, help="output packed .npz artifact")
    parser.add_argument("--z-roi", required=True, type=_parse_z_roi,
                        help="output-coordinate half-open z range BEGIN,END")
    parser.add_argument(
        "--output-scale", type=float, default=1.0,
        help="base /0 voxels per output/fitter voxel (use 4 for pyramid /2)")
    parser.add_argument("--presence-threshold", type=int, default=160)
    parser.add_argument("--cell-size", type=int, default=2,
                        help="deduplication cell edge in prediction voxels")
    parser.add_argument("--group", default="3",
                        help="local Zarr pyramid group (default: 3)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel chunk readers (default: 1)")
    args = parser.parse_args()
    if not 0 <= args.presence_threshold <= 255:
        parser.error("--presence-threshold must be in [0, 255]")
    if args.cell_size <= 0:
        parser.error("--cell-size must be positive")
    if args.output_scale <= 0:
        parser.error("--output-scale must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if Path(args.manifest).is_dir():
        extract_local(args.manifest, args.z_roi, args.output, group=args.group,
                      output_scale=args.output_scale, threshold=args.presence_threshold,
                      cell_size=args.cell_size, overwrite=args.overwrite,
                      workers=args.workers)
    elif urlparse(args.manifest).scheme in ('http', 'https'):
        extract(args.manifest, args.z_roi, args.output,
                output_scale=args.output_scale,
                threshold=args.presence_threshold, cell_size=args.cell_size,
                overwrite=args.overwrite, workers=args.workers)
    else:
        parser.error(f'fiber source directory does not exist: {args.manifest}')


if __name__ == "__main__":
    main()
