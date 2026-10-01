"""Per-volume quiet-background calibration and fast foreground-only CT normalization."""
import json
import os
from pathlib import Path

import numba
import numpy as np
from scipy.ndimage import gaussian_filter1d

METHOD = 'background_3noise_foreground_mad_v1'
BACKGROUND = -4.
LIMIT = 4.
CALIBRATION_SEED = 532


def chunk_paths(root, separator):
    if separator == '.':
        return sorted(Path(e.path) for e in os.scandir(root)
                      if e.is_file() and len(e.name.split('.')) == 3 and all(v.isdigit() for v in e.name.split('.')))
    paths = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.isdigit()]
        paths.extend(Path(directory)/f for f in files if f.isdigit())
    return sorted(paths)


def background_from_blocks(rows):
    """Dominant median intensity among the quietest quarter of 8-cubed blocks.

    Restricting local MAD measures texture, not global bright-tail percentiles.
    No assumption that zero-fill or air dominates the volume histogram.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) < 100 or not np.isfinite(rows).all():
        raise ValueError('CT calibration requires at least 100 finite populated background candidate blocks')
    cutoff = float(np.quantile(rows[:, 1], .25))
    quiet = rows[rows[:, 1] <= cutoff]
    hist = np.bincount(quiet[:, 0].astype(int), minlength=256)
    mode = int(np.argmax(gaussian_filter1d(hist.astype(float), 2)))
    near = quiet[np.abs(quiet[:, 0]-mode) <= 5]
    if not len(near):
        raise ValueError('No populated CT background mode found')
    center, noise = np.median(near, axis=0)
    noise = max(2., float(noise))
    return dict(center=float(center), noise=noise, threshold=float(center+3*noise),
                modal_bin=mode, sampled_blocks=len(rows), quiet_blocks=len(quiet),
                modal_blocks=len(near), quiet_mad_cutoff=cutoff)


def volume_key(spec):
    source = str(spec.ct_zarr).rstrip('/')
    if '://' not in source:
        source = str(Path(source).resolve())
    return source+'::'+str(spec.ct_level)


def open_ct(spec):
    from .volume import ChunkedArray, RemoteChunkedArray
    if str(spec.ct_zarr).startswith(('s3://', 'http://', 'https://')):
        return RemoteChunkedArray(spec.ct_zarr, spec.ct_level, spec.cache_dir, 32 << 20)
    return ChunkedArray(Path(spec.ct_zarr)/str(spec.ct_level), 32 << 20)


def array_identity(ct):
    return dict(shape=list(ct.shape), chunks=list(ct.chunks), dtype=ct.dtype.str)


def calibrate(ct, *, count=128, seed=CALIBRATION_SEED):
    """Sample available chunks, or bounded remote reads when the cache is empty.

    Reads through the normal reader, supporting compressed local stores too.
    Cache coverage is not a uniform whole-volume sample; record exact provenance.
    """
    if ct.dtype != np.dtype('uint8') or min(ct.chunks) < 8:
        raise ValueError('CT calibration requires uint8 data with chunks at least 8 voxels wide')
    rng = np.random.default_rng(seed)
    paths = chunk_paths(ct.path, ct.sep)
    rows, used = [], []
    candidates = []
    if paths:
        for i in rng.choice(len(paths), min(4*count, len(paths)), replace=False):
            relative = str(paths[i].relative_to(ct.path))
            candidates.append(tuple(int(v) for v in relative.split(ct.sep)))
    else:
        shape = (np.asarray(ct.shape)+ct.chunks-1)//ct.chunks
        candidates = list(dict.fromkeys(tuple(int(v) for v in rng.integers(0, shape)) for _ in range(4*count)))
    for key in candidates:
        if len(used) >= count and len(rows) >= 100:
            break
        raw = ct.chunk(key)
        if raw is None:
            continue
        extent = np.minimum(ct.chunks, np.asarray(ct.shape)-np.asarray(key)*ct.chunks)
        if min(extent) < 8:
            continue
        used.append(list(key))
        for z, y, x in rng.integers(0, extent-7, (64, 3)):
            block = np.array(raw[z:z+8, y:y+8, x:x+8], dtype=np.float32).ravel()
            if np.count_nonzero(block) < .98*block.size:
                continue
            center = np.median(block)
            rows.append((center, 1.4826*np.median(np.abs(block-center))))
    return dict(background_from_blocks(rows), seed=seed, sampled_chunks=used,
                sampling='available_chunks' if paths else 'random_volume_chunks', **array_identity(ct))


def validate_record(record, spec):
    if record is None or record.get('method') != METHOD or record.get('volume') != volume_key(spec):
        raise ValueError('CT background calibration missing or belongs to a different volume')
    center, noise, threshold = (record[k] for k in ('center', 'noise', 'threshold'))
    if not np.isfinite([center, noise, threshold]).all() or not 0 <= center <= 255 or noise < 2:
        raise ValueError('Invalid CT background calibration')
    if not np.isclose(threshold, center+3*noise, rtol=0, atol=1e-9):
        raise ValueError('CT foreground threshold must equal background + 3 noise scales')
    return record


def prepare_normalization(out, specs, *, resume=None, known=None):
    """Persist calibration once, bind it to readers, and enforce exact resume reuse.

    Checkpoints embed the document too, allowing a missing JSON to be restored
    without estimating again. A changed JSON on resume is an error.
    """
    from .volume import RemoteChunkedArray
    path = Path(out)/'ct_normalization.json'
    document = json.loads(path.read_text()) if path.exists() else None
    if resume is not None:
        if document is not None and document != resume:
            raise ValueError('CT normalization JSON differs from the resumed checkpoint')
        document = resume
    elif document is None:
        document = dict(method=METHOD, volumes={})
    if document.get('method') != METHOD:
        raise ValueError('Unsupported CT normalization policy')
    # Do not mutate an embedded checkpoint document while adding an inference volume.
    document = json.loads(json.dumps(document))
    for spec in specs:
        key = volume_key(spec)
        ct = open_ct(spec)
        if (known is not None and key in known['volumes'] and key in document['volumes']
                and known['volumes'][key] != document['volumes'][key]):
            raise ValueError(f'Inference CT normalization differs from the checkpoint: {key}')
        if key not in document['volumes']:
            if resume is not None:
                raise ValueError(f'Resumed checkpoint lacks CT calibration: {key}')
            if known is not None and key in known['volumes']:
                record = known['volumes'][key]
            else:
                print(f'Calibrating CT background: {key}', flush=True)
                record = dict(calibrate(ct), method=METHOD, volume=key)
            document['volumes'][key] = record
        record = validate_record(document['volumes'][key], spec)
        if any(record[k] != v for k, v in array_identity(ct).items()):
            raise ValueError(f'CT array metadata changed since calibration: {key}')
        spec.ct_normalization = dict(record)
        print(f'CT background {record["center"]:.2f}, noise {record["noise"]:.2f}, '
              f'foreground > {record["threshold"]:.2f}: {key}', flush=True)
    if not path.exists() or json.loads(path.read_text()) != document:
        RemoteChunkedArray._atomic_write(path, (json.dumps(document, indent=2)+'\n').encode())
    return document


@numba.njit(cache=True)
def _foreground_histogram(image, threshold, step):
    hist = np.zeros(256, np.int64)
    for z in range(0, image.shape[0], step):
        for y in range(0, image.shape[1], step):
            for x in range(0, image.shape[2], step):
                value = image[z, y, x]
                if np.isfinite(value) and value > threshold:
                    hist[max(0, min(255, int(value*255.+.5)))] += 1
    return hist


@numba.njit(cache=True)
def _median(hist):
    half, running = hist.sum()*.5, 0
    for i in range(256):
        running += hist[i]
        if running >= half:
            return i
    return 0


@numba.njit(cache=True)
def _normalize(image, threshold, noise):
    hist = _foreground_histogram(image, threshold, 4)
    if hist.sum() == 0:
        hist = _foreground_histogram(image, threshold, 1)  # Sparse thin foreground.
    center = _median(hist)
    deviations = np.zeros(256, np.int64)
    for i in range(256):
        deviations[abs(i-center)] += hist[i]
    scale = max(1.4826*_median(deviations), 2*noise)
    flat = image.reshape(-1)
    inverse = 255./scale
    offset = center/scale
    for i in range(flat.size):
        value = flat[i]
        flat[i] = (min(LIMIT, max(BACKGROUND, value*inverse-offset))
                   if np.isfinite(value) and value > threshold else BACKGROUND)


def normalize_ct(image, record):
    """In-place float32 crop, sampled in [0,1], to masked robust z-scores.

    Threshold continuous interpolated values before quantizing statistics only.
    The background sentinel is -4; retained foreground is clipped to [-4,4].
    """
    if record is None:
        raise ValueError('CT background calibration is required before sampling model inputs')
    if image.dtype != np.float32 or image.ndim != 3 or not image.flags.c_contiguous:
        raise ValueError('CT normalization requires a contiguous float32 3D crop')
    _normalize(image, np.float32(record['threshold']/255.), record['noise'])
