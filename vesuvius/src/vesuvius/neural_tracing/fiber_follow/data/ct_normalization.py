"""Per-crop z-score normalization and persisted volume provenance."""
import json
from pathlib import Path
import numpy as np

ZSCORE_METHOD = 'crop_zscore_v1'
ZSCORE_EPSILON = 1e-6


def volume_key(spec):
    source = str(spec.ct_zarr).rstrip('/')
    if '://' not in source:
        source = str(Path(source).resolve())
    return source+'::'+str(spec.ct_level)


def open_ct(spec):
    from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray, RemoteChunkedArray
    if str(spec.ct_zarr).startswith(('s3://', 'http://', 'https://')):
        return RemoteChunkedArray(spec.ct_zarr, spec.ct_level, spec.cache_dir, 32 << 20)
    return ChunkedArray(Path(spec.ct_zarr)/str(spec.ct_level), 32 << 20)


def array_identity(ct):
    return dict(shape=list(ct.shape), chunks=list(ct.chunks), dtype=ct.dtype.str)


def validate_record(record, spec):
    if record is None or record.get('method') != ZSCORE_METHOD:
        raise ValueError('Only per-crop z-score CT normalization is supported')
    if record.get('volume') != volume_key(spec) or record.get('epsilon') != ZSCORE_EPSILON:
        raise ValueError('Invalid z-score normalization record or different volume')
    return record


def prepare_normalization(out, specs, *, resume=None, known=None):
    """Persist calibration once, bind it to readers, and enforce exact resume reuse.

    Checkpoints embed the document too, allowing a missing JSON to be restored
    without estimating again. A changed JSON on resume is an error.
    """
    from vesuvius.neural_tracing.fiber_follow.data.volume import RemoteChunkedArray
    path = Path(out)/'ct_normalization.json'
    document = json.loads(path.read_text()) if path.exists() else None
    if resume is not None:
        if document is not None and document != resume:
            raise ValueError('CT normalization JSON differs from the resumed checkpoint')
        document = resume
    elif document is None:
        document = dict(method=ZSCORE_METHOD, volumes={})
    if document.get('method') != ZSCORE_METHOD:
        raise ValueError('Unsupported CT normalization policy')
    if known is not None and document['method'] != known['method']:
        raise ValueError('Inference CT normalization differs from the checkpoint')
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
                record = dict(method=ZSCORE_METHOD, volume=key, epsilon=ZSCORE_EPSILON, **array_identity(ct))
            document['volumes'][key] = record
        record = validate_record(document['volumes'][key], spec)
        if record['method'] != document['method']:
            raise ValueError('CT record and document normalization methods differ')
        if any(record[k] != v for k, v in array_identity(ct).items()):
            raise ValueError(f'CT array metadata changed since calibration: {key}')
        spec.ct_normalization = dict(record)
        print(f'CT per-crop z-score, epsilon {ZSCORE_EPSILON:g}: {key}', flush=True)
    if not path.exists() or json.loads(path.read_text()) != document:
        RemoteChunkedArray._atomic_write(path, (json.dumps(document, indent=2)+'\n').encode())
    return document


def normalize_ct(image, record):
    """Normalize a sampled crop in place, including all values without clipping."""
    if record is None or record.get('method') != ZSCORE_METHOD:
        raise ValueError('A z-score normalization record is required before sampling model inputs')
    if image.dtype != np.float32 or image.ndim != 3 or not image.flags.c_contiguous:
        raise ValueError('CT normalization requires a contiguous float32 3D crop')
    if not np.isfinite(image).all():
        raise ValueError('Z-score CT crop contains nonfinite values')
    mean = image.mean(dtype=np.float64)
    std = image.std(dtype=np.float64)
    image -= np.float32(mean)
    image /= np.float32(max(float(std), ZSCORE_EPSILON))
