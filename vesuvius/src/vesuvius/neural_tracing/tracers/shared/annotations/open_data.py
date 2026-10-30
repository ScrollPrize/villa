"""Identification of Vesuvius open-data CT stores, and whole-file hashing for provenance."""
import hashlib
import re
from urllib.parse import urlsplit


OPEN_DATA_BUCKET = 'vesuvius-challenge-open-data'
OPEN_DATA_HTTPS = f'https://{OPEN_DATA_BUCKET}.s3.us-east-1.amazonaws.com'


def open_data_volume(url):
    """(sample, volume id, base voxel um, https path) of a Vesuvius open-data CT store
    (``s3://vesuvius-challenge-open-data/SAMPLE/volumes/ID-2.400um-....zarr`` or its https form), else None.
    The volume id is what object-store fibers name as their ``coordinates.volume_uuid``."""
    source = urlsplit(str(url).rstrip('/'))
    if (source.scheme, source.netloc) not in (('s3', OPEN_DATA_BUCKET), ('https', urlsplit(OPEN_DATA_HTTPS).netloc)):
        return None
    parts = source.path.strip('/').split('/')
    match = re.match(r'(\d+)-(\d+(?:\.\d+)?)um', parts[-1])
    if len(parts) != 3 or parts[1] != 'volumes' or not match:
        return None
    return parts[0], match[1], float(match[2]), f'{OPEN_DATA_HTTPS}/{"/".join(parts)}'


def file_sha256(path):
    """SHA-256 of a whole file, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
