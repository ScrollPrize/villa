import json

import numpy as np
import pytest

from fake_s3 import BUCKET, FakeS3Session
from remote_zarr import mirror_array


def write_array(path, separator, chunks):
    """A 16^3 uint8 Zarr v2 array of 8^3 chunks; only ``chunks`` are stored."""
    path.mkdir(parents=True)
    (path / '.zarray').write_text(json.dumps({
        'zarr_format': 2, 'shape': [16, 16, 16], 'chunks': [8, 8, 8], 'dtype': '|u1',
        'compressor': None, 'fill_value': 0, 'order': 'C', 'filters': None,
        'dimension_separator': separator}))
    for index in chunks:
        chunk = path / separator.join(map(str, index))
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_bytes(bytes([sum(index) + 1]) * 512)


@pytest.mark.parametrize('separator', ['.', '/'])
def test_mirror_copies_the_listed_chunks_and_resumes(tmp_path, separator):
    chunks = [(0, 0, 0), (1, 0, 1), (1, 1, 1)]
    write_array(tmp_path / 'bucket/a/s.zarr/4', separator, chunks)
    gets = []

    def session():
        return FakeS3Session(tmp_path / 'bucket', gets)
    reported = []
    mirror = tmp_path / 'mirror'
    assert mirror_array(f'{BUCKET}/a/s.zarr/4', mirror, workers=2, session_factory=session,
                        progress=lambda done, total: reported.append((done, total))) == str(mirror)
    for index in chunks:
        name = separator.join(map(str, index))
        assert (mirror / name).read_bytes() == (tmp_path / 'bucket/a/s.zarr/4' / name).read_bytes()
    assert json.loads((mirror / '.zarray').read_text())['dimension_separator'] == separator
    assert reported[0] == (0, 3) and reported[-1] == (3, 3)
    assert not list(mirror.rglob('*.partial'))

    # Complete: nothing is fetched again.
    gets.clear()
    mirror_array(f'{BUCKET}/a/s.zarr/4', mirror, session_factory=session)
    assert gets == []
    # Interrupted: only the chunks not yet written are fetched.
    (mirror / '.vc-mirrored').unlink()
    (mirror / separator.join(map(str, chunks[1]))).unlink()
    mirror_array(f'{BUCKET}/a/s.zarr/4', mirror, session_factory=session)
    assert sorted(key for key in gets if not key.endswith(('.zarray', '.zattrs'))) == [
        'a/s.zarr/4/' + separator.join(map(str, chunks[1]))]


def test_an_unlistable_array_is_refused(tmp_path):
    write_array(tmp_path / 'bucket/s.zarr', '.', [(0, 0, 0)])
    with pytest.raises(ValueError, match='cannot be listed'):
        mirror_array('https://data.example/s.zarr', tmp_path / 'mirror',
                      session_factory=lambda: FakeS3Session(tmp_path / 'bucket'))
