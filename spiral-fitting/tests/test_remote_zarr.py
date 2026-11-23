import json
from pathlib import Path

import pytest

pytest.importorskip('vc_reader')

from fake_s3 import FakeS3Server  # noqa: E402
from remote_zarr import is_listable, listable_location, mirror_array  # noqa: E402


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
    source = tmp_path / 'bucket/a/s.zarr/4'
    write_array(source, separator, chunks)
    cache = tmp_path / 'cache'
    reported = []
    with FakeS3Server(tmp_path / 'bucket') as s3:
        url, access = s3.location('a/s.zarr/4'), s3.access()
        mirror = Path(mirror_array(url, cache, workers=2, access=access,
                                   progress=lambda done, total: reported.append((done, total))))
        # vc_reader's layout under the cache: the array at its s3:// place.
        assert mirror == cache / 's3/bucket/a/s.zarr/4'
        for index in chunks:
            name = separator.join(map(str, index))
            assert (mirror / name).read_bytes() == (source / name).read_bytes()
        assert json.loads((mirror / '.zarray').read_text())['dimension_separator'] == separator
        assert reported and reported[-1] == (4, 4)  # .zarray and three chunks
        assert (mirror / '.vc-mirrored').read_text().strip() == url

        # Complete: nothing is fetched again.
        s3.gets.clear()
        assert mirror_array(url, cache, access=access) == str(mirror)
        assert s3.gets == []
        # Interrupted: only the objects not yet written are fetched.
        (mirror / '.vc-mirrored').unlink()
        missing = separator.join(map(str, chunks[1]))
        (mirror / missing).unlink()
        mirror_array(url, cache, access=access)
        assert s3.gets == [f'a/s.zarr/4/{missing}']
        assert (mirror / missing).read_bytes() == (source / missing).read_bytes()


def test_listable_locations_are_s3_and_virtual_hosted_amazon_urls():
    assert listable_location('s3://bucket/a.zarr/0/') == 's3://bucket/a.zarr/0'
    assert listable_location('s3+eu-west-1://bucket/a.zarr') == 's3+eu-west-1://bucket/a.zarr'
    assert listable_location('https://bucket.s3.eu-central-1.amazonaws.com/a/s.zarr/4') \
        == 's3+eu-central-1://bucket/a/s.zarr/4'
    assert listable_location('https://bucket.s3.amazonaws.com/a%20b/s.zarr') == 's3://bucket/a b/s.zarr'
    for location in ('https://data.example/s.zarr', 'https://bucket.s3.example.com/a',
                     'http://127.0.0.1:8000/s.zarr', '/local/s.zarr'):
        assert not is_listable(location), location


def test_an_unlistable_array_is_refused(tmp_path):
    with pytest.raises(ValueError, match='cannot be listed'):
        mirror_array('https://data.example/s.zarr', tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


def test_a_prefix_without_a_zarray_is_refused(tmp_path):
    (tmp_path / 'bucket/notes').mkdir(parents=True)
    (tmp_path / 'bucket/notes/readme.txt').write_text('no array here')
    with FakeS3Server(tmp_path / 'bucket') as s3:
        with pytest.raises(ValueError, match='no .zarray'):
            mirror_array(s3.location('notes'), tmp_path / 'cache', access=s3.access())
