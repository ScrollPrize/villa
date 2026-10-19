"""VC3D cache routing keeps the checkpoint's CT identity and physical level."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec, open_ct_array


@pytest.fixture
def native(monkeypatch):
    opened, reads = [], []

    class Volume:
        dtype = 'uint8'
        fill_value = 0
        remote_cache_path = '/viewer/cache/volume'

        @staticmethod
        def open_url(url):
            opened.append(url)
            value = Volume()
            value.base_scale_level = int(url.split('#vc-base-scale=')[-1]) if '#' in url else 0
            return value

        def has_scale_level(self, level):
            return 0 <= level <= 5

        def shape_at(self, level):
            return (8, 8, 8)

        def chunk_shape(self, level):
            return (4, 4, 4)

        def read_chunk(self, level, key, blocking=True):
            reads.append((level, key, blocking))
            return np.full((4, 4, 4), 173, dtype=np.uint8)

    monkeypatch.setitem(sys.modules, 'vc.volume', SimpleNamespace(Volume=Volume, set_chunk_cache_budget=lambda _: None))
    monkeypatch.setenv('VESUVIUS_CT_READER', 'vc3d')
    monkeypatch.setenv('VESUVIUS_VC3D_VOLUME_SOURCE', 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/test.zarr')
    monkeypatch.setenv('VESUVIUS_VC3D_VOLUME_LOCATOR', 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/test.zarr#vc-base-scale=2')
    return opened, reads


def test_matching_viewer_reuses_locator_and_translates_level(native):
    opened, reads = native
    spec = CTVolumeSpec('s3://vesuvius-challenge-open-data/test.zarr', ct_level=3)
    reader = open_ct_array(spec, 1 << 20)
    assert opened == ['https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/test.zarr#vc-base-scale=2']
    np.testing.assert_array_equal(reader.read((1, 1, 1), (2, 2, 2)), np.full((2, 2, 2), 173, np.uint8))
    assert reads == [(1, (0, 0, 0), True)]
    assert reader.path == '/viewer/cache/volume'
    reader.chunk((0, 0, 0))
    assert len(reads) == 1


@pytest.mark.parametrize('url,level', [
    ('s3://vesuvius-challenge-open-data/test.zarr', 1),
    ('s3://vesuvius-challenge-open-data/different.zarr', 3),
])
def test_finer_level_or_different_source_keeps_original_pyramid(native, url, level):
    opened, reads = native
    reader = open_ct_array(CTVolumeSpec(url, ct_level=level), 1 << 20)
    reader.chunk((0, 0, 0))
    assert opened == [url]
    assert reads == [(level, (0, 0, 0), True)]


def test_native_fetch_errors_are_not_silent_fill(native, monkeypatch):
    reader = open_ct_array(CTVolumeSpec('s3://vesuvius-challenge-open-data/test.zarr', ct_level=2), 1 << 20)
    monkeypatch.setattr(reader._volume, 'read_chunk', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='no data'):
        reader.chunk((0, 0, 0))
