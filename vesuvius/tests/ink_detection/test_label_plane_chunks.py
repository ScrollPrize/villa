from pathlib import Path

import numpy as np
import pytest
import tifffile
from vc_reader import images, write_image

from vesuvius.ink_detection.preprocessing import create_label_zarrs as labels
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding


@pytest.mark.parametrize(
    'kind,dtype',
    [('png', np.uint8), ('striped', np.uint8), ('striped', np.uint16),
     ('tiled', np.uint8), ('tiled', np.uint16)],
)
@pytest.mark.parametrize('reduction', ['nearest', 'mean'])
def test_real_labels_use_one_plane_chunks_and_keep_training_reads(tmp_path, kind, dtype, reduction):
    source = Path(__file__).parent / 'fixtures/label_png_memory/supervision-mask.png'
    expected = images.read(source).astype(dtype)
    if dtype == np.uint16:
        expected *= 257
    directory = tmp_path / 'w00'
    directory.mkdir()
    stem = 'w00_supervision_mask' if reduction == 'nearest' else 'w00_composite'
    path = directory / (stem + ('.png' if kind == 'png' else '.tif'))
    if kind == 'png':
        write_image(path, expected)
        labels.write_label_pyramid(
            labels.load_image(path), path.with_suffix('.zarr'),
            levels=4, downsample_mode=reduction, chunk_workers=2,
        )
    else:
        options = {'tile': (16, 16)} if kind == 'tiled' else {}
        tifffile.imwrite(path, expected, **options)
        labels.convert_image(path, levels=4, chunk_workers=2)
    for level in range(4):
        volume = open_volume(path.with_suffix('.zarr'), level)
        np.testing.assert_array_equal(volume[32], expected)
        assert np.count_nonzero(volume[:32]) == 0
        assert np.count_nonzero(volume[33:]) == 0
        height, width = expected.shape
        crop, valid = read_bbox_with_padding(volume, (25, -2, -3, 42, height + 2, width + 3))
        target = np.zeros((17, height + 4, width + 6), dtype=dtype)
        target[7, 2:height + 2, 3:width + 3] = expected
        np.testing.assert_array_equal(crop, target)
        assert valid is not None
        assert volume.shape == (65, height, width)
        assert volume.chunks == (1, 128, 128)
        if reduction == 'nearest':
            expected = expected[::2, ::2]
        else:
            padded = np.pad(expected, ((0, height % 2), (0, width % 2)), mode='edge')
            blocks = padded.reshape((height + 1) // 2, 2, (width + 1) // 2, 2)
            expected = np.rint(blocks.mean(axis=(1, 3))).astype(dtype)


def test_existing_depth65_chunks_remain_readable(tmp_path, monkeypatch):
    source = Path(__file__).parent / 'fixtures/label_png_memory/supervision-mask.png'
    image = images.read(source)
    monkeypatch.setattr(labels, 'CHUNK_SHAPE', (65, 128, 128))
    output = tmp_path / 'old.zarr'
    labels.write_label_pyramid(image, output, levels=3)
    for level in range(3):
        volume = open_volume(output, level)
        assert volume.chunks == (65, 128, 128)
        np.testing.assert_array_equal(volume[32], image[::2**level, ::2**level])
