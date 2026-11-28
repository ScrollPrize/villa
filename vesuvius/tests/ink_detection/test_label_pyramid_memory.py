from pathlib import Path
import tracemalloc

import numpy as np
import pytest
from vc_reader import images, write_image
from vc_reader.reader import open_group

from vesuvius.ink_detection.preprocessing.create_label_zarrs import (
    DEFAULT_LABEL_SLICE,
    convert_image,
    write_label_pyramid,
)


@pytest.fixture
def supervision_mask():
    path = Path(__file__).parent / 'fixtures/label_png_memory/supervision-mask.png'
    return images.read(path)


@pytest.mark.parametrize('channels', [1, 3, 4])
def test_png_conversion_does_not_allocate_a_full_label_volume(tmp_path, supervision_mask, channels):
    image = supervision_mask
    path = tmp_path / 'w00_supervision_mask.png'
    encoded = image if channels == 1 else np.repeat(image[..., None], channels, axis=-1)
    write_image(path, encoded)
    tracemalloc.start()
    try:
        result = convert_image(path, levels=3, chunk_workers=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result['status'] == 'written'
    assert peak < image.nbytes * 8 + (512 << 10)
    group = open_group(path.with_suffix('.zarr'))
    for level in range(3):
        array = group[str(level)]
        expected = image[::2**level, ::2**level]
        assert array.shape == (65, *expected.shape)
        assert array.dtype == image.dtype
        np.testing.assert_array_equal(array[DEFAULT_LABEL_SLICE], expected)
        assert not array[:DEFAULT_LABEL_SLICE].any()
        assert not array[DEFAULT_LABEL_SLICE + 1:].any()


@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('reduction', ['nearest', 'mean'])
def test_label_pyramid_preserves_real_mask_values_and_empty_planes(
    tmp_path, supervision_mask, dtype, reduction
):
    expected = supervision_mask.astype(dtype)
    output = tmp_path / 'mask.zarr'
    write_label_pyramid(expected, output, levels=3, downsample_mode=reduction, chunk_workers=1)
    group = open_group(output)
    for level in range(3):
        array = group[str(level)]
        assert array.shape == (65, *expected.shape)
        np.testing.assert_array_equal(array[DEFAULT_LABEL_SLICE], expected)
        assert not array[:DEFAULT_LABEL_SLICE].any()
        assert not array[DEFAULT_LABEL_SLICE + 1:].any()
        if reduction == 'nearest':
            expected = expected[::2, ::2]
        else:
            padded = np.pad(expected, tuple((0, size % 2) for size in expected.shape), mode='edge')
            expected = np.rint(padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2)
                               .mean(axis=(1, 3))).astype(dtype)
