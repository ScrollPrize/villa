"""Logits parts as inference writes them, blended and finalized, on local stores."""
import os

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter, open_zarr

from vesuvius.models.run.blending import merge_inference_outputs
from vesuvius.models.run.finalize_outputs import finalize_logits


def write_part(parent, part_id, patches, coords, volume_shape):
    patches = np.asarray(patches, np.float16)
    logits = ZarrWriter.create(
        f"{parent}/logits_part_{part_id}.zarr",
        shape=patches.shape,
        chunks=(1, *patches.shape[1:]),
        dtype=np.float16,
        cname="zstd",
        level=1,
        skip_empty=True,
    )
    logits[...] = patches
    logits.attrs.update(patch_size=list(patches.shape[2:]), original_volume_shape=list(volume_shape), overlap=0.5)
    positions = ZarrWriter.create(f"{parent}/coordinates_part_{part_id}.zarr", shape=(len(coords), 3),
                                  chunks=(max(1, len(coords)), 3), dtype=np.int32)
    if len(coords):
        positions[...] = np.asarray(coords, np.int32)


@pytest.fixture
def parts(tmp_path):
    parent = str(tmp_path)
    volume = (4, 4, 8)
    patch = np.zeros((2, 2, 4, 4, 4))
    patch[0, 1] = 2.0  # class 1 wins in the first patch
    patch[1, 1] = 2.0  # and in the second; an all-zero part follows
    write_part(parent, 0, patch, [(0, 0, 0), (0, 0, 4)], volume)
    write_part(parent, 1, np.zeros((1, 2, 4, 4, 4)), [(0, 0, 0)], volume)
    write_part(parent, 2, np.zeros((0, 2, 4, 4, 4)), [], volume)
    return parent, volume


def test_blend_then_finalize(tmp_path, parts):
    parent, volume = parts
    blended = f"{parent}/blended.zarr"
    merge_inference_outputs(parent, blended, num_workers=1, verbose=False)
    merged = open_zarr(blended)
    assert merged.shape == (2, *volume)
    assert merged.dtype == np.float16
    assert merged.attrs["sigma_scale"] == 8.0
    assert merged.attrs["overlap"] == 0.5
    values = merged[...]
    # Patch corners have almost no Gaussian weight; a patch's centre keeps its value.
    np.testing.assert_allclose(values[1][1:3, 1:3, 1:3], 2.0, atol=1e-2)
    np.testing.assert_allclose(values[0], 0.0)
    # The empty patches of part 1 contributed nothing, and that is cached with the part.
    assert (tmp_path / "logits_part_1.zarr" / ".non_empty_patch_idxs.json").is_file()

    final = f"{parent}/final.zarr"
    finalize_logits(blended, final, mode="binary", num_workers=1, verbose=False, delete_intermediates=True)
    output = open_zarr(final)
    assert output.dtype == np.uint8
    assert output.attrs["processing_mode"] == "binary"
    expected = np.round(255 * np.exp(2.0) / (1 + np.exp(2.0)))
    assert abs(int(output[...].max()) - expected) <= 1
    assert not os.path.exists(blended)
