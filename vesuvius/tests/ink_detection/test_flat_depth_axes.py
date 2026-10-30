"""Declared surface-volume axes must survive the actual flat inference path."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import torch
from torch import nn
import zarr

from vesuvius.ink_detection.inference import infer


class RecordingModel(nn.Module):
    """Untrained instrument: records model input; its output is not ink evidence."""

    def __init__(self):
        super().__init__()
        self.first_input = None
        self.batches = 0

    def forward(self, image):
        if self.first_input is None:
            self.first_input = image[0, 0].detach().cpu().numpy().copy()
        self.batches += 1
        return image[:, :, 0] - 0.5


def make_input(path, canonical, *, axes="zyx", group=False, level="0",
               declare=True, conflict=False, unrelated_entry=False):
    """Write a permutation of the same ZYX samples, with no resampling."""
    values = np.ascontiguousarray(canonical.transpose(tuple("zyx".index(a) for a in axes)))
    version3 = int(zarr.__version__.split(".", 1)[0]) >= 3
    kwargs = {"zarr_format": 2} if version3 else {}
    if group:
        root = zarr.open_group(str(path), mode="w", **kwargs)
        create = root.create_array if version3 else root.create_dataset
        array = create(level, shape=values.shape, chunks=values.shape, dtype="u1")
    else:
        array = zarr.open_array(str(path), mode="w", shape=values.shape,
                                chunks=values.shape, dtype="u1", **kwargs)
        root = array
    array[:] = values
    if declare and not group:
        array.attrs["_ARRAY_DIMENSIONS"] = list(axes)
    if declare and group:
        entries = []
        if unrelated_entry:
            entries.append({"axes": ["x", "y", "z"], "datasets": [{"path": "99"}]})
        entries.append({"version": "0.4", "axes": [
            {"name": a, "type": "space", "unit": "micrometer"} for a in axes],
            "datasets": [{"path": level, "coordinateTransformations": [
                {"type": "scale", "scale": [9.366] * 3}]}]})
        root.attrs["multiscales"] = entries
    if conflict:
        array.attrs["_ARRAY_DIMENSIONS"] = list("yxz" if axes == "zyx" else "zyx")
    return root, array


def capture(path, output, *, direction="forward", resolution="0", input_depth=3):
    model = RecordingModel()
    configured = infer.ConfiguredModel(model=model, patch_size=16,
        input_depth=input_depth, preprocessing="divide_255", amp_dtype=None)
    args = argparse.Namespace(resolution=resolution, overlap=0.0, stride=16,
        blend_mode="constant", layer_start=None, layer_end=None, mask_path=None,
        batch_size=2, gpu_ids=[], num_workers=0, prefetch_factor=2,
        tta_mirror=False, tta_batch_size=None)
    infer.infer_single_zarr(args=args, input_zarr=path, configured_model=configured,
        device=torch.device("cpu"), output_tiff=output, layer_direction=direction)
    return model, tifffile.imread(output)


def expected_first_input(canonical, *, direction="forward", input_depth=3):
    # Independent named-axis oracle: choose the centred native planes, place
    # source-backed samples in the top-left, and divide by the fixed divisor.
    start = canonical.shape[0] // 2 - input_depth // 2
    planes = list(range(start, start + input_depth))
    if direction == "reverse":
        planes.reverse()
    expected = np.zeros((input_depth, 16, 16), dtype=np.float32)
    h, w = min(16, canonical.shape[1]), min(16, canonical.shape[2])
    expected[:, :h, :w] = canonical[planes, :h, :w].astype(np.float32) * (1.0 / 255.0)
    return expected


class FlatDepthAxesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def assert_pipeline(self, *, axes, narrow=True, group=False, declare=True,
                        unrelated_entry=False):
        shape = (9, 4, 12) if narrow else (9, 24, 32)
        canonical = (np.arange(np.prod(shape), dtype=np.uint32) % 251 + 1).astype(np.uint8).reshape(shape)
        path = self.directory / "input.zarr"
        make_input(path, canonical, axes=axes, group=group, declare=declare,
                   unrelated_entry=unrelated_entry)
        for direction in ("forward", "reverse"):
            model, output = capture(path, self.directory / f"{direction}.tif", direction=direction)
            self.assertEqual(output.shape, canonical.shape[1:])
            np.testing.assert_array_equal(model.first_input,
                expected_first_input(canonical, direction=direction))

    def test_zyx_narrow_array_keeps_declared_depth(self):
        self.assert_pipeline(axes="zyx")

    def test_yxz_narrow_array_keeps_declared_depth(self):
        self.assert_pipeline(axes="yxz")

    def test_group_named_axes_reach_the_model(self):
        self.assert_pipeline(axes="zyx", group=True)

    def test_only_matching_multiscale_entry_is_used(self):
        self.assert_pipeline(axes="zyx", group=True, unrelated_entry=True)

    def test_legacy_unlabelled_wide_array_is_unchanged(self):
        self.assert_pipeline(axes="zyx", narrow=False, declare=False)

    def test_named_wide_array_is_unchanged(self):
        self.assert_pipeline(axes="zyx", narrow=False)

    def test_conflicting_axes_reject_before_output_write(self):
        path = self.directory / "input.zarr"
        make_input(path, np.ones((9, 4, 12), dtype=np.uint8), group=True, conflict=True)
        output = self.directory / "existing.tif"
        output.write_bytes(b"existing output must remain unchanged")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            capture(path, output)
        self.assertEqual(output.read_bytes(), b"existing output must remain unchanged")

    def test_unsupported_declared_order_rejects_before_write(self):
        path = self.directory / "input.zarr"
        _, array = make_input(path, np.ones((9, 4, 12), dtype=np.uint8))
        array.attrs["_ARRAY_DIMENSIONS"] = ["x", "y", "z"]
        output = self.directory / "out.tif"
        with self.assertRaisesRegex(ValueError, "ZYX or YXZ"):
            capture(path, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
