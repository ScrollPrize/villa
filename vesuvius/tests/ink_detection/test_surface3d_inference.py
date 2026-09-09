"""Independent reference checks for 3D surface tiling, boundaries and quantization."""

import itertools

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.inference.infer_surface3d import (
    SurfacePredictor, axis_starts, blend_window, depth_support,
)


class LocalLogits(torch.nn.Module):
    def forward(self, image):
        return 3 * image - 1


@pytest.mark.parametrize('depth,support', [(65, (2,64)), (109, (24,86))])
def test_canonical_full_segment_reversal_and_export_alignment(depth, support):
    from vesuvius.ink_detection.inference.infer_surface3d import ReversedDepthVolume, evaluated_depth_support, Canonical3DOnly
    class Only3D(torch.nn.Module):
        def forward(self, image):
            pytest.fail('2D forward executed')
        def forward_3d(self, image):
            return image
    raw = np.broadcast_to(np.linspace(-3,3,depth,dtype=np.float32)[:,None,None],(depth,8,8)).copy()
    source = ReversedDepthVolume(raw)
    np.testing.assert_array_equal(source[0:64,:,:],raw[::-1][:64])
    plan={'patch_zyx':(64,8,8),'stride_zyx':(32,4,4),'batch_size':2,
          'depth_mode':'centered','valid_depth_margin':1}
    predictor=SurfacePredictor(Canonical3DOnly(Only3D()),NormalizationConfig.from_value('none'),plan,torch.device('cpu'))
    try: values,_=predictor.predict_tile(source,(0,8,0,8))
    finally: predictor.close()
    exported=values[::-1]
    start, stop = support
    assert evaluated_depth_support(raw.shape,plan['patch_zyx'],'centered',1,True)==support
    assert not exported[:start].any() and not exported[stop:].any()
    expected=torch.sigmoid(torch.from_numpy(raw[start:stop])).mul(255).round().byte().numpy()
    np.testing.assert_array_equal(exported[start:stop],expected)


def reference(raw, patch, stride, normalization):
    """Blend the complete global lattice on CPU, independently of tile planning."""
    num, den = np.zeros(raw.shape, np.float32), np.zeros(raw.shape, np.float32)
    window = blend_window(patch).numpy()
    for origin in itertools.product(*(axis_starts(n, p, s) for n, p, s in zip(raw.shape, patch, stride))):
        bounds = tuple(slice(a, min(a + p, n)) for a, p, n in zip(origin, patch, raw.shape))
        crop = raw[bounds]
        valid_shape = crop.shape
        crop = np.pad(crop, [(0, p - n) for p, n in zip(patch, valid_shape)])
        image = normalize_image(crop, normalization)
        prediction = torch.sigmoid(3 * torch.from_numpy(image) - 1).numpy()
        prediction *= crop.any(axis=0)[None]
        slices = tuple(slice(0, n) for n in valid_shape)
        num[bounds] += (prediction * window)[slices]
        den[bounds] += window[slices]
    assert np.all(den > 0)
    return np.rint(255 * num / den).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("shape", [(11, 19, 23), (3, 5, 7)])
@pytest.mark.parametrize("batch", [1, 3])
def test_tiles_match_independent_full_volume_reference(shape, batch):
    raw = np.random.default_rng(27).integers(0, 256, shape, dtype=np.uint8)
    raw[:, :3, :4] = 0
    patch, stride = (4, 8, 8), (2, 4, 4)
    normalization = NormalizationConfig.from_value("percentile_minmax")
    plan = {"patch_zyx": patch, "stride_zyx": stride, "batch_size": batch, "normalization_workers": 2}
    expected = reference(raw, patch, stride, normalization)
    actual = np.zeros(shape, np.uint8)
    predictor = SurfacePredictor(LocalLogits(), normalization, plan, torch.device("cpu"))
    try:
        for y in range(0, shape[1], 7):
            for x in range(0, shape[2], 9):
                bounds = (y, min(y+7, shape[1]), x, min(x+9, shape[2]))
                result, _ = predictor.predict_tile(raw, bounds)
                actual[:, bounds[0]:bounds[1], bounds[2]:bounds[3]] = result
    finally:
        predictor.close()
    np.testing.assert_array_equal(actual, expected)
    assert not actual[:, :3, :4].any()


def test_blank_volume_never_runs_the_network():
    class Forbidden(torch.nn.Module):
        def forward(self, image):
            pytest.fail("Blank input reached the network")
    plan = {"patch_zyx": (4,8,8), "stride_zyx": (2,4,4), "batch_size": 3}
    predictor = SurfacePredictor(Forbidden(), NormalizationConfig.from_value("none"), plan, torch.device("cpu"))
    try:
        result, stats = predictor.predict_tile(np.zeros((9,17,21), np.uint8), (0,17,0,21))
    finally:
        predictor.close()
    assert result.shape == (9,17,21) and not result.any() and stats["patches"] == 0
