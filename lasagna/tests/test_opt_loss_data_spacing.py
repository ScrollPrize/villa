"""Normal-import CPU regressions using the actual dense manifest loader.

No extracted functions, replaced samplers, external data, or optimizer runs.
Run: python -m unittest discover -s tests -p test_opt_loss_data_spacing.py -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_data
import opt_loss_data
from lasagna_volume import ChannelGroup, LasagnaVolume


def make_manifest(root, *, cos_level=1, density_level=2, source_to_base=1.0):
    groups = {}
    for channel, level in (("cos", cos_level), ("grad_mag", density_level),
                           ("nx", density_level), ("ny", density_level)):
        size = 16 // (2 ** level)
        values = (np.broadcast_to(16 + 2 * np.arange(size), (size, size, size))
                  if channel == "cos" else np.full((size, size, size), 128))
        array = zarr.open(str(root / (channel + ".zarr")), mode="w",
                          shape=values.shape, chunks=values.shape, dtype="uint8",
                          zarr_format=2)
        array[:] = values.astype(np.uint8)
        groups[channel] = ChannelGroup(channel + ".zarr", level, [channel])
    manifest = LasagnaVolume(path=root / "volume.lasagna.json", groups=groups,
                            source_to_base=source_to_base,
                            base_shape_zyx=(16, 16, 16))
    manifest.save()
    return manifest.path


def load_manifest(path, crop=None, cuda_gridsample=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return fit_data.load_3d(path=str(path), device=torch.device("cpu"),
                               crop=crop, cuda_gridsample=cuda_gridsample)


def sample_pair(data, xyz):
    query = torch.tensor(xyz, dtype=torch.float32).reshape(1, 1, -1, 3)
    direct_query = query.clone().requires_grad_(True)
    reference_query = query.clone().requires_grad_(True)
    direct = opt_loss_data._sample_cos_diff(SimpleNamespace(data=data, xyz_hr=direct_query))
    reference = data.grid_sample_fullres(reference_query, diff=True, channels={"cos"})
    reference = reference.cos.squeeze(0).permute(1, 0, 2, 3)
    direct_grad, = torch.autograd.grad(direct.sum(), direct_query)
    reference_grad, = torch.autograd.grad(reference.sum(), reference_query)
    return direct, reference, direct_grad, reference_grad


class DenseCosSpacingTests(unittest.TestCase):
    def check_case(self, *, cos_level=1, density_level=2, source_to_base=1.0,
                   crop=None, xyz=((8, 8, 8),), cuda_gridsample=True):
        with tempfile.TemporaryDirectory(prefix="lasagna-cos-spacing-") as td:
            manifest = make_manifest(Path(td), cos_level=cos_level,
                                     density_level=density_level,
                                     source_to_base=source_to_base)
            data = load_manifest(manifest, crop, cuda_gridsample)
            self.assertEqual(data._spacing_for("cos"), (2 ** cos_level * source_to_base,) * 3)
            self.assertEqual(data.spacing, (2 ** density_level * source_to_base,) * 3)
            pair = sample_pair(data, xyz)
            torch.testing.assert_close(pair[0], pair[1], rtol=0, atol=2e-7)
            torch.testing.assert_close(pair[2], pair[3], rtol=0, atol=2e-7)
            return pair

    def test_mixed_spacing_analytic_value_and_gradient(self):
        direct, _, grad, _ = self.check_case()
        self.assertAlmostEqual(float(direct.detach().item()), 24 / 255, delta=2e-7)
        torch.testing.assert_close(grad.flatten(), torch.tensor([1 / 255, 0, 0]), rtol=0, atol=2e-7)

    def test_equal_spacing_negative_control(self):
        self.check_case(cos_level=1, density_level=1)

    def test_reverse_spacing_ratio(self):
        self.check_case(cos_level=2, density_level=1)

    def test_aligned_nonzero_crop_origin(self):
        self.check_case(crop=(4, 4, 4, 12, 12, 12), xyz=((8, 8, 8), (10, 10, 10)))

    def test_nonunit_source_to_base(self):
        self.check_case(source_to_base=2.0, xyz=((16, 16, 16), (20, 18, 22)))

    def test_boundary_and_zero_padding(self):
        self.check_case(xyz=((-2, 4, 4), (0, 4, 4), (14, 4, 4), (16, 4, 4)))

    def test_torch_sampler_parity(self):
        self.check_case(cuda_gridsample=False, xyz=((7.25, 7.5, 8.75),))

    def test_public_loss_maps_at_same_physical_positions(self):
        with tempfile.TemporaryDirectory(prefix="lasagna-cos-loss-") as td:
            data = load_manifest(make_manifest(Path(td)))
            query = torch.tensor([[[[6., 8., 8.], [8., 8., 8.], [10., 8., 8.]]]],
                                 requires_grad=True)
            target = data.grid_sample_fullres(query, diff=True, channels={"cos"})
            target = target.cos.squeeze(0).permute(1, 0, 2, 3).detach()
            res = SimpleNamespace(data=data, xyz_hr=query, target_mod=target,
                                  target_plain=target, mask_hr=torch.ones_like(target))
            for loss_fn in (opt_loss_data.data_loss, opt_loss_data.data_plain_loss):
                loss, _, _ = loss_fn(res=res)
                self.assertAlmostEqual(float(loss.detach()), 0.0, delta=1e-12)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
