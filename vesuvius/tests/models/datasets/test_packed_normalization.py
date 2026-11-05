"""Packed-sample boundary regressions. Scientific execution is PBS-only.

Actual packer layout/mapping functions and actual ZarrDataset constructor/items
execute on generated stores. Config loading is supplied metadata; no TIFF read,
model, optimizer, training, scan or actual annotation is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing import pack_tifs_to_sparse_zarr as packer
from vesuvius.models.training.normalization import get_normalization


def tree(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class PackedNormalizationTests(unittest.TestCase):
    def fixture(self, scheme, *, neighbor=4000, ndim=3, gap=0, overlap=True, mapped=True):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        shape = (8,) * ndim
        layout = packer.calculate_layout(2, shape, gap, 4)
        source = (100 + np.arange(8 ** ndim, dtype=np.uint16)).reshape(shape)
        images = np.zeros(layout.volume_shape, dtype=np.uint16)
        labels = np.zeros(layout.volume_shape, dtype=np.uint8)
        own = (slice(0, 8),) * ndim
        images[own] = source; labels[own] = 1
        other = (slice(0, 8),) * (ndim - 1) + (slice(8 + gap, 16 + gap),)
        images[other] = neighbor; labels[other] = 2
        for path, values in [(root/'images/volume.zarr', images), (root/'labels/volume_ink.zarr', labels)]:
            array = create_level_dataset(str(path), '0', values.shape, (4,) * ndim, values.dtype, None)
            array[:] = values
        samples = [packer.SampleInfo(i, f'engineered{i}.tif', root/f'image{i}.tif', root/f'label{i}.tif',
                    None, None, shape, True, 1.0) for i in range(2)]
        if mapped:
            packer.save_mapping(root, samples, layout, root/'supplied-source-metadata', 'ink', 1)
        width = 8 + gap + 4 if overlap else 8
        patch_size = (8,) * (ndim - 1) + (width,)
        properties = {'mean': 120.0, 'std': 30.0, 'percentile_00_5': 0.0, 'percentile_99_5': 2000.0}
        manager = SimpleNamespace(data_path=root, train_patch_size=patch_size,
            targets={'ink': {'valid_patch_value': 1}}, dataset_config={'valid_patch_value': 1},
            allow_gap_extension=True, cache_valid_patches=False, skip_patch_validation=not mapped,
            unlabeled_foreground_enabled=False, normalization_scheme=scheme,
            intensity_properties=properties, skip_intensity_sampling=True, ome_zarr_resolution=0)
        before = tree(root)
        dataset = ZarrDataset(manager, is_training=False)
        index = next(i for i, p in enumerate(dataset.valid_patches) if p.position == (0,) * ndim)
        item = dataset[index]
        self.assertEqual(tree(root), before)
        return item, source, images, labels, patch_size, properties

    def assert_expected(self, fixture, scheme, *, safe_width):
        item, source, images, labels, size, properties = fixture
        region = (slice(0, 8),) * (source.ndim - 1) + (slice(0, safe_width),)
        normalizer = get_normalization(scheme, properties)
        actual_region = images[region]
        normalized = normalizer.run(actual_region) if normalizer else actual_region.astype(np.float32)
        expected = np.zeros(size, dtype=np.float32)
        expected[region] = normalized
        np.testing.assert_allclose(item['image'].numpy()[0], expected, rtol=1e-6, atol=1e-6)
        expected_labels = np.zeros(size, dtype=np.float32); expected_labels[region] = labels[region]
        np.testing.assert_array_equal(item['ink'].numpy()[0], expected_labels)
        mask = np.zeros(size, dtype=np.float32); mask[region] = 1
        np.testing.assert_array_equal(item['padding_mask'].numpy()[0], mask)
        self.assertFalse(bool(item['is_unlabeled']))

    def independent_of_neighbor(self, scheme, *, ndim=3, gap=0):
        low = self.fixture(scheme, neighbor=0, ndim=ndim, gap=gap)
        high = self.fixture(scheme, neighbor=4000, ndim=ndim, gap=gap)
        # Both the exact own-region reference and neighbor independence matter.
        self.assert_expected(low, scheme, safe_width=8 + gap)
        self.assert_expected(high, scheme, safe_width=8 + gap)
        np.testing.assert_array_equal(low[0]['image'].numpy(), high[0]['image'].numpy())

    def test_zscore_excludes_neighbor(self):
        self.independent_of_neighbor('zscore')

    def test_minmax_excludes_neighbor(self):
        self.independent_of_neighbor('minmax')

    def test_robust_excludes_neighbor(self):
        self.independent_of_neighbor('robust')

    def test_2d_normalization_excludes_neighbor(self):
        self.independent_of_neighbor('zscore', ndim=2)

    def test_gap_extension_retains_own_gap_but_excludes_neighbor(self):
        self.independent_of_neighbor('zscore', gap=4)

    def test_identity_and_global_ct_control(self):
        for scheme in ['none', 'ct']:
            with self.subTest(scheme=scheme): self.independent_of_neighbor(scheme)

    def test_boundary_fitting_patch_control(self):
        for scheme in ['zscore', 'minmax', 'robust']:
            with self.subTest(scheme=scheme):
                self.assert_expected(self.fixture(scheme, overlap=False), scheme, safe_width=8)

    def test_unmapped_dense_patch_control(self):
        fixture = self.fixture('zscore', mapped=False)
        self.assert_expected(fixture, 'zscore', safe_width=fixture[4][-1])


if __name__ == '__main__': unittest.main(verbosity=2)
