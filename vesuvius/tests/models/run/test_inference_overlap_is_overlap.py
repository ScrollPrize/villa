"""--overlap is the fraction of a patch that neighbouring patches share.

The dataset's sliding window takes the stride as a fraction of the patch size, and the
Inferer passed the overlap straight through as that stride. 0.5 is the one value where
the two agree, so the default looked right while --overlap 0.25 produced 75% overlap and
--overlap 0.75 produced 25%. The structure-tensor Inferer computes its own step_size from
the overlap and was ignored the same way.
"""

from __future__ import annotations

import pytest
import torch

from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer


def _captured_step(monkeypatch, *, overlap, step_size=None):
    captured = {}

    class FakeDataset:
        collate_fn = staticmethod(lambda batch: batch)

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.all_positions = []

        def __len__(self):
            return 0

    monkeypatch.setattr(inference, 'VCDataset', FakeDataset)
    inferer = Inferer.__new__(Inferer)
    if step_size is not None:
        inferer.step_size = step_size
    inferer.model_normalization_scheme = None
    inferer.model_intensity_properties = None
    inferer.normalization_scheme = 'instance_zscore'
    inferer.input = 'unused.zarr'
    inferer.patch_size = (8, 8, 8)
    inferer.overlap = overlap
    inferer.num_parts = 1
    inferer.part_id = 0
    inferer.input_format = 'zarr'
    inferer.verbose = False
    inferer.skip_empty_patches = False
    inferer.scroll_id = None
    inferer.segment_id = None
    inferer.energy = None
    inferer.resolution = None
    inferer.input_anon = False
    inferer.bbox = None
    inferer.read_retries = 1
    inferer.device = torch.device('cpu')
    inferer.max_patches = None
    inferer._create_dataset_and_loader()
    return captured['step_size']


@pytest.mark.parametrize('overlap, stride', [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)])
def test_stride_is_the_complement_of_the_overlap(monkeypatch, overlap, stride):
    assert _captured_step(monkeypatch, overlap=overlap) == pytest.approx(stride)


def test_a_subclass_step_size_is_used(monkeypatch):
    assert _captured_step(monkeypatch, overlap=0.0, step_size=0.5) == pytest.approx(0.5)


@pytest.mark.parametrize('overlap', [-0.1, 1.0, 1.5])
def test_overlap_outside_zero_to_one_is_rejected(overlap):
    with pytest.raises(ValueError, match='overlap must be in'):
        Inferer(model_path='m', input_dir='i', output_dir='o', overlap=overlap)
