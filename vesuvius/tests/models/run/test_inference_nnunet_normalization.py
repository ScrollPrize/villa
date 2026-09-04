from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    CTNormalization,
)

from vesuvius.data.volume import Volume
from vesuvius.models.run import inference
from vesuvius.models.run.inference import (
    Inferer,
    _nnunet_normalization_from_model_info,
)


CT_PROPERTIES = {
    'mean': 87.5442,
    'std': 47.7438,
    'percentile_00_5': 0.0,
    'percentile_99_5': 212.0,
}


def nnunet_model_info(
    scheme='CTNormalization',
    *,
    intensity_properties=None,
    use_mask=False,
):
    if intensity_properties is None:
        intensity_properties = CT_PROPERTIES
    return {
        'configuration_manager': SimpleNamespace(
            normalization_schemes=[scheme],
            use_mask_for_norm=[use_mask],
        ),
        'plans_manager': SimpleNamespace(
            foreground_intensity_properties_per_channel={
                '0': intensity_properties,
            },
        ),
    }


def test_nnunet_ct_normalization_uses_channel_zero_fingerprint():
    scheme, properties = _nnunet_normalization_from_model_info(
        nnunet_model_info())

    assert scheme == 'ct'
    assert properties == CT_PROPERTIES
    assert properties is not CT_PROPERTIES


def test_volume_ct_normalization_matches_nnunet(tmp_path):
    source = np.array(
        [[[-10, 0, 42], [88, 150, 212]], [[213, 255, 100], [1, 87, 200]]],
        dtype=np.int16,
    )
    path = tmp_path / 'input.zarr'
    array = zarr.open(
        str(path), mode='w', shape=source.shape, chunks=source.shape,
        dtype=source.dtype,
    )
    array[:] = source

    volume = Volume(
        type='zarr',
        path=str(path),
        normalization_scheme='ct',
        intensity_props=CT_PROPERTIES,
        return_as_type='np.float32',
        return_as_tensor=False,
    )
    expected = CTNormalization(
        use_mask_for_norm=False,
        intensityproperties=CT_PROPERTIES,
    ).run(source.copy())

    np.testing.assert_array_equal(volume[:, :, :], expected)


@pytest.mark.parametrize(
    ('native', 'expected'),
    [
        ('ZScoreNormalization', 'instance_zscore'),
        ('NoNormalization', 'none'),
        ('RescaleTo01Normalization', 'instance_minmax'),
    ],
)
def test_nnunet_supported_normalization_mapping(native, expected):
    scheme, properties = _nnunet_normalization_from_model_info(
        nnunet_model_info(native))

    assert scheme == expected
    assert properties is None


def test_nnunet_unknown_normalization_fails_instead_of_silent_fallback():
    with pytest.raises(ValueError, match='Unsupported nnU-Net normalization'):
        _nnunet_normalization_from_model_info(
            nnunet_model_info('RGBTo01Normalization'))


def test_nnunet_masked_zscore_fails_instead_of_changing_semantics():
    with pytest.raises(ValueError, match='masked Z-score'):
        _nnunet_normalization_from_model_info(
            nnunet_model_info('ZScoreNormalization', use_mask=True))


def test_nnunet_multichannel_normalization_fails_explicitly():
    model_info = nnunet_model_info()
    model_info['configuration_manager'].normalization_schemes = [
        'CTNormalization', 'CTNormalization',
    ]

    with pytest.raises(ValueError, match='2 input channels'):
        _nnunet_normalization_from_model_info(model_info)


def test_dataset_receives_ct_intensity_properties(monkeypatch):
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
    inferer.model_normalization_scheme = 'ct'
    inferer.model_intensity_properties = dict(CT_PROPERTIES)
    inferer.normalization_scheme = 'instance_zscore'
    inferer.input = 'unused.zarr'
    inferer.patch_size = (8, 8, 8)
    inferer.overlap = 0.5
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
    inferer.chunk_cache_dir = None
    inferer.chunk_cache_max_gb = None
    inferer.device = torch.device('cpu')
    inferer.max_patches = None

    dataset, dataloader = inferer._create_dataset_and_loader()

    assert dataset is inferer.dataset
    assert dataloader is None
    assert captured['normalization_scheme'] == 'ct'
    assert captured['intensity_props'] == CT_PROPERTIES
