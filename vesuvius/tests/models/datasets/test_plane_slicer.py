import numpy as np
import pytest

from vesuvius.models.datasets.slicers import (
    PlaneSliceConfig,
    PlaneSliceVolume,
    PlaneSlicer,
)


def make_config(**overrides):
    base = dict(
        sample_planes=('z',),
        plane_weights={'z': 1.0},
        plane_patch_sizes={'z': (2, 2)},
        primary_plane='z',
        min_labeled_ratio=0.1,
        min_bbox_percent=0.1,
        allow_unlabeled=False,
        random_rotation_planes={},
        random_tilt_planes={},
        label_interpolation={},
        save_plane_masks=False,
        plane_mask_mode='plane',
    )
    base.update(overrides)
    return PlaneSliceConfig(**base)


def test_plane_slicer_builds_index_and_extracts_patch():
    image = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    label = (image > 10).astype(np.float32)

    volume = PlaneSliceVolume(
        index=0,
        name='vol0',
        image=image,
        labels={'main': label},
    )

    slicer = PlaneSlicer(config=make_config(), target_names=['main'])
    slicer.register_volume(volume)
    patches, weights = slicer.build_index(validate=True)

    assert patches, "Expected at least one patch to be produced"
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0

    slicer.set_normalizer(None)
    result = slicer.extract(patches[0])

    assert result.image.shape[0] == 1
    assert result.image.dtype == np.float32
    assert result.labels['main'].dtype == np.float32
    assert 'plane' in result.patch_info


def test_plane_slicer_requires_patch_size():
    config = make_config(plane_patch_sizes={})
    slicer = PlaneSlicer(config=config, target_names=['main'])
    slicer.register_volume(
        PlaneSliceVolume(
            index=0,
            name='v',
            image=np.zeros((4, 4, 4), dtype=np.float32),
            labels={'main': np.zeros((4, 4, 4), dtype=np.float32)},
        )
    )
    with pytest.raises(KeyError):
        slicer.build_index(validate=True)
