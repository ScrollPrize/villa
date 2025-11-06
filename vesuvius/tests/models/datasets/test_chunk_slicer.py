import numpy as np
import pytest

from vesuvius.models.datasets.slicers import (
    ChunkSliceConfig,
    ChunkVolume,
    ChunkSlicer,
)


def make_chunk_config(patch_size=(2, 2, 2), **overrides):
    base = dict(
        patch_size=tuple(patch_size),
        stride=None,
        min_labeled_ratio=0.1,
        min_bbox_percent=0.1,
        allow_unlabeled=False,
        downsample_level=0,
        num_workers=0,
        cache_enabled=False,
        cache_dir=None,
    )
    base.update(overrides)
    return ChunkSliceConfig(**base)


def build_volume(index=0, has_label=True):
    image = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    label = (image > 10).astype(np.float32) if has_label else None
    labels = {'main': label}
    return ChunkVolume(
        index=index,
        name=f'vol{index}',
        image=image,
        labels=labels,
        label_source=label,
        cache_key_path=None,
    )


def test_chunk_slicer_builds_index_and_extracts_patch():
    config = make_chunk_config()
    slicer = ChunkSlicer(config=config, target_names=['main'])
    slicer.register_volume(build_volume())

    patches, weights = slicer.build_index(validate=True)

    assert patches, 'Expected chunk slicer to produce patches'
    assert weights is None

    slicer.set_normalizer(None)
    result = slicer.extract(patches[0])

    assert result.image.shape == (1, 2, 2, 2)
    assert result.image.dtype == np.float32
    assert result.labels['main'].dtype == np.float32
    assert result.labels['main'].shape == (1, 2, 2, 2)
    assert result.patch_info['plane'] == 'volume'


def test_chunk_slicer_enumerates_unlabeled_when_allowed():
    config = make_chunk_config(allow_unlabeled=True)
    slicer = ChunkSlicer(config=config, target_names=['main'])
    unlabeled_volume = build_volume(has_label=False)
    slicer.register_volume(unlabeled_volume)

    patches, _ = slicer.build_index(validate=True)

    assert patches, 'Expected enumeration even without labels when allowed'
    assert all(p.volume_index == unlabeled_volume.index for p in patches)


def test_chunk_slicer_supports_2d_data():
    image = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    label = (image > 5).astype(np.float32)
    volume = ChunkVolume(
        index=0,
        name='vol2d',
        image=image,
        labels={'main': label},
        label_source=label,
        cache_key_path=None,
    )

    config = make_chunk_config(patch_size=(2, 2))
    slicer = ChunkSlicer(config=config, target_names=['main'])
    slicer.register_volume(volume)

    patches, _ = slicer.build_index(validate=True)
    assert patches, 'Expected patches for 2D data'

    result = slicer.extract(patches[0])
    assert result.image.shape == (1, 2, 2)
    assert result.labels['main'].shape == (1, 2, 2)
