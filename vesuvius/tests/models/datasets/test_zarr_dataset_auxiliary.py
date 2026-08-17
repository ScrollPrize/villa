from types import SimpleNamespace

import pytest

from vesuvius.models.datasets.zarr_dataset import ZarrDataset


def test_zarr_dataset_rejects_unsupported_auxiliary_targets(tmp_path):
    mgr = SimpleNamespace(
        data_path=tmp_path,
        train_patch_size=(8, 8, 8),
        targets={
            "ink": {"auxiliary_task": False},
            "distance_transform": {
                "auxiliary_task": True,
                "task_type": "distance_transform",
                "source_target": "ink",
            },
        },
    )

    with pytest.raises(ValueError, match="cannot generate auxiliary targets"):
        ZarrDataset(mgr, is_training=False)
