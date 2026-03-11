from __future__ import annotations

import unittest
from unittest.mock import patch

import aiohttp
import numpy as np

from vesuvius.models.datasets import ssl_zarr_dataset


class OpenZarrTest(unittest.TestCase):
    def test_open_zarr_uses_anon_true_by_default_for_s3(self) -> None:
        with patch.object(ssl_zarr_dataset, "load_volume_auth", return_value=(None, None)), patch.object(
            ssl_zarr_dataset.zarr,
            "open",
            return_value="opened",
        ) as mock_open:
            opened = ssl_zarr_dataset.open_zarr("s3://bucket/example.zarr", 0)

        self.assertEqual(opened, "opened")
        mock_open.assert_called_once_with(
            "s3://bucket/example.zarr",
            path="0",
            mode="r",
            storage_options={"anon": True},
        )

    def test_open_zarr_merges_custom_s3_storage_options(self) -> None:
        with patch.object(ssl_zarr_dataset, "load_volume_auth", return_value=(None, None)), patch.object(
            ssl_zarr_dataset.zarr,
            "open",
            return_value="opened",
        ) as mock_open:
            opened = ssl_zarr_dataset.open_zarr(
                "s3://bucket/example.zarr",
                1,
                s3_storage_options={"anon": False, "key": "abc"},
            )

        self.assertEqual(opened, "opened")
        mock_open.assert_called_once_with(
            "s3://bucket/example.zarr",
            path="1",
            mode="r",
            storage_options={"anon": False, "key": "abc"},
        )

    def test_open_zarr_wraps_missing_s3fs_dependency(self) -> None:
        with patch.object(ssl_zarr_dataset, "load_volume_auth", return_value=(None, None)), patch.object(
            ssl_zarr_dataset.zarr,
            "open",
            side_effect=ImportError("missing s3fs"),
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, "s3fs"):
                ssl_zarr_dataset.open_zarr("s3://bucket/example.zarr", 0)

    def test_open_zarr_uses_fsspec_store_for_authenticated_https(self) -> None:
        fake_store = object()
        patchers = [
            patch.object(ssl_zarr_dataset, "load_volume_auth", return_value=("user", "pass")),
            patch.object(ssl_zarr_dataset.fsspec, "filesystem", return_value="fs"),
            patch.object(ssl_zarr_dataset.zarr, "open", return_value="opened"),
        ]
        if hasattr(ssl_zarr_dataset.zarr.storage, "FsspecStore"):
            patchers.append(patch.object(ssl_zarr_dataset.zarr.storage, "FsspecStore", return_value=fake_store))
        else:
            patchers.append(patch.object(ssl_zarr_dataset.zarr.storage, "FSStore", return_value=fake_store))

        with patchers[0] as _, patchers[1] as mock_filesystem, patchers[2] as mock_open, patchers[3] as mock_store:
            opened = ssl_zarr_dataset.open_zarr("https://example.test/volume.zarr/", 2)

        self.assertEqual(opened, "opened")
        mock_filesystem.assert_called_once_with(
            "https",
            asynchronous=True,
            client_kwargs={"auth": aiohttp.BasicAuth("user", "pass")},
        )
        if hasattr(ssl_zarr_dataset.zarr.storage, "FsspecStore"):
            mock_store.assert_called_once_with(
                fs="fs",
                path="https://example.test/volume.zarr",
                read_only=True,
                allowed_exceptions=(
                    KeyError,
                    FileNotFoundError,
                    PermissionError,
                    OSError,
                    aiohttp.ClientResponseError,
                ),
            )
        else:
            mock_store.assert_called_once_with(
                "https://example.test/volume.zarr",
                fs="fs",
                mode="r",
                check=False,
                create=False,
                exceptions=(KeyError, FileNotFoundError, PermissionError, OSError, aiohttp.ClientResponseError),
            )
        mock_open.assert_called_once_with(fake_store, path="2", mode="r")

    def test_dataset_len_uses_epoch_length_when_configured(self) -> None:
        config = {
            "epoch_length": 123,
            "global_crop_size": [16, 16, 16],
            "datasets": [{"volume_path": "s3://bucket/example.zarr", "volume_scale": 0}],
        }

        with (
            patch.object(ssl_zarr_dataset, "open_zarr", return_value=np.zeros((32, 32, 32), dtype=np.uint8)),
            patch.object(ssl_zarr_dataset, "create_training_transforms", side_effect=lambda size: f"tfm-{size}"),
            patch.object(ssl_zarr_dataset, "get_normalization", return_value=None),
        ):
            dataset = ssl_zarr_dataset.SSLZarrDataset(config, do_augmentations=False)

        self.assertEqual(len(dataset), 123)


if __name__ == "__main__":
    unittest.main()
