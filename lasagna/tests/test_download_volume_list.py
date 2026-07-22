import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from lasagna.scripts import download_volume_list as batch


class DownloadVolumeListTests(unittest.TestCase):
    def test_parse_uri_derives_destination(self):
        volume = batch.parse_volume_uri(
            "s3://vesuvius-challenge-open-data/PHerc0125/volumes/vol-masked.zarr/",
            line_number=1,
        )

        self.assertEqual(volume.scroll, "PHerc0125")
        self.assertEqual(volume.name, "vol-masked.zarr")
        self.assertEqual(volume.relative_path(), Path("PHerc0125/volumes/vol-masked.zarr"))

    def test_list_rejects_multiple_volumes_for_one_scroll(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "volumes.txt"
            path.write_text(
                "s3://vesuvius-challenge-open-data/PHerc0125/volumes/a.zarr\n"
                "s3://vesuvius-challenge-open-data/PHerc0125/volumes/b.zarr\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "overwrite info.json"):
                batch.read_volume_list(path)

    def test_main_calls_scale_zero_downloader_and_writes_info(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            list_path = root / "volumes.txt"
            downloader = root / "download_omezarr.py"
            source = (
                "s3://vesuvius-challenge-open-data/PHerc0125/volumes/"
                "20250821151825-9.362um-1.2m-113keV-masked.zarr"
            )
            list_path.write_text(source + "\n", encoding="utf-8")
            downloader.write_text("# test placeholder\n", encoding="utf-8")

            with mock.patch.object(
                batch.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                result = batch.main([
                    str(list_path),
                    "--output-root", str(root / "scrolls"),
                    "--downloader", str(downloader),
                ])

            self.assertEqual(result, 0)
            command = run.call_args.args[0]
            self.assertIn("--scales", command)
            self.assertEqual(command[command.index("--scales") + 1], "0")
            self.assertIn("--anon", command)
            self.assertEqual(command[command.index("--workers") + 1], "512")
            info_path = root / "scrolls/PHerc0125/info.json"
            self.assertEqual(
                json.loads(info_path.read_text(encoding="utf-8")),
                {
                    "s3_path": source,
                    "local_volume_path": (
                        "volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr"
                    ),
                },
            )


if __name__ == "__main__":
    unittest.main()
