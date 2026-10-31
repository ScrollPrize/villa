"""Offline metadata contract tests for the VC3D zero-copy archive proposal.

Synthetic records only: do not confuse with GitHub API permission or scroll tests.
Run: python -m unittest discover -s .github/scripts/tests -p 'test_vc3d_archive_contract.py'
"""
import copy
import importlib.util
import pathlib
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "vc3d_archive_zero_copy.py"
SPEC = importlib.util.spec_from_file_location("vc3d_archive_zero_copy", SOURCE)
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)

SHA = "ab" * 20
BASE_NAMES = (
    "win64.exe", "win64.zip", "macos-arm64.dmg", "macos-arm64.zip",
    "linux-x86_64.AppImage", "linux-x86_64-debug.tar.zst",
)


def release(channel, start_id):
    return {
        "id": start_id - 1,
        "name": "VC3D Latest" if channel == "latest" else "VC3D Stable",
        "body": (f"Automatically packaged from main at commit {SHA}."
                 if channel == "latest" else
                 f"Promoted from tested package artifacts built at commit {SHA}."),
        "tag_name": channel,
        "assets": [
            {"id": start_id + n, "name": f"VC3D-{SHA[:7]}-2026-10-24-{suffix}",
             "size": 200 + n, "digest": "sha256:" + f"{n+1:064x}"}
            for n, suffix in enumerate(BASE_NAMES)
        ],
    }


class ArchiveMetadataContract(unittest.TestCase):
    def test_both_channels_keep_twelve_distinct_identifiers(self):
        a = ARCHIVE.verify_release_origin(release("latest", 100), "latest", SHA)
        b = ARCHIVE.verify_release_origin(release("stable", 200), "stable", SHA)
        self.assertEqual(sorted(a), list(range(100, 106)))
        self.assertEqual(sorted(b), list(range(200, 206)))
        self.assertTrue(set(a).isdisjoint(b))

    def test_missing_asset_fails(self):
        example = release("latest", 100)
        example["assets"].pop()
        with self.assertRaises(ARCHIVE.ArchiveError):
            ARCHIVE.verify_release_origin(example, "latest", SHA)

    def test_duplicate_asset_id_fails(self):
        example = release("stable", 200)
        example["assets"][3]["id"] = example["assets"][0]["id"]
        with self.assertRaises(ARCHIVE.ArchiveError):
            ARCHIVE.verify_release_origin(example, "stable", SHA)

    def test_duplicate_platform_fails(self):
        example = release("latest", 100)
        example["assets"][3]["name"] = example["assets"][1]["name"]
        with self.assertRaises(ARCHIVE.ArchiveError):
            ARCHIVE.verify_release_origin(example, "latest", SHA)

    def test_unexpected_commit_in_filename_fails(self):
        example = release("latest", 100)
        example["assets"][0]["name"] = example["assets"][0]["name"].replace(SHA[:7], "0123456")
        with self.assertRaises(ARCHIVE.ArchiveError):
            ARCHIVE.verify_release_origin(example, "latest", SHA)

    def test_missing_digest_fails(self):
        example = release("stable", 200)
        example["assets"][0]["digest"] = None
        with self.assertRaises(ARCHIVE.ArchiveError):
            ARCHIVE.verify_release_origin(example, "stable", SHA)

    def test_release_body_and_title_must_match(self):
        for field, replacement in (("name", "Unexpected"), ("body", "Unverified")):
            with self.subTest(field=field):
                example = release("latest", 100)
                example[field] = replacement
                with self.assertRaises(ARCHIVE.ArchiveError):
                    ARCHIVE.verify_release_origin(example, "latest", SHA)

    def test_wrong_channel_fails(self):
        with self.assertRaises(ARCHIVE.ArchiveError):
            ARCHIVE.verify_release_origin(release("stable", 200), "latest", SHA)


if __name__ == "__main__":
    unittest.main()
