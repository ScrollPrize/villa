"""Released labels are cached by content identity; local overrides need no network."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vesuvius.ink_detection.paris4_3d_gambiarra import labels


class Downloads(unittest.TestCase):
    def test_download_cache_and_corruption_recovery(self):
        payload = b'released scribbles'
        with tempfile.TemporaryDirectory() as tmp, \
                patch('huggingface_hub.cached_assets_path', return_value=Path(tmp)), \
                patch.object(labels, 'SCRIBBLES_SHA256', hashlib.sha256(payload).hexdigest()):
            def download(bucket, files, **kwargs):
                self.assertEqual(bucket, 'scrollprize/datasets')
                self.assertEqual(files[0][0], labels.SCRIBBLES_FILE)
                self.assertTrue(kwargs['raise_on_missing_files'])
                files[0][1].write_bytes(payload)
            with patch('huggingface_hub.download_bucket_files', side_effect=download) as fetch:
                path = labels.resolve_scribbles()
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(labels.resolve_scribbles(), path)
                self.assertEqual(fetch.call_count, 1)
                path.write_bytes(b'corrupt')
                self.assertEqual(labels.resolve_scribbles().read_bytes(), payload)
                self.assertEqual(fetch.call_count, 2)
            with patch('huggingface_hub.download_bucket_files', side_effect=lambda b, f, **kw: f[0][1].write_bytes(b'wrong')):
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                    labels.resolve_scribbles()
                self.assertFalse(path.exists())

    def test_local_override_does_not_download(self):
        with patch('huggingface_hub.download_bucket_files', side_effect=AssertionError('network used')):
            self.assertEqual(labels.resolve_scribbles('/local/scribbles.json'), Path('/local/scribbles.json'))
