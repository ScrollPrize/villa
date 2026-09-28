"""Temporary-directory regression tests for interrupted-output preservation."""
import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_failure_mining import archive_incomplete_output


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'full'

    def test_missing_and_repeated_call(self):
        self.assertIsNone(archive_incomplete_output(self.output))
        self.output.mkdir()
        (self.output / 'trace.json').write_text('evidence')
        (self.output.parent / 'full.json').write_text('binding')
        archived = archive_incomplete_output(self.output)
        self.assertEqual((archived / 'trace.json').read_text(), 'evidence')
        self.assertEqual((self.output.parent / 'full.json').read_text(), 'binding')
        self.assertIsNone(archive_incomplete_output(self.output))

    def test_incomplete_summary(self):
        self.output.mkdir()
        (self.output / 'summary.json').write_text(json.dumps({'completed': False}))
        archived = archive_incomplete_output(self.output)
        self.assertTrue((archived / 'summary.json').exists())

    def test_completed_and_malformed_fail_closed(self):
        self.output.mkdir()
        summary = self.output / 'summary.json'
        for text, error in [(' {"completed": true}', RuntimeError), ('invalid', ValueError)]:
            summary.write_text(text)
            with self.assertRaises(error):
                archive_incomplete_output(self.output)
            self.assertTrue(summary.exists())

    def test_reject_file_and_symlink(self):
        self.output.write_text('not a directory')
        with self.assertRaises(RuntimeError):
            archive_incomplete_output(self.output)
        self.output.unlink()
        target = self.output.parent / 'cache'
        target.mkdir()
        self.output.symlink_to(target)
        with self.assertRaises(RuntimeError):
            archive_incomplete_output(self.output)
        self.assertTrue(target.exists())

    def test_collision_does_not_overwrite(self):
        self.output.mkdir()
        fixed = datetime.datetime(2026, 9, 28, tzinfo=datetime.timezone.utc)
        target = self.output.with_name('full.interrupted.20260928T000000000000Z')
        target.mkdir()
        with patch('training_failure_mining.datetime.datetime') as clock:
            clock.now.return_value = fixed
            with self.assertRaises(RuntimeError):
                archive_incomplete_output(self.output)
        self.assertTrue(self.output.exists())
        self.assertTrue(target.exists())


if __name__ == '__main__':
    unittest.main()
