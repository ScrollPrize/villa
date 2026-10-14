"""Concurrent writers must not lose each other's previews and pins.

``PublishedPreviewIndex`` re-reads the file for every operation so two
service processes on one output root keep a shared view, but each write is
a read-modify-write of the whole document. Without cross-process
serialization a writer that loaded the file before another stored loses
that other's entry when it stores, and two writers sharing one
``.incoming`` temporary file corrupt the store on Windows (``os.replace``
of a file another process holds open). These tests pin both failure modes
down with real concurrent writers.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview_index import PublishedPreviewIndex

DIGEST = "d" * 64
OTHER_DIGEST = "e" * 64


def _generation(base, name, digest):
    """A published generation manifest that ``_entry_is_live`` accepts."""
    root = base / name
    root.mkdir(parents=True)
    manifest = {"model_state_sha256": digest}
    (root / "surface.tifxyz").mkdir()
    manifest["surface_path"] = str(root / "surface.tifxyz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json"


class StaleSnapshotTests(unittest.TestCase):
    """A stale in-memory snapshot must not erase a concurrent update."""

    def _run_against_paused_writer(self, pause, attempt):
        """Drive one writer paused after ``_load`` and a second update.

        ``pause(writer)`` runs in the first writer's thread after it has
        loaded the document and before it stores: whatever the second
        writer commits in that window must survive the first writer's
        store. The first writer is released only after the second has
        attempted its update, so the overlap is a handshake, not a
        scheduler race and not a timeout.
        """
        writer = PublishedPreviewIndex(self.base)
        loaded = threading.Event()
        release = threading.Event()
        outcomes = {}

        original_load = PublishedPreviewIndex._load

        def pausing_load(index):
            document = original_load(index)
            if index is writer:
                loaded.set()
                if not release.wait(timeout=10):
                    raise AssertionError(
                        "first writer was never released")
            return document

        with mock.patch.object(
                PublishedPreviewIndex, "_load", pausing_load):
            def first_write():
                try:
                    pause(writer)
                    outcomes["first"] = True
                except Exception as exc:
                    outcomes["first"] = exc

            first = threading.Thread(target=first_write, daemon=True)
            first.start()
            self.assertTrue(loaded.wait(timeout=5),
                            "first writer never loaded the document")

            attempted = threading.Event()

            def second_write():
                attempted.set()
                try:
                    attempt(PublishedPreviewIndex(self.base))
                    outcomes["second"] = True
                except Exception as exc:
                    outcomes["second"] = exc

            second = threading.Thread(target=second_write, daemon=True)
            second.start()
            self.assertTrue(attempted.wait(timeout=5),
                            "second writer never attempted its update")
            release.set()
            first.join(timeout=10)
            second.join(timeout=10)
            self.assertFalse(first.is_alive(), "first writer hung")
            self.assertFalse(second.is_alive(), "second writer hung")
        self.assertEqual(outcomes, {"first": True, "second": True})
        return writer

    def test_a_second_record_made_while_one_is_in_flight_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.base = Path(temporary)
            first_manifest = _generation(self.base, "gen-1", DIGEST)
            second_manifest = _generation(self.base, "gen-2", OTHER_DIGEST)

            def pause(writer):
                writer.record(DIGEST, manifest_path=first_manifest,
                              session_id="s1", generation=1)

            def attempt(index):
                index.record(OTHER_DIGEST, manifest_path=second_manifest,
                             session_id="s2", generation=2)

            writer = self._run_against_paused_writer(pause, attempt)
            document = json.loads(writer.path.read_text())
            self.assertEqual(
                set(document["previews"]), {DIGEST, OTHER_DIGEST})
            self.assertEqual(
                document["previews"][OTHER_DIGEST]["session_id"], "s2")
            self.assertEqual(
                document["previews"][OTHER_DIGEST]["manifest_path"],
                str(second_manifest))

    def test_a_pin_made_while_a_record_is_in_flight_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.base = Path(temporary)
            manifest = _generation(self.base, "gen-1", DIGEST)
            checkpoint = self.base / "checkpoints" / "keep.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")

            def pause(writer):
                writer.record(DIGEST, manifest_path=manifest,
                              session_id="s1", generation=1)

            def attempt(index):
                index.pin(str(checkpoint), DIGEST)

            writer = self._run_against_paused_writer(pause, attempt)
            document = json.loads(writer.path.read_text())
            self.assertIn(DIGEST, document["previews"])
            self.assertEqual(document["pins"], {str(checkpoint): DIGEST})


class CrossProcessTests(unittest.TestCase):
    def test_two_processes_lose_neither_previews_nor_pins(self):
        script = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[4])\n"
            "from preview_index import PublishedPreviewIndex\n"
            "base = Path(sys.argv[1]); worker = int(sys.argv[2])\n"
            "index = PublishedPreviewIndex(base)\n"
            "for i in range(int(sys.argv[3])):\n"
            "    digest = f'{(worker << 8) | i:064x}'\n"
            "    manifest = base / f'w{worker}-i{i}' / 'manifest.json'\n"
            "    manifest.parent.mkdir(parents=True, exist_ok=True)\n"
            "    manifest.write_text(\n"
            "        json.dumps({'model_state_sha256': digest}))\n"
            "    checkpoint = base / f'w{worker}-i{i}.ckpt'\n"
            "    checkpoint.write_bytes(b'checkpoint')\n"
            "    index.record(digest, manifest_path=manifest,\n"
            "                 session_id=f'w{worker}', generation=i)\n"
            "    index.pin(str(checkpoint), digest)\n"
            "    index.lookup(digest)\n")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            count = 20
            root = str(Path(__file__).resolve().parents[1])
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(base),
                     str(worker), str(count), root],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for worker in range(2)]
            try:
                for process in processes:
                    _, stderr = process.communicate(timeout=120)
                    self.assertEqual(
                        process.returncode, 0, stderr.decode())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=30)
            index = PublishedPreviewIndex(base)
            # The two processes raced over the whole read-modify-write
            # document; before the file lock one could replace the other's
            # updates wholesale, and both could also fail outright on the
            # shared ``.incoming`` temporary.
            for worker in range(2):
                for i in range(count):
                    digest = f"{(worker << 8) | i:064x}"
                    entry = index.lookup(digest)
                    self.assertIsNotNone(
                        entry, f"worker {worker} iteration {i} lost")
                    self.assertEqual(entry["session_id"], f"w{worker}")
            document = json.loads(index.path.read_text())
            for worker in range(2):
                for i in range(count):
                    self.assertIn(
                        str(base / f"w{worker}-i{i}.ckpt"),
                        document["pins"])
            self.assertEqual(
                [entry.name for entry in (base / ".spiral-published").iterdir()
                 if entry.name.endswith(".incoming")], [])


if __name__ == "__main__":
    unittest.main()
