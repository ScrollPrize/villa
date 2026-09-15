"""Re-showing the flattened preview a loaded checkpoint already has.

A checkpoint and a raw preview export made from the same model state carry
the same content digest. The service indexes published (flattened) previews
by that digest, pins the ones a saved checkpoint names against retention,
and when the fitter reports that its model now equals a checkpoint - after a
save, a load or a resume - re-shows the surface it already flattened for it.
"""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import torch
from checkpoint_io import model_state_sha256
from fit_session import SpiralInputPaths
from lasagna_publish import PublishedPreview
from preview_index import PublishedPreviewIndex
from spiral_service import PREVIEW_ARTIFACTS_KEPT, ServiceState
from checkpoint_fixtures import _FakeContext, _idle_session


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


DIGEST = "d" * 64
OTHER_DIGEST = "e" * 64


def _frozen_epoch(value):
    return {
        "spiral_and_transform": {"w": torch.full((2,), float(value))},
        "umbilicus_zyx": torch.zeros(3, 3),
        "flow_min_corner_zyx": torch.zeros(3),
        "flow_max_corner_zyx": torch.ones(3),
        "spiral_outward_sense": "CW",
        "flow_integration_steps": 4,
        "flow_integration_solver": "rk4",
        "model_config": {"a": 1},
    }


class ModelStateDigestTests(unittest.TestCase):
    def test_digest_is_content_only_and_covers_what_places_the_surface(self):
        state = {"a": torch.arange(6.0).reshape(2, 3), "b": torch.tensor(True)}
        clone = {key: value.clone() for key, value in state.items()}
        self.assertEqual(model_state_sha256(state, (), 0, 10),
                         model_state_sha256(clone, (), 0, 10))
        # A parameter change, the run window and the frozen-epoch stack each
        # change the surface, so each changes the digest.
        moved = dict(state, a=state["a"] + 1e-6)
        self.assertNotEqual(model_state_sha256(state, (), 0, 10),
                            model_state_sha256(moved, (), 0, 10))
        self.assertNotEqual(model_state_sha256(state, (), 0, 10),
                            model_state_sha256(state, (), 0, 11))
        self.assertNotEqual(
            model_state_sha256(state, (), 0, 10),
            model_state_sha256(state, [_frozen_epoch(1.0)], 0, 10))
        self.assertNotEqual(
            model_state_sha256(state, [_frozen_epoch(1.0)], 0, 10),
            model_state_sha256(state, [_frozen_epoch(2.0)], 0, 10))
        self.assertEqual(
            model_state_sha256(state, [_frozen_epoch(1.0)], 0, 10),
            model_state_sha256(clone, [_frozen_epoch(1.0)], 0, 10))


class _DigestContext(_FakeContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved = []

    def save_checkpoint(self, path, completed_iterations):
        self.saved.append((path, completed_iterations))
        return path

    def model_state_digest(self):
        return DIGEST


def _session(completed=5):
    """An idle session with enough state for status() to be read."""
    session = _idle_session(completed)
    session._warnings = []
    session._error = None
    session._preview_manifest = None
    session._preview_generation = 0
    session._run_config = None
    session._run_config_limits = None
    session._default_advanced_config = None
    return session


class PublishedPreviewIndexTests(unittest.TestCase):
    def _generation(self, base, name, digest, with_surface=True):
        root = base / name
        root.mkdir(parents=True)
        manifest = {"model_state_sha256": digest}
        if with_surface:
            (root / "surface.tifxyz").mkdir()
            manifest["surface_path"] = str(root / "surface.tifxyz")
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root / "manifest.json"

    def test_lookup_returns_only_live_matching_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            index = PublishedPreviewIndex(base)
            self.assertIsNone(index.lookup(DIGEST))
            manifest = self._generation(base, "gen-1", DIGEST)
            index.record(DIGEST, manifest_path=manifest, session_id="s",
                         generation=1, source_fit_iteration=40)
            entry = index.lookup(DIGEST)
            self.assertEqual(entry["manifest_path"], str(manifest))
            self.assertEqual(entry["source_fit_iteration"], 40)
            # A generation whose manifest names another digest, or whose
            # directory is gone, is not this model state's preview.
            manifest.write_text(json.dumps(
                {"model_state_sha256": OTHER_DIGEST}))
            self.assertIsNone(index.lookup(DIGEST))
            manifest.write_text(json.dumps({"model_state_sha256": DIGEST}))
            index.record(DIGEST, manifest_path=manifest, session_id="s",
                         generation=1)
            self.assertIsNotNone(index.lookup(DIGEST))
            manifest.unlink()
            self.assertIsNone(index.lookup(DIGEST))
            document = json.loads(index.path.read_text())
            self.assertEqual(document["previews"], {})

    def test_pins_follow_their_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            index = PublishedPreviewIndex(base)
            first = self._generation(base, "gen-1", DIGEST)
            second = self._generation(base, "gen-2", OTHER_DIGEST)
            index.record(DIGEST, manifest_path=first, session_id="s",
                         generation=1)
            index.record(OTHER_DIGEST, manifest_path=second, session_id="s",
                         generation=2)
            checkpoint = base / "checkpoint_autosave.ckpt"
            checkpoint.write_bytes(b"x")
            index.pin(str(checkpoint), DIGEST)
            self.assertEqual(index.pinned_roots(), {first.parent.resolve()})
            # The autosave is re-saved with another model state: the new
            # digest is pinned, the old one released.
            index.pin(str(checkpoint), OTHER_DIGEST)
            self.assertEqual(index.pinned_roots(), {second.parent.resolve()})
            checkpoint.unlink()
            self.assertEqual(index.pinned_roots(), set())
            # Pins for missing checkpoints and dead digests are harmless.
            index.pin(str(base / "missing.ckpt"), DIGEST)
            index.pin(str(checkpoint), "f" * 64)
            self.assertEqual(index.pinned_roots(), set())


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class ServiceRestoreTests(unittest.TestCase):
    def _state(self, output):
        state = ServiceState()
        state.session_id = "session"
        state.session_paths = SpiralInputPaths.from_mapping({
            "dataset_root": "", "output_directory": str(output)})
        return state

    def _published(self, output, generation, digest, iteration=40):
        root = output / ".spiral-published" / "session" / f"generation-{generation}"
        root.mkdir(parents=True)
        (root / "surface.tifxyz").mkdir()
        (root / "surface.tifxyz" / "x.tif").write_bytes(b"x" * 16)
        (root / "manifest.json").write_text(json.dumps({
            "model_state_sha256": digest,
            "surface_path": str(root / "surface.tifxyz"),
            "source_fit_iteration": iteration}))
        return PublishedPreview(
            manifest_path=root / "manifest.json",
            surface_id=f"surface-{generation}", generation=generation,
            raw_manifest={"model_state_sha256": digest},
            raw_manifest_path=output / "raw" / "manifest.json",
            publish_parent=root.parent, correspondence=None,
            flattened_valid=None,
            source_fit_iteration=iteration)

    def _publish(self, state, output, generation, digest, iteration=40):
        published = self._published(output, generation, digest, iteration)
        raw = output / "raw" / f"g{generation}"
        raw.mkdir(parents=True)
        (raw / "manifest.json").write_text("{}")
        with mock.patch.object(
                state, "_publish_flattened_preview",
                return_value=(mock.Mock(), published)):
            state._maybe_register_artifacts({
                "preview_generation": generation,
                "preview_manifest_path": str(raw / "manifest.json"),
            })
            self.assertTrue(_wait(
                lambda: state._preview.completed_generation >= generation))
        return published

    def test_loading_a_checkpoint_re_shows_its_published_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = self._state(output)
            first = self._publish(state, output, 1, DIGEST, iteration=40)
            restored_ref = state._preview.artifact
            # Training moved on and a newer surface is on display.
            self._publish(state, output, 2, OTHER_DIGEST, iteration=80)
            self.assertNotEqual(state._preview.artifact, restored_ref)
            self.assertEqual(state._preview.source_fit_iteration, 80)

            checkpoint = output / "checkpoints" / "before.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"PK\x03\x04checkpoint")
            state._status_changed({"checkpoint_state": {
                "path": str(checkpoint), "model_state_sha256": DIGEST,
                "completed_iterations": 40}})
            self.assertTrue(_wait(
                lambda: state._preview.model_state_sha256 == DIGEST))
            # The directory was already registered this session, so the very
            # same artifact is shown again rather than a second owner of it.
            self.assertEqual(state._preview.artifact, restored_ref)
            self.assertEqual(state._preview.source_fit_iteration, 40)
            self.assertIsNone(state._preview.diagnostics_artifact)
            self.assertEqual(state.status()["preview_artifact"], restored_ref)
            self.assertEqual(state.status()["preview_source_iteration"], 40)
            # The checkpoint pins that surface against retention.
            self.assertEqual(
                PublishedPreviewIndex(output).pinned_roots(),
                {first.manifest_path.parent.resolve()})
            records = state.events.read_after(0)["events"]
            self.assertTrue(any(
                "Preview restored" in str(record.get("text", ""))
                for record in records), records)

    def test_a_surface_from_an_earlier_service_is_registered_afresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            index = PublishedPreviewIndex(output)
            published = self._published(output, 7, DIGEST, iteration=40)
            index.record(DIGEST, manifest_path=published.manifest_path,
                         session_id="earlier-session", generation=7,
                         source_fit_iteration=40)
            state = self._state(output)
            checkpoint = output / "checkpoint_autosave.ckpt"
            checkpoint.write_bytes(b"PK\x03\x04checkpoint")
            state._status_changed({"checkpoint_state": {
                "path": str(checkpoint), "model_state_sha256": DIGEST,
                "completed_iterations": 40}})
            self.assertTrue(_wait(lambda: state._preview.artifact is not None))
            ref = state._preview.artifact
            self.assertEqual(ref["kind"], "spiral-preview")
            manifest = state.artifacts.manifest(ref["id"])
            self.assertEqual(
                {entry["name"] for entry in manifest["files"]},
                {"manifest.json", "surface.tifxyz/x.tif"})
            self.assertEqual(state._preview.source_fit_iteration, 40)

    def test_pinned_surfaces_survive_fixed_count_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = self._state(output)
            first = self._publish(state, output, 1, DIGEST)
            checkpoint = output / "checkpoints" / "keep.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"PK\x03\x04checkpoint")
            state._status_changed({"checkpoint_state": {
                "path": str(checkpoint), "model_state_sha256": DIGEST,
                "completed_iterations": 40}})
            self.assertTrue(_wait(lambda: PublishedPreviewIndex(
                output).pinned_roots() == {
                    first.manifest_path.parent.resolve()}))
            for generation in range(2, PREVIEW_ARTIFACTS_KEPT + 3):
                self._publish(state, output, generation, f"{generation:064x}")
            self.assertTrue(first.manifest_path.is_file())
            second = (output / ".spiral-published" / "session"
                      / "generation-2")
            self.assertFalse(second.exists())
            # Once the checkpoint is gone the pin lapses and the next prune
            # takes the surface with it.
            checkpoint.unlink()
            self._publish(state, output, PREVIEW_ARTIFACTS_KEPT + 3,
                          "a" * 64)
            self.assertFalse(first.manifest_path.exists())
