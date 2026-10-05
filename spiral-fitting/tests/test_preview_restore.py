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
import threading
import time
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_io import model_state_sha256
from fit_session import SessionState, SpiralInputPaths
from lasagna_publish import PublishedPreview
from preview_index import PublishedPreviewIndex
from service_artifacts import ArtifactRegistry
from spiral_runtime import (ApplyCheckpointCommand, PreflightCheckpointCommand,
                            SaveCheckpointCommand)
from spiral_service import PREVIEW_ARTIFACTS_KEPT, ServiceState
from test_checkpoint_load import _FakeContext, _idle_session

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

    def test_every_tensor_dtype_hashes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                digest = model_state_sha256(
                    {"x": torch.ones(3, dtype=dtype), "e": torch.zeros(0)})
                self.assertEqual(len(digest), 64)


class _DigestContext(_FakeContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved = []

    def model_state_digest(self):
        return DIGEST

    def save_checkpoint(self, path, completed_iterations):
        self.saved.append((path, completed_iterations))
        return path


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


class RuntimeCheckpointStateTests(unittest.TestCase):
    def setUp(self):
        import checkpoint_io
        self._original = checkpoint_io.load_checkpoint_cpu
        checkpoint_io.load_checkpoint_cpu = lambda path: {
            "completed_iterations": 99}
        self.addCleanup(
            setattr, checkpoint_io, "load_checkpoint_cpu", self._original)

    def test_a_save_names_the_checkpoint_the_model_now_equals(self):
        session = _session(completed=5)
        session._context = _DigestContext()
        self.assertIsNone(session.status()["checkpoint_state"])
        command = SaveCheckpointCommand(
            session_generation=0, expected_iteration=5, path="/out/a.ckpt")
        session._run_checkpoint_save(command)
        self.assertEqual(command.result["model_state_sha256"], DIGEST)
        self.assertEqual(session.status()["checkpoint_state"], {
            "path": "/out/a.ckpt", "model_state_sha256": DIGEST,
            "completed_iterations": 5})

    def test_a_load_names_the_loaded_checkpoint(self):
        session = _session(completed=5)
        session._context = _DigestContext()
        preflight = PreflightCheckpointCommand(
            session_generation=0, epoch=1, path="/ckpt/a.ckpt")
        session._run_checkpoint_preflight(preflight)
        self.assertIsNone(preflight.error)
        apply = ApplyCheckpointCommand(
            session_generation=0, epoch=2, path="/ckpt/a.ckpt")
        session._run_checkpoint_apply(apply)
        self.assertIsNone(apply.error)
        self.assertEqual(apply.result["model_state_sha256"], DIGEST)
        self.assertEqual(session.status()["checkpoint_state"], {
            "path": "/ckpt/a.ckpt", "model_state_sha256": DIGEST,
            "completed_iterations": 99})

    def test_a_step_forgets_the_checkpoint(self):
        session = _session(completed=5)
        session._context = _DigestContext()
        session._checkpoint_state = {
            "path": "/ckpt/a.ckpt", "model_state_sha256": DIGEST,
            "completed_iterations": 5}
        session._pending = 2
        session._state = SessionState.Running
        session.iteration_completed(
            completed_iterations=6, total_loss=1.0, losses={},
            learning_rate=1e-4)
        self.assertIsNone(session.status()["checkpoint_state"])

    def test_a_context_without_a_digest_reports_nothing(self):
        session = _session(completed=5)
        session._context = _FakeContext()
        self.assertIsNone(session._record_checkpoint_state("/ckpt/a.ckpt", 5))
        self.assertIsNone(session.status()["checkpoint_state"])


class RegistryRetentionTests(unittest.TestCase):
    def _register(self, registry, root, generation):
        root.mkdir()
        (root / "manifest.json").write_text("{}")
        return registry.register_directory(
            "spiral-preview", "session", generation, root, "manifest.json",
            delete_root_on_prune=True)

    def test_retain_exempts_older_artifacts_from_fixed_count_pruning(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            registry = ArtifactRegistry()
            refs = [self._register(registry, base / f"g{n}", n)
                    for n in range(1, 5)]
            pinned = (base / "g1").resolve()
            registry.prune(
                "spiral-preview", "session", 2,
                retain=lambda artifact: Path(artifact.root) == pinned)
            self.assertTrue((base / "g1").exists())
            self.assertFalse((base / "g2").exists())
            self.assertIsNotNone(registry.manifest(refs[0]["id"]))
            self.assertEqual(
                registry.find("spiral-preview", base / "g1", "session"),
                refs[0])
            self.assertIsNone(registry.find("spiral-preview", base / "g2"))
            self.assertIsNone(
                registry.find("spiral-preview", base / "g1", "other-session"))


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

    def test_a_corrupt_index_file_starts_over(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            index = PublishedPreviewIndex(base)
            index.path.parent.mkdir(parents=True)
            index.path.write_text("not json")
            self.assertIsNone(index.lookup(DIGEST))
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

    def test_a_publication_is_indexed_by_its_model_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = self._state(output)
            published = self._publish(state, output, 1, DIGEST)
            self.assertEqual(state._preview.model_state_sha256, DIGEST)
            index = PublishedPreviewIndex(output)
            entry = index.lookup(DIGEST)
            self.assertEqual(entry["manifest_path"],
                             str(published.manifest_path))
            self.assertEqual(entry["source_fit_iteration"], 40)

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

    def test_the_same_checkpoint_state_is_handled_once_and_a_shown_surface_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = self._state(output)
            self._publish(state, output, 1, DIGEST)
            shown = state._preview.artifact
            checkpoint = output / "checkpoint_autosave.ckpt"
            checkpoint.write_bytes(b"PK\x03\x04checkpoint")
            status = {"checkpoint_state": {
                "path": str(checkpoint), "model_state_sha256": DIGEST,
                "completed_iterations": 40}}
            with mock.patch.object(
                    state, "_restore_published_preview",
                    wraps=state._restore_published_preview) as restore:
                state._status_changed(status)
                state._status_changed(status)
                state._status_changed({"checkpoint_state": None})
                self.assertTrue(_wait(
                    lambda: checkpoint.as_posix() in json.loads(
                        PublishedPreviewIndex(output).path.read_text())["pins"]))
            self.assertEqual(restore.call_count, 1)
            self.assertEqual(state._preview.artifact, shown)
            self.assertEqual(state._preview.model_state_sha256, DIGEST)

    def test_an_unknown_model_state_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = self._state(output)
            self._publish(state, output, 1, DIGEST)
            shown = state._preview.artifact
            checkpoint = output / "checkpoints" / "elsewhere.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"PK\x03\x04checkpoint")
            state._status_changed({"checkpoint_state": {
                "path": str(checkpoint), "model_state_sha256": "f" * 64,
                "completed_iterations": 1}})
            self.assertTrue(_wait(lambda: not any(
                thread.name == "spiral-preview-restore"
                for thread in threading.enumerate())))
            self.assertEqual(state._preview.artifact, shown)
            self.assertEqual(state._preview.model_state_sha256, DIGEST)

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


if __name__ == "__main__":
    unittest.main()
