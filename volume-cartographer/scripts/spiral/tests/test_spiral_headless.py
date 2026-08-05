import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

import numpy as np
import torch

from fit_session import (PclInputSpec, PclRole, resolve_dataset_root,
                         resolve_logical_dbm, validate_checkpoint_container)
from spiral_runtime import DistributedInteractiveFitSession, InteractiveFitSession
from spiral_helpers import compute_winding_range_and_input_extents
from spiral_service import ServiceState
from tifxyz import save_combined_tifxyz


class DatasetResolverTests(unittest.TestCase):
    def test_truncated_torch_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "truncated.ckpt"
            checkpoint.write_bytes(b"PK\x03\x04" + bytes(128))
            with self.assertRaisesRegex(ValueError, "incomplete or corrupt"):
                validate_checkpoint_container(checkpoint)

    def test_legacy_torch_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "legacy.ckpt"
            torch.save({"value": 1}, checkpoint, _use_new_zipfile_serialization=False)
            with self.assertRaisesRegex(ValueError, "Legacy pickle checkpoints are not supported"):
                validate_checkpoint_container(checkpoint)

    def test_conventional_resolution_and_logical_dbm_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "umbilicus.json").write_text("{}")
            (root / "verified_patches").mkdir()
            (root / "unverified_patches").mkdir()
            (root / "fibers").mkdir()
            (root / "tracks").mkdir()
            (root / "tracks" / "only.dbm.db").write_bytes(b"")
            (root / "abs_winding.json").write_text("{}")
            (root / "relative_windings.json").write_text("{}")
            (root / "same_windings.json").write_text("{}")
            (root / "patch-overlap-pcls.json").write_text("{}")
            result = resolve_dataset_root(root)
            self.assertTrue(result.ok)
            self.assertEqual(result.resolved["tracks_dbm"], str(root / "tracks" / "only.dbm"))
            self.assertEqual(result.resolved["verified_patches"],
                             str(root / "verified_patches"))
            self.assertEqual(result.resolved["fibers"], str(root / "fibers"))
            self.assertNotIn("unverified_patches", result.resolved)
            self.assertEqual([item["role"] for item in result.pcl_inputs],
                             ["absolute", "relative", "same_winding"])
            self.assertEqual(resolve_logical_dbm(root / "tracks" / "only.dbm.db"),
                             str(root / "tracks" / "only.dbm"))

    def test_dbm_ambiguity_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "umbilicus.json").write_text("{}")
            (root / "verified_patches").mkdir()
            (root / "tracks").mkdir()
            for name in ("z.dbm.db", "a.dbm.db"):
                (root / "tracks" / name).write_bytes(b"")
            result = resolve_dataset_root(root)
            self.assertEqual(result.ambiguities["tracks_dbm"], [
                str(root / "tracks" / "a.dbm"), str(root / "tracks" / "z.dbm")])


class HandoffTests(unittest.TestCase):
    def test_combined_preview_is_connected_with_ordered_winding_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "generation-1"
            blocks = {winding: np.full((3, 2, 3), winding, dtype=np.float32)
                      for winding in range(10, 13)}
            save_combined_tifxyz(blocks, destination, "preview", 20, 9.6, "test")
            metadata = json.loads((destination / "preview" / "meta.json").read_text())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                metadata["winding_column_ranges"], [[0, 2], [2, 4], [4, 6]]
            )
            self.assertNotIn("components", metadata)
            self.assertEqual(metadata["component_winding_ids"], [10, 11, 12])
            from PIL import Image
            x = np.asarray(Image.open(destination / "preview" / "x.tif"))
            self.assertEqual(x.shape, (3, 6))
            self.assertTrue(np.all(x[:, 1] == 10))
            self.assertTrue(np.all(x[:, 2] == 11))

    def test_combined_preview_cleanup_publishes_one_authoritative_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "generation-1"
            block = np.full((18, 32, 3), -1.0, dtype=np.float32)
            block[1:15, 1:16] = [10.0, 20.0, 30.0]
            block[5:13, 23:30] = [40.0, 50.0, 60.0]
            save_combined_tifxyz(
                {10: block}, destination, "preview", 1, 9.6, "test",
                cleanup_erosion_cells=3)

            metadata = json.loads(
                (destination / "preview" / "meta.json").read_text())
            self.assertNotIn("components", metadata)
            self.assertEqual(metadata["lasagna_input_cleanup"], {
                "erosion_cells": 3,
                "component_connectivity": 4,
                "components_after_erosion": 2,
            })
            from PIL import Image
            coordinates = [
                np.asarray(Image.open(destination / "preview" / f"{axis}.tif"))
                for axis in "xyz"
            ]
            valid = np.isfinite(coordinates).all(axis=0) & ~np.all(
                np.stack(coordinates, axis=-1) == -1.0, axis=-1)
            self.assertEqual(int(valid.sum()), 72)
            self.assertTrue(valid[4:12, 4:13].all())
            self.assertFalse(valid[:, 20:].any())


class PreviewRangeTests(unittest.TestCase):
    def test_many_short_tracks_are_transformed_in_point_batches(self):
        class CountingIdentity:
            def __init__(self):
                self.calls = 0

            def __call__(self, value):
                self.calls += 1
                return value

        transform = CountingIdentity()
        tracks = [np.array([[50, 0, x]], dtype=np.float32) for x in range(70_000)]
        winding_range, patch_extents, pcl_extents = compute_winding_range_and_input_extents(
            transform,
            torch.tensor(10.0),
            [],
            [],
            {"output_first_winding": 10, "output_winding_margin": 4},
            0,
            100,
            lambda *_: None,
            authoritative_zyx_lines=tracks,
        )

        self.assertEqual(transform.calls, 2)
        self.assertEqual(winding_range, (10, 7005))
        self.assertEqual(patch_extents, [])
        self.assertEqual(pcl_extents, [])


class ProtocolTests(unittest.TestCase):
    def test_distributed_session_waits_for_every_rank_before_ready(self):
        session = DistributedInteractiveFitSession.__new__(DistributedInteractiveFitSession)
        session._gpu_ids = (0, 1)
        session._condition = threading.Condition()
        session._events = queue.Queue()
        session._rank_statuses = {}
        session._acks = {}
        session._incorporation_callbacks = {}
        session._failed_error = None
        session._status = {"state": "Loading", "warnings": []}
        published = []
        session._status_callback = lambda status: published.append(status)
        listener = threading.Thread(target=session._listen)
        listener.start()
        ready = {
            "state": "Ready", "phase": "Ready", "warnings": [],
            "error": None, "current_iteration": 0, "target_iteration": 0,
        }
        session._events.put(("status", 0, ready))
        session._events.put(("status", 1, {
            **ready,
            "state": "Loading",
            "progress": {
                "operation": "loading",
                "stage_name": "Loading tracks",
                "detail": None,
                "step": 2,
                "total_steps": 10,
                "unit": "DB keys",
                "elapsed_seconds": 3.0,
                "eta_seconds": 12.0,
            },
        }))
        deadline = time.time() + 2
        while len(published) < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(published[-1]["state"], "Loading")
        self.assertEqual(published[-1]["phase"], "Loading tracks")
        self.assertIn(
            "GPU worker 2/2", published[-1]["progress"]["detail"])

        session._events.put(("status", 1, ready))
        deadline = time.time() + 2
        while published[-1]["state"] != "Ready" and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(published[-1]["state"], "Ready")
        session._events.put(None)
        listener.join(2)

    def test_interactive_run_can_continue_past_checkpoint_training_steps(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Ready"
        session._completed = 30_000
        session._pending = 0
        session._target = 30_000

        target = session.run(250)

        self.assertEqual(target, 30_250)
        self.assertEqual(session._pending, 250)
        self.assertEqual(session._target, 30_250)
        self.assertEqual(session._state, "Running")

    def test_interactive_run_extends_training_horizon_to_target_iteration(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Paused"
        session._completed = 30_000
        session._pending = 0
        session._target = 30_000
        session._configure_run = lambda *_: None
        session._idle_actions = []
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        target = session.run(250)

        self.assertEqual(target, 30_250)
        self.assertEqual(session._idle_actions[0][0], "configure")
        self.assertEqual(
            session._idle_actions[0][1]["optimizer_num_training_steps"],
            30_250,
        )
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_250)

    def test_interactive_run_preserves_horizon_when_target_is_within_it(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Ready"
        session._completed = 100
        session._pending = 0
        session._target = 100
        session._configure_run = lambda *_: None
        session._idle_actions = []
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        session.run(250)

        self.assertEqual(session._idle_actions, [])
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_000)

    def test_interactive_run_preserves_horizon_when_target_equals_it(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Ready"
        session._completed = 29_750
        session._pending = 0
        session._target = 29_750
        session._configure_run = lambda *_: None
        session._idle_actions = []
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        target = session.run(250)

        self.assertEqual(target, 30_000)
        self.assertEqual(session._idle_actions, [])
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_000)

    def test_interactive_run_extends_horizon_by_run_count_when_crossed(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Ready"
        session._completed = 29_751
        session._pending = 0
        session._target = 29_751
        session._configure_run = lambda *_: None
        session._idle_actions = []
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        target = session.run(250)

        self.assertEqual(target, 30_001)
        self.assertEqual(session._idle_actions[0][0], "configure")
        self.assertEqual(
            session._idle_actions[0][1]["optimizer_num_training_steps"],
            30_250,
        )
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_250)

    def test_run_queues_influence_config_with_only_pending_inputs(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Ready"
        session._completed = 10
        session._pending = 0
        session._target = 10
        session._incorporate_inputs = lambda *_: None
        session._idle_actions = []
        pending = [{"id": "new-patch"}]
        influence = {"influence_theta_frac": 0.25}

        session.run(20, pending_inputs=pending, influence_config=influence)

        action = session._idle_actions[0]
        self.assertEqual(action[0], "incorporate")
        self.assertEqual(action[1], pending)
        self.assertEqual(action[3], influence)

    def test_run_configuration_is_queued_before_input_incorporation(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Ready"
        session._completed = 10
        session._pending = 0
        session._target = 10
        session._incorporate_inputs = lambda *_: None
        session._configure_run = lambda *_: None
        session._idle_actions = []
        session.requested_config = {"loss_weight_patch_radius": 8.0}
        session._run_config = {"loss_weight_patch_radius": 8.0}

        session.run(
            20,
            pending_inputs=[{"id": "new-patch"}],
            run_config={"loss_weight_patch_radius": 4.0},
        )

        self.assertEqual([action[0] for action in session._idle_actions],
                         ["configure", "incorporate"])
        self.assertEqual(session._run_config["loss_weight_patch_radius"], 4.0)

    def test_run_configuration_applies_active_host_values_exactly(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._configure_run = lambda config: setattr(session, "applied", config)

        session._run_configuration({
            "sample_count_patches_per_step": 101,
            "loss_weight_patch_radius": 3.5,
            "loss_start_patch_dt": 123,
        })

        self.assertEqual(session.applied, {
            "sample_count_patches_per_step": 101,
            "loss_weight_patch_radius": 3.5,
            "loss_start_patch_dt": 123,
        })

    def test_run_finish_callback_precedes_autosave(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Running"
        session._completed = 9
        session._pending = 1
        session._target = 10
        session._stop_requested = False
        session._latest_metrics = {}
        session._output_path = "/tmp"
        session._status_callback = None
        calls = []
        session._finish_run = lambda: calls.append("finish")
        session._save_checkpoint = lambda *_: calls.append("save")
        session._publish_preview = lambda: calls.append("preview")

        session.iteration_completed(
            completed_iterations=10, total_loss=1.0, losses={}, learning_rate=1.e-3)

        self.assertEqual(calls, ["finish", "save", "preview"])
        self.assertEqual(session._state, "Paused")

    def test_preview_is_announced_before_exporting_state_can_pause(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = InteractiveFitSession.__new__(InteractiveFitSession)
            session._condition = threading.Condition()
            session._preview_generation = 0
            session._preview_session_id = "test-session"
            session.paths = type(
                "Paths", (), {"output_directory": temporary})()
            states = []
            session._state = "ExportingPreview"
            session._export_preview = lambda destination, surface_id: {
                "manifest_path": str(Path(destination) / "manifest.json"),
            }
            session._publish_status = lambda: states.append((
                session._state,
                session._preview_generation,
                session._preview_manifest,
            ))

            session._publish_preview()

            self.assertEqual(states, [(
                "ExportingPreview",
                1,
                str(Path(temporary) / ".spiral-preview" / "test-session"
                    / "generation-1" / "manifest.json"),
            )])

    def test_secondary_gpu_rank_pauses_without_publishing_outputs(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Running"
        session._completed = 9
        session._pending = 1
        session._target = 10
        session._stop_requested = False
        session._latest_metrics = {}
        session._status_callback = None
        session.publishes_outputs = False
        calls = []
        session._finish_run = lambda: calls.append("finish")
        session._save_checkpoint = lambda *_: calls.append("save")
        session._publish_preview = lambda: calls.append("preview")

        session.iteration_completed(
            completed_iterations=10, total_loss=1.0, losses={}, learning_rate=1.e-3)

        self.assertEqual(calls, ["finish"])
        self.assertEqual(session._state, "Paused")

    def test_mutating_command_is_deduplicated(self):
        service = ServiceState()
        calls = []
        first = service.deduplicated("same-command", lambda: calls.append(1) or {"accepted": True})
        second = service.deduplicated("same-command", lambda: calls.append(2) or {"accepted": True})
        self.assertEqual(calls, [1])
        self.assertEqual(first, second)

    def test_concurrent_duplicate_waits_for_one_execution(self):
        service = ServiceState()
        entered = threading.Event()
        release = threading.Event()
        calls = []
        results = []

        def operation():
            calls.append(1)
            entered.set()
            release.wait(2)
            return {"accepted": True}

        first = threading.Thread(target=lambda: results.append(
            service.deduplicated("concurrent-command", operation)))
        second = threading.Thread(target=lambda: results.append(
            service.deduplicated("concurrent-command", operation)))
        first.start()
        self.assertTrue(entered.wait(1))
        second.start()
        time.sleep(0.02)
        release.set()
        first.join(2)
        second.join(2)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(calls, [1])
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
