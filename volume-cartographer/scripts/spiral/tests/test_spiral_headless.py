import dataclasses
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from fit_session import (AUTOSAVE_CHECKPOINT_NAME, AUTOSAVE_METADATA_NAME,
                         AUTOSAVE_METADATA_SCHEMA, PclInputSpec, PclRole,
                         ScrollSpecError, SessionState, SpiralInputPaths,
                         load_scroll_spec, resolve_dataset_root,
                         resolve_logical_dbm, validate_checkpoint_container)
import spiral_runtime
from spiral_progress import NullProgressReporter
from spiral_runtime import (CommandBarrier, CommandBarrierViolation,
                            ConfigureCommand,
                            DistributedInteractiveFitSession,
                            FileStoreRendezvous, IncorporateCommand,
                            InteractiveFitSession, SaveCheckpointCommand,
                            collective_view)
from spiral_helpers import compute_winding_range_and_input_extents
from spiral_service import ServiceState
from tifxyz import save_combined_tifxyz


def _zip_checkpoint_bytes(payload=b"payload"):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.pkl", payload)
    return buffer.getvalue()


def write_scroll_spec(root, **extra):
    document = {
        "schema_version": 1,
        "name": "s1",
        "voxel_size_um": 9.6,
        "spiral_outward_sense": "CW",
        **extra,
    }
    (Path(root) / "spiral-scroll.json").write_text(json.dumps(document))


class ScrollSpecTests(unittest.TestCase):
    def test_missing_file_names_the_conventional_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ScrollSpecError, "spiral-scroll.json"):
                load_scroll_spec(temporary)

    def test_schema_version_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "spiral-scroll.json").write_text(json.dumps({
                "name": "s1", "voxel_size_um": 9.6,
                "spiral_outward_sense": "CW"}))
            with self.assertRaisesRegex(ScrollSpecError, "schema_version"):
                load_scroll_spec(temporary)

    def test_unknown_keys_are_rejected_by_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_scroll_spec(temporary, render_volume_scale=16)
            with self.assertRaisesRegex(
                    ScrollSpecError, r"unknown keys: \['render_volume_scale'\]"):
                load_scroll_spec(temporary)

    def test_unknown_path_override_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_scroll_spec(temporary, paths={"scroll_zarr": "volume.zarr"})
            with self.assertRaisesRegex(
                    ScrollSpecError, r"unknown path override keys: \['scroll_zarr'\]"):
                load_scroll_spec(temporary)

    def test_required_physical_facts_and_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "spiral-scroll.json").write_text(json.dumps({
                "schema_version": 1, "name": "s1"}))
            with self.assertRaisesRegex(
                    ScrollSpecError,
                    r"missing required keys: \['spiral_outward_sense', "
                    r"'voxel_size_um'\]"):
                load_scroll_spec(temporary)
            write_scroll_spec(temporary)
            spec = load_scroll_spec(temporary)
            self.assertEqual(spec.name, "s1")
            self.assertEqual(spec.spiral_outward_sense, "CW")
            self.assertEqual(spec.umbilicus_coordinate_scale, 1.0)
            self.assertEqual(spec.normal_zarr_group, "4")
            self.assertEqual(spec.surf_sdt_zarr_group, "1")
            self.assertEqual(spec.lasagna_scale, 4)
            self.assertEqual(spec.path_overrides, ())

    def test_relative_path_overrides_resolve_against_dataset_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_scroll_spec(
                temporary,
                paths={"umbilicus": "annotations/umbilicus.json",
                       "tracks_dbm": "/elsewhere/tracks.dbm"})
            spec = load_scroll_spec(temporary)
            self.assertEqual(
                spec.path_override("umbilicus"),
                str(Path(temporary).resolve() / "annotations" / "umbilicus.json"))
            self.assertEqual(spec.path_override("tracks_dbm"),
                             "/elsewhere/tracks.dbm")

    def test_dataset_resolution_requires_and_honors_the_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations").mkdir()
            (root / "annotations" / "umbilicus.json").write_text("{}")
            (root / "verified_patches").mkdir()
            missing = resolve_dataset_root(root)
            self.assertIn("scroll_spec", missing.missing_required)
            self.assertTrue(any("spiral-scroll.json" in warning
                                for warning in missing.warnings))
            write_scroll_spec(
                root, paths={"umbilicus": "annotations/umbilicus.json"})
            result = resolve_dataset_root(root)
            self.assertTrue(result.ok)
            self.assertEqual(result.scroll_spec["name"], "s1")
            self.assertEqual(result.resolved["umbilicus"],
                             str(root.resolve() / "annotations" / "umbilicus.json"))


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
            write_scroll_spec(root)
            (root / "umbilicus.json").write_text("{}")
            (root / "verified_patches").mkdir()
            (root / "unverified_patches").mkdir()
            (root / "fibers").mkdir()
            (root / "tracks").mkdir()
            (root / "tracks" / "only.dbm.db").write_bytes(b"")
            (root / "abs_winding.json").write_text("{}")
            (root / "relative_windings.json").write_text("{}")
            (root / "same_windings.json").write_text("{}")
            (root / "drawn_control_points.json").write_text("{}")
            result = resolve_dataset_root(root)
            self.assertTrue(result.ok)
            self.assertEqual(result.resolved["tracks_dbm"], str(root / "tracks" / "only.dbm"))
            self.assertEqual(result.resolved["verified_patches"],
                             str(root / "verified_patches"))
            self.assertEqual(result.resolved["fibers"], str(root / "fibers"))
            self.assertNotIn("unverified_patches", result.resolved)
            self.assertEqual([item["role"] for item in result.pcl_inputs],
                             ["absolute", "relative", "same_winding",
                              "drawn_control_points"])
            self.assertEqual(resolve_logical_dbm(root / "tracks" / "only.dbm.db"),
                             str(root / "tracks" / "only.dbm"))

    def test_dbm_ambiguity_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_scroll_spec(root)
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


class _FakeWorker:
    """A worker process stand-in for the parent watchdog and fail-stop paths."""

    def __init__(self, rank):
        self.rank = rank
        self.alive = True
        self.exitcode = None
        self.terminated = 0
        self.killed = 0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated += 1
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -15

    def kill(self):
        self.killed += 1
        self.alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        return None


def rank_status(state, epoch=0, config_revision=0, **extra):
    status = {
        "state": state, "phase": str(state), "warnings": [], "error": None,
        "current_iteration": 0, "target_iteration": 0,
        "command_epoch": epoch, "config_revision": config_revision,
    }
    status.update(extra)
    return status


class ProtocolTests(unittest.TestCase):
    def _proxy(self, world_size=2, published=None):
        """A coordinator with fake workers and no spawned processes."""
        session = DistributedInteractiveFitSession.__new__(
            DistributedInteractiveFitSession)
        session._init_coordinator_state(
            tuple(range(world_size)),
            (published.append if published is not None else None),
            None)
        session._events = queue.Queue()
        session._commands = [queue.Queue() for _ in range(world_size)]
        session._processes = [_FakeWorker(rank) for rank in range(world_size)]
        return session

    def _wait_for(self, predicate, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_distributed_session_waits_for_every_rank_before_ready(self):
        published = []
        session = self._proxy(published=published)
        listener = threading.Thread(target=session._listen)
        listener.start()
        ready = rank_status(SessionState.Idle, epoch=3)
        ready["phase"] = "Idle"
        session._events.put(("status", 0, ready))
        session._events.put(("status", 1, rank_status(
            SessionState.Loading, epoch=3, progress={
                "operation": "loading",
                "stage_name": "Loading tracks",
                "detail": None,
                "step": 2,
                "total_steps": 10,
                "unit": "DB keys",
                "elapsed_seconds": 3.0,
                "eta_seconds": 12.0,
            })))
        self.assertTrue(self._wait_for(lambda: len(published) >= 2))
        # Rank 0 being ready is not a collective fact, so the last state every
        # rank agreed on (Loading) stands and the phase names the laggard.
        self.assertEqual(published[-1]["state"], SessionState.Loading)
        self.assertEqual(published[-1]["phase"], "Loading tracks")
        self.assertIn(
            "GPU worker 2/2", published[-1]["progress"]["detail"])

        session._events.put(("status", 1, rank_status(
            SessionState.Idle, epoch=3)))
        self.assertTrue(self._wait_for(
            lambda: published[-1]["state"] == SessionState.Idle))
        self.assertEqual(session._collective_state, SessionState.Idle)
        session._events.put(None)
        listener.join(2)

    def test_collective_state_is_visible_only_when_every_rank_agrees(self):
        idle = rank_status(SessionState.Idle, epoch=4)
        running = rank_status(SessionState.Running, epoch=4)

        # One rank has not reported at all.
        view = collective_view({0: idle}, 2, SessionState.Loading)
        self.assertFalse(view.visible)
        self.assertEqual(view.state, SessionState.Loading)

        # Both reported, but they are in different states.
        view = collective_view({0: idle, 1: running}, 2, SessionState.Loading)
        self.assertFalse(view.visible)
        self.assertEqual(view.state, SessionState.Loading)
        self.assertEqual(view.laggard, 1)

        # Same state, different command epochs: still not a collective fact,
        # because the ranks are executing different commands.
        view = collective_view(
            {0: running, 1: rank_status(SessionState.Running, epoch=5)},
            2, SessionState.Idle)
        self.assertFalse(view.visible)
        self.assertEqual(view.state, SessionState.Idle)
        self.assertEqual(view.laggard, 1)

        # Same epoch and same state.
        view = collective_view({0: running, 1: running}, 2, SessionState.Idle)
        self.assertTrue(view.visible)
        self.assertEqual(view.state, SessionState.Running)

    def test_rank_zero_publication_is_named_a_coordinator_sub_operation(self):
        published = []
        session = self._proxy(published=published)
        session._collective_state = SessionState.Idle
        listener = threading.Thread(target=session._listen)
        listener.start()
        saving = rank_status(SessionState.Saving, epoch=6)
        saving["phase"] = "Autosaving checkpoint"
        saving["progress"] = {"operation": "saving_checkpoint",
                              "stage_name": "Autosaving checkpoint",
                              "detail": "checkpoint_autosave.ckpt"}
        session._events.put(("status", 1, rank_status(
            SessionState.Idle, epoch=6)))
        session._events.put(("status", 0, saving))
        self.assertTrue(self._wait_for(
            lambda: published and published[-1].get("coordinator_operation")))
        latest = published[-1]
        self.assertEqual(latest["state"], SessionState.Saving)
        self.assertEqual(latest["coordinator_operation"], "saving")
        self.assertIn("Coordinator saving (rank 0)", latest["phase"])
        self.assertIn("coordinator rank 1/2", latest["progress"]["detail"])
        # The agreed collective state is untouched by rank-0-only work.
        self.assertEqual(session._collective_state, SessionState.Idle)
        session._events.put(None)
        listener.join(2)

    def test_all_rank_commands_carry_a_monotonic_epoch(self):
        session = self._proxy()
        with session._condition:
            session._rank_statuses = {
                0: rank_status(SessionState.Idle, config_revision=7),
                1: rank_status(SessionState.Idle, config_revision=7),
            }
            first = session._issue_barrier("run")
            second = session._issue_barrier("stop")
        self.assertEqual((first.epoch, first.kind), (1, "run"))
        self.assertEqual((second.epoch, second.kind), (2, "stop"))
        # A run may only be admitted by a quiescent rank; stop is
        # asynchronous with respect to the step loop.
        self.assertEqual(first.pending, 0)
        self.assertIsNone(second.pending)
        self.assertEqual(first.config_revision, 7)

        # Without unanimous revisions there is nothing to assert.
        with session._condition:
            session._rank_statuses[1] = rank_status(
                SessionState.Idle, config_revision=8)
            third = session._issue_barrier("run")
        self.assertIsNone(third.config_revision)

    def test_checkpoint_save_is_a_coordinator_sub_operation_without_a_barrier(self):
        session = self._proxy()
        session._collective_state = SessionState.Idle
        session._status["state"] = SessionState.Idle
        thread = threading.Thread(
            target=lambda: session._call(
                "save_checkpoint", {"path": "/tmp/x.ckpt"}, ranks=(0,),
                timeout=5.0, collective=False))
        thread.start()
        barrier, command_id, name, _ = session._commands[0].get(timeout=5)
        self.assertIsNone(barrier)
        self.assertEqual(name, "save_checkpoint")
        self.assertEqual(session._command_epoch, 0)
        self.assertTrue(session._commands[1].empty())
        session._events.put(("ack", command_id, 0, True, "/tmp/x.ckpt"))
        listener = threading.Thread(target=session._listen)
        listener.start()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        session._events.put(None)
        listener.join(2)

    def test_worker_error_fails_the_session_and_terminates_siblings(self):
        published = []
        session = self._proxy(published=published)
        session._start_coordinator_threads()
        try:
            started = time.monotonic()
            session._events.put((
                "worker_error", 1, "RuntimeError: boom", "Traceback: boom"))
            self.assertTrue(self._wait_for(
                lambda: session.status()["state"] == SessionState.Error))
            self.assertTrue(self._wait_for(
                lambda: all(worker.terminated
                            for worker in session._processes)))
            elapsed = time.monotonic() - started
        finally:
            session._stop_watchdog.set()
            session._events.put(None)
        self.assertLess(elapsed, 60.0)
        self.assertIn("rank 1", session.status()["error"])
        self.assertIn("boom", session.status()["error"])
        self.assertTrue(all(worker.terminated
                            for worker in session._processes))
        self.assertTrue(published)

    def test_unexpected_worker_exit_fails_the_session_in_bounded_time(self):
        session = self._proxy()
        session._start_coordinator_threads()
        try:
            started = time.monotonic()
            session._processes[1].alive = False
            session._processes[1].exitcode = 7
            self.assertTrue(self._wait_for(
                lambda: session.status()["state"] == SessionState.Error))
            self.assertTrue(self._wait_for(
                lambda: session._processes[0].terminated == 1))
            elapsed = time.monotonic() - started
        finally:
            session._stop_watchdog.set()
            session._events.put(None)
        self.assertLess(elapsed, 60.0)
        error = session.status()["error"]
        self.assertIn("rank 1", error)
        self.assertIn("exit code 7", error)
        # The surviving sibling is taken down with it.
        self.assertEqual(session._processes[0].terminated, 1)

    def test_command_timeout_fails_the_session_and_aborts_the_workers(self):
        session = self._proxy()
        session._collective_state = SessionState.Idle
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as caught:
            session._call("run", {"count": 1}, timeout=0.05)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 60.0)
        self.assertIn("Timed out", str(caught.exception))
        self.assertIn("[0, 1]", str(caught.exception))
        self.assertEqual(session.status()["state"], SessionState.Error)
        self.assertTrue(all(worker.terminated
                            for worker in session._processes))
        # A failed session refuses further work with the original cause.
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            session._call("stop")

    def test_first_failure_cause_wins_and_workers_are_aborted_once(self):
        session = self._proxy()
        session._fail_session("first cause", rank=1)
        session._fail_session("second cause", rank=0)
        self.assertIn("first cause", session.status()["error"])
        self.assertTrue(all(worker.terminated == 1
                            for worker in session._processes))

    def test_rendezvous_owns_its_endpoint_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as root:
            rendezvous = FileStoreRendezvous(Path(root) / ".spiral-rendezvous")
            directory = Path(rendezvous.directory)
            self.assertTrue(directory.is_dir())
            self.assertEqual(
                Path(rendezvous.endpoint.store_path).parent, directory)
            # Two sessions never share an endpoint.
            other = FileStoreRendezvous(Path(root) / ".spiral-rendezvous")
            self.assertNotEqual(other.directory, rendezvous.directory)
            other.close()
            # Closing one leaves the other's endpoint owned.
            self.assertTrue(directory.is_dir())

            session = self._proxy()
            session._rendezvous = rendezvous
            session._close_rendezvous()
            self.assertFalse(directory.exists())
            self.assertIsNone(session._rendezvous)

    def test_run_barrier_is_validated_against_the_rank_state(self):
        session = self._idle_session(rank=1, world_size=2)
        session._config_revision = 4
        good = CommandBarrier(epoch=1, kind="run", config_revision=4,
                              pending=0)

        for barrier, expected in [
            (dataclasses.replace(good, epoch=2), "expected command epoch 1"),
            (dataclasses.replace(good, kind="stop"),
             "coordinator issued as stop"),
            (dataclasses.replace(good, config_revision=9),
             "configuration revision 4"),
            (dataclasses.replace(good, pending=3), "requires 3"),
        ]:
            with self.assertRaises(CommandBarrierViolation) as caught:
                session.run(5, barrier=barrier)
            self.assertIn("rank 1", str(caught.exception))
            self.assertIn(expected, str(caught.exception))
            # A refused barrier leaves the rank exactly where it was.
            self.assertEqual(session._state, SessionState.Idle)
            self.assertEqual(session._command_epoch, 0)

        session.run(5, barrier=good)
        self.assertEqual(session._command_epoch, 1)
        self.assertEqual(session._step_epoch, 1)
        self.assertEqual(session._state, SessionState.Running)

    def test_command_from_another_epoch_fail_stops_the_rank(self):
        session = self._idle_session(rank=1, world_size=2)
        session._commands.append(IncorporateCommand(
            session_generation=0, epoch=99, records=[]))
        with self.assertRaisesRegex(CommandBarrierViolation,
                                    "from epoch 99 while in epoch 0"):
            session.wait_for_iteration(0)

    def test_step_boundary_refuses_an_epoch_the_run_was_not_admitted_in(self):
        session = self._idle_session(rank=1, world_size=2)
        session.run(5, barrier=CommandBarrier(
            epoch=1, kind="run", config_revision=0, pending=0))
        # A step under the admitting epoch is allowed...
        session.wait_for_iteration(0)
        # ...and one under any other epoch is not.
        with session._condition:
            session._command_epoch = 2
        with self.assertRaisesRegex(CommandBarrierViolation,
                                    "admitted in epoch 1"):
            session.wait_for_iteration(0)
        with session._condition:
            session._command_epoch = 1
            session._config_revision = 3
        with self.assertRaisesRegex(CommandBarrierViolation,
                                    "against revision 0"):
            session.wait_for_iteration(0)

    def _idle_session(self, completed=0, rank=0, world_size=1):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._phase = "Idle"
        session._completed = completed
        session._target = completed
        session._pending = 0
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session._stop_requested = False
        session._shutdown = False
        session._run_start_completed = completed
        session.rank = rank
        session.world_size = world_size
        session._status_callback = None
        session._event_callback = None
        return session

    def test_illegal_lifecycle_transition_is_a_programming_error(self):
        session = self._idle_session()
        with session._condition:
            # An idle session may now start an explicit preview export, but a
            # preview export cannot start a run.
            session._transition_locked(SessionState.ExportingPreview)
            with self.assertRaisesRegex(RuntimeError,
                                        "ExportingPreview -> Running"):
                session._transition_locked(SessionState.Running)
            # Every state may still fail or shut down.
            session._transition_locked(SessionState.Error, "Error")
            with self.assertRaisesRegex(RuntimeError, "Error -> Running"):
                session._transition_locked(SessionState.Running)

    def test_idle_reports_one_phase_and_the_iteration_count(self):
        never_run = self._idle_session()
        paused = self._idle_session(completed=42)
        for session in (never_run, paused):
            with session._condition:
                session._transition_locked(
                    SessionState.Idle, spiral_runtime.IDLE_PHASE)
        self.assertEqual(never_run._state, SessionState.Idle)
        self.assertEqual(paused._state, SessionState.Idle)
        # Both idle sessions look the same to a client except for the work
        # they have done; the phase does not editorialise about it.
        self.assertEqual(never_run._phase, "Idle")
        self.assertEqual(paused._phase, "Idle")
        self.assertEqual(never_run.completed_iterations, 0)
        self.assertEqual(paused.completed_iterations, 42)

    def test_save_command_carries_identity_and_completes_with_a_result(self):
        session = self._idle_session(completed=7)
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        session._context = SimpleNamespace(
            save_checkpoint=lambda path, iteration: f"{path}#{iteration}")
        saved = threading.Thread(
            target=lambda: results.append(
                session.save_checkpoint("/tmp/c.ckpt", timeout=5.0)))
        results = []
        saved.start()
        deadline = time.time() + 5
        while not session._commands and time.time() < deadline:
            time.sleep(0.005)
        command = session._commands[0]
        self.assertIsInstance(command, SaveCheckpointCommand)
        self.assertTrue(command.command_id)
        self.assertEqual(command.session_generation, 0)
        self.assertEqual(command.expected_iteration, 7)
        session._commands.pop(0)
        session._run_checkpoint_save(command)
        saved.join(5)
        self.assertEqual(results, ["/tmp/c.ckpt#7"])
        self.assertEqual(command.result["path"], "/tmp/c.ckpt#7")
        self.assertFalse(command.cancelled)
        self.assertEqual(session._state, SessionState.Idle)

    def test_command_queued_against_a_closed_session_is_cancelled(self):
        session = self._idle_session(completed=3)
        command = SaveCheckpointCommand(
            session_generation=session.session_generation,
            expected_iteration=3, path="/tmp/c.ckpt")
        session.session_generation += 1
        stale = command.stale_reason(
            session_generation=session.session_generation,
            iteration=3, config_revision=0)
        self.assertIsNotNone(stale)
        command.cancel(stale)
        self.assertTrue(command.cancelled)
        self.assertIn("no longer current", command.error)

    def test_interactive_run_can_continue_past_checkpoint_training_steps(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 30_000
        session._pending = 0
        session._target = 30_000
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1

        target = session.run(250)

        self.assertEqual(target, 30_250)
        self.assertEqual(session._pending, 250)
        self.assertEqual(session._target, 30_250)
        self.assertEqual(session._state, SessionState.Running)

    def test_interactive_run_extends_training_horizon_to_target_iteration(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 30_000
        session._pending = 0
        session._target = 30_000
        session._context = object()
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        target = session.run(250)

        self.assertEqual(target, 30_250)
        self.assertIsInstance(session._commands[0], ConfigureCommand)
        self.assertEqual(
            session._commands[0].config["optimizer_num_training_steps"],
            30_250,
        )
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_250)

    def test_interactive_run_preserves_horizon_when_target_is_within_it(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 100
        session._pending = 0
        session._target = 100
        session._context = object()
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        session.run(250)

        self.assertEqual(session._commands, [])
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_000)

    def test_interactive_run_preserves_horizon_when_target_equals_it(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 29_750
        session._pending = 0
        session._target = 29_750
        session._context = object()
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        target = session.run(250)

        self.assertEqual(target, 30_000)
        self.assertEqual(session._commands, [])
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_000)

    def test_interactive_run_extends_horizon_by_run_count_when_crossed(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 29_751
        session._pending = 0
        session._target = 29_751
        session._context = object()
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)

        target = session.run(250)

        self.assertEqual(target, 30_001)
        self.assertIsInstance(session._commands[0], ConfigureCommand)
        self.assertEqual(
            session._commands[0].config["optimizer_num_training_steps"],
            30_250,
        )
        self.assertEqual(
            session._run_config["optimizer_num_training_steps"], 30_250)

    def test_run_queues_influence_config_with_only_pending_inputs(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 10
        session._pending = 0
        session._target = 10
        session._context = object()
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session.requested_config = {
            "optimizer_num_training_steps": 30_000,
        }
        session._run_config = dict(session.requested_config)
        pending = [{"id": "new-patch"}]
        influence = {"influence_theta_frac": 0.25}

        session.run(20, pending_inputs=pending, influence_config=influence)

        command = session._commands[0]
        self.assertIsInstance(command, IncorporateCommand)
        self.assertEqual(command.records, pending)
        self.assertEqual(command.influence_config, influence)
        self.assertTrue(command.command_id)
        self.assertEqual(command.session_generation, 0)
        self.assertEqual(command.expected_iteration, 10)

    def test_run_configuration_is_queued_before_input_incorporation(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._completed = 10
        session._pending = 0
        session._target = 10
        session._context = object()
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session.requested_config = {"loss_weight_patch_radius": 8.0}
        session._run_config = {"loss_weight_patch_radius": 8.0}

        session.run(
            20,
            pending_inputs=[{"id": "new-patch"}],
            run_config={"loss_weight_patch_radius": 4.0},
        )

        self.assertEqual([command.kind for command in session._commands],
                         ["configure", "incorporate"])
        self.assertEqual(session._run_config["loss_weight_patch_radius"], 4.0)

    def test_incorporation_warnings_reach_the_session_status(self):
        # The context takes the inputs but reports what it could not honour;
        # those warnings ride the status the panel already displays.
        session = self._idle_session(completed=10)
        session._warnings = []
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        session._context = SimpleNamespace(
            incorporate_interactive_inputs=lambda *args, **kwargs: [
                "2 cross-fiber link(s) on 1 added fiber(s) are not used by this session"])
        command = IncorporateCommand(
            records=[{"id": "fiber-a", "kind": "fiber"}],
            influence_config=None,
            mark_incorporated=None)

        session._run_incorporation(command)

        self.assertEqual(session._warnings, [
            "2 cross-fiber link(s) on 1 added fiber(s) are not used by this session"])

    def test_incorporation_without_warnings_leaves_the_status_clean(self):
        session = self._idle_session(completed=10)
        session._warnings = []
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        # A context predating the warning return value must not break the path.
        session._context = SimpleNamespace(
            incorporate_interactive_inputs=lambda *args, **kwargs: None)
        command = IncorporateCommand(
            records=[{"id": "patch-a", "kind": "patch"}],
            influence_config=None,
            mark_incorporated=None)

        session._run_incorporation(command)

        self.assertEqual(session._warnings, [])

    def test_run_configuration_applies_active_host_values_exactly(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._completed = 7
        applied = {}

        def apply_config(config, path_changes=None, *, current_iteration):
            applied["config"] = config
            applied["path_changes"] = path_changes
            applied["current_iteration"] = current_iteration

        session._context = SimpleNamespace(apply_config=apply_config)

        session._state = SessionState.Idle
        session._commands = []
        session._config_revision = 0
        session._applied_config = None
        command = ConfigureCommand(config={
            "sample_count_patches_per_step": 101,
            "loss_weight_patch_radius": 3.5,
            "loss_start_patch_dt": 123,
        })
        session._run_configuration(command)

        self.assertEqual(applied["config"], {
            "sample_count_patches_per_step": 101,
            "loss_weight_patch_radius": 3.5,
            "loss_start_patch_dt": 123,
        })
        self.assertEqual(applied["path_changes"], {})
        self.assertEqual(applied["current_iteration"], 7)
        self.assertTrue(command.done.is_set())
        self.assertIsNone(command.error)
        self.assertEqual(command.result["config_revision"], 1)

    def _paused_session(self, calls, output_path, autosave_on_pause=True):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Running
        session._completed = 9
        session._pending = 1
        session._target = 10
        session._stop_requested = False
        session._latest_metrics = {}
        session._output_path = str(output_path)
        session._status_callback = None
        session._autosave_on_pause = autosave_on_pause
        session.paths = SpiralInputPaths.from_mapping({
            "dataset_root": "/datasets/scroll1",
            "output_directory": str(output_path),
        })

        def save_checkpoint(path, *_):
            calls.append("save")
            Path(path).write_bytes(_zip_checkpoint_bytes())

        session._context = SimpleNamespace(
            clear_interactive_influence=lambda: calls.append("finish"),
            save_checkpoint=save_checkpoint)
        session._publish_preview = lambda: calls.append("preview")
        return session

    def test_run_finish_callback_precedes_autosave(self):
        calls = []
        with tempfile.TemporaryDirectory() as output:
            session = self._paused_session(calls, output)

            session.iteration_completed(
                completed_iterations=10, total_loss=1.0, losses={},
                learning_rate=1.e-3)

            # Pausing writes the autosave and nothing else: a preview is an
            # explicit request now, not a side effect of stopping.
            self.assertEqual(calls, ["finish", "save"])
            self.assertEqual(session._state, SessionState.Idle)

            # The autosave names itself, so an always-loaded service can
            # select it at startup without guessing from filenames.
            metadata = json.loads(
                (Path(output) / AUTOSAVE_METADATA_NAME).read_text())
            self.assertEqual(metadata["schema"], AUTOSAVE_METADATA_SCHEMA)
            self.assertEqual(metadata["session_namespace"], str(output))
            self.assertEqual(metadata["dataset_root"], "/datasets/scroll1")
            self.assertEqual(metadata["completed_iterations"], 10)
            self.assertEqual(metadata["checkpoint"], AUTOSAVE_CHECKPOINT_NAME)

    def test_a_run_can_ask_for_no_autosave_on_pause(self):
        calls = []
        with tempfile.TemporaryDirectory() as output:
            session = self._paused_session(calls, output,
                                           autosave_on_pause=False)

            session.iteration_completed(
                completed_iterations=10, total_loss=1.0, losses={},
                learning_rate=1.e-3)

            self.assertEqual(calls, ["finish"])
            self.assertEqual(session._state, SessionState.Idle)
            # No autosave means no metadata claiming one exists.
            self.assertFalse((Path(output) / AUTOSAVE_METADATA_NAME).exists())

    def test_the_autosave_flag_is_decided_when_a_run_is_admitted(self):
        session = self._idle_session(completed=4)
        session.requested_config = {"optimizer_num_training_steps": 30_000}
        session._progress_reporter = lambda: NullProgressReporter()

        session.run(2, autosave_on_pause=False)
        self.assertFalse(session._autosave_on_pause)
        with session._condition:
            session._state = SessionState.Idle
            session._pending = 0
        session.run(2)
        self.assertTrue(session._autosave_on_pause)

    def test_preview_is_announced_before_exporting_state_can_pause(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = InteractiveFitSession.__new__(InteractiveFitSession)
            session._condition = threading.Condition()
            session._preview_generation = 0
            session._preview_session_id = "test-session"
            session.paths = type(
                "Paths", (), {"output_directory": temporary})()
            states = []
            session._state = SessionState.ExportingPreview
            session._context = SimpleNamespace(
                export_preview=lambda destination, surface_id: {
                    "manifest_path": str(Path(destination) / "manifest.json"),
                })
            session._publish_status = lambda: states.append((
                session._state,
                session._preview_generation,
                session._preview_manifest,
            ))

            session._publish_preview()

            self.assertEqual(states, [(
                SessionState.ExportingPreview,
                1,
                str(Path(temporary) / ".spiral-preview" / "test-session"
                    / "generation-1" / "manifest.json"),
            )])

    def test_a_restored_checkpoint_session_does_not_export_a_preview(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Loading
        session._phase = "Loading"
        session._completed = 0
        session._target = 0
        session._status_callback = None
        session._event_callback = None
        session.publishes_outputs = True
        session.paths = SimpleNamespace(checkpoint="/ckpt/a.ckpt",
                                        output_directory="/tmp")
        calls = []
        session._publish_preview = lambda: calls.append("preview")
        session._progress_reporter = lambda: NullProgressReporter()

        session._session_ready(SimpleNamespace(start_iteration=4200,
                                               out_path="/tmp/out"))

        # Inspecting a checkpoint costs a load, not a preview export.
        self.assertEqual(calls, [])
        self.assertEqual(session._state, SessionState.Idle)
        self.assertEqual(session._phase, "Idle")
        self.assertEqual(session._completed, 4200)

    def test_export_preview_is_a_requested_coordinator_operation(self):
        session = self._idle_session(completed=12)
        session.publishes_outputs = True
        session._preview_manifest = None
        session._preview_generation = 0
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        states = []

        def publish_preview():
            states.append(session._state)
            session._preview_manifest = "/preview/manifest.json"
            session._preview_generation = 3

        session._publish_preview = publish_preview
        results = []
        requester = threading.Thread(
            target=lambda: results.append(session.export_preview(timeout=5.0)))
        requester.start()
        deadline = time.time() + 5
        while not session._commands and time.time() < deadline:
            time.sleep(0.005)
        command = session._commands.pop(0)
        self.assertIsInstance(command, spiral_runtime.ExportPreviewCommand)
        # A coordinator sub-operation: no command epoch of its own.
        self.assertIsNone(command.epoch)
        self.assertEqual(command.expected_iteration, 12)

        session._run_export_preview(command)
        requester.join(5)

        self.assertEqual(states, [SessionState.ExportingPreview])
        self.assertEqual(session._state, SessionState.Idle)
        self.assertEqual(results, [{"preview_manifest_path": "/preview/manifest.json",
                                    "preview_generation": 3}])

    def test_export_preview_is_refused_outside_idle(self):
        session = self._idle_session()
        with session._condition:
            session._state = SessionState.Running
        with self.assertRaisesRegex(RuntimeError, "not allowed in Running"):
            session.export_preview(timeout=0.1)
        self.assertEqual(session._commands, [])

    def test_secondary_gpu_rank_pauses_without_publishing_outputs(self):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Running
        session._completed = 9
        session._pending = 1
        session._target = 10
        session._stop_requested = False
        session._latest_metrics = {}
        session._status_callback = None
        session.publishes_outputs = False
        calls = []
        session._context = SimpleNamespace(
            clear_interactive_influence=lambda: calls.append("finish"),
            save_checkpoint=lambda *_: calls.append("save"))
        session._publish_preview = lambda: calls.append("preview")

        session.iteration_completed(
            completed_iterations=10, total_loss=1.0, losses={}, learning_rate=1.e-3)

        self.assertEqual(calls, ["finish"])
        self.assertEqual(session._state, SessionState.Idle)

    def test_mutating_command_is_deduplicated(self):
        service = ServiceState()
        calls = []
        first = service.replay_command(
            "session_run", "same-command",
            lambda: calls.append(1) or {"accepted": True})
        second = service.replay_command(
            "session_run", "same-command",
            lambda: calls.append(2) or {"accepted": True})
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
            service.replay_command("session_run", "concurrent-command",
                                   operation)))
        second = threading.Thread(target=lambda: results.append(
            service.replay_command("session_run", "concurrent-command",
                                   operation)))
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
