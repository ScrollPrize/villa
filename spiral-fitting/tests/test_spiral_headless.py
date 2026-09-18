import pytest
from runtime_fixtures import RuntimeFixture
import dataclasses
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from fit_session import (
    AUTOSAVE_CHECKPOINT_NAME,
    AUTOSAVE_METADATA_NAME,
    AUTOSAVE_METADATA_SCHEMA,
    ScrollSpecError,
    SessionState,
    load_scroll_spec,
    resolve_dataset_root,
    resolve_logical_dbm,
    validate_checkpoint_container,
)
import spiral_runtime
from spiral_progress import NullProgressReporter
from spiral_runtime import (
    CommandBarrier,
    CommandBarrierViolation,
    ConfigureCommand,
    DtLossScheduleCommand,
    InteractiveFitSession,
    collective_view,
)
from spiral_helpers import compute_winding_range_and_input_extents
from spiral_service import ServiceState
from tifxyz import save_combined_tifxyz


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
            self.assertIsNone(spec.base_shape_zyx)
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


class PreviewRangeTests(unittest.TestCase):
    def test_a_point_budget_bounds_the_transformed_track_points(self):
        class CountingIdentity:
            def __init__(self):
                self.points = 0

            def __call__(self, value):
                self.points += int(value.shape[0])
                return value

        # One long track, so the budget has to thin within it. Points lie a
        # voxel apart and dr_per_winding is 2000, the realistic ratio: a
        # 200-point stride then costs a tenth of a winding.
        dr_per_winding = torch.tensor(2000.0)
        tracks = [torch.tensor(
            [[50.0, 0.0, float(x)] for x in range(200_000)],
            dtype=torch.float32)]
        cfg = {"output_first_winding": 10, "output_winding_margin": 4}
        exact = CountingIdentity()
        exact_range, _, _ = compute_winding_range_and_input_extents(
            exact, dr_per_winding, [], [], cfg, 0, 100, lambda *_: None,
            authoritative_zyx_lines=tracks)
        budgeted = CountingIdentity()
        budgeted_range, _, _ = compute_winding_range_and_input_extents(
            budgeted, dr_per_winding, [], [], cfg, 0, 100,
            lambda *_: None, authoritative_zyx_lines=tracks,
            point_budget=1000)

        self.assertEqual(exact.points, 200_000)
        self.assertLessEqual(budgeted.points, 1000)
        # Thinning a track by a fixed stride moves the observed extreme by a
        # fraction of a winding, which the output margin already covers.
        self.assertEqual(exact_range[0], budgeted_range[0])
        self.assertLessEqual(exact_range[1] - budgeted_range[1], 1)
        self.assertLessEqual(budgeted_range[1], exact_range[1])


def rank_status(state, epoch=0, config_revision=0, **extra):
    status = {
        "state": state, "phase": str(state), "warnings": [], "error": None,
        "current_iteration": 0, "target_iteration": 0,
        "command_epoch": epoch, "config_revision": config_revision,
    }
    status.update(extra)
    return status


class ProtocolTests(RuntimeFixture):
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

    def test_every_run_queues_and_applies_dt_schedule_before_first_step(self):
        session = self._idle_session(completed=100)
        calls = []
        session._context = SimpleNamespace(
            run_dt_resume_iteration=175,
            configure_dt_loss_schedule=lambda start, count, schedule: calls.append(
                (start, count, schedule)))
        schedule = {"enabled": True, "last_fraction": 0.25}

        session.run(100, dt_loss_schedule=schedule)

        self.assertEqual([command.kind for command in session._commands],
                         ["dt_loss_schedule"])
        session.wait_for_iteration(100)
        self.assertEqual(calls, [(100, 100, schedule)])
        self.assertEqual(session._iteration_in_progress, 100)

    def test_failed_run_setup_cancels_schedule_and_clears_run_state(self):
        session = self._idle_session(completed=10)
        session._state = SessionState.Running
        session._pending = 20
        session._target = 30
        session._warnings = []
        session._run_config = {"loss_weight_patch_radius": 8.0}
        calls = []
        session._context = SimpleNamespace(
            apply_config=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("bad setup")),
            clear_interactive_run_state=lambda: calls.append("clear"))
        schedule = DtLossScheduleCommand(
            session_generation=0, run_start=10, requested_iterations=20,
            schedule={"enabled": True, "last_fraction": 0.25})
        session._commands = [schedule]
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        command = ConfigureCommand(
            config={"loss_weight_patch_radius": 4.0},
            previous_run_config={"loss_weight_patch_radius": 8.0})

        session._run_configuration(command)

        self.assertEqual(calls, ["clear"])
        self.assertTrue(schedule.cancelled)
        self.assertEqual(session._commands, [])
        self.assertEqual(session._state, SessionState.Idle)
        self.assertEqual(session._target, 10)

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

    def test_a_long_run_autosaves_every_thousand_iterations(self):
        calls = []
        with tempfile.TemporaryDirectory() as output:
            session = self._mid_run_session(calls, output, completed=3000)

            session.iteration_completed(
                completed_iterations=3000, total_loss=1.0, losses={},
                learning_rate=1.e-3)

            # The cadence save neither pauses the run nor clears its state.
            self.assertEqual(calls, ["save"])
            self.assertEqual(session._state, SessionState.Running)
            self.assertEqual(session._phase, "Optimizing")
            self.assertEqual(session._pending, 499)
            metadata = json.loads(
                (Path(output) / AUTOSAVE_METADATA_NAME).read_text())
            self.assertEqual(metadata["completed_iterations"], 3000)
            self.assertEqual(metadata["checkpoint"], AUTOSAVE_CHECKPOINT_NAME)

    def test_a_failed_cadence_autosave_keeps_the_previous_one_and_warns(self):
        calls = []
        with tempfile.TemporaryDirectory() as output:
            session = self._mid_run_session(calls, output, completed=3000)
            previous = Path(output) / AUTOSAVE_CHECKPOINT_NAME
            previous.write_bytes(b"previous autosave")

            def failing_save(path, *_):
                calls.append("save")
                raise OSError("disk full")
            session._context.save_checkpoint = failing_save

            session.iteration_completed(
                completed_iterations=3000, total_loss=1.0, losses={},
                learning_rate=1.e-3)

            self.assertEqual(calls, ["save"])
            self.assertEqual(session._state, SessionState.Running)
            self.assertEqual(previous.read_bytes(), b"previous autosave")
            self.assertEqual(len(session._warnings), 1)
            self.assertIn("disk full", session._warnings[0])

    def test_scheduled_preview_is_queued_at_cadence_and_final_boundaries(self):
        calls = []
        with tempfile.TemporaryDirectory() as output:
            session = self._paused_session(
                calls, output, autosave_on_pause=False)
            session._commands = []
            session.session_generation = 0
            session._preview_schedule = {
                "cadence_iterations": 10, "diagnostics": True}
            session._next_preview_iteration = 10
            session._automatic_previews_disabled = False
            session._preview_source_iteration = None

            session.iteration_completed(
                completed_iterations=10, total_loss=1.0, losses={},
                learning_rate=1.e-3)

            self.assertEqual(session._state, SessionState.Idle)
            self.assertEqual(len(session._commands), 1)
            command = session._commands[0]
            self.assertIsInstance(command, spiral_runtime.ExportPreviewCommand)
            self.assertTrue(command.automatic)
            self.assertTrue(command.diagnostics)
            self.assertEqual(command.expected_iteration, 10)
            self.assertEqual(session._next_preview_iteration, 20)

    def test_scheduled_preview_failure_disables_future_captures(self):
        session = self._idle_session(completed=10)
        session._state = SessionState.Running
        session._phase = "Optimizing"
        session._warnings = []
        session._automatic_previews_disabled = False
        session._preview_manifest = None
        session._preview_generation = 0
        session._preview_source_iteration = None
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        session.publishes_outputs = True
        session._publish_preview = lambda diagnostics=False: (_ for _ in ()).throw(
            RuntimeError("preview OOM"))
        command = spiral_runtime.ExportPreviewCommand(
            session_generation=0, expected_iteration=10,
            automatic=True)
        session._run_export_preview(command)
        self.assertTrue(session._automatic_previews_disabled)
        self.assertIn("preview OOM", "\n".join(session._warnings))

    def test_preview_capture_barriers_surround_rank_zero_export(self):
        session = self._idle_session(completed=10)
        session._state = SessionState.Running
        session._phase = "Optimizing"
        session._warnings = []
        session._preview_manifest = None
        session._preview_generation = 0
        session._preview_source_iteration = None
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        session.publishes_outputs = True
        events = []
        session._preview_snapshot_barrier = lambda: events.append("barrier")
        session._share_preview_capture_error = lambda error: (
            events.append(("share", error)), error)[1]
        session._publish_preview = lambda diagnostics=False: events.append("export")
        command = spiral_runtime.ExportPreviewCommand(
            session_generation=0, expected_iteration=10, automatic=True)

        session._run_export_preview(command)

        self.assertEqual(
            events, ["barrier", "export", ("share", None), "barrier"])
        self.assertTrue(command.done.is_set())
        self.assertIsNone(command.error)

    def test_export_preview_can_capture_a_running_iteration_boundary(self):
        session = self._idle_session()
        with session._condition:
            session._state = SessionState.Running
            session._phase = "Optimizing"
            session._pending = 2
            session._target = 2
        session._progress_reporter = lambda: NullProgressReporter()
        session._publish_status = lambda: None
        session.publishes_outputs = True
        session._preview_manifest = None
        session._preview_generation = 0
        session._preview_source_iteration = None
        session._publish_preview = lambda diagnostics=False: None
        results = []
        requester = threading.Thread(
            target=lambda: results.append(
                session.export_preview(timeout=5.0)))
        requester.start()
        deadline = time.time() + 5
        while not session._commands and time.time() < deadline:
            time.sleep(0.005)
        command = session._commands.pop(0)
        self.assertEqual(command.expected_iteration, 0)
        session._run_export_preview(command)
        requester.join(5)
        self.assertEqual(session._state, SessionState.Running)
        self.assertEqual(session._phase, "Optimizing")
        self.assertEqual(results, [{
            "preview_manifest_path": None,
            "preview_generation": 0,
        }])

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
            clear_interactive_run_state=lambda: calls.append("finish"),
            save_checkpoint=lambda *_: calls.append("save"))
        session._publish_preview = lambda: calls.append("preview")

        session.iteration_completed(
            completed_iterations=10, total_loss=1.0, losses={}, learning_rate=1.e-3)

        self.assertEqual(calls, ["finish"])
        self.assertEqual(session._state, SessionState.Idle)

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


@pytest.mark.parametrize("completed, expected_horizon", [
    (100, 30_000), (29_750, 30_000), (29_751, 30_250), (30_000, 30_250),
])
def test_run_preserves_or_extends_training_horizon(completed, expected_horizon):
    session = RuntimeFixture()._idle_session(completed=completed)
    assert session.run(250) == completed + 250
    assert session._run_config["optimizer_num_training_steps"] == expected_horizon
    configured = [command.config["optimizer_num_training_steps"]
                  for command in session._commands
                  if isinstance(command, ConfigureCommand)]
    assert configured == ([expected_horizon] if expected_horizon > 30_000 else [])
    assert any(isinstance(command, DtLossScheduleCommand) for command in session._commands)
