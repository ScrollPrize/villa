"""Strict two-phase in-session checkpoint loading.

Phase 1 (``FitContext.inspect_checkpoint``) is a pure CPU-side verdict: it
must catch every structural invariant class and must not touch the live fit.
Phase 2 (``FitContext.apply_checkpoint``) restores the durable iteration and
realigns the LR schedule to it. The runtime tests cover the coordination:
where the session goes during a load, that a refusal leaves it exactly as it
was, and that the verb is only valid in Idle.
"""

import copy
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fit_spiral
from config import Config, durable_config
from fit_session import SessionState
from spiral_progress import NullProgressReporter
import spiral_runtime
from spiral_runtime import (ApplyCheckpointCommand, DiscardCheckpointCommand,
                            InteractiveFitSession,
                            PreflightCheckpointCommand)


CONFIG = Config().as_dict()

MODEL_STATE = {
    "linear_logits": torch.zeros(4, 3),
    "flow_fields.0.flows.0": torch.zeros(2, 5, 5),
}

OPTIMISER_STATE = {
    "state": {},
    "param_groups": [{"params": [0, 1], "lr": 3e-5},
                     {"params": [2], "lr": 3e-5}],
}

SCHEDULER_STATE = {"gamma": 0.9, "base_lrs": [3e-5, 3e-5], "last_epoch": 7,
                   "_step_count": 8, "_last_lr": [3e-5, 3e-5]}


def _live_context():
    """A stand-in for the live fit the preflight compares against."""
    return SimpleNamespace(
        lasagna_scale=4,
        normal_zarr_group="4",
        spiral_outward_sense="CW",
        paths=SimpleNamespace(dataset_root="/data/scroll1"),
        phase_mode=True,
        sdt_volume={"fingerprint": {"sha256": "abc", "path": "/store/a",
                                    "complete": True}},
        config=dict(CONFIG),
        model_z_begin=1000,
        model_z_end=2000,
        spiral_and_transform=SimpleNamespace(
            state_dict=lambda: dict(MODEL_STATE)),
        optimiser=SimpleNamespace(
            state_dict=lambda: copy.deepcopy(OPTIMISER_STATE)),
        lr_scheduler=SimpleNamespace(
            state_dict=lambda: dict(SCHEDULER_STATE)),
    )


def _checkpoint(**overrides):
    payload = {
        "schema_version": 2,
        "completed_iterations": 4200,
        "spiral_and_transform": dict(MODEL_STATE),
        "optimiser": copy.deepcopy(OPTIMISER_STATE),
        "scheduler": dict(SCHEDULER_STATE),
        "cfg": durable_config(CONFIG),
        "lasagna_scale": 4,
        "lasagna_group": "4",
        "spiral_outward_sense": "CW",
        # Only the content-identity fields compare: the store may legitimately
        # have moved and grown since the checkpoint was written.
        "surf_sdt_fingerprint": {"sha256": "abc", "path": "/store/b",
                                 "complete": False},
        "z_begin": 1000,
        "z_end": 2000,
        "input_manifest": {"dataset_root": "/data/scroll1"},
    }
    payload.update(overrides)
    return payload


def _inspect(checkpoint, context=None, **kwargs):
    return fit_spiral.FitContext.inspect_checkpoint(
        context or _live_context(), checkpoint, source="/ckpt/a.ckpt",
        **kwargs)


class CheckpointPreflightTests(unittest.TestCase):
    def test_a_matching_checkpoint_is_accepted_with_its_durable_iteration(self):
        verdict = _inspect(_checkpoint())
        self.assertTrue(verdict.accepted, verdict.reasons)
        self.assertEqual(verdict.reasons, ())
        self.assertEqual(verdict.completed_iterations, 4200)

    def test_preflight_does_not_touch_the_live_fit(self):
        context = _live_context()
        context.spiral_and_transform.load_state_dict = (
            lambda *_: self.fail("preflight loaded model state"))
        context.optimiser.load_state_dict = (
            lambda *_: self.fail("preflight loaded optimiser state"))
        context.lr_scheduler.load_state_dict = (
            lambda *_: self.fail("preflight loaded scheduler state"))
        before = copy.deepcopy(context.config)
        self.assertFalse(_inspect(_checkpoint(z_begin=0)).accepted)
        self.assertTrue(_inspect(_checkpoint()).accepted)
        self.assertEqual(context.config, before)

    def test_schema_invariants(self):
        self.assertIn("not a state dictionary",
                      _inspect(["not", "a", "dict"]).message())
        legacy = _inspect(_checkpoint(schema_version=1))
        self.assertFalse(legacy.accepted)
        self.assertIn("schema version 1", legacy.message())
        # The startup/CLI restore has no live session to protect and still
        # accepts pre-v2 checkpoints through the same implementation.
        self.assertTrue(
            _inspect(_checkpoint(schema_version=1),
                     allow_legacy_schema=True).accepted)
        missing = _inspect(_checkpoint(optimiser=None))
        self.assertFalse(missing.accepted)
        self.assertIn("'optimiser'", missing.message())

    def test_scroll_and_dataset_identity_invariants(self):
        for override, expected in (
                ({"lasagna_scale": 2}, "lasagna_scale"),
                ({"lasagna_group": "8"}, "Lasagna group"),
                ({"spiral_outward_sense": "CCW"}, "outward sense"),
                ({"input_manifest": {"dataset_root": "/data/other"}},
                 "written against dataset"),
        ):
            with self.subTest(override=override):
                verdict = _inspect(_checkpoint(**override))
                self.assertFalse(verdict.accepted)
                self.assertIn(expected, verdict.message())

    def test_sdt_identity_invariant_applies_when_an_sdt_loss_is_enabled(self):
        stale = _checkpoint(surf_sdt_fingerprint={"sha256": "def"})
        verdict = _inspect(stale)
        self.assertFalse(verdict.accepted)
        self.assertIn("surf-SDT fingerprint", verdict.message())
        # With no SDT-driven loss active the store is not an input at all.
        context = _live_context()
        context.phase_mode = False
        self.assertTrue(_inspect(stale, context).accepted)

    def test_model_z_domain_must_match_exactly(self):
        for domain in ((900, 2000), (1000, 2100)):
            with self.subTest(domain=domain):
                verdict = _inspect(
                    _checkpoint(z_begin=domain[0], z_end=domain[1]))
                self.assertFalse(verdict.accepted)
                self.assertIn("model z-domain", verdict.message())
                self.assertIn("rebuild", verdict.message())
        undeclared = _checkpoint()
        del undeclared["z_begin"], undeclared["z_end"]
        self.assertFalse(_inspect(undeclared).accepted)

    def test_structural_configuration_invariants(self):
        unknown = _checkpoint(cfg={**durable_config(CONFIG), "who_am_i": 1})
        self.assertIn("does not match the current schema",
                      _inspect(unknown).message())
        incomplete = durable_config(CONFIG)
        del incomplete["optimizer_learning_rate"]
        self.assertIn("optimizer_learning_rate",
                      _inspect(_checkpoint(cfg=incomplete)).message())
        # z_begin/z_end joined the schema late and are session-owned anyway.
        carve_out = durable_config(CONFIG)
        carve_out.pop("z_begin", None)
        carve_out.pop("z_end", None)
        self.assertTrue(_inspect(_checkpoint(cfg=carve_out)).accepted)
        shaped = durable_config(CONFIG)
        shaped["model_flow_bounds_radius"] = (
            int(shaped["model_flow_bounds_radius"]) + 1)
        verdict = _inspect(_checkpoint(cfg=shaped))
        self.assertFalse(verdict.accepted)
        self.assertIn("model-shaping config mismatch", verdict.message())

    def test_model_key_and_tensor_geometry_invariants(self):
        renamed = {"linear_logits": torch.zeros(4, 3),
                   "flow_fields.0.flows.9": torch.zeros(2, 5, 5)}
        verdict = _inspect(_checkpoint(spiral_and_transform=renamed))
        self.assertFalse(verdict.accepted)
        self.assertIn("model keys differ", verdict.message())
        reshaped = dict(MODEL_STATE)
        reshaped["linear_logits"] = torch.zeros(4, 4)
        verdict = _inspect(_checkpoint(spiral_and_transform=reshaped))
        self.assertFalse(verdict.accepted)
        self.assertIn("tensor geometry", verdict.message())
        retyped = dict(MODEL_STATE)
        retyped["linear_logits"] = torch.zeros(4, 3, dtype=torch.float64)
        self.assertIn("dtype",
                      _inspect(_checkpoint(spiral_and_transform=retyped)).message())

    def test_optimiser_and_scheduler_compatibility_invariants(self):
        fewer = {"state": {}, "param_groups": [{"params": [0, 1, 2]}]}
        verdict = _inspect(_checkpoint(optimiser=fewer))
        self.assertFalse(verdict.accepted)
        self.assertIn("parameter groups", verdict.message())
        regrouped = copy.deepcopy(OPTIMISER_STATE)
        regrouped["param_groups"][0]["params"] = [0, 2]
        verdict = _inspect(_checkpoint(optimiser=regrouped))
        self.assertFalse(verdict.accepted)
        self.assertIn("do not cover the same parameters", verdict.message())
        lambda_schedule = {"base_lrs": [3e-5, 3e-5], "last_epoch": 7,
                           "_step_count": 8, "_last_lr": [3e-5, 3e-5],
                           "lr_lambdas": [None, None]}
        verdict = _inspect(_checkpoint(scheduler=lambda_schedule))
        self.assertFalse(verdict.accepted)
        self.assertIn("different kind of schedule", verdict.message())
        narrower = dict(SCHEDULER_STATE, base_lrs=[3e-5])
        self.assertIn("parameter groups",
                      _inspect(_checkpoint(scheduler=narrower)).message())

    def test_every_failing_invariant_is_reported_not_just_the_first(self):
        verdict = _inspect(_checkpoint(lasagna_scale=2, z_begin=0))
        self.assertFalse(verdict.accepted)
        self.assertEqual(len(verdict.reasons), 2, verdict.reasons)


class _StubContext(fit_spiral.FitContext):
    """A FitContext with a real optimiser/scheduler and nothing else."""

    def __init__(self):  # noqa: D107 - deliberately skips FitContext.__init__
        self.config = dict(CONFIG)
        self.model = torch.nn.Linear(3, 2)
        self.spiral_and_transform = self.model
        self.gap_expander_params = [self.model.bias]
        self.optimiser = torch.optim.AdamW(
            [{"params": [self.model.weight], "weight_decay": 0.0},
             {"params": [self.model.bias], "weight_decay": 0.0}],
            lr=self.config["optimizer_learning_rate"])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimiser, gamma=0.5)
        self.start_iteration = 0


class CheckpointApplyTests(unittest.TestCase):
    def test_apply_restores_the_durable_step_and_realigns_the_schedule(self):
        context = _StubContext()
        horizon = int(context.config["optimizer_num_training_steps"])
        payload = {
            "completed_iterations": 1234,
            "spiral_and_transform": context.model.state_dict(),
            "optimiser": context.optimiser.state_dict(),
            # A schedule saved under a much shorter horizon: the live
            # configuration is authoritative and the realignment is what
            # decides the resulting LR.
            "scheduler": {**context.lr_scheduler.state_dict(),
                          "last_epoch": 3, "_step_count": 4},
        }

        completed = context.apply_checkpoint(payload, realign_lr=True)

        self.assertEqual(completed, 1234)
        self.assertEqual(context.start_iteration, 1234)
        self.assertEqual(context.lr_scheduler.last_epoch, 1234)
        expected = fit_spiral.get_exponential_lr_at_step(
            context.config["optimizer_learning_rate"],
            context.config["optimizer_lr_final_factor"], 1234, horizon)
        self.assertAlmostEqual(
            context.optimiser.param_groups[0]["lr"], expected, places=12)
        self.assertNotAlmostEqual(
            context.optimiser.param_groups[0]["lr"],
            context.config["optimizer_learning_rate"], places=12)

    def test_a_checkpoint_without_an_iteration_falls_back_to_the_caller(self):
        context = _StubContext()
        payload = {
            "spiral_and_transform": context.model.state_dict(),
            "optimiser": context.optimiser.state_dict(),
            "scheduler": context.lr_scheduler.state_dict(),
        }
        self.assertEqual(
            context.apply_checkpoint(payload, fallback_iteration=17), 17)
        self.assertEqual(context.start_iteration, 17)


class _FakeContext:
    """The fitter-thread context the runtime load commands drive."""

    def __init__(self, accepted=True, apply_error=None):
        self.accepted = accepted
        self.apply_error = apply_error
        self.inspected = []
        self.applied = []

    def inspect_checkpoint(self, checkpoint, source=""):
        self.inspected.append(source)
        return fit_spiral.CheckpointVerdict(
            self.accepted, () if self.accepted else ("z-domain differs",),
            completed_iterations=99, source=source)

    def apply_checkpoint(self, checkpoint, realign_lr=False):
        self.applied.append(realign_lr)
        if self.apply_error is not None:
            raise self.apply_error
        return 99


def _idle_session(completed=5):
    session = InteractiveFitSession.__new__(InteractiveFitSession)
    session._condition = threading.Condition()
    session._state = SessionState.Idle
    session._phase = "Idle"
    session._completed = completed
    session._target = completed
    session._pending = 0
    session._commands = []
    session._pending_checkpoint = None
    session.session_generation = 0
    session._config_revision = 0
    session._command_epoch = 0
    session._step_epoch = 0
    session._step_config_revision = 0
    session._stop_requested = False
    session._shutdown = False
    session._run_start_completed = completed
    session._latest_metrics = {"total_loss": 1.0}
    session.input_manifest = {}
    session.rank = 0
    session.world_size = 1
    session._status_callback = None
    session._event_callback = None
    session._progress_reporter = lambda: NullProgressReporter()
    session._publish_status = lambda: None
    return session


class InSessionCheckpointLoadTests(unittest.TestCase):
    def setUp(self):
        self.loaded = []

        def load_checkpoint_cpu(path):
            self.loaded.append(path)
            return {"completed_iterations": 99}

        import checkpoint_io
        self._original = checkpoint_io.load_checkpoint_cpu
        checkpoint_io.load_checkpoint_cpu = load_checkpoint_cpu
        self.addCleanup(
            setattr, checkpoint_io, "load_checkpoint_cpu", self._original)

    def _preflight(self, session, path="/ckpt/a.ckpt"):
        command = PreflightCheckpointCommand(
            session_generation=0, epoch=1, path=path)
        session._run_checkpoint_preflight(command)
        return command

    def test_accepted_load_restores_the_iteration_and_bumps_the_revision(self):
        session = _idle_session(completed=5)
        session._context = _FakeContext()

        preflight = self._preflight(session)
        self.assertIsNone(preflight.error)
        self.assertTrue(preflight.result["accepted"])
        # The session is out of Idle for the whole two-phase load, so nothing
        # can be admitted against the model being replaced.
        self.assertEqual(session._state, SessionState.Loading)
        self.assertEqual(session._completed, 5)

        apply_command = ApplyCheckpointCommand(
            session_generation=0, epoch=2, path="/ckpt/a.ckpt")
        session._run_checkpoint_apply(apply_command)

        self.assertIsNone(apply_command.error)
        self.assertEqual(session._state, SessionState.Idle)
        self.assertEqual(session._phase, "Idle")
        self.assertEqual(session._completed, 99)
        self.assertEqual(session._target, 99)
        self.assertEqual(session._config_revision, 1)
        self.assertEqual(session.input_manifest["checkpoint"], "/ckpt/a.ckpt")
        # The realignment is requested by the session, not left to the CLI.
        self.assertEqual(session._context.applied, [True])
        self.assertEqual(apply_command.result["completed_iterations"], 99)

    def test_a_refused_preflight_leaves_the_session_untouched(self):
        session = _idle_session(completed=5)
        session._context = _FakeContext(accepted=False)

        command = self._preflight(session)

        self.assertIn("z-domain differs", command.error)
        self.assertEqual(session._state, SessionState.Idle)
        self.assertEqual(session._phase, "Idle")
        self.assertEqual(session._completed, 5)
        self.assertEqual(session._config_revision, 0)
        self.assertEqual(session._context.applied, [])
        self.assertIsNone(session._pending_checkpoint)
        self.assertNotIn("checkpoint", session.input_manifest)

    def test_an_unreadable_checkpoint_is_a_refusal_not_a_failure(self):
        import checkpoint_io
        checkpoint_io.load_checkpoint_cpu = lambda path: (_ for _ in ()).throw(
            OSError("no such file"))
        session = _idle_session()
        session._context = _FakeContext()

        command = self._preflight(session)

        self.assertIn("no such file", command.error)
        self.assertEqual(session._state, SessionState.Idle)

    def test_apply_without_a_preflight_refuses_without_touching_the_model(self):
        session = _idle_session()
        session._context = _FakeContext()
        with session._condition:
            session._transition_locked(SessionState.Loading, "Inspecting")
        command = ApplyCheckpointCommand(
            session_generation=0, epoch=2, path="/ckpt/b.ckpt")

        session._run_checkpoint_apply(command)

        self.assertIn("No inspected checkpoint is pending", command.error)
        self.assertEqual(session._context.applied, [])
        self.assertEqual(session._state, SessionState.Idle)

    def test_a_failure_while_applying_is_fatal_to_the_session(self):
        session = _idle_session()
        session._context = _FakeContext(apply_error=RuntimeError("size mismatch"))
        self._preflight(session)
        command = ApplyCheckpointCommand(
            session_generation=0, epoch=2, path="/ckpt/a.ckpt")

        with self.assertRaisesRegex(RuntimeError, "state are partial"):
            session._run_checkpoint_apply(command)

        # The waiter is released with the same diagnosis the fitter thread
        # unwinds with; the session does not return to Idle.
        self.assertIn("size mismatch", command.error)
        self.assertEqual(session._state, SessionState.Loading)

    def test_discard_releases_the_payload_and_returns_to_idle(self):
        session = _idle_session()
        session._context = _FakeContext()
        self._preflight(session)
        self.assertIsNotNone(session._pending_checkpoint)

        command = DiscardCheckpointCommand(session_generation=0, epoch=2)
        session._run_checkpoint_discard(command)

        self.assertIsNone(session._pending_checkpoint)
        self.assertEqual(session._state, SessionState.Idle)
        self.assertTrue(command.result["discarded"])

    def test_load_is_refused_outside_idle(self):
        for state in (SessionState.Running, SessionState.Saving,
                      SessionState.Loading, SessionState.Error):
            with self.subTest(state=state):
                session = _idle_session()
                with session._condition:
                    session._state = state
                with self.assertRaisesRegex(
                        RuntimeError, "not allowed while session state"):
                    session.preflight_checkpoint("/ckpt/a.ckpt", timeout=0.1)
                self.assertEqual(session._commands, [])

    def test_the_load_commands_are_all_rank_commands_on_the_epoch_barrier(self):
        session = _idle_session()
        good = spiral_runtime.CommandBarrier(
            epoch=1, kind="preflight_checkpoint", config_revision=0, pending=0)
        wrong_kind = spiral_runtime.CommandBarrier(
            epoch=1, kind="run", config_revision=0, pending=0)
        with self.assertRaises(spiral_runtime.CommandBarrierViolation):
            session.preflight_checkpoint("/ckpt/a.ckpt", timeout=0.1,
                                         barrier=wrong_kind)
        self.assertEqual(session._command_epoch, 0)
        self.assertEqual(session._commands, [])

        # A matching barrier queues the command in the coordinator's epoch.
        def queue_and_wait():
            try:
                session.preflight_checkpoint(
                    "/ckpt/a.ckpt", timeout=0.2, barrier=good)
            except TimeoutError:
                pass  # No fitter thread is draining the queue here.

        thread = threading.Thread(target=queue_and_wait, daemon=True)
        thread.start()
        thread.join(2.0)
        self.assertEqual(session._command_epoch, 1)
        self.assertEqual(len(session._commands), 1)
        self.assertEqual(session._commands[0].epoch, 1)

    def test_a_rank_with_iterations_pending_refuses_the_load_barrier(self):
        session = _idle_session()
        session._pending = 3
        with self.assertRaisesRegex(spiral_runtime.CommandBarrierViolation,
                                    "requires 0"):
            session.preflight_checkpoint(
                "/ckpt/a.ckpt", timeout=0.1,
                barrier=spiral_runtime.CommandBarrier(
                    epoch=1, kind="preflight_checkpoint", config_revision=0,
                    pending=0))


if __name__ == "__main__":
    unittest.main()
