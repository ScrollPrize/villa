"""Prepared CPU save/load controls; run on PBS only, never local training.

The revised source retains the previously tested0033 EMA seed behavior. The
future runtime also composes the separately validated0030 AMP guard in both arms.
There are no forward/backward passes, real optimizer training or scan inputs.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import vesuvius.ink_detection.training.train as training
from vesuvius.ink_detection.models.checkpoint import restore_training_state


def _model(value):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def _state(value=-1, lr=0.9):
    model = _model(value)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    return model, optimizer, scheduler


def _payload(*, ema=False, save_ema=False, count=12):
    model, optimizer, scheduler = _state(3, 0.1)
    config = SimpleNamespace(ema=SimpleNamespace(enabled=ema, save_in_checkpoint=save_ema),
        wandb_run_id=None, best_checkpoint_metric=None)
    config.to_mapping = lambda: {"ema": {"enabled": ema, "save_in_checkpoint": save_ema}}
    return training.build_training_checkpoint_payload(
        accelerator=SimpleNamespace(get_state_dict=lambda item: item.state_dict()),
        model=model, optimizer=optimizer, scheduler=scheduler, config=config,
        step=99, optimizer_step=count, ema_model=_model(5) if ema else None,
        wandb_module=None)


def _manual_payload():
    payload = _payload()
    payload.pop("optimizer_step", None)
    return payload


def _roundtrip(payload, path):
    torch.save(payload, path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _update_block():
    source = ast.parse(inspect.getsource(training))
    matches = [node for node in ast.walk(source) if isinstance(node, ast.If)
        and any(isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "optimizer_step" for stmt in node.body)]
    assert len(matches) == 1
    return compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])),
                   training.__file__, "exec")


def _advance(model, ema_model, counter, synchronized):
    calls = []
    namespace = {"torch": torch, "optimizer_step": counter, "unwrapped_model": model,
        "ema_model": ema_model, "scheduler": SimpleNamespace(step=lambda: calls.append("step")),
        "accelerator": SimpleNamespace(sync_gradients=synchronized, optimizer_step_was_skipped=False),
        "config": SimpleNamespace(ema=SimpleNamespace(start_step=14, update_every_steps=3, decay=0.5))}
    exec(_update_block(), namespace)
    return namespace["optimizer_step"], len(calls)


class ResumeCounterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def restore(self, payload, *, ema=True, weights_only=False):
        model, optimizer, scheduler = _state()
        ema_model = _model(-2) if ema else None
        result = restore_training_state(model, optimizer, scheduler, payload,
            self.root / "checkpoint.pth", load_weights_only=weights_only, ema_model=ema_model)
        return result, model, ema_model

    def assert_invalid_unchanged(self, payload):
        model, optimizer, scheduler = _state()
        before_lr = optimizer.param_groups[0]["lr"]
        before_scheduler = scheduler.state_dict()
        with self.assertRaises(ValueError):
            restore_training_state(model, optimizer, scheduler, payload, "synthetic.pth")
        self.assertEqual(model.weight.item(), -1)
        self.assertEqual(optimizer.param_groups[0]["lr"], before_lr)
        self.assertEqual(scheduler.state_dict(), before_scheduler)

    def test_payload_without_ema_persists_counter(self):
        payload = _payload()
        self.assertEqual(payload.get("optimizer_step"), 12)
        self.assertNotIn("ema_model", payload)
        self.assertNotIn("ema_optimizer_step", payload)

    def test_payload_ema_without_saved_weights_persists_counter(self):
        payload = _payload(ema=True, save_ema=False)
        self.assertEqual(payload.get("optimizer_step"), 12)
        self.assertNotIn("ema_model", payload)

    def test_payload_saved_ema_has_matching_counters(self):
        payload = _payload(ema=True, save_ema=True)
        self.assertEqual(payload.get("optimizer_step"), 12)
        self.assertEqual(payload["ema_optimizer_step"], 12)
        self.assertIn("ema_model", payload)

    def test_zero_counter_is_explicit(self):
        self.assertEqual(_payload(count=0).get("optimizer_step"), 0)

    def test_full_resume_counter_without_saved_ema(self):
        payload = _manual_payload()
        payload["optimizer_step"] = 12
        (iteration, counter), model, ema = self.restore(payload)
        self.assertEqual((iteration, counter), (100, 12))
        self.assertEqual(model.weight.item(), 3)
        self.assertEqual(ema.weight.item(), 3)

    def test_full_resume_counter_with_ema_disabled(self):
        payload = _manual_payload()
        payload["optimizer_step"] = 12
        result, _, ema = self.restore(payload, ema=False)
        self.assertEqual(result, (100, 12))
        self.assertIsNone(ema)

    def test_legacy_counter_without_saved_ema(self):
        payload = _manual_payload()
        payload["ema_optimizer_step"] = 12
        result, _, ema = self.restore(payload)
        self.assertEqual(result, (100, 12))
        self.assertEqual(ema.weight.item(), 3)

    def test_legacy_counter_with_ema_disabled(self):
        payload = _manual_payload()
        payload["ema_optimizer_step"] = 12
        self.assertEqual(self.restore(payload, ema=False)[0], (100, 12))

    def test_legacy_saved_ema_and_counter_control(self):
        payload = _manual_payload()
        payload.update(ema_model=_model(5).state_dict(), ema_optimizer_step=12)
        result, model, ema = self.restore(payload)
        self.assertEqual(result, (100, 12))
        self.assertEqual(model.weight.item(), 3)
        self.assertEqual(ema.weight.item(), 5)

    def test_missing_history_stays_zero_control(self):
        result, model, ema = self.restore(_manual_payload())
        self.assertEqual(result, (100, 0))
        self.assertEqual(model.weight.item(), 3)
        self.assertEqual(ema.weight.item(), 3)

    def test_weights_only_ignores_counter_history_control(self):
        payload = _manual_payload()
        payload.update(optimizer_step=-1, ema_optimizer_step="malformed", ema_model=_model(5).state_dict())
        result, model, ema = self.restore(payload, weights_only=True)
        self.assertEqual(result, (0, 0))
        self.assertEqual(model.weight.item(), 3)
        self.assertEqual(ema.weight.item(), 3)

    def test_matching_new_and_legacy_state_control(self):
        payload = _manual_payload()
        payload.update(optimizer_step=12, ema_optimizer_step=12, ema_model=_model(5).state_dict())
        self.assertEqual(self.restore(payload)[0], (100, 12))

    def test_conflicting_counters_reject_before_state_changes(self):
        payload = _manual_payload()
        payload.update(optimizer_step=12, ema_optimizer_step=11)
        self.assert_invalid_unchanged(payload)

    def test_negative_new_counter_rejects_before_state_changes(self):
        payload = _manual_payload()
        payload["optimizer_step"] = -1
        self.assert_invalid_unchanged(payload)

    def test_boolean_new_counter_rejects_before_state_changes(self):
        payload = _manual_payload()
        payload["optimizer_step"] = True
        self.assert_invalid_unchanged(payload)

    def test_float_new_counter_rejects_before_state_changes(self):
        payload = _manual_payload()
        payload["optimizer_step"] = 12.0
        self.assert_invalid_unchanged(payload)

    def test_null_new_counter_rejects_before_state_changes(self):
        payload = _manual_payload()
        payload["optimizer_step"] = None
        self.assert_invalid_unchanged(payload)

    def test_malformed_legacy_counter_rejects_before_state_changes(self):
        payload = _manual_payload()
        payload["ema_optimizer_step"] = "12"
        self.assert_invalid_unchanged(payload)

    def test_negative_save_counter_rejects(self):
        with self.assertRaises(ValueError):
            _payload(count=-1)

    def test_boolean_save_counter_rejects(self):
        with self.assertRaises(ValueError):
            _payload(count=True)

    def test_resumed_ema_cadence_matches_update_history_without_saved_weights(self):
        payload = _roundtrip(_payload(ema=True, save_ema=False), self.root / "first.pth")
        (_, counter), model, ema = self.restore(payload)
        events = []
        calls = 0
        for offset in range(4):
            previous = ema.weight.item()
            with torch.no_grad():
                model.weight.fill_(7 + offset)
            counter, advanced = _advance(model, ema, counter, True)
            calls += advanced
            if ema.weight.item() != previous:
                events.append(counter)
        self.assertEqual(counter, 16)
        self.assertEqual(calls, 4)
        self.assertEqual(events, [14])

    def test_roundtrip_retains_updates_distinct_from_microbatch_iterations(self):
        payload = _roundtrip(_payload(ema=True, save_ema=False), self.root / "first.pth")
        result, model, ema = self.restore(payload)
        self.assertEqual(result, (100, 12))
        counter, calls = _advance(model, ema, result[1], False)
        self.assertEqual((counter, calls), (12, 0))
        counter, calls = _advance(model, ema, counter, True)
        self.assertEqual((counter, calls), (13, 1))

    def test_repeated_roundtrip_keeps_count_after_unsynchronized_microbatch(self):
        payload = _roundtrip(_payload(ema=True, save_ema=False), self.root / "first.pth")
        (_, counter), model, ema = self.restore(payload)
        counter, calls = _advance(model, ema, counter, False)
        self.assertEqual(calls, 0)
        second = _payload(ema=True, save_ema=False, count=counter)
        second = _roundtrip(second, self.root / "second.pth")
        self.assertEqual(self.restore(second)[0], (100, 12))

    def test_upstream_canonical_payload_contract_control(self):
        # Run the real fixture and test function without importing pytest or the
        # unrelated model/forward tests. Each arm retains its own frozen schema.
        upstream = Path(training.__file__).resolve().parents[4] / "tests/ink_detection"
        namespace = {"torch": torch, "TrainingConfig": training.TrainingConfig,
            "resolve_training_mapping": training.resolve_training_mapping,
            "build_training_checkpoint_payload": training.build_training_checkpoint_payload}
        selections = [("test_model_foundation.py", {"_config_mapping"}),
            ("test_training_artifacts.py", {"_LocalAccelerator",
             "test_checkpoint_payload_uses_canonical_mapping_and_optional_ema_state"})]
        for filename, names in selections:
            source = ast.parse((upstream / filename).read_text())
            nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
            self.assertEqual(len(nodes), len(names))
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(upstream / filename), "exec"), namespace)
        namespace["test_checkpoint_payload_uses_canonical_mapping_and_optional_ema_state"]()


if __name__ == "__main__":
    unittest.main(verbosity=2)
