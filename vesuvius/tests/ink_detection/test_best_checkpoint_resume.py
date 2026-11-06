"""PBS-only checkpoint lifecycle controls; no training loop or scan data.

Uses the actual initialization, nested writer and selection blocks from native
training with real CPU checkpoint files. It does not run the whole trainer.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import vesuvius.ink_detection.training.train as training


def _config(folder, metric="val_loss", weights_only=False):
    config = SimpleNamespace(out_dir=folder, best_checkpoint_metric=metric,
        ink=SimpleNamespace(checkpoint=SimpleNamespace(weights_only=weights_only)),
        ema=SimpleNamespace(save_in_checkpoint=False), wandb_run_id=None,
        save_every=1, save_iterations=None)
    config.to_mapping = lambda: {"best_checkpoint_metric": metric, "out_dir": str(folder)}
    return config


def _session(request, marker=9):
    """Use exactly the real entry's initialization and nested save function."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(marker)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    tree = ast.parse(inspect.getsource(training))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_training")
    initializers = [n for n in function.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "best_checkpoint_value" for t in n.targets)]
    assert len(initializers) == 1
    save = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint")
    selection = [n for n in ast.walk(function) if isinstance(n, ast.If)
                 and n.body and isinstance(n.body[0], ast.Assign)
                 and isinstance(n.body[0].targets[0], ast.Name)
                 and n.body[0].targets[0].id == "candidate"]
    assert len(selection) == 1
    namespace = dict(training.__dict__, request=request, config=request.config,
        model=model, optimizer=optimizer, scheduler=scheduler, optimizer_step=0,
        ema_model=None, wandb_module=None,
        accelerator=SimpleNamespace(is_main_process=True, get_state_dict=lambda model: model.state_dict()))
    exec(compile(ast.Module(body=[*initializers, save], type_ignores=[]), "native-checkpoint-lifecycle", "exec"), namespace)
    namespace["_selection_code"] = compile(ast.Module(body=selection, type_ignores=[]), "native-best-selection", "exec")
    return namespace


def _event(session, value, step=3):
    session.update(record={session["config"].best_checkpoint_metric: value}, step=step)
    exec(session["_selection_code"], session)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class TestBestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def request(self, *, metric="val_loss", checkpoint=None, weights_only=False, folder=None, source=None):
        folder = self.root if folder is None else folder
        folder.mkdir(parents=True, exist_ok=True)
        return training.TrainingRequest(_config(folder, metric, weights_only),
                                        source or folder / "ckpt_000003.pth", checkpoint)

    def paired_best(self, folder, value=0.2, metric="val_loss", marker=3):
        folder.mkdir(parents=True, exist_ok=True)
        name = f"best_{metric}.pth"
        torch.save({"model": {"weight": torch.tensor([[float(marker)]])}}, folder / name)
        (folder / "best_checkpoint.json").write_text(json.dumps(
            {"metric": metric, "value": value, "checkpoint": name, "step": 1}))
        return folder / name

    def state(self, value=0.2, metric="val_loss"):
        return {"best_checkpoint": {"metric": metric, "value": value}}

    def test_new_best_payload_carries_metric_and_score(self):
        session = _session(self.request())
        _event(session, 0.2)
        payload = _load(self.root / "best_val_loss.pth")
        self.assertIn("best_checkpoint", payload)
        self.assertEqual(payload["best_checkpoint"], {"metric": "val_loss", "value": 0.2})

    def test_regular_checkpoint_carries_previous_best(self):
        session = _session(self.request())
        _event(session, 0.2)
        session["save_checkpoint"](4)
        payload = _load(self.root / "ckpt_000005.pth")
        self.assertIn("best_checkpoint", payload)
        self.assertEqual(payload["best_checkpoint"]["value"], 0.2)

    def test_persisted_loss_restores_in_new_directory(self):
        request = self.request(checkpoint=self.state(), source=self.root / "old" / "ckpt.pth")
        self.assertEqual(_session(request)["best_checkpoint_value"], 0.2)

    def test_persisted_accuracy_restores_in_new_directory(self):
        request = self.request(metric="val_balanced_accuracy", checkpoint=self.state(0.8, "val_balanced_accuracy"),
                               source=self.root / "old" / "ckpt.pth")
        self.assertEqual(_session(request)["best_checkpoint_value"], 0.8)

    def test_worse_resumed_loss_preserves_best_file_bytes(self):
        best = self.paired_best(self.root)
        before = best.read_bytes()
        session = _session(self.request(checkpoint=self.state()))
        _event(session, 0.3)
        self.assertEqual(best.read_bytes(), before)
        self.assertEqual(session["best_checkpoint_value"], 0.2)

    def test_worse_resumed_accuracy_preserves_best_file_bytes(self):
        best = self.paired_best(self.root, 0.8, "val_balanced_accuracy")
        before = best.read_bytes()
        session = _session(self.request(metric="val_balanced_accuracy", checkpoint=self.state(0.8, "val_balanced_accuracy")))
        _event(session, 0.7)
        self.assertEqual(best.read_bytes(), before)

    def test_better_resumed_loss_updates_payload_and_sidecar(self):
        self.paired_best(self.root)
        session = _session(self.request(checkpoint=self.state()), marker=7)
        _event(session, 0.1)
        payload = _load(self.root / "best_val_loss.pth")
        self.assertEqual(payload["model"]["weight"].item(), 7)
        self.assertIn("best_checkpoint", payload)
        self.assertEqual(payload["best_checkpoint"]["value"], 0.1)
        self.assertEqual(json.loads((self.root / "best_checkpoint.json").read_text())["value"], 0.1)

    def test_legacy_same_directory_uses_paired_metadata(self):
        self.paired_best(self.root)
        session = _session(self.request(checkpoint={"config": {"best_checkpoint_metric": "val_loss"}}))
        self.assertEqual(session["best_checkpoint_value"], 0.2)

    def test_legacy_other_source_does_not_borrow_output_history(self):
        self.paired_best(self.root, 0.01)
        request = self.request(checkpoint={}, source=self.root / "other" / "ckpt.pth")
        self.assertIsNone(_session(request)["best_checkpoint_value"])

    def test_same_directory_rollback_preserves_better_existing_artifact(self):
        best = self.paired_best(self.root, 0.2)
        before = best.read_bytes()
        session = _session(self.request(checkpoint=self.state(0.4)))
        _event(session, 0.3)
        self.assertEqual(best.read_bytes(), before)
        self.assertEqual(session["best_checkpoint_value"], 0.2)

    def test_new_directory_uses_checkpoint_history_instead_of_output_sidecar(self):
        self.paired_best(self.root, 0.01)
        request = self.request(checkpoint=self.state(0.4), source=self.root / "other" / "ckpt.pth")
        self.assertEqual(_session(request)["best_checkpoint_value"], 0.4)

    def test_weights_only_restarts_tracking(self):
        self.paired_best(self.root)
        self.assertIsNone(_session(self.request(checkpoint=self.state(), weights_only=True))["best_checkpoint_value"])

    def test_changed_metric_restarts_tracking(self):
        self.assertIsNone(_session(self.request(metric="val_balanced_accuracy", checkpoint=self.state()))["best_checkpoint_value"])

    def test_changed_legacy_metric_restarts_tracking(self):
        self.paired_best(self.root)
        request = self.request(checkpoint={"config": {"best_checkpoint_metric": "val_balanced_accuracy"}})
        self.assertIsNone(_session(request)["best_checkpoint_value"])

    def test_disabled_tracking_keeps_old_payload_schema(self):
        session = _session(self.request(metric=None, checkpoint=self.state()))
        self.assertIsNone(session["best_checkpoint_value"])
        session["save_checkpoint"](0, force=True)
        self.assertEqual(list(_load(self.root / "ckpt_000001.pth")),
                         ["model", "optimizer", "lr_scheduler", "config", "step", "wandb_run_id"])

    def test_unknown_history_does_not_add_state(self):
        session = _session(self.request(checkpoint={}))
        session["save_checkpoint"](0, force=True)
        self.assertNotIn("best_checkpoint", _load(self.root / "ckpt_000001.pth"))

    def test_orphan_sidecar_is_ignored(self):
        (self.root / "best_checkpoint.json").write_text('{"metric":"val_loss","value":0.1,"checkpoint":"best_val_loss.pth"}')
        self.assertIsNone(_session(self.request(checkpoint={}))["best_checkpoint_value"])

    def test_wrong_sidecar_metric_is_ignored(self):
        self.paired_best(self.root)
        (self.root / "best_checkpoint.json").write_text('{"metric":"val_balanced_accuracy","value":0.9,"checkpoint":"best_val_loss.pth"}')
        self.assertIsNone(_session(self.request(checkpoint={}))["best_checkpoint_value"])

    def test_wrong_sidecar_filename_is_ignored(self):
        self.paired_best(self.root)
        (self.root / "best_checkpoint.json").write_text('{"metric":"val_loss","value":0.1,"checkpoint":"unrelated.pth"}')
        self.assertIsNone(_session(self.request(checkpoint={}))["best_checkpoint_value"])

    def test_nan_first_metric_does_not_poison_later_selection(self):
        session = _session(self.request())
        _event(session, float("nan"))
        self.assertIsNone(session["best_checkpoint_value"])
        self.assertFalse((self.root / "best_val_loss.pth").exists())
        _event(session, 0.2)
        self.assertEqual(session["best_checkpoint_value"], 0.2)

    def test_infinite_first_loss_is_not_saved_as_best(self):
        session = _session(self.request())
        _event(session, float("-inf"))
        self.assertIsNone(session["best_checkpoint_value"])
        self.assertFalse((self.root / "best_val_loss.pth").exists())

    def test_infinite_first_accuracy_is_not_saved_as_best(self):
        session = _session(self.request(metric="val_balanced_accuracy"))
        _event(session, float("inf"))
        self.assertIsNone(session["best_checkpoint_value"])

    def test_equal_finite_score_keeps_existing_best(self):
        best = self.paired_best(self.root)
        session = _session(self.request())
        session["best_checkpoint_value"] = 0.2
        before = best.read_bytes()
        _event(session, 0.2)
        self.assertEqual(best.read_bytes(), before)

    def test_nan_after_valid_score_keeps_existing_best(self):
        best = self.paired_best(self.root)
        session = _session(self.request())
        session["best_checkpoint_value"] = 0.2
        before = best.read_bytes()
        _event(session, float("nan"))
        self.assertEqual(best.read_bytes(), before)

    def test_nonfinite_checkpoint_history_rejects_before_writes(self):
        before = sorted(p.name for p in self.root.iterdir())
        with self.assertRaises(ValueError):
            _session(self.request(checkpoint=self.state(float("nan"))))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), before)

    def test_nonmapping_checkpoint_history_rejects(self):
        with self.assertRaises(ValueError):
            _session(self.request(checkpoint={"best_checkpoint": []}))

    def test_malformed_paired_sidecar_rejects_before_writes(self):
        best = self.paired_best(self.root)
        before = best.read_bytes()
        (self.root / "best_checkpoint.json").write_text("{")
        with self.assertRaises(ValueError):
            _session(self.request(checkpoint={}))
        self.assertEqual(best.read_bytes(), before)

    def test_save_load_resume_lifecycle_preserves_best_until_improvement(self):
        session = _session(self.request(), marker=3)
        _event(session, 0.2, 0)
        best = self.root / "best_val_loss.pth"
        before = hashlib.sha256(best.read_bytes()).hexdigest()
        with torch.no_grad():
            session["model"].weight.fill_(7)
        session["save_checkpoint"](2)
        checkpoint_path = self.root / "ckpt_000003.pth"
        checkpoint = _load(checkpoint_path)
        resumed = _session(self.request(checkpoint=checkpoint, source=checkpoint_path))
        training.restore_training_state(resumed["model"], resumed["optimizer"], resumed["scheduler"], checkpoint, checkpoint_path)
        self.assertEqual(resumed["model"].weight.item(), 7)
        _event(resumed, 0.3, 3)
        self.assertEqual(hashlib.sha256(best.read_bytes()).hexdigest(), before)
        _event(resumed, 0.1, 4)
        self.assertEqual(_load(best)["model"]["weight"].item(), 7)
        self.assertEqual(_load(best)["best_checkpoint"]["value"], 0.1)
