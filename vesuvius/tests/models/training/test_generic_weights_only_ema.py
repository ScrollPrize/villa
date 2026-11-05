"""PBS-only real CPU checkpoint I/O and exact generic caller/EMA source methods.

No complete BaseTrainer construction, model forward, training, CUDA or DDP.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import inspect
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.models.training.save_checkpoint import save_checkpoint
from vesuvius.models.utilities.load_checkpoint import load_checkpoint


def source_harness():
    source = Path(inspect.getsourcefile(load_checkpoint)).resolve().parents[1] / 'training/train.py'
    cls = next(n for n in ast.parse(source.read_text(encoding='utf8')).body
               if isinstance(n, ast.ClassDef) and n.name == 'BaseTrainer')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    selected = ['_unwrap_model', '_create_ema_model', '_initialize_ema_model',
                '_update_ema_model', '_get_scheduler', '_get_additional_checkpoint_data']
    body = [deepcopy(methods[name]) for name in selected]
    init = methods['_initialize_training']
    branch = next(n for n in init.body if isinstance(n, ast.If)
                  and ast.unparse(n.test).startswith("hasattr(self.mgr, 'checkpoint_path')"))
    fn = ast.parse('def restore(self, model, optimizer, scheduler):\n'
                   '    start_epoch = 0\n    checkpoint_loaded = False\n'
                   '    loss_overrides = {}\n').body[0]
    fn.body.append(deepcopy(branch))
    fn.body.extend(ast.parse('return model, optimizer, scheduler, start_epoch, checkpoint_loaded').body)
    body.append(fn)
    harness = ast.ClassDef(name='Harness', bases=[], keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    ns = {'torch': torch, 'deepcopy': deepcopy, 'get_scheduler': get_scheduler,
          'load_checkpoint': load_checkpoint}
    exec(compile(module, str(source), 'exec'), ns)
    return ns['Harness']


class TinyState(torch.nn.Module):
    def __init__(self, weight, bias, seen):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([float(weight)]))
        self.bias = torch.nn.Parameter(torch.tensor([float(bias)]))
        self.register_buffer('seen', torch.tensor(seen, dtype=torch.int64))


def values(model):
    return {name: value.item() for name, value in model.state_dict().items()}


class GenericWeightsOnlyEmaTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'checkpoint.pth'
        self.Harness = source_harness()

    def trainer(self, *, weights_only, enabled=True):
        obj = self.Harness()
        obj.mgr = SimpleNamespace(checkpoint_path=self.path, load_weights_only=weights_only,
                                  scheduler='cosine', scheduler_kwargs={}, initial_lr=0.2,
                                  max_epoch=8, rebuild_from_checkpoint_config=False)
        obj.device = torch.device('cpu')
        obj.ema_enabled = enabled
        obj.ema_save_in_checkpoint = True
        obj.ema_model = None
        obj.ema_decay = 0.5
        obj.ema_start_step = 5
        obj.ema_update_every_steps = 3
        obj._ema_optimizer_step = 0
        obj._checkpoint_ema_state = None
        obj._checkpoint_ema_optimizer_step = None
        obj.loss_override_calls = []
        obj._apply_loss_overrides = obj.loss_override_calls.append
        return obj

    def write_checkpoint(self, *, ema=True, legacy=False, partial=False, raw=False):
        main = TinyState(12, 4, 5)
        opt = torch.optim.SGD(main.parameters(), lr=0.05, momentum=0.9)
        for param in main.parameters():
            opt.state[param]['momentum_buffer'] = torch.full_like(param, 7)
        sched = get_scheduler('cosine', opt, initial_lr=0.05, max_steps=8)
        producer = self.trainer(weights_only=False)
        producer.ema_model = TinyState(90, 81, 11) if ema else None
        producer._ema_optimizer_step = 17
        extra = producer._get_additional_checkpoint_data()
        if legacy:
            extra.pop('ema_optimizer_step', None)
        with redirect_stdout(io.StringIO()):
            payload = save_checkpoint(main, opt, sched, 3, self.path, additional_data=extra)
        if partial:
            payload['model'] = {'weight': payload['model']['weight']}
            torch.save(payload, self.path)
        if raw:
            torch.save(main.state_dict(), self.path)

    def restore(self, *, weights_only, enabled=True):
        obj = self.trainer(weights_only=weights_only, enabled=enabled)
        target = TinyState(-2, -3, 0)
        opt = torch.optim.SGD(target.parameters(), lr=0.2, momentum=0.9)
        sched = get_scheduler('cosine', opt, initial_lr=0.2, max_steps=8)
        before = self.path.read_bytes()
        with redirect_stdout(io.StringIO()):
            model, opt, sched, epoch, loaded = obj.restore(target, opt, sched)
            obj._initialize_ema_model(model)
        self.assertTrue(loaded)
        self.assertEqual(obj.loss_override_calls, [{}])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIs(sched.optimizer, opt)
        if weights_only:
            self.assertEqual(epoch, 0)
            self.assertEqual(len(opt.state), 0)
            self.assertEqual(opt.param_groups[0]['lr'], 0.2)
        else:
            self.assertEqual(epoch, 4)
            self.assertEqual(len(opt.state), 2)
            self.assertEqual(opt.param_groups[0]['lr'], 0.05)
        return obj, model

    def assert_fresh(self, obj, model, expected):
        self.assertEqual(values(model), expected)
        self.assertEqual(values(obj.ema_model), expected)
        self.assertEqual(obj._ema_optimizer_step, 0)
        self.assertFalse(obj.ema_model.training)
        self.assertTrue(all(not p.requires_grad for p in obj.ema_model.parameters()))
        obj._update_ema_model(model)
        self.assertEqual(obj._ema_optimizer_step, 1)
        self.assertEqual(values(obj.ema_model), expected)

    def test_weights_only_ignores_ema_weights_and_counter(self):
        self.write_checkpoint()
        self.assert_fresh(*self.restore(weights_only=True), {'weight': 12., 'bias': 4., 'seen': 5})

    def test_weights_only_legacy_ema_without_counter_starts_fresh(self):
        self.write_checkpoint(legacy=True)
        self.assert_fresh(*self.restore(weights_only=True), {'weight': 12., 'bias': 4., 'seen': 5})

    def test_partial_weights_only_seeds_ema_from_loaded_and_retained_main(self):
        self.write_checkpoint(partial=True)
        self.assert_fresh(*self.restore(weights_only=True), {'weight': 12., 'bias': -3., 'seen': 0})

    def test_full_resume_restores_ema_and_optimizer_history_control(self):
        self.write_checkpoint()
        obj, model = self.restore(weights_only=False)
        self.assertEqual(values(model), {'weight': 12., 'bias': 4., 'seen': 5})
        self.assertEqual(values(obj.ema_model), {'weight': 90., 'bias': 81., 'seen': 11})
        self.assertEqual(obj._ema_optimizer_step, 17)
        obj._update_ema_model(model)
        self.assertEqual(obj._ema_optimizer_step, 18)
        self.assertEqual(values(obj.ema_model), {'weight': 90., 'bias': 81., 'seen': 11})

    def test_full_resume_without_ema_seeds_loaded_main_control(self):
        self.write_checkpoint(ema=False)
        self.assert_fresh(*self.restore(weights_only=False), {'weight': 12., 'bias': 4., 'seen': 5})

    def test_raw_weights_only_control(self):
        self.write_checkpoint(raw=True)
        self.assert_fresh(*self.restore(weights_only=True), {'weight': 12., 'bias': 4., 'seen': 5})

    def test_disabled_ema_weights_only_control(self):
        self.write_checkpoint()
        obj, model = self.restore(weights_only=True, enabled=False)
        self.assertEqual(values(model), {'weight': 12., 'bias': 4., 'seen': 5})
        self.assertIsNone(obj.ema_model)
        self.assertEqual(obj._ema_optimizer_step, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
