"""Prepared PBS-only checkpoint/epoch tests with supplied epoch completions.

Actual CPU save/load and source range/stop/finalization; no full trainer/training.
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
from vesuvius.models.training.save_checkpoint import save_checkpoint, save_final_checkpoint
from vesuvius.models.utilities.load_checkpoint import load_checkpoint


CASES = {
    'test_early_stop_final_epoch_matches_completed_range': {
        'input_epoch': -1, 'limit': 5, 'patience': 1, 'scores': [2, 2, 2, 2, 2],
        'completed': [0, 1], 'saved': 1, 'next': 2, 'remaining': [2, 3, 4]},
    'test_resumed_early_stop_uses_current_completed_epoch': {
        'input_epoch': 2, 'limit': 10, 'patience': 2, 'scores': [2, 2, 2, 2, 2, 2, 2],
        'completed': [3, 4, 5], 'saved': 5, 'next': 6, 'remaining': [6, 7, 8, 9]},
    'test_no_loop_above_horizon_preserves_loaded_epoch': {
        'input_epoch': 5, 'limit': 4, 'patience': 20, 'scores': [],
        'completed': [], 'saved': 5, 'next': 6, 'remaining': []},
    'test_ordinary_completion_control': {
        'input_epoch': -1, 'limit': 4, 'patience': 0, 'scores': [4, 3, 2, 1],
        'completed': [0, 1, 2, 3], 'saved': 3, 'next': 4, 'remaining': []},
    'test_already_complete_no_loop_control': {
        'input_epoch': 3, 'limit': 4, 'patience': 20, 'scores': [],
        'completed': [], 'saved': 3, 'next': 4, 'remaining': []},
    'test_single_epoch_completion_control': {
        'input_epoch': -1, 'limit': 1, 'patience': 1, 'scores': [2],
        'completed': [0], 'saved': 0, 'next': 1, 'remaining': []},
}


def epoch_program(source):
    """Keep real control expressions; supply heavy train/validation completion."""
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BaseTrainer')
    train = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    loop = next(n for n in train.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    start = next(n for n in train.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'start_epoch' for t in n.targets))
    patience = next(n for n in train.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'early_stopping_patience' for t in n.targets))
    initial = next(n for n in train.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'early_stopping_patience > 0')
    between = train.body[train.body.index(initial) + 1:train.body.index(loop)]
    assert all(isinstance(n, ast.Assign) for n in between)
    stopping = next(n for n in ast.walk(loop) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'early_stopping_patience > 0'
                    and any(isinstance(b, ast.Break) for b in ast.walk(n)))
    replacement = ast.parse('completed_epoch(epoch)\navg_val_loss = supplied_scores[epoch - supplied_start]').body
    shell = ast.For(target=deepcopy(loop.target), iter=deepcopy(loop.iter),
                    body=replacement + [deepcopy(stopping)], orelse=[], type_comment=None)
    assert not loop.orelse
    final = next(n for n in train.body if isinstance(n, ast.If)
                 and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                         and c.func.id == 'save_final_checkpoint' for c in ast.walk(n)))
    nodes = [deepcopy(start), deepcopy(patience), deepcopy(initial), *deepcopy(between), shell, deepcopy(final)]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, 'actual-epoch-control-with-supplied-completion', 'exec'), compile(
        ast.Expression(body=deepcopy(loop.iter)), 'actual-resume-range', 'eval')


class TinyState(torch.nn.Module):
    def __init__(self, value=12, seen=5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([float(value)]))
        self.register_buffer('seen', torch.tensor(seen, dtype=torch.int64))


class GenericFinalEpochTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def objects(self):
        model = TinyState()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        optimizer.state[model.weight]['momentum_buffer'] = torch.full_like(model.weight, 7)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return model, optimizer, scheduler

    def run_case(self, name):
        case = CASES[name]
        model, optimizer, scheduler = self.objects()
        mgr = SimpleNamespace(model_name='fixture', max_epoch=case['limit'],
            early_stopping_patience=case['patience'], initial_lr=0.05, scheduler='step',
            scheduler_kwargs={'step_size': 2, 'gamma': 0.5}, rebuild_from_checkpoint_config=False)
        initial_path = self.root / 'initial.pth'
        # All state/history is a prescribed fixture, not generated by training.
        with redirect_stdout(io.StringIO()):
            save_checkpoint(model, optimizer, scheduler, case['input_epoch'], initial_path)
        initial_bytes = initial_path.read_bytes()
        with redirect_stdout(io.StringIO()):
            model, optimizer, scheduler, start, loaded = load_checkpoint(initial_path, model, optimizer,
                scheduler, mgr, torch.device('cpu'), load_weights_only=False)
        self.assertTrue(loaded)
        self.assertEqual(start, case['input_epoch'] + 1)
        completed = []

        def complete(epoch):
            completed.append(epoch)
            # Prescribed completion markers only; no forward/backward/update.
            with torch.no_grad():
                model.weight.fill_(epoch + 1)
                model.seen.fill_(epoch)

        owner = SimpleNamespace(mgr=mgr, is_distributed=False, rank=0)
        source = Path(inspect.getsourcefile(load_checkpoint)).resolve().parents[1] / 'training/train.py'
        program, range_code = epoch_program(source.read_text(encoding='utf8'))
        ns = {'self': owner, 'training_state': {'start_epoch': start}, 'supplied_start': start,
              'supplied_scores': case['scores'], 'completed_epoch': complete,
              'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
              'model_ckpt_dir': self.root, 'train_dataset': None, 'save_final_checkpoint': save_final_checkpoint}
        with redirect_stdout(io.StringIO()):
            exec(program, ns)
        path = Path(ns['final_model_path'])
        before = path.read_bytes()
        restored, restored_opt, restored_sched = self.objects()
        with redirect_stdout(io.StringIO()):
            restored, restored_opt, restored_sched, next_epoch, loaded = load_checkpoint(
                path, restored, restored_opt, restored_sched, mgr, torch.device('cpu'), load_weights_only=False)
        self.assertTrue(loaded)
        self.assertEqual(completed, case['completed'])
        self.assertTrue(torch.equal(restored.weight, model.weight))
        self.assertTrue(torch.equal(restored.seen, model.seen))
        self.assertTrue(torch.equal(restored_opt.state[restored.weight]['momentum_buffer'],
                                    optimizer.state[model.weight]['momentum_buffer']))
        self.assertEqual(restored_sched.state_dict(), scheduler.state_dict())
        self.assertIs(restored_sched.optimizer, restored_opt)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(initial_path.read_bytes(), initial_bytes)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        remaining = list(eval(range_code, {'start_epoch': next_epoch, 'self': owner}))
        self.assertEqual(payload['epoch'], case['saved'])
        self.assertEqual(next_epoch, case['next'])
        self.assertEqual(remaining, case['remaining'])

    def test_early_stop_final_epoch_matches_completed_range(self):
        self.run_case('test_early_stop_final_epoch_matches_completed_range')

    def test_resumed_early_stop_uses_current_completed_epoch(self):
        self.run_case('test_resumed_early_stop_uses_current_completed_epoch')

    def test_no_loop_above_horizon_preserves_loaded_epoch(self):
        self.run_case('test_no_loop_above_horizon_preserves_loaded_epoch')

    def test_ordinary_completion_control(self):
        self.run_case('test_ordinary_completion_control')

    def test_already_complete_no_loop_control(self):
        self.run_case('test_already_complete_no_loop_control')

    def test_single_epoch_completion_control(self):
        self.run_case('test_single_epoch_completion_control')


if __name__ == '__main__':
    unittest.main(verbosity=2)
