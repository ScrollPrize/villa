"""Prepared CPU regressions for the actual native-ink update block.

Run only inside PBS. This isolates control flow; it does not reproduce CUDA AMP.
"""
import ast
import inspect
from types import SimpleNamespace
import torch
import vesuvius.ink_detection.training.train as training

def _update_block():
    tree = ast.parse(inspect.getsource(training))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and any((isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name) and (child.target.id == 'optimizer_step') for child in node.body))]
    assert len(matches) == 1
    return compile(ast.Module(body=matches, type_ignores=[]), 'native-update', 'exec')
import unittest

class RegressionTests(unittest.TestCase):

    def test_scheduler_counter_and_ema_follow_successful_updates__case_0(self):
        sync = False
        skipped = False
        calls = []
        ema_state = {'weight': torch.tensor(1.0), 'counter': torch.tensor(4)}
        model_state = {'weight': torch.tensor(3.0), 'counter': torch.tensor(8)}
        namespace = {'torch': torch, 'accelerator': SimpleNamespace(sync_gradients=sync, optimizer_step_was_skipped=skipped), 'scheduler': SimpleNamespace(step=lambda: calls.append('scheduler')), 'optimizer_step': 0, 'config': SimpleNamespace(ema=SimpleNamespace(start_step=0, update_every_steps=1, decay=0.5)), 'ema_model': SimpleNamespace(state_dict=lambda: ema_state), 'unwrapped_model': SimpleNamespace(state_dict=lambda: model_state)}
        exec(_update_block(), namespace)
        updated = sync and (not skipped)
        assert len(calls) == int(updated)
        assert namespace['optimizer_step'] == int(updated)
        assert ema_state['weight'].item() == (2.0 if updated else 1.0)
        assert ema_state['counter'].item() == (8 if updated else 4)

    def test_scheduler_counter_and_ema_follow_successful_updates__case_1(self):
        sync = False
        skipped = True
        calls = []
        ema_state = {'weight': torch.tensor(1.0), 'counter': torch.tensor(4)}
        model_state = {'weight': torch.tensor(3.0), 'counter': torch.tensor(8)}
        namespace = {'torch': torch, 'accelerator': SimpleNamespace(sync_gradients=sync, optimizer_step_was_skipped=skipped), 'scheduler': SimpleNamespace(step=lambda: calls.append('scheduler')), 'optimizer_step': 0, 'config': SimpleNamespace(ema=SimpleNamespace(start_step=0, update_every_steps=1, decay=0.5)), 'ema_model': SimpleNamespace(state_dict=lambda: ema_state), 'unwrapped_model': SimpleNamespace(state_dict=lambda: model_state)}
        exec(_update_block(), namespace)
        updated = sync and (not skipped)
        assert len(calls) == int(updated)
        assert namespace['optimizer_step'] == int(updated)
        assert ema_state['weight'].item() == (2.0 if updated else 1.0)
        assert ema_state['counter'].item() == (8 if updated else 4)

    def test_scheduler_counter_and_ema_follow_successful_updates__case_2(self):
        sync = True
        skipped = False
        calls = []
        ema_state = {'weight': torch.tensor(1.0), 'counter': torch.tensor(4)}
        model_state = {'weight': torch.tensor(3.0), 'counter': torch.tensor(8)}
        namespace = {'torch': torch, 'accelerator': SimpleNamespace(sync_gradients=sync, optimizer_step_was_skipped=skipped), 'scheduler': SimpleNamespace(step=lambda: calls.append('scheduler')), 'optimizer_step': 0, 'config': SimpleNamespace(ema=SimpleNamespace(start_step=0, update_every_steps=1, decay=0.5)), 'ema_model': SimpleNamespace(state_dict=lambda: ema_state), 'unwrapped_model': SimpleNamespace(state_dict=lambda: model_state)}
        exec(_update_block(), namespace)
        updated = sync and (not skipped)
        assert len(calls) == int(updated)
        assert namespace['optimizer_step'] == int(updated)
        assert ema_state['weight'].item() == (2.0 if updated else 1.0)
        assert ema_state['counter'].item() == (8 if updated else 4)

    def test_scheduler_counter_and_ema_follow_successful_updates__case_3(self):
        sync = True
        skipped = True
        calls = []
        ema_state = {'weight': torch.tensor(1.0), 'counter': torch.tensor(4)}
        model_state = {'weight': torch.tensor(3.0), 'counter': torch.tensor(8)}
        namespace = {'torch': torch, 'accelerator': SimpleNamespace(sync_gradients=sync, optimizer_step_was_skipped=skipped), 'scheduler': SimpleNamespace(step=lambda: calls.append('scheduler')), 'optimizer_step': 0, 'config': SimpleNamespace(ema=SimpleNamespace(start_step=0, update_every_steps=1, decay=0.5)), 'ema_model': SimpleNamespace(state_dict=lambda: ema_state), 'unwrapped_model': SimpleNamespace(state_dict=lambda: model_state)}
        exec(_update_block(), namespace)
        updated = sync and (not skipped)
        assert len(calls) == int(updated)
        assert namespace['optimizer_step'] == int(updated)
        assert ema_state['weight'].item() == (2.0 if updated else 1.0)
        assert ema_state['counter'].item() == (8 if updated else 4)
