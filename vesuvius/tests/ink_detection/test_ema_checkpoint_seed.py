"""Prepared real-PyTorch checkpoint regressions; run only inside PBS."""
import torch
from torch import nn
from vesuvius.ink_detection.models.checkpoint import restore_training_state

def _model(value):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model
import unittest

class RegressionTests(unittest.TestCase):

    def test_ema_uses_restored_main_weights_unless_full_ema_state_is_available__case_0(self):
        weights_only = True
        has_ema = False
        model, ema_model = (_model(-1), _model(-2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint = {'model': _model(3).state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'step': 7}
        if has_ema:
            checkpoint['ema_model'] = _model(5).state_dict()
            checkpoint['ema_optimizer_step'] = 6
        next_step, ema_step = restore_training_state(model, optimizer, scheduler, checkpoint, 'memory.pth', load_weights_only=weights_only, ema_model=ema_model)
        assert model.weight.item() == 3
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)
        assert next_step == (0 if weights_only else 8)
        assert ema_step == (6 if has_ema and (not weights_only) else 0)
        with torch.no_grad():
            model.weight.add_(10)
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)

    def test_ema_uses_restored_main_weights_unless_full_ema_state_is_available__case_1(self):
        weights_only = True
        has_ema = True
        model, ema_model = (_model(-1), _model(-2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint = {'model': _model(3).state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'step': 7}
        if has_ema:
            checkpoint['ema_model'] = _model(5).state_dict()
            checkpoint['ema_optimizer_step'] = 6
        next_step, ema_step = restore_training_state(model, optimizer, scheduler, checkpoint, 'memory.pth', load_weights_only=weights_only, ema_model=ema_model)
        assert model.weight.item() == 3
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)
        assert next_step == (0 if weights_only else 8)
        assert ema_step == (6 if has_ema and (not weights_only) else 0)
        with torch.no_grad():
            model.weight.add_(10)
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)

    def test_ema_uses_restored_main_weights_unless_full_ema_state_is_available__case_2(self):
        weights_only = False
        has_ema = False
        model, ema_model = (_model(-1), _model(-2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint = {'model': _model(3).state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'step': 7}
        if has_ema:
            checkpoint['ema_model'] = _model(5).state_dict()
            checkpoint['ema_optimizer_step'] = 6
        next_step, ema_step = restore_training_state(model, optimizer, scheduler, checkpoint, 'memory.pth', load_weights_only=weights_only, ema_model=ema_model)
        assert model.weight.item() == 3
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)
        assert next_step == (0 if weights_only else 8)
        assert ema_step == (6 if has_ema and (not weights_only) else 0)
        with torch.no_grad():
            model.weight.add_(10)
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)

    def test_ema_uses_restored_main_weights_unless_full_ema_state_is_available__case_3(self):
        weights_only = False
        has_ema = True
        model, ema_model = (_model(-1), _model(-2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint = {'model': _model(3).state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'step': 7}
        if has_ema:
            checkpoint['ema_model'] = _model(5).state_dict()
            checkpoint['ema_optimizer_step'] = 6
        next_step, ema_step = restore_training_state(model, optimizer, scheduler, checkpoint, 'memory.pth', load_weights_only=weights_only, ema_model=ema_model)
        assert model.weight.item() == 3
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)
        assert next_step == (0 if weights_only else 8)
        assert ema_step == (6 if has_ema and (not weights_only) else 0)
        with torch.no_grad():
            model.weight.add_(10)
        assert ema_model.weight.item() == (5 if has_ema and (not weights_only) else 3)

    def test_checkpoint_without_requested_ema_retains_existing_resume_behavior__case_0(self):
        model = _model(-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint = {'model': _model(3).state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'step': 7}
        assert restore_training_state(model, optimizer, scheduler, checkpoint, 'memory.pth') == (8, 0)
        assert model.weight.item() == 3
