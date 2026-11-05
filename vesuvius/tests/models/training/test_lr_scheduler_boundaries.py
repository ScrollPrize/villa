"""Prepared real-PyTorch CPU regressions; run only in a PBS allocation."""
import torch
from vesuvius.models.training.lr_schedulers import PolyLRScheduler, WarmupPolyLRScheduler, get_scheduler

def _optimizer():
    return torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
import unittest

class RegressionTests(unittest.TestCase):

    def test_polynomial_decay_stays_real_and_zero_after_horizon__case_0(self):
        warmup = False
        optimizer = _optimizer()
        scheduler = WarmupPolyLRScheduler(optimizer, 0.1, 10, 2) if warmup else PolyLRScheduler(optimizer, 0.1, 10)
        for step in [10, 11, 20]:
            scheduler.step(step)
            assert optimizer.param_groups[0]['lr'] == 0.0

    def test_polynomial_decay_stays_real_and_zero_after_horizon__case_1(self):
        warmup = True
        optimizer = _optimizer()
        scheduler = WarmupPolyLRScheduler(optimizer, 0.1, 10, 2) if warmup else PolyLRScheduler(optimizer, 0.1, 10)
        for step in [10, 11, 20]:
            scheduler.step(step)
            assert optimizer.param_groups[0]['lr'] == 0.0

    def test_in_horizon_polynomial_formula_is_preserved__case_0(self):
        warmup = False
        optimizer = _optimizer()
        scheduler = WarmupPolyLRScheduler(optimizer, 0.1, 10, 2) if warmup else PolyLRScheduler(optimizer, 0.1, 10)
        for step in [0, 1, 2, 5, 9]:
            scheduler.step(step)
            expected = 0.1 * step / 2 if warmup and step < 2 else 0.1 * (1 - (step - 2) / 8) ** 0.9 if warmup else 0.1 * (1 - step / 10) ** 0.9
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)

    def test_in_horizon_polynomial_formula_is_preserved__case_1(self):
        warmup = True
        optimizer = _optimizer()
        scheduler = WarmupPolyLRScheduler(optimizer, 0.1, 10, 2) if warmup else PolyLRScheduler(optimizer, 0.1, 10)
        for step in [0, 1, 2, 5, 9]:
            scheduler.step(step)
            expected = 0.1 * step / 2 if warmup and step < 2 else 0.1 * (1 - (step - 2) / 8) ** 0.9 if warmup else 0.1 * (1 - step / 10) ** 0.9
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)

    def test_invalid_horizon_is_rejected_before_optimizer_changes__case_0(self):
        warmup = False
        max_steps = 0
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'max_steps'):
            if warmup:
                WarmupPolyLRScheduler(optimizer, 0.1, max_steps, 0)
            else:
                PolyLRScheduler(optimizer, 0.1, max_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_invalid_horizon_is_rejected_before_optimizer_changes__case_1(self):
        warmup = False
        max_steps = -1
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'max_steps'):
            if warmup:
                WarmupPolyLRScheduler(optimizer, 0.1, max_steps, 0)
            else:
                PolyLRScheduler(optimizer, 0.1, max_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_invalid_horizon_is_rejected_before_optimizer_changes__case_2(self):
        warmup = True
        max_steps = 0
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'max_steps'):
            if warmup:
                WarmupPolyLRScheduler(optimizer, 0.1, max_steps, 0)
            else:
                PolyLRScheduler(optimizer, 0.1, max_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_invalid_horizon_is_rejected_before_optimizer_changes__case_3(self):
        warmup = True
        max_steps = -1
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'max_steps'):
            if warmup:
                WarmupPolyLRScheduler(optimizer, 0.1, max_steps, 0)
            else:
                PolyLRScheduler(optimizer, 0.1, max_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_invalid_warmup_is_rejected_before_optimizer_changes__case_0(self):
        warmup_steps = -1
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'warmup_steps'):
            WarmupPolyLRScheduler(optimizer, 0.1, 10, warmup_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_invalid_warmup_is_rejected_before_optimizer_changes__case_1(self):
        warmup_steps = 10
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'warmup_steps'):
            WarmupPolyLRScheduler(optimizer, 0.1, 10, warmup_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_invalid_warmup_is_rejected_before_optimizer_changes__case_2(self):
        warmup_steps = 11
        optimizer = _optimizer()
        with self.assertRaisesRegex(ValueError, 'warmup_steps'):
            WarmupPolyLRScheduler(optimizer, 0.1, 10, warmup_steps)
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_default_step_scheduler_can_advance_for_short_runs__case_0(self):
        horizon = 1
        optimizer = _optimizer()
        scheduler = get_scheduler('step', optimizer, 0.1, horizon)
        assert scheduler.step_size == max(1, horizon // 3)
        for _ in range(horizon):
            optimizer.step()
            scheduler.step()
        assert 0.0 <= optimizer.param_groups[0]['lr'] <= 0.1

    def test_default_step_scheduler_can_advance_for_short_runs__case_1(self):
        horizon = 2
        optimizer = _optimizer()
        scheduler = get_scheduler('step', optimizer, 0.1, horizon)
        assert scheduler.step_size == max(1, horizon // 3)
        for _ in range(horizon):
            optimizer.step()
            scheduler.step()
        assert 0.0 <= optimizer.param_groups[0]['lr'] <= 0.1

    def test_default_step_scheduler_can_advance_for_short_runs__case_2(self):
        horizon = 3
        optimizer = _optimizer()
        scheduler = get_scheduler('step', optimizer, 0.1, horizon)
        assert scheduler.step_size == max(1, horizon // 3)
        for _ in range(horizon):
            optimizer.step()
            scheduler.step()
        assert 0.0 <= optimizer.param_groups[0]['lr'] <= 0.1

    def test_default_step_scheduler_can_advance_for_short_runs__case_3(self):
        horizon = 9
        optimizer = _optimizer()
        scheduler = get_scheduler('step', optimizer, 0.1, horizon)
        assert scheduler.step_size == max(1, horizon // 3)
        for _ in range(horizon):
            optimizer.step()
            scheduler.step()
        assert 0.0 <= optimizer.param_groups[0]['lr'] <= 0.1

    def test_nonpositive_explicit_step_size_is_rejected__case_0(self):
        step_size = 0
        with self.assertRaisesRegex(ValueError, 'step_size'):
            get_scheduler('step', _optimizer(), 0.1, 10, step_size=step_size)

    def test_nonpositive_explicit_step_size_is_rejected__case_1(self):
        step_size = -1
        with self.assertRaisesRegex(ValueError, 'step_size'):
            get_scheduler('step', _optimizer(), 0.1, 10, step_size=step_size)
