"""Prepared CPU controls: actual source methods/factory, supplied tiny finite work.

PBS only. No full trainer, real data loader, CUDA/DDP or scientific training.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext, redirect_stdout
import io
import math
from pathlib import Path
from time import perf_counter
from types import MethodType, SimpleNamespace
import unittest

import torch
from vesuvius.models.training import lr_schedulers


CASES = {
    # scheduler, epochs, loader batches, cap, accumulation, horizon, updates, cycle
    "divisible_cosine": ("cosine_warmup", 2, 8, None, 4, 4, 4, None),
    "partial_cosine": ("cosine_warmup", 3, 5, None, 2, 9, 9, None),
    "capped_cosine": ("cosine_warmup", 2, 12, 7, 4, 4, 4, None),
    "large_cosine": ("cosine_warmup", 2, 3, None, 8, 2, 2, None),
    "diffusers_partial": ("diffusers_cosine_warmup", 3, 5, None, 2, 9, 9, None),
    "unit_accumulation_control": ("cosine_warmup", 2, 4, None, 1, 8, 8, None),
    "single_batch_control": ("cosine_warmup", 2, 1, None, 8, 2, 2, None),
    "epoch_poly_control": ("poly", 3, 5, None, 2, 3, 9, None),
    "explicit_cycle_control": ("cosine_warmup", 2, 8, None, 4, 11, 4, 11),
}


def source_programs(factory):
    path = Path(lr_schedulers.__file__).with_name("train.py")
    cls = next(n for n in ast.parse(path.read_text(encoding="utf8")).body
               if isinstance(n, ast.ClassDef) and n.name == "BaseTrainer")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    namespace = dict(torch=torch, get_scheduler=factory, perf_counter=perf_counter)
    names = ("_get_scheduler", "_train_step", "_get_scaler")
    exec(compile(ast.Module(body=[methods[n] for n in names], type_ignores=[]),
                 str(path), "exec"), namespace)

    body = methods["_initialize_training"].body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "steps_per_epoch" for t in n.targets))
    stop = next(i for i in range(start, len(body)) if isinstance(body[i], ast.Assign)
                and any(isinstance(t, ast.Tuple) and ast.unparse(t) == "(scheduler, is_per_iteration_scheduler)"
                        for t in body[i].targets))
    train = methods["train"]
    by_test = {ast.unparse(n.test): n for n in ast.walk(train) if isinstance(n, ast.If)}
    flag = next(n for n in ast.walk(train) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "step_scheduler_at_epoch_begin" for t in n.targets))
    def program(nodes):
        return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    namespace["initialize"] = program(body[start:stop + 1])
    namespace["after_update"] = program([by_test["optimizer_stepped and is_per_iteration_scheduler"]])
    namespace["epoch_begin"] = program([flag, by_test["step_scheduler_at_epoch_begin"]])
    namespace["epoch_end"] = program([by_test["not is_per_iteration_scheduler and (not step_scheduler_at_epoch_begin)"]])
    return namespace


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.05)
        self.updates = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.updates += 1
        return result


class GenericSchedulerAccumulationTests(unittest.TestCase):
    def run_case(self, name):
        kind, epochs, batches, cap, accumulation, horizon, updates, cycle = CASES[name]
        calls = []
        def factory(**kwargs):
            calls.append(dict(kwargs))
            return lr_schedulers.get_scheduler(**kwargs)
        code = source_programs(factory)
        kwargs = {"warmup_steps": 0} if "warmup" in kind else {}
        if cycle is not None:
            kwargs["first_cycle_steps"] = cycle
        manager = SimpleNamespace(scheduler=kind, scheduler_kwargs=kwargs,
            max_epoch=epochs, max_steps_per_epoch=cap, initial_lr=0.05,
            gradient_accumulation=accumulation, gradient_clip=12.0,
            augment_on_device=False)
        trainer = SimpleNamespace(mgr=manager, startup_timing=False,
            _first_optimizer_timing_recorded=False)
        for method in ("_get_scheduler", "_train_step", "_get_scaler"):
            setattr(trainer, method, MethodType(code[method], trainer))
        trainer._prepare_batch = lambda data, **unused: data
        trainer._accumulate_aug_timers = lambda data: None
        trainer._get_model_outputs = lambda model, data: (data, {}, {"task": model(data)})
        trainer._compute_train_loss = lambda outputs, targets, losses: (
            outputs["task"].square().mean(), {})
        trainer._update_ema_model = lambda model: None
        model = torch.nn.Linear(1, 1, bias=False, device="cpu")
        with torch.no_grad():
            model.weight.fill_(0.25)
        optimizer = CountingSGD(model.parameters())
        context = dict(self=trainer, train_dataloader=range(batches), optimizer=optimizer)
        with redirect_stdout(io.StringIO()):
            exec(code["initialize"], {}, context)
        scheduler = context["scheduler"]
        effective = scheduler.first_cycle_steps if cycle is not None else calls[0]["max_steps"]
        self.assertEqual(effective, horizon)
        self.assertEqual(len(calls), 1)
        if kind == "cosine_warmup":
            self.assertEqual(scheduler.first_cycle_steps, horizon)
        if kind == "poly":
            self.assertEqual(scheduler.max_steps, horizon)
        advances = []
        real_step = scheduler.step
        def counted_step(*args, **kwargs):
            advances.append((args, kwargs))
            return real_step(*args, **kwargs)
        scheduler.step = counted_step
        scaler = trainer._get_scaler("cpu", use_amp=False)
        num_iters = batches if cap is None else min(batches, cap)
        for epoch in range(epochs):
            context["epoch"] = epoch
            exec(code["epoch_begin"], {}, context)
            for i in range(num_iters):
                if i % accumulation == 0:
                    optimizer.zero_grad(set_to_none=True)
                result = trainer._train_step(model=model,
                    data_dict=torch.ones((1, 1), device="cpu"), loss_fns={},
                    use_amp=False, autocast_ctx=nullcontext(), epoch=epoch, step=i,
                    scaler=scaler, optimizer=optimizer, num_iters=num_iters,
                    grad_accumulate_n=accumulation)
                context["optimizer_stepped"] = result[-1]
                exec(code["after_update"], {}, context)
                self.assertTrue(all(math.isfinite(p["lr"]) for p in optimizer.param_groups))
            exec(code["epoch_end"], {}, context)
        self.assertEqual(optimizer.updates, updates)
        expected_advances = updates if context["is_per_iteration_scheduler"] else epochs
        self.assertEqual(len(advances), expected_advances)
        self.assertTrue(torch.isfinite(model.weight).all().item())

    def test_divisible_cosine(self): self.run_case("divisible_cosine")
    def test_partial_cosine(self): self.run_case("partial_cosine")
    def test_capped_cosine(self): self.run_case("capped_cosine")
    def test_large_cosine(self): self.run_case("large_cosine")
    def test_diffusers_partial(self): self.run_case("diffusers_partial")
    def test_unit_accumulation_control(self): self.run_case("unit_accumulation_control")
    def test_single_batch_control(self): self.run_case("single_batch_control")
    def test_epoch_poly_control(self): self.run_case("epoch_poly_control")
    def test_explicit_cycle_control(self): self.run_case("explicit_cycle_control")
