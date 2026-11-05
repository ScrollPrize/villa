"""Prepared PBS-only CPU packing controls;no real collective or training.

Executes actual native validation assignments and actual Accelerate trimming.
Rank payloads are captured from that same native code;the collective and its
gradient-state metadata are explicitly supplied. All logits/labels are engineered.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import inspect
import textwrap
from types import MethodType, SimpleNamespace
import unittest

import accelerate
from accelerate import Accelerator
from accelerate.data_loader import BatchSamplerShard
import torch
from torch.utils.data import BatchSampler, SequentialSampler

import vesuvius.ink_detection.training.train as training
from vesuvius.ink_detection.training.metrics import BalancedAccuracy, Confusion

GATHER_AST_SHA256 = "6f76875f8ed470badf1779d8b504fada1258d788d2962e50bdf33b0677c1e5b6"
SHARD_AST_SHA256 = "99b45b5b371dd29c81e51da37338953caa8eb7abe2adfd185b0c5cf093510ab3"


def _semantic_digest(value):
    node = ast.parse(textwrap.dedent(inspect.getsource(value))).body[0]
    for item in ast.walk(node):
        if isinstance(item, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.body and isinstance(item.body[0], ast.Expr) and isinstance(item.body[0].value, ast.Constant) and isinstance(item.body[0].value.value, str):
                item.body.pop(0)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _packing_code():
    function = next(n for n in ast.parse(inspect.getsource(training)).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_run_training")
    loop = next(n for n in ast.walk(function) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "val_index")
    assignments = [n for n in loop.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in {"batch_counts", "gathered", "counts"} for t in n.targets)]
    assert len(assignments) == 3
    shape_checks = [n for n in loop.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) in {"primary_val_predictions.shape != val_targets.shape",
                                               "val_ignore.shape != val_targets.shape"}]
    assert len(shape_checks) in {0, 2}
    selected = {id(n) for n in assignments + shape_checks}
    # Keep source order and execute the repaired preflight as well as packing.
    # The original arm has no caller preflight;its unchanged helper rejects.
    nodes = [n for n in loop.body if id(n) in selected]
    return compile(ast.Module(body=nodes, type_ignores=[]), training.__file__, "exec")


def _sample(logits, targets, valid=None, *, volume=False):
    shape = (1, 1, 1, len(logits)) if volume else (1, 1, len(logits))
    return SimpleNamespace(
        logits=torch.tensor(logits, dtype=torch.float32, device="cpu").reshape(shape),
        targets=torch.tensor(targets, dtype=torch.float32, device="cpu").reshape(shape),
        valid=torch.tensor([True]*len(logits) if valid is None else valid,
                           dtype=torch.bool, device="cpu").reshape(shape))


def _standard_samples():
    return [_sample([4], [1]), _sample([-4], [0]), _sample([-4], [1]),
            _sample([4], [0]), _sample([-4], [0])]


def _batch(samples, indices):
    return SimpleNamespace(logits=torch.stack([samples[i].logits for i in indices]),
        targets=torch.stack([samples[i].targets for i in indices]),
        ignore=(~torch.stack([samples[i].valid for i in indices])).to(torch.float32))


def _execute(code, batch, accelerator, initial):
    namespace = dict(training.__dict__, primary_val_predictions=batch.logits,
        val_targets=batch.targets, val_ignore=batch.ignore, accelerator=accelerator,
        counts=initial, confusion_metric=Confusion())
    exec(code, namespace)
    return namespace["counts"]


class _Payload(Exception):
    def __init__(self, tensor):
        super().__init__("capture native payload before collective")
        self.tensor = tensor.detach().clone()


def _capture(code, batch):
    def intercept(payload):
        raise _Payload(payload)
    try:
        _execute(code, batch, SimpleNamespace(gather_for_metrics=intercept),
                 Confusion.zero_counts(device=torch.device("cpu")))
    except _Payload as captured:
        return captured.tensor
    raise AssertionError("native code did not reach metric gather")


def _scenario(samples, batch_size, ranks, cap=None):
    code = _packing_code()
    def sampler():
        return BatchSampler(SequentialSampler(range(len(samples))), batch_size=batch_size, drop_last=False)
    # Match ordinary map-style prepare_data_loader:single process is unsharded.
    shards = ([list(sampler())] if ranks == 1 else [list(BatchSamplerShard(
        sampler(), num_processes=ranks, process_index=rank,
        split_batches=False, even_batches=True)) for rank in range(ranks)])
    assert len({len(shard) for shard in shards}) == 1
    available = len(shards[0])
    steps = available if cap is None else min(available, cap)
    counts = Confusion.zero_counts(device=torch.device("cpu"))
    trace = []
    for step in range(steps):
        batches = [_batch(samples, shard[step]) for shard in shards]
        # Capture both rank payloads from the actual production assignments.
        # No mirrored implementation decides whether they are aggregate rows.
        payloads = [_capture(code, batch) for batch in batches]
        def collective(local):
            if not torch.equal(local, payloads[0]):
                raise AssertionError("local payload changed between capture and gather")
            return torch.cat(payloads, dim=0)
        supplied = SimpleNamespace(gather=collective,
            gradient_state=SimpleNamespace(end_of_dataloader=step == available-1,
                remainder=len(samples) % (batch_size*ranks)))
        supplied.gather_for_metrics = MethodType(Accelerator.gather_for_metrics, supplied)
        counts = _execute(code, batches[0], supplied, counts)
        trace.append({"indices": [shard[step] for shard in shards], "payloads": payloads})
    return counts, trace


def _values(counts):
    return [value.item() for value in (counts.tp, counts.fp, counts.fn, counts.tn)]


class ConfusionSampleRowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The prospective job uses the already installed pinned version,never
        # downloads/replaces a dependency to make a changed contract pass.
        if accelerate.__version__ != "1.14.0":
            raise AssertionError("expected existing Accelerate1.14.0 environment")
        if _semantic_digest(Accelerator.gather_for_metrics) != GATHER_AST_SHA256:
            raise AssertionError("gather_for_metrics differs from pinned source")
        if _semantic_digest(BatchSamplerShard) != SHARD_AST_SHA256:
            raise AssertionError("BatchSamplerShard differs from pinned source")

    def assert_counts(self, result, expected, accuracy):
        self.assertEqual(_values(result), expected)
        for value in (result.tp, result.fp, result.fn, result.tn):
            self.assertEqual(value.dtype, torch.int64)
            self.assertEqual(value.device.type, "cpu")
        self.assertAlmostEqual(BalancedAccuracy.from_counts(result).item(), float(accuracy), places=12)

    def test_padded_three_sample_tail_removes_duplicate_counts(self):
        result, _ = _scenario(_standard_samples()[:3], 2, 2)
        self.assert_counts(result, [1, 0, 1, 1], Fraction(3, 4))

    def test_one_sample_tail_after_full_step_removes_duplicate_counts(self):
        result, _ = _scenario(_standard_samples(), 2, 2)
        self.assert_counts(result, [1, 1, 1, 2], Fraction(7, 12))

    def test_complete_global_batch_control(self):
        result, _ = _scenario(_standard_samples()[:4], 2, 2)
        self.assert_counts(result, [1, 1, 1, 1], Fraction(1, 2))

    def test_local_batch_one_tail_control(self):
        result, _ = _scenario(_standard_samples()[:3], 1, 2)
        self.assert_counts(result, [1, 0, 1, 1], Fraction(3, 4))

    def test_ordinary_single_process_tail_control(self):
        result, _ = _scenario(_standard_samples(), 2, 1)
        self.assert_counts(result, [1, 1, 1, 2], Fraction(7, 12))

    def test_validation_cap_before_tail_control(self):
        result, _ = _scenario(_standard_samples(), 2, 2, cap=1)
        self.assert_counts(result, [1, 1, 1, 1], Fraction(1, 2))

    def test_all_ignored_sample_retains_zero_row_until_trim(self):
        samples = [_sample([4], [1]), _sample([4], [0], [False]), _sample([-4], [0])]
        result, trace = _scenario(samples, 2, 2)
        self.assert_counts(result, [1, 0, 0, 1], Fraction(1))
        self.assertEqual(tuple(trace[0]["payloads"][0].shape), (2, 4))
        self.assertTrue(torch.equal(trace[0]["payloads"][0][1], torch.zeros(4, dtype=torch.int64, device="cpu")))

    def test_unequal_valid_voxel_support_is_counted_not_averaged(self):
        samples = [_sample([4]*4, [1]*4),
            _sample([-4]*4, [0]*4, [True, False, False, False]),
            _sample([-4]*4, [1]*4, [True, False, False, False])]
        result, _ = _scenario(samples, 2, 2)
        self.assert_counts(result, [4, 0, 1, 1], Fraction(9, 10))

    def test_int64_payload_column_order_control(self):
        samples = [_sample([4,4,4,-4,-4,-4,-4,-4,-4,-4], [1,1,0,1,1,1,0,0,0,0])]
        result, trace = _scenario(samples, 1, 1)
        self.assert_counts(result, [2, 1, 3, 4], Fraction(3, 5))
        self.assertTrue(torch.equal(trace[0]["payloads"][0], torch.tensor([[2,1,3,4]], dtype=torch.int64, device="cpu")))

    def test_legacy_mask_empty_class_and_threshold_controls(self):
        recipes = [([_sample([4], [1], [False])], [0,0,0,0], Fraction(0)),
            ([_sample([4], [1]), _sample([4], [0], [False])], [1,0,0,0], Fraction(1)),
            ([_sample([0], [1])], [1,0,0,0], Fraction(1)),
            ([_sample([4,-4], [1,0], volume=True)], [1,0,0,1], Fraction(1))]
        for samples, expected, accuracy in recipes:
            with self.subTest(expected=expected, samples=len(samples)):
                result, _ = _scenario(samples, len(samples), 1)
                self.assert_counts(result, expected, accuracy)
                batch = _batch(samples, list(range(len(samples))))
                legacy = Confusion().compute_batch(training.MetricBatch(logits=batch.logits,
                    targets=batch.targets, valid_mask=batch.ignore <= 0))
                self.assertEqual(_values(result), _values(legacy))

    def test_mismatched_batch_shapes_preserve_legacy_rejection_control(self):
        good = _batch(_standard_samples()[:2], [0, 1])
        invalid = [SimpleNamespace(logits=good.logits, targets=good.targets[:1], ignore=good.ignore),
            SimpleNamespace(logits=good.logits[:1], targets=good.targets, ignore=good.ignore),
            SimpleNamespace(logits=good.logits, targets=good.targets, ignore=good.ignore[:1])]
        def forbidden(payload):
            raise AssertionError("malformed shapes reached the collective")
        for batch in invalid:
            with self.subTest(logits=tuple(batch.logits.shape), targets=tuple(batch.targets.shape),
                              mask=tuple(batch.ignore.shape)):
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    _execute(_packing_code(), batch, SimpleNamespace(gather_for_metrics=forbidden),
                             Confusion.zero_counts(device=torch.device("cpu")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
