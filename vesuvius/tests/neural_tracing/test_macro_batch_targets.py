"""Production loss statistics and source-extracted geometry prepass on real CT."""
import copy
from dataclasses import replace
from datetime import timedelta
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader
from tests.neural_tracing.test_effective_batch_accumulation import (
    _pointwise_fiber_model, _compare_parameters,
)
from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DBatch
from vesuvius.neural_tracing.fiber_trace_3d.cuda_batch_pipeline import FiberTrace3DCudaBatchPipeline
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
    MacroBatchTargetSource, FiberTrace3DSampleQueue, SampleQueueSource,
    apply_deferred_batch_augmentation,
)
from vesuvius.neural_tracing.fiber_trace_3d.queued_batches import QueuedBatchView
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets, target_normalization_counts
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    compute_losses, compute_conditioned_losses, _slice_batch, _finalize_loss_statistics,
)
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator


class _Source:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.releases = 0
        self.closed = False

    def __next__(self):
        return next(self.batches)

    def release_current(self):
        self.releases += 1

    def close(self):
        self.closed = True


def _prepass(source, config, sizes=(1, 1), stride=1, conditioned=False, device="cpu"):
    return MacroBatchTargetSource(source, config=config, device=torch.device(device),
        output_stride=stride, batch_sizes=sizes, conditioned=conditioned)


def _raw_pair(tmp_path):
    loader = _real_paris4_loader(tmp_path)
    raw = loader.load_batch(0, sample_mode="flat")
    return loader.config, raw, [_slice_batch(raw, i, i + 1) for i in range(2)]


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("mask", ["all", "partial", "empty"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_prepass_matches_masked_target_counts_without_image_transfer(tmp_path, monkeypatch, stride, mask, conditioned):
    config, raw, _ = _raw_pair(tmp_path)
    valid = raw.valid_mask.clone()
    if mask == "partial":
        valid[0, :, :16] = False
    elif mask == "empty":
        valid.zero_()
    raw = replace(raw, valid_mask=valid)
    parts = [_slice_batch(raw, i, i + 1) for i in range(2)]
    expected = sum((target_normalization_counts(materialize_targets(part, config,
                    output_stride=stride), conditioned=conditioned) for part in parts), torch.zeros(3))
    source = _Source(parts)
    prepass = _prepass(source, config, stride=stride, conditioned=conditioned)
    calls = []
    original = FiberTrace3DBatch.to

    def transfer(batch, *args, **kwargs):
        calls.append(batch.volume.numel())
        assert batch.valid_mask is not None
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(FiberTrace3DBatch, "to", transfer)
    first, second = next(prepass), next(prepass)
    assert source.releases == 2 and calls == [0, 0]
    assert torch.equal(first.normalization_counts, expected)
    assert torch.equal(second.normalization_counts, expected)
    assert first.normalization_counts.device.type == "cpu"
    assert torch.equal(first.volume, parts[0].volume)
    assert first.volume.data_ptr() != parts[0].volume.data_ptr()
    with pytest.raises(StopIteration):
        next(prepass)
    prepass.close()
    assert source.closed and not prepass.pending


def test_macro_batch_owns_all_tensor_storage_before_release(tmp_path):
    config, _, parts = _raw_pair(tmp_path)
    expected = [replace(part, **{name: value.clone() for name, value in vars(part).items()
                               if isinstance(value, torch.Tensor)}) for part in parts]

    class ReusingSource(_Source):
        def __next__(self):
            self.current = super().__next__()
            return self.current

        def release_current(self):
            super().release_current()
            for value in vars(self.current).values():
                if isinstance(value, torch.Tensor):
                    value.zero_()

    prepass = _prepass(ReusingSource(parts), config)
    for reference in expected:
        actual = next(prepass)
        for name, value in vars(reference).items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(getattr(actual, name), value), name
    prepass.close()


@pytest.mark.parametrize("materialized", [False, True])
def test_stride_change_requires_complete_normalization_prepass(tmp_path, materialized):
    config, raw, parts = _raw_pair(tmp_path)
    first = _prepass(_Source(parts), config, stride=1)
    try:
        staged = [next(first), next(first)]
    finally:
        first.close()
    if materialized:
        staged = [materialize_targets(batch, config) for batch in staged]
    for batch in staged:
        with pytest.raises(ValueError, match="complete optimizer batch"):
            materialize_targets(batch, config, output_stride=2)
    # Restaging clears the obsolete counts, rebuilding the whole logical batch.
    second = _prepass(_Source(staged), config, stride=2)
    expected = target_normalization_counts(materialize_targets(raw, config, output_stride=2))
    try:
        for _ in range(2):
            batch = next(second)
            targets = materialize_targets(batch, config, output_stride=2)
            assert targets.target_output_stride == targets.normalization_output_stride == 2
            torch.testing.assert_close(targets.normalization_counts, expected, rtol=0, atol=0)
    finally:
        second.close()


def test_prepass_releases_on_shape_and_geometry_errors_and_closes(tmp_path):
    config, raw, parts = _raw_pair(tmp_path)
    source = _Source([raw])
    prepass = _prepass(source, config, sizes=(1,))
    with pytest.raises(RuntimeError, match="canonical partition"):
        next(prepass)
    assert source.releases == 1
    prepass.close()
    assert source.closed
    source = _Source(parts)
    prepass = _prepass(source, config)

    def fail(operation):
        raise RuntimeError("interrupted geometry")

    prepass.prepare_geometry = fail
    with pytest.raises(RuntimeError, match="interrupted geometry"):
        next(prepass)
    assert source.releases == 1
    prepass.close()
    assert source.closed
    with pytest.raises(ValueError, match="nonempty local"):
        _prepass(_Source([]), config, sizes=())


def test_actual_queue_prepass_wraps_slots_and_reuses_workers(tmp_path):
    config, _, _ = _raw_pair(tmp_path)
    owner = FiberTrace3DSampleQueue(sources={"train": SampleQueueSource(config, sample_mode="flat")},
                                  workers=2, slot_count=2, initial_in_flight=1)
    pids = [p.pid for p in owner.pipeline.processes]
    view = QueuedBatchView(owner, "train", [(0,), (1,), (2,), (3,)])
    source = _prepass(view, config)
    try:
        batches = [next(source) for _ in range(4)]
        assert [b.stream_indices.item() for b in batches] == [0, 1, 2, 3]
        for offset in (0, 2):
            expected = sum((target_normalization_counts(materialize_targets(b, config))
                            for b in batches[offset:offset + 2]), torch.zeros(3))
            assert torch.equal(batches[offset].normalization_counts, expected)
        assert [p.pid for p in owner.pipeline.processes] == pids
    finally:
        source.close()
        owner.close()


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("mask", ["all", "no-positive", "empty"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_statistics_counts_are_target_only_for_real_masked_patches(tmp_path, conditioned, mask, dtype):
    config, raw, _ = _raw_pair(tmp_path)
    batch = materialize_targets(raw, replace(config, presence_radius_voxels=3.0))
    presence_mask = batch.presence_mask.clone()
    if mask == "no-positive":
        presence_mask &= batch.presence_target <= .5
    elif mask == "empty":
        presence_mask.zero_()
    batch = replace(batch, presence_mask=presence_mask)

    class ConditionedHead(torch.nn.Module):
        conditioned_decoder_enabled = True

        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, volume, *, conditioned_point_indices_bzyx, **kwargs):
            output = torch.full((len(volume), 7, *volume.shape[-3:]), self.value, dtype=dtype)
            idx = conditioned_point_indices_bzyx
            point = None if idx is None else output[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]
            return {"zero_output": output, "random_output": output, "point_output": point}

    expected = target_normalization_counts(batch, conditioned=conditioned)
    for value in (.1, .9):
        if conditioned:
            stats = compute_conditioned_losses(ConditionedHead(value), batch,
                direction_weight=1, presence_weight=1, _return_statistics=True)
        else:
            output = torch.full((len(batch.volume), 7, *batch.volume.shape[-3:]), value, dtype=dtype)
            stats = compute_losses(output, batch, direction_weight=1, presence_weight=1,
                                   _return_statistics=True)
        counts = torch.stack((stats.direction_count, stats.positive_presence_count,
                              stats.negative_presence_count))
        assert torch.equal(counts, expected)
        assert not counts.requires_grad
        metrics = _finalize_loss_statistics(stats, direction_weight=1, presence_weight=1)
        assert all(torch.isfinite(v) for v in metrics.values())


def test_statistics_reject_multibranch_but_legacy_loss_still_works(tmp_path):
    config, raw, _ = _raw_pair(tmp_path)
    batch = materialize_targets(raw, config)
    output = torch.full((len(batch.volume), 14, *batch.volume.shape[-3:]), .6)
    assert torch.isfinite(compute_losses(output, batch, direction_weight=1, presence_weight=1)["total"])
    with pytest.raises(ValueError, match="prediction-dependent counts"):
        compute_losses(output, batch, direction_weight=1, presence_weight=1, _return_statistics=True)


def _distributed_finalizer(rank, init_file, batch):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(41)
        model = _pointwise_fiber_model("cpu")
        oracle = copy.deepcopy(model)
        expected = compute_losses(oracle(batch.volume), batch, direction_weight=2, presence_weight=3)
        expected["total"].backward()
        part = _slice_batch(batch, rank, rank + 1)
        stats = compute_losses(model(part.volume), part, direction_weight=2,
                               presence_weight=3, _return_statistics=True)
        metrics = _finalize_loss_statistics(stats, direction_weight=2, presence_weight=3,
            distributed=SimpleNamespace(enabled=True, world_size=2))
        metrics["total"].backward()
        # This API retains the existing DDP gradient-average compensation.
        # Bounded accumulation instead finalizes already-reduced statistics
        # with distributed=None and must not call this path again.
        for p in model.parameters():
            dist.all_reduce(p.grad)
            p.grad.div_(2)
        _compare_parameters(model, oracle)
        for name in metrics:
            torch.testing.assert_close(metrics[name], expected[name].detach(), atol=3e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_real_distributed_finalizer_matches_global_values_and_gradients(tmp_path):
    config, raw, _ = _raw_pair(tmp_path)
    batch = materialize_targets(raw, config)
    mp.spawn(_distributed_finalizer, args=(str(tmp_path / "finalizer"), batch), nprocs=2, join=True)


def _reference_single_loss(output, batch):
    """Original per-patch objective, calculated independently from statistics."""
    idx = batch.direction_indices_bzyx
    dirs = output[idx[:, 0], :6, idx[:, 1], idx[:, 2], idx[:, 3]]
    direction = ((dirs - batch.direction_target_sparse).square() * batch.direction_weight_sparse).mean()
    presence = output[:, 6:7].clamp(1e-6, 1 - 1e-6)
    positive_losses, negative_losses = [], []
    for b in range(len(output)):
        selected = idx[idx[:, 0] == b]
        mask = batch.presence_mask[b, 0, selected[:, 1], selected[:, 2], selected[:, 3]]
        values = presence[b, 0, selected[:, 1], selected[:, 2], selected[:, 3]][mask]
        if values.numel():
            positive_losses.append(F.binary_cross_entropy(values.float(), torch.ones_like(values).float()))
        negative = presence[b][batch.presence_mask[b] & (batch.presence_target[b] <= .5)]
        if negative.numel():
            negative_losses.append(F.binary_cross_entropy(negative.float(), torch.zeros_like(negative).float()))
    zero = output.sum() * 0
    pos = torch.stack(positive_losses).mean() if positive_losses else zero
    neg = torch.stack(negative_losses).mean() if negative_losses else zero
    return 2 * direction + 3 * (pos + neg)


class _ConditionedPointwiseHead(torch.nn.Module):
    """Small sample-independent head exercising the real conditioned loss API."""
    conditioned_decoder_enabled = True

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 7, 1)

    def forward(self, volume, *, conditioned_random_query,
                conditioned_point_indices_bzyx, conditioned_point_query_n6, **kwargs):
        logits = self.conv(volume)
        zero = logits.sigmoid()
        random = (logits + conditioned_random_query[:, :1, None, None, None] * .05).sigmoid()
        idx = conditioned_point_indices_bzyx
        point = None
        if idx is not None:
            values = logits[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]
            bias = torch.cat((conditioned_point_query_n6 * .05,
                              values.new_zeros((len(idx), 1))), dim=1)
            point = (values + bias).sigmoid()
        return {"zero_output": zero, "random_output": random, "point_output": point}


def _production_update(tmp_path, device, stride, conditioned):
    config, raw, parts = _raw_pair(tmp_path)
    # Sparse positives masked out in one sample; direction remains supervised.
    raw = replace(raw, valid_mask=raw.valid_mask.clone())
    raw.valid_mask[0, :, :16] = False
    parts = [_slice_batch(raw, i, i + 1) for i in range(2)]
    full = materialize_targets(raw.to(device), config, output_stride=stride)
    torch.manual_seed(31)
    model = _ConditionedPointwiseHead().to(device) if conditioned else _pointwise_fiber_model(device)
    if stride > 1:
        if conditioned:
            class PooledHead(_ConditionedPointwiseHead):
                def forward(self, volume, **kwargs):
                    return super().forward(F.avg_pool3d(volume, stride), **kwargs)
            model = PooledHead().to(device)
        else:
            model = torch.nn.Sequential(torch.nn.AvgPool3d(stride), model)
    oracle = copy.deepcopy(model)
    opts = [torch.optim.AdamW(m.parameters(), lr=.01) for m in (model, oracle)]
    for _ in range(2):
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        if conditioned:
            reported = compute_conditioned_losses(oracle, full, direction_weight=2, presence_weight=3)
            expected = reported["total"]
        else:
            output = oracle(full.volume)
            expected = _reference_single_loss(output, full)
            reported = compute_losses(output, full, direction_weight=2, presence_weight=3)
        torch.testing.assert_close(reported["total"], expected, atol=3e-6, rtol=2e-5)
        expected.backward()
        staging = _prepass(_Source(parts), config, device=device, stride=stride, conditioned=conditioned)
        first = next(staging)
        accumulator = NormalizedGradientAccumulator(model, first.normalization_counts,
            direction_weight=2, presence_weight=3)
        for raw_part in (first, next(staging)):
            part = materialize_targets(raw_part.to(device), config, output_stride=stride)
            stats = (compute_conditioned_losses(model, part, direction_weight=2,
                        presence_weight=3, _return_statistics=True) if conditioned else
                     compute_losses(model(part.volume), part, direction_weight=2,
                        presence_weight=3, _return_statistics=True))
            accumulator.add(stats)
        statistics = accumulator.reduce_and_assign()
        _compare_parameters(model, oracle)
        for opt in opts:
            opt.step()
        _compare_parameters(model, oracle, gradients=False)
        combined = _finalize_loss_statistics(statistics, direction_weight=2, presence_weight=3)
        for name in reported:
            torch.testing.assert_close(combined[name], reported[name].detach(), atol=3e-6, rtol=2e-5)
        staging.close()


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("conditioned", [False, True])
def test_real_production_prepass_loss_and_adam_match_full_batch(tmp_path, stride, conditioned):
    _production_update(tmp_path, torch.device("cpu"), stride, conditioned)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager GPU reservation")
@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("conditioned", [False, True])
def test_cuda_real_production_prepass_loss_and_adam_match_full_batch(tmp_path, stride, conditioned):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    _production_update(tmp_path, torch.device("cuda:0"), stride, conditioned)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager GPU reservation")
@pytest.mark.parametrize("mode", ["all", "blur"])
def test_cuda_real_macro_batch_and_async_suffix_preserve_counts(tmp_path, mode):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    loader = _real_paris4_loader(tmp_path)
    config = loader.config
    owner = FiberTrace3DSampleQueue(sources={"train": SampleQueueSource(config,
        sample_mode="flat", defer_cuda_value_augmentation=mode)},
        workers=2, slot_count=3, initial_in_flight=2)
    view = QueuedBatchView(owner, "train", ((0, 1), (2,)))
    staged = _prepass(view, config, sizes=(2, 1), stride=2, device="cuda:0")
    # Own CPU batches have already released all shared source slots.
    first, second = next(staged), next(staged)
    expected = []
    for batch in (first, second):
        direct = apply_deferred_batch_augmentation(batch.to("cuda:0"), config, mode=mode)
        expected.append(materialize_targets(direct, config, output_stride=2))
    torch.cuda.synchronize()
    pipeline = FiberTrace3DCudaBatchPipeline(_Source([first, second]), config=config,
        device=torch.device("cuda:0"), output_stride=2, value_augmentation_suffix=mode)
    try:
        counts = sum((target_normalization_counts(b) for b in expected), torch.zeros(3, device="cuda:0"))
        for reference in expected:
            actual = next(pipeline)
            assert torch.equal(actual.volume, reference.volume)
            assert torch.equal(actual.presence_mask, reference.presence_mask)
            assert torch.equal(actual.normalization_counts, counts)
            assert actual.normalization_counts.is_cuda
    finally:
        pipeline.close()
        staged.close()
        owner.close()
