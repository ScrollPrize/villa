"""Extracted one-backward oracles plus canonical batch and real CT checks."""
import copy
import os
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import (
    NormalizedGradientAccumulator, effective_batch_partition,
    validate_sample_independent_model, validate_target_only_loss_config,
)
from vesuvius.neural_tracing.fiber_trace_3d.loss_statistics import LossStatistics
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import VirtualBatchSampleIndex
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets, target_normalization_counts
from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses, _slice_batch


@pytest.mark.parametrize("raw", [{"model_3d": {"direction_branch_count": 2}}, {"model": {"output_channels": 21}}])
def test_unsupported_losses_fail_early(raw):
    with pytest.raises(ValueError, match="TODO.*prediction-dependent"):
        validate_target_only_loss_config(raw)


def statistics(output, inputs):
    zero = output.sum() * 0
    selected = (inputs[:, 0] > 0).float()
    values = {name: zero for name in LossStatistics.__dataclass_fields__}
    values.update(direction_sum=output.square().sum(), direction_count=output.new_tensor(output.numel()),
                  positive_presence_sum=(output[:, 0].square() * selected).sum(), positive_presence_count=selected.sum(),
                  negative_presence_sum=(output[:, 1] - 1).square().sum(), negative_presence_count=output.new_tensor(len(output)))
    values.update(positive_embedding_sum=zero.detach(), negative_embedding_sum=zero.detach())
    return LossStatistics(**values)


@pytest.mark.parametrize("microbatch", [1, 3, 16, 128])
def test_single_backward_matches_global_loss(microbatch, monkeypatch):
    torch.manual_seed(18)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    oracle = copy.deepcopy(model)
    inputs = torch.randn(128, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    other_optimizer = torch.optim.AdamW(oracle.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        other_optimizer.zero_grad(set_to_none=True)
        s = statistics(oracle(inputs), inputs)
        (2*s.direction_sum/s.direction_count + 3*s.positive_presence_sum/s.positive_presence_count
         + .75*s.negative_presence_sum/s.negative_presence_count).backward()
        calls = []
        original = torch.Tensor.backward
        with monkeypatch.context() as patch:
            def backward(tensor, *args, **kwargs):
                calls.append(1)
                return original(tensor, *args, **kwargs)
            patch.setattr(torch.Tensor, "backward", backward)
            a = NormalizedGradientAccumulator(model, torch.tensor([256, (inputs[:,0]>0).sum(), 128]),
                                              direction_weight=2, presence_weight=3, negative_query_weight=.25)
            for part in inputs.split(microbatch):
                a.add(statistics(model(part), part))
            a.reduce_and_assign()
        assert len(calls) == len(inputs.split(microbatch))
        for p, q in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-5)
        optimizer.step()
        other_optimizer.step()
        for p, q in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(p, q, atol=3e-6, rtol=2e-5)


def direction_only(output):
    value = output.float()
    zero = value.sum() * 0
    values = {name: zero for name in LossStatistics.__dataclass_fields__ if "embedding" not in name}
    values.update(direction_sum=value.square().sum(), direction_count=value.new_tensor(value.numel()))
    return LossStatistics(**values)


def test_changed_counts_fail_before_optimizer_step():
    model = torch.nn.Linear(3,2)
    a = NormalizedGradientAccumulator(model, torch.zeros(3), direction_weight=1, presence_weight=1)
    x = torch.ones(1,3)
    a.add(statistics(model(x), x))
    with pytest.raises(RuntimeError, match="counts changed"):
        a.reduce_and_assign()


def _distributed_single(rank, init_file):
    import torch.distributed as dist
    from datetime import timedelta
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+init_file, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(5)
        model = torch.nn.Linear(3,2)
        model.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
        oracle = copy.deepcopy(model)
        x = torch.ones(1,3)
        s = statistics(oracle(x), x)
        (s.direction_sum/2 + s.positive_presence_sum + s.negative_presence_sum).backward()
        a = NormalizedGradientAccumulator(model, torch.tensor([2,1,1]) if rank==0 else torch.zeros(3),
                                          direction_weight=1, presence_weight=1, distributed=True)
        if rank==0: a.add(statistics(model(x), x))
        a.reduce_and_assign()
        for p,q in zip(model.parameters(),oracle.parameters()):
            if q.grad is None: assert p.grad is None
            else: torch.testing.assert_close(p.grad,q.grad)
    finally:
        dist.destroy_process_group()


def test_distributed_single_backward_zero_work_rank(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_single, args=(str(tmp_path/'gloo'),), nprocs=2, join=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_backward_scaled_amp(dtype):
    model = torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): model.weight.fill_(.001)
    oracle = copy.deepcopy(model)
    inputs = torch.full((16,1), .001)
    with torch.autocast("cpu", dtype=dtype):
        s = direction_only(oracle(inputs))
    (s.direction_sum/s.direction_count*65536).backward()
    a = NormalizedGradientAccumulator(model, torch.tensor([16,0,0]),
                                      direction_weight=1,presence_weight=0,loss_scale=65536)
    for part in inputs.split(1):
        with torch.autocast("cpu", dtype=dtype): s = direction_only(model(part))
        a.add(s)
    a.reduce_and_assign()
    assert torch.isfinite(model.weight.grad).all() and model.weight.grad.abs().max()>0
    torch.testing.assert_close(model.weight.grad,oracle.weight.grad,atol=1e-8,rtol=.01)


@pytest.mark.parametrize("world", [1, 2, 6, 8, 17])
@pytest.mark.parametrize("microbatch", [1, 3, 16])
def test_partition_membership_unequal_and_zero_work(world, microbatch):
    ids = [i for rank in range(world)
           for batch in effective_batch_partition(3, 7, rank, world, microbatch) for i in batch]
    assert ids == list(range(21, 28))


@pytest.mark.parametrize("world", [1, 2, 6, 8])
def test_virtual_mapping_covers_eight_logical_batches_on_fewer_ranks(world):
    for step in (0, 1, 7):
        ids = []
        for rank in range(world):
            mapping = VirtualBatchSampleIndex(batch_size=16, physical_world_size=world,
                rank=rank, virtual_world_size=8, start_step=step)
            ids.extend(mapping(i) for i in range(mapping.local_batches_per_step * 16))
        assert sorted(ids) == list(range(step * 128, (step + 1) * 128))


@pytest.mark.parametrize("values", [(-1, 128, 0, 2, 1), (0, 0, 0, 2, 1),
    (0, 128, 2, 2, 1), (0, 128, 0, 0, 1), (0, 128, 0, 2, 0), (True, 128, 0, 2, 1)])
def test_partition_rejects_invalid_inputs(values):
    with pytest.raises(ValueError, match="invalid"):
        effective_batch_partition(*values)


@pytest.mark.parametrize("module", [torch.nn.BatchNorm3d(2), torch.nn.SyncBatchNorm(2),
    torch.nn.Dropout(), torch.nn.RNN(3, 2)])
def test_reject_sample_dependent_modules(module):
    with pytest.raises(ValueError, match="rejects"):
        validate_sample_independent_model(module)


def test_statistics_pack_add_and_detach_preserve_all_fields():
    output = torch.randn(5, 2, requires_grad=True)
    s = statistics(output, torch.ones(5, 3))
    round_trip = LossStatistics.unpack(s.packed())
    detached = s.detached()
    doubled = detached + detached
    for name in LossStatistics.__dataclass_fields__:
        assert not getattr(round_trip, name).requires_grad
        assert torch.equal(getattr(round_trip, name), getattr(detached, name))
        assert torch.equal(getattr(doubled, name), 2 * getattr(detached, name))
    with pytest.raises(ValueError, match="incorrect loss statistics size"):
        LossStatistics.unpack(torch.zeros(2))


def test_add_has_one_backward_no_collective_and_unused_parameters_stay_none(monkeypatch):
    model = torch.nn.Linear(3, 2)
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
    accumulator = NormalizedGradientAccumulator(model, torch.tensor([2, 1, 1]),
        direction_weight=1, presence_weight=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("add must not communicate")
    monkeypatch.setattr(dist, "all_reduce", forbidden)
    accumulator.add(statistics(model(torch.ones(1, 3)), torch.ones(1, 3)))
    accumulator.reduce_and_assign()
    assert model.unused.grad is None
    with pytest.raises(RuntimeError, match="finalized"):
        accumulator.reduce_and_assign()
    with pytest.raises(RuntimeError, match="finalized"):
        accumulator.add(statistics(model(torch.ones(1, 3)), torch.ones(1, 3)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reject_non_fp32_master_parameters(dtype):
    model = torch.nn.Linear(3, 2).to(dtype=dtype)
    with pytest.raises(ValueError, match="FP32 master"):
        NormalizedGradientAccumulator(model, torch.zeros(3), direction_weight=1, presence_weight=1)


def _distributed_mismatch(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        model = torch.nn.Linear(3, 2)
        counts = torch.tensor([3, 1, 1]) if rank == 0 else torch.tensor([2, 1, 1])
        accumulator = NormalizedGradientAccumulator(model, counts,
            direction_weight=1, presence_weight=1, distributed=True)
        accumulator.add(statistics(model(torch.ones(1, 3)), torch.ones(1, 3)))
        with pytest.raises(RuntimeError, match="counts changed on one or more ranks"):
            accumulator.reduce_and_assign()
    finally:
        dist.destroy_process_group()


def test_distributed_mismatch_fails_on_every_rank_without_hanging(tmp_path):
    mp.spawn(_distributed_mismatch, args=(str(tmp_path / "mismatch"),), nprocs=2, join=True)


def _fiber_counts(batch):
    return target_normalization_counts(batch)


def _fiber_statistics(output, batch):
    return compute_losses(output, batch, direction_weight=1., presence_weight=1.,
                          _return_statistics=True)


def _pointwise_fiber_model(device):
    # A sample-independent seven-channel head keeps this an accumulation test,
    # not an architecture/convergence experiment. It sees real augmented CT.
    return torch.nn.Sequential(torch.nn.Conv3d(1, 7, 1), torch.nn.Sigmoid()).to(device)


def _real_batch(tmp_path, count=2):
    loader = _real_paris4_loader(tmp_path)
    if count != loader.config.batch_size:
        loader = type(loader)(replace(loader.config, batch_size=count))
    return materialize_targets(loader.load_batch(0, sample_mode="flat"), loader.config)


def _compare_parameters(model, oracle, *, gradients=True, atol=3e-6, rtol=2e-5):
    for actual, expected in zip(model.parameters(), oracle.parameters(), strict=True):
        if gradients:
            if expected.grad is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, expected.grad, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_real_paris4_accumulation_matches_full_batch_loss_and_adam(tmp_path):
    batch = _real_batch(tmp_path)
    torch.manual_seed(97)
    model = _pointwise_fiber_model("cpu")
    oracle = copy.deepcopy(model)
    opt, ref_opt = (torch.optim.AdamW(m.parameters(), lr=.01) for m in (model, oracle))
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        full = compute_losses(oracle(batch.volume), batch, direction_weight=2, presence_weight=3)
        full["total"].backward()
        counts = torch.stack([_fiber_counts(_slice_batch(batch, i, i + 1)) for i in range(2)]).sum(0)
        acc = NormalizedGradientAccumulator(model, counts, direction_weight=2, presence_weight=3)
        for i in range(2):
            part = _slice_batch(batch, i, i + 1)
            acc.add(_fiber_statistics(model(part.volume), part))
        acc.reduce_and_assign()
        _compare_parameters(model, oracle)
        opt.step()
        ref_opt.step()
        _compare_parameters(model, oracle, gradients=False)
    print("Paris4: two bounded updates match unchanged full-batch fiber loss/Adam")


def _require_cuda_lease():
    if os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1":
        pytest.skip("requires the manager-authorized CUDA test launcher")
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    assert torch.cuda.is_available()


def test_cuda_real_paris4_microbatches_match_unchanged_fiber_loss(tmp_path):
    _require_cuda_lease()
    batch = _real_batch(tmp_path).to("cuda:0")
    torch.manual_seed(98)
    model = _pointwise_fiber_model("cuda:0")
    oracle = copy.deepcopy(model)
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(1, device="cuda:0")))
    # Register the same unused master parameter after copying for an independent
    # check that reduction does not turn a genuinely unused parameter into zero.
    oracle.register_parameter("unused", torch.nn.Parameter(torch.ones(1, device="cuda:0")))
    opts = [torch.optim.AdamW(m.parameters(), lr=.01) for m in (model, oracle)]
    for _ in range(2):
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        compute_losses(oracle(batch.volume), batch, direction_weight=2, presence_weight=3)["total"].backward()
        parts = [_slice_batch(batch, i, i + 1) for i in range(2)]
        counts = torch.stack([_fiber_counts(part) for part in parts]).sum(0)
        acc = NormalizedGradientAccumulator(model, counts, direction_weight=2, presence_weight=3)
        for part in parts:
            acc.add(_fiber_statistics(model(part.volume), part))
        acc.reduce_and_assign()
        _compare_parameters(model, oracle)
        for opt in opts:
            opt.step()
        _compare_parameters(model, oracle, gradients=False)
    print("CUDA Paris4: unchanged full-batch fiber objective/Adam matches one-sample accumulation")


def _distributed_real_cuda(rank, init_file, cpu_batch):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method="file://" + init_file,
        rank=rank, world_size=2, timeout=timedelta(seconds=30), device_id=device)
    try:
        torch.manual_seed(99)
        model = _pointwise_fiber_model(device)
        model.register_parameter("unused", torch.nn.Parameter(torch.ones(1, device=device)))
        oracle = copy.deepcopy(model)
        full = cpu_batch.to(device)
        parts = [_slice_batch(cpu_batch, ids[0], ids[-1] + 1).to(device)
                 for ids in effective_batch_partition(0, len(cpu_batch.volume), rank, 2, 1)]
        counts = torch.zeros(3, device=device)
        for part in parts:
            counts += _fiber_counts(part)
        opts = [torch.optim.AdamW(m.parameters(), lr=.01) for m in (model, oracle)]
        for _ in range(2):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            compute_losses(oracle(full.volume), full, direction_weight=2, presence_weight=3)["total"].backward()
            acc = NormalizedGradientAccumulator(model, counts, direction_weight=2,
                                               presence_weight=3, distributed=True)
            for part in parts:
                acc.add(_fiber_statistics(model(part.volume), part))
            acc.reduce_and_assign()
            _compare_parameters(model, oracle)
            for opt in opts:
                opt.step()
            _compare_parameters(model, oracle, gradients=False)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("samples", [1, 3])
def test_cuda_real_nccl_unequal_and_zero_work_rank(tmp_path, samples):
    _require_cuda_lease()
    assert torch.cuda.device_count() >= 2, "two manager-allocated devices are required"
    assert dist.is_nccl_available()
    batch = _real_batch(tmp_path, count=samples)
    mp.spawn(_distributed_real_cuda, args=(str(tmp_path / "nccl"), batch), nprocs=2, join=True)
    print(f"NCCL Paris4: {samples} real samples over two ranks match full batch, including unused parameters")
