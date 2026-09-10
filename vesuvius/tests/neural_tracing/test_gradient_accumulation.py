import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import (
    ComponentGradientAccumulator, effective_batch_partition,
    validate_sample_independent_model,
)
from vesuvius.neural_tracing.fiber_trace_3d.loss_statistics import LossStatistics


def stats(output):
    # Unequal and prediction-selected detached counts, including empty components.
    selected = (output.detach()[:, 0] > 0).float()
    zero = output.sum() * 0
    return LossStatistics(
        direction_sum=output.square().sum(), direction_count=output.new_tensor(output.numel()),
        positive_presence_sum=(output[:, 0].square() * selected).sum(), positive_presence_count=selected.sum(),
        negative_presence_sum=(output[:, 1] - 1).square().sum(), negative_presence_count=output.new_tensor(output.shape[0]),
        angle_sum=zero, angle_count=zero.detach(), branch0_sum=zero, branch1_sum=zero,
        branch_count=zero.detach(), selected_score_sum=zero, selected_score_count=zero.detach())


def loss(s):
    return (2 * s.direction_sum / s.direction_count.clamp_min(1)
            + 3 * s.positive_presence_sum / s.positive_presence_count.clamp_min(1)
            + 0.75 * s.negative_presence_sum / s.negative_presence_count.clamp_min(1))


@pytest.mark.parametrize("microbatch", [1, 3, 8, 16, 128])
def test_gradients_and_adam_match_full_batch(microbatch):
    torch.manual_seed(13)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    oracle = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    baseline_optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.01)
    for _ in range(3):
        inputs = torch.randn(128, 3)
        optimizer.zero_grad(set_to_none=True)
        baseline_optimizer.zero_grad(set_to_none=True)
        expected = stats(oracle(inputs))
        loss(expected).backward()
        accumulator = ComponentGradientAccumulator(model, loss_scale=64)
        for part in inputs.split(microbatch):
            accumulator.add(stats(model(part)))
        actual = accumulator.reduce_and_assign(direction_weight=2, presence_weight=3,
                                               negative_query_weight=0.25)
        torch.testing.assert_close(loss(actual), loss(expected))
        for a, b in zip(model.parameters(), oracle.parameters()):
            a.grad.div_(64)
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.5)
        optimizer.step()
        baseline_optimizer.step()
        for a, b in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-5)


def test_partition_membership_unequal_and_zero_work():
    for world in (1, 2, 6, 8, 17):
        for microbatch in (1, 3, 16):
            ids = [i for rank in range(world) for batch in effective_batch_partition(3, 7, rank, world, microbatch) for i in batch]
            assert ids == list(range(21, 28))


def test_reject_batch_state_and_random_modules():
    for module in (torch.nn.BatchNorm3d(2), torch.nn.SyncBatchNorm(2), torch.nn.Dropout()):
        with pytest.raises(ValueError):
            validate_sample_independent_model(module)


def test_unused_parameters_remain_unused():
    model = torch.nn.Linear(3, 2)
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
    a = ComponentGradientAccumulator(model)
    a.add(stats(model(torch.ones(1, 3))))
    a.reduce_and_assign(direction_weight=1, presence_weight=1)
    assert model.unused.grad is None
    with pytest.raises(RuntimeError):
        a.add(stats(model(torch.ones(1, 3))))


def direction_only(output):
    value = output.float()
    zero = value.sum() * 0
    values = {name: zero for name in LossStatistics.__dataclass_fields__}
    values.update(direction_sum=value.square().sum(), direction_count=value.new_tensor(value.numel()))
    return LossStatistics(**values)


@pytest.mark.parametrize("weight,batch_size", [(0.001, 1), (1000.0, 128)])
def test_cpu_fp16_sparse_underflow_and_local_overflow(weight, batch_size):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    oracle = copy.deepcopy(model)
    inputs = torch.full((batch_size, 1), 0.001)
    with torch.autocast("cpu", dtype=torch.float16):
        expected = direction_only(oracle(inputs))
    (expected.direction_sum / expected.direction_count * 65536).backward()
    assert torch.isfinite(oracle.weight.grad).all()
    assert oracle.weight.grad.abs().max() > 0
    a = ComponentGradientAccumulator(model, loss_scale=65536, protect_fp16=True)
    for part in inputs.split(1):
        with torch.autocast("cpu", dtype=torch.float16):
            s = direction_only(model(part))
        a.add(s)
    a.reduce_and_assign(direction_weight=1, presence_weight=0)
    torch.testing.assert_close(model.weight.grad, oracle.weight.grad, atol=1e-8, rtol=1e-3)
    if weight == 1000:
        assert a.local_scale_retries > 0


def _distributed_worker(rank, world, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        torch.manual_seed(15)
        model = torch.nn.Linear(3, 2)
        a = ComponentGradientAccumulator(model)
        inputs = torch.arange(6).float().reshape(2, 3) / 5
        # Three ranks for two samples intentionally includes a zero-work rank.
        for ids in effective_batch_partition(0, 2, rank, world, 1):
            a.add(stats(model(inputs[list(ids)])))
        result = a.reduce_and_assign(direction_weight=2, presence_weight=3,
                                    negative_query_weight=0.25, distributed=True)
        torch.save({"gradients": [p.grad for p in model.parameters()], "loss": loss(result)},
                   f"{output_dir}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_real_gloo_unequal_and_zero_sample_ranks(tmp_path, monkeypatch):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    mp.spawn(_distributed_worker, args=(3, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=3)
    torch.manual_seed(15)
    model = torch.nn.Linear(3, 2)
    expected = loss(stats(model(torch.arange(6).float().reshape(2, 3) / 5)))
    expected.backward()
    for rank in range(3):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual["loss"], expected)
        for gradient, parameter in zip(actual["gradients"], model.parameters()):
            torch.testing.assert_close(gradient, parameter.grad)


def test_gradient_observer_sees_exact_bank_contributions():
    torch.manual_seed(81)
    model = torch.nn.Linear(3, 2)
    received = {}

    def observe(component, parameter, contribution):
        key = (component, parameter)
        if key not in received:
            received[key] = torch.zeros_like(contribution)
        received[key].add_(contribution)

    accumulator = ComponentGradientAccumulator(model, gradient_observer=observe)
    for inputs in torch.randn(5, 3).split(2):
        accumulator.add(stats(model(inputs)))
    for (component, parameter), value in received.items():
        assert torch.equal(value, accumulator.banks[component][parameter])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_amp_handoff_restores_exact_state(dtype):
    if os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("explicit CUDA test opt-in required")
    from vesuvius.neural_tracing.fiber_trace_3d.training_state import (
        capture_handoff, replica_signature, restore_handoff,
    )
    torch.cuda.set_per_process_memory_fraction(
        4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    device = torch.device("cuda:0")
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=128)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        value = model(torch.ones(4, 3, device=device)).float().square().mean()
    scaler.scale(value).backward()
    scaler.step(optimizer)
    scaler.update()
    progress = {"step": 1, "sample_frontier": 128, "pending_test": {"rows": [[0, 2.0]]}}
    state = capture_handoff(model=model, optimizer=optimizer, grad_scaler=scaler,
                            progress=progress, cuda_device=device)
    restored = torch.nn.Linear(3, 2).to(device)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=1)
    new_scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=2)
    result = restore_handoff(state, model=restored, optimizer=new_optimizer,
                             grad_scaler=new_scaler, cuda_device=device)
    actual = capture_handoff(model=restored, optimizer=new_optimizer, grad_scaler=new_scaler,
                             progress=result, cuda_device=device)
    # Byte-exact model/optimizer/scaler/progress, independently of any subsequent
    # floating-point arithmetic or GPU partitioning.
    assert replica_signature(actual) == replica_signature(state)
    torch.testing.assert_close(actual["rng"]["cuda"], state["rng"]["cuda"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("microbatch", [1, 3, 16])
def test_cuda_amp_accumulation_matches_full_batch(dtype, microbatch):
    if os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("explicit CUDA test opt-in required")
    # Tiny numerical oracle; cap allocator at 4 GiB, well below the user limit.
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.manual_seed(13)
    model = torch.nn.Linear(3, 2).cuda()
    oracle = copy.deepcopy(model)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001) for m in (model, oracle)]
    scalers = [torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=128) for _ in range(2)]
    for _ in range(3):
        inputs = torch.randn(128, 3, device="cuda")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            expected = stats(oracle(inputs).float())
        scalers[1].scale(loss(expected)).backward()
        accumulator = ComponentGradientAccumulator(model, loss_scale=scalers[0].get_scale(),
                                                    protect_fp16=dtype == torch.float16)
        # Initialize the lazy scaler just as the shared trainer does.
        scalers[0].scale(torch.zeros((), device="cuda"))
        for part in inputs.split(microbatch):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                current = stats(model(part).float())
            accumulator.add(current)
        actual = accumulator.reduce_and_assign(direction_weight=2, presence_weight=3,
                                               negative_query_weight=0.25)
        for scaler, optimizer in zip(scalers, optimizers):
            scaler.unscale_(optimizer)
        tolerance = dict(atol=3e-3, rtol=3e-2) if dtype != torch.float32 else dict(atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(loss(actual), loss(expected), **tolerance)
        for a, b in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(a.grad, b.grad, **tolerance)
        for module, optimizer, scaler in zip((model, oracle), optimizers, scalers):
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        assert scalers[0].get_scale() == scalers[1].get_scale()
        for a, b in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(a, b, **tolerance)
