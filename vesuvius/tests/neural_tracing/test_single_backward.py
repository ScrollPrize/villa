"""Target-count prepass and single-backward regression oracles."""
import copy
import os
import time
from statistics import mean, median
from dataclasses import replace

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator, validate_target_only_loss_config
from vesuvius.neural_tracing.fiber_trace_3d.loss_statistics import LossStatistics
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses
from test_arc_sampling import _gaussian_line_batch, make_loader


@pytest.mark.parametrize("raw", [{"model_3d": {"direction_branch_count": 2}}, {"model": {"output_channels": 21}}])
def test_unsupported_losses_fail_early(raw):
    with pytest.raises(ValueError, match="TODO.*prediction-dependent"):
        validate_target_only_loss_config(raw)


@pytest.mark.parametrize("embedding,channels", [(False, 14)])
def test_unsupported_losses_fail_in_evaluation(embedding, channels):
    with pytest.raises(ValueError, match="temporarily unsupported"):
        compute_losses(torch.zeros(1, channels, 1, 1, 1), None,
                       embeddings=torch.ones(1) if embedding else None,
                       direction_weight=1, presence_weight=1)


def test_conditioned_target_counts():
    loader = make_loader(dense=True, balanced=True)
    batch = loader.load_batch(0)
    source = MacroBatchTargetSource(None, config=loader.config, device=torch.device("cpu"),
        output_stride=4, batch_sizes=[1], conditioned=True)
    target = materialize_targets(batch, loader.config, output_stride=4)
    counts = source._counts(batch)
    torch.testing.assert_close(counts, torch.tensor([12 * len(target.direction_indices_bzyx),
        2 * torch.unique(target.direction_indices_bzyx[:, 0]).numel(), 2.]))
    validate_target_only_loss_config({"model_3d": {"conditioned_decoder_enabled": True}})


def test_conditioned_single_backward_matches_full_batch():
    from test_fiber_trace_3d import _loader, _load_materialized_batch, _slice_batch
    from vesuvius.neural_tracing.fiber_trace_3d.model import FiberTrace3DNet, FiberTrace3DModelConfig
    from vesuvius.neural_tracing.fiber_trace_3d.train import compute_conditioned_losses
    batch = _load_materialized_batch(_loader(augment_enabled=False), 0)
    torch.manual_seed(31)
    model = FiberTrace3DNet(FiberTrace3DModelConfig(
        conditioned_decoder_enabled=True, conditioned_latent_channels=8,
        conditioned_decoder_hidden_channels=8, conditioned_decoder_layers=2,
        features_per_stage=(4,8), strides=((1,1,1),(2,2,2)), normalization="group"))
    oracle = copy.deepcopy(model)
    options = dict(direction_weight=2, presence_weight=3, positive_query_weight=.7, negative_query_weight=.4)
    compute_conditioned_losses(oracle, batch, **options)["total"].backward()
    counts = torch.tensor([12 * len(batch.direction_indices_bzyx),
        2 * torch.unique(batch.direction_indices_bzyx[:,0]).numel(), 2 * len(batch.volume)])
    accumulator = NormalizedGradientAccumulator(model, counts, **options)
    for i in range(len(batch.volume)):
        accumulator.add(compute_conditioned_losses(model, _slice_batch(batch,i,i+1),
            **options, _return_statistics=True))
    accumulator.reduce_and_assign()
    for actual, expected in zip(model.parameters(), oracle.parameters()):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-4)


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


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("stride", [1, 4])
def test_target_prepass_owns_reused_queue_slots(dense, stride):
    loader = make_loader(dense=dense, balanced=dense)
    batches = [loader.load_batch(i) for i in range(3)]
    class Source:
        def __iter__(self): return self
        def __next__(self):
            self.current = batches.pop(0)
            return self.current
        def release_current(self):
            self.current.volume.fill_(123)
            self.current.target_segment_starts_zyx.fill_(999)
        def close(self): pass
    source = MacroBatchTargetSource(Source(), config=loader.config, device=torch.device("cpu"),
                                   output_stride=stride, batch_sizes=[1,1,1])
    result = [next(source) for _ in range(3)]
    assert all(not (b.volume == 123).any() for b in result)
    expected = torch.zeros(3)
    for batch in result:
        targets = materialize_targets(batch, loader.config, output_stride=stride)
        output = torch.full((1,7,*targets.presence_target.shape[-3:]), .5, requires_grad=True)
        s = compute_losses(output, targets, direction_weight=10, presence_weight=1, _return_statistics=True)
        expected += torch.stack((s.direction_count, s.positive_presence_count, s.negative_presence_count))
    torch.testing.assert_close(result[0].normalization_counts, expected, rtol=0, atol=0)


def test_gaussian_target_prepass_includes_distribution_negative_groups():
    raw, config, targets = _gaussian_line_batch(stride=1)
    source = MacroBatchTargetSource(
        None, config=config, device=torch.device("cpu"), output_stride=1,
        batch_sizes=[1],
    )
    counts = source._counts(raw)
    output = torch.full(
        (1, 7, *targets.presence_target.shape[-3:]),
        0.5,
        requires_grad=True,
    )
    statistics = compute_losses(
        output, targets, direction_weight=10, presence_weight=1,
        _return_statistics=True,
    )
    expected = torch.stack((
        statistics.direction_count,
        statistics.positive_presence_count,
        statistics.negative_presence_count,
        statistics.distribution_presence_count,
    ))
    torch.testing.assert_close(counts, expected, rtol=0, atol=0)


def test_gaussian_single_backward_matches_explicit_three_way_presence_loss():
    torch.manual_seed(23)
    model = torch.nn.Linear(3, 4, bias=False)
    oracle = copy.deepcopy(model)
    inputs = torch.randn(7, 3)

    def gaussian_statistics(output):
        zero = output.sum() * 0.0
        return LossStatistics(
            direction_sum=output[:, 0].square().sum(),
            direction_count=output.new_tensor(output.shape[0]),
            positive_presence_sum=(output[:, 1] - 1.0).square().sum(),
            positive_presence_count=output.new_tensor(output.shape[0]),
            negative_presence_sum=output[:, 2].square().sum(),
            negative_presence_count=output.new_tensor(output.shape[0]),
            distribution_presence_sum=output[:, 3].square().sum(),
            distribution_presence_count=output.new_tensor(output.shape[0]),
            angle_sum=zero,
            angle_count=zero.detach(),
            branch0_sum=zero,
            branch1_sum=zero,
            branch_count=zero.detach(),
            selected_score_sum=zero,
            selected_score_count=zero.detach(),
        )

    expected = gaussian_statistics(oracle(inputs))
    (
        2.0 * expected.direction_sum / expected.direction_count
        + 3.0 * expected.positive_presence_sum / expected.positive_presence_count
        + 1.5 * expected.negative_presence_sum / expected.negative_presence_count
        + 1.5 * expected.distribution_presence_sum
        / expected.distribution_presence_count
    ).backward()

    accumulator = NormalizedGradientAccumulator(
        model,
        torch.tensor([7.0, 7.0, 7.0, 7.0]),
        direction_weight=2.0,
        presence_weight=3.0,
        gaussian_presence=True,
    )
    for part in inputs.split(2):
        accumulator.add(gaussian_statistics(model(part)))
    accumulator.reduce_and_assign()
    for actual, reference in zip(model.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad)


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
    from test_gradient_accumulation import direction_only
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


def test_cuda_dense_single_backward_equivalence_and_speed():
    if os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("requires manager-authorized CUDA test launcher")
    from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import ComponentGradientAccumulator
    from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
    from pathlib import Path
    import json
    torch.cuda.set_per_process_memory_fraction(16*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.manual_seed(18)
    cfg = json.loads((Path(__file__).parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_16x8.json").read_text())
    loader = make_loader()
    loader.config = replace(loader.config, patch_shape_zyx=(128,128,128), batch_size=4)
    raw = loader.load_batch(0)
    raw.volume.copy_(torch.randn_like(raw.volume))
    batch = materialize_targets(raw.to("cuda:0"), loader.config, output_stride=4)
    counts = torch.tensor([batch.direction_indices_bzyx.shape[0]*6,
                           torch.unique(batch.direction_indices_bzyx[:,0]).numel(),
                           (batch.presence_target<=.5).flatten(1).any(dim=1).sum()], device="cuda")
    model = build_fiber_trace_3d_model(cfg).cuda()
    result = {}
    gradients = {}
    for mode in ("component", "single"):
        durations = []
        for iteration in range(15):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            a = (ComponentGradientAccumulator(model) if mode=="component" else
                 NormalizedGradientAccumulator(model, counts, direction_weight=10, presence_weight=1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                stats = compute_losses(model(batch.volume), batch, direction_weight=10,
                                       presence_weight=1, _return_statistics=True)
            a.add(stats)
            if mode=="component": a.reduce_and_assign(direction_weight=10, presence_weight=1)
            else: a.reduce_and_assign()
            del stats, a
            torch.cuda.synchronize()
            if iteration>=5: durations.append(time.perf_counter()-start)
        gradients[mode] = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        result[mode] = {"samples_per_second_1gpu": 4/mean(durations),
                        "mean_ms": mean(durations)*1000, "p50_ms":median(durations)*1000,
                        "min_ms":min(durations)*1000, "max_ms":max(durations)*1000}
    # Separate BF16 component VJPs have different rounding from a combined VJP.
    relative = (gradients['single']-gradients['component']).norm()/gradients['component'].norm()
    assert relative < .02, relative.item()
    print("SINGLE_BACKWARD_PROFILE", json.dumps({"device_count":1, "batch_per_gpu":4,
          "patch":128, "stride":4, "precision":"bf16", "warmup":5, "measured":10,
          "gradient_relative_l2":relative.item(), "results":result}), flush=True)
