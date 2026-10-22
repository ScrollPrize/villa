"""Exercise the extracted production trainer on real Paris4 CT and annotation."""
import copy
from dataclasses import asdict, replace
from datetime import timedelta
import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader, _assert_same_batch
from vesuvius.neural_tracing.fiber_trace_3d import train
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DLoader, config_from_mapping, collate_fiber_trace_3d_samples,
    prepare_fiber_trace_3d_cuda_suffix,
)
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import apply_deferred_batch_augmentation
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.training_state import cpu_clone


def _config(tmp_path, *, device="cpu", depth=0, mode=None, start=0):
    raw = asdict(_real_paris4_loader(tmp_path).config)
    raw.pop("config_dir")
    # output_stride is derived from the model, not an embedding setting.
    raw["embedding"].pop("output_stride")
    raw["test_datasets"] = copy.deepcopy(raw["datasets"])
    raw["model"] = {"features_per_stage": [2, 4], "normalization": "none"}
    raw["training"] = {
        "device": device, "max_steps": start + 4, "run_path": str(tmp_path / "runs"),
        "run_name": "queue", "run_datestr": "fixed", "tensorboard_enabled": True,
        "learning_rate": .003, "weight_decay": .125, "scalar_log_interval": 1,
        "test_interval": 2, "checkpoint_interval": 2, "kept_snapshot_interval": 0,
        "test_control_points": 3, "test_start_sample_index": 5, "test_trace2cp_enabled": False,
        "sample_vis_interval": 2, "test_sample_vis_interval": 2, "sample_vis_count": 1,
        "sample_pipeline_workers": 2, "sample_pipeline_slots": 3,
        "sample_pipeline_initial_in_flight": 2,
        "sample_pipeline_cuda_prefetch_depth": depth,
    }
    if mode is not None:
        raw["training"]["sample_pipeline_cuda_value_augmentation"] = mode
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return raw, path


class _Writer:
    def __init__(self):
        self.scalars = {}
        self.images = []
        self.closed = False
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.scalars[tag, step] = float(value)

    def add_text(self, *args, **kwargs):
        pass

    def add_image(self, tag, image, step, **kwargs):
        self.images.append((tag, step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


def _observe_queue(monkeypatch):
    queues = []
    original = train.FiberTrace3DSampleQueue

    def make(*args, **kwargs):
        queue = original(*args, **kwargs)
        queues.append(queue)
        queue.observed_processes = tuple(queue.pipeline.processes)
        queue.observed_pids = tuple(process.pid for process in queue.observed_processes)
        queue.admissions = []
        submit = queue.submit

        def record(source, indices):
            ticket = submit(source, indices)
            if ticket is not None:
                queue.admissions.append((source, tuple(indices)))
            return ticket

        queue.submit = record
        return queue

    monkeypatch.setattr(train, "FiberTrace3DSampleQueue", make)
    # A production run must no longer instantiate either legacy whole-batch
    # loader or the serial fallback, even with loader_workers=0.
    monkeypatch.setattr(train, "_make_batch_dataloader", lambda *a, **k: pytest.fail("legacy loader"))
    return queues


def _assert_workers_exited(queue):
    for pid in queue.observed_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _serial_batch(loader, index, device, mode):
    if mode is None:
        return materialize_targets(loader.load_batch(index).to(device), loader.config)
    raws = [prepare_fiber_trace_3d_cuda_suffix(loader.config,
        loader.load_raw_sample(index + offset), mode=mode)
        for offset in range(loader.config.batch_size)]
    batch = collate_fiber_trace_3d_samples([raw.sample for raw in raws],
        deferred_augmentation_params=tuple(raw.augmentation for raw in raws),
        deferred_blur_kernels=tuple(raw.blur_kernels for raw in raws))
    batch = apply_deferred_batch_augmentation(batch.to(device), loader.config, mode=mode)
    return materialize_targets(batch, loader.config)


def _serial_test_rows(model, loader, options):
    size = loader.config.batch_size
    count = options["sample_count"]
    start = options["start_sample_index"]
    # The native geometry backend is CPU-only. Keep the serial oracle on that
    # same CPU backend, then transfer; do not compare to GPU coord generation.
    source = (loader.load_batch(start + consumed)
              for consumed in range(0, count, size))
    return train._evaluate_dense_loss_rows(model, loader, batch_iterator=source, **options)


def _trainer_equivalence(tmp_path, monkeypatch, *, device="cpu", depth=0, mode=None, start=0):
    raw, path = _config(tmp_path, device=device, depth=depth, mode=mode, start=start)
    queues = _observe_queue(monkeypatch)
    writer = _Writer()
    monkeypatch.setattr(train, "_make_summary_writer", lambda *a, **k: writer)
    loader = FiberTrace3DLoader(config_from_mapping(raw))
    test_loader = FiberTrace3DLoader(config_from_mapping(
        train._make_test_loader_raw_config(raw, raw["training"])))
    dev = torch.device(device)
    torch.manual_seed(151)
    reference = train.build_fiber_trace_3d_model(raw).to(dev)
    initial = cpu_clone(reference.state_dict())
    hparams = train._optimizer_hparams_from_training(raw["training"])
    optimizer = torch.optim.AdamW(reference.parameters(), **hparams)
    resume = None
    if start:
        resume = tmp_path / "resume.pt"
        train._save_snapshot(resume, model=reference, optimizer=optimizer,
            step=start, config=raw, metric=None, metric_name=None,
            sample_progress=train._SampleProgress(0, 0, loader.config.batch_size))

    # Serial oracle calls the same production losses, model and target builder,
    # but obtains each real augmented crop directly, as before queue integration.
    expected_rows = {}
    test_options = dict(device=dev, start_sample_index=5, sample_count=3,
        direction_weight=10., presence_weight=1.)
    expected_rows[start] = _serial_test_rows(reference, test_loader, test_options)
    expected_train, expected_batches = {}, []
    for step in range(start + 1, start + 5):
        reference.train()
        batch = _serial_batch(loader, (step - 1) * loader.config.batch_size, dev, mode)
        expected_batches.append(batch.to("cpu"))
        optimizer.zero_grad(set_to_none=True)
        losses = train._forward_loss_tensors(reference, batch, direction_weight=10.,
            presence_weight=1., training_loss=True)
        losses["total"].backward()
        optimizer.step()
        expected_train[step] = float(losses["total"].detach())
        if step % 2 == 0:
            expected_rows[step] = _serial_test_rows(reference, test_loader, test_options)

    rows, observed_batches = {}, []
    evaluate = train._evaluate_dense_loss_rows
    forward = train._forward_loss_statistics

    def check_input(model, batch, **kwargs):
        if kwargs.get("training_loss", True):
            _assert_same_batch(batch.to("cpu"), expected_batches[len(observed_batches)])
            observed_batches.append(tuple(batch.stream_indices.tolist()))
        return forward(model, batch, **kwargs)

    def capture(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        rows[kwargs["diagnostic_step"]] = result
        return result

    monkeypatch.setattr(train, "_evaluate_dense_loss_rows", capture)
    monkeypatch.setattr(train, "_forward_loss_statistics", check_input)
    # Init consumes exactly the same RNG as the serial oracle; input sampling
    # has its own canonical per-sample seeds and cannot depend on worker order.
    torch.manual_seed(151)
    train.run_training(path, resume_checkpoint=resume, unmanaged=True)
    assert writer.closed and writer.flushes
    assert len(observed_batches) == 4
    assert rows.keys() == expected_rows.keys()
    for step, result in rows.items():
        for (index, actual), (ref_index, expected) in zip(result, expected_rows[step], strict=True):
            assert index == ref_index
            assert actual == pytest.approx(expected, rel=3e-5, abs=3e-6)
    for step, loss in expected_train.items():
        assert writer.scalars["train/loss_total", step * loader.config.batch_size] == pytest.approx(loss, rel=3e-5, abs=3e-6)
    state = torch.load(tmp_path / "runs/queue_fixed/snapshots/current.pt", weights_only=False,
                       map_location="cpu")
    assert state["step"] == start + 4
    assert state["model"].keys() == initial.keys()
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(state["model"][key], expected.cpu(), rtol=3e-5, atol=3e-6)
    for actual, expected in zip(state["optimizer"]["state"].values(),
                                optimizer.state_dict()["state"].values(), strict=True):
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value.cpu(), rtol=3e-5, atol=3e-6)
    sheets = [("test_sample_3d/principal_slices", start)] + (
        [("train_sample_3d/principal_slices", 1)] if start == 0 else []) + [
        ("train_sample_3d/principal_slices", start + 2),
        ("test_sample_3d/principal_slices", start + 2),
        ("train_sample_3d/principal_slices", start + 4),
        ("test_sample_3d/principal_slices", start + 4)]
    assert writer.images == [(tag, step * loader.config.batch_size) for tag, step in sheets]
    assert len(queues) == 1
    queue = queues[0]
    assert len(queue.observed_processes) == 2
    _assert_workers_exited(queue)
    assert not queue._owned and not queue._ready
    assert queue.admissions == [
        ("test", (5, 6)), ("test", (7,)),
        ("train", (start * 2, start * 2 + 1)),
        ("train", (start * 2 + 2, start * 2 + 3)),
        ("test", (5, 6)), ("test", (7,)),
        ("train", (start * 2 + 4, start * 2 + 5)),
        ("train", (start * 2 + 6, start * 2 + 7)),
        ("test", (5, 6)), ("test", (7,)),
    ]


@pytest.mark.parametrize("start", [0, 6])
def test_real_paris4_production_training_matches_serial_and_reuses_test_workers(tmp_path, monkeypatch, start):
    _trainer_equivalence(tmp_path, monkeypatch, start=start)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager GPU reservation")
@pytest.mark.parametrize("mode", [None, "blur", "all"])
def test_cuda_real_production_async_training_matches_serial(tmp_path, monkeypatch, mode):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    # Fix convolution algorithm selection for the oracle; production settings
    # are not changed. Both paths must see exactly identical input tensors.
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        _trainer_equivalence(tmp_path, monkeypatch, device="cuda:0", depth=2, mode=mode, start=6)


@pytest.mark.parametrize("failure", ["initial", "periodic", "train"])
def test_real_paris4_trainer_failure_closes_shared_workers_and_writer(tmp_path, monkeypatch, failure):
    _, path = _config(tmp_path)
    queues = _observe_queue(monkeypatch)
    writer = _Writer()
    monkeypatch.setattr(train, "_make_summary_writer", lambda *a, **k: writer)
    original = train._forward_loss_statistics

    def fail(model, batch, **kwargs):
        training_loss = kwargs.get("training_loss", True)
        if ((failure == "train" and training_loss)
                or (failure == "initial" and not training_loss)
                or (failure == "periodic" and not training_loss
                    and "train/loss_total" in {tag for tag, _ in writer.scalars})):
            raise RuntimeError("injected trainer failure")
        return original(model, batch, **kwargs)

    monkeypatch.setattr(train, "_forward_loss_statistics", fail)
    with pytest.raises(RuntimeError, match="injected trainer failure"):
        train.run_training(path, unmanaged=True)
    assert writer.closed
    assert len(queues) == 1
    assert not queues[0]._owned
    _assert_workers_exited(queues[0])


@pytest.mark.parametrize("prepare", [False, True])
def test_real_paris4_cpu_transfer_owns_storage_before_source_release(tmp_path, prepare):
    loader = _real_paris4_loader(tmp_path)
    original = loader.load_batch(5)
    owned = replace(original, **{name: value.clone() for name, value in vars(original).items()
                                if isinstance(value, torch.Tensor)})
    expected = materialize_targets(owned, loader.config) if prepare else owned

    class PoisonOnRelease:
        def __next__(self):
            return original

        def release_current(self):
            for value in vars(original).values():
                if isinstance(value, torch.Tensor):
                    value.zero_()

    batch, *_ = train._next_training_batch(iterator=PoisonOnRelease(), loader=loader,
        sample_index=5, sample_mode="random", device=torch.device("cpu"), prepare_targets=prepare)
    _assert_same_batch(batch, expected)


def _distributed_worker(rank, path, rendezvous, device):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    torch.manual_seed(151)

    def initialize(config, *, init_method=None):
        if config.device.type == "cuda":
            torch.cuda.set_device(config.device)
        dist.init_process_group(config.backend, init_method=rendezvous, rank=rank,
                                world_size=2, timeout=timedelta(seconds=60))

    train._distributed_init = initialize
    train.run_training(path, unmanaged=True)


def _distributed_equivalence(tmp_path, device, test_count):
    raw, path = _config(tmp_path, device=device, depth=2 if device == "cuda" else 0)
    raw["training"].update(tensorboard_enabled=False, test_control_points=test_count)
    path.write_text(json.dumps(raw))
    loader = FiberTrace3DLoader(config_from_mapping(raw))
    torch.manual_seed(151)
    reference_device = torch.device("cuda:0" if device == "cuda" else "cpu")
    reference = train.build_fiber_trace_3d_model(raw).to(reference_device)
    optimizer = torch.optim.AdamW(reference.parameters(),
        **train._optimizer_hparams_from_training(raw["training"]))
    # Preserve the current DDP objective: mean the two rank-local losses. The
    # global target-only accumulation objective is a separate extraction stage.
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        for rank in range(2):
            batch = _serial_batch(loader, (step * 2 + rank) * 2, reference_device, None)
            loss = train._forward_loss_tensors(reference, batch, direction_weight=10.,
                presence_weight=1., training_loss=True)["total"]
            (loss / 2).backward()
        optimizer.step()
    mp.spawn(_distributed_worker, args=(str(path), (tmp_path / "rendezvous").as_uri(), device),
             nprocs=2, join=True)
    state = torch.load(tmp_path / "runs/queue_fixed/snapshots/current.pt", weights_only=False,
                       map_location="cpu")
    assert state["step"] == 4
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(state["model"][key], expected.cpu(), rtol=8e-5, atol=8e-6)


@pytest.mark.parametrize("test_count", [1, 3])
def test_real_paris4_distributed_trainer_handles_partial_and_empty_test_ranks(tmp_path, test_count):
    _distributed_equivalence(tmp_path, "cpu", test_count)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager GPU reservation")
@pytest.mark.parametrize("test_count", [1, 3])
def test_cuda_real_distributed_queued_training_matches_previous_ddp_objective(tmp_path, test_count):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    assert torch.cuda.device_count() >= 2
    _distributed_equivalence(tmp_path, "cuda", test_count)


def test_real_paris4_queue_requires_finite_steps_before_starting_workers(tmp_path, monkeypatch):
    raw, path = _config(tmp_path)
    raw.pop("test_datasets")
    raw["training"].update(max_steps=0, test_interval=100, checkpoint_interval=100,
                           tensorboard_enabled=False)
    path.write_text(json.dumps(raw))
    queues = _observe_queue(monkeypatch)
    with pytest.raises(ValueError, match="finite training.max_steps"):
        train.run_training(path, unmanaged=True)
    assert not queues


@pytest.mark.parametrize("total,world,expected", [(8, 3, [3, 3, 2]), (2, 4, [1, 1, 0, 0])])
def test_worker_partition(total, world, expected):
    assert [train._rank_partition(total, world_size=world, rank=rank) for rank in range(world)] == expected


@pytest.mark.parametrize("key,value,match", [
    ("sample_pipeline_cuda_prefetch_depth", -1, "nonnegative"),
    ("sample_pipeline_cuda_prefetch_depth", 1, "CUDA training device"),
    ("sample_pipeline_cuda_value_augmentation", "invalid", "blur or all"),
    ("sample_pipeline_cuda_value_augmentation", "all", "prefetch_depth"),
])
def test_invalid_preparation_options_fail_before_worker_start(tmp_path, monkeypatch, key, value, match):
    raw = {"training": {"device": "cpu", "sample_pipeline_workers": 1, key: value}}
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(raw))
    monkeypatch.setattr(train, "load_config", lambda _: None)
    monkeypatch.setattr(train, "FiberTrace3DLoader", lambda _: pytest.fail("started loader"))
    with pytest.raises(ValueError, match=match):
        train.run_training(config, unmanaged=True)
