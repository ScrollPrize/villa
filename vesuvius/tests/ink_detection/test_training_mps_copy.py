"""MPS host-to-device copy safety for ink training.

On torch 2.11 and 2.12, a non_blocking host-to-MPS copy can read its CPU source
after the source has been freed (pytorch/pytorch#189690). The trainer's
Accelerate loader frees each host batch right after queueing its copy, and the
pseudo-label engines convert host inputs to the teacher dtype on the way to the
device, which frees a host temporary as soon as the copy is queued. The teacher
stubs below fill fresh host tensors with a sentinel before the queued copies
run, so a copy that reads freed memory delivers the sentinel instead of the
input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.training import dynamic_labels as labels
from vesuvius.ink_detection.training import train as train_module


_CONFIGS = (
    Path(__file__).parents[2]
    / "src"
    / "vesuvius"
    / "ink_detection"
    / "configs"
)
SENTINEL = -7777.0
CHUNK = 32
BATCH = 4
ROUNDS = 4

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)


class _AcceleratorCaptured(Exception):
    """Stop the trainer once it has built its Accelerator arguments."""


@pytest.mark.parametrize("cuda_available", [False, True])
def test_trainer_requests_non_blocking_loader_copies_only_with_cuda(
    tmp_path: Path, monkeypatch, cuda_available
):
    import accelerate

    authored = json.loads(
        (_CONFIGS / "aligned21_hybrid_3d2d.json").read_text(encoding="utf-8")
    )
    authored["out_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")
    captured = {}

    def capture_accelerator(**kwargs):
        captured.update(kwargs)
        raise _AcceleratorCaptured

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(accelerate, "Accelerator", capture_accelerator)
    with pytest.raises(_AcceleratorCaptured):
        train_module._run_training(
            train_module.stage_training_request(config_path)
        )

    assert captured["dataloader_config"].non_blocking is cuda_available


def _host_sentinels(*shapes):
    """Fresh float32 host tensors that reuse freed blocks and hold SENTINEL."""

    return [
        torch.empty(shape).fill_(SENTINEL) for shape in shapes for _ in range(4)
    ]


class _SentinelTeacher(nn.Module):
    """Overwrite freed host memory, record the input range, emit sure ink."""

    def __init__(self, *scratch_shapes):
        super().__init__()
        self.scratch_shapes = scratch_shapes
        self.scratch = []
        self.ranges = []

    def forward(self, image):
        self.scratch.extend(_host_sentinels(*self.scratch_shapes))
        dims = tuple(range(1, image.ndim))
        ranges = torch.stack([image.amin(dim=dims), image.amax(dim=dims)], dim=1)
        self.ranges.extend(ranges.cpu().tolist())
        return {"ink": torch.full_like(image, 20.0)}


class _UpperHalfDino(nn.Module):
    """Overwrite freed host memory, then match the reference in upper Z only."""

    def __init__(self, *scratch_shapes):
        super().__init__()
        self.scratch_shapes = scratch_shapes
        self.scratch = []

    def forward_features(self, windows):
        self.scratch.extend(_host_sentinels(*self.scratch_shapes))
        token_count = windows[0, 0].numel()
        upper = torch.arange(token_count, device=windows.device) >= token_count // 2
        tokens = torch.stack((upper, ~upper), dim=-1).to(windows.dtype)
        return {"x_norm_patchtokens": tokens.expand(int(windows.shape[0]), -1, -1)}


def _host_sources(source_dtype):
    """Image b holds b + 1 in the source dtype; the foreground mask is all on."""

    shape = (BATCH, 1, CHUNK, CHUNK, CHUNK)
    values = (torch.arange(BATCH) + 1).to(source_dtype)
    image = values.reshape(BATCH, 1, 1, 1, 1).expand(shape).contiguous()
    return image, torch.ones(shape, dtype=source_dtype), values.float().tolist()


def _corrupted_inputs(ranges, values):
    return [
        index
        for index, (low, high) in enumerate(ranges)
        if low != values[index % BATCH] or high != values[index % BATCH]
    ]


def _corrupted_labels(outputs, expected):
    return [
        index
        for index, sample in enumerate(torch.cat(outputs))
        if not torch.equal(sample, expected)
    ]


@requires_mps
@pytest.mark.parametrize(
    "source_dtype", [torch.uint8, torch.bool], ids=["uint8", "bool"]
)
def test_dino_guided_labels_keep_host_inputs_on_mps(source_dtype):
    shape = (BATCH, 1, CHUNK, CHUNK, CHUNK)
    unet = _SentinelTeacher(shape)
    generator = labels.DinoGuidedLabelGenerator(
        unet=unet,
        dino=_UpperHalfDino(shape),
        reference_embedding=torch.tensor([1.0, 0.0]),
        device="mps",
        dtype=torch.float32,
        dino_stride=CHUNK,
        dino_minibatch=BATCH,
        dino_blend_sigma=1000.0,
        threshold=0.5,
        grid=labels.DinoGridSpec(
            chunk_size=CHUNK, window_size=CHUNK, patch_size=1, embedding_dim=2
        ),
    )
    image, mask, values = _host_sources(source_dtype)

    outputs = [
        generator.generate(image, mask_b1zyx=mask).cpu() for _ in range(ROUNDS)
    ]

    expected = torch.zeros((1, CHUNK, CHUNK, CHUNK))
    expected[:, CHUNK // 2 :] = 1.0
    inputs = _corrupted_inputs(unet.ranges, values)
    masked = _corrupted_labels(outputs, expected)
    assert (inputs, masked) == ([], []), (
        f"{len(inputs)} of {len(unet.ranges)} teacher inputs and {len(masked)} "
        f"of {ROUNDS * BATCH} label samples differ from their host batch"
    )


@requires_mps
@pytest.mark.parametrize(
    "source_dtype", [torch.uint8, torch.bool], ids=["uint8", "bool"]
)
def test_self_distill_labels_keep_host_inputs_on_mps(source_dtype):
    sample_shape = (1, 1, CHUNK, CHUNK, CHUNK)
    primary = _SentinelTeacher((BATCH, *sample_shape[1:]), sample_shape)
    generator = labels.SelfDistillLabelGenerator(
        primary=primary,
        ensemble=_SentinelTeacher(),
        primary_threshold=0.5,
        ensemble_threshold=0.5,
        mean_hi=float("inf"),
        std_lo=0.0,
        tta=False,
        device="mps",
        dtype=torch.float32,
        patch_size_zyx=(CHUNK, CHUNK, CHUNK),
    )
    image, mask, values = _host_sources(source_dtype)

    outputs = []
    for _ in range(ROUNDS):
        labels_b1zyx = generator.generate(
            image,
            mask_b1zyx=mask,
            raw_mean=torch.zeros(BATCH),
            raw_std=torch.ones(BATCH),
        )
        # The last sample's mask copy has no later teacher call to overwrite it.
        scratch = _host_sentinels(sample_shape)
        outputs.append(labels_b1zyx.cpu())
        del scratch

    expected = torch.ones((1, CHUNK, CHUNK, CHUNK))
    inputs = _corrupted_inputs(primary.ranges, values)
    masked = _corrupted_labels(outputs, expected)
    assert (inputs, masked) == ([], []), (
        f"{len(inputs)} of {len(primary.ranges)} teacher inputs and {len(masked)} "
        f"of {ROUNDS * BATCH} label samples differ from their host batch"
    )
