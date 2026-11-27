"""The flat block loop accumulates batch k while the device runs batch k+1.

`run_block_inference` defers each batch's accumulation by one batch (on CUDA the
probabilities come back through two reused host buffers). The accumulator must still
see the original calls, in the original order, with the original tile values: skips
of batch k, adds of batch k, skips of batch k+1, ...
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from torch import nn

from vesuvius.ink_detection.inference.infer import run_block_inference

PATCH = 8
# GPU spin in each CUDA forward, so a batch's device-to-host copy is still pending
# when the loop moves on to the next batch (about 10 ms on a laptop GPU).
SPIN_CYCLES = 20_000_000

# (occupancy per block, first block value): whole, empty, partial, partial, whole.
BATCHES = [([1, 1], 0.0), ([0, 0], 2.0), ([1, 0], 4.0), ([0, 1], 6.0), ([1, 1], 8.0)]


def _loader():
    x0 = 0
    for occupancy, first_value in BATCHES:
        count = len(occupancy)
        values = first_value + torch.arange(count, dtype=torch.float32)
        images = values.view(count, 1, 1, 1, 1).expand(count, 1, 2, PATCH, PATCH).contiguous()
        metadata = torch.tensor(
            [[0, x0 + i * PATCH, PATCH, PATCH, int(o)] for i, o in enumerate(occupancy)]
        )
        x0 += count * PATCH
        yield images, metadata


class _ValueLogits(nn.Module):
    """Logit of every pixel = the block's value (scaled), so tiles are distinct."""

    def forward(self, images):
        if images.is_cuda:
            torch.cuda._sleep(SPIN_CYCLES)
        values = images[:, 0, 0, :1, :1] / 10.0
        return values.expand(images.shape[0], PATCH, PATCH).unsqueeze(1).contiguous()


class _RecordingAccumulator:
    def __init__(self):
        self.calls = []

    def add_tile(self, *, y0, x0, tile, tile_weights):
        self.calls.append(("add", y0, x0, tile.copy()))

    def skip_tile(self, *, y0, x0, valid_h, valid_w):
        self.calls.append(("skip", y0, x0))


def _expected_calls(device):
    calls, x0 = [], 0
    for occupancy, first_value in BATCHES:
        blocks = [(x0 + i * PATCH, first_value + i, o) for i, o in enumerate(occupancy)]
        calls += [("skip", 0, bx) for bx, _, o in blocks if not o]
        for bx, value, o in blocks:
            if o:
                logit = torch.full((PATCH, PATCH), value, device=device) / 10.0
                calls.append(("add", 0, bx, logit.sigmoid().cpu().numpy()))
        x0 += len(occupancy) * PATCH
    return calls


def _run(device):
    accumulator = _RecordingAccumulator()
    run_block_inference(
        loader=_loader(),
        model=_ValueLogits(),
        accumulator=accumulator,
        weight_map=np.ones((PATCH, PATCH), dtype=np.float32),
        mask=None,
        device=device,
        amp_dtype=None,
        tta_axes=(),
        tta_batch_size=None,
    )
    return accumulator.calls


def _assert_same_calls(actual, expected):
    assert [call[:3] for call in actual] == [call[:3] for call in expected]
    for got, want in zip(actual, expected):
        if got[0] == "add":
            np.testing.assert_array_equal(got[3], want[3])


def test_deferred_accumulation_keeps_call_order_and_values_on_cpu():
    _assert_same_calls(_run(torch.device("cpu")), _expected_calls(torch.device("cpu")))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


@cuda
def test_deferred_accumulation_keeps_call_order_and_values_on_cuda():
    _assert_same_calls(_run(torch.device("cuda")), _expected_calls(torch.device("cuda")))


@cuda
def test_unavailable_pinned_memory_falls_back_to_pageable_copies(monkeypatch, caplog):
    empty = torch.empty

    def empty_without_pinning(*args, pin_memory=False, **kwargs):
        if pin_memory:
            raise RuntimeError("CUDA error: out of memory")
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_without_pinning)
    with caplog.at_level(logging.WARNING):
        calls = _run(torch.device("cuda"))
    _assert_same_calls(calls, _expected_calls(torch.device("cuda")))
    warnings = [r for r in caplog.records if "Pinned host memory unavailable" in r.message]
    assert len(warnings) == 1
