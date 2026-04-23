"""Tests for DinoGuidedLabelGenerator helpers and (when available) end-to-end."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from koine_machines.data.dino_guided_labels import (
    _CHUNK_SIZE,
    _DINO_PATCH_SIZE,
    _DINO_WINDOW,
    _TOKENS_PER_CHUNK_AXIS,
    _TOKENS_PER_WINDOW_AXIS,
    _gaussian_window_3d,
    _window_starts,
)


def test_window_starts_stride_64_full_coverage():
    starts = _window_starts(64)
    assert starts == [0, 64, 128]
    last = _CHUNK_SIZE - _DINO_WINDOW
    assert starts[0] == 0 and starts[-1] == last


def test_window_starts_stride_128_two_positions():
    starts = _window_starts(128)
    assert starts == [0, 128]


def test_window_starts_includes_last_when_stride_doesnt_divide():
    starts = _window_starts(96)
    # natural 96-stride positions are 0, 96; we must also include 128 so the
    # chunk's tail (192..255) is covered by the last window.
    assert starts[0] == 0 and starts[-1] == _CHUNK_SIZE - _DINO_WINDOW
    assert all(s + _DINO_WINDOW <= _CHUNK_SIZE for s in starts)


def test_gaussian_window_shape_and_center_max():
    w = _gaussian_window_3d(_TOKENS_PER_WINDOW_AXIS, sigma=4.0)
    assert tuple(w.shape) == (_TOKENS_PER_WINDOW_AXIS,) * 3
    # center cells should be the global maximum
    c = _TOKENS_PER_WINDOW_AXIS // 2
    assert torch.allclose(w.max(), w[c, c, c], atol=0.0) or w[c, c, c] >= w.max() - 1e-6


def test_token_grid_constants_consistent():
    assert _TOKENS_PER_WINDOW_AXIS * _DINO_PATCH_SIZE == _DINO_WINDOW
    assert _TOKENS_PER_CHUNK_AXIS * _DINO_PATCH_SIZE == _CHUNK_SIZE


# ---------------------------------------------------------------------------
# Optional end-to-end test: only runs when DINOGUIDED_E2E=1 in env (and a
# CUDA device + checkpoints are present). It's slow and requires GPU.
# ---------------------------------------------------------------------------
def _e2e_paths_present():
    paths = [
        "/ephemeral/dinov2_ckpts/teacher_unet_ckpt_060000.pth",
        "/ephemeral/dinov2_ckpts/checkpoint_step_352500_paris4.pt",
        "/ephemeral/dinov2_ckpts/avg_ref_embedding.npy",
    ]
    return all(Path(p).exists() for p in paths)


@pytest.mark.skipif(
    os.environ.get("DINOGUIDED_E2E", "0") != "1"
    or not torch.cuda.is_available()
    or not _e2e_paths_present(),
    reason="end-to-end test requires DINOGUIDED_E2E=1, CUDA, and snapshotted ckpts",
)
def test_generate_smoke_and_determinism():
    from koine_machines.data.dino_guided_labels import DinoGuidedLabelGenerator
    gen = DinoGuidedLabelGenerator(
        unet_ckpt="/ephemeral/dinov2_ckpts/teacher_unet_ckpt_060000.pth",
        dino_ckpt="/ephemeral/dinov2_ckpts/checkpoint_step_352500_paris4.pt",
        ref_embedding="/ephemeral/dinov2_ckpts/avg_ref_embedding.npy",
        device="cuda",
        dtype=torch.bfloat16,
    )
    img = torch.rand(1, 1, 256, 256, 256, device="cuda", dtype=torch.bfloat16)
    a = gen.generate(img)
    b = gen.generate(img)
    assert a.shape == (1, 1, 256, 256, 256)
    assert a.dtype == torch.float32
    assert torch.equal(a, b), "generator must be deterministic on a fixed input"
    assert torch.unique(a).tolist() in ([0.0], [1.0], [0.0, 1.0]), "output must be binary"
    sim = gen._dino_sim_map(img)
    assert float(sim.min()) >= 0.0 and float(sim.max()) <= 1.0
