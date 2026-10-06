"""vesuvius.predict on machines without CUDA (CPU, Apple Silicon MPS).

Three things assumed CUDA: the --device default was 'cuda', which fails at model load
anywhere else; the batched mirroring TTA (8 flips in one forward pass, ~8x the activation
memory) relies on catching torch.cuda.OutOfMemoryError to fall back, which CPU and MPS
never raise; and float16 input patches only match the weights under CUDA autocast.
"""

from __future__ import annotations

import torch

from vesuvius.models.run import tta
from vesuvius.models.run.inference import build_parser


def test_device_defaults_to_none_so_the_accelerator_is_detected():
    args = build_parser().parse_args(
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o"])
    assert args.device is None


def test_mirroring_tta_runs_sequentially_off_cuda(monkeypatch):
    def batched(*args, **kwargs):
        raise AssertionError("batched TTA must not run off CUDA")

    monkeypatch.setattr(tta, "infer_with_tta_batched_3d", batched)
    model = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)
    inputs = torch.randn(1, 1, 8, 8, 8)
    with torch.inference_mode():
        out = tta.infer_with_tta(model, inputs, "mirroring")
        expected = tta.infer_with_tta_sequential(model, inputs, "mirroring")
    torch.testing.assert_close(out, expected)
