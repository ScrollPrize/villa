"""``--amp-dtype default`` means full precision, not float16 autocast."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from vesuvius.ink_detection.inference.inference_runtime import inference_autocast


def test_no_resolved_dtype_means_no_autocast_context():
    # torch.autocast("cuda", dtype=None) would ENABLE float16 autocast, which
    # is what run_block_inference used to do for --amp-dtype default.
    assert isinstance(inference_autocast(torch.device("cuda"), None), nullcontext)


def test_cpu_never_autocasts():
    assert isinstance(inference_autocast(torch.device("cpu"), torch.float16), nullcontext)
    assert isinstance(inference_autocast(torch.device("cpu"), None), nullcontext)


def test_resolved_dtype_opens_autocast_with_that_dtype():
    context = inference_autocast(torch.device("cuda"), torch.float16)
    assert isinstance(context, torch.autocast)
    assert context.fast_dtype == torch.float16
    context = inference_autocast(torch.device("cuda"), torch.bfloat16)
    assert context.fast_dtype == torch.bfloat16


def test_default_really_disables_autocast_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    with inference_autocast(torch.device("cuda"), None):
        assert not torch.is_autocast_enabled("cuda")
    with inference_autocast(torch.device("cuda"), torch.float16):
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") == torch.float16
