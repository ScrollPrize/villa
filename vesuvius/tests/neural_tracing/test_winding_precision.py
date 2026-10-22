"""Tests for infer_winding_volume --precision and --decode-chunk-l."""
import os, sys
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util  # noqa: E402
WM_DIR = os.path.join(HERE, "../../src/vesuvius/neural_tracing/winding_models")


def _load(name, fn):
    # Load by file path: the vesuvius package __init__ pulls heavy optional deps.
    spec = importlib.util.spec_from_file_location(name, os.path.join(WM_DIR, fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


iwv = _load("patched_infer_winding_volume", "infer_winding_volume.py")
WindingModel = _load("patched_winding_model", "winding_model.py").WindingModel

CKPT = os.environ.get("WINDING_MODEL_CKPT", "")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        m = WindingModel(ck["config"]["model"]); m.load_state_dict(ck["model"])
    else:  # pragma: no cover
        pytest.skip("checkpoint not present")
    return m.eval()


def _slab(B=2, H=32, W=32, L=96, seed=1):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(B, H, W, L, generator=g) * 1000
    valid = torch.ones(B, H, W, L, dtype=torch.bool); valid[:, :4] = False; valid[..., -7:] = False
    return img, valid


@pytest.mark.parametrize("chunk", [1, 7, 24, 40, 96, 500])
def test_chunked_equals_stock_fp32(model, chunk):
    torch.set_num_threads(4)
    img, valid = _slab()
    with torch.inference_mode():
        ref = model(img, valid)["phase"]
        got = iwv.chunked_winding_forward(model, img, valid, chunk)["phase"]
    assert got.shape == ref.shape
    d = (got - ref).abs().max().item()
    assert d < 1e-5, d


def _parse(extra):
    argv = ["infer_winding_volume.py", "fit.ckpt", "out.zarr", "--reference-zarr", "r.zarr", "--model-ckpt", "m.pth"] + extra
    with mock.patch.object(sys, "argv", argv):
        return iwv.parse_args()


def test_flag_defaults_keep_stock_behaviour():
    a = _parse([])
    assert a.precision == "bf16" and a.decode_chunk_l is None
    assert iwv.resolve_autocast_dtype(a.precision) is torch.bfloat16


@pytest.mark.parametrize("p,dt", [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", None)])
def test_flag_parsing(p, dt):
    a = _parse(["--precision", p, "--decode-chunk-l", "64"])
    assert a.precision == p and a.decode_chunk_l == 64
    assert iwv.resolve_autocast_dtype(p) is dt


def test_bad_flags_rejected():
    with pytest.raises(SystemExit):
        _parse(["--precision", "int8"])
    with pytest.raises(SystemExit):
        _parse(["--decode-chunk-l", "0"])


def test_auto_selects_by_native_bf16():
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False) as f:
            assert iwv.resolve_autocast_dtype("auto") is torch.float16
            f.assert_called_with(including_emulation=False)
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            assert iwv.resolve_autocast_dtype("auto") is torch.bfloat16


def test_auto_older_torch_without_kwarg():
    def old(*a, **k):
        if k: raise TypeError("unexpected kwarg")
        return True  # old torch reports emulated bf16 as supported
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=old):
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            assert iwv.resolve_autocast_dtype("auto") is torch.float16  # Turing
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)):
            assert iwv.resolve_autocast_dtype("auto") is torch.bfloat16


def test_auto_without_cuda_is_fp16():
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert iwv.resolve_autocast_dtype("auto") is torch.float16


def test_forward_batch_threads_options(model):
    """_forward_batch uses forward_args (phase-cache path passes args=None)."""
    img, valid = _slab(B=1, L=48)
    batch = [(0, img[0].numpy(), valid[0].numpy(), None)]
    seen = {}
    real = iwv.chunked_winding_forward
    def spy(m, a, b, n):
        seen["n"] = n; return real(m, a, b, n)
    with mock.patch.object(iwv, "chunked_winding_forward", side_effect=spy), \
         mock.patch.object(iwv, "_postprocess_model_fields", side_effect=lambda ph, *a, **k: ph):
        with torch.inference_mode():
            ph = iwv._forward_batch(batch, model, torch.device("cpu"), 1, args=None,
                                    forward_args=SimpleNamespace(precision="fp32", decode_chunk_l=16))
            ref = model(img, valid)["phase"]
    assert seen["n"] == 16
    assert (ph - ref).abs().max().item() < 1e-5
