from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest
import tifffile
import torch
from vc_reader.reader import ZarrWriter, create_group

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference import engines
from vesuvius.ink_detection.inference.engines import EngineUnavailable, InkrsEngine
from vesuvius.ink_detection.inference.infer import main
from vesuvius.ink_detection.models.model import make_model

from . import inkrs_stand_in
from .helpers import config_mapping as make_config_mapping

DEPTH = 3
SIDE = 16
COMMON = ["--workers", "0", "--no-compile", "--blend-mode", "hann", "--batch-size", "2", "--device", "cpu"]


def _save_checkpoint(path, *, mixed_precision="fp16"):
    mapping = make_config_mapping("vesuvius_unet_2p5d", depth=DEPTH, side=SIDE)
    mapping["image_normalization"] = "robust_mad"
    mapping["mixed_precision"] = mixed_precision
    torch.manual_seed(0)
    model = make_model(InkConfig.from_mapping(mapping))
    torch.save({"config": mapping, "model": model.state_dict()}, path)
    return path


def _volume(layers=5, height=40, width=48):
    volume = np.random.default_rng(0).integers(1, 255, size=(layers, height, width)).astype(np.uint8)
    # an all-zero corner exercises the empty-tile skip and partial coverage
    volume[:, :24, :24] = 0
    return volume


def _bare_surface(path):
    volume = _volume()
    ZarrWriter.create(path, shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u1")[:] = volume
    return path


def _pyramid_surface(path):
    volume = _volume()
    level3 = np.zeros((5, 6), dtype=np.uint8)
    level3[1:4, 2:5] = 1
    group = create_group(path)
    group.create_array("0", shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u1")[:] = volume
    group.create_array("3", shape=level3.shape, chunks=level3.shape, dtype="u1")[:] = level3
    return path


@pytest.fixture
def stand_in(monkeypatch):
    """Install the PyTorch stand-in as ``inkrs`` and let it run on the CPU."""

    monkeypatch.setitem(sys.modules, "inkrs", inkrs_stand_in.stand_in_module())
    monkeypatch.setattr(engines, "INKRS_DEVICE_TYPES", ("cpu",))
    monkeypatch.setattr(inkrs_stand_in.Engine, "runs", 0)
    reads = []
    real_read = engines._read_span
    monkeypatch.setattr(engines, "_read_span", lambda *a: reads.append(a[2:]) or real_read(*a))
    return reads


def _infer(surface, checkpoint, output, *extra):
    assert main([str(surface), str(checkpoint), str(output), *COMMON, *extra]) == 0
    return output


@pytest.mark.parametrize("make_surface", [_bare_surface, _pyramid_surface])
def test_inkrs_path_writes_pytorchs_maps(tmp_path, stand_in, make_surface):
    # the stand-in runs villa's own forward and blend, so a difference can only
    # come from what villa hands inkrs: the window, its layer order or the occupancy
    surface = make_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    maps = []
    for start in (0, 1):
        window = ["--layer-start", str(start), "--layer-end", str(start + DEPTH), "--direction", "both"]
        _infer(surface, checkpoint, tmp_path / f"torch_L{start}.tif", *window)
        _infer(surface, checkpoint, tmp_path / f"inkrs_L{start}.tif", *window, "--engine", "inkrs")
        for suffix in ("", "_reverse"):
            reference = tifffile.imread(tmp_path / f"torch_L{start}{suffix}.tif")
            np.testing.assert_array_equal(tifffile.imread(tmp_path / f"inkrs_L{start}{suffix}.tif"), reference)
            maps.append(reference.tobytes())

    # four distinct maps, so one window or direction written everywhere would fail
    assert len(set(maps)) == 4
    assert inkrs_stand_in.Engine.runs == 4
    # each window's reverse run reuses its forward run's read
    assert stand_in == [(0, DEPTH), (1, 1 + DEPTH)]


@pytest.mark.parametrize(
    ("setup", "extra", "reason"),
    [
        ("missing", [], "the inkrs package is not installed"),
        ("gpu_only", [], "it needs a CUDA device, or MPS on macOS, got 'cpu'"),
        ("bf16", [], "it needs a checkpoint that stores mixed_precision fp16"),
        ("", ["--tta-mirror"], "it does not run --tta-mirror"),
        ("", ["--blend-mode", "gaussian"], "it blends with Hann only, got 'gaussian'"),
    ],
)
def test_command_inkrs_cannot_run_falls_back_or_stops(tmp_path, stand_in, monkeypatch, caplog, setup, extra, reason):
    if setup == "missing":
        monkeypatch.setitem(sys.modules, "inkrs", None)
    if setup == "gpu_only":
        monkeypatch.setattr(engines, "INKRS_DEVICE_TYPES", ("cuda", "mps"))
    surface = _bare_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth", mixed_precision="bf16" if setup == "bf16" else "fp16")

    reference = _infer(surface, checkpoint, tmp_path / "torch.tif", *extra)
    with caplog.at_level("WARNING"):
        fallback = _infer(surface, checkpoint, tmp_path / "fallback.tif", *extra, "--engine", "inkrs")

    assert f"The inkrs engine cannot run this command: {reason}" in caplog.text
    np.testing.assert_array_equal(tifffile.imread(fallback), tifffile.imread(reference))
    assert inkrs_stand_in.Engine.runs == 0
    with pytest.raises(EngineUnavailable, match=reason):
        _infer(surface, checkpoint, tmp_path / "strict.tif", *extra, "--engine", "inkrs", "--engine-fallback", "error")


def test_input_inkrs_declines_runs_on_pytorch_and_other_errors_raise(tmp_path, stand_in, monkeypatch, caplog):
    surface = _bare_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    reference = _infer(surface, checkpoint, tmp_path / "torch.tif")

    def failing(error):
        def run(self, *args, **kwargs):
            raise error

        monkeypatch.setattr(inkrs_stand_in.Engine, "run", run)

    failing(inkrs_stand_in.UnsupportedError("a volume of 4 GiB or more"))
    with caplog.at_level("WARNING"):
        fallback = _infer(surface, checkpoint, tmp_path / "fallback.tif", "--engine", "inkrs")
    assert "inkrs does not support this input: a volume of 4 GiB or more. Running it with PyTorch" in caplog.text
    np.testing.assert_array_equal(tifffile.imread(fallback), tifffile.imread(reference))

    with pytest.raises(EngineUnavailable, match="4 GiB"):
        _infer(surface, checkpoint, tmp_path / "strict.tif", "--engine", "inkrs", "--engine-fallback", "error")

    # out of GPU memory or a missing library is a real failure, not a reason to fall back
    failing(inkrs_stand_in.InkrsError("CUDA out of memory"))
    with pytest.raises(inkrs_stand_in.InkrsError, match="out of memory"):
        _infer(surface, checkpoint, tmp_path / "oom.tif", "--engine", "inkrs")
    assert not (tmp_path / "strict.tif").exists() and not (tmp_path / "oom.tif").exists()


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        # index-less cuda is PyTorch's current device, which a fallback also uses
        (torch.device("cuda"), 1),
        (torch.device("cuda", 2), 2),
        (torch.device("mps"), 0),
    ],
)
def test_inkrs_loads_on_the_device_pytorch_uses(tmp_path, monkeypatch, device, expected):
    opened = []

    def engine(checkpoint, *, device):
        opened.append(device)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setitem(sys.modules, "inkrs", inkrs_stand_in.stand_in_module(engine))
    InkrsEngine.load(tmp_path / "m.pth", device)
    assert opened == [expected]


def test_villa_calls_match_the_installed_inkrs():
    inkrs = pytest.importorskip("inkrs")
    assert issubclass(inkrs.UnsupportedError, inkrs.InkrsError)
    # villa calls Engine(checkpoint, device=) and run(volume, windows, stride=, occupancy=, coverage=)
    assert "device" in inspect.signature(inkrs.Engine.__init__).parameters
    assert {"stride", "occupancy", "coverage"} <= set(inspect.signature(inkrs.Engine.run).parameters)
