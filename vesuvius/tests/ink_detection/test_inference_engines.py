from __future__ import annotations

import gc
import sys
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from vc_reader.reader import ZarrWriter, create_group

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference import engines
from vesuvius.ink_detection.inference.engines import (
    EngineUnavailable,
    FlatEngineResult,
    InkrsEngine,
    WindowCache,
    select_engine,
)
from vesuvius.ink_detection.inference.infer import (
    main,
    normalize_inference_paths,
    parse_args,
    run,
)
from vesuvius.ink_detection.models.model import make_model

from .helpers import config_mapping as make_config_mapping
from .inkrs_stand_in import TorchStandInEngine, load_torch_stand_in

DEPTH = 3
SIDE = 16


def _save_checkpoint(path, *, side=SIDE):
    mapping = make_config_mapping("vesuvius_unet_2p5d", depth=DEPTH, side=side)
    mapping["image_normalization"] = "robust_mad"
    torch.manual_seed(0)
    model = make_model(InkConfig.from_mapping(mapping))
    torch.save({"config": mapping, "model": model.state_dict()}, path)
    return path


def _write_surface(path, *, layers=5, height=40, width=48):
    volume = np.random.default_rng(0).integers(1, 255, size=(layers, height, width))
    # an all-zero corner exercises the empty-tile skip and partial coverage
    volume[:, :24, :24] = 0
    array = ZarrWriter.create(
        path, shape=volume.shape, chunks=(layers, 16, 16), dtype="u1"
    )
    array[:] = volume
    return path, volume.astype(np.uint8)


COMMON = ["--workers", "0", "--no-compile", "--blend-mode", "hann", "--batch-size", "2"]


def _run_flat(surface, checkpoint, output, *extra, engine_factory=None):
    args = normalize_inference_paths(
        parse_args([str(surface), str(checkpoint), str(output), *COMMON, *extra])
    )
    assert run(args, engine_factory=engine_factory) == 0


def _record_stand_in_coverage(monkeypatch):
    # how many pixels each stand-in run covered, which also shows the engine ran rather than fell back
    covered = []
    real_run = TorchStandInEngine.run

    def recording_run(self, *args, **kwargs):
        results = real_run(self, *args, **kwargs)
        covered.extend(int(c.sum()) for _, c in results)
        return results

    monkeypatch.setattr(TorchStandInEngine, "run", recording_run)
    return covered


def _pyramid_surface(tmp_path):
    _, volume = _write_surface(tmp_path / "bare.zarr")
    level3 = np.zeros((5, 6), dtype=np.uint8)
    level3[1:4, 2:5] = 1
    return _write_group(tmp_path / "group.zarr", volume, level3)


@pytest.mark.parametrize("pyramid", [False, True])
def test_stand_in_through_the_inkrs_adapter_writes_pytorchs_maps(tmp_path, monkeypatch, pyramid):
    # the stand-in runs villa's own forward and blend behind inkrs.Engine's signatures, so a
    # difference can only come from the adapter: the window, its layer order or the occupancy hand-off
    surface = _pyramid_surface(tmp_path) if pyramid else _write_surface(tmp_path / "surface.zarr")[0]
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    covered = _record_stand_in_coverage(monkeypatch)

    for start in (0, 1):
        window = ("--layer-start", str(start), "--layer-end", str(start + DEPTH), "--direction", "both")
        _run_flat(surface, checkpoint, tmp_path / f"torch_L{start}.tif", *window)
        _run_flat(
            surface, checkpoint, tmp_path / f"stand_in_L{start}.tif", *window,
            engine_factory=load_torch_stand_in,
        )

    names = [f"L{start}{suffix}.tif" for start in (0, 1) for suffix in ("", "_reverse")]
    for name in names:
        np.testing.assert_array_equal(
            tifffile.imread(tmp_path / f"stand_in_{name}"), tifffile.imread(tmp_path / f"torch_{name}")
        )
    reference = [tifffile.imread(tmp_path / f"torch_{name}") for name in names]
    # four distinct maps, so one window or direction written everywhere would fail above
    assert all(m.any() for m in reference)
    assert len({m.tobytes() for m in reference}) == len(names)
    assert len(covered) == len(names)
    if pyramid:
        # the level-3 mask drops tiles that the bare array would run
        assert all(c < 40 * 48 - 16 * 16 for c in covered)
    else:
        # only the 16 x 16 corner lies under skipped tiles alone
        assert covered == [40 * 48 - 16 * 16] * len(names)


def test_maps_differ_when_the_adapter_drops_villas_occupancy(tmp_path, monkeypatch):
    # the check above must be able to fail: without villa's level-3 mask the stand-in
    # schedules tiles that PyTorch skips and writes ink where PyTorch writes none
    surface = _pyramid_surface(tmp_path)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    real = engines._inkrs_infer
    monkeypatch.setattr(
        engines,
        "_inkrs_infer",
        lambda module, handle, window, occupancy, *, stride: real(module, handle, window, None, stride=stride),
    )
    window = ("--layer-start", "1", "--layer-end", str(1 + DEPTH))

    _run_flat(surface, checkpoint, tmp_path / "torch.tif", *window)
    _run_flat(surface, checkpoint, tmp_path / "lost.tif", *window, engine_factory=load_torch_stand_in)

    reference, lost = tifffile.imread(tmp_path / "torch.tif"), tifffile.imread(tmp_path / "lost.tif")
    assert ((lost != 0) & (reference == 0)).any()


def _inkrs_shaped_model(**overrides):
    values = dict(
        amp_dtype=torch.float16,
        preprocessing="tifxyz_robust",
        input_depth=17,
        patch_size=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _engine_args(tmp_path, *extra):
    return normalize_inference_paths(
        parse_args(["in.zarr", str(tmp_path / "m.pth"), "out.tif", "--engine", "inkrs", *extra])
    )


def test_inkrs_missing_falls_back_or_stops_with_the_reason(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "inkrs", None)
    device = torch.device("cuda", 0)

    with caplog.at_level("WARNING"):
        engine = select_engine(_engine_args(tmp_path), _inkrs_shaped_model(), device)
    assert engine is None
    assert "inkrs package is not installed" in caplog.text
    assert "Continuing with PyTorch" in caplog.text

    with pytest.raises(EngineUnavailable, match="not installed"):
        select_engine(
            _engine_args(tmp_path, "--engine-fallback", "error"),
            _inkrs_shaped_model(),
            device,
        )


def test_inkrs_refuses_unsupported_models_and_options_before_importing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "inkrs", None)
    strict = _engine_args(tmp_path, "--engine-fallback", "error", "--tta-mirror")

    with pytest.raises(EngineUnavailable) as raised:
        select_engine(
            strict,
            _inkrs_shaped_model(amp_dtype=torch.bfloat16),
            torch.device("cpu"),
        )

    message = str(raised.value)
    for reason in ("CUDA device", "--tta-mirror", "fp16 autocast"):
        assert reason in message
    assert "not installed" not in message


@pytest.mark.parametrize(
    ("checkpoint", "extra", "reason"),
    [
        ("m.pth", ("--roi", "0", "8", "0", "8"), "it does not crop to --roi"),
        ("m.pth", ("--mask-path", "mask.png"), "it does not apply --mask-path"),
        ("m.pth", ("--blend-mode", "gaussian"), "it blends with Hann only, got 'gaussian'"),
        # inkrs would read any other path as its own safetensors export and fail, not decline
        ("m.pt", (), "it loads .pth checkpoints only, got m.pt"),
    ],
)
def test_inkrs_declines_whole_commands_without_loading(tmp_path, monkeypatch, caplog, checkpoint, extra, reason):
    # these hold for every input, so loading inkrs for them would only take GPU memory PyTorch needs
    def never_load(checkpoint, *, device):
        raise AssertionError("inkrs loaded for a command it declines")

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(never_load))
    args = normalize_inference_paths(
        parse_args(["in.zarr", str(tmp_path / checkpoint), "out.tif", "--engine", "inkrs", *extra])
    )
    with caplog.at_level("WARNING"):
        engine = select_engine(args, _inkrs_shaped_model(), torch.device("cuda", 0))
    assert engine is None
    assert reason in caplog.text


def test_cpu_run_requesting_inkrs_writes_the_pytorch_map(tmp_path, monkeypatch, caplog):
    from vesuvius.ink_detection.inference import infer

    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    # inkrs importable, so only the CPU refusal can stop the read ahead
    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(None))
    reads = []
    monkeypatch.setattr(infer, "read_span", lambda *a: reads.append(a))

    assert main([str(surface), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON, "--device", "cpu"]) == 0
    with caplog.at_level("WARNING"):
        assert (
            main(
                [str(surface), str(checkpoint), str(tmp_path / "inkrs.tif"), *COMMON, "--device", "cpu", "--engine", "inkrs"]
            )
            == 0
        )
    assert "Continuing with PyTorch" in caplog.text
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "inkrs.tif"), tifffile.imread(tmp_path / "torch.tif")
    )
    # a command inkrs refuses reads nothing ahead for it
    assert reads == []

    with pytest.raises(EngineUnavailable):
        main(
            [
                str(surface),
                str(checkpoint),
                str(tmp_path / "strict.tif"),
                *COMMON, "--device", "cpu",
                "--engine",
                "inkrs",
                "--engine-fallback",
                "error",
            ]
        )


class _RecordingHandle:
    """Stands in for inkrs.Engine and records what villa sends to run()."""

    def __init__(self):
        self.calls = []

    def run(self, volume, windows, *, stride, occupancy, coverage):
        self.calls.append((volume.copy(), windows, stride, occupancy, coverage))
        height, width = volume.shape[1:]
        probability = np.full((height, width), 9, dtype=np.uint8)
        return [(probability, np.ones((height, width), dtype=np.uint8))]


def _write_group(path, volume, level3):
    group = create_group(path)
    group.create_array("0", shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u1")[:] = volume
    group.create_array("3", shape=level3.shape, chunks=level3.shape, dtype="u1")[:] = level3
    return path


class _InkrsError(RuntimeError):
    pass


class _UnsupportedError(_InkrsError):
    pass


def _fake_inkrs(engine):
    return SimpleNamespace(
        InkrsError=_InkrsError, UnsupportedError=_UnsupportedError, Engine=engine, __version__="0"
    )


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        # index-less cuda is the current device, which set_device moved to 1
        (torch.device("cuda"), 1),
        (torch.device("cuda", 2), 2),
        (torch.device("mps"), 0),
    ],
)
def test_inkrs_loads_on_the_device_pytorch_uses(tmp_path, monkeypatch, device, expected):
    opened = []

    def load(checkpoint, *, device):
        opened.append(device)
        return object()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(load))
    engine = select_engine(_engine_args(tmp_path), _inkrs_shaped_model(), device)

    assert opened == [expected]
    assert engine.device_index == expected


def test_inkrs_loads_the_checkpoint_pytorch_would_after_expanding_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    checkpoint = tmp_path / "models" / "m.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    opened = []

    def load(path, *, device):
        # inkrs checks the file itself and raises InkrsError, not UnsupportedError
        if not Path(path).is_file():
            raise _InkrsError(f"checkpoint not found: {path}")
        opened.append(Path(path))
        return object()

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(load))
    # a quoted ~ reaches villa unexpanded, and the PyTorch loader expands it
    args = normalize_inference_paths(
        parse_args(["in.zarr", "~/models/m.pth", "out.tif", "--engine", "inkrs"])
    )
    engine = select_engine(args, _inkrs_shaped_model(), torch.device("cuda", 0))

    assert engine is not None
    assert opened == [checkpoint]


def test_inkrs_load_falls_back_only_when_inkrs_calls_it_unsupported(
    tmp_path, monkeypatch, caplog
):
    def refuse(checkpoint, *, device):
        raise _UnsupportedError("checkpoint is not the hybrid_3d2d-seed42 contract")

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(refuse))
    with caplog.at_level("WARNING"):
        engine = select_engine(
            _engine_args(tmp_path), _inkrs_shaped_model(), torch.device("cuda", 0)
        )
    assert engine is None
    assert "inkrs cannot load m.pth on device 0" in caplog.text
    assert "hybrid_3d2d-seed42 contract" in caplog.text

    def broken(checkpoint, *, device):
        raise _InkrsError("no cuDNN 9 found")

    # a broken install is a real failure, not a reason to run quietly on PyTorch
    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(broken))
    with pytest.raises(_InkrsError, match="cuDNN"):
        select_engine(_engine_args(tmp_path), _inkrs_shaped_model(), torch.device("cuda", 0))


@pytest.mark.parametrize("edit", ["unexpected", "missing"])
def test_installed_inkrs_declines_a_checkpoint_villa_loads_non_strictly(
    tmp_path, monkeypatch, caplog, edit
):
    # villa loads with strict=False and runs such a checkpoint; inkrs refuses it in its
    # export, which runs on the CPU before inkrs opens the GPU, so this needs no GPU
    inkrs = pytest.importorskip("inkrs")
    pytest.importorskip("safetensors")
    payload = torch.load(_save_checkpoint(tmp_path / "model.pth"), weights_only=True)
    state = payload["model"]
    if edit == "unexpected":
        state["extra.weight"] = torch.zeros(1)
        expected = "unexpected keys: extra.weight"
    else:
        key = sorted(state)[0]
        del state[key]
        expected = f"missing keys: {key}"
    checkpoint = tmp_path / f"{edit}.pth"
    torch.save(payload, checkpoint)
    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)

    def command(*extra):
        return normalize_inference_paths(
            parse_args(["in.zarr", str(checkpoint), "out.tif", "--engine", "inkrs", *extra])
        )

    with caplog.at_level("WARNING"):
        engine = select_engine(command(), _inkrs_shaped_model(), torch.device("cuda", 0))
    assert engine is None
    assert f"inkrs cannot load {edit}.pth on device 0" in caplog.text
    assert expected in caplog.text
    assert "Continuing with PyTorch" in caplog.text

    with pytest.raises(EngineUnavailable) as raised:
        select_engine(
            command("--engine-fallback", "error"), _inkrs_shaped_model(), torch.device("cuda", 0)
        )
    assert isinstance(raised.value.__cause__, inkrs.UnsupportedError)
    assert expected in str(raised.value)


def test_input_inkrs_calls_unsupported_runs_on_pytorch_and_other_errors_raise(tmp_path, caplog):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    assert main([str(surface), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON]) == 0

    class _Handle:
        def __init__(self, error):
            self.error = error

        def run(self, volume, windows, *, stride, occupancy, coverage):
            raise self.error

    def run_with(error, output, *extra):
        args = normalize_inference_paths(
            parse_args([str(surface), str(checkpoint), str(tmp_path / output), *COMMON, *extra])
        )
        engine = InkrsEngine(_fake_inkrs(None), _Handle(error), device_index=0)
        return run(args, engine_factory=lambda *_: engine)

    with caplog.at_level("WARNING"):
        run_with(_UnsupportedError("a volume of 4 GiB or more"), "fallback.tif")
    assert "a volume of 4 GiB or more. Running this input with PyTorch" in caplog.text
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "fallback.tif"), tifffile.imread(tmp_path / "torch.tif")
    )

    with pytest.raises(EngineUnavailable, match="4 GiB"):
        run_with(_UnsupportedError("a volume of 4 GiB or more"), "strict.tif", "--engine-fallback", "error")
    with pytest.raises(_InkrsError, match="out of memory"):
        run_with(_InkrsError("CUDA out of memory"), "oom.tif")
    assert not (tmp_path / "strict.tif").exists() and not (tmp_path / "oom.tif").exists()


def test_stand_in_and_adapter_match_the_installed_inkrs_signatures():
    inkrs = pytest.importorskip("inkrs")
    import inspect

    assert issubclass(inkrs.UnsupportedError, inkrs.InkrsError)
    real_init = inspect.signature(inkrs.Engine.__init__).parameters
    real_run = inspect.signature(inkrs.Engine.run).parameters
    # the adapter's calls: Engine(checkpoint, device=) and run(volume, windows, stride=, occupancy=, coverage=)
    assert {"checkpoint", "device"} <= real_init.keys()
    assert {"volume", "windows", "stride", "occupancy", "coverage"} <= real_run.keys()
    stand_in_run = inspect.signature(TorchStandInEngine.run).parameters
    for name in ("stride", "occupancy", "coverage", "outputs"):
        assert stand_in_run[name].default == real_run[name].default, name


@pytest.mark.parametrize("pyramid", [False, True])
def test_inkrs_adapter_sends_layers_in_model_order_and_villas_occupancy(tmp_path, pyramid):
    surface, volume = _write_surface(tmp_path / "surface.zarr")
    level3 = np.zeros((5, 6), dtype=np.uint8)
    level3[1:4, 2:5] = 1
    if pyramid:
        surface = _write_group(tmp_path / "group.zarr", volume, level3)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    handle = _RecordingHandle()
    engine = InkrsEngine(_fake_inkrs(None), handle, device_index=0)
    args = normalize_inference_paths(
        parse_args(
            [
                str(surface),
                str(checkpoint),
                str(tmp_path / "out.tif"),
                *COMMON,
                "--layer-start",
                "1",
                "--layer-end",
                "4",
                "--direction",
                "reverse",
            ]
        )
    )

    assert run(args, engine_factory=lambda *_: engine) == 0

    [(window, windows, stride, occupancy, coverage)] = handle.calls
    np.testing.assert_array_equal(window, volume[[3, 2, 1]])
    assert window.flags.c_contiguous and window.dtype == np.uint8
    assert windows == [(0, 1)] and stride == SIDE // 2 and coverage is True
    if pyramid:
        # inkrs scales villa's own low-resolution mask by round(40 / 5) = 8
        np.testing.assert_array_equal(occupancy, level3 != 0)
    else:
        # a bare array has no pyramid, so villa scans nothing and neither may inkrs
        assert occupancy is None
    assert (tifffile.imread(tmp_path / "out.tif") == 9).all()


def test_both_directions_read_the_source_layers_once_beside_the_engine_load(
    tmp_path, monkeypatch
):
    from vesuvius.ink_detection.inference import infer

    surface, volume = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    reads = []
    real_read = engines.read_span
    read_started, load_done = threading.Event(), threading.Event()

    def recording_read(*a):
        reads.append((threading.current_thread().name, a[2:]))
        read_started.set()
        # the read finishes only once the engine has loaded, so a run that read
        # before or after the load, not during it, times out here or below
        assert load_done.wait(10), "the engine never loaded while the read waited"
        return real_read(*a)

    def load(*_):
        assert read_started.wait(10), "the read did not start before the engine load"
        load_done.set()
        return engine

    monkeypatch.setattr(engines, "read_span", recording_read)
    monkeypatch.setattr(infer, "read_span", recording_read)
    handle = _RecordingHandle()
    engine = InkrsEngine(_fake_inkrs(None), handle, device_index=0)
    args = normalize_inference_paths(
        parse_args(
            [str(surface), str(checkpoint), str(tmp_path / "out.tif"), *COMMON,
             "--layer-start", "1", "--layer-end", "4", "--direction", "both"]
        )
    )

    assert run(args, engine_factory=load) == 0

    # one read, on a thread, spanning the engine load
    assert reads == [("engine-window-prefetch", (1, 4))]
    assert not any(t.name == "engine-window-prefetch" for t in threading.enumerate())
    [forward, reverse] = [call[0] for call in handle.calls]
    np.testing.assert_array_equal(forward, volume[[1, 2, 3]])
    np.testing.assert_array_equal(reverse, volume[[3, 2, 1]])


def test_window_read_ahead_only_for_runs_inkrs_would_take(tmp_path, monkeypatch):
    from vesuvius.ink_detection.inference.infer import engine_window_wanted

    model, cuda = _inkrs_shaped_model(), torch.device("cuda", 0)

    def wanted(*extra, device=cuda):
        return engine_window_wanted(_engine_args(tmp_path, *extra), model, device, None)

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(None))
    assert wanted("--blend-mode", "hann", "--stride", "64")
    # inkrs's macOS build runs on Metal through villa's MPS device
    assert wanted("--blend-mode", "hann", "--stride", "64", device=torch.device("mps"))
    # every refusal inkrs or villa would make before a request leaves the read undone
    assert not wanted("--blend-mode", "constant", "--stride", "64")
    assert not wanted("--blend-mode", "hann", "--stride", "999")
    assert not wanted("--blend-mode", "hann", "--roi", "0", "1", "0", "1")
    assert not wanted("--blend-mode", "hann", "--tta-mirror")
    assert not wanted("--blend-mode", "hann", device=torch.device("cpu"))
    monkeypatch.setitem(sys.modules, "inkrs", None)
    assert not wanted("--blend-mode", "hann", "--stride", "64")


def test_inkrs_declines_windows_it_cannot_take_before_reading_them(tmp_path, monkeypatch, caplog):
    # inkrs checks dtype, size and the 4 GiB limit only once it has the window in memory, so villa
    # applies the same limits from metadata first; an oversized input must never be read whole
    from vesuvius.ink_detection.inference import infer
    from vesuvius.ink_detection.inference.engines import FlatEngineRequest

    reads = []
    real_read = engines.read_span

    def recording_read(*a):
        reads.append(a[2:])
        return real_read(*a)

    monkeypatch.setattr(engines, "read_span", recording_read)
    monkeypatch.setattr(infer, "read_span", recording_read)
    handle = _RecordingHandle()
    engine = InkrsEngine(_fake_inkrs(None), handle, device_index=0)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    # a uint16 input runs on PyTorch with neither the read ahead nor the request reading it
    _, volume = _write_surface(tmp_path / "u8.zarr")
    wide = tmp_path / "u16.zarr"
    ZarrWriter.create(wide, shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u2")[:] = (
        volume.astype(np.uint16) * 200
    )
    assert main([str(wide), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON]) == 0
    args = normalize_inference_paths(parse_args([str(wide), str(checkpoint), str(tmp_path / "engine.tif"), *COMMON]))
    with caplog.at_level("WARNING"):
        assert run(args, engine_factory=lambda *_: engine) == 0
    assert "it reads uint8 volumes only, got uint16" in caplog.text
    assert reads == [] and handle.calls == []
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "engine.tif"), tifffile.imread(tmp_path / "torch.tif")
    )

    # 3 x 40000 x 36000 bytes is just over 4 GiB; with no chunk written it takes no disk
    big = ZarrWriter.create(tmp_path / "big.zarr", shape=(DEPTH, 40000, 36000), chunks=(DEPTH, 1024, 1024), dtype="u1")
    request = FlatEngineRequest(
        input_zarr=tmp_path / "big.zarr", resolution="0", volume=big, depth_axis_first=True,
        height=40000, width=36000, layer_indices=np.arange(DEPTH), input_depth=DEPTH,
        patch_size=SIDE, stride=SIDE // 2, blend_mode="hann", preprocessing="tifxyz_robust", blocks=(),
    )
    assert engine.unsupported_reason(request) == "its 3-layer window of 4320000000 bytes is 4 GiB or more"
    # a 2-layer input is shallower than the 3-layer model, so the engine declines it too
    _write_surface(tmp_path / "shallow.zarr", layers=2)
    model = _inkrs_shaped_model(input_depth=DEPTH, patch_size=SIDE)
    for surface in ("big.zarr", "shallow.zarr", "u8.zarr"):
        cache = WindowCache()
        args = normalize_inference_paths(
            parse_args([str(tmp_path / surface), str(checkpoint), "out.tif", "--engine", "inkrs"])
        )
        infer.prefetch_engine_window(args, model, cache)
        cache.close()
    # only the small uint8 input was read ahead
    assert reads == [(1, 4)]
    narrow = SimpleNamespace(dtype=np.dtype("u1"))
    assert engines.inkrs_window_reason(narrow, DEPTH, 40, SIDE - 1, SIDE) == (
        "the input's 40x15 layers are smaller than one 16 px tile"
    )
    assert engines.inkrs_window_reason(narrow, DEPTH, SIDE, SIDE, SIDE) is None


def test_folder_runs_load_the_engine_once_and_fall_back_per_segment(tmp_path, monkeypatch):
    # one uint8 segment runs on the engine and one uint16 segment falls back to PyTorch, in one folder run
    _, volume = _write_surface(tmp_path / "u8.zarr")
    folder = tmp_path / "segments"
    _write_surface(folder / "a" / "a.zarr")
    (folder / "b").mkdir(parents=True)
    ZarrWriter.create(folder / "b" / "b.zarr", shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u2")[:] = (
        volume.astype(np.uint16) * 200
    )
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    loads, handle = [], _RecordingHandle()

    def factory(*_):
        loads.append(1)
        return InkrsEngine(_fake_inkrs(None), handle, device_index=0)

    args = normalize_inference_paths(parse_args(["--folder", str(folder), str(checkpoint), *COMMON, "--direction", "both"]))
    assert run(args, engine_factory=factory) == 0
    assert main([str(folder / "b" / "b.zarr"), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON]) == 0

    assert loads == [1]
    # segment a's two directions went to the engine, in model order, and b's to neither
    [forward, reverse] = [call[0] for call in handle.calls]
    np.testing.assert_array_equal(reverse, forward[::-1])
    for direction in ("forward", "reverse"):
        [a_map] = (folder / "a" / "preds").glob(f"a_model_{direction}_*.tif")
        assert (tifffile.imread(a_map) == 9).all()
    [b_forward] = (folder / "b" / "preds").glob("b_model_forward_*.tif")
    np.testing.assert_array_equal(tifffile.imread(b_forward), tifffile.imread(tmp_path / "torch.tif"))


def test_window_cache_keeps_one_span_and_never_a_failed_read():
    cache = WindowCache()
    calls = []

    def read(value):
        def inner():
            calls.append(value)
            return np.full((1, 2, 2), value, np.uint8)
        return inner

    cache.prefetch(("a", "0", True, 0, 1), read(1))
    first = cache.span(("a", "0", True, 0, 1), read(2))
    assert cache.span(("a", "0", True, 0, 1), read(2)) is first
    # the shared span is read-only, so one engine cannot change another's input
    assert not first.flags.writeable
    assert (cache.span(("b", "0", True, 0, 1), read(3)) == 3).all()
    assert (cache.span(("a", "0", True, 0, 1), read(4)) == 4).all()
    assert calls == [1, 3, 4]

    def broken():
        raise OSError("unreadable chunk")

    with pytest.raises(OSError):
        cache.span(("c", "0", True, 0, 1), broken)
    assert (cache.span(("c", "0", True, 0, 1), read(5)) == 5).all()
    # a failed read ahead is read again by the request, which then owns any error
    cache.prefetch(("d", "0", True, 0, 1), broken)
    assert (cache.span(("d", "0", True, 0, 1), read(6)) == 6).all()
    assert calls == [1, 3, 4, 5, 6]

    # close waits for a read ahead nobody asked for and keeps nothing
    release = threading.Event()

    def slow():
        release.wait(10)
        return read(7)()

    cache.prefetch(("e", "0", True, 0, 1), slow)
    threading.Timer(0.05, release.set).start()
    cache.close()
    assert calls[-1] == 7
    assert (cache.span(("e", "0", True, 0, 1), read(8)) == 8).all()


def test_engine_shape_mismatch_is_an_error_not_a_written_map(tmp_path):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    class _WrongShape:
        name = "wrong"

        def unsupported_reason(self, request):
            return None

        def predict(self, request):
            return FlatEngineResult(
                probability=np.zeros((2, 2), dtype=np.uint8),
                coverage=np.zeros((2, 2), dtype=bool),
            )

    args = normalize_inference_paths(
        parse_args([str(surface), str(checkpoint), str(tmp_path / "out.tif"), *COMMON])
    )
    with pytest.raises(RuntimeError, match="expected \\(40, 48\\)"):
        run(args, engine_factory=lambda *_: _WrongShape())
    assert not (tmp_path / "out.tif").exists()


def test_inkrs_accepts_autocast_without_an_explicit_dtype(tmp_path, monkeypatch):
    # --amp-dtype default leaves the dtype to torch.autocast, which is fp16 on CUDA
    monkeypatch.setitem(sys.modules, "inkrs", None)

    with pytest.raises(EngineUnavailable) as raised:
        select_engine(
            _engine_args(tmp_path, "--engine-fallback", "error"),
            _inkrs_shaped_model(amp_dtype=None),
            torch.device("cuda", 0),
        )

    assert "not installed" in str(raised.value)
    assert "fp16" not in str(raised.value)


def test_engine_runs_leave_pytorch_off_the_device_until_an_input_falls_back(
    tmp_path, monkeypatch
):
    from vesuvius.ink_detection.inference import infer

    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    prepared = []
    real_prepare = infer.prepare_model_for_inference

    def counting_prepare(model, **kwargs):
        prepared.append(model)
        return real_prepare(model, **kwargs)

    monkeypatch.setattr(infer, "prepare_model_for_inference", counting_prepare)
    built = []
    real_make = infer.make_model
    monkeypatch.setattr(infer, "make_model", lambda config: built.append(config) or real_make(config))

    class _Engine:
        name = "fixed"

        def __init__(self, decline):
            self.decline = decline

        def unsupported_reason(self, request):
            return "declined" if self.decline else None

        def predict(self, request):
            shape = (request.height, request.width)
            return FlatEngineResult(np.full(shape, 9, np.uint8), np.ones(shape, bool))

    def run_with(engine, output):
        args = normalize_inference_paths(
            parse_args([str(surface), str(checkpoint), str(tmp_path / output), *COMMON,
                        "--direction", "both"])
        )
        run(args, engine_factory=None if engine is None else lambda *_: engine)

    engine = _Engine(decline=False)
    released = weakref.ref(engine)
    gc.disable()
    try:
        run_with(engine, "engine.tif")
        del engine
        # nothing villa keeps may hold the engine, or its GPU memory, past run()
        assert released() is None
    finally:
        gc.enable()
    # an engine that runs every input never needs the PyTorch model built
    assert prepared == [] and built == []

    run_with(_Engine(decline=True), "fallback.tif")
    # both directions fell back, and the model was built and prepared once for both
    assert len(prepared) == 1 and len(built) == 1
    run_with(None, "torch.tif")
    assert len(built) == 2
    for name in ("", "_reverse"):
        np.testing.assert_array_equal(
            tifffile.imread(tmp_path / f"fallback{name}.tif"),
            tifffile.imread(tmp_path / f"torch{name}.tif"),
        )


class _WindowHandle(_RecordingHandle):
    """A recording stand-in whose map depends on the window it is sent."""

    def run(self, volume, windows, *, stride, occupancy, coverage):
        super().run(volume, windows, stride=stride, occupancy=occupancy, coverage=coverage)
        weights = np.arange(1, volume.shape[0] + 1, dtype=np.int64)[:, None, None]
        probability = ((volume * weights).sum(axis=0) % 251).astype(np.uint8)
        return [(probability, np.ones(volume.shape[1:], dtype=np.uint8))]


@pytest.mark.parametrize("engine", [False, True])
def test_layer_starts_write_each_single_window_runs_bytes_with_one_load(tmp_path, monkeypatch, engine):
    import vesuvius.ink_detection.inference.infer as infer_module

    surface, volume = _write_surface(tmp_path / "surface.zarr", layers=6)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    loads, handles, builds = [], [], []
    # every PyTorch model build or preparation, so the multi-window run must show one of each at most
    for name in ("configure_model", "prepare_model_for_inference"):
        real = getattr(infer_module, name)
        monkeypatch.setattr(infer_module, name, lambda *a, _real=real, _name=name, **k: builds.append(_name) or _real(*a, **k))

    def factory(*_):
        handles.append(_WindowHandle())
        loads.append(1)
        return InkrsEngine(_fake_inkrs(None), handles[-1], device_index=0)

    def infer(out, *window):
        args = normalize_inference_paths(
            parse_args([str(surface), str(checkpoint), str(out), *COMMON, "--direction", "both", *window])
        )
        assert run(args, engine_factory=factory if engine else None) == 0

    infer(tmp_path / "multi" / "map.tif", "--layer-starts", "2,0,1")
    multi_loads = len(loads)
    # one reference contract read, which checks the windows; PyTorch then builds and prepares its
    # model once, the engine path never
    torch_build = ["configure_model", "prepare_model_for_inference"]
    assert builds == ["configure_model"] + ([] if engine else torch_build), builds
    for start in (0, 1, 2):
        infer(tmp_path / "single" / f"map_L{start}.tif", "--layer-start", str(start), "--layer-end", str(start + DEPTH))

    names = sorted(p.name for p in (tmp_path / "multi").iterdir())
    assert names == sorted(f"map_L{s}{r}.tif" for s in (0, 1, 2) for r in ("", "_reverse"))
    for name in names:
        multi = (tmp_path / "multi" / name).read_bytes()
        assert multi == (tmp_path / "single" / name).read_bytes(), name
    # the maps differ by window, so a run that wrote one window everywhere would fail above
    assert len({(tmp_path / "multi" / n).read_bytes() for n in names}) == len(names)
    if engine:
        # one engine for all six maps, sent windows in the order given, each pair forward then reverse
        assert multi_loads == 1
        sent = [call[0] for call in handles[0].calls]
        want = [volume[s : s + DEPTH][::step] for s in (2, 0, 1) for step in (1, -1)]
        assert len(sent) == len(want)
        for got, expected in zip(sent, want):
            np.testing.assert_array_equal(got, expected)


def test_layer_starts_parse_and_refuse_conflicts(tmp_path):
    from vesuvius.ink_detection.inference.infer import parse_layer_starts, resolve_layer_runs

    assert parse_layer_starts("21:24") == [21, 22, 23]
    assert parse_layer_starts("5,3,1") == [5, 3, 1]
    for bad in ("4:4", "1,1", "a:b", "-1:2", "5,-3"):
        with pytest.raises(ValueError):
            parse_layer_starts(bad)
    base = [str(tmp_path / "in.zarr"), str(tmp_path / "m.pth"), str(tmp_path / "o.tif")]
    for extra in (
        ["--layer-starts", "0:2", "--layer-start", "0"],
        ["--layer-starts", "0:2", "--layer-end", "3"],
        ["--layer-starts", "3,3"],
        ["--layer-starts", "0:0"],
    ):
        with pytest.raises(SystemExit):
            parse_args([*base, *extra])
    with pytest.raises(SystemExit):
        parse_args([str(tmp_path / "m.pth"), "--folder", str(tmp_path), "--layer-starts", "0:2"])
    args = normalize_inference_paths(parse_args([*base, "--layer-starts", "20,4"]))
    runs = resolve_layer_runs(args, 17)
    assert [(w.layer_start, w.layer_end, w.layer_starts, o.name) for w, o in runs] == [
        (20, 37, None, "o_L20.tif"),
        (4, 21, None, "o_L4.tif"),
    ]
    # the caller's namespace is left as parsed
    assert args.layer_starts == [20, 4] and args.layer_start is None
    # without --layer-starts, the one run is the command itself
    plain = normalize_inference_paths(parse_args(base))
    assert resolve_layer_runs(plain, 17) == [(plain, plain.output_tiff)]


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("starts", ["1000", "0,1000", "2:5"])
def test_layer_starts_refuse_windows_past_the_input_before_any_work(tmp_path, monkeypatch, engine, starts):
    # 6 layers hold starts 0..3 at the 3-layer test depth; a window past that would clamp to nothing
    # and fall back to the full depth under its own name, so the whole list is refused before the
    # model or engine loads and before any map is written
    import vesuvius.ink_detection.inference.infer as infer_module

    surface, _ = _write_surface(tmp_path / "surface.zarr", layers=6)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    loads = []
    real_configure = infer_module.configure_model

    def counting_configure(*a, **k):
        loads.append("configure")
        return real_configure(*a, **k)

    def factory(*_):
        loads.append("engine")
        return InkrsEngine(_fake_inkrs(None), _WindowHandle(), device_index=0)

    def counting_make(config):
        loads.append("make_model")
        return real_make(config)

    real_make = infer_module.make_model
    monkeypatch.setattr(infer_module, "configure_model", counting_configure)
    monkeypatch.setattr(infer_module, "make_model", counting_make)
    out = tmp_path / "multi" / "map.tif"
    args = normalize_inference_paths(
        parse_args([str(surface), str(checkpoint), str(out), *COMMON, "--direction", "both", "--layer-starts", starts])
    )
    with pytest.raises(ValueError, match="end past"):
        run(args, engine_factory=factory if engine else None)
    # only the reference contract was read; no engine, no PyTorch model build, no output
    assert loads == ["configure"]
    assert not out.parent.exists() or not any(out.parent.iterdir())
