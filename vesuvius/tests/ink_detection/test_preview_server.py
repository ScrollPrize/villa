"""The preview server must return what flat inference writes for the same pixels."""

from __future__ import annotations

from contextlib import contextmanager
import http.client
import inspect
import io
import json
import socket
import threading
from urllib.parse import urlencode

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from vc_reader.reader import ZarrWriter

import vesuvius.ink_detection.inference.infer as infer
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference import preview_server
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from vesuvius.ink_detection.models.model import make_model

from .helpers import config_mapping


PATCH, STRIDE, DEPTH = 16, 8, 5
CANVAS_HW = (45, 61)  # neither (45 - 16) nor (61 - 16) is a multiple of 8: flush last patches


class TinyInkModel(nn.Module):
    """A small fixed network whose output depends on layer order and neighbors."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(1234)
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.head = nn.Conv2d(4 * depth, 1, kernel_size=3, padding=1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)

    def forward(self, image):
        features = torch.tanh(self.conv(image))
        batch, channels, depth, height, width = features.shape
        return {"ink": self.head(features.reshape(batch, channels * depth, height, width))}


def tiny_configured(patch: int = PATCH, depth: int = DEPTH) -> infer.ConfiguredModel:
    return infer.ConfiguredModel(
        model=TargetModel(TinyInkModel(depth)).eval(),
        patch_size=patch,
        input_depth=depth,
        preprocessing="tifxyz_robust",
        amp_dtype=None,
    )


def random_stack(dtype=np.uint8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    high = 256 if dtype == np.uint8 else 65536
    stack = rng.integers(0, high, (DEPTH, *CANVAS_HW), dtype=dtype)
    stack[:, :, 45:] = 0  # the flush column of patches is all zero and skipped
    stack[:, 30:, 20:36] = 0  # patches that are partly zero (normalization fallbacks)
    return stack


def villa_roi_reference(configured, stack, roi, *, stride=STRIDE, direction="forward"):
    """Flat inference's --roi path (infer_single_zarr) on an in-memory array, before quantization."""

    depth, height, width = stack.shape
    patch = configured.patch_size
    reader = infer.FlatPatchReader(
        input_path="memory", resolution="0", depth_axis_first=True, height=height, width=width,
        layer_indices=infer.select_layer_indices(
            depth, layer_start=None, layer_end=None, output_depth=configured.input_depth, direction=direction
        ),
        output_depth=configured.input_depth, preprocessing=configured.preprocessing, array=stack,
    )
    blocks = infer.iter_blocks((height, width), patch, stride, roi_bounds=roi)
    shape = (roi[1] - roi[0], roi[3] - roi[2])
    chunks = (min(patch, shape[0]), min(patch, shape[1]))
    probability = np.zeros(shape, np.float32)
    weight = np.zeros(shape, np.float32)
    accumulator = infer.ChunkAccumulator(
        shape=shape, chunk_shape=chunks, prob_sum_store=probability, weight_sum_store=weight,
        contribution_counts=infer.compute_chunk_contribution_counts(
            [infer.clip_block_to_roi(block, roi) for block in blocks], chunk_shape=chunks
        ),
    )
    dataset = infer.FlatBlockDataset(
        reader=reader, blocks=blocks, patch_size=patch, preprocessing=configured.preprocessing
    )
    blend_mode = infer.resolve_blend_mode("auto", stride=stride, patch_size=patch)
    infer.run_block_inference(
        loader=DataLoader(dataset, batch_size=1), model=configured.model, accumulator=accumulator,
        weight_map=infer.compute_importance_map_2d(patch_size=(patch, patch), mode=blend_mode).numpy(),
        mask=None, device=torch.device("cpu"), amp_dtype=None, tta_axes=(), tta_batch_size=None, roi_bounds=roi,
    )
    accumulator.flush_remaining()
    return infer.normalize_probability_sums(probability, weight)


def floor255(probability: np.ndarray) -> np.ndarray:
    return infer.probabilities_to_uint8(probability.copy())


@contextmanager
def serving(engine):
    server = preview_server.make_server(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def call(port, method, path, params=None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(method, path + ("?" + urlencode(params) if params else ""), body=body)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def view_params(stack, view, *, key="segment-a", variant="f", fmt="f32"):
    x, y, w, h = view
    return {"key": key, "cw": stack.shape[2], "ch": stack.shape[1], "x": x, "y": y, "w": w, "h": h,
            "variant": variant, "format": fmt}


def upload(stack, need, params, dtype="uint8"):
    x, y, w, h = need["x"], need["y"], need["w"], need["h"]
    window = np.ascontiguousarray(stack[:, y : y + h, x : x + w]).astype("<u1" if dtype == "uint8" else "<u2")
    return dict(params, sx=x, sy=y, sw=w, sh=h, dtype=dtype), window.tobytes()


def fetch(port, stack, view, *, dtype="uint8", **kwargs):
    """Ask for a view, upload what the server needs, and decode the 200 reply."""

    params = view_params(stack, view, **kwargs)
    status, headers, data = call(port, "POST", "/preview", params)
    if status == 409:
        params, body = upload(stack, json.loads(data)["need"], params, dtype)
        status, headers, data = call(port, "POST", "/preview", params, body)
    assert status == 200, data
    shape = tuple(int(value) for value in headers["X-Shape"].split(","))
    assert shape == (view[3], view[2])
    if params["format"] == "f32":
        image = np.frombuffer(data, dtype="<f4").reshape(shape)
    elif params["format"] == "u8":
        image = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    else:
        image = np.asarray(Image.open(io.BytesIO(data)))
    return image, headers


def test_need_rectangle_of_the_worked_example():
    assert infer._sliding_positions_1d(300, 128, 64) == [0, 64, 128, 172]
    assert infer._sliding_positions_1d(200, 128, 64) == [0, 64, 72]
    engine = preview_server.PreviewEngine(tiny_configured(patch=128, depth=17))
    params = {"key": "k", "cw": 300, "ch": 200, "x": 100, "y": 50, "w": 120, "h": 100,
              "variant": "f", "format": "u8"}
    with serving(engine) as port:
        status, _, data = call(port, "POST", "/preview", params)
    assert status == 409
    assert json.loads(data) == {"need": {"x": 0, "y": 0, "w": 300, "h": 200}, "depth": 17, "missing": 12}
    assert engine.cache.stats()["tiles"] == 0  # nothing is computed for a 409


@pytest.mark.parametrize("view", [
    (13, 11, 20, 17),  # interior, off the patch grid
    (40, 30, 21, 15),  # bottom-right edge: flush patches and an all-zero patch
    (0, 0, 61, 45),  # the whole canvas
    (60, 44, 1, 1),  # one corner pixel
    (3, 27, 50, 6),  # crosses the partly zero region
])
def test_preview_equals_flat_inference_roi_in_f32_and_u8(view):
    configured = tiny_configured()
    stack = random_stack()
    x, y, w, h = view
    expected = villa_roi_reference(configured, stack, (y, y + h, x, x + w))
    with serving(preview_server.PreviewEngine(configured)) as port:
        params = view_params(stack, view)
        status, _, data = call(port, "POST", "/preview", params)
        assert status == 409
        params, body = upload(stack, json.loads(data)["need"], params)
        status, headers, data = call(port, "POST", "/preview", params, body)
        assert status == 200
        assert headers["X-Backend"] == "torch:cpu"
        computed, cached = (int(value) for value in headers["X-Tiles"].split(","))
        assert computed == len(infer.iter_blocks(CANVAS_HW, PATCH, STRIDE, roi_bounds=(y, y + h, x, x + w)))
        assert cached == 0
        f32 = np.frombuffer(data, dtype="<f4").reshape(h, w)
        u8, _ = fetch(port, stack, view, fmt="u8")
        png, _ = fetch(port, stack, view, fmt="png")
    np.testing.assert_array_equal(f32, expected)
    np.testing.assert_array_equal(u8, floor255(expected))
    np.testing.assert_array_equal(png, floor255(expected))
    assert expected.any() or view == (60, 44, 1, 1)


def test_uint16_upload_equals_flat_inference_on_uint16():
    configured = tiny_configured()
    stack = random_stack(np.uint16, seed=3)
    view = (5, 9, 40, 30)
    with serving(preview_server.PreviewEngine(configured)) as port:
        result, _ = fetch(port, stack, view, dtype="uint16")
    np.testing.assert_array_equal(result, villa_roi_reference(configured, stack, (9, 39, 5, 45)))


def test_reversed_variant_equals_reverse_direction_and_differs_from_forward():
    configured = tiny_configured()
    stack = random_stack(seed=1)
    view = (6, 4, 48, 37)
    roi = (4, 41, 6, 54)
    with serving(preview_server.PreviewEngine(configured)) as port:
        reversed_view, _ = fetch(port, stack, view, variant="r")
        forward_view, _ = fetch(port, stack, view, variant="f")
    np.testing.assert_array_equal(reversed_view, villa_roi_reference(configured, stack, roi, direction="reverse"))
    np.testing.assert_array_equal(forward_view, villa_roi_reference(configured, stack, roi))
    covered = forward_view > 0
    assert np.abs(reversed_view - forward_view).max() > 0.05
    assert np.mean(floor255(reversed_view)[covered] != floor255(forward_view)[covered]) > 0.5


def test_pan_reuses_cached_tiles_and_needs_only_the_new_strip():
    configured = tiny_configured()
    stack = random_stack(seed=2)
    engine = preview_server.PreviewEngine(configured)
    first, panned = (2, 3, 24, 30), (2 + 2 * STRIDE, 3, 24, 30)

    def positions(view):
        x, y, w, h = view
        return {(b.y0, b.x0): b for b in infer.iter_blocks(CANVAS_HW, PATCH, STRIDE, roi_bounds=(y, y + h, x, x + w))}

    old, new = positions(first), positions(panned)
    fresh = [block for position, block in new.items() if position not in old]
    assert fresh and len(fresh) < len(new)
    with serving(engine) as port:
        fetch(port, stack, first)
        params = view_params(stack, panned)
        status, _, data = call(port, "POST", "/preview", params)
        assert status == 409
        reply = json.loads(data)
        assert reply["missing"] == len(fresh)
        assert reply["need"] == {
            "x": min(b.x0 for b in fresh), "y": min(b.y0 for b in fresh),
            "w": max(b.x0 + b.valid_w for b in fresh) - min(b.x0 for b in fresh),
            "h": max(b.y0 + b.valid_h for b in fresh) - min(b.y0 for b in fresh),
        }
        assert reply["need"]["w"] < 24 + 2 * PATCH  # a strip, not the whole neighborhood
        params, body = upload(stack, reply["need"], params)
        status, headers, data = call(port, "POST", "/preview", params, body)
        assert status == 200
        assert headers["X-Tiles"] == f"{len(fresh)},{len(new) - len(fresh)}"
        health = json.loads(call(port, "GET", "/health")[2])
    x, y, w, h = panned
    np.testing.assert_array_equal(
        np.frombuffer(data, dtype="<f4").reshape(h, w),
        villa_roi_reference(configured, stack, (y, y + h, x, x + w)),
    )
    # The panned view looked its cached patches up twice: before and with the upload.
    assert health["cache"]["hits"] == 2 * (len(new) - len(fresh))
    assert health["cache"]["tiles"] == len(set(old) | set(new))


@pytest.mark.parametrize("shift,should_match", [(STRIDE // 2, False), (STRIDE, True)])
def test_displaced_grid_control(shift, should_match):
    """The same pixels on a grid moved by half a stride must not equal the reference."""

    configured = tiny_configured()
    stack = random_stack(seed=4)
    stack[:, :, 45:] = 7  # keep every patch nonempty so only the grid differs
    view = (24, 24, 14, 12)  # one patch away from the cropped origin, inside both grids
    x, y, w, h = view
    expected = villa_roi_reference(configured, stack, (y, y + h, x, x + w))
    cropped = np.ascontiguousarray(stack[:, shift:, shift:])
    with serving(preview_server.PreviewEngine(configured)) as port:
        moved, _ = fetch(port, cropped, (x - shift, y - shift, w, h), key="cropped")
    if should_match:
        np.testing.assert_array_equal(moved, expected)
    else:
        assert np.mean(floor255(moved) != floor255(expected)) > 0.5


def test_keys_never_share_patches():
    """A view cached under one key is a miss under another, and each key gets its own data's result."""

    configured = tiny_configured()
    first, second = random_stack(seed=5), random_stack(seed=6)
    view = (9, 7, 30, 20)
    x, y, w, h = view
    engine = preview_server.PreviewEngine(configured)
    with serving(engine) as port:
        result_a, _ = fetch(port, first, view, key="surface-a")
        status, _, data = call(port, "POST", "/preview", view_params(second, view, key="surface-b"))
        assert status == 409, data
        assert json.loads(data)["missing"] == len(infer.iter_blocks(CANVAS_HW, PATCH, STRIDE, roi_bounds=(y, y + h, x, x + w)))
        result_b, _ = fetch(port, second, view, key="surface-b")
        again, headers = fetch(port, first, view, key="surface-a")
    assert headers["X-Tiles"].startswith("0,")  # key A is still cached, untouched by key B
    np.testing.assert_array_equal(result_a, villa_roi_reference(configured, first, (y, y + h, x, x + w)))
    np.testing.assert_array_equal(result_b, villa_roi_reference(configured, second, (y, y + h, x, x + w)))
    np.testing.assert_array_equal(again, result_a)
    assert np.mean(floor255(result_a) != floor255(result_b)) > 0.5


def test_server_binds_loopback_only_with_a_large_backlog():
    assert "host" not in inspect.signature(preview_server.make_server).parameters
    server = preview_server.make_server(preview_server.PreviewEngine(tiny_configured()), port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.request_queue_size >= 64
    finally:
        server.server_close()


def test_a_burst_of_concurrent_connections_is_answered():
    """120 clients connect at once; the stdlib backlog of 5 resets some of them in most runs."""

    clients = 120  # below the backlog, so every connection fits in the queue
    stack = random_stack()
    barrier = threading.Barrier(clients)
    statuses, errors = [], []

    def client(index, port):
        params = view_params(stack, (index % 40, index % 30, 20, 15), key=f"k{index % 4}", fmt="u8")
        barrier.wait()
        try:
            statuses.append(call(port, "POST", "/preview", params)[0])
        except OSError as exc:
            errors.append(repr(exc))

    with serving(preview_server.PreviewEngine(tiny_configured())) as port:
        threads = [threading.Thread(target=client, args=(index, port)) for index in range(clients)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    assert statuses == [409] * clients


def test_canvas_and_upload_limits_are_checked_before_any_work():
    engine = preview_server.PreviewEngine(tiny_configured())
    side = preview_server.MAX_CANVAS_SIDE
    with serving(engine) as port:
        for cw, ch in ((side + 1, 45), (61, side + 1), (10**18, 10**18)):
            status, _, data = call(port, "POST", "/preview", {"key": "k", "cw": cw, "ch": ch, "x": 0, "y": 0,
                                                              "w": 1, "h": 1})
            assert status == 400 and "at most" in json.loads(data)["error"], (cw, ch, data)
        status, _, data = call(port, "POST", "/preview", {"key": "k", "cw": side, "ch": side, "x": 0, "y": 0,
                                                          "w": 1, "h": 1})
        assert status == 409 and json.loads(data)["missing"] == 1
        # A thin view inside max_view_px whose patches would need a 6.9 GB upload is refused from
        # its headers alone: the server answers without reading the body.
        thin = {"key": "thin", "cw": 16_000_000, "ch": 64, "x": 0, "y": 20, "w": 16_000_000, "h": 1,
                "sx": 0, "sy": 5, "sw": 16_000_000, "sh": 2 * PATCH - 1, "dtype": "uint16"}
        assert thin["w"] * thin["h"] <= engine.max_view_px
        length = DEPTH * thin["sh"] * thin["sw"] * 2
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            connection.sendall(f"POST /preview?{urlencode(thin)} HTTP/1.1\r\nHost: test\r\n"
                               f"Content-Length: {length}\r\n\r\n".encode())
            assert connection.recv(64).startswith(b"HTTP/1.0 413")
        status, _, data = call(port, "POST", "/preview", {k: v for k, v in thin.items() if not k.startswith("s")
                                                          and k != "dtype"})
        assert status == 413 and "max_upload_mb" in json.loads(data)["error"]
    # The bound is the view plus one patch on each side, at 2 bytes per voxel.
    exact = (10 + 2 * PATCH) * (10 + 2 * PATCH) * DEPTH * 2
    tight = preview_server.PreviewEngine(tiny_configured(), max_upload_mb=exact / 2**20)
    with serving(tight) as port:
        stack = random_stack()
        assert call(port, "POST", "/preview", view_params(stack, (0, 0, 10, 10)))[0] == 409
        assert call(port, "POST", "/preview", view_params(stack, (0, 0, 11, 10)))[0] == 413
        result, _ = fetch(port, stack, (0, 0, 10, 10))
    np.testing.assert_array_equal(result, villa_roi_reference(tiny_configured(), stack, (0, 10, 0, 10)))


def test_health_and_error_codes():
    engine = preview_server.PreviewEngine(
        tiny_configured(), cache_mb=1, max_view_px=500, model_name="tiny", voxel_um=9.36
    )
    stack = random_stack()
    good = view_params(stack, (0, 0, 10, 10), fmt="u8")
    with serving(engine) as port:
        status, _, data = call(port, "GET", "/health")
        health = json.loads(data)
        assert status == 200
        assert {key: health[key] for key in ("service", "api", "backend", "model", "patch", "stride", "depth",
                                             "voxel_um", "max_view_px", "max_upload_mb")} == {
            "service": "ink-preview", "api": 1, "backend": "torch:cpu", "model": "tiny", "patch": PATCH,
            "stride": STRIDE, "depth": DEPTH, "voxel_um": 9.36, "max_view_px": 500, "max_upload_mb": 1024.0,
        }
        assert set(health["cache"]) == {"tiles", "mb", "cap_mb", "hits", "misses"}
        assert call(port, "GET", "/nowhere")[0] == 404
        assert call(port, "POST", "/nowhere", good)[0] == 404
        bad_queries = [
            {key: value for key, value in good.items() if key != "key"},
            dict(good, key="k" * 201),
            dict(good, cw="wide"),
            dict(good, x=55),  # view leaves the canvas
            dict(good, w=0),
            dict(good, variant="s"),
            dict(good, format="jpeg"),
            dict(good, sx=0, sy=0, sw=10),  # incomplete upload rectangle
            dict(good, sx=0, sy=0, sw=10, sh=10, dtype="float32"),
            dict(good, sx=60, sy=0, sw=10, sh=10),  # upload leaves the canvas
        ]
        for params in bad_queries:
            status, _, data = call(port, "POST", "/preview", params, b"")
            assert status == 400, params
            assert "error" in json.loads(data)
        status, _, data = call(port, "POST", "/preview", good, b"\0" * 10)  # body without a rectangle
        assert status == 400
        with_rectangle = dict(good, sx=0, sy=0, sw=10, sh=10, dtype="uint8")
        status, _, data = call(port, "POST", "/preview", with_rectangle, b"\0" * (DEPTH * 100 - 1))
        assert status == 400 and "needs 500" in json.loads(data)["error"]
        status, _, data = call(port, "POST", "/preview", dict(good, w=30, h=20))
        assert status == 413 and "error" in json.loads(data)
        too_big = dict(good, sx=0, sy=0, sw=61, sh=45)  # more than the view plus one patch each side
        status, _, _ = call(port, "POST", "/preview", too_big, b"\0" * (DEPTH * 61 * 45))
        assert status == 413
        status, _, _ = call(port, "GET", "/health")  # the server is still up
        assert status == 200


def test_server_errors_are_500_and_the_server_stays_up():
    class Broken(nn.Module):
        def forward(self, image):
            raise RuntimeError("model failed")

    configured = infer.ConfiguredModel(
        model=Broken(), patch_size=PATCH, input_depth=DEPTH, preprocessing="tifxyz_robust", amp_dtype=None
    )
    stack = random_stack()
    with serving(preview_server.PreviewEngine(configured)) as port:
        params = view_params(stack, (0, 0, 8, 8))
        status, _, data = call(port, "POST", "/preview", params)
        params, body = upload(stack, json.loads(data)["need"], params)
        status, _, data = call(port, "POST", "/preview", params, body)
        assert status == 500 and "model failed" in json.loads(data)["error"]
        assert call(port, "GET", "/health")[0] == 200


def test_tile_cache_is_least_recently_used_under_its_cap():
    tile = np.zeros((16, 16), np.float32)
    size = tile.nbytes + preview_server._ENTRY_BYTES
    cache = preview_server.TileCache(2 * size)
    cache.put(("k", "f", 0, 0), (16, 16), tile)
    cache.put(("k", "f", 0, 8), (16, 16), tile)
    assert cache.get(("k", "f", 0, 0), (16, 16))[0]
    cache.put(("k", "f", 0, 16), (16, 16), tile)  # evicts (0, 8), the least recently used
    assert not cache.get(("k", "f", 0, 8), (16, 16))[0]
    assert cache.get(("k", "f", 0, 16), (16, 16))[0]
    assert not cache.get(("k", "f", 0, 16), (8, 16))[0]  # another canvas edge is not a hit
    assert not cache.get(("k", "r", 0, 0), (16, 16))[0]  # variants never share tiles
    stats = cache.stats()
    assert (stats["tiles"], stats["hits"], stats["misses"]) == (2, 2, 3)
    assert stats["mb"] <= stats["cap_mb"]
    disabled = preview_server.TileCache(0)
    disabled.put(("k", "f", 0, 0), (16, 16), None)
    assert disabled.stats()["tiles"] == 0


def test_resolve_blend_mode_matches_the_cli_rule():
    assert infer.resolve_blend_mode("auto", stride=64, patch_size=128) == "hann"
    assert infer.resolve_blend_mode("auto", stride=128, patch_size=128) == "constant"
    assert infer.resolve_blend_mode("gaussian", stride=64, patch_size=128) == "gaussian"


@pytest.mark.parametrize("direction,variant", [("forward", "f"), ("reverse", "r")])
def test_checkpoint_server_equals_flat_inference_cli_roi(tmp_path, direction, variant):
    """Load a checkpoint like infer.py and compare with the TIFF that infer.py --roi writes."""

    mapping = config_mapping("vesuvius_unet_2p5d", depth=3, side=16)
    mapping["image_normalization"] = "robust_mad"
    torch.manual_seed(7)
    model = make_model(InkConfig.from_mapping(mapping))
    checkpoint = tmp_path / "tiny-ink.pth"
    torch.save({"config": mapping, "model": model.state_dict()}, checkpoint)
    rng = np.random.default_rng(5)
    volume = rng.integers(0, 256, (3, *CANVAS_HW), dtype=np.uint8)
    volume[:, :, 50:] = 0
    input_path = tmp_path / "surface.zarr"
    array = ZarrWriter.create(input_path, shape=volume.shape, chunks=(3, 16, 16), dtype="u1")
    array[:] = volume
    output = tmp_path / "roi.tif"
    roi = (7, 45, 19, 61)  # reaches the bottom and right edges
    assert infer.main([
        str(input_path), str(checkpoint), str(output), "--workers", "0", "--no-compile",
        "--device", "cpu", "--direction", direction, "--roi", *(str(value) for value in roi),
    ]) == 0
    expected = tifffile.imread(output)

    engine = preview_server.load_engine(
        preview_server.parse_args(["--checkpoint", str(checkpoint), "--device", "cpu"])
    )
    assert engine.health()["model"] == "tiny-ink" and engine.health()["voxel_um"] is None
    stated = preview_server.parse_args(["--checkpoint", str(checkpoint), "--voxel-um", "9.36"])
    assert preview_server.load_engine(stated).health()["voxel_um"] == 9.36
    assert (engine.patch, engine.stride, engine.depth, engine.blend_mode) == (16, 8, 3, "hann")
    with serving(engine) as port:
        result, _ = fetch(port, volume, (roi[2], roi[0], roi[3] - roi[2], roi[1] - roi[0]),
                          variant=variant, fmt="u8")
    assert expected.shape == result.shape
    np.testing.assert_array_equal(result, expected)
    assert expected.any()
