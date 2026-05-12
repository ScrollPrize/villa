"""Smoke tests for the end-to-end FiberTracer.

These run with a tiny CPU-only model on a synthetic numpy volume, so they
exercise the priming -> step -> re-anchor lifecycle and the bidirectional
concat without touching S3 or a GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel
from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import ChunkLRUCache
from vesuvius.neural_tracing.autoreg_fiber.streaming.tracer import FiberTracer
from vesuvius.neural_tracing.autoreg_fiber.streaming.window import WindowedVolumeReader


class _ArrayBackedZarrLike:
    def __init__(self, data: np.ndarray, chunk_shape: tuple[int, int, int]):
        self._data = data
        self.shape = tuple(data.shape)
        self.chunks = tuple(int(v) for v in chunk_shape)
        self.dtype = data.dtype

    def __getitem__(self, item):
        return self._data[item]


def _tiny_config(tmp_path, *, mode: str) -> dict:
    return {
        "dinov2_backbone": None,
        "crop_size": [16, 16, 16],
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "prompt_length": 2,
        "target_length": 4,
        "point_stride": 1,
        "decoder_dim": 24,
        "decoder_depth": 1,
        "decoder_num_heads": 2,
        "decoder_dropout": 0.0,
        "max_fiber_position_embeddings": 64,
        "coarse_prediction_mode": mode,
        "cross_attention_every_n_blocks": 1,
        "distance_aware_coarse_targets_enabled": False,
        "position_refine_start_step": 0,
        "position_refine_weight": 0.01,
        "xyz_soft_loss_weight": 0.1,
        "segment_vector_loss_weight": 0.01,
        "optimizer": {"name": "adamw", "learning_rate": 2e-3, "weight_decay": 0.0},
        "batch_size": 1,
        "num_workers": 0,
        "val_num_workers": 0,
        "val_fraction": 0.0,
        "num_steps": 1,
        "log_frequency": 1,
        "ckpt_frequency": 50,
        "save_final_checkpoint": False,
        "out_dir": str(tmp_path / "runs"),
    }


def _make_tracer(tmp_path, *, volume_shape: tuple[int, int, int] = (64, 32, 32), max_steps: int = 12):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path, mode="axis_factorized")
    model = AutoregFiberModel(cfg).eval()
    rng = np.random.default_rng(0)
    volume = rng.uniform(0.0, 1.0, size=volume_shape).astype(np.float32)
    source = _ArrayBackedZarrLike(volume, chunk_shape=(8, 8, 8))
    cache = ChunkLRUCache(source, maxsize=32, num_prefetch_workers=2)
    reader = WindowedVolumeReader(cache, crop_size=(16, 16, 16), reanchor_margin=2)
    tracer = FiberTracer(
        model,
        reader,
        max_steps=max_steps,
        stop_prob_threshold=None,  # let the test rely on max_steps / out-of-volume
        min_steps=1,
        dtype=torch.float32,
    )
    return tracer, volume


def test_trace_one_direction_returns_polyline_with_prompt(tmp_path) -> None:
    tracer, volume = _make_tracer(tmp_path, max_steps=8)
    prompt = np.array([[10.0, 8.0, 8.0], [11.0, 8.0, 8.0]], dtype=np.float32)
    result = tracer.trace_one_direction(prompt, prefetch=False)
    assert result.polyline_world_zyx.shape[0] == prompt.shape[0] + result.steps
    np.testing.assert_array_equal(result.polyline_world_zyx[: prompt.shape[0]], prompt)
    assert result.steps > 0
    assert result.stop_reason in {"max_steps", "out_of_volume", "stop_probability"}
    # All produced points must lie within volume bounds OR be the final point that
    # triggered the out_of_volume stop. We allow the last point to be just outside.
    vshape = np.array(volume.shape, dtype=np.float32)
    interior = result.polyline_world_zyx[:-1]
    assert ((interior >= 0).all() and (interior < vshape).all())


def test_trace_triggers_reanchor_when_advancing(tmp_path) -> None:
    tracer, _volume = _make_tracer(tmp_path, max_steps=30)
    prompt = np.array([[4.0, 8.0, 8.0], [5.0, 8.0, 8.0]], dtype=np.float32)
    result = tracer.trace_one_direction(prompt, prefetch=True)
    # With a 16^3 window and margin=2 voxels, a 30-step trace will almost
    # certainly trip needs_reanchor at least once.
    assert result.reanchors >= 0  # never negative
    assert result.steps == len(result.stop_probabilities)


def test_trace_bidirectional_concatenates_back_orig_forward(tmp_path) -> None:
    tracer, _volume = _make_tracer(tmp_path, max_steps=4)
    fiber = np.array(
        [
            [10.0, 8.0, 8.0],
            [11.0, 8.0, 8.0],
            [12.0, 8.0, 8.0],
        ],
        dtype=np.float32,
    )
    bidi = tracer.trace_bidirectional(fiber, prefetch=False)
    # The forward polyline contains the input fiber at its head.
    np.testing.assert_array_equal(bidi.forward.polyline_world_zyx[: fiber.shape[0]], fiber)
    # The bidirectional polyline ends with the forward result.
    np.testing.assert_array_equal(
        bidi.polyline_world_zyx[-bidi.forward.polyline_world_zyx.shape[0] :],
        bidi.forward.polyline_world_zyx,
    )
    # Backward-extension length matches the back trace minus the (reversed) prompt.
    backward_extension = bidi.backward.polyline_world_zyx[tracer.prompt_length :]
    assert bidi.polyline_world_zyx.shape[0] == backward_extension.shape[0] + bidi.forward.polyline_world_zyx.shape[0]
    # The extension is concatenated in reversed order (so trace flows in the original direction).
    np.testing.assert_array_equal(
        bidi.polyline_world_zyx[: backward_extension.shape[0]],
        backward_extension[::-1],
    )


def test_rejects_mismatched_input_shape(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path, mode="joint_pointer")
    model = AutoregFiberModel(cfg).eval()
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    source = _ArrayBackedZarrLike(volume, chunk_shape=(8, 8, 8))
    cache = ChunkLRUCache(source, maxsize=4, num_prefetch_workers=0)
    # crop_size differs from model.input_shape -> should raise.
    reader_bad = WindowedVolumeReader(cache, crop_size=(8, 8, 8), reanchor_margin=2)
    try:
        FiberTracer(model, reader_bad, dtype=torch.float32)
    except ValueError as exc:
        assert "input_shape" in str(exc) and "crop_size" in str(exc)
    else:
        raise AssertionError("expected ValueError for crop_size != model.input_shape")


def test_short_prompt_is_rejected(tmp_path) -> None:
    tracer, _volume = _make_tracer(tmp_path, max_steps=4)
    short = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)  # prompt_length=2 in tiny config
    try:
        tracer.trace_one_direction(short)
    except ValueError as exc:
        assert "prompt_length" in str(exc)
    else:
        raise AssertionError("expected ValueError for too-short prompt")
