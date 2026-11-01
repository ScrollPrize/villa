from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run import inference
from vesuvius.models.run.blending import InMemoryBlender, merge_inference_outputs
from vesuvius.models.run.finalize_outputs import FinalizeConfig
from vesuvius.models.run.inference import Inferer, build_parser
from vesuvius.models.run.patch_writer import BoundedPatchWriter

PATCH = (16, 16, 16)
VOLUME = (40, 36, 44)  # not a multiple of the patch or chunk size, so edge chunks are partial


def _patches(num_classes):
    """Overlapping patch logits on a stride-8 grid, as predict would produce them (float16)."""
    rng = np.random.default_rng(0)
    starts = [range(0, v - p + 1, 8) for v, p in zip(VOLUME, PATCH)]
    coords = np.array([(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]], dtype=np.int32)
    logits = rng.normal(0.0, 3.0, (len(coords), num_classes, *PATCH)).astype(np.float16)
    logits[3] = 0  # an empty patch: skipped by blending, adds no weight
    return coords, logits


def _write_parts(parent: Path, coords, logits):
    """The two stores vesuvius.predict writes for one part."""
    store = open_zarr(str(parent / "logits_part_0.zarr"), mode="w", shape=logits.shape,
                      chunks=(1, logits.shape[1], *PATCH), dtype=np.float16, write_empty_chunks=False)
    store[:] = logits
    store.attrs["patch_size"] = list(PATCH)
    store.attrs["original_volume_shape"] = list(VOLUME)
    coords_store = open_zarr(str(parent / "coordinates_part_0.zarr"), mode="w", shape=coords.shape,
                             chunks=coords.shape, dtype=np.int32)
    coords_store[:] = coords


def _read(path):
    return np.asarray(zarr.open(str(path), mode="r")[:])


@pytest.mark.parametrize(
    ("num_classes", "finalize_config", "through_writer"),
    [
        (2, FinalizeConfig(mode="binary"), False),
        (2, FinalizeConfig(mode="binary", threshold=0.3), False),
        (3, FinalizeConfig(mode="multiclass"), False),
        (2, None, False),  # blended float16 logits, as blend_logits writes them
        (2, FinalizeConfig(mode="binary"), True),  # the way predict feeds it: one writer thread
    ],
)
def test_in_memory_blend_matches_blend_and_finalize(tmp_path: Path, num_classes, finalize_config,
                                                    through_writer) -> None:
    coords, logits = _patches(num_classes)
    parent = tmp_path / "parts"
    parent.mkdir()
    _write_parts(parent, coords, logits)

    staged = tmp_path / "staged.zarr"
    merge_inference_outputs(str(parent), str(staged), chunk_size=(12, 12, 12), num_workers=1,
                            verbose=False, finalize_config=finalize_config)

    blender = InMemoryBlender(coords, PATCH, num_classes, VOLUME)
    if through_writer:
        with BoundedPatchWriter(blender, max_workers=1, max_inflight=4) as writer:
            for index, patch in enumerate(logits):
                writer.submit(index, patch)
    else:
        for index, patch in enumerate(logits):
            blender.add(index, patch)
    in_memory = tmp_path / "in_memory.zarr"
    blender.write(str(in_memory), chunk_size=(12, 12, 12), finalize_config=finalize_config, verbose=False)

    assert blender.patches_skipped_empty == 1
    expected, actual = _read(staged), _read(in_memory)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_in_memory_blend_covers_only_the_patch_extent(tmp_path: Path) -> None:
    coords = np.array([(8, 8, 8)], dtype=np.int32)
    origin, shape = InMemoryBlender.extent(coords, PATCH, VOLUME)
    assert origin == (8, 8, 8) and shape == PATCH
    assert InMemoryBlender.required_bytes(coords, PATCH, 2, VOLUME) == 3 * 4 * 16 ** 3

    blender = InMemoryBlender(coords, PATCH, 2, VOLUME)
    blender.add(0, np.ones((2, *PATCH), dtype=np.float16))
    out = tmp_path / "out.zarr"
    blender.write(str(out), finalize_config=FinalizeConfig(mode="binary"), verbose=False)
    result = _read(out)
    assert result.shape == (1, *VOLUME)
    assert result[0, 8:24, 8:24, 8:24].min() > 0
    # Only the 16^3 output chunks the patch touches are written; the rest stay at the fill value.
    assert not result[0, 32:].any() and not result[0, :, 32:].any() and not result[0, :, :, 32:].any()


def _required_args(tmp_path: Path) -> list[str]:
    return ["--model_path", str(tmp_path / "model.pth"), "--input_dir", "https://example.org/volume.zarr",
            "--output_dir", str(tmp_path / "out")]


def test_blend_in_memory_is_off_by_default(tmp_path: Path) -> None:
    args = build_parser().parse_args(_required_args(tmp_path))
    assert args.blend_in_memory is None


def test_main_passes_in_memory_blend_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    output = str(tmp_path / "blended.zarr")

    class FakeInferer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def infer(self):
            return output, None

    monkeypatch.setattr(inference, "Inferer", FakeInferer)
    monkeypatch.setattr(sys, "argv", [
        "vesuvius.predict", *_required_args(tmp_path), "--device", "cpu",
        "--blend-in-memory", output, "--threshold", "--threshold_value", "0.3",
        "--blend_chunk_size", "128,128,128", "--blend_compression_level", "3",
    ])

    assert inference.main() == 0
    config = captured["in_memory_blend"]
    assert config["output_path"] == output
    assert config["chunk_size"] == (128, 128, 128)
    assert config["compression_level"] == 3
    assert config["sigma_scale"] == 8.0
    assert config["finalize_config"].mode == "binary"
    assert config["finalize_config"].threshold == 0.3


def test_main_rejects_blend_in_memory_with_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", [
        "vesuvius.predict", *_required_args(tmp_path), "--device", "cpu",
        "--blend_in_memory", str(tmp_path / "blended.zarr"), "--num_parts", "2",
    ])
    with pytest.raises(SystemExit):
        inference.main()


def test_inferer_rejects_in_memory_blend_with_parts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_parts"):
        Inferer(model_path=str(tmp_path / "model.pth"), input_dir="https://example.org/volume.zarr",
                output_dir=str(tmp_path / "out"), device="cpu", num_parts=2,
                in_memory_blend={"output_path": str(tmp_path / "blended.zarr")})
