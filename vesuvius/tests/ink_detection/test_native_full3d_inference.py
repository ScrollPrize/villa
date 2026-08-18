"""Native volume planning, prediction, and OME-Zarr lifecycle tests."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from vesuvius.ink_detection.inference import infer_full3d_tifxyz as native


def _plan(**overrides) -> native.NativePlan:
    values = {
        "volume_path": "memory://native",
        "array_shape_zyx": (8, 8, 8),
        "chunk_shape_zyx": (4, 4, 4),
        "x": np.array([0.25, 4.0, 7.9, np.nan, -1.0, 8.0]),
        "y": np.array([0.25, 0.0, 7.9, 1.0, 1.0, 1.0]),
        "z": np.array([0.25, 0.0, 7.9, 1.0, 1.0, 1.0]),
        "valid": np.array([True, True, True, False, False, True]),
        "patch_size_zyx": (5, 5, 5),
        "overlap": 0.5,
        "chunk_halo": 0,
        "write_region": "occupied",
        "max_target_chunks": None,
    }
    values.update(overrides)
    return native.build_native_plan(**values)


def test_native_weight_selection_uses_only_ema_then_model():
    ema = {"weight": torch.tensor(1.0)}
    flat_alias = {"weight": torch.tensor(2.0)}
    model = {"weight": torch.tensor(3.0)}

    selected, state = native.select_native_inference_weights(
        {"ema_model": ema, "state_dict": flat_alias, "model": model}
    )
    assert selected == "ema_model"
    assert state is ema

    selected, state = native.select_native_inference_weights(
        {"state_dict": flat_alias, "model": model}
    )
    assert selected == "model"
    assert state is model


def test_negative_resolution_fails_before_native_volume_open(monkeypatch):
    config = SimpleNamespace(
        data=SimpleNamespace(mode="full_3d"),
        model=SimpleNamespace(crop_size=(3, 3, 3)),
    )
    monkeypatch.setattr(
        native,
        "_load_checkpoint_config",
        lambda checkpoint: ({}, config),
    )

    with pytest.raises(ValueError, match="must be >= 0"):
        monkeypatch.setattr(
            native,
            "_open_shared_volume",
            lambda *args, **kwargs: pytest.fail(
                "negative resolution opened the volume"
            ),
        )
        native._plan_from_args(
            SimpleNamespace(checkpoint=Path("unused"), resolution="-1")
        )


def test_cli_accepts_both_flag_spellings_and_explicit_workers_alias(tmp_path):
    common = [str(tmp_path), str(tmp_path / "model.ckpt"), str(tmp_path / "out")]
    hyphen = native.parse_args(
        [
            *common,
            "--num-workers",
            "0",
            "--downsample-workers",
            "3",
            "--max-target-chunks",
            "2",
            "--no-compile",
        ]
    )
    underscore = native.parse_args(
        [
            *common,
            "--num_workers",
            "0",
            "--downsample_workers",
            "3",
            "--max_target_chunks",
            "2",
            "--no_compile",
        ]
    )
    legacy = native.parse_args([*common, "--workers", "7"])
    assert hyphen.num_workers == underscore.num_workers == 0
    assert hyphen.downsample_workers == underscore.downsample_workers == 3
    assert hyphen.max_target_chunks == underscore.max_target_chunks == 2
    assert hyphen.compile_model is underscore.compile_model is False
    assert legacy.num_workers == 7
    assert native.parse_args([*common, "--overlap", "1"]).overlap == 1.0
    with pytest.raises(SystemExit):
        native.parse_args([*common, "--foreground-channel", "0"])


def test_volume_source_is_anchored_to_tifxyz_directory(tmp_path, monkeypatch):
    tifxyz = tmp_path / "map"
    tifxyz.mkdir()
    source = tifxyz / "volume_source.txt"

    with pytest.raises(ValueError, match="empty"):
        source.write_text("\n", encoding="utf-8")
        native.read_volume_source(tifxyz)

    source.write_text("https://example.invalid/a.zarr\n", encoding="utf-8")
    assert native.read_volume_source(tifxyz) == "https://example.invalid/a.zarr"

    absolute = tmp_path / "absolute.zarr"
    source.write_text(str(absolute), encoding="utf-8")
    assert native.read_volume_source(tifxyz) == str(absolute)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    source.write_text("../native/volume.zarr", encoding="utf-8")
    assert native.read_volume_source(tifxyz) == str(
        (tifxyz / "../native/volume.zarr").resolve()
    )

    source.unlink()
    with pytest.raises(FileNotFoundError, match="volume_source.txt"):
        native.read_volume_source(tifxyz)


def test_occupied_plan_handles_boundaries_halo_stride_and_truncation():
    occupied = {
        native.ChunkId(0, 0, 0),
        native.ChunkId(0, 0, 1),
        native.ChunkId(1, 1, 1),
    }
    plan = _plan()
    assert plan.occupied_chunks == occupied
    assert plan.target_chunks == occupied
    assert plan.stride_zyx == (2, 2, 2)
    assert list(plan.patches) == sorted(set(plan.patches))
    assert set(plan.contribution_counts) == occupied
    assert all(count > 0 for count in plan.contribution_counts.values())

    halo = _plan(chunk_halo=1, write_region="expanded")
    assert halo.expanded_chunks == {
        native.ChunkId(z, y, x)
        for z in range(2)
        for y in range(2)
        for x in range(2)
    }
    truncated = _plan(chunk_halo=1, write_region="expanded", max_target_chunks=2)
    assert truncated.target_chunks == frozenset(sorted(halo.expanded_chunks)[:2])


def test_patch_lattice_covers_small_volume_and_validates_direct_calls():
    small = _plan(
        array_shape_zyx=(3, 4, 2),
        chunk_shape_zyx=(2, 2, 2),
        x=np.array([1.9]),
        y=np.array([3.9]),
        z=np.array([2.9]),
        valid=np.array([True]),
        patch_size_zyx=(7, 6, 5),
        overlap=0.0,
    )
    assert small.patches == (native.PatchSpec(0, 0, 0),)
    assert small.contribution_counts == {native.ChunkId(1, 1, 0): 1}

    for field, bad in (
        ("array_shape_zyx", (0, 2, 2)),
        ("chunk_shape_zyx", (1, 2)),
        ("patch_size_zyx", (1, -1, 1)),
        ("overlap", 1.0),
        ("chunk_halo", -1),
        ("max_target_chunks", 0),
        ("write_region", "everything"),
    ):
        with pytest.raises(ValueError):
            _plan(**{field: bad})


def test_probability_rules_resize_and_exact_eight_way_tta():
    shape = (3, 4, 5)
    one = native.logits_to_probabilities(
        torch.zeros((1, 1, 1, 2, 2)),
        patch_size_zyx=shape,
    )
    assert one.shape == (1, 1, *shape)
    assert torch.equal(one, torch.full_like(one, 0.5))

    variants = native.tta_variants(True)
    assert variants == [
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    images = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    expected = torch.sigmoid(images)
    for batch_size in (1, 3, None):
        actual = native.predict_batch(
            lambda value: value,
            images,
            variants=variants,
            tta_batch_size=batch_size,
            patch_size_zyx=(2, 3, 4),
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-7)


def test_importance_maps_and_sparse_accumulator_round_half_to_even():
    gaussian = native.create_importance_map((3, 4, 5), mode="gaussian")
    assert gaussian.shape == (3, 4, 5)
    assert gaussian.dtype == np.float32
    assert gaussian.max() == 1.0
    assert gaussian.min() >= np.finfo(np.float32).eps
    assert np.array_equal(
        native.create_importance_map((2, 2, 2), mode="constant"),
        np.ones((2, 2, 2), dtype=np.float32),
    )

    output = np.zeros((4, 4, 4), dtype=np.uint8)
    target = native.ChunkId(0, 0, 0)
    accumulator = native.ChunkAccumulator3D(
        output=output,
        target_chunks={target},
        contribution_counts={target: 2},
        chunk_shape_zyx=(2, 2, 2),
    )
    weights = np.ones((3, 3, 3), dtype=np.float32)
    accumulator.add_patch(
        patch_start_zyx=(0, 0, 0),
        probabilities=np.full((3, 3, 3), 0.25, dtype=np.float32),
        weights=weights,
    )
    assert not output.any()
    accumulator.add_patch(
        patch_start_zyx=(0, 0, 0),
        probabilities=np.full((3, 3, 3), 0.75, dtype=np.float32),
        weights=weights,
    )
    assert np.all(output[:2, :2, :2] == 128)
    assert not output[2:, :, :].any()
    assert not output[:, 2:, :].any()
    assert not output[:, :, 2:].any()


def test_native_dataset_normalizes_only_valid_voxels_and_builds_surface_channel(
    monkeypatch,
):
    from vesuvius.ink_detection.config import NormalizationConfig

    config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d",
            normalization=NormalizationConfig.from_value(
                {"mode": "divide", "divisor": 10}
            ),
        )
    )
    current_volume = {"value": np.full((2, 2, 2), 10, dtype=np.uint8)}

    def open_shared_volume_stub(path, resolution, *, cache_dir, cache_max_gb):
        return current_volume["value"]

    monkeypatch.setattr(native, "_open_shared_volume", open_shared_volume_stub)
    original_params = native.native_tifxyz_pyramid_params
    params_calls = []
    monkeypatch.setattr(
        native,
        "native_tifxyz_pyramid_params",
        lambda resolution: (
            params_calls.append(resolution) or original_params(resolution)
        ),
    )
    dataset = native.NativePatchDataset(
        tifxyz_dir=Path("unused"),
        volume_path="memory://native",
        resolution="0",
        patches=(native.PatchSpec(-1, -1, -1),),
        patch_size_zyx=(3, 3, 3),
        config=config,
    )
    image, metadata = dataset[0]
    assert metadata.tolist() == [-1, -1, -1]
    assert image.shape == (1, 3, 3, 3)
    assert not image[0, 0].any()
    assert not image[0, :, 0].any()
    assert not image[0, :, :, 0].any()
    assert torch.all(image[0, 1:, 1:, 1:] == 1)

    class FakeTifxyz:
        full_resolution_shape = (2, 2)

        def __init__(self):
            yy, xx = np.mgrid[:2, :2]
            self.x = xx.astype(np.float32)
            self.y = yy.astype(np.float32)
            self.z = np.zeros((2, 2), dtype=np.float32)

        def get_zyxs(self, stored_resolution=False):
            return np.stack((self.z, self.y, self.x), axis=-1)

        def __getitem__(self, slices):
            return (
                self.x[slices],
                self.y[slices],
                self.z[slices],
                np.ones(self.x[slices].shape, dtype=bool),
            )

    wrap_config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d_single_wrap",
            normalization=NormalizationConfig.from_value("none"),
        )
    )
    current_volume["value"] = np.zeros((2, 2, 2), dtype=np.uint8)
    wrapped = native.NativePatchDataset(
        tifxyz_dir=Path("unused"),
        volume_path="memory://native",
        resolution="0",
        patches=(native.PatchSpec(0, 0, 0),),
        patch_size_zyx=(2, 2, 2),
        config=wrap_config,
    )
    wrapped._tifxyz = FakeTifxyz()
    wrapped_image, _ = wrapped[0]
    assert wrapped_image.shape == (2, 2, 2, 2)
    assert torch.all(wrapped_image[1, 0] == 1)
    assert torch.allclose(wrapped_image[1, 1], torch.full((2, 2), 0.9))
    assert params_calls == [0, 0]

    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored._volume is None
    assert restored._tifxyz is None
    assert restored._coarse_positions is None


def test_explicit_v2_sparse_ome_zarr_and_encoded_uint8_pyramid(tmp_path):
    path = tmp_path / "prediction.zarr"
    arrays = native.create_output_zarr(
        path,
        shape_zyx=(5, 4, 3),
        chunks_zyx=(2, 2, 2),
    )
    assert [tuple(array.shape) for array in arrays] == [
        (5, 4, 3),
        (3, 2, 2),
        (2, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ]
    root_metadata = json.loads((path / ".zgroup").read_text(encoding="utf-8"))
    array_metadata = json.loads((path / "0" / ".zarray").read_text(encoding="utf-8"))
    assert root_metadata == {"zarr_format": 2}
    assert array_metadata["dimension_separator"] == "/"
    assert array_metadata["compressor"] == {
        "id": "blosc",
        "cname": "zstd",
        "clevel": 3,
        "shuffle": 2,
        "blocksize": 0,
    }
    assert not any((path / "0").glob("[0-9]*"))

    values = np.array(
        [[[0, 1], [2, 3]], [[252, 253], [254, 255]]], dtype=np.uint8
    )
    target = native.ChunkId(0, 0, 0)
    accumulator = native.ChunkAccumulator3D(
        output=arrays[0],
        target_chunks={target},
        contribution_counts={target: 1},
        chunk_shape_zyx=(2, 2, 2),
    )
    accumulator.add_patch(
        patch_start_zyx=(0, 0, 0),
        probabilities=values.astype(np.float32) / 255.0,
        weights=np.ones(values.shape, dtype=np.float32),
    )
    assert (path / "0" / "0" / "0" / "0").is_file()
    assert not (path / "0" / "0" / "0" / "1").exists()

    native.build_downsample_levels(arrays, level0_written_chunks={target}, workers=2)
    expected = np.zeros((5, 4, 3), dtype=np.uint8)
    expected[:2, :2, :2] = values
    multiscales = zarr.open_group(path, mode="r").attrs["multiscales"][0]
    assert multiscales["version"] == "0.4"
    assert multiscales["axes"] == native.AXES
    assert [dataset["path"] for dataset in multiscales["datasets"]] == [
        str(level) for level in range(6)
    ]
    for level, array in enumerate(arrays):
        assert array.attrs["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]
        assert np.array_equal(np.asarray(array), expected)
        assert multiscales["datasets"][level]["coordinateTransformations"] == [
            {"type": "scale", "scale": [float(2**level)] * 3}
        ]
        if level + 1 < len(arrays):
            expected = native.downsample_mean_3d(expected)


def test_output_reject_overwrite_and_coherent_rerun(tmp_path):
    output = tmp_path / "existing.zarr"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError):
        native.create_output_zarr(
            output,
            shape_zyx=(2, 2, 2),
            chunks_zyx=(2, 2, 2),
            overwrite=False,
        )
    assert sentinel.read_bytes() == b"keep-me"

    first = native.create_output_zarr(
        output,
        shape_zyx=(2, 2, 2),
        chunks_zyx=(2, 2, 2),
        overwrite=True,
    )
    assert not sentinel.exists()
    first[0][:] = 17
    second = native.create_output_zarr(
        output,
        shape_zyx=(2, 2, 2),
        chunks_zyx=(2, 2, 2),
        overwrite=True,
    )
    assert not np.asarray(second[0]).any()
    assert len(list(zarr.open_group(output, mode="r").array_keys())) == 6


def test_plan_only_does_not_load_runtime_or_touch_existing_output(tmp_path, monkeypatch):
    plan = _plan(max_target_chunks=1)
    sentinel = tmp_path / "prediction.zarr"
    sentinel.write_bytes(b"immutable")
    calls = []

    def fake_plan(args):
        calls.append(args.checkpoint)
        return plan, object(), object()

    monkeypatch.setattr(native, "_plan_from_args", fake_plan)
    monkeypatch.setattr(
        native,
        "_load_native_model",
        lambda *args: pytest.fail("model must remain lazy"),
    )
    args = SimpleNamespace(
        plan_only=True,
        checkpoint=tmp_path / "model.ckpt",
        output_zarr=sentinel,
    )
    assert native.run_command(args) == 0
    assert calls == [args.checkpoint]
    assert sentinel.read_bytes() == b"immutable"


def test_injected_command_runtime_writes_then_builds_all_levels(tmp_path, monkeypatch):
    target = native.ChunkId(0, 0, 0)
    plan = native.NativePlan(
        volume_path="memory://native",
        array_shape_zyx=(2, 2, 2),
        chunk_shape_zyx=(2, 2, 2),
        occupied_chunks=frozenset({target}),
        expanded_chunks=frozenset({target}),
        target_chunks=frozenset({target}),
        patches=(native.PatchSpec(0, 0, 0),),
        contribution_counts={target: 1},
        patch_size_zyx=(2, 2, 2),
        stride_zyx=(1, 1, 1),
    )
    from vesuvius.ink_detection.config import NormalizationConfig

    blocker = tmp_path / "embedded-io-blocker"
    blocker.write_bytes(b"")
    config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d",
            normalization=NormalizationConfig.from_value("none"),
            volume_auth_json=str(blocker / "auth.json"),
            volume_cache_dir=str(blocker / "cache"),
            volume_cache_max_gb=120.0,
        )
    )
    volume = np.zeros((2, 2, 2), dtype=np.uint8)
    monkeypatch.setattr(
        native,
        "_plan_from_args",
        lambda args: (plan, object(), config),
    )

    class ZeroModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(
                (images.shape[0], 1, *images.shape[-3:]),
                dtype=images.dtype,
                device=images.device,
            )

    bundle = native.NativeModelBundle(
        model=ZeroModel(),
        device=torch.device("cpu"),
        amp_dtype=None,
    )
    monkeypatch.setattr(native, "_load_native_model", lambda *args: bundle)

    def open_shared_volume_stub(path, resolution, *, cache_dir, cache_max_gb):
        assert (cache_dir, cache_max_gb) == (None, None)
        return volume

    monkeypatch.setattr(native, "_open_shared_volume", open_shared_volume_stub)

    output = tmp_path / "command.zarr"
    args = SimpleNamespace(
        plan_only=False,
        checkpoint=tmp_path / "model.ckpt",
        tifxyz_dir=tmp_path,
        output_zarr=output,
        resolution="0",
        overwrite=False,
        blend_mode="constant",
        tta=False,
        tta_batch_size=None,
        batch_size=1,
        gpu_ids=[],
        num_workers=0,
        prefetch_factor=2,
        cache_dir=None,
        cache_max_gb=None,
        downsample_workers=1,
    )
    assert native.run_command(args) == 0
    group = zarr.open_group(output, mode="r")
    assert sorted(group.array_keys()) == [str(level) for level in range(6)]
    assert all(np.all(np.asarray(group[str(level)]) == 128) for level in range(6))
    with pytest.raises(FileExistsError):
        native.run_command(args)
    assert np.all(np.asarray(zarr.open_group(output, mode="r")["0"]) == 128)
    args.overwrite = True
    assert native.run_command(args) == 0
    assert all(
        np.all(np.asarray(zarr.open_group(output, mode="r")[str(level)]) == 128)
        for level in range(6)
    )
