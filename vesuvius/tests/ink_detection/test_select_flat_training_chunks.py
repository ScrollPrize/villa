from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from vesuvius.ink_detection.preprocessing.select_flat_training_chunks import main


SHAPE_ZYX = (4, 16, 16)
CHUNKS_ZYX = (4, 4, 4)


def _write_level0(path: Path, array: np.ndarray) -> None:
    kwargs = {"mode": "w"}
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        kwargs["zarr_format"] = 2
    root = zarr.open_group(path, **kwargs)
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        root.create_array("0", data=array, chunks=CHUNKS_ZYX)
    else:
        root.create_dataset("0", data=array, chunks=CHUNKS_ZYX)


def _segment(tmp_path: Path) -> Path:
    segment = tmp_path / "ink-dataset" / "seg-a"
    segment.mkdir(parents=True)
    (segment / "x.tif").write_bytes(b"")
    _write_level0(segment / "seg-a.zarr", np.full(SHAPE_ZYX, 100, dtype=np.uint8))
    labels = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    labels[:, 0:4, 0:4] = 1  # one labeled chunk in the top-left corner
    _write_level0(segment / "seg-a_supervision_mask.zarr", labels)
    _write_level0(segment / "seg-a_inklabels.zarr", labels)
    return segment


def _config(tmp_path: Path) -> Path:
    authored = {
        "out_dir": str(tmp_path / "run"),
        "mode": "flat",
        "model_type": "vesuvius_unet",
        "in_channels": 1,
        "model_config": {"autoconfigure": True, "z_projection_mode": "max"},
        "targets": {"ink": {"out_channels": 1, "activation": "none", "z_projection_mode": "max"}},
        "patch_size": [4, 8, 8],
        "patch_overlap": 0.5,
        "patch_min_labeled_coverage": 0.0,
        "batch_size": 1,
        "num_iterations": 1,
        "seed": 0,
        "learning_rate": 0.01,
        "datasets": [{"segments_path": str(tmp_path / "ink-dataset"), "volume_scale": "0"}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(authored), encoding="utf-8")
    return path


def _plan(tmp_path: Path) -> Path:
    paths = ["x.tif", "meta.json", "seg-a.zarr/.zgroup", "seg-a.zarr/0/.zarray"]
    paths += [f"seg-a.zarr/0/0.{y}.{x}" for y in range(4) for x in range(4)]
    paths += ["seg-a.zarr/1/.zarray", "seg-a.zarr/1/0.0.0", "preds/prediction.tif"]
    paths += ["seg-a_supervision_mask.zarr/0/0.0.0", "seg-a_inklabels.zarr/1/0.0.0"]
    records = [{"type": "header", "source": "hf://buckets/x/y/seg-a", "dest": "./ink-dataset/seg-a"}]
    records += [
        {"type": "operation", "action": "download", "path": p, "size": 10} for p in paths
    ]
    path = tmp_path / "full.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_keeps_only_level0_chunks_under_labeled_patches(tmp_path):
    _segment(tmp_path)
    output = tmp_path / "subset.jsonl"

    assert main([str(_config(tmp_path)), str(_plan(tmp_path)), str(output), "--plan-root", str(tmp_path)]) == 0

    records = [json.loads(line) for line in output.read_text().splitlines()]
    kept = {r["path"] for r in records[1:]}
    # The only patch starts at (y, x) = (0, 0) and spans 8x8 pixels: chunks 0..1 on both axes.
    assert {p for p in kept if p.startswith("seg-a.zarr/0/0.")} == {
        "seg-a.zarr/0/0.0.0",
        "seg-a.zarr/0/0.0.1",
        "seg-a.zarr/0/0.1.0",
        "seg-a.zarr/0/0.1.1",
    }
    assert {"x.tif", "meta.json", "seg-a.zarr/.zgroup", "seg-a.zarr/0/.zarray"} <= kept
    assert {"seg-a_supervision_mask.zarr/0/0.0.0", "seg-a_inklabels.zarr/1/0.0.0"} <= kept
    assert not kept & {"seg-a.zarr/1/.zarray", "seg-a.zarr/1/0.0.0", "preds/prediction.tif"}
    assert records[0]["summary"]["downloads"] == len(kept)


def test_keeps_every_chunk_a_z_jittered_window_can_read(tmp_path, monkeypatch):
    # Review on #1821: flat_z_window_jitter shifts the read window in Z, so selecting by the
    # stored patch bbox alone leaves out chunks that training then reads as zeros.
    import random

    from vesuvius.ink_detection.data.dataset import InkDataset, flat_z_window_bbox
    from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config

    shape = (16, 16, 16)
    segment = tmp_path / "ink-dataset" / "seg-a"
    segment.mkdir(parents=True)
    (segment / "x.tif").write_bytes(b"")
    _write_level0(segment / "seg-a.zarr", np.full(shape, 100, dtype=np.uint8))
    labels = np.zeros(shape, dtype=np.uint8)
    labels[:, 0:4, 0:4] = 1
    _write_level0(segment / "seg-a_supervision_mask.zarr", labels)
    _write_level0(segment / "seg-a_inklabels.zarr", labels)
    config_path = _config(tmp_path)
    authored = json.loads(config_path.read_text())
    authored["patch_size"] = [8, 8, 8]
    authored["flat_z_window_jitter"] = {"enabled": True, "window_depth": 4, "max_offset": 4, "probability": 1.0}
    config_path.write_text(json.dumps(authored))

    records = [{"type": "header", "source": "hf://buckets/x/y/seg-a", "dest": "./ink-dataset/seg-a"}]
    records += [
        {"type": "operation", "action": "download", "path": f"seg-a.zarr/0/{z}.{y}.{x}", "size": 10}
        for z in range(4) for y in range(4) for x in range(4)
    ]
    plan = tmp_path / "full.jsonl"
    plan.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    output = tmp_path / "subset.jsonl"
    assert main([str(config_path), str(plan), str(output), "--plan-root", str(tmp_path)]) == 0
    kept = {json.loads(line)["path"] for line in output.read_text().splitlines()[1:]}

    # Every window the trainer can draw, one offset at a time.
    config = training_dataset_config(stage_training_request(config_path).config)
    dataset = InkDataset(config, do_augmentations=False)
    needed = set()
    for offset in range(-4, 5):
        monkeypatch.setattr(random, "randint", lambda a, b, offset=offset: offset)
        for patch in dataset.patches:
            (z0, y0, x0, z1, y1, x1), got = flat_z_window_bbox(
                patch.bbox, config=config, do_augmentations=True, is_validation=False
            )
            assert got == offset
            if z0 < 0 or z1 > shape[0]:
                continue  # the trainer rejects windows outside the volume (Z padding is forbidden)
            needed |= {
                f"seg-a.zarr/0/{iz}.{iy}.{ix}"
                for iz in range(z0 // 4, (z1 - 1) // 4 + 1)
                for iy in range(y0 // 4, (y1 - 1) // 4 + 1)
                for ix in range(x0 // 4, (x1 - 1) // 4 + 1)
            }
    assert needed - kept == set()
    assert len({p.split("/")[-1].split(".")[0] for p in needed}) > 2  # the jitter really reaches more Z chunks


def test_chunk_keys_follow_the_arrays_own_key_encoding():
    # Review on #1821: Zarr 2 arrays have no `.metadata`; '/'-separated stores must keep their separator.
    from vesuvius.ink_detection.preprocessing.select_flat_training_chunks import chunk_key_encoder

    class Zarr2Array:
        _dimension_separator = "/"

    assert chunk_key_encoder(Zarr2Array())((1, 2, 3)) == "1/2/3"

    class Zarr2DefaultArray:
        _dimension_separator = None

    assert chunk_key_encoder(Zarr2DefaultArray())((1, 2, 3)) == "1.2.3"
