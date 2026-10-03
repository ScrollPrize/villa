from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import torch
import zarr

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference.infer import main as infer_main
from vesuvius.ink_detection.inference.infer import select_layer_indices
from vesuvius.ink_detection.inference.sweep import (
    checkpoint_labels,
    main as sweep_main,
    parse_args,
    plan_depth_windows,
    roc_auc_uint8,
)
from vesuvius.ink_detection.models.model import make_model

from .test_model_foundation import _config_mapping


def test_offset_zero_is_the_default_infer_window_and_out_of_range_is_skipped():
    windows, skipped = plan_depth_windows(28, 17, [6, -8, 0, 2, -2, 5, 0])

    default = select_layer_indices(
        28, layer_start=None, layer_end=None, output_depth=17, direction="forward"
    )
    assert [(w.offset, w.layer_start, w.layer_end) for w in windows] == [
        (-2, 4, 21),
        (0, 6, 23),
        (2, 8, 25),
        (5, 11, 28),
    ]
    assert windows[1].layer_start == default[0]
    assert skipped == [-8, 6]

    short, short_skipped = plan_depth_windows(3, 5, [-1, 0, 1])
    assert [(w.offset, w.layer_start, w.layer_end) for w in short] == [(0, 0, 3)]
    assert short_skipped == [-1, 1]


def test_roc_auc_uint8_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 12, size=(9, 11), dtype=np.uint8)
    ink = rng.random((9, 11)) < 0.4
    region = rng.random((9, 11)) < 0.8

    positive = scores[region & ink].astype(float)
    negative = scores[region & ~ink].astype(float)
    pairs = positive[:, None] - negative[None, :]
    expected = ((pairs > 0).sum() + 0.5 * (pairs == 0).sum()) / pairs.size

    assert roc_auc_uint8(scores, ink, region) == pytest.approx(expected)
    assert roc_auc_uint8(scores, np.zeros_like(ink), region) is None
    with pytest.raises(ValueError, match="uint8"):
        roc_auc_uint8(scores.astype(np.float32), ink, region)


def test_checkpoint_labels_disambiguate_equal_stems(tmp_path):
    paths = [
        tmp_path / "seed42" / "step-075000.pth",
        tmp_path / "seed43" / "step-075000.pth",
        tmp_path / "other.pth",
    ]
    assert checkpoint_labels(paths) == [
        "seed42_step-075000",
        "seed43_step-075000",
        "other",
    ]


def test_labels_and_eval_mask_are_required_together(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "in.zarr",
                str(tmp_path),
                "--checkpoints",
                "a.pth",
                "--ink-labels",
                "labels.tif",
            ]
        )


def test_sweep_outputs_match_separate_infer_runs(tmp_path):
    torch.manual_seed(0)
    config_mapping = _config_mapping("vesuvius_unet_2p5d", depth=3, side=16)
    config_mapping["image_normalization"] = "robust_mad"
    model = make_model(InkConfig.from_mapping(config_mapping))
    checkpoint = tmp_path / "model.pth"
    torch.save({"config": config_mapping, "model": model.state_dict()}, checkpoint)

    rng = np.random.default_rng(1)
    input_path = tmp_path / "surface.zarr"
    volume = zarr.open(
        input_path,
        mode="w",
        shape=(7, 16, 16),
        chunks=(7, 16, 16),
        dtype="u1",
        zarr_format=2,
    )
    volume[:] = rng.integers(1, 256, size=(7, 16, 16), dtype=np.uint8)
    labels_path = tmp_path / "inklabels.tif"
    tifffile.imwrite(
        labels_path, np.where(rng.random((16, 16)) < 0.3, 255, 0).astype(np.uint8)
    )
    mask_path = tmp_path / "validation.zarr"
    mask = zarr.open(mask_path, mode="w", shape=(16, 16), dtype="u1", zarr_format=2)
    mask[:] = 0
    mask[2:14, 2:14] = 255

    common = ["--workers", "0", "--no-compile", "--blend-mode", "constant"]
    out_dir = tmp_path / "sweep"
    assert (
        sweep_main(
            [
                str(input_path),
                str(out_dir),
                "--checkpoints",
                str(checkpoint),
                "--offsets",
                "-3",
                "-1",
                "0",
                "1",
                "3",
                "--ink-labels",
                str(labels_path),
                "--eval-mask",
                str(mask_path),
                *common,
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "sweep.json").read_text())
    assert summary["checkpoints"][0]["skipped_offsets"] == [-3, 3]
    runs = summary["runs"]
    assert [(run["offset"], run["direction"]) for run in runs] == [
        (-1, "forward"),
        (-1, "reverse"),
        (0, "forward"),
        (0, "reverse"),
        (1, "forward"),
        (1, "reverse"),
    ]
    assert (out_dir / "sweep.png").is_file()

    ink = tifffile.imread(labels_path) != 0
    region = np.asarray(mask[:]) != 0
    outputs = set()
    for run in runs:
        start = 2 + run["offset"]
        expected_indices = list(range(start, start + 3))
        if run["direction"] == "reverse":
            expected_indices.reverse()
        assert run["layer_indices"] == expected_indices

        reference = tmp_path / f"reference_{run['output']}"
        assert (
            infer_main(
                [
                    str(input_path),
                    str(checkpoint),
                    str(reference),
                    "--layer-start",
                    str(start),
                    "--layer-end",
                    str(start + 3),
                    "--direction",
                    run["direction"],
                    *common,
                ]
            )
            == 0
        )
        swept = tifffile.imread(out_dir / run["output"])
        np.testing.assert_array_equal(swept, tifffile.imread(reference))
        assert run["roc_auc"] == pytest.approx(roc_auc_uint8(swept, ink, region))
        outputs.add(swept.tobytes())
    # The windows and directions really are different inputs to the model.
    assert len(outputs) == len(runs)
