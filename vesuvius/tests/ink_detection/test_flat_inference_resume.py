"""Resumable flat inference must be bit-identical to an uninterrupted run."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
import torch
import zarr

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference import infer
from vesuvius.ink_detection.inference.infer import main
from vesuvius.ink_detection.models.model import make_model

from .test_model_foundation import _config_mapping


class _Interrupted(Exception):
    pass


def _setup(tmp_path):
    config_mapping = _config_mapping("vesuvius_unet_2p5d", depth=3, side=16)
    config = InkConfig.from_mapping(config_mapping)
    torch.manual_seed(0)
    model = make_model(config)
    checkpoint = tmp_path / "model.pth"
    torch.save({"config": config_mapping, "model": model.state_dict()}, checkpoint)
    input_path = tmp_path / "surface.zarr"
    array = zarr.open(
        input_path,
        mode="w",
        shape=(3, 72, 56),
        chunks=(3, 16, 16),
        dtype="u1",
        zarr_format=2,
    )
    rng = np.random.default_rng(0)
    array[:] = rng.integers(1, 255, size=(3, 72, 56), dtype=np.uint8)
    return input_path, checkpoint


def _run(input_path, checkpoint, output, *extra):
    return main(
        [
            str(input_path),
            str(checkpoint),
            str(output),
            "--workers",
            "0",
            "--no-compile",
            *extra,
        ]
    )


def _interrupt_after_snapshots(monkeypatch, snapshots):
    original = infer._save_resume_snapshot
    calls = {"n": 0}

    def save_then_maybe_die(*args, **kwargs):
        original(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] >= snapshots:
            raise _Interrupted

    monkeypatch.setattr(infer, "_save_resume_snapshot", save_then_maybe_die)


def test_resumed_run_is_bit_identical_to_uninterrupted(tmp_path, monkeypatch):
    input_path, checkpoint = _setup(tmp_path)
    reference = tmp_path / "reference.tif"
    assert _run(input_path, checkpoint, reference) == 0
    expected = tifffile.imread(reference)
    assert expected.any()

    resume_dir = tmp_path / "resume"
    resumed = tmp_path / "resumed.tif"
    with monkeypatch.context() as patch:
        _interrupt_after_snapshots(patch, snapshots=3)
        with pytest.raises(_Interrupted):
            _run(
                input_path, checkpoint, resumed,
                "--resume-dir", str(resume_dir), "--resume-every", "7",
            )
    assert not resumed.exists()
    assert (resume_dir / "state_00003.json").exists()

    assert _run(
        input_path, checkpoint, resumed,
        "--resume-dir", str(resume_dir), "--resume-every", "7",
    ) == 0
    np.testing.assert_array_equal(tifffile.imread(resumed), expected)


def test_resume_falls_back_when_newest_snapshot_is_incomplete(tmp_path, monkeypatch):
    """On a lazily-uploading mount, the newest state file can land before the
    files it references; loading must fall back to the previous snapshot."""

    input_path, checkpoint = _setup(tmp_path)
    reference = tmp_path / "reference.tif"
    assert _run(input_path, checkpoint, reference) == 0

    resume_dir = tmp_path / "resume"
    resumed = tmp_path / "resumed.tif"
    with monkeypatch.context() as patch:
        _interrupt_after_snapshots(patch, snapshots=4)
        with pytest.raises(_Interrupted):
            _run(
                input_path, checkpoint, resumed,
                "--resume-dir", str(resume_dir), "--resume-every", "5",
            )
    (resume_dir / "open_00004.npz").unlink()

    assert _run(
        input_path, checkpoint, resumed,
        "--resume-dir", str(resume_dir), "--resume-every", "5",
    ) == 0
    np.testing.assert_array_equal(
        tifffile.imread(resumed), tifffile.imread(reference)
    )


def test_resume_refuses_a_different_run(tmp_path, monkeypatch):
    input_path, checkpoint = _setup(tmp_path)
    resume_dir = tmp_path / "resume"
    with monkeypatch.context() as patch:
        _interrupt_after_snapshots(patch, snapshots=1)
        with pytest.raises(_Interrupted):
            _run(
                input_path, checkpoint, tmp_path / "a.tif",
                "--resume-dir", str(resume_dir), "--resume-every", "5",
            )
    with pytest.raises(ValueError, match="different inference run"):
        _run(
            input_path, checkpoint, tmp_path / "b.tif",
            "--resume-dir", str(resume_dir), "--resume-every", "5",
            "--blend-mode", "constant",
        )
