"""End-to-end CLI smoke for ``vesuvius.trace_fiber``.

Builds a tiny CPU-only model and a fake "remote" zarr on disk, saves a
checkpoint + a prompt .npz in the expected formats, then invokes the
streaming CLI as a Python function and asserts the local artefacts on disk.

No S3 and no actual WK upload (``--upload-to-wk`` is omitted).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import zarr

from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel
from vesuvius.neural_tracing.autoreg_fiber.streaming.cli import main as cli_main


def _tiny_config(tmp_path: Path) -> dict:
    return {
        "dinov2_backbone": None,
        "crop_size": [16, 16, 16],
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "prompt_length": 2,
        "target_length": 3,
        "point_stride": 1,
        "decoder_dim": 24,
        "decoder_depth": 1,
        "decoder_num_heads": 2,
        "decoder_dropout": 0.0,
        "max_fiber_position_embeddings": 64,
        "coarse_prediction_mode": "axis_factorized",
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
        "volumes": {
            "TestVolume": {
                "storage_options": {},
                "volume_shape": [64, 32, 32],
                "volume_zarr_url": str(tmp_path / "volume.zarr"),
            },
        },
    }


def _make_zarr(path: Path, shape: tuple[int, int, int], chunk_shape: tuple[int, int, int]) -> None:
    arr = zarr.open(
        str(path),
        mode="w",
        shape=shape,
        chunks=chunk_shape,
        dtype="float32",
    )
    rng = np.random.default_rng(0)
    arr[:] = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)


def _write_prompt_npz(tmp_path: Path) -> Path:
    points_zyx = np.array(
        [[8.0, 16.0, 16.0], [9.0, 16.0, 16.0], [10.0, 16.0, 16.0]],
        dtype=np.float32,
    )
    fiber = FiberPath(
        annotation_id="ann-cli",
        tree_id="tree-cli",
        target_volume="TestVolume",
        marker="fibers_s1a",
        source_points_xyz=points_zyx[:, ::-1].astype(np.float32, copy=False),
        points_zyx=points_zyx,
        transform_checksum="identity",
        densify_step=1.0,
    )
    return write_fiber_cache(fiber, tmp_path / "prompts")


def _save_checkpoint(tmp_path: Path, cfg: dict) -> Path:
    torch.manual_seed(0)
    model = AutoregFiberModel(cfg).eval()
    ckpt = {"config": cfg, "model": model.state_dict()}
    path = tmp_path / "ckpt.pth"
    torch.save(ckpt, path)
    return path


def test_cli_writes_local_artifacts_for_bidirectional_trace(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    _make_zarr(Path(cfg["volumes"]["TestVolume"]["volume_zarr_url"]), shape=(64, 32, 32), chunk_shape=(16, 16, 16))
    ckpt_path = _save_checkpoint(tmp_path, cfg)
    prompt_path = _write_prompt_npz(tmp_path)
    out_dir = tmp_path / "out"

    rc = cli_main(
        [
            "--config", str(config_path),
            "--ckpt", str(ckpt_path),
            "--prompt", str(prompt_path),
            "--volume", "TestVolume",
            "--out", str(out_dir),
            "--direction", "both",
            "--max-steps", "5",
            "--stop-prob-threshold", "-1.0",
            "--min-steps", "1",
            "--cache-chunks", "16",
            "--prefetch-workers", "0",
            "--device", "cpu",
            "--dtype", "fp32",
            "--reanchor-margin", "1",
        ]
    )
    assert rc == 0
    assert (out_dir / "trace.npz").exists()
    assert (out_dir / "trace.nml").exists()
    assert (out_dir / "trace.zip").exists()
    summary = json.loads((out_dir / "trace_summary.json").read_text(encoding="utf-8"))
    assert summary["forward_stop_reason"] in {"max_steps", "out_of_volume", "stop_probability"}
    assert summary["backward_stop_reason"] in {"max_steps", "out_of_volume", "stop_probability"}
    chunk_stats = summary["chunk_cache_stats"]
    assert chunk_stats["misses"] >= 1  # we read something from the zarr
    # And the saved trace polyline has at least the prompt + a few steps.
    npz = np.load(out_dir / "trace.npz")
    assert npz["polyline_world_zyx"].shape[1] == 3
    assert npz["polyline_world_zyx"].shape[0] >= 3


def test_cli_rejects_missing_volume_key(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    _make_zarr(Path(cfg["volumes"]["TestVolume"]["volume_zarr_url"]), shape=(32, 32, 32), chunk_shape=(16, 16, 16))
    ckpt_path = _save_checkpoint(tmp_path, cfg)
    prompt_path = _write_prompt_npz(tmp_path)

    try:
        cli_main(
            [
                "--config", str(config_path),
                "--ckpt", str(ckpt_path),
                "--prompt", str(prompt_path),
                "--volume", "DoesNotExist",
                "--out", str(tmp_path / "out"),
                "--direction", "forward",
                "--max-steps", "1",
                "--stop-prob-threshold", "-1.0",
                "--min-steps", "1",
                "--cache-chunks", "4",
                "--prefetch-workers", "0",
                "--device", "cpu",
                "--dtype", "fp32",
                "--reanchor-margin", "1",
            ]
        )
    except SystemExit as exc:
        assert "DoesNotExist" in str(exc) or "TestVolume" in str(exc)
    else:
        raise AssertionError("expected SystemExit for missing volume key")
