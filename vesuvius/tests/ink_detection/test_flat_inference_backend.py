"""OpenVINO CPU backend for flat inference: CLI contract and torch parity."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile
import torch
from vc_reader.reader import ZarrWriter

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference.infer import parse_args
from vesuvius.ink_detection.models.model import make_model

from .helpers import config_mapping
from .test_flat_inference_ensemble import _write_surface


def test_openvino_backend_is_opt_in_and_cpu_only():
    positional = ["surface.zarr", "model.pth", "out.tif"]
    assert parse_args(positional).backend == "torch"
    assert parse_args([*positional, "--backend", "openvino"]).backend == "openvino"
    for extra in (["--device", "cuda"], ["--device", "mps"], ["--gpus", "0"]):
        with pytest.raises(SystemExit):
            parse_args([*positional, "--backend", "openvino", *extra])


def _save_checkpoint(path, model_type: str, *, seed: int):
    mapping = config_mapping(model_type, depth=3, side=16)
    torch.manual_seed(seed)
    model = make_model(InkConfig.from_mapping(mapping))
    torch.save({"config": mapping, "model": model.state_dict()}, path)
    return path


# find_spec, not importorskip: importing OpenVINO here would load its TBB pool
# into the pytest process, which the subprocess below exists to avoid.
@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None,
    reason="needs the cpu-inference extra",
)
@pytest.mark.parametrize(
    "model_type", ["vesuvius_unet_2p5d", "vesuvius_unet_3d_stem_2d"]
)
def test_openvino_ensemble_tta_matches_torch_within_one_level(tmp_path, model_type):
    # 40x40 at patch 16 and overlap 0.5 is 16 patches: batch 3 leaves a partial
    # last batch, mirror TTA multiplies the batch by the variant count, and the
    # all-zero corner patch is dropped from its batch before the forward.
    surface = _write_surface(tmp_path / "surface.zarr", side=40)
    ZarrWriter.open(surface)[:, :16, :16] = 0
    first = _save_checkpoint(tmp_path / "seed0.pth", model_type, seed=0)
    second = _save_checkpoint(tmp_path / "seed1.pth", model_type, seed=1)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")

    def run(backend):
        output = tmp_path / f"{backend}.tif"
        # A child process keeps OpenVINO's TBB thread pool out of the pytest
        # process: with it loaded, a later subprocess-spawning test segfaulted
        # pytest on Linux (OpenVINO 2026.4.1, torch 2.15). The CLI itself runs fine.
        result = subprocess.run(
            [
                sys.executable, "-m", "vesuvius.ink_detection.inference.infer",
                str(surface), str(first), str(output),
                "--ensemble-checkpoint", str(second),
                "--tta-mirror", "--batch-size", "3",
                "--workers", "0", "--no-compile", "--device", "cpu",
                "--backend", backend,
            ],
            env=environment,
            check=True,
            capture_output=True,
        )
        converted = b"OpenVINO inference threads" in result.stderr
        assert converted == (backend == "openvino")
        return tifffile.imread(output).astype(np.int16)

    reference, converted = run("torch"), run("openvino")
    assert np.unique(reference).size > 10
    assert np.abs(converted - reference).max() <= 1
