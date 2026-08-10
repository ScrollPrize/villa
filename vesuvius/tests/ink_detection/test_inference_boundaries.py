"""Checkpoint, runtime, and root-I/O boundary tests."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
import zarr

from vesuvius.ink_detection.models.checkpoint import (
    config_from_checkpoint,
    resolve_pretrained_backbone_config,
    select_inference_weights,
)
from vesuvius.ink_detection.inference.inference_runtime import (
    checkpoint_amp_dtype,
    parse_gpu_ids,
    prepare_model_for_inference,
)
from vesuvius.ink_detection.inference.infer import load_flat_inference_state
from vesuvius.ink_detection.volume_io import (
    open_volume,
    open_volume_root,
    select_volume_level,
)


def test_inference_import_does_not_add_runtime_side_effects():
    source = """
import multiprocessing as mp
import os
import sys
import vesuvius
os.environ.pop('OPENCV_IO_MAX_IMAGE_PIXELS', None)
for name in tuple(sys.modules):
    if name == 'vesuvius.tifxyz' or name.startswith('vesuvius.tifxyz.'):
        sys.modules.pop(name)
before = mp.get_start_method(allow_none=True)
import vesuvius.ink_detection.inference.infer
assert os.environ.get('OPENCV_IO_MAX_IMAGE_PIXELS') is None
assert mp.get_start_method(allow_none=True) == before
assert 'vesuvius.tifxyz.reader' not in sys.modules
"""
    environment = os.environ.copy()
    source_root = Path(__file__).parents[2] / "src"
    environment["PYTHONPATH"] = str(source_root)
    subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_state_alias_precedence_and_root_tensor_acceptance():
    states = {
        name: {"weight": torch.tensor([float(index)])}
        for index, name in enumerate(
            ("ema_model", "state_dict", "model_state_dict", "model"), start=1
        )
    }
    selected, state = select_inference_weights(states, prefer_ema=True)
    assert selected == "ema_model"
    assert state is states["ema_model"]
    selected, state = select_inference_weights(states, prefer_ema=False)
    assert selected == "state_dict"
    assert state is states["state_dict"]

    for expected in ("state_dict", "model_state_dict", "model"):
        payload = {
            key: value
            for key, value in states.items()
            if key in ("state_dict", "model_state_dict", "model")
            and tuple(("state_dict", "model_state_dict", "model")).index(key)
            >= tuple(("state_dict", "model_state_dict", "model")).index(expected)
        }
        assert select_inference_weights(payload, prefer_ema=False)[0] == expected

    root = {"weight": torch.tensor([17.0]), "bias": torch.tensor([-3.0])}
    assert select_inference_weights(root) == ("<root>", root)
    with pytest.raises(KeyError, match="config"):
        config_from_checkpoint(root)
    assert select_inference_weights({}) == ("<root>", {})
    with pytest.raises(ValueError, match="supported model state"):
        select_inference_weights({"step": 7})


def test_recursive_backbone_resolution_is_relative_cycle_safe_and_pure(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    terminal = nested_dir / "terminal.pth"
    middle = nested_dir / "middle.pth"
    torch.save(
        {"config": {"model_config": {"pretrained_backbone": "dinov2"}}},
        terminal,
    )
    torch.save(
        {
            "config": {
                "model_config": {"pretrained_backbone": terminal.name}
            }
        },
        middle,
    )
    outer_path = tmp_path / "outer.pth"
    authored = {
        "model_config": {"pretrained_backbone": "nested/middle.pth"},
        "untouched": [1, 2, 3],
    }
    before = deepcopy(authored)

    resolved = resolve_pretrained_backbone_config(
        authored, checkpoint_path=outer_path
    )

    assert resolved["model_config"]["pretrained_backbone"] == "dinov2"
    assert authored == before

    torch.save(
        {
            "config": {
                "model_config": {"pretrained_backbone": "../outer.pth"}
            }
        },
        middle,
    )
    torch.save({"config": authored}, outer_path)
    with pytest.raises(ValueError, match="recursive pretrained_backbone"):
        resolve_pretrained_backbone_config(authored, checkpoint_path=outer_path)

    missing = {"model_config": {"pretrained_backbone": "missing.pth"}}
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_pretrained_backbone_config(missing, checkpoint_path=outer_path)


def test_backbone_resolution_falls_back_to_current_directory(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    terminal = tmp_path / "backbone.pth"
    torch.save(
        {"config": {"model_config": {"pretrained_backbone": "dinov2"}}},
        terminal,
    )
    monkeypatch.chdir(tmp_path)

    resolved = resolve_pretrained_backbone_config(
        {"model_config": {"pretrained_backbone": terminal.name}},
        checkpoint_path=checkpoint_dir / "model.pth",
    )

    assert resolved["model_config"]["pretrained_backbone"] == "dinov2"


def test_backbone_resolution_expands_home_in_checkpoint_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    run = home / "run"
    run.mkdir(parents=True)
    terminal = run / "backbone.pth"
    torch.save(
        {"config": {"model_config": {"pretrained_backbone": "dinov2"}}},
        terminal,
    )
    monkeypatch.setenv("HOME", str(home))

    resolved = resolve_pretrained_backbone_config(
        {"model_config": {"pretrained_backbone": terminal.name}},
        checkpoint_path="~/run/model.pth",
    )

    assert resolved["model_config"]["pretrained_backbone"] == "dinov2"


def test_flat_inference_state_loading_is_nonstrict_and_ddp_compatible():
    model = nn.Linear(2, 1)
    partial = {"module.weight": torch.ones_like(model.weight)}

    incompatibility = load_flat_inference_state(model, partial)

    assert incompatibility.missing_keys == ["bias"]
    assert incompatibility.unexpected_keys == []
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))


def test_runtime_gpu_and_amp_boundaries():
    assert parse_gpu_ids(None) == ()
    assert parse_gpu_ids(" 2,0 ") == (2, 0)
    for malformed in ("0,", "-1", "a", "1,1"):
        with pytest.raises(ValueError):
            parse_gpu_ids(malformed)
    assert checkpoint_amp_dtype({"config": {"mixed_precision": "fp16"}}) is torch.float16
    assert checkpoint_amp_dtype({"config": {"mixed_precision": "bfloat16"}}) is torch.bfloat16
    assert checkpoint_amp_dtype({"config": {"mixed_precision": "no"}}) is None


def test_compile_fallback_returns_eager_model_when_compiler_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch, "compile", None)

    source = nn.Linear(1, 1)
    prepared, device = prepare_model_for_inference(
        source,
        gpu_ids=(),
        compile_model=True,
        compile_mode="default",
    )

    assert prepared is source
    assert device.type in {"cpu", "cuda"}


def test_root_volume_view_and_level_selection_share_one_open(tmp_path):
    root_path = tmp_path / "pyramid.zarr"
    kwargs = {"mode": "w"}
    zarr3 = int(zarr.__version__.split(".", 1)[0]) >= 3
    if zarr3:
        kwargs["zarr_format"] = 2
    root = zarr.open_group(root_path, **kwargs)
    create = root.create_array if zarr3 else root.create_dataset
    create("0", shape=(3, 4, 5), chunks=(3, 4, 5), dtype="u1")
    create("3", shape=(1, 1, 1), chunks=(1, 1, 1), dtype="u1")

    opened = open_volume_root(root_path)

    assert tuple(select_volume_level(opened, "0", source=str(root_path)).shape) == (3, 4, 5)
    assert tuple(select_volume_level(opened, "3", source=str(root_path)).shape) == (1, 1, 1)
    assert tuple(open_volume(root_path, 0).shape) == (3, 4, 5)
