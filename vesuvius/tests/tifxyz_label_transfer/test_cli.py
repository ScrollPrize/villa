"""Public command and optional-import boundaries for TIFXYZ label transfer."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from vesuvius.tifxyz_label_transfer import estimate_canvas_offset
from vesuvius.tifxyz_label_transfer import estimate_canvas_offset_evidence
from vesuvius.tifxyz_label_transfer import prepare_canvas_offset_evidence
from vesuvius.tifxyz_label_transfer import self_render_tifxyz
from vesuvius.tifxyz_label_transfer import transfer
from vesuvius.tifxyz_label_transfer import view_alignment_napari


MODULES = (
    "build_native",
    "transfer",
    "make_label_zarrs",
    "prepare_canvas_offset_evidence",
    "estimate_canvas_offset_evidence",
    "estimate_canvas_offset",
    "self_render_tifxyz",
    "view_alignment_napari",
)


@pytest.mark.parametrize("module", MODULES)
def test_module_help_exits_zero_without_torch_or_napari(module: str) -> None:
    script = f"""
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    root = name.split('.', 1)[0]
    if root == 'torch':
        raise ModuleNotFoundError('simulated volume-only install')
    if root in {{'napari', 'qtpy'}}:
        raise AssertionError(f'unexpected optional import: {{name}}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from vesuvius.tifxyz_label_transfer import {module}
raise SystemExit({module}.main(['-h']))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_every_multiword_parser_accepts_underscore_aliases() -> None:
    single = transfer.build_parser().parse_args(
        [
            "single",
            "--source_tifxyz",
            "source",
            "--target_tifxyz",
            "target",
            "--label",
            "label",
            "--output",
            "output",
            "--dry_run",
        ]
    )
    assert single.source_tifxyz == "source"
    assert single.dry_run

    offset = estimate_canvas_offset.build_parser().parse_args(
        [
            "--source_tifxyz",
            "source",
            "--target_tifxyz",
            "target",
            "--source_render",
            "source.tif",
            "--target_render",
            "target.tif",
            "--output",
            "offset.json",
        ]
    )
    assert offset.source_tifxyz == "source"

    evidence = estimate_canvas_offset_evidence.build_parser().parse_args(
        ["--case_dir", "case", "--output", "offset.json"]
    )
    assert evidence.case_dir == Path("case")

    preparation = prepare_canvas_offset_evidence.build_parser().parse_args(
        ["--case_dir", "case", "--ink_rclone_root", "remote:ink"]
    )
    assert preparation.ink_rclone_root == "remote:ink"

    rendering = self_render_tifxyz.build_parser().parse_args(
        ["--case_dir", "case", "--plan_only"]
    )
    assert rendering.plan_only

    viewer = view_alignment_napari.build_parser().parse_args(
        ["--case_dir", "case", "--validate_only"]
    )
    assert viewer.validate_only
