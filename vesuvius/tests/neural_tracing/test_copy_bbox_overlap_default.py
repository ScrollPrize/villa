"""Regression tests for Copy's CLI overlap default (issue #2040).

Execute the actual parse_args function from production source without importing
its unrelated GPU/native dependencies. This verifies argparse and its own
validation logic, not just the source text's default declaration.
"""

import argparse
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py"
)
REQUIRED_ARGS = [
    "--tifxyz-path", "example.tifxyz",
    "--volume-path", "example.zarr",
    "--checkpoint-path", "checkpoint.pth",
]


def _run_production_parser(extra_argv):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    ]
    assert len(functions) == 1, "Expected exactly one production parse_args"

    # Only the parser function is compiled. Its imports, model initialization
    # and native dependencies are not needed for the CLI behavior under test.
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "argparse": argparse,
        "os": os,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "np": SimpleNamespace(isfinite=math.isfinite),
        "Path": Path,
        "_DEFAULT_VOLUME_CHUNK_CACHE_GB": 20.0,
        "TTA_MERGE_METHODS": ("vector_geomedian",),
        "_TTA_MERGE_ALIASES": {},
        "TTA_TRANSFORM_MODES": ("mirror",),
        "_TTA_TRANSFORM_ALIASES": {},
        "is_auto_displacement_scale": lambda value: str(value).lower() == "auto",
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["parse_args"](REQUIRED_ARGS + list(extra_argv))


def test_copy_bbox_overlap_default_is_half():
    assert _run_production_parser([]).bbox_overlap == pytest.approx(0.5)


def test_copy_bbox_overlap_allows_legacy_explicit_zero():
    assert _run_production_parser(["--bbox-overlap", "0.0"]).bbox_overlap == pytest.approx(0.0)


def test_copy_bbox_overlap_allows_explicit_custom_value():
    assert _run_production_parser(["--bbox-overlap", "0.25"]).bbox_overlap == pytest.approx(0.25)


@pytest.mark.parametrize("invalid", ["-0.01", "1.0", "1.5", "nan", "NaN", "inf", "-inf", "not-a-number"])
def test_copy_bbox_overlap_rejects_invalid_values(invalid):
    with pytest.raises(SystemExit) as exc:
        _run_production_parser(["--bbox-overlap", invalid])
    assert exc.value.code == 2
