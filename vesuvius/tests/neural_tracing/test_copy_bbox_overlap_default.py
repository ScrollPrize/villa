"""Regression check for Copy's bbox-overlap CLI default (issue #2040).

Inspect the production argparse declaration without importing GPU/native dependencies.
The CLI argument and its literal default are extracted directly from the source.
"""

import argparse
import ast
from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py"
)


def _overlap_parser():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--bbox-overlap"
    ]
    assert len(matches) == 1, "Expected one --bbox-overlap argument declaration"
    call = matches[0]
    kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(kwargs["type"], ast.Name) and kwargs["type"].id == "float"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbox-overlap",
        type=float,
        default=ast.literal_eval(kwargs["default"]),
    )
    return parser


def test_copy_bbox_overlap_default_is_half():
    assert _overlap_parser().parse_args([]).bbox_overlap == pytest.approx(0.5)


def test_copy_bbox_overlap_allows_legacy_explicit_zero():
    assert _overlap_parser().parse_args(["--bbox-overlap", "0.0"]).bbox_overlap == pytest.approx(0.0)


def test_copy_bbox_overlap_allows_explicit_custom_value():
    assert _overlap_parser().parse_args(["--bbox-overlap", "0.25"]).bbox_overlap == pytest.approx(0.25)
