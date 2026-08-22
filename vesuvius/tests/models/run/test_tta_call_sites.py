"""Signature-contract test for `infer_with_tta` call sites.

Covers a real bug: `neural_tracing/infer.py` called
`infer_with_tta(..., batched=self.tta_batched)`, but the function's
keyword-only parameter is named `use_batched`. There is no `batched`
parameter, so the call raised
`TypeError: infer_with_tta() got an unexpected keyword argument 'batched'`
the moment TTA was enabled (`do_tta: true` in the config).

These tests deliberately verify the contract via `inspect.signature` and
static source inspection rather than by importing
`vesuvius.neural_tracing.infer`, which pulls in `cupy` and therefore
requires a GPU environment.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from vesuvius.models.run.tta import infer_with_tta


def test_infer_with_tta_accepts_use_batched_not_batched():
    """Pin the parameter name the call sites must use."""
    params = inspect.signature(infer_with_tta).parameters
    assert "use_batched" in params
    assert "batched" not in params, (
        "if this parameter is ever renamed to 'batched', the call sites "
        "checked below must be updated in the same change"
    )


def test_infer_with_tta_binds_with_use_batched():
    sig = inspect.signature(infer_with_tta)
    sig.bind(object(), None, "mirroring", use_batched=True)


def test_infer_with_tta_rejects_batched():
    sig = inspect.signature(infer_with_tta)
    with pytest.raises(TypeError):
        sig.bind(object(), None, "mirroring", batched=True)


def _package_root() -> Path:
    """Locate `src/vesuvius` by walking up, rather than assuming a fixed depth."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "vesuvius"
        if candidate.is_dir():
            return candidate
    raise AssertionError("could not locate the vesuvius source tree")


def _iter_infer_with_tta_calls():
    """Yield (path, lineno, keyword_names) for every infer_with_tta call
    in the package, parsed statically so no GPU-only module is imported."""
    root = _package_root()
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "infer_with_tta":
                continue
            yield path, node.lineno, {kw.arg for kw in node.keywords if kw.arg}


def test_no_call_site_passes_an_unknown_keyword():
    """Every in-repo call must use keywords the function actually accepts."""
    accepted = set(inspect.signature(infer_with_tta).parameters)
    offenders = []
    checked = 0
    for path, lineno, kwargs in _iter_infer_with_tta_calls():
        checked += 1
        unknown = kwargs - accepted
        if unknown:
            offenders.append(f"{path}:{lineno} passes {sorted(unknown)}")

    assert checked > 0, "expected to find at least one infer_with_tta call site"
    assert not offenders, (
        "infer_with_tta call site(s) pass keywords the function does not "
        "accept, which raises TypeError at runtime:\n  "
        + "\n  ".join(offenders)
    )
