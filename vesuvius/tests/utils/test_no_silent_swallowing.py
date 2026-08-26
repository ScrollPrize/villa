"""In the data-access layer, an error must not be swallowed into a plausible answer.

The general form of the zarr_array_exists bug. There, a broad `except Exception:
return False` turned a credentials failure into "the array does not exist", and a
caller five minutes away reported that an array had never been created. The
danger is specific: in code whose job is to answer "is this there / can I read
this", swallowing an error produces an answer that is indistinguishable from a
real negative, and every caller downstream then acts on a confident lie.

This is a RATCHET over the data-access layer only. Handlers that legitimately
degrade are listed with a reason. The list may shrink; it must never grow.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "vesuvius"
LAYER = ("utils/io", "data")

# (path relative to src/vesuvius, function name) -> why swallowing is correct here.
ALLOWED = {
    ("utils/io/zarr_utils.py", "_is_ome_zarr"):
        "reads .zattrs which the code documents as optional; the result does not depend on it",
    ("data/zarr_chunk_index.py", "_zarray_signature"):
        "computes a cache signature; None means 'no signature', a documented degraded mode",
}

pytestmark = pytest.mark.skipif(
    not SRC.is_dir(), reason="source tree not present outside a checkout"
)


def _enclosing_function(tree, node):
    best = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fn.lineno <= node.lineno and (best is None or fn.lineno > best.lineno):
                end = getattr(fn, "end_lineno", None)
                if end is None or node.lineno <= end:
                    best = fn
    return best.name if best else "<module>"


def _swallowers():
    """Broad handlers whose whole body is `pass` or a constant return."""
    found = []
    for sub in LAYER:
        base = SRC / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                t = node.type
                broad = t is None or (isinstance(t, ast.Name) and t.id in ("Exception", "BaseException"))
                if not broad or len(node.body) != 1:
                    continue
                body = node.body[0]
                swallow = (
                    isinstance(body, ast.Pass)
                    or (isinstance(body, ast.Return)
                        and (body.value is None or isinstance(body.value, ast.Constant)))
                )
                if swallow:
                    rel = str(path.relative_to(SRC))
                    found.append((rel, _enclosing_function(tree, node), node.lineno))
    return found


def test_no_new_error_swallowing_in_the_data_layer():
    unexplained = [
        (rel, fn, line) for rel, fn, line in _swallowers() if (rel, fn) not in ALLOWED
    ]
    assert not unexplained, (
        "these broad handlers in the data-access layer turn an error into a plausible "
        "answer, so callers cannot tell a real negative from a failure:\n"
        + "\n".join(f"  {rel}:{line} in {fn}()" for rel, fn, line in unexplained)
        + "\nEither report the failure, or add an entry to ALLOWED explaining why "
          "degrading silently is correct here."
    )


def test_allowlist_does_not_go_stale():
    """If a listed handler no longer swallows, take it off the list."""
    present = {(rel, fn) for rel, fn, _ in _swallowers()}
    gone = sorted(set(ALLOWED) - present)
    assert not gone, f"no longer swallow, remove from ALLOWED: {gone}"
