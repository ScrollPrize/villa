"""CLI validation errors should exit with a usage message, not a NameError.

`main()` in these entry points built its parser inline as
``build_parser().parse_args()`` and then called ``parser.error(...)`` on the
validation paths, but no name ``parser`` existed at that point. Every such path
raised ``NameError`` instead of the intended argparse usage error -- and in
finalize_outputs the call sat on the unconditional path, so the command failed
before doing any work.

The tests below exercise `main()` itself (the flag-spelling suite only reaches
`build_parser()`), so a regression shows up as the wrong exception type rather
than as a silently different message.
"""

from __future__ import annotations

import pytest


def _run_main(monkeypatch, module, argv):
    monkeypatch.setattr("sys.argv", [module.__name__] + argv)
    return module.main()


def test_finalize_bad_part_id_exits_with_usage(monkeypatch):
    from vesuvius.models.run import finalize_outputs

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, finalize_outputs,
                  ["in.zarr", "out.zarr", "--num_parts", "2", "--part_id", "5"])
    assert excinfo.value.code == 2


def test_finalize_bad_chunk_size_exits_with_usage(monkeypatch):
    from vesuvius.models.run import finalize_outputs

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, finalize_outputs,
                  ["in.zarr", "out.zarr", "--chunk_size", "128,128"])
    assert excinfo.value.code == 2


def test_finalize_threshold_value_without_threshold_exits_with_usage(monkeypatch):
    from vesuvius.models.run import finalize_outputs

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, finalize_outputs,
                  ["in.zarr", "out.zarr", "--threshold_value", "0.5"])
    assert excinfo.value.code == 2


def test_predict_bad_bbox_exits_with_usage(monkeypatch):
    from vesuvius.models.run import inference

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, inference,
                  ["--model_path", "model", "--input_dir", "in.zarr",
                   "--output_dir", "out", "--bbox", "1000-1400,:,:"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("module_name", [
    "vesuvius.models.run.finalize_outputs",
    "vesuvius.models.run.inference",
])
def test_main_binds_the_parser_it_reports_errors_on(module_name):
    """`main()` must keep a reference to its parser, not discard it inline."""
    import ast
    import importlib

    module = importlib.import_module(module_name)
    tree = ast.parse(open(module.__file__).read())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    uses_parser = any(isinstance(n, ast.Name) and n.id == "parser"
                      and isinstance(n.ctx, ast.Load) for n in ast.walk(main))
    binds_parser = any(isinstance(t, ast.Name) and t.id == "parser"
                       for n in ast.walk(main) if isinstance(n, ast.Assign)
                       for t in n.targets)
    assert not uses_parser or binds_parser
