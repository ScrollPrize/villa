"""`vesuvius.finalize_outputs` must reach finalize_logits instead of dying on its own parser.

On main 23adee0, main() did `args = build_parser().parse_args()` and then called
`parser.error(...)` / `resolve_threshold(parser, args)`, so every invocation,
for any arguments and either mode, raised NameError: name 'parser' is not defined.
"""

from __future__ import annotations

import sys

import pytest

from vesuvius.models.run import finalize_outputs


def test_cli_main_reaches_finalize_logits(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(finalize_outputs, "finalize_logits", lambda **kw: calls.update(kw))
    monkeypatch.setattr(sys, "argv", ["vesuvius.finalize_outputs", str(tmp_path / "in.zarr"),
                                      str(tmp_path / "out.zarr"), "--mode", "multiclass", "--threshold"])
    assert finalize_outputs.main() == 0
    assert calls["mode"] == "multiclass" and calls["threshold"] == 0.5


def test_cli_main_reports_bad_partition_through_the_parser(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["vesuvius.finalize_outputs", "in.zarr", "out.zarr",
                                      "--num_parts", "2", "--part_id", "5"])
    with pytest.raises(SystemExit) as excinfo:  # parser.error, not NameError
        finalize_outputs.main()
    assert excinfo.value.code == 2
