"""vesuvius.predict must report the real error and exit 1 when inference fails (#1360).

``Inferer.infer`` catches its own exception, prints the traceback and returns
``None``; ``main`` then unpacked that ``None`` and reported
``cannot unpack non-iterable NoneType object`` as the failure. The exit code
was already 1, but the message pointed at the wrong thing.
"""

from __future__ import annotations

import sys

import pytest

from vesuvius.models.run import inference


class _FailingInferer:
    """Stands in for Inferer: construction succeeds, infer() fails the way the real one does."""

    def __init__(self, *args, **kwargs):
        self.skip_empty_patches = False

    def infer(self):
        try:
            raise FileNotFoundError("Checkpoint file not found: /models/x/fold_0/checkpoint_final.pth")
        except Exception as exc:  # mirrors Inferer.infer on main
            print(f"An error occurred during inference: {exc}")
            return None


def test_main_reports_infer_failure_and_exits_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(inference, "Inferer", _FailingInferer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesuvius.predict",
            "--model_path", str(tmp_path / "model"),
            "--input_dir", str(tmp_path / "in.zarr"),
            "--output_dir", str(tmp_path / "out"),
            "--device", "cpu",
        ],
    )
    rc = inference.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "Checkpoint file not found" in out
    assert "--- Inference Failed ---" in out
    assert "cannot unpack" not in out
