"""glibc heap setting for command-line CPU runs of flat inference."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from vesuvius.ink_detection.inference import infer

from .test_flat_inference_ensemble import _save_checkpoint, _write_surface


def test_keep_freed_memory_in_heap_applies_on_linux_only():
    # A child process, because the setting is process-wide and would outlive this test.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from vesuvius.ink_detection.inference.infer import keep_freed_memory_in_heap; "
            "print(keep_freed_memory_in_heap())",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(sys.platform == "linux")


def test_main_applies_it_only_to_command_line_cpu_runs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(infer, "keep_freed_memory_in_heap", lambda: calls.append(1) or True)
    surface = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    def argv(output):
        return [str(surface), str(checkpoint), str(tmp_path / output),
                "--workers", "0", "--no-compile", "--device", "cpu"]

    assert infer.main(argv("library.tif")) == 0
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["infer", *argv("cli.tif")])
    assert infer.main() == 0
    assert calls == [1]
