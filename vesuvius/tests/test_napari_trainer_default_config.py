"""vesuvius.napari_trainer starts from a default config that exists in the package.

The path climbed one folder too many (src/models/... instead of
src/vesuvius/models/...), so the trainer window failed at launch with
FileNotFoundError on every install.

main_window builds Qt widgets when imported, so it is imported in a child
process on Qt's offscreen platform, where a missing display cannot abort pytest.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")


def test_default_config_exists():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c",
         "from vesuvius.napari_trainer import main_window as m; print(m.DEFAULT_CONFIG_PATH)"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if result.returncode != 0 and any(k in result.stderr for k in ("Qt", "qt.qpa", "display")):
        pytest.skip(f"no usable Qt here: {result.stderr.strip()[-300:]}")
    assert result.returncode == 0, result.stderr
    path = Path(result.stdout.strip().splitlines()[-1])
    assert path.is_file()
    assert path.parts[-5:-3] == ("vesuvius", "models")
