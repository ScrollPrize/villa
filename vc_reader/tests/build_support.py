"""Select an existing CMake build of the native modules; nothing is installed."""
import os
from pathlib import Path
import sys


def load_build(directory=None):
    """Use the build in ``directory`` (default VC_READER_BUILD_DIR), including in spawned workers."""
    directory = directory or os.environ.get('VC_READER_BUILD_DIR')
    if directory:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
        from extension_build import load_extensions
        load_extensions(Path(directory))
