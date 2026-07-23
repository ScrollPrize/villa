"""Optional loader for the VC native Spiral sampling helpers."""

import importlib
import os
from pathlib import Path


def load_native_spiral_sampling():
    if os.environ.get('VC_DISABLE_NATIVE_SPIRAL_SAMPLING') == '1':
        return None
    try:
        return importlib.import_module('vc.spiral_sampling')
    except ImportError:
        try:
            import vc
            build_package = Path(__file__).resolve().parents[2] / 'build/python/vc'
            if build_package.is_dir() and str(build_package) not in vc.__path__:
                vc.__path__.append(str(build_package))
            return importlib.import_module('vc.spiral_sampling')
        except ImportError:
            return None

