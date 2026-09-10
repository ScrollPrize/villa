"""Loader for the compiled Spiral sampling helpers."""

import importlib
import os


def load_spiral_sampling():
    if os.environ.get('VC_DISABLE_NATIVE_SPIRAL_SAMPLING') == '1':
        return None
    try:
        return importlib.import_module('vc_spiral.spiral_sampling')
    except ImportError:
        return None
