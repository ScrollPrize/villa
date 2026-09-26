"""Learned native beam scoring.

An explicit VC_PYTHON_BUILD_DIR lets development runs use rebuilt bindings
without reinstalling the VC package, including forkserver training workers.
"""
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path

_build = os.environ.get('VC_PYTHON_BUILD_DIR')
if _build:
    _python_dir = Path(_build).resolve() / 'python'
    if not (_python_dir / 'vc' / '__init__.py').is_file():
        raise ImportError(f'VC_PYTHON_BUILD_DIR={_build} has no python/vc package')
    if 'vc' in sys.modules and Path(sys.modules['vc'].__file__).resolve().parent != _python_dir / 'vc':
        raise ImportError('Set VC_PYTHON_BUILD_DIR before importing vc')

    class _VCBuildFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'vc':
                return PathFinder.find_spec(fullname, [str(_python_dir)])
            if fullname.startswith('vc.'):
                return PathFinder.find_spec(fullname, path)
            return None

    sys.meta_path.insert(0, _VCBuildFinder())
