"""Python bindings for Volume Cartographer's fiber tooling: the fiber tracer
(vc.fiber_trace), Fiber3D training kernels and the Lasagna normal sampler
(vc.volume) and fiber evaluation.

Volume reads and sampling are vc_reader's (vc_reader.reader.open_zarr)."""

import os as _os
import sys as _sys

if _sys.platform == "win32":
    # Windows has no RPATH, and Python 3.8+ does not search PATH for an
    # extension module's dependent DLLs. The vc_* DLLs the extension modules
    # link install to the site-packages root and, on Windows, as the full
    # runtime closure (object store, SFTP) to site-packages/bin, so register
    # both before any extension module is imported.
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for _dir in (_root, _os.path.join(_root, "bin")):
        if _os.path.isdir(_dir):
            _os.add_dll_directory(_dir)
    del _root, _dir
del _os, _sys
