# Host-RAM offload hook for fit_spiral. Put this directory FIRST on PYTHONPATH of the fit process (only the first sitecustomize.py on the path runs).
# 1. Optionally chains another sitecustomize.py (FIT_CHAIN_HOOK=<path>).
# 2. FIT_SPIRAL_POOL_OFFLOAD=1 (default OFF): when fit_spiral imports sparse_cuda_cache, ResidentBrickPool is replaced by rb_offload.OffloadBrickPool
#    (pools in mapped host RAM, zero-copy gather kernel; bit-identical values).  Without the variable nothing is patched.
import os, sys
_chain = os.environ.get("FIT_CHAIN_HOOK", "")
if _chain and os.path.exists(_chain):
    exec(compile(open(_chain).read(), _chain, "exec"), {"__name__": "chained_sitecustomize", "__file__": _chain})
if os.environ.get("FIT_SPIRAL_POOL_OFFLOAD") == "1" and "fit_spiral" in " ".join(sys.argv):      # only the fit process (run142.sh also starts helper pythons)
    import importlib.abc, importlib.machinery
    _here = os.path.dirname(os.path.abspath(__file__))

    class _PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "sparse_cuda_cache": return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None: return spec
            orig = spec.loader.exec_module
            def exec_module(module, _orig=orig):
                _orig(module)
                sys.path.insert(0, _here); import rb_offload; rb_offload.install(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchFinder())
    print("pool_offload hook: FIT_SPIRAL_POOL_OFFLOAD=1 armed (sparse_cuda_cache.ResidentBrickPool will be replaced on import)", flush=True)
