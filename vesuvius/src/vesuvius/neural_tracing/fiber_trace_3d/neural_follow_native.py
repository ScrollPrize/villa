"""Select the workspace-built binding explicitly, without replacing an installation."""
import importlib.util
import os
from pathlib import Path
import sys
import sysconfig


def load_native():
    directory = os.environ.get('FIBER_FOLLOW_NATIVE_DIR')
    if not directory:
        from vc import volume
        if not hasattr(volume.NormalSampler, 'replay'):
            raise RuntimeError('rebuild vc_volume or set FIBER_FOLLOW_NATIVE_DIR to workspace build/python/vc')
        return volume
    directory=Path(directory)
    path=directory/('volume'+sysconfig.get_config_var('EXT_SUFFIX'))
    if 'vc.volume' in sys.modules:
        module=sys.modules['vc.volume']
        if Path(module.__file__).resolve()!=path.resolve():
            raise RuntimeError('a different vc.volume binding was loaded before the benchmark')
        return module
    if 'vc' not in sys.modules:
        package=importlib.util.spec_from_file_location('vc',directory/'__init__.py',submodule_search_locations=[str(directory)])
        sys.modules['vc']=importlib.util.module_from_spec(package)
    spec=importlib.util.spec_from_file_location('vc.volume',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules['vc.volume']=module
    spec.loader.exec_module(module)
    sys.modules['vc'].volume=module
    # Match the ordinary vc package exports without invoking editable finders.
    for name in ('Volume','set_chunk_cache_budget','set_chunk_cache_decode_threads','set_chunk_cache_io_threads'):
        setattr(sys.modules['vc'],name,getattr(module,name))
    return module
