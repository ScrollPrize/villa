"""Descriptor-limit startup and inheritance by mmap-backed data-loader workers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vesuvius.neural_tracing.fiber_follow import runloop, train


@pytest.mark.parametrize('platform,soft,hard,expected', [
    ('linux', 1024, 524288, (524288, 524288)),
    ('linux', 1024, -1, (1 << 20, -1)),
    ('linux', 524288, 524288, None),
    ('linux', -1, -1, None),
    ('darwin', 256, -1, (10240, -1)),
    ('darwin', 256, 4096, (4096, 4096)),
    ('darwin', 16384, -1, None),
])
def test_raise_soft_limit_without_lowering_it_or_changing_hard_limit(monkeypatch,platform,soft,hard,expected):
    calls=[]
    resource=SimpleNamespace(RLIMIT_NOFILE=7,RLIM_INFINITY=-1,
        getrlimit=lambda kind:(soft,hard),setrlimit=lambda kind,value:calls.append((kind,value)))
    monkeypatch.setitem(sys.modules,'resource',resource)
    monkeypatch.setattr(runloop.sys,'platform',platform)
    runloop.raise_open_file_limit()
    assert calls==([] if expected is None else [(7,expected)])


@pytest.mark.parametrize('failure',[ImportError,ValueError,OSError])
def test_unavailable_or_denied_limit_change_is_best_effort(monkeypatch,failure):
    def denied(*args): raise failure('unavailable')
    monkeypatch.setitem(sys.modules,'resource',SimpleNamespace(RLIMIT_NOFILE=7,RLIM_INFINITY=-1,
        getrlimit=lambda kind:(1024,4096),setrlimit=denied))
    runloop.raise_open_file_limit()


def test_training_and_preflight_raise_limit_before_parsing(monkeypatch):
    path=Path(__file__).parents[1]/'scripts/benchmark_single_path.py'
    spec=importlib.util.spec_from_file_location('benchmark_file_limit',path)
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    for module in (train,benchmark):
        calls=[]
        monkeypatch.setattr(module,'raise_open_file_limit',lambda:calls.append('raised'))
        with pytest.raises(SystemExit) as result:
            module.main(['--help'])
        assert result.value.code==0 and calls==['raised']


@pytest.mark.skipif(os.name!='posix',reason='POSIX resource limits')
@pytest.mark.parametrize('context',['spawn','forkserver'])
def test_low_limit_mmap_failure_is_fixed_and_workers_inherit_limit(tmp_path,context):
    import multiprocessing
    if context not in multiprocessing.get_all_start_methods():
        pytest.skip(f'{context} is unavailable')
    script=tmp_path/'probe.py'
    script.write_text('''
import contextlib
import io
import json
from pathlib import Path
import resource
import sys
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow import train

class Maps(torch.utils.data.IterableDataset):
    def __init__(self,path): self.path=path
    def __iter__(self):
        maps=[np.memmap(self.path,dtype='u1',mode='r',shape=(1,)) for _ in range(128)]
        try:
            assert all(int(m[0])==7 for m in maps)
            yield dict(limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],mappings=len(maps))
        finally:
            for m in maps: m._mmap.close()

if __name__=='__main__':
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard!=resource.RLIM_INFINITY and hard<256: sys.exit(77)
    path=Path(sys.argv[1]);path.write_bytes(bytes([7]))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,hard))
    maps=[];failure=None
    try:
        for _ in range(128): maps.append(np.memmap(path,dtype='u1',mode='r',shape=(1,)))
    except OSError as exc: failure=exc.errno
    finally:
        for m in maps: m._mmap.close()
    assert failure==24, failure
    # Exercise the actual trainer startup, stopping before data/training setup.
    with contextlib.redirect_stdout(io.StringIO()):
        try: train.main(['--help'])
        except SystemExit as exc: assert exc.code==0
    after=resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    assert after==resource.RLIM_INFINITY or after>=256
    loader=torch.utils.data.DataLoader(Maps(str(path)),batch_size=None,num_workers=2,
                                       multiprocessing_context=sys.argv[2])
    rows=list(loader)
    assert len(rows)==2 and all(r['limit']==after and r['mappings']==128 for r in rows)
    print(json.dumps(dict(before=64,failure_errno=failure,after=after,workers=rows)))
''')
    result=subprocess.run([sys.executable,str(script),str(tmp_path/'chunk'),context],
                          capture_output=True,text=True,timeout=60)
    if result.returncode==77: pytest.skip('Hard limit is too low for this regression test')
    assert result.returncode==0, result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report['failure_errno']==24 and len(report['workers'])==2
