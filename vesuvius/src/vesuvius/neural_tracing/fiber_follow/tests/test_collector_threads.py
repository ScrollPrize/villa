"""DAgger always bounds intra-op threads before loading its model."""
import subprocess
import sys
from types import SimpleNamespace

from vesuvius.neural_tracing.fiber_follow.train.online import OnlineCollector


def test_online_collector_passes_thread_limit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kw: calls.append(command) or SimpleNamespace())
    collector = OnlineCollector(tmp_path/'dagger', 'fibers', (0, 1), 'cpu', every=1, threads=3)
    collector.launch(1, lambda path: None)
    collector.log.close()
    assert calls[0][calls[0].index('--threads')+1] == '3'
    assert collector.settings()['threads'] == 3


def test_collector_child_applies_limit_before_loading_checkpoint(tmp_path):
    script = '''
import torch
from vesuvius.neural_tracing.fiber_follow.tracing.collection import main
class Loaded(Exception): pass
def load(*args):
    assert torch.get_num_threads() == 3
    raise Loaded()
try:
    main(['--checkpoint','unused','--fibers','unused','--out','unused','--threads','3'], checkpoint_loader=load)
except Loaded:
    print('THREAD_LIMIT_VERIFIED')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'THREAD_LIMIT_VERIFIED' in result.stdout
