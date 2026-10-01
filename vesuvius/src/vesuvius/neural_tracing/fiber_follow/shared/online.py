"""One background DAgger collector; atomic replay publication to live workers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.data import OnPolicyStates, REPLAY_FAILURES
from vesuvius.neural_tracing.fiber_follow.shared.trace import DEFAULT_CONFIDENCE, DEFAULT_N_COMMIT


def publish_replay(index, paths):
    index = Path(index)
    temp = index.with_suffix('.partial.json')
    temp.write_text(json.dumps([os.path.abspath(p) for p in paths]))
    os.replace(temp, index)


class MultiSourceCollector:
    """Round-robin source-local replay; at most one GPU collector at a time."""
    def __init__(self, collectors):
        self.collectors = list(collectors)
        self.next_source = 0
        self.active = None

    def poll(self):
        if self.active is None:
            return None
        name, collector = self.collectors[self.active]
        event = collector.poll()
        if event is not None:
            self.active = None
            return dict(event, dataset=name)
        return None

    def launch(self, step, save):
        if self.active is not None:
            return False
        index = self.next_source
        _,collector = self.collectors[index]
        if not collector.launch(step,save):
            return False
        self.active = index
        self.next_source = (index+1)%len(self.collectors)
        return True

    def close(self):
        events = []
        for name,collector in self.collectors:
            event = collector.close()
            if event:
                events.append(dict(event,dataset=name))
        return dict(dagger_shutdown=events) if events else None


class OnlineCollector:
    """Training owns this process and continues updating its existing optimizer.

    At most one snapshot is being collected. Busy collection skips a launch;
    the next eligible snapshot is taken after completion. Failed collection is
    reported and does not publish incomplete data. A still-running collector is
    terminated when training ends; already published caches remain reusable.
    """
    def __init__(self, directory, fibers, val_z, device, every=1000, max_seeds=64,
                 batch=1, explore_calls=8, seed=0, replay_keep=4, initial=(),
                 trace_len=6000., confidence=DEFAULT_CONFIDENCE, n_commit=DEFAULT_N_COMMIT,
                 collector_module='vesuvius.neural_tracing.fiber_follow.flow_matching.collect', seeds_per_fiber=2,
                 extra_args=()):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = self.directory/'replay.json'
        self.fibers, self.val_z, self.device = fibers, val_z, device
        self.every, self.max_seeds, self.batch = every, max_seeds, batch
        self.explore_calls, self.seed, self.replay_keep = explore_calls, seed, replay_keep
        self.trace_len, self.confidence, self.n_commit = trace_len, confidence, n_commit
        self.seeds_per_fiber = seeds_per_fiber
        self.collector_module = collector_module
        self.extra_args = [str(v) for v in extra_args]
        self.paths = list(initial)
        self.process = self.log = None
        self.output = None
        publish_replay(self.index, self.paths)

    def poll(self):
        if self.process is None or self.process.poll() is None:
            return None
        code = self.process.returncode
        self.log.close()
        self.process = self.log = None
        if code:
            return dict(dagger_error=code, collection_log=str(self.output.with_suffix('.log')))
        states = OnPolicyStates.load(self.output)
        self.paths = (self.paths+[states._dir])[-self.replay_keep:]
        publish_replay(self.index, self.paths)
        return dict(dagger_states=len(states), dagger_source_step=states.provenance['step'],
                    dagger_caches=len(self.paths), dagger_cache=str(self.output),
                    dagger_fibers=int(len(np.unique(states.fiber_idx))),
                    dagger_hard=int(np.count_nonzero(states.hard)),
                    dagger_failures={name:int(np.count_nonzero(states.failure_kind == i))
                                     for i, name in enumerate(REPLAY_FAILURES)},
                    dagger_exploratory=int(np.count_nonzero(states.exploratory)))

    def launch(self, step, save):
        if not self.every or step % self.every or self.process is not None:
            return False
        checkpoint = self.directory/f'source_{step:06d}.pt'
        self.output = self.directory/f'decisions_{step:06d}.npz'
        save(checkpoint)
        command = [sys.executable, '-m', self.collector_module,
                   '--checkpoint', str(checkpoint), '--fibers', self.fibers,
                   '--val-z', *map(str, self.val_z), '--device', self.device,
                   '--max-seeds', str(self.max_seeds), '--batch', str(self.batch),
                   '--seeds-per-fiber', str(self.seeds_per_fiber),
                   '--explore-calls', str(self.explore_calls), '--trace-len', str(self.trace_len),
                   '--confidence', str(self.confidence), '--n-commit', str(self.n_commit),
                   '--seed', str(self.seed+step), '--out', str(self.output), *self.extra_args]
        self.log = self.output.with_suffix('.log').open('w')
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        return True

    def close(self):
        event = self.poll()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.log.close()
            self.process = self.log = None
            return dict(dagger_unfinished=str(self.output), message='Training ended; unfinished collection was not published')
        return event
