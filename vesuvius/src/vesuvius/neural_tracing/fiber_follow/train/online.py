"""One background DAgger collector; atomic replay publication to live workers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates


def publish_replay(index, paths):
    index = Path(index)
    temp = index.with_suffix('.partial.json')
    temp.write_text(json.dumps([os.path.abspath(p) for p in paths]))
    os.replace(temp, index)


class MultiSourceCollector:
    """Round-robin source-local replay; at most one GPU collector at a time.

    A launch while a collection is running is skipped and counted; the achieved
    cadence and publication age are reported with each completed collection.
    """
    def __init__(self, collectors):
        self.collectors = list(collectors)
        self.next_source = 0
        self.active = None

    def poll(self, step=None):
        if self.active is None:
            return None
        name, collector = self.collectors[self.active]
        event = collector.poll(step)
        if event is not None:
            self.active = None
            return dict(event, dataset=name)
        return None

    def launch(self, step, save):
        if self.active is not None:
            _, collector = self.collectors[self.active]
            if collector.due(step):
                collector.busy_skips += 1
            return False
        index = self.next_source
        _, collector = self.collectors[index]
        if not collector.launch(step, save):
            return False
        self.active = index
        self.next_source = (index+1) % len(self.collectors)
        return True

    def close(self):
        events = []
        for name, collector in self.collectors:
            event = collector.close()
            if event:
                events.append(dict(event, dataset=name))
        return dict(dagger_shutdown=events) if events else None


class OnlineCollector:
    """Training owns this process and continues updating its existing optimizer.

    At most one snapshot is being collected. A busy collection skips the launch;
    the next eligible snapshot is taken after completion. Failed collection is
    reported and does not publish incomplete data. A still-running collector is
    terminated when training ends; already published caches remain reusable.
    The fiber coverage cursor advances only when a collection is published.
    """
    def __init__(self, directory, fibers, val_z, device, every=1000, fibers_per_collection=64,
                 batch=8, forward_chunk=0, seed=0, replay_keep=4, initial=(), trace_len=768.,
                 before=48., after=64., stride=16., max_states=192, confidence=None, n_commit=None, gate=None,
                 collector_module='vesuvius.neural_tracing.fiber_follow.tracing.collect', extra_args=(), threads=4,
                 length_power=0.):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = self.directory/'replay.json'
        self.coverage = self.directory/'coverage.json'
        self.fibers, self.val_z, self.device = fibers, val_z, device
        self.every, self.fibers_per_collection, self.batch = every, fibers_per_collection, batch
        self.forward_chunk, self.seed, self.replay_keep = forward_chunk, seed, replay_keep
        self.trace_len, self.before, self.after, self.stride = trace_len, before, after, stride
        self.max_states = max_states
        self.confidence, self.n_commit, self.gate = confidence, n_commit, gate
        self.collector_module = collector_module
        if threads < 1:
            raise ValueError('Positive collector thread count required')
        self.threads = threads
        self.length_power = float(length_power)
        self.extra_args = [str(v) for v in extra_args]
        self.paths = list(initial)
        self.process = self.log = None
        self.output = None
        self.launched_step = self.previous_launch = None
        self.busy_skips = 0
        publish_replay(self.index, self.paths)

    def settings(self):
        return dict(every=self.every, fibers_per_collection=self.fibers_per_collection, batch=self.batch,
                    threads=self.threads, forward_chunk=self.forward_chunk, trace_len=self.trace_len, before=self.before,
                    after=self.after, stride=self.stride, max_states=self.max_states, replay_keep=self.replay_keep,
                    confidence=self.confidence, n_commit=self.n_commit, exploration='none',
                    length_power=self.length_power, **({} if self.gate is None else dict(gate=self.gate)))

    def due(self, step):
        return bool(self.every) and step % self.every == 0

    def poll(self, step=None):
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
        os.replace(self.output.with_suffix('.coverage.json'), self.coverage)
        source = int(states.provenance['step'])
        event = dict(dagger_states=len(states), dagger_source_step=source,
                     dagger_caches=len(self.paths), dagger_cache=str(self.output),
                     dagger_fibers=int(len(np.unique(states.fiber_idx))),
                     dagger_supply=states.provenance['supply'],
                     dagger_coverage=states.provenance['coverage'],
                     dagger_operating_policy=states.provenance['operating_policy'],
                     dagger_busy_skips=self.busy_skips,
                     dagger_launch_interval=(None if self.previous_launch is None
                                             else self.launched_step-self.previous_launch))
        if step is not None:
            event['dagger_publication_age'] = int(step)-source
        self.busy_skips = 0
        return event

    def launch(self, step, save):
        if not self.due(step):
            return False
        if self.process is not None:
            self.busy_skips += 1
            return False
        checkpoint = self.directory/f'source_{step:06d}.pt'
        self.output = self.directory/f'decisions_{step:06d}.npz'
        save(checkpoint)
        command = [sys.executable, '-m', self.collector_module,
                   '--threads', str(self.threads), '--checkpoint', str(checkpoint), '--fibers', self.fibers,
                   '--val-z', *map(str, self.val_z), '--device', self.device,
                   '--fibers-per-collection', str(self.fibers_per_collection), '--batch', str(self.batch),
                   '--forward-chunk', str(self.forward_chunk), '--trace-len', str(self.trace_len),
                   '--before', str(self.before), '--after', str(self.after), '--stride', str(self.stride),
                   '--max-states', str(self.max_states),
                   '--seed', str(self.seed+step), '--out', str(self.output), *self.extra_args]
        if self.coverage.exists():
            command += ['--coverage-state', str(self.coverage)]
        if self.length_power:
            command += ['--length-power', str(self.length_power)]
        for flag, value in (('--confidence', self.confidence), ('--n-commit', self.n_commit), ('--gate', self.gate)):
            if value is not None:
                command += [flag, str(value)]
        self.log = self.output.with_suffix('.log').open('w')
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        self.previous_launch, self.launched_step = self.launched_step, step
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
