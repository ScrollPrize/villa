"""Bounded, ordered CPU diagnostics over immutable optimizer snapshots."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time


class OrderedDiagnostics:
    def __init__(self, workers, publish):
        if workers<1:raise ValueError('Diagnostic workers must be positive')
        self.executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='uv-audit')
        self.pending=deque();self.limit=workers;self.publish=publish
        self.wait_seconds=0.
        self.lock=Lock();self.running=0;self.peak_running=0;self.max_pending=0

    def _execute(self,fn,args):
        with self.lock:
            self.running+=1;self.peak_running=max(self.peak_running,self.running)
        try:return fn(*args)
        finally:
            with self.lock:self.running-=1

    def _drain_one(self):
        started=time.monotonic()
        result=self.pending.popleft().result()
        self.wait_seconds+=time.monotonic()-started
        self.publish(result)

    def reserve(self):
        while self.pending and (len(self.pending)>=self.limit or self.pending[0].done()):
            self._drain_one()

    def submit(self, fn, *args):
        self.reserve()
        self.pending.append(self.executor.submit(self._execute,fn,args))
        self.max_pending=max(self.max_pending,len(self.pending))

    def shutdown(self):
        self.executor.shutdown(wait=True,cancel_futures=True)

    def close(self):
        try:
            while self.pending:self._drain_one()
        finally:self.shutdown()


def evaluate_snapshot(d,snapshot,widths,config,indices,projection,record,copy_seconds,profile_path=None,gt_scorer=None):
    """Shared sync/async CPU-only evaluation; no tensors or mutable model access."""
    from . import fiber_uv_layout as uv
    from .fiber_reference_comparison import reference_winding_metrics
    profile=None
    if profile_path is not None:
        import cProfile
        profile=cProfile.Profile();profile.enable()
    started=time.monotonic()
    geometry=(uv.hard_geometry(d,snapshot,widths,config) if config.get('diagnostic_backend','optimized')=='optimized' else None)
    record.update(uv.audit(d,snapshot,widths,config,indices,geometry=geometry))
    audited=time.monotonic()
    if projection is not None:
        record['reference']=reference_winding_metrics(d,snapshot,widths,config,projection,geometry=geometry)
    if gt_scorer is not None:record['gt']=gt_scorer.score_state(snapshot,widths,config)
    done=time.monotonic()
    timing=dict(step=record['step'],copy_sync_seconds=copy_seconds,
        audit_seconds=audited-started,reference_seconds=done-audited,
        total_seconds=done-started+copy_seconds)
    if profile is not None:
        profile.disable();profile.dump_stats(str(profile_path))
    return record,timing
