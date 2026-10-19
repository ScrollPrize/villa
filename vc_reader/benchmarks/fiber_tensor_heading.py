"""Sheet-normal heading calculation and CUDA service for the fiber benchmark.

The service runs in the training process. Workers send small query descriptions;
only headings/frames return to CPU. CT patches and tensor intermediates remain
on CUDA. The trainer's continuation prediction-axis policy stays in effect.
"""
from collections import OrderedDict
from functools import partial
import multiprocessing as mp
from multiprocessing.connection import wait
import threading
import time
import traceback
from types import SimpleNamespace
from fiber_timeline import ENABLED as TIMELINE, LOCAL, span

import numpy as np
import torch

from fiber_cuda_pipeline import reader_spec, install_worker_reader, ReaderDataLoader

CLIENT = None
SERVICE = None


class Client:
    def __init__(self, connection):
        self.connection = connection
        self.lock = threading.Lock()
        self.seeds = OrderedDict()
        self.trace_context = None
        self.trace_seconds = 0.
        self.trace_requests = 0

    def request(self, vol, positions, priors, families, mode='frame', previous=None):
        spec = reader_spec(vol.ct)
        # Training workers have passed the shared prefetch readiness gate.
        # Manifest construction before timing may still fetch uncached context.
        if spec.get('offline'):
            spec['cache_writes'] = False
        n = len(positions)
        anchors = np.stack([np.zeros(3) if f is None else np.asarray(f)[:,0]
                            for f in (previous if previous is not None else [None]*n)])
        request = dict(reader=spec, points=np.asarray(positions)*vol.input_scale,
                       priors=np.asarray(priors, np.float32), families=list(families),
                       anchors=anchors.astype(np.float32), mode=mode)
        with self.lock:
            began = time.perf_counter()
            if TIMELINE:
                request['trace_sent'] = began
                request['trace_context'] = self.trace_context
            ok, value = self.exchange(request)
            if TIMELINE:
                self.trace_seconds += time.perf_counter()-began
                self.trace_requests += 1
        if not ok:
            raise RuntimeError('Heading calculation failed:\n'+value)
        return value

    def exchange(self, request):
        self.connection.send(request)
        return self.connection.recv()

    def seed_key(self, vol, pos, family, direction):
        return (id(vol), tuple(pos), str(family), tuple(direction))

    def prepare_seeds(self, items, vol):
        pending = [i for i in items if i.get('seed_heading_family') is not None]
        if not pending:
            return
        heads = self.request(vol, [i['seed_pos'] for i in pending], [i['seed_tangent'] for i in pending],
                             [i['seed_heading_family'] for i in pending], mode='seed')
        for item, head in zip(pending, heads):
            self.seeds[self.seed_key(vol,item['seed_pos'],item['seed_heading_family'],item['seed_tangent'])] = head
        while len(self.seeds) > 256:
            self.seeds.popitem(last=False)


def normal_context(pos_xyz, input_scale):
    center = np.asarray(pos_xyz, np.float64)[::-1]*input_scale
    return np.floor(center/3).astype(np.int64)*3-54, np.full(3,109,np.int64)


def crop_frame_bounds(item, crop, vol, predictor=None):
    from vesuvius.neural_tracing.tracers.fibers.frames import ct
    if item.get('frame_policy') in ct.FRAME_POLICIES:
        yield from ct.frame_prefetch_bounds(item, crop, vol.input_scale)
    else:
        yield ct.heading_free_bounds(item['pos'], crop, vol.input_scale)
        yield normal_context(item['pos'], vol.input_scale)
    if item.get('seed_heading_family') is not None:
        yield normal_context(item['seed_pos'], vol.input_scale)


def seed_heading(vol, pos, family, direction):
    from vesuvius.neural_tracing.tracers.fibers.frames.ct import SeedHeadingError
    key = CLIENT.seed_key(vol,pos,family,direction)
    head = CLIENT.seeds.pop(key, None)
    if head is None:
        head = CLIENT.request(vol,[pos],[direction],[family],mode='seed')[0]
    if not np.isfinite(head).all():
        raise SeedHeadingError('Anisotropic CT tensor has no usable H/V seed heading')
    return head.astype(np.float64)


def held_frame(vol, pos, heading, previous=None, *, fallback=None, diagnostics=None):
    frame = CLIENT.request(vol,[pos],[heading],['H'],mode='held',previous=[previous if previous is not None else fallback])[0]
    if diagnostics is not None:
        diagnostics.update(source=0, energy=0., gap=0.)
    return frame.astype(np.float64)


def forbid_trainer_tensor(*args, **kwargs):
    raise RuntimeError('Trainer tensor calculation bypassed the selected vc_reader backend')


class Predictor:
    model = SimpleNamespace(cfg=SimpleNamespace(predict_frames=True))

    def predict_frames(self, vol, positions, priors, paths, families, previous=None, pool=None):
        return list(CLIENT.request(vol,positions,priors,families,previous=previous).astype(np.float64))


def load_predictor(*args):
    return Predictor()


def install_heading(connection=None, *, client=None):
    global CLIENT
    CLIENT = Client(connection) if client is None else client
    from vesuvius.neural_tracing.tracers.fibers.frames import ct, learned
    from vesuvius.neural_tracing.tracers.fibers.data import observations, targets, follow_dataset, seeds
    from vesuvius.neural_tracing.tracers.fibers.train import live_continuation
    ct.normal_context = learned.normal_context = normal_context
    learned._load_predictor = load_predictor
    learned.crop_frame_bounds = observations.crop_frame_bounds = crop_frame_bounds
    ct.oriented_seed_heading = targets.oriented_seed_heading = seeds.oriented_seed_heading = seed_heading
    ct.ct_frame = live_continuation.ct_frame = held_frame
    ct.ct_normal = ct.ct_tensor = forbid_trainer_tensor
    original = follow_dataset.FollowDataset.resolve_seeds
    if not getattr(original, '_tensor_heading', False):
        def resolve_seeds(self, items, vol, rng, windows):
            CLIENT.prepare_seeds(items, vol)
            return original(self, items, vol, rng, windows)
        resolve_seeds._tensor_heading = True
        follow_dataset.FollowDataset.resolve_seeds = resolve_seeds


def init_worker(worker_id, connections):
    install_worker_reader(worker_id)
    install_heading(connections[worker_id])
    if TIMELINE:
        from fiber_timeline import install_worker
        install_worker(CLIENT)


class HeadingDataLoader(ReaderDataLoader.__bases__[0]):
    def __init__(self, *args, **kwargs):
        SERVICE.pool.finish_startup()
        connections = SERVICE.add_clients(kwargs.get('num_workers',0))
        super().__init__(*args, worker_init_fn=partial(init_worker, connections=connections), **kwargs)


class HeadingCalculator:
    def __init__(self, pool, *, device):
        self.pool, self.device = pool, device
        self.counters = dict(requests=0, queries=0, seed_queries=0, frame_queries=0,
                             invalid_normals=0, result_download_bytes=0, seconds=0.)

    @torch.no_grad()
    def calculate(self, request):
        from vc_reader.sheet_normal import sample_sheet_normals
        from vesuvius.neural_tracing.tracers.fibers.heading_model.frames import orthonormal_frame
        reader = self.pool.get(request['reader'], 'heading')
        outputs = []
        for start in range(0,len(request['points']),8):
            part = slice(start,start+8)
            result = torch.empty((len(request['points'][part]),4),device=self.device,dtype=torch.float32)
            sample_sheet_normals(reader,request['points'][part],
                                 out=result if self.device == 'cuda' else result.numpy())
            normal, gap = result[:,:3],result[:,3]
            prior = torch.as_tensor(request['priors'][part],device=self.device)
            vertical = torch.tensor([str(f).upper() in ('V','VERTICAL') for f in request['families'][part]],device=self.device)
            z = normal.new_tensor([0.,0.,1.]).expand_as(normal)
            heading = torch.where(vertical[:,None],z-normal[:,2:3]*normal,torch.linalg.cross(z,normal,dim=-1))
            length = heading.norm(dim=-1,keepdim=True)
            valid = (gap>0) & (length[:,0]>=1e-3)
            heading = heading/length.clamp_min(1e-3)
            heading = torch.where((heading*prior).sum(-1,keepdim=True)<0,-heading,heading)
            # Invalid seeds are rejected; an unresolved frame holds its prior
            # heading and obtains a deterministic/transported roll as in training.
            if request['mode'] == 'seed':
                output = torch.where(valid[:,None],heading,torch.nan)
            else:
                heading = prior if request['mode']=='held' else torch.where(valid[:,None],heading,prior)
                anchors = torch.as_tensor(request['anchors'][part],device=self.device)
                output = orthonormal_frame(heading,normal,anchors)
            # One small result download includes the validity flag for accounting.
            packed = torch.cat((output.flatten(1),(gap<=0).float()[:,None]),dim=1).cpu().numpy()
            self.counters['result_download_bytes'] += packed.nbytes if self.device == 'cuda' else 0
            self.counters['invalid_normals'] += int(packed[:,-1].sum())
            outputs.append(packed[:,:-1].reshape(output.shape))
        return np.concatenate(outputs)


class Service(HeadingCalculator):
    def __init__(self, pool, *, stream='default'):
        super().__init__(pool, device='cuda')
        self.stream_mode = stream
        self.connections = []
        self.timeline = []
        self.pool = pool
        self.stop = threading.Event()
        self.lock = threading.Lock()
        main = self.add_clients(1)[0]
        install_heading(main)
        self.thread = threading.Thread(target=self.run, name='cuda-sheet-heading', daemon=True)
        self.thread.start()

    def add_clients(self, n):
        pairs = [mp.get_context().Pipe() for _ in range(n)]
        with self.lock:
            self.connections.extend(a for a,b in pairs)
        return [b for a,b in pairs]

    def run(self):
        if self.stream_mode == 'separate':
            from vc_reader import _native
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                _native._set_stream(stream.cuda_stream)
                try:
                    self.serve()
                finally:
                    stream.synchronize()
                    _native._set_stream(0)
        else:
            self.serve()

    def serve(self):
        while not self.stop.is_set():
            with self.lock:
                connections = list(self.connections)
            for conn in wait(connections,timeout=.05):
                try:
                    request = conn.recv()
                except (EOFError,OSError):
                    with self.lock:
                        self.connections.remove(conn)
                    continue
                began = time.perf_counter()
                cpu = time.thread_time()
                if TIMELINE:
                    LOCAL.stages = {}
                try:
                    with span('heading.request'):
                        result = self.calculate(request)
                    self.counters['requests'] += 1
                    self.counters['queries'] += len(result)
                    self.counters['seed_queries' if request['mode']=='seed' else 'frame_queries'] += len(result)
                    conn.send((True,result))
                except Exception:
                    conn.send((False,traceback.format_exc()))
                finished = time.perf_counter()
                self.counters['seconds'] += finished-began
                if TIMELINE:
                    self.timeline.append(dict(sent=request['trace_sent'], began=began, finished=finished,
                        context=request['trace_context'], mode=request['mode'], queries=len(request['points']),
                        cpu_seconds=time.thread_time()-cpu, stages=LOCAL.stages))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError('CUDA heading service did not stop')

    def stats(self):
        from vc_reader.sheet_normal import RECIPE
        return dict(recipe=RECIPE,backend='cuda',counters_including_startup_warmup_prefetch=self.counters,
                    stream=self.stream_mode,
                    chunk_cache='statistics in reader_stats.chunk_cache',
                    reference_commit='cd0ee74e3a4f8dd7f3b4884b01c7e411fc49ee6e',
                    result_transfer='GPU to CPU: headings/frames and validity only',
                    continuation_axis='trained previous-prediction policy retained',
                    recorded_replay_frames='retained')


def install(pool, *, stream='default'):
    global SERVICE
    SERVICE = Service(pool, stream=stream)
    torch.utils.data.DataLoader = HeadingDataLoader
    return SERVICE
