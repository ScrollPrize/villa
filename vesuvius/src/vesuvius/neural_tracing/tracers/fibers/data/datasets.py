"""Explicit dataset provenance and weighted, source-local training batches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from vc_reader.reader import open_zarr

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import FiberStore
from vesuvius.neural_tracing.tracers.shared.annotations.open_data import open_data_volume
from vesuvius.neural_tracing.tracers.shared.annotations.splits import heldout_ids, validate_split_policy
from vesuvius.neural_tracing.tracers.shared.annotations.traced import fiber_manifest
from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
from vesuvius.neural_tracing.tracers.fibers.data.follow_dataset import FollowDataset
from vesuvius.neural_tracing.tracers.shared.paths import moved_path
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.fibers.data.observations import IdentityObservationBuilder
from vesuvius.neural_tracing.tracers.fibers.data.fiber_bank import FiberBank
from vesuvius.neural_tracing.tracers.fibers.data.seeds import make_seeds, read_manifest
from vesuvius.neural_tracing.tracers.shared.io import jsonable
from vesuvius.neural_tracing.tracers.fibers.frames.ct import SEED_HEADING_POLICY, TRACE_HEADING_POLICY, FRAME_POLICY
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import volume_key
from vesuvius.neural_tracing.tracers.fibers.data.replay import load_replay, own_replay_paths, usable_replay
from vesuvius.neural_tracing.tracers.fibers.data.episodes import loader_chunk


PRIMARY_OPTIONS = ('fibers', 'ct', 'manifest', 'val_z')


def ct_source_spec(source, cache_dir):
    return CTVolumeSpec(source['ct'], ct_level=int(source.get('ct_level', 0)),
        ct_grid_scale=float(source.get('ct_grid_scale', 1.)), grid_scale=float(source['grid_scale']), cache_dir=cache_dir)


def primary_source_spec(document):
    """The Paris 4 CT volume spec the trainer builds from a dataset config."""
    source = next(s for s in document['sources'] if s['kind'] == 'paris4')
    return CTVolumeSpec(source['ct'], ct_level=source.get('ct_level', 0), ct_grid_scale=source.get('ct_grid_scale', 4.),
        grid_scale=source.get('grid_scale', 8.), cache_dir=document['cache_dir'])


def read_dataset_config(path):
    path = Path(path).resolve()
    return parse_dataset_config(json.loads(path.read_text()), path.parent)


def parse_dataset_config(document, base):
    """A dataset configuration (dict) with relative paths resolved against ``base``, and its digest."""
    if document.get('version') != 1 or not document.get('sources'):
        raise ValueError('Dataset config requires version 1 and nonempty sources')
    document = json.loads(json.dumps(document))
    base = Path(base)
    def resolve(value):
        if value is None or '://' in str(value):
            return value
        return str(Path(moved_path(base / value)).resolve())
    document['cache_dir'] = resolve(document['cache_dir'])
    names = []
    for source in document['sources']:
        names.append(source['name'])
        weight = float(source['weight'])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Every dataset weight must be finite and positive')
        source['weight'] = weight
        validate_split_policy(source['validation'])  # Validate policy before dataset I/O.
        if source['kind'] == 'paris4':
            for key in PRIMARY_OPTIONS:
                if key in source and key != 'val_z':
                    source[key] = resolve(source[key])
        elif source['kind'] == 'fibers':
            source['path'] = resolve(source['path'])
            source['ct'] = resolve(source['ct'])
            if not source.get('coordinate_space'):
                raise ValueError('Fiber store sources require a coordinate identity')
        else:
            raise ValueError(f'Unknown dataset kind: {source["kind"]}')
    if len(set(names)) != len(names) or sum(s['kind'] == 'paris4' for s in document['sources']) != 1:
        raise ValueError('Require unique source names and exactly one Paris 4 source')
    primary = next(s for s in document['sources'] if s['kind'] == 'paris4')
    for key in ('fibers', 'ct', 'manifest', 'val_z'):
        if key not in primary:
            raise ValueError(f'Paris 4 source is missing {key}')
    digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    return document, digest


def validation_manifest(fibers, spec, seed=0, monitor_count=32):
    """Freeze distinct monitor/calibration/final fiber IDs and CT-only seeds."""
    ct = open_zarr(spec.ct_zarr, level=spec.ct_level, cache_dir=spec.cache_dir, memory_cache=256 << 20)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(fibers)).tolist()
    n = min(monitor_count, max(1,len(ids)//3))
    groups = dict(monitor=ids[:n],calibration=ids[n:n+max(1,(len(ids)-n)//2)],
                  final=ids[n+max(1,(len(ids)-n)//2):])
    result = dict(version=3,fibers=fiber_manifest(fibers),volume=spec.to_dict(),
                  seed_heading_policy=SEED_HEADING_POLICY,heading_policy=TRACE_HEADING_POLICY,frame_policy=FRAME_POLICY)
    for name,members in groups.items():
        result[name+'_fibers'] = members
        seeds = []
        # Keep manifest generation bounded: reserve all IDs, but only generate
        # monitor-sized seed subsets for calibration/final on large fiber catalogs.
        for fi in members[:monitor_count]:
            candidates = make_seeds([fibers[fi]],ct,spec,per_fiber=1,seed=seed+fi)
            if candidates:
                value = candidates[0]
                seeds.append(dict(value,fiber=fi))
        result[name] = seeds
    result = json.loads(json.dumps(result,default=jsonable))
    result['sha256'] = hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def load_primary_dataset(document, spec):
    """Keep legacy reserved fibers held out, plus a random split of the rest.

    Fibers whose annotation doubles back are kept out of training; the held-out
    split is decided before that and does not change.
    """
    source = next(s for s in document['sources'] if s['kind']=='paris4')
    fibers = load_fibers(source['fibers'],grid_scale=spec.grid_scale)
    legacy = read_manifest(source['manifest'])
    reserved = {f['name'] for f in legacy['fibers']}
    extra = heldout_ids([f.name for f in fibers if f.name not in reserved],source['validation'])
    val = [f for f in fibers if f.name in reserved|extra]
    train = [f for f in fibers if f.name not in reserved|extra and not f.foldbacks]
    manifest = validation_manifest(val,spec,source['validation']['seed'])
    return fibers,train,val,manifest


class WeightedDatasets(torch.utils.data.IterableDataset):
    """Choose a source per equal-sized batch; each source applies its own task budget.

    Workers use independent deterministic RNG streams. Never fall back to a
    different source on an I/O failure, which would silently change weights.
    """
    def __init__(self, datasets, names, weights, seed=0):
        if not datasets or len(datasets) != len(names) or len(names) != len(weights):
            raise ValueError('Dataset names, weights and datasets must align')
        weights = np.asarray(weights, float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Invalid dataset sampling weights')
        if len({d.chunk for d in datasets}) != 1:
            raise ValueError('Weighted datasets require equal batch sizes')
        self.datasets, self.names = datasets, names
        self.weights, self.seed = weights/weights.sum(), seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7349, 0 if worker is None else worker.id]))
        streams = [None]*len(self.datasets)
        while True:
            index = int(rng.choice(len(streams), p=self.weights))
            if streams[index] is None:
                streams[index] = iter(self.datasets[index])
            batch = next(streams[index])
            rows = len(batch['hist']) if 'hist' in batch else self.datasets[index].chunk  # episode batches: chunk*steps
            batch['dataset_id'] = torch.full((rows,), index, dtype=torch.int64)
            yield batch


def open_fiber_source(source, cache_dir, normalization=None):
    """Checked fiber-store training/validation fibers and CT spec for one dataset-config source.

    The fibers are those of the store bound to the volume ``coordinate_space`` (SAMPLE/VOLUME_ID) names, and the
    configured CT must be that open-data volume. Returns (training fibers, validation fibers, CT spec).
    """
    path = Path(source['path'])
    volume = source['coordinate_space'].split('/')[-1]
    # A local copy of the CT (same coordinate frame, read at ct_grid_scale) names the store it mirrors.
    ct = open_data_volume(source.get('ct_mirror_of', source['ct']))
    if ct is None or f'{ct[0]}/{ct[1]}' != source['coordinate_space']:
        raise ValueError(f'Fiber source CT is not the open-data volume {source["coordinate_space"]}')
    scale = float(source['grid_scale'])
    fibers = FiberStore(path, volume, scale, validation=source['validation'])
    validation_fibers = FiberStore(path, volume, scale, validation=source['validation'], split='validation')
    spec = ct_source_spec(source, cache_dir)
    if normalization is not None:
        spec.ct_normalization = normalization['volumes'][volume_key(spec)]
    return fibers, validation_fibers, spec


def build_mixed_dataset(primary, document, cfg, sample, sampling, args, *, seed, budget, out=None, resume=False,
                        normalization=None):
    datasets, names, weights, provenance = [], [], [], []
    for index, source in enumerate(document['sources']):
        if source['kind'] == 'paris4':
            dataset = primary
            entry = dict(name=source['name'], kind='paris4', fibers=len(primary.fibers))
        else:
            fibers, validation_fibers, spec = open_fiber_source(source, document['cache_dir'], normalization)
            scale = float(source['grid_scale'])
            builder = IdentityObservationBuilder(cfg, fibers, sampling, augment=True, neighbors=FiberBank(fibers))
            replay_index = Path(out)/'dagger'/source['name']/'replay.json' if out else None
            replay_paths = (own_replay_paths(json.loads(replay_index.read_text()), Path(out)/'dagger')
                            if resume and replay_index and replay_index.exists() else [])
            replay = usable_replay(load_replay(replay_paths), fibers, sample.n_history, spec.grid_scale)
            dataset = FollowDataset(fibers, spec, sample, chunk=loader_chunk(cfg, args.batch),
                seed=seed+100003*(index+1), cache_bytes=int(args.worker_cache_gb*(1<<30)),
                batch_builder=builder, budget=budget, length_power=args.store_length_power,
                onpolicy=replay, replay_index=str(replay_index) if replay_index else None)
            dataset.validation_fibers = validation_fibers
            dataset.validation_manifest = validation_manifest(validation_fibers,spec,source['validation']['seed'])
            entry = dict(name=source['name'], kind='fibers', fibers=len(fibers),
                excluded_fibers=len(validation_fibers), digest=fibers.manifest_entries()[0]['digest'],
                coordinate_space=source['coordinate_space'],
                validation=source['validation'], grid_scale=scale)
        datasets.append(dataset)
        names.append(source['name'])
        weights.append(source['weight'])
        provenance.append(entry)
    return WeightedDatasets(datasets, names, weights, seed), provenance
