"""Explicit dataset provenance and weighted, source-local training batches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.data.data import FollowDataset, fiber_manifest, load_fibers
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder
from vesuvius.neural_tracing.fiber_follow.data.afv_neighbors import AFVBank


PRIMARY_OPTIONS = ('fibers', 'ct', 'manifest', 'val_z')


def ct_source_spec(source, cache_dir):
    return FiberVolumeSpec(source['ct'], ct_level=int(source.get('ct_level', 0)),
        ct_grid_scale=float(source.get('ct_grid_scale', 1.)), grid_scale=float(source['grid_scale']), cache_dir=cache_dir)


def primary_source_spec(document):
    """The Paris 4 CT volume spec the trainer builds from a dataset config."""
    source = next(s for s in document['sources'] if s['kind'] == 'paris4')
    return FiberVolumeSpec(source['ct'], ct_level=source.get('ct_level', 0), ct_grid_scale=source.get('ct_grid_scale', 4.),
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
        return str((base / value).resolve())
    document['cache_dir'] = resolve(document['cache_dir'])
    names = []
    for source in document['sources']:
        names.append(source['name'])
        weight = float(source['weight'])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Every dataset weight must be finite and positive')
        source['weight'] = weight
        from vesuvius.neural_tracing.fiber_follow.data.dataset_split import validate_split_policy
        validate_split_policy(source['validation'])  # Validate policy before dataset I/O.
        if source['kind'] == 'paris4':
            for key in PRIMARY_OPTIONS:
                if key in source and key != 'val_z':
                    source[key] = resolve(source[key])
        elif source['kind'] == 'afv':
            source['path'] = resolve(source['path'])
            source['ct'] = resolve(source['ct'])
            if not source.get('sha256') or not source.get('coordinate_space'):
                raise ValueError('AFV sources require a SHA256 and coordinate identity')
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
    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import make_seeds
    from vesuvius.neural_tracing.fiber_follow.shared.experiment import jsonable
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import SEED_HEADING_POLICY, TRACE_HEADING_POLICY, FRAME_POLICY
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
    seed_volume = FiberVolume(spec, cache_bytes=256 << 20)
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
        # monitor-sized seed subsets for calibration/final on large AFV catalogs.
        for fi in members[:monitor_count]:
            candidates = make_seeds([fibers[fi]],seed_volume,per_fiber=1,seed=seed+fi)
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
    from vesuvius.neural_tracing.fiber_follow.data.dataset_split import heldout_ids
    from vesuvius.neural_tracing.fiber_follow.shared.experiment import read_manifest
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


def open_afv_source(source, cache_dir, normalization=None):
    """Checked AFV training/validation fibers and CT spec for one dataset-config source.

    Verifies the file checksum, coordinate space and that the configured CT is the recorded CT store.
    Returns (training fibers, validation fibers, CT spec, sha256).
    """
    path = Path(source['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != source['sha256']:
        raise ValueError(f'AFV checksum changed: {path}')
    scale = float(source['grid_scale'])
    fibers = AFVFibers(path, scale, validation=source['validation'],sha256=digest.hexdigest())
    validation_fibers = AFVFibers(path,scale,validation=source['validation'],split='validation',sha256=digest.hexdigest())
    if fibers.metadata['frame']['vc_open_data_coordinate_space'] != source['coordinate_space']:
        raise ValueError('AFV coordinate identity mismatch')
    root = fibers.metadata['root']
    native_url = root.get('vc_open_data_source_path', '').rstrip('/')
    # The configured source must refer to the same recorded CT store.
    def canonical(url):
        return str(url).rstrip('/').replace('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/',
            's3://vesuvius-challenge-open-data/').replace('https://vesuvius-challenge-open-data.s3.amazonaws.com/',
            's3://vesuvius-challenge-open-data/')
    # A local copy of the CT (same coordinate frame, read at ct_grid_scale) names the store it mirrors.
    if canonical(source.get('ct_mirror_of', source['ct'])) != canonical(native_url):
        raise ValueError('AFV source CT differs from its embedded metadata')
    spec = ct_source_spec(source, cache_dir)
    if normalization is not None:
        from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
        spec.ct_normalization = normalization['volumes'][volume_key(spec)]
    return fibers, validation_fibers, spec, digest.hexdigest()


def build_mixed_dataset(primary, document, cfg, sample, sampling, args, *, seed, budget, out=None, resume=False,
                        normalization=None):
    datasets, names, weights, provenance = [], [], [], []
    for index, source in enumerate(document['sources']):
        if source['kind'] == 'paris4':
            dataset = primary
            entry = dict(name=source['name'], kind='paris4', fibers=len(primary.fibers))
        else:
            fibers, validation_fibers, spec, sha256 = open_afv_source(source, document['cache_dir'], normalization)
            scale = float(source['grid_scale'])
            builder = IdentityObservationBuilder(cfg, fibers, sampling, augment=True, neighbors=AFVBank(fibers))
            replay_index = Path(out)/'dagger'/source['name']/'replay.json' if out else None
            from vesuvius.neural_tracing.fiber_follow.data.data import load_replay, usable_replay
            replay_paths = json.loads(replay_index.read_text()) if resume and replay_index and replay_index.exists() else []
            replay = usable_replay(load_replay(replay_paths), fibers, sample.n_history, spec.grid_scale)
            dataset = FollowDataset(fibers, spec, sample, chunk=args.batch,
                seed=seed+100003*(index+1), cache_bytes=int(args.worker_cache_gb*(1<<30)),
                batch_builder=builder, budget=budget, length_power=args.afv_length_power,
                onpolicy=replay, replay_index=str(replay_index) if replay_index else None)
            dataset.validation_fibers = validation_fibers
            dataset.validation_manifest = validation_manifest(validation_fibers,spec,source['validation']['seed'])
            entry = dict(name=source['name'], kind='afv', fibers=len(fibers),
                excluded_fibers=fibers.metadata['fiber_count']-len(fibers),
                sha256=sha256, coordinate_space=source['coordinate_space'],
                validation=source['validation'], grid_scale=scale)
        datasets.append(dataset)
        names.append(source['name'])
        weights.append(source['weight'])
        provenance.append(entry)
    return WeightedDatasets(datasets, names, weights, seed), provenance
