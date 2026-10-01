"""Explicit dataset provenance and weighted, source-local training batches."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..shared.afv import AFVFibers
from ..shared.data import FollowDataset, ZBand, fiber_manifest, load_fibers
from ..shared.volume import FiberVolumeSpec
from .data import IdentityObservationBuilder
from .neighbor_bank import NeighborBank


PRIMARY_OPTIONS = ('fibers', 'fiber_zarrs', 'ct', 'manifest', 'val_z', 'negative_bank',
                   'near_negative_bank', 'following_bank', 'continuation_bank')


def ct_source_spec(source, cache_dir):
    return FiberVolumeSpec(source.get('fiber_zarrs', ''), ct_zarr=source['ct'],
        ct_level=int(source.get('ct_level', 0)), ct_grid_scale=float(source.get('ct_grid_scale', 1.)),
        grid_scale=float(source['grid_scale']), inputs='ct', load_presence=False, cache_dir=cache_dir)


def read_dataset_config(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get('version') != 1 or not document.get('sources'):
        raise ValueError('Dataset config requires version 1 and nonempty sources')
    document = json.loads(json.dumps(document))
    def resolve(value):
        if value is None or '://' in str(value):
            return value
        return str((path.parent / value).resolve())
    document['cache_dir'] = resolve(document['cache_dir'])
    names = []
    for source in document['sources']:
        names.append(source['name'])
        weight = float(source['weight'])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Every dataset weight must be finite and positive')
        source['weight'] = weight
        from ..shared.dataset_split import validate_split_policy
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
    for key in ('fibers', 'fiber_zarrs', 'ct', 'manifest', 'negative_bank', 'val_z'):
        if key not in primary:
            raise ValueError(f'Paris 4 source is missing {key}')
    digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    return document, digest


def validate_dataset_resume(checkpoint, document, digest):
    """Allow cache relocation while requiring identical resolved training data."""
    if checkpoint.get('dataset_config_sha256') == digest:
        return
    recorded = checkpoint.get('dataset_config')
    if isinstance(recorded,dict) and isinstance(document,dict):
        without_cache = lambda value: {k:v for k,v in value.items() if k != 'cache_dir'}
        if without_cache(recorded) == without_cache(document):
            return
    raise ValueError('Resume dataset configuration changed')


def validation_manifest(fibers, spec, seed=0, monitor_count=32):
    """Freeze distinct monitor/calibration/final fiber IDs and CT-only seeds."""
    from ..shared.evaluate import make_seeds
    from ..shared.experiment import jsonable
    from ..shared.heading import SEED_HEADING_POLICY, TRACE_HEADING_POLICY, FRAME_POLICY
    from ..shared.volume import FiberVolume
    from dataclasses import replace
    # Seed geometry reads CT only, even when the model itself has extra channels.
    seed_volume = FiberVolume(replace(spec, inputs='ct', load_presence=False), cache_bytes=256 << 20)
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

    Existing Paris 4 neighbor shards only support the legacy training parents.
    Their old mining band remains provenance, not a training sampling filter.
    """
    from ..shared.dataset_split import heldout_ids
    from ..shared.experiment import read_manifest
    source = next(s for s in document['sources'] if s['kind']=='paris4')
    fibers = load_fibers(source['fibers'],grid_scale=spec.grid_scale)
    legacy = read_manifest(source['manifest'])
    reserved = {f['name'] for f in legacy['fibers']}
    extra = heldout_ids([f.name for f in fibers if f.name not in reserved],source['validation'])
    val = [f for f in fibers if f.name in reserved|extra]
    train = [f for f in fibers if f.name not in reserved|extra]
    manifest = validation_manifest(val,spec,source['validation']['seed'])
    return fibers,train,val,manifest


class HoldoutFilteredBank(NeighborBank):
    """Prevent mined neighbors from reintroducing a reserved Paris 4 fiber."""
    def __init__(self,*args,heldout=(),**kwargs):
        super().__init__(*args,**kwargs)
        from scipy.spatial import cKDTree
        self._heldout_tree = cKDTree(np.concatenate([f.points for f in heldout])) if heldout else None
        self._heldout_cache = OrderedDict()

    def allowed_path(self,points):
        if self._heldout_tree is None:
            return True
        key = hashlib.sha256(np.asarray(points).tobytes()).digest()
        if key not in self._heldout_cache:
            # Quarter-voxel neighbor samples and <=1 voxel heldout vertices:
            # a 2-voxel guard conservatively covers their interpolation gaps.
            self._heldout_cache[key] = bool((self._heldout_tree.query(points)[0] > 2.).all())
            if len(self._heldout_cache)>256:
                self._heldout_cache.popitem(last=False)
        return self._heldout_cache[key]

    def draw_path(self,*args,**kwargs):
        for _ in range(8):
            result = super().draw_path(*args,**kwargs)
            if result is not None and self.allowed_path(result[1]):
                return result
        return None

    def spatial_records(self,*args,**kwargs):
        return [r for r in super().spatial_records(*args,**kwargs) if self.allowed_path(r['samples'])]


def apply_primary_source(args, document):
    primary = next(s for s in document['sources'] if s['kind'] == 'paris4')
    for key in PRIMARY_OPTIONS:
        setattr(args, key, primary.get(key))


class AFVBank(NeighborBank):
    """Reuse existing foreign-mask clearance rules on AFV spatial queries.

    These are supervision-only paths, never image inputs. Nearby paths also
    supply existing matched decisions, neighbor following and memory switches.
    """
    def __init__(self, fibers):
        self.fibers = fibers
        self.exclusion = 1.5
        self._trees = OrderedDict()
        self.root = Path(fibers.path)
        self.run = {'mining': {'min_distance': self.exclusion}, 'digest': fibers.metadata['uuid']}
        self._cdf = np.cumsum(fibers.lengths)/fibers.lengths.sum()

    @property
    def shard_count(self):
        return 1

    def draw_path(self, rng, *, unique=True, min_length=0., hard_fraction=0.):
        """Draw a nearby, target-clear continuous path with arc correspondence.

        Parent locations are length weighted; neighbor IDs are sampled without
        duplicate RTree blocks. Whole-target clearance rejects overlapping
        duplicates. Unknown/short/nonmatching geometry is retried or skipped.
        """
        if not 0 <= hard_fraction <= 1 or min_length < 0:
            raise ValueError('Invalid AFV neighbor sampling options')
        from scipy.spatial import cKDTree
        from ..shared.geometry import arclength, interp_at
        from .neighbor_mining import exact_nearest
        from .bank_geometry import difficulty_scores, DIFFICULTY_KINDS
        proposals = []
        hard = rng.random() < hard_fraction
        for _ in range(16):
            fi = int(np.searchsorted(self._cdf, rng.random(), side='right'))
            parent = self.fibers[fi]
            at = float(rng.uniform(0., parent.length))
            anchor = interp_at(parent.points, parent.s, [at])[0]
            radius = 6. if rng.random() < .5 else 32.
            ids = self.fibers.nearby_fiber_ids(anchor, radius, self.fibers.catalog[fi][0])
            if not ids:
                continue
            tree, gap = self._target_tree(fi)
            for neighbor_id in rng.permutation(ids)[:8]:
                neighbor = self.fibers[self.fibers.id_to_index[int(neighbor_id)]]
                j = int(np.argmin(np.linalg.norm(neighbor.points-anchor, axis=1)))
                span = max(160., min_length+32.)
                a, b = max(0., neighbor.s[j]-span), min(neighbor.length, neighbor.s[j]+span)
                line = interp_at(neighbor.points, neighbor.s, np.arange(a,b+1e-9,.25))
                if len(line) < 2:
                    continue
                distance, nearest = tree.query(line)
                supported = ((distance-gap > self.exclusion) & (distance <= 32.)
                             & (np.abs(parent.s[nearest]-at) <= span))
                edges = np.diff(np.r_[False,supported,False].astype(np.int8))
                starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                choices = [(a,b) for a,b in zip(starts,stops) if (b-a-1)*.25 >= max(32.,min_length)]
                if not choices:
                    continue
                begin,end = choices[int(rng.integers(len(choices)))]
                curve = line[begin:end]
                if arclength(curve)[-1] < min_length:
                    continue
                # Match against the local parent window, then leave stricter
                # monotonicity/seed observability checks to the existing tasks.
                near = nearest[begin:end]
                lo,hi = max(0,int(near.min())-2),min(len(parent.s),int(near.max())+3)
                _,_,segment,u = exact_nearest(curve,parent.points[lo:hi])
                matched = parent.s[lo+segment]+u*np.diff(parent.s[lo:hi])[segment]
                if matched[-1] < matched[0]:
                    curve,matched = curve[::-1].copy(),matched[::-1]
                if np.any(np.diff(matched) < -1e-5):
                    continue
                proposals.append((fi,curve,(float(matched.min()),float(matched.max()))))
                break
            if proposals and (not hard or len(proposals) >= 4):
                break
        if not proposals:
            return None
        if not hard:
            return proposals[0]
        kind = int(rng.integers(len(DIFFICULTY_KINDS)))
        scores = [difficulty_scores(line,self.fibers[fi])[kind] for fi,line,_ in proposals]
        return proposals[int(np.argmax(scores))]

    def provenance(self):
        return dict(kind='afv', path=self.fibers.path, manifest=self.fibers.manifest_entries())

    def spatial_records(self, fi, world, radius=0.):
        bounds = np.stack((np.min(world, axis=0)-radius, np.max(world, axis=0)+radius))
        own = self.fibers.catalog[fi][0]
        return list(self.fibers.nearby_blocks(bounds, own, records=True))


class WeightedDatasets(torch.utils.data.IterableDataset):
    """Choose a source per equal-sized batch; preserve matched pairs.

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
            batch['dataset_id'] = torch.full((len(batch['hist']),), index, dtype=torch.int64)
            yield batch


def build_mixed_dataset(primary, document, cfg, sample, sampling, args, *, seed, out=None, resume=False,
                        normalization=None):
    if cfg.input_mode != 'ct' or cfg.direction_inputs:
        raise ValueError('The AFV sources have CT only; enable --input-mode ct --no-direction-inputs')
    datasets, names, weights, provenance = [], [], [], []
    for index, source in enumerate(document['sources']):
        if source['kind'] == 'paris4':
            dataset = primary
            entry = dict(name=source['name'], kind='paris4', fibers=len(primary.fibers))
        else:
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
            if canonical(source['ct']) != canonical(native_url):
                raise ValueError('AFV source CT differs from its embedded metadata')
            spec = ct_source_spec(source, document['cache_dir'])
            if normalization is not None:
                from ..shared.ct_normalization import volume_key
                spec.ct_normalization = normalization['volumes'][volume_key(spec)]
            local_sampling = sampling
            builder = IdentityObservationBuilder(cfg, fibers, local_sampling, augment=True,
                                                  negative_bank=AFVBank(fibers))
            band = None
            replay_index = Path(out)/'dagger'/source['name']/'replay.json' if out else None
            from ..shared.data import OnPolicyStates
            replay_paths = json.loads(replay_index.read_text()) if resume and replay_index and replay_index.exists() else []
            replay = [OnPolicyStates.load(p) for p in replay_paths]
            dataset = FollowDataset(fibers, spec, sample, band, chunk=args.batch,
                seed=seed+100003*(index+1), cache_bytes=int(args.worker_cache_gb*(1<<30)),
                batch_builder=builder, fresh_fraction=args.fresh_fraction,
                clean_fraction=getattr(args, 'clean_fraction', None),
                onpolicy=replay, replay_index=str(replay_index) if replay_index else None)
            dataset.validation_fibers = validation_fibers
            dataset.validation_manifest = validation_manifest(validation_fibers,spec,source['validation']['seed'])
            entry = dict(name=source['name'], kind='afv', fibers=len(fibers),
                excluded_fibers=fibers.metadata['fiber_count']-len(fibers),
                sha256=digest.hexdigest(), coordinate_space=source['coordinate_space'],
                validation=source['validation'], grid_scale=scale)
        datasets.append(dataset)
        names.append(source['name'])
        weights.append(source['weight'])
        provenance.append(entry)
    return WeightedDatasets(datasets, names, weights, seed), provenance
