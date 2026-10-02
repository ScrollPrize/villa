"""Training states for the crop-heading model, built with the follower's own data code.

A share of states are seeds (an annotated point with no observed path, the follower's seed-only state); the rest
are simulated tracer decisions from ``shared.data.simulated_trace`` (``make_sample``'s observed path, without its
labels) with the follower's own startup mix and tracing error, on the same Paris 4 / AFV sources, fiber splits, CT volumes and CT normalization (``regression.datasets``).
The prior heading is what the tracer holds: its trailing 12-voxel fit (``linear12_heading``) once 12 voxels exist.
Seeds and shorter paths hold a seed heading the model must not rely on, so their prior is the true continuation
tilted within a wide cone (also used on a share of long paths). The target is ``targets.in_crop_heading`` over the
follower crop's forward extent.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.model import (
    HeadingConfig, ct_shift, model_inputs, patch_volume_spec, prior_frames)
from vesuvius.neural_tracing.fiber_follow.heading_model.targets import in_crop_heading
from vesuvius.neural_tracing.fiber_follow.regression.datasets import (
    WeightedDatasets, load_primary_dataset, open_afv_source, primary_source_spec, read_dataset_config)
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, SampleConfig, simulated_trace, tight_block, traversal_curve
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, normalize
from vesuvius.neural_tracing.fiber_follow.shared.heading import linear12_heading
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume

HISTORY_BINS = (('seed', 0., 1e-9), ('1-11', 1e-9, 12.), ('12-31', 12., 32.), ('32+', 32., float('inf')))


@dataclass
class HeadingSampling:
    seed_probability: float = .3  # seed decisions: annotated point, no observed path
    cone_deg: float = 25.  # |N(0, cone_deg)| tilt of the true continuation for held-seed priors
    cone_cap_deg: float = 75.
    long_cone_probability: float = .1  # share of long paths also given a cone prior (robustness)
    sample_config: dict = field(default_factory=dict)  # overrides of the follower's SampleConfig defaults

    def __post_init__(self):
        if not all(0 <= v <= 1 for v in (self.seed_probability, self.long_cone_probability)):
            raise ValueError('Heading sampling probabilities must lie in [0, 1]')
        if not (self.cone_deg >= 0 and 0 <= self.cone_cap_deg <= 90):
            raise ValueError('Cone angles must be nonnegative, capped at 90 degrees')

    def follower_sample_config(self):
        # Tracing error model, startup mix and history exactly as the follower trains, unless overridden.
        return SampleConfig(**self.sample_config)


@dataclass
class Source:
    name: str
    kind: str
    weight: float
    spec: object
    train: object
    validation: object


def load_sources(dataset_config, out, *, ct_normalization=None, ct_downsample_levels=0):
    """Paris 4 and AFV sources from a follower dataset config, CT normalization bound to each volume.

    Each source's ``spec`` is the CT the heading patches read: the follower's volume, or ``ct_downsample_levels``
    coarser pyramid levels of it (``model.patch_volume_spec``). ``ct_normalization`` (a follower run's
    ct_normalization.json) reuses that run's exact records; a coarser level gets its own per-crop z-score record.
    """
    from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import prepare_normalization
    document, digest = read_dataset_config(dataset_config)
    known = json.loads(Path(ct_normalization).read_text()) if ct_normalization else None
    sources = []
    for entry in document['sources']:
        if entry['kind'] == 'paris4':
            spec = primary_source_spec(document)
            _, train, validation, _ = load_primary_dataset(document, spec)
        else:
            train, validation, spec, _ = open_afv_source(entry, document['cache_dir'])
        sources.append(Source(entry['name'], entry['kind'], float(entry.get('weight', 1.)),
                              patch_volume_spec(spec, ct_downsample_levels), train, validation))
    Path(out).mkdir(parents=True, exist_ok=True)
    normalization = prepare_normalization(out, [s.spec for s in sources], known=known)
    return document, digest, sources, normalization


def fiber_weights(fibers, minimum):
    lengths = np.asarray(fibers.lengths if hasattr(fibers, 'lengths') else [f.length for f in fibers], np.float64)
    weights = np.where(lengths >= minimum, lengths, 0.)  # uniform over usable annotated arclength
    if not weights.sum():
        raise ValueError('No fibers long enough for the heading target span')
    return weights/weights.sum()


def cone(direction, rng, scale, cap):
    """The direction tilted by |N(0, scale)| degrees (capped) in a random azimuth."""
    angle = np.radians(min(abs(rng.normal(0., scale)), cap))
    axis = normalize(np.cross(direction, rng.normal(size=3)))
    return normalize(np.cos(angle)*direction+np.sin(angle)*axis)


def heading_state(fiber, rng, cfg: HeadingConfig, sampling: HeadingSampling, sample_cfg):
    """One tracer decision (head, observed path, prior heading) and the annotated fiber ahead of it."""
    reverse = bool(rng.integers(2))
    p, s = traversal_curve(fiber, reverse)
    if s[-1] < cfg.forward+8:
        return None
    t = float(rng.uniform(0., s[-1]-cfg.forward-2))
    if rng.random() < sampling.seed_probability:
        pos = interp_at(p, s, np.array([t]))[0]
        path = pos[None]
    else:
        # make_sample's observed path and head, with the same draws, without the follower's labels.
        path = simulated_trace(p, s, t, sample_cfg, rng)[0]
        pos = path[-1]
    future = interp_at(p, s, t+np.arange(1., cfg.forward+1e-9))-pos
    init = normalize(future[-1])
    fit = linear12_heading(path, 0)  # the tracer's heading; None while it still holds the seed heading
    if fit is None or rng.random() < sampling.long_cone_probability:
        prior = cone(init, rng, sampling.cone_deg, sampling.cone_cap_deg)
    else:
        prior = fit
    return dict(pos=pos, path=path, prior=prior, future=future, init=init,
                history=float(arclength(path)[-1]) if len(path) > 1 else 0.)


def plan_states(fibers, weights, n, rng, cfg, sampling, *, roll_rng=None):
    """n states (geometry only, no CT) with their patch frames; random patch roll when ``roll_rng`` is given."""
    sample_cfg = sampling.follower_sample_config()
    # rng.choice(len(weights), p=weights) without rebuilding its CDF on every draw (same draws).
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    states = []
    while len(states) < n:
        state = heading_state(fibers[int(cdf.searchsorted(rng.random(), side='right'))], rng, cfg, sampling, sample_cfg)
        if state is not None:
            states.append(state)
    for state, frame in zip(states, prior_frames([s['prior'] for s in states], roll_rng)):
        state['frame'] = frame
    return states


def finish_states(states, vol, cfg, pool=None):
    """Targets, CT patches and path features for planned states; targets in each patch frame."""
    targets = in_crop_heading(np.stack([s['future'] for s in states]), np.stack([s['init'] for s in states]))
    frames = [s['frame'] for s in states]
    patch, path = model_inputs(vol, cfg, [s['pos'] for s in states], frames, [s['path'] for s in states], pool)
    for state, target in zip(states, targets):
        state['target'] = target
    local = torch.from_numpy(np.stack([t @ f for t, f in zip(targets, frames)]).astype(np.float32))
    return patch, path, local


def build_states(fibers, weights, vol, n, rng, cfg, sampling, *, roll_rng=None, pool=None):
    """Planned and finished states in one call (held-out sets)."""
    states = plan_states(fibers, weights, n, rng, cfg, sampling, roll_rng=roll_rng)
    return (states, *finish_states(states, vol, cfg, pool))


class HeadingBatchBuilder:
    """FollowDataset batch builder: exact patch footprints for prefetch, then targets and model inputs."""
    def __init__(self, cfg: HeadingConfig):
        self.cfg = cfg

    def prefetch_bounds(self, item, vol):
        yield tight_block(np.asarray(item['pos'])-ct_shift(self.cfg, vol), item['frame'], self.cfg.patch, vol.input_scale)

    def __call__(self, items, vol):
        patch, path, target = finish_states(items, vol, self.cfg)
        return dict(patch=patch, path=path, target=target,
                    history=torch.tensor([s['history'] for s in items], dtype=torch.float32))


class HeadingStates(FollowDataset):
    """One source's endless heading batches through the follower's loader pipeline.

    Only the plans (``_iter_plans``) and batch builder differ from the follower: cache-only volumes,
    remote prefetch lookahead, ``ensure`` before reads and the length-weighted fiber draws are FollowDataset's.
    """
    def __init__(self, source: Source, cfg: HeadingConfig, sampling: HeadingSampling, batch, seed=0,
                 cache_bytes=512 << 20):
        super().__init__(source.train, source.spec, sampling.follower_sample_config(), None, chunk=batch, seed=seed,
                         cache_bytes=cache_bytes, batch_builder=HeadingBatchBuilder(cfg))
        self.heading_cfg, self.sampling = cfg, sampling
        # Length-weighted like the follower, restricted to fibers long enough for the target span.
        self.weights = fiber_weights(source.train, cfg.forward+8)

    def _iter_plans(self, vol, windows):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0 if info is None else info.id]))
        while True:
            states = plan_states(self.fibers, self.weights, self.chunk, rng, self.heading_cfg, self.sampling, roll_rng=rng)
            self.prefetch_items(states, vol)
            yield states


def mixed_heading_states(sources, cfg, sampling, batch, seed=0, cache_bytes=512 << 20):
    """Per-source datasets mixed by dataset-config weight with the follower's WeightedDatasets."""
    datasets = [HeadingStates(s, cfg, sampling, batch, seed=seed+100003*(k+1), cache_bytes=cache_bytes)
                for k, s in enumerate(sources)]
    return WeightedDatasets(datasets, [s.name for s in sources], [s.weight for s in sources], seed=seed), datasets


def validation_states(sources, cfg, sampling, n, seed, pool=None):
    """Fixed held-out states per source, with the deterministic inference roll."""
    out = {}
    for k, source in enumerate(sources):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7349, k]))
        vol = FiberVolume(source.spec)
        out[source.name] = build_states(source.validation, fiber_weights(source.validation, cfg.forward+8), vol, n,
                                        rng, cfg, sampling, pool=pool)
    return out


def sampling_from_dict(value):
    return HeadingSampling(**(value or {}))


def sampling_to_dict(sampling):
    return asdict(sampling)
