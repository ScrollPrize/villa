"""One JSON config with sections ``data``, ``model``, ``trace``, ``loss``, ``train``, ``eval``.

Every field has a default here, so a config file only lists what it changes. Command-line
``--set section.field=value`` overrides parse ``value`` as JSON when possible (``--set model.dim=128``,
``--set model.crop=[24,16,24]``), otherwise as a string. Distances are level-0 voxels throughout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from vesuvius.neural_tracing.tracers.shared.paths import parse_path_map, path_map, recorded_path

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE/'output'


@dataclass
class DataConfig:
    patches_dir: str | list = '/home/sean/Desktop/spiral_dataset/verified_patches'  # one directory or a list
    volume: str = '/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr'
    volume_level: int = 0  # pyramid level read for crops; positions stay level-0
    voxel_size_um: float = 9.6  # level-0 voxel size of the volume (s1_ds2: 9.6 um), for area_cm2 in written tifxyz
    front_points: int = 5  # number of points on the front (validation; training: front_points_range)
    point_spacing: float = 60.  # seed distance along the front between neighbouring points
    step: float = 20.  # distance advanced per tracing step
    steps: int = 12  # rollout steps per training episode
    history: int = 1  # GT rows behind the seed row given as the initial direction
    gt_resolution: float = 5.  # spacing of the canonical GT grid each episode resamples its patch onto
    gt_margin: float = 40.  # GT grid extent beyond the region the front can reach
    roi: int = 640  # largest extent (plus view margin) of an episode's block: its reachable GT region
    min_valid_fraction: float = .9  # of the supervised GT region
    patch_weight_power: float = .5  # patch sampling weight = valid_points ** power
    split_block: int = 1024  # spatial train/val split cell
    val_fraction: float = .1  # fraction of split cells held out
    split_seed: int = 0
    seed_noise_normal: float = 1.5  # Gaussian noise on seed rows, along the sheet normal
    seed_noise_tangent: float = 3.  # ... and within the sheet
    worker_cache_mb: int = 256  # per-process chunk cache of the volume reader
    max_tries: int = 200  # patches drawn per episode before giving up
    # Hard spots (``evaluation/hard_spots.py``): this fraction of training episodes starts at a recorded failure spot of a
    # model (JSON from evaluation/hard_spots.py; patch directories by their position in patches_dir), jittered by up to two
    # steps along the advance and half a point spacing along the front. Validation episodes are unaffected.
    hard_windows: str = ''
    hard_fraction: float = 0.
    anchor_tries: int = 8  # orientation/anchor attempts on a drawn patch before drawing another
    # Per patch directory (same order as patches_dir): the CT volume its patches' coordinates are in, e.g. segments
    # of other scrolls (``scripts/prepare_scroll.py``); an empty entry (or an empty list) means data.volume. The
    # volumes share data.volume_level.
    patch_volumes: list = field(default_factory=list)
    # Per patch directory: its share of the sampling weight (normalized; within a directory patches keep their
    # size weights). Empty: every patch by size alone.
    dir_weights: list = field(default_factory=list)
    # Training only: each batch's front has N points, N drawn uniformly from [min, max] (data.front_points still sets
    # validation). Batches hold about train.points_per_batch points (batch = that / N, rounded).
    front_points_range: list = field(default_factory=list)
    # Training batches keep their blocks unpadded on the GPU (a list; crops sample each episode's own block) instead
    # of padding them to the batch's largest extent per axis, which long fronts at mixed orientations blow up.
    unpadded_blocks: bool = False
    # Training only: this fraction of episodes starts from a fiber matched to its patch (JSON lines of
    # scripts/match_fibers_to_patches.py: runs with 3D samples ``xyz`` and grid positions ``rc``) instead of a GT
    # grid row: the fiber is the seed row, the GT window is rotated to follow it (``EpisodeSampler._fiber_attempt``).
    fiber_runs: str = ''
    fiber_fraction: float = 0.
    fiber_min_valid: float = .5  # fiber episodes' min_valid_fraction: matched runs often lie near patch edges
    # Fiber-following batches (training): this fraction of batches has a 3-point front across a matched fiber,
    # middle point on it, advancing along it (``EpisodeSampler._follow_attempt``), with the model's mode input 1
    # (model.fiber_follow_mode). ``fiber_follow`` marks such a batch's sampler config; set by the trainer.
    follow_fraction: float = 0.
    fiber_follow: bool = False
    # Fiber inputs for fiber-started episodes (needs model.fiber_tokens): every fiber around the episode from this
    # pool (scripts/build_fiber_pool.py), the seed fiber as instance 0. In fiber-following episodes the seed fiber is
    # cut ahead of the start (exactly at it with probability fiber_cut_at_start, else anywhere up to the rollout's
    # end), so the model learns to extend a fiber past its known end.
    fiber_pool: str = ''
    fiber_cut_at_start: float = .5
    fiber_max_instances: int = 32
    fiber_max_points: int = 4096


@dataclass
class ModelConfig:
    crop: list = field(default_factory=lambda: [24, 48, 32])  # point crop samples along (normal, front tangent, advance)
    crop_spacing: float = 1.  # sample spacing along every axis
    crop_forward: float = 10.  # crop centre ahead of the point along the advance direction
    # Point CNN: ResNet-D basic blocks of these widths and strides (each crop size divisible by 2 per stride-2 stage),
    # then a 1x1 conv to point_bottleneck channels before the flattening Linear.
    channels: list = field(default_factory=lambda: [32, 64, 128])
    strides: list = field(default_factory=lambda: [1, 2, 2])
    point_bottleneck: int = 32
    dim: int = 256
    layers: int = 8  # transformer blocks over the point, grid (and fiber) tokens together
    heads: int = 8
    mlp_ratio: float = 8.
    memory_rows: int = 3  # past grid rows kept besides the current one
    dropout: float = 0.
    # Grid tokens: small CT crops along the curve (columns, trace.spline_samples per gap) for the current row and
    # the last memory_rows rows, through their own residual CNN.
    token_crop: list = field(default_factory=lambda: [16, 16, 16])  # samples along (normal, curve tangent, advance)
    token_spacing: float = 1.
    token_channels: list = field(default_factory=lambda: [32, 64, 96])
    token_strides: list = field(default_factory=lambda: [1, 2, 2])
    # A learned mode embedding (zero at start) added to every token: 0 sheet tracing, 1 following a fiber with a
    # 3-point front across it (data.follow_fraction). Starts as the checkpoint it is initialised from.
    fiber_follow_mode: bool = False
    # Fiber tokens: the fiber points nearest the front each step (data.fiber_pool), as extra tokens of the
    # transformer: position and direction in the front frame, family, seed-fiber flag and the fiber's instance code
    # (random per episode, the same for all points of one fiber, different between fibers). 0: none.
    fiber_tokens: int = 0


@dataclass
class TraceConfig:
    """The front points are the control points of a cubic Hermite curve: confident points move to their predictions
    and tilt their tangents; stopped points take the displacement interpolated from their confident neighbours and
    are put on the curve through them, so they can land back on the sheet and rejoin."""
    spline_samples: int = 6  # dense curve samples per point gap (curve loss, arc-length respacing, output columns)
    confidence_threshold: float = .5  # below: the point is stopped (masked) and rides the curve


@dataclass
class LossConfig:
    normal: float = 1.  # prediction error along the GT sheet normal
    advance: float = .3  # ... along the GT advance direction
    # ... along the front, against the point's seed column (v = k * point_spacing) rather than its current position,
    # so cumulative shrinking and sideways drift of the front are penalised, not just each step's sliding.
    tangent: float = .1
    spline: float = .5  # dense spline samples' distance to the sheet
    # Class-balanced: on- and off-sheet labels weigh equally (off-sheet points are rare, so an unbalanced BCE learns
    # to keep all).
    confidence: float = .5
    # Front length: Huber on (arc length of the predicted curve - seed length) per gap, in voxels, so the
    # fixed set of front points neither stretches nor shrinks along the sheet. Needs no GT. 0 disables.
    front_length: float = 0.
    huber: float = 2.  # Huber transition (voxels)
    # Same-sheet test, about half the sheet spacing. Confidence predicts whether the candidate point lies on
    # the seed's sheet; points farther than this from the GT sheet get no position loss, only confidence 0.
    sheet_tolerance: float = 6.
    good_distance: float = 3.  # on-sheet distance counted as accurate (metrics only)


@dataclass
class TrainConfig:
    name: str = 'sheet_tracer'
    iterations: int = 3000
    batch: int = 16
    workers: int = 6
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    grad_clip: float = 1.
    amp: str = 'bf16'  # 'bf16' or 'none'
    state_noise_normal: float = .75  # noise added to the front between rollout steps
    state_noise_tangent: float = 1.5
    reset_distance: float = 8.  # in snapped episodes, points farther than this are snapped back onto the sheet
    free_fraction: float = .5  # episodes never snapped back, so points can drift onto other sheets
    jump_probability: float = .3  # episodes in which a run of points is pushed onto a neighbouring sheet
    jump_min: float = 7.  # push along the sheet normal, either sign, onto the brightest CT ridge in this range
    jump_max: float = 24.
    prefetch: int = 2  # loader batches in flight per worker (each holds batch x block bytes of shared memory)
    compile_geometry: bool = False  # torch.compile the GT projection: ~1.4x faster, not bit-identical (geometry.py)
    compile_model: bool = False  # torch.compile the model (model.compile_model): ~1.7x faster at level 0, not bit-identical
    points_per_batch: int = 0  # with data.front_points_range: front points per batch (0: train.batch * data.front_points)
    val_episodes: int = 64
    log_every: int = 50
    val_every: int = 500
    viz_every: int = 500
    ckpt_every: int = 1000  # last.pt
    keep_every: int = 0  # also keep ckpt_ITER.pt every this many iterations (0: off)
    max_minutes: float = 0.  # 0: no time budget; otherwise training stops and the cosine lr reaches zero there
    # Before each training rollout, trace up to this many GT rows behind the seed (uniform per batch, without
    # loss) with the front put back on GT after every step, so the grid memory starts out filled as it is deep
    # into a real trace. Training episodes then start that many rows further back; validation episodes do not
    # change.
    history_rows: int = 0
    # Exponential moving average of the weights (decay per iteration, e.g. 0.999; 0: off), warmed up as
    # min(ema, (1 + n) / (10 + n)) over the first updates. Validation, best.pt and evaluation use the averaged
    # weights; checkpoints keep both (``model`` raw, ``ema`` averaged).
    ema: float = 0.
    # Live chains (like the fiber tracer's live continuation): a front is not restarted from GT after its rollout but
    # continues in a later iteration from exactly where the model left it (positions, curve, confidences, grid
    # memory) on the next GT window and block of its patch, which a loader worker reads on request (train/live.py).
    # A chain ends after a random number of steps in chain_steps, when under chain_on_sheet of its points end a
    # rollout on the seed's sheet, when a row has no confident point connected to the surface traced so far (the
    # inference stop), or when its patch has no next window (chain_min_valid of it valid, in the split).
    live_chains: bool = False
    chain_steps: list = field(default_factory=lambda: [24, 72])
    chain_on_sheet: float = .5
    chain_min_valid: float = .5
    chain_max_age: int = 200  # iterations a chain may wait for its next window before it is dropped
    seed: int = 0
    device: str = 'cuda'


@dataclass
class EvalConfig:
    """Free rollouts after training (and the evaluate CLI default). Keep it short between quick runs."""
    episodes: int = 32
    steps: int = 10
    roi: int = 512
    after_training: bool = True


SECTIONS = dict(data=DataConfig, model=ModelConfig, trace=TraceConfig, loss=LossConfig,
                train=TrainConfig, eval=EvalConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return OUTPUT/self.train.name



# Options this version no longer has. A loaded config (e.g. from a checkpoint) may still list them, at the value the
# supported models had; another value means a model this code cannot run. None: any value (options that only shaped
# training: rollout steps are always detached and backpropagated one by one, the confidence loss is class-balanced
# and the along-front target is the seed column).
REMOVED = dict(
    data=dict(extra_volumes=[[]], extra_levels=[[]], extra_level_dirs=[[]], tight_blocks=[True]),
    model=dict(attention=[True], surface_tokens=[True], joint_transformer=[True], residual=[True], memory_lags=[[]],
               memory_anchor=[False], context_crop=[[]], profile_crop=[[]], grid_layers=None, context_spacing=None,
               context_channels=None, context_strides=None, profile_spacing=None, profile_channels=None,
               profile_strides=None),
    trace=dict(curve=['hermite'], stopped=['curve'], respace=[True], spline_ctrl=None, spline_smooth=None,
               min_weight=None),
    loss=dict(column_target=None, confidence_balance=None),
    train=dict(detach_steps=None, backward_per_step=None),
)


def from_dict(document: dict, strict: bool = True) -> Config:
    """Options of removed model types (``REMOVED``) are dropped if they hold the supported value, rejected
    otherwise. ``strict=False`` (loading older checkpoints) also drops other unknown fields, with a note."""
    unknown = set(document)-set(SECTIONS)
    if unknown:
        raise ValueError(f'Unknown config sections: {sorted(unknown)}')
    parts = {}
    for name, cls in SECTIONS.items():
        values = dict(document.get(name, {}))
        for key, allowed in REMOVED.get(name, {}).items():
            if key in values:
                value = values.pop(key)
                if allowed is not None and value not in allowed:
                    raise ValueError(f'{name}.{key}={value!r}: this model type is no longer supported')
        names = {f.name for f in fields(cls)}
        bad = set(values)-names
        if bad and strict:
            raise ValueError(f'Unknown {name} fields: {sorted(bad)}')
        if bad:
            print(f'config: ignoring obsolete {name} fields {sorted(bad)}', flush=True)
            values = {k: v for k, v in values.items() if k in names}
        for f in fields(cls):  # '--set loss.front_length=1.' arrives as the string '1.' (not valid JSON)
            if f.name in values and isinstance(values[f.name], str) and isinstance(f.default, (int, float)) \
                    and not isinstance(f.default, bool):
                try:
                    values[f.name] = type(f.default)(values[f.name])
                except ValueError as exc:
                    raise ValueError(f'{name}.{f.name} must be a number, not {values[f.name]!r}') from exc
        parts[name] = cls(**values)
    config = Config(**parts)
    validate(config)
    return config


def parse_override(text: str) -> tuple[str, str, object]:
    key, sep, raw = text.partition('=')
    section, dot, name = key.partition('.')
    if not sep or not dot:
        raise ValueError(f'Override must look like section.field=value, not {text!r}')
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return section, name, value


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Defaults, then the JSON file (relative paths: cwd, then the package's configs/), then overrides."""
    document = {}
    if path:
        path = Path(path)
        if not path.is_file() and (PACKAGE/'configs'/path).is_file():
            path = PACKAGE/'configs'/path
        document = json.loads(path.read_text())
    for text in overrides or []:
        section, name, value = parse_override(text)
        document.setdefault(section, {})[name] = value
    return from_dict(document)


def validate(config: Config) -> None:
    d, m = config.data, config.model
    if d.follow_fraction and (not 0 < d.follow_fraction < 1 or not d.fiber_runs or not d.front_points_range
                              or not m.fiber_follow_mode):
        raise ValueError('data.follow_fraction needs data.fiber_runs, data.front_points_range and model.fiber_follow_mode')
    if not 0 <= d.fiber_fraction <= 1 or (d.fiber_fraction and not d.fiber_runs):
        raise ValueError('data.fiber_fraction must be in [0, 1] and needs data.fiber_runs')
    if not 0 <= d.hard_fraction <= 1 or (d.hard_fraction and not d.hard_windows):
        raise ValueError('data.hard_fraction must be in [0, 1] and needs data.hard_windows')
    dirs = len(d.patches_dir) if isinstance(d.patches_dir, (list, tuple)) else 1
    if d.patch_volumes and len(d.patch_volumes) != dirs:
        raise ValueError('data.patch_volumes needs one entry per patch directory')
    if d.dir_weights and (len(d.dir_weights) != dirs or min(d.dir_weights) < 0 or not sum(d.dir_weights)):
        raise ValueError('data.dir_weights needs one non-negative weight per patch directory, not all zero')
    if d.front_points_range and (len(d.front_points_range) != 2 or not 3 <= int(d.front_points_range[0])
                                 <= int(d.front_points_range[1]) or config.train.live_chains):
        raise ValueError('data.front_points_range needs [min, max] with 3 <= min <= max, and no train.live_chains')
    if d.front_points < 3:
        raise ValueError('data.front_points must be at least 3 (the Hermite curve)')
    if len(m.crop) != 3 or any(int(c) < 2 for c in m.crop):
        raise ValueError('model.crop needs three sizes >= 2')
    if not all(isinstance(s, (int, float)) for s in (m.crop_spacing, m.token_spacing)):
        raise ValueError('model.crop_spacing and model.token_spacing are one number (per-axis spacing was removed)')
    if d.volume_level < 0 or d.roi % (1 << d.volume_level):
        raise ValueError('data.roi must be divisible by 2**data.volume_level')
    if config.train.amp not in ('bf16', 'none'):
        raise ValueError("train.amp must be 'bf16' or 'none'")
    for name, crop, channels, strides in (('', m.crop, m.channels, m.strides),
                                          ('token_', m.token_crop, m.token_channels, m.token_strides)):
        if len(strides) != len(channels) or any(int(s) not in (1, 2) for s in strides):
            raise ValueError(f'model.{name}strides needs one stride (1 or 2) per model.{name}channels entry')
        factor = 2**sum(int(s) == 2 for s in strides)
        if any(int(c) % factor for c in crop):
            raise ValueError(f'model.{name}crop sizes must be divisible by {factor} (residual stages)')
        if any(int(c) % 4 for c in channels):
            raise ValueError(f'model.{name}channels must be divisible by 4 (GroupNorm(4))')
    tc = config.train
    if not 0 <= tc.ema < 1:
        raise ValueError('train.ema must be in [0, 1)')
    if tc.history_rows < 0:
        raise ValueError('train.history_rows must be >= 0')
    if tc.live_chains and (tc.history_rows or len(tc.chain_steps) != 2 or m.memory_rows > d.steps
                           or not d.steps <= int(tc.chain_steps[0]) <= int(tc.chain_steps[1])):
        # Continued chains of different depths share batches, so every continued front must hold the same memory:
        # all of it, once a rollout of data.steps has filled it.
        raise ValueError('train.live_chains needs no train.history_rows, model.memory_rows <= data.steps, and '
                         'train.chain_steps [min, max] with data.steps <= min <= max')
    if m.dim % m.heads:
        raise ValueError('model.dim must be divisible by model.heads')


def path_mapper(texts):
    """``--path-map OLD=NEW`` texts (none: the ``TRACERS_PATH_MAP`` environment, ``shared/paths.py``) -> a function
    relocating every data path of a config in place."""
    mapping = parse_path_map(texts) if texts else path_map()
    remap = lambda path: recorded_path(path, mapping) if path and '://' not in str(path) else path

    def apply(cfg):
        d = cfg.data
        d.volume = remap(d.volume)
        d.patch_volumes = [remap(p) for p in d.patch_volumes]
        d.patches_dir = [remap(p) for p in d.patches_dir] if isinstance(d.patches_dir, list) else remap(d.patches_dir)
        d.hard_windows, d.fiber_runs, d.fiber_pool = remap(d.hard_windows), remap(d.fiber_runs), remap(d.fiber_pool)
        return cfg
    return apply
