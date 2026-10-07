"""One JSON config with sections ``data``, ``model``, ``trace``, ``loss``, ``train``, ``eval``.

Every field has a default here, so a config file only lists what it changes. Command-line
``--set section.field=value`` overrides parse ``value`` as JSON when possible (``--set model.dim=128``,
``--set model.crop=[24,16,24]``), otherwise as a string. Distances are level-0 voxels throughout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE/'output'


@dataclass
class DataConfig:
    patches_dir: str | list = '/home/sean/Desktop/spiral_dataset/verified_patches'  # one directory or a list
    volume: str = '/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr'
    volume_level: int = 1  # pyramid level read for crops; positions stay level-0
    voxel_size_um: float = 9.6  # level-0 voxel size of the volume (s1_ds2: 9.6 um), for area_cm2 in written tifxyz
    front_points: int = 12  # number of sampled points on the front
    point_spacing: float = 20.  # seed distance along the front between neighbouring points
    step: float = 20.  # distance advanced per tracing step
    steps: int = 6  # rollout steps per training episode
    history: int = 1  # GT rows behind the seed row given as the initial direction
    gt_resolution: float = 5.  # spacing of the canonical GT grid each episode resamples its patch onto
    gt_margin: float = 40.  # GT grid extent beyond the region the front can reach
    roi: int = 384  # edge of the cubic volume block loaded per episode
    min_valid_fraction: float = .9  # of the supervised GT region
    patch_weight_power: float = .5  # patch sampling weight = valid_points ** power
    split_block: int = 1024  # spatial train/val split cell
    val_fraction: float = .1  # fraction of split cells held out
    split_seed: int = 0
    seed_noise_normal: float = 1.5  # Gaussian noise on seed rows, along the sheet normal
    seed_noise_tangent: float = 3.  # ... and within the sheet
    worker_cache_mb: int = 256  # per-process chunk cache of the volume reader
    # Extra input channels: volumes on the CT's grid (same shape and pyramid, read at volume_level), e.g. a
    # surface-prediction volume. Crops are scaled to [0, 1] (the CT channel stays z-scored per crop).
    extra_volumes: list = field(default_factory=list)
    # Pyramid level of each extra volume (empty: the CT's volume_level). A coarser level is read for the same
    # region and upsampled by repetition to the CT grid, e.g. a level-1 surface prediction with level-0 CT.
    extra_levels: list = field(default_factory=list)
    # Directory of each extra volume's array inside its zarr, when it is not named by that level (e.g. a pyramid
    # whose "3" is on the CT's level-1 grid: extra_levels [1], extra_level_dirs ["3"]). Empty: str(level).
    extra_level_dirs: list = field(default_factory=list)
    # Read only the reachable GT region plus the view margin per episode instead of the whole roi cube (batches
    # are zero-padded to their largest block). Much less data per batch, especially at volume level 0.
    tight_blocks: bool = False
    max_tries: int = 200  # patches drawn per episode before giving up
    # Hard spots (``hard_spots.py``): this fraction of training episodes starts at a recorded failure spot of a
    # model (JSON from hard_spots.py; patch directories by their position in patches_dir), jittered by up to two
    # steps along the advance and half a point spacing along the front. Validation episodes are unaffected.
    hard_windows: str = ''
    hard_fraction: float = 0.
    anchor_tries: int = 8  # orientation/anchor attempts on a drawn patch before drawing another
    # Per patch directory (same order as patches_dir): the CT volume its patches' coordinates are in, e.g. segments
    # of other scrolls (``scripts/prepare_scroll.py``); an empty entry (or an empty list) means data.volume. The
    # volumes share data.volume_level. Not with extra_volumes.
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
    crop: list = field(default_factory=lambda: [16, 12, 20])  # samples along (normal, front tangent, advance)
    crop_spacing: float | list = 2.  # one number, or [normal, tangent, advance] (e.g. [1, 2, 2] at level 0)
    crop_forward: float = 10.  # crop centre ahead of the point along the advance direction
    channels: list = field(default_factory=lambda: [16, 32, 48])  # CNN widths, one 3x3x3 conv each
    strides: list = field(default_factory=lambda: [2, 2, 1])  # per conv; early downsampling keeps the CNN cheap
    dim: int = 96
    layers: int = 2
    heads: int = 4
    mlp_ratio: float = 2.
    memory_rows: int = 3  # previous rows of point tokens the front cross-attends to
    attention: bool = True  # False: points predict independently (no self- or memory attention)
    dropout: float = 0.
    # Surface-token grid: small CT crops along the curve (columns, trace.spline_samples per gap) for the current
    # row and the last memory_rows rows. Axial attention runs along rows and columns; front points cross-attend to the
    # grid (replacing memory over past point tokens) and write their own crop token into it at their columns.
    surface_tokens: bool = False
    token_crop: list = field(default_factory=lambda: [8, 8, 8])  # samples along (normal, curve tangent, advance)
    token_spacing: float | list = 2.
    token_channels: list = field(default_factory=lambda: [8, 16, 32])
    token_strides: list = field(default_factory=lambda: [2, 2, 1])
    grid_layers: int = 2  # axial (row + column) attention blocks over the grid
    # Coarse context crop per front point (empty: off): a wide, low-resolution view of the surrounding sheets
    # (e.g. [16, 16, 16] at 6 vx is a 96 vx cube), through its own CNN, added to the point's token through a
    # zero-initialised projection (so a model initialised from a checkpoint without it starts out unchanged).
    context_crop: list = field(default_factory=list)
    context_spacing: float | list = 6.
    context_channels: list = field(default_factory=lambda: [8, 16, 32])
    context_strides: list = field(default_factory=lambda: [2, 2, 1])
    # Grid memory at these lags (rows back, increasing), e.g. [1, 2, 3, 6, 12], instead of the last memory_rows
    # rows (empty). Rows get two more geometry features (log lag, anchor flag), zero-initialised from a checkpoint
    # without them; memory_rows still scales the original age feature (lags beyond it saturate there).
    memory_lags: list = field(default_factory=list)
    # Keep the trace's first grid row (the seed row) in the grid memory for good: a fixed reference of the sheet
    # the trace started on.
    memory_anchor: bool = False
    # Profile along the normal at every grid column (empty: off): a long, thin crop, e.g. [64, 3, 3] at 2 vx
    # (+-63 vx, several sheets either side), through its own CNN, added to the column's token through a
    # zero-initialised projection. Shows the stack of neighbouring sheets the small token crops cannot.
    profile_crop: list = field(default_factory=list)
    profile_spacing: float | list = 2.
    profile_channels: list = field(default_factory=lambda: [16, 32, 48])
    profile_strides: list = field(default_factory=lambda: [2, 2, 2])
    # Crop CNNs of ResNet-D basic blocks (two convs per stage, residual skip) instead of single convs; every crop
    # size must then be divisible by 2 per stride-2 stage.
    residual: bool = False
    point_bottleneck: int = 0  # residual point CNN: 1x1 conv to this many channels before flattening (0: none)
    # One transformer (``layers`` blocks, width ``dim``, FFN ``dim * mlp_ratio``) over the point and grid tokens
    # together, with learned type embeddings, instead of the axial grid blocks plus point blocks.
    joint_transformer: bool = False
    # A learned mode embedding (zero at start) added to every token: 0 sheet tracing, 1 following a fiber with a
    # 3-point front across it (data.follow_fraction). Starts as the checkpoint it is initialised from.
    fiber_follow_mode: bool = False
    # Fiber tokens: the fiber points nearest the front each step (data.fiber_pool), as extra tokens of the joint
    # transformer: position and direction in the front frame, family, seed-fiber flag and the fiber's instance code
    # (random per episode, the same for all points of one fiber, different between fibers). 0: none.
    fiber_tokens: int = 0


@dataclass
class TraceConfig:
    # 'smoothing': a confidence-weighted smoothing B-spline (spline_ctrl control points) is fitted through the
    # front-point predictions. 'hermite': front points are the curve's control points; each moves its position
    # and tilts its tangent, and the curve between them is the cubic Hermite through those (points can be spaced
    # widely; widen model.crop along the front to about point_spacing).
    curve: str = 'smoothing'
    spline_ctrl: int = 0  # smoothing: B-spline control points; 0 means max(4, front_points // 2)
    spline_smooth: float = 1e-2  # smoothing: second-difference penalty on the change of control points
    spline_samples: int = 4  # dense curve samples per point gap (curve loss, arc-length respacing, hermite output)
    respace: bool = True  # re-space front points uniformly by arc length after every step
    confidence_threshold: float = .5  # below: the point is stopped (masked) and rides the spline
    # hermite: where a stopped point goes. 'displacement' moves it by its confident neighbours' interpolated
    # displacement (keeps any offset, e.g. a sheet away); 'curve' also puts it on the Hermite curve through its
    # confident neighbours when it has one on each side, so it can land back on the sheet and rejoin.
    stopped: str = 'displacement'
    min_weight: float = 1e-3  # spline-fit weight floor for low-confidence points


@dataclass
class LossConfig:
    normal: float = 1.  # prediction error along the GT sheet normal
    advance: float = .3  # ... along the GT advance direction
    tangent: float = .1  # ... along the front, against the point's seed column (see column_target)
    spline: float = .5  # dense spline samples' distance to the sheet
    confidence: float = .5
    # Weight on- and off-sheet labels equally (off-sheet points are rare, so an unbalanced BCE learns to keep all).
    confidence_balance: bool = True
    # Along-front target is the point's seed column (v = k * point_spacing) rather than its current position, so
    # cumulative shrinking and sideways drift of the front are penalised, not just each step's sliding.
    column_target: bool = True
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
    detach_steps: bool = True  # False backpropagates through the whole rollout
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
    # into a real trace (model.memory_lags). Training episodes then start that many rows further back; validation
    # episodes do not change.
    history_rows: int = 0
    # Exponential moving average of the weights (decay per iteration, e.g. 0.999; 0: off), warmed up as
    # min(ema, (1 + n) / (10 + n)) over the first updates. Validation, best.pt and evaluation use the averaged
    # weights; checkpoints keep both (``model`` raw, ``ema`` averaged).
    ema: float = 0.
    # Backpropagate each rollout step's loss right after the step instead of the summed loss at the end: the same
    # gradient (steps are detached from each other), with only one step's activations in memory. Needs detach_steps.
    backward_per_step: bool = False
    # Live chains (like fiber_follow's live continuation): a front is not restarted from GT after its rollout but
    # continues in a later iteration from exactly where the model left it (positions, curve, confidences, grid
    # memory) on the next GT window and block of its patch, which a loader worker reads on request (live.py).
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

    def spline_ctrl(self) -> int:
        return self.trace.spline_ctrl or max(4, self.data.front_points//2)


def from_dict(document: dict, strict: bool = True) -> Config:
    """``strict=False`` (loading older checkpoints) drops fields this version no longer has, with a note."""
    unknown = set(document)-set(SECTIONS)
    if unknown:
        raise ValueError(f'Unknown config sections: {sorted(unknown)}')
    parts = {}
    for name, cls in SECTIONS.items():
        values = dict(document.get(name, {}))
        names = {f.name for f in fields(cls)}
        if name == 'data' and 'front_points' not in values and 'point_spacing' not in values:
            # Earlier run configs serialized these two fields between volume_level and step.
            keys = list(values)
            if 'volume_level' in values and 'step' in values:
                i = keys.index('volume_level')+1
                previous = keys[i:i+2]
                if (len(previous) == 2 and keys[i+2:i+3] == ['step']
                        and set(previous).isdisjoint(names)
                        and isinstance(values[previous[0]], int)
                        and isinstance(values[previous[1]], (int, float))):
                    values['front_points'] = values.pop(previous[0])
                    values['point_spacing'] = values.pop(previous[1])
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
    d, m, t = config.data, config.model, config.trace
    if d.extra_levels and (len(d.extra_levels) != len(d.extra_volumes)
                           or any(int(level) < d.volume_level for level in d.extra_levels)):
        raise ValueError('data.extra_levels needs one level per extra volume, none finer than data.volume_level')
    if d.follow_fraction and (not 0 < d.follow_fraction < 1 or not d.fiber_runs or not d.front_points_range
                              or not m.fiber_follow_mode):
        raise ValueError('data.follow_fraction needs data.fiber_runs, data.front_points_range and model.fiber_follow_mode')
    if m.fiber_tokens and not m.joint_transformer:
        raise ValueError('model.fiber_tokens needs model.joint_transformer')
    if not 0 <= d.fiber_fraction <= 1 or (d.fiber_fraction and not d.fiber_runs):
        raise ValueError('data.fiber_fraction must be in [0, 1] and needs data.fiber_runs')
    if not 0 <= d.hard_fraction <= 1 or (d.hard_fraction and not d.hard_windows):
        raise ValueError('data.hard_fraction must be in [0, 1] and needs data.hard_windows')
    if d.extra_level_dirs and len(d.extra_level_dirs) != len(d.extra_volumes):
        raise ValueError('data.extra_level_dirs needs one directory per extra volume')
    dirs = len(d.patches_dir) if isinstance(d.patches_dir, (list, tuple)) else 1
    if d.patch_volumes and (len(d.patch_volumes) != dirs or (d.extra_volumes and any(d.patch_volumes))):
        raise ValueError('data.patch_volumes needs one entry per patch directory, and no data.extra_volumes')
    if d.dir_weights and (len(d.dir_weights) != dirs or min(d.dir_weights) < 0 or not sum(d.dir_weights)):
        raise ValueError('data.dir_weights needs one non-negative weight per patch directory, not all zero')
    if d.front_points_range and (len(d.front_points_range) != 2 or not 3 <= int(d.front_points_range[0])
                                 <= int(d.front_points_range[1]) or config.train.live_chains):
        raise ValueError('data.front_points_range needs [min, max] with 3 <= min <= max, and no train.live_chains')
    if t.stopped not in ('displacement', 'curve'):
        raise ValueError("trace.stopped must be 'displacement' or 'curve'")
    if t.curve not in ('smoothing', 'hermite'):
        raise ValueError("trace.curve must be 'smoothing' or 'hermite'")
    if t.curve == 'smoothing' and d.front_points < 4:
        raise ValueError('data.front_points must be at least 4 for the smoothing spline')
    if t.curve == 'hermite' and d.front_points < 3:
        raise ValueError('data.front_points must be at least 3 for the hermite curve')
    if t.curve == 'smoothing' and not 4 <= config.spline_ctrl() <= d.front_points:
        raise ValueError('trace.spline_ctrl must lie in [4, data.front_points]')
    if len(m.crop) != 3 or any(int(c) < 2 for c in m.crop):
        raise ValueError('model.crop needs three sizes >= 2')
    if d.volume_level < 0 or d.roi % (1 << d.volume_level):
        raise ValueError('data.roi must be divisible by 2**data.volume_level')
    if config.train.amp not in ('bf16', 'none'):
        raise ValueError("train.amp must be 'bf16' or 'none'")
    if len(m.strides) != len(m.channels) or any(int(s) not in (1, 2) for s in m.strides):
        raise ValueError('model.strides needs one stride (1 or 2) per model.channels entry')
    if len(m.token_strides) != len(m.token_channels) or any(int(s) not in (1, 2) for s in m.token_strides):
        raise ValueError('model.token_strides needs one stride (1 or 2) per model.token_channels entry')
    if m.context_crop and (len(m.context_crop) != 3 or len(m.context_strides) != len(m.context_channels)):
        raise ValueError('model.context_crop needs three sizes and one context stride per context channel')
    if m.residual:
        for name, crop, strides in (('crop', m.crop, m.strides), ('token_crop', m.token_crop, m.token_strides),
                                    ('context_crop', m.context_crop, m.context_strides),
                                    ('profile_crop', m.profile_crop, m.profile_strides)):
            factor = 2**sum(int(s) == 2 for s in strides)
            if crop and any(int(c) % factor for c in crop):
                raise ValueError(f'model.{name} sizes must be divisible by {factor} with model.residual')
        if any(int(c) % 4 for c in list(m.channels)+list(m.token_channels)+list(m.context_channels)):
            raise ValueError('model.residual needs channel counts divisible by 4 (GroupNorm(4))')
    if m.point_bottleneck and not m.residual:
        raise ValueError('model.point_bottleneck needs model.residual')
    if m.joint_transformer and not (m.surface_tokens and m.attention):
        raise ValueError('model.joint_transformer needs model.surface_tokens and model.attention')
    lags = [int(lag) for lag in m.memory_lags]
    if lags and (not m.surface_tokens or lags[0] < 1 or any(b <= a for a, b in zip(lags, lags[1:]))):
        raise ValueError('model.memory_lags needs model.surface_tokens and increasing lags >= 1')
    if m.memory_anchor and not m.surface_tokens:
        raise ValueError('model.memory_anchor needs model.surface_tokens')
    if m.profile_crop and (not m.surface_tokens or len(m.profile_crop) != 3
                           or len(m.profile_strides) != len(m.profile_channels)):
        raise ValueError('model.profile_crop needs model.surface_tokens, three sizes and one profile stride per '
                         'profile channel')
    tc = config.train
    if tc.backward_per_step and not tc.detach_steps:
        raise ValueError('train.backward_per_step needs train.detach_steps')
    if not 0 <= tc.ema < 1:
        raise ValueError('train.ema must be in [0, 1)')
    if tc.history_rows < 0:
        raise ValueError('train.history_rows must be >= 0')
    if tc.live_chains and (not d.tight_blocks or tc.history_rows or m.memory_anchor or len(tc.chain_steps) != 2
                           or not d.steps <= int(tc.chain_steps[0]) <= int(tc.chain_steps[1])
                           or max(lags or [m.memory_rows]) > d.steps):
        # Continued chains of different depths share batches, so every continued front must hold the same memory:
        # all of it, once a rollout of data.steps has filled it.
        raise ValueError('train.live_chains needs data.tight_blocks, no train.history_rows or model.memory_anchor, '
                         'memory lags (or memory_rows) <= data.steps, and train.chain_steps [min, max] with '
                         'data.steps <= min <= max')
    if m.surface_tokens and not m.attention:
        raise ValueError('model.surface_tokens needs model.attention (front_points read the grid by attention)')
    if m.dim % m.heads:
        raise ValueError('model.dim must be divisible by model.heads')


def path_mapper(texts):
    """``--path-map OLD=NEW`` texts -> a function applying them to every data path of a config (in place)."""
    pairs = [text.split('=', 1) for text in texts]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError('--path-map needs OLD=NEW')
    remap = lambda path: next((new+str(path)[len(old):] for old, new in pairs if str(path).startswith(old)), path)

    def apply(cfg):
        d = cfg.data
        d.volume, d.extra_volumes = remap(d.volume), [remap(p) for p in d.extra_volumes]
        d.patch_volumes = [remap(p) if p else p for p in d.patch_volumes]
        d.patches_dir = [remap(p) for p in d.patches_dir] if isinstance(d.patches_dir, list) else remap(d.patches_dir)
        return cfg
    return apply
