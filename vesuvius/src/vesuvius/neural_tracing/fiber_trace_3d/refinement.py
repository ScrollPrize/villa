"""Opt-in training feedback. Geometry is base ZYX, never output-grid voxels.

The queue carries immutable observations. Only the training boundary commits
proposals, ordered by canonical sample index, to shared memory-mapped curves.
"""
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
import functools
import hashlib
import json
import os
import tempfile
import time

import numpy as np


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = False
    iterations: int = 4
    spacing_base: float = 8.0
    samples_per_segment: int = 4
    normal_radius_base: float = 40.0
    tangent_radius_base: float = 80.0
    gt_weight: float = 1.0
    shape_weight: float = 0.01
    learning_rate: float = 0.01
    max_step_fraction: float = 0.125
    border_fraction: float = 0.125

    @classmethod
    def parse(cls, raw):
        values = raw.get("refinement", {})
        if not isinstance(values, dict):
            raise ValueError("refinement must be an object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown refinement settings: {sorted(unknown)}")
        result = cls(**values)
        if type(result.enabled) is not bool:
            raise ValueError("refinement.enabled must be boolean")
        for name in ("iterations", "samples_per_segment"):
            if type(getattr(result, name)) is not int or not 1 <= getattr(result, name) <= 64:
                raise ValueError(f"refinement.{name} must be an integer in [1,64]")
        for name in ("spacing_base", "normal_radius_base", "tangent_radius_base", "learning_rate",
                     "max_step_fraction", "border_fraction"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"refinement.{name} must be positive finite")
        for name in ("gt_weight", "shape_weight"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"refinement.{name} must be nonnegative finite")
        if result.border_fraction > 0.5 or result.max_step_fraction > 1:
            raise ValueError("refinement border_fraction<=0.5 and max_step_fraction<=1 required")
        if result.enabled and raw.get("model_3d", raw.get("model", {})).get("conditioned_decoder_enabled", False):
            raise ValueError("refinement requires an unconditioned presence output")
        return result


def arclength(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def interpolate(values, knots, locations):
    return np.stack([np.interp(locations, knots, values[:, i]) for i in range(values.shape[1])], axis=1)


def resample(points, material, count=None):
    """Carry immutable-reference arc coordinates through reparameterization."""
    arc = arclength(points)
    keep = np.r_[True, np.diff(arc) > 1e-10]
    if not np.isfinite(points).all() or arc[-1] <= 1e-10:
        raise ValueError("refinement produced a degenerate/nonfinite curve")
    at = np.linspace(0., arc[-1], len(points) if count is None else count)
    return interpolate(np.c_[points, material][keep], arc[keep], at)


def ellipse_metric(normals, cfg):
    n = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(n, axis=1)
    if not np.isfinite(n).all() or np.any(lengths < 1e-10):
        raise ValueError("refinement requires valid native Lasagna normals")
    n = n / lengths[:, None]
    return (np.eye(3)[None] / cfg.tangent_radius_base**2 +
            n[:, :, None] * n[:, None, :] *
            (1 / cfg.normal_radius_base**2 - 1 / cfg.tangent_radius_base**2))


def taper(points_patch, shape, fraction):
    distances = np.minimum(points_patch, np.asarray(shape) - 1 - points_patch)
    return np.clip(np.min(distances / (np.asarray(shape) * fraction), axis=1), 0, 1)


def clamp_update(delta, points, metric, limits, tangent=None):
    tangent = np.gradient(points, axis=0) if tangent is None else tangent
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    delta = delta - tangent * np.sum(delta * tangent, axis=1, keepdims=True)
    norm = np.sqrt(np.maximum(0., np.einsum("ni,nij,nj->n", delta, metric, delta)))
    return delta * np.minimum(1., limits / np.maximum(norm, 1e-12))[:, None]


def step_limits(points_patch, shape, cfg, independent, endpoints):
    """Test pieces have no stitching seam: uniform cap, including endpoints."""
    independent = np.broadcast_to(independent, (len(points_patch),))
    limits = np.full(len(points_patch), cfg.max_step_fraction)
    stitched = ~independent
    limits[stitched] *= taper(points_patch[stitched], shape, cfg.border_fraction)
    pinned = np.asarray(endpoints)[~independent[endpoints]]
    limits[pinned] = 0.
    return limits


def semi_implicit_step(points, baseline, metric, force, cfg, links=None, spacing=None):
    """Cartesian 3x3 block-tridiagonal solve followed by plane projection.

    I + dt*(2*gt_weight*Q + 2*shape_weight*L/h^2). SciPy executes
    the narrow-band solve in native code, with no Python node iteration.
    """
    from scipy.linalg import solve_banded
    n = len(points)
    dt = cfg.learning_rate * cfg.normal_radius_base**2
    h = max(arclength(baseline)[-1] / max(1, n - 1), 1e-6) if spacing is None else spacing
    k = np.broadcast_to(dt * 2 * cfg.shape_weight / np.asarray(h)**2, (n,))
    edge_k = .5 * (k[:-1] + k[1:])
    if links is not None:
        edge_k = edge_k * links
    displacement = points - baseline
    lap = np.zeros_like(displacement)
    edge = (displacement[1:] - displacement[:-1]) * edge_k[:, None]
    lap[:-1] -= edge
    lap[1:] += edge
    rhs = dt * force - dt * 2 * cfg.gt_weight * np.einsum("nij,nj->ni", metric, displacement) - lap
    blocks = dt * 2 * cfg.gt_weight * metric
    degree = np.r_[edge_k, 0.] + np.r_[0., edge_k]
    blocks = blocks + np.eye(3)[None] * (1 + degree)[:, None, None]
    band = np.zeros((7, n * 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            columns = np.arange(n) * 3 + j
            band[3 + i - j, columns] = blocks[:, i, j]
    band[0, 3:] = np.repeat(-edge_k, 3)
    band[6, :-3] = np.repeat(-edge_k, 3)
    return solve_banded((3, 3), band, rhs.reshape(-1), check_finite=False).reshape(n, 3)


def presence_force(points, to_patch, presence, patch_shape, cfg, output_stride=1, segment_weights=None, independent_piece=False):
    """Uniform segment quadrature; each observation pushes both endpoints.

    Derivatives of the existing nonlinear base->patch mapping are evaluated
    with symmetric finite differences. No affine-only smooth-warp shortcut.
    """
    from scipy.ndimage import map_coordinates
    t = (np.arange(cfg.samples_per_segment) + .5) / cfg.samples_per_segment
    q = (points[:-1, None] * (1 - t)[None, :, None] + points[1:, None] * t[None, :, None]).reshape(-1, 3)
    # Native stride-k predictions/targets use output index j -> input j*k.
    # This is not an align_corners=False image-resizing lattice.
    def sample(base):
        mapped = to_patch(base)
        coords = mapped / output_stride
        return map_coordinates(presence, coords.T, order=1, mode="nearest", prefilter=False)
    epsilon = .5
    offsets = np.concatenate((np.eye(3), -np.eye(3))) * epsilon
    sampled = sample((q[None] + offsets[:, None]).reshape(-1, 3)).reshape(6, -1)
    gradients = ((sampled[:3] - sampled[3:]) / (2 * epsilon)).T.reshape(-1, len(t), 3)
    if independent_piece:
        centers = to_patch(q) / output_stride
        supported = np.all((centers >= 0) & (centers <= np.array(presence.shape)-1), axis=1)
        gradients *= supported.reshape(-1, len(t), 1)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) if segment_weights is None else segment_weights
    gradients *= lengths[:, None, None] / len(t)
    force = np.zeros_like(points)
    force[:-1] += np.sum(gradients * (1 - t)[None, :, None], axis=1)
    force[1:] += np.sum(gradients * t[None, :, None], axis=1)
    mass = np.r_[lengths, 0.] / 2 + np.r_[0., lengths] / 2
    return force / np.maximum(mass[:, None], 1e-12)


def refine_curve(current, baseline_points, baseline_arc, normals, presence, to_patch, patch_shape, cfg, output_stride=1, *, independent_piece=False):
    curve = np.asarray(current, dtype=np.float64).copy()
    normal_arc = curve[:, 3].copy()
    # Sheet axes are unoriented: interpolate nn^T metrics, never signed normals.
    metrics = ellipse_metric(normals, cfg).reshape(-1, 9)
    for _ in range(cfg.iterations):
        points = curve[:, :3]
        baseline = interpolate(baseline_points, baseline_arc, curve[:, 3])
        metric = interpolate(metrics, normal_arc, curve[:, 3]).reshape(-1, 3, 3)
        force = presence_force(points, to_patch, presence, patch_shape, cfg, output_stride,
                               segment_weights=np.diff(curve[:, 3]), independent_piece=independent_piece)
        update = semi_implicit_step(points, baseline, metric, force, cfg)
        limits = step_limits(to_patch(points), patch_shape, cfg, independent_piece, np.array([0, len(points)-1]))
        update = clamp_update(update, points, metric, limits)
        curve = resample(points + update, curve[:, 3])
    return curve


_ENV = "VILLA_FIBER_REFINEMENT_STATE"


@contextmanager
def file_lock(path, shared=False):
    # flock is supported on both target platforms (Ubuntu/macOS).
    import fcntl
    with open(path, "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield


class GeometryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, key, points, metadata, cfg):
        path = self.root / key
        with file_lock(self.root / "catalog.lock"):
            if path.exists():
                if json.loads((path / "metadata.json").read_text()) != json.loads(json.dumps(metadata)):
                    raise ValueError("refinement geometry metadata changed for an existing source")
                return
            if (self.root / "restored.json").exists():
                raise ValueError("refinement resume sources differ from the saved geometry; refusing to reset GT")
            arc = arclength(points)
            curve = resample(points, arc, max(2, int(np.ceil(arc[-1] / cfg.spacing_base)) + 1))
            path.mkdir()
            np.save(path / "baseline.npy", np.c_[points, arc])
            np.save(path / "current.npy", curve)
            (path / "metadata.json").write_text(json.dumps(metadata))

    def validate_split(self, keys, split):
        if (self.root / "restored.json").exists():
            expected = {p.name for p in self.root.iterdir() if p.is_dir() and
                        json.loads((p / "metadata.json").read_text()).get("split") == split}
            if expected != set(keys):
                raise ValueError(f"refinement resume {split} dataset membership changed")

    def test_patch(self, parent, sample_index, params, shape, spacing, start, current, baseline):
        """Lazy independent local state; parent identity was validated by register.

        New deterministic crops may be first encountered after a checkpoint.
        They always start from immutable GT, including after legacy resumes.
        """
        from dataclasses import asdict
        identity = dict(protocol="fixed-test-crop-v1", parent=parent,
                        sample_index=int(sample_index), params=asdict(params),
                        shape=shape, spacing=float(spacing), start=int(start))
        serialized = json.dumps(identity, sort_keys=True,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
        key = hashlib.sha256(serialized.encode()).hexdigest()
        path = self.root / key
        with file_lock(self.root / "catalog.lock"):
            if not path.exists():
                parent_metadata = json.loads((self.root / parent / "metadata.json").read_text())
                metadata = {**parent_metadata, "split": "test_patch", "crop": json.loads(serialized)}
                metadata.pop("export", None)  # cropped endpoints, not whole-fiber CP metadata
                path.mkdir()
                np.save(path / "baseline.npy", baseline)
                np.save(path / "current.npy", current)
                (path / "metadata.json").write_text(json.dumps(metadata))
        return key, self.read(key)

    def read(self, key):
        path = self.root / key
        with file_lock(path / "lock", shared=True):
            return np.load(path / "current.npy", mmap_mode="r").copy()

    def baseline(self, key):
        return np.load(self.root / key / "baseline.npy", mmap_mode="r")

    def commit(self, key, start, before, after):
        path = self.root / key
        with file_lock(path / "lock"):
            target = np.load(path / "current.npy", mmap_mode="r+")
            view = target[start:start + len(before)]
            if not np.array_equal(view, before):
                return False
            if after.shape != before.shape or not np.isfinite(after).all() or np.any(np.diff(after[:, 3]) <= 0):
                raise ValueError("invalid refinement proposal")
            view[:] = after
            return True

    def save(self, target):
        """One portable archive shared by raw/EMA snapshots at this step."""
        import zipfile
        target = Path(target)
        temporary = target.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(self.root.iterdir()):
                if not path.is_dir():
                    continue
                with file_lock(path / "lock", shared=True):
                    for name in ("baseline.npy", "current.npy", "metadata.json"):
                        archive.write(path / name, f"{path.name}/{name}")
                    curve = np.load(path / "current.npy")
                    metadata = json.loads((path / "metadata.json").read_text())
                    if metadata.get("split") == "test":
                        # Whole-test entries are identity/baseline records only;
                        # legacy current.npy is retained for archive compatibility.
                        curve = np.load(path / "baseline.npy")
                        metadata = {**metadata, "evaluation_role": "immutable_source_only"}
                    # An eligible component is exported independently: never
                    # reconnect gaps or fabricate extrapolation.
                    export = metadata.get("export", {})
                    cp_arc = export.get("control_arc", [curve[0, 3], curve[-1, 3]])
                    at = np.unique(np.r_[curve[:, 3], cp_arc])
                    points = interpolate(curve[:, :3], curve[:, 3], at)[:, ::-1].tolist()
                    cps = interpolate(curve[:, :3], curve[:, 3], cp_arc)[:, ::-1].tolist()
                    segments = export.get("segments")
                    if segments is not None:
                        cps = [dict(position=p, **({"segment_to_next": m} if m is not None else {}))
                               for p, m in zip(cps, segments, strict=True)]
                    obj = {"type": "vc3d_fiber", "version": 3 if segments is not None else 1,
                           "optimization_mode": "native_fiber_trace3d" if segments is not None else "lasagna",
                           "generation": export.get("generation", 0), "line_points": points,
                           "control_points": cps, "source_format": export.get("source_format", "json"),
                           "refinement": {k:v for k,v in metadata.items() if k != "export"}}
                    archive.writestr(f"{path.name}/fiber.json", json.dumps(obj))
        temporary.replace(target)
        return {"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    def restore(self, target, digest):
        import zipfile
        target = Path(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("refined-fiber snapshot checksum mismatch")
        with zipfile.ZipFile(target) as archive:
            for name in archive.namelist():
                pieces = Path(name).parts
                if len(pieces) != 2 or len(pieces[0]) != 64 or any(c not in "0123456789abcdef" for c in pieces[0]):
                    raise ValueError("invalid refinement archive entry")
                if pieces[1] not in {"baseline.npy", "current.npy", "metadata.json", "fiber.json"}:
                    raise ValueError("invalid refinement archive file")
                archive.extract(name, self.root)
        (self.root / "restored.json").write_text("{}")


def refinement_session(function):
    @functools.wraps(function)
    def wrapped(config_path, **kwargs):
        raw = json.loads(Path(config_path).read_text())
        cfg = RefinementConfig.parse(raw)
        if not cfg.enabled:
            return function(config_path, **kwargs)
        if _ENV in os.environ:
            raise RuntimeError("refinement state must be owned by one training supervisor")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("refinement multi-GPU launch must use the cooperative manager, not torchrun")
        resume = kwargs.get("resume_checkpoint") or raw.get("training", {}).get("resume") or raw.get("resume")
        with tempfile.TemporaryDirectory(prefix="fiber-refinement-") as root:
            if resume:
                import torch
                state = torch.load(resume, map_location="cpu", weights_only=False)
                saved = state.get("refinement_state")
                if saved:
                    GeometryStore(root).restore(Path(resume).parent / saved["file"], saved["sha256"])
                elif state.get("config", {}).get("refinement", {}).get("enabled"):
                    raise ValueError("refinement checkpoint is missing its fiber state")
            os.environ[_ENV] = root
            try:
                return function(config_path, **kwargs)
            finally:
                os.environ.pop(_ENV, None)
    return wrapped


def active_store():
    if _ENV not in os.environ:
        raise RuntimeError("enabled refinement requires the training launcher (missing shared state)")
    return GeometryStore(os.environ[_ENV])


def export_metadata(fiber):
    from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index
    from vc3d_fiber_format import serialize_segment_metadata
    arc = arclength(fiber.line_points_zyx.astype(np.float64))
    positions = [float(arc[control_point_line_index(fiber, i)]) for i in range(len(fiber.control_points_xyz))]
    nml = fiber.metadata.get("source_format") == "nml"
    segments = ([serialize_segment_metadata(m) if m is not None else None for m in fiber.control_point_segments]
                if fiber.version == 3 and not nml else None)
    return {"control_arc": positions, "segments": segments,
            "source_format": "nml" if nml else "json", "generation": fiber.generation}


@dataclass(frozen=True)
class RefinementObservation:
    key: str
    start: int
    current: np.ndarray
    baseline: np.ndarray
    normals: np.ndarray
    params: object
    spacing: float
    shape: tuple
    config: RefinementConfig
    output_stride: int = 1
    independent_piece: bool = False


class BatchedPatchTransform:
    """Batch sparse queries through the loader's shared ordered shear mapping."""
    def __init__(self, observations):
        self.spacing = np.array([o.spacing for o in observations])
        self.centers = np.stack([o.params.cp_volume_zyx for o in observations])
        self.local = np.stack([o.params.cp_local_zyx for o in observations])
        self.matrices = np.stack([o.params.source_to_output_zyx for o in observations])
        groups = {}
        for i, o in enumerate(observations):
            s = o.params.smooth
            if s is None or s.mode == "none":
                continue
            key = (s.mode, tuple(s.control_min_zyx), tuple(s.control_spacing_zyx), tuple(c.shape for c in s.controls))
            groups.setdefault(key, []).append(i)
        self.groups = []
        for ids in groups.values():
            s = observations[ids[0]].params.smooth
            lookup = np.full(len(observations), -1, dtype=int)
            lookup[ids] = np.arange(len(ids))
            controls = tuple(np.stack([observations[i].params.smooth.controls[j] for i in ids]).astype(np.float64)
                             for j in range(len(s.controls)))
            self.groups.append((lookup, s, controls))

    def __call__(self, points, ids):
        from .loader import _smooth_forward_rel_zyx
        relative = points / self.spacing[ids, None] - self.centers[ids]
        for lookup, s, controls in self.groups:
            local_ids = lookup[ids]
            selected = local_ids >= 0
            if np.any(selected):
                relative[selected] = _smooth_forward_rel_zyx(relative[selected], s,
                    controls=controls, batch_index=local_ids[selected])
        return self.local[ids] + np.einsum("ni,nji->nj", relative, self.matrices[ids])


def refine_observations(observations, maps):
    """Solve independent ragged curves as one narrow-band system.

    Zero off-diagonal links separate patches. Grouping changes dispatch only;
    quadrature, ellipse metric and semi-implicit update match refine_curve.
    """
    from scipy.ndimage import map_coordinates
    if not observations:
        return []
    cfg = observations[0].config
    if any(o.config != cfg or o.shape != observations[0].shape or o.output_stride != observations[0].output_stride for o in observations):
        raise ValueError("refinement batch must share settings and prediction lattice")
    transform = BatchedPatchTransform(observations)
    sizes = np.array([len(o.current) for o in observations])
    offsets = np.r_[0, np.cumsum(sizes)]
    ids = np.repeat(np.arange(len(sizes)), sizes)
    links = ids[1:] == ids[:-1]
    edges = np.flatnonzero(links)
    endpoint = np.r_[offsets[:-1], offsets[1:]-1]
    previous, following = np.arange(len(ids))-1, np.arange(len(ids))+1
    previous[offsets[:-1]] = offsets[:-1]
    following[offsets[1:]-1] = offsets[1:]-1
    curves = [o.current.copy() for o in observations]
    metrics = [ellipse_metric(o.normals, cfg).reshape(-1, 9) for o in observations]
    maps = np.stack(maps)
    t = (np.arange(cfg.samples_per_segment) + .5) / cfg.samples_per_segment
    qids = np.repeat(ids[edges], len(t))
    query_ids = np.tile(qids, 6)
    perturbations = np.concatenate((np.eye(3), -np.eye(3))) * .5
    for _ in range(cfg.iterations):
        curve = np.concatenate(curves)
        points = curve[:,:3]
        baseline_parts = [interpolate(o.baseline[:,:3], o.baseline[:,3], c[:,3]) for o,c in zip(observations, curves)]
        baseline = np.concatenate(baseline_parts)
        metric = np.concatenate([interpolate(q, o.current[:,3], c[:,3]).reshape(-1,3,3)
                                 for q,o,c in zip(metrics, observations, curves)])
        q = (points[edges,None]*(1-t)[None,:,None] + points[edges+1,None]*t[None,:,None]).reshape(-1,3)
        mapped = transform((q[None]+perturbations[:,None]).reshape(-1,3), query_ids) / observations[0].output_stride
        sampled = map_coordinates(maps, np.concatenate((query_ids[None], mapped.T)),
                                  order=1, mode="nearest", prefilter=False).reshape(6,-1)
        gradient = (sampled[:3]-sampled[3:]).T.reshape(-1,len(t),3)
        independent_queries = np.array([o.independent_piece for o in observations], dtype=bool)[qids]
        if np.any(independent_queries):
            centers = transform(q, qids) / observations[0].output_stride
            supported = np.all((centers >= 0) & (centers <= np.array(maps.shape[1:])-1), axis=1)
            gradient *= (supported | ~independent_queries).reshape(-1,len(t),1)
        weights = curve[edges+1,3]-curve[edges,3]
        gradient *= weights[:,None,None]/len(t)
        force = np.zeros_like(points)
        force[edges] += np.sum(gradient*(1-t)[None,:,None], axis=1)
        force[edges+1] += np.sum(gradient*t[None,:,None], axis=1)
        mass = np.zeros(len(points))
        mass[edges] += weights/2
        mass[edges+1] += weights/2
        force /= np.maximum(mass[:,None], 1e-12)
        spacing = np.concatenate([np.full(len(b), max(arclength(b)[-1]/(len(b)-1),1e-6)) for b in baseline_parts])
        delta = semi_implicit_step(points, baseline, metric, force, cfg, links=links, spacing=spacing)
        independent = np.array([o.independent_piece for o in observations], dtype=bool)[ids]
        limits = step_limits(transform(points, ids), observations[0].shape, cfg, independent, endpoint)
        delta = clamp_update(delta, points, metric, limits, tangent=points[following]-points[previous])
        updated = points+delta
        curves = [resample(updated[a:b], curve[a:b,3]) for a,b in zip(offsets[:-1], offsets[1:])]
    return curves


class RefinementObserver:
    def __init__(self, cfg):
        self.config = cfg
        self.pending = []
        self.seconds = 0.
        self.accepted = self.stale = 0

    def __call__(self, batch, output):
        if not self.config.enabled:
            return
        from .model import presence_outputs
        # Retain only the small output probabilities, not the training graph.
        self.pending.append((batch.stream_indices.detach(), batch.refinement_geometry,
                             presence_outputs(output.detach()).amax(dim=1).squeeze(1).float()))

    def flush(self, distributed):
        if not self.config.enabled:
            return
        import torch.distributed as dist
        started = time.perf_counter()
        proposals = []
        todo, images = [], []
        for indices, observations, presence in self.pending:
            indices = indices.cpu().tolist()
            maps = presence.cpu().numpy()
            for index, observation, values in zip(indices, observations, maps):
                if observation is None:
                    continue
                todo.append((index, observation))
                images.append(values)
        afters = refine_observations([o for _,o in todo], images)
        for (index, observation), after in zip(todo, afters):
            proposals.append((index, observation.key, observation.start, observation.current, after))
        self.pending.clear()
        if distributed.enabled:
            gathered = [None] * distributed.world_size if distributed.is_main else None
            dist.gather_object(proposals, gathered, dst=0)
        else:
            gathered = [proposals]
        if distributed.is_main:
            store = active_store()
            for _, key, start, before, after in sorted((p for rows in gathered for p in rows), key=lambda p: p[0]):
                if store.commit(key, start, before, after):
                    self.accepted += 1
                else:
                    self.stale += 1
        # Do not let another rank start preparing a new test phase before all
        # proposals have been committed. Queue lookahead stays immutable.
        if distributed.enabled:
            dist.barrier()
        self.seconds += time.perf_counter() - started
