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

REFINEMENT_LOSS_NAMES = ("presence", "smoothing_normal", "smoothing_tangent", "localization", "direction")


def empty_refinement_statistics():
    return [0.] * (1+len(REFINEMENT_LOSS_NAMES))


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = False
    update_enabled: bool = True
    state_checkpoint: str | None = None
    state_sha256: str | None = None
    iterations: int = 4
    spacing_base: float = 8.0
    samples_per_segment: int = 4
    normal_radius_base: float = 40.0
    tangent_radius_base: float = 80.0
    gt_weight: float = 1.0
    shape_weight: float = 0.01
    shape_normal_weight: float | None = None
    shape_tangent_weight: float | None = None
    direction_weight: float = 0.0
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
        if type(result.enabled) is not bool or type(result.update_enabled) is not bool:
            raise ValueError("refinement enabled/update_enabled must be boolean")
        if result.state_checkpoint is not None and (not isinstance(result.state_checkpoint, str) or not result.state_checkpoint):
            raise ValueError("refinement.state_checkpoint must be a nonempty path string")
        if result.state_sha256 is not None and (not isinstance(result.state_sha256, str) or len(result.state_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in result.state_sha256)):
            raise ValueError("refinement.state_sha256 must be a lowercase SHA256")
        if not result.update_enabled and (not result.enabled or not result.state_checkpoint or not result.state_sha256):
            raise ValueError("frozen refinement requires enabled=true plus state_checkpoint/state_sha256")
        if (result.state_checkpoint is None) != (result.state_sha256 is None):
            raise ValueError("refinement state_checkpoint and state_sha256 must be specified together")
        for name in ("iterations", "samples_per_segment"):
            if type(getattr(result, name)) is not int or not 1 <= getattr(result, name) <= 64:
                raise ValueError(f"refinement.{name} must be an integer in [1,64]")
        for name in ("spacing_base", "normal_radius_base", "tangent_radius_base", "learning_rate",
                     "max_step_fraction", "border_fraction"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"refinement.{name} must be positive finite")
        for name in ("gt_weight", "shape_weight", "direction_weight", "shape_normal_weight", "shape_tangent_weight"):
            value = getattr(result, name)
            if value is None and name.startswith("shape_") and name != "shape_weight":
                continue
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


def bending_terms(points, links=None, metric=None):
    """Lumped arc mass and B=D2.T W D2 for integral |d²x/ds²|² ds.

    Current arc lengths are frozen within each solve. Interior three-point
    stencils only: free ends, no virtual fixed neighbours or cross-piece links.
    Return diagonal/upper bands and B*x, using differences for translation stability.
    """
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    connected = np.ones(len(lengths), dtype=bool) if links is None else np.asarray(links, dtype=bool)
    h = np.maximum(lengths, 1e-6)
    edge_mass = np.where(connected, h, 0.) / 2
    mass = np.maximum(np.r_[edge_mass, 0.] + np.r_[0., edge_mass], 1e-6)
    left, right = h[:-1], h[1:]
    weights = .5 * (left + right) * (connected[:-1] & connected[1:])
    a = 2 / (left * (left + right))
    c = 2 / (right * (left + right))
    b = -a-c
    if metric is not None:
        # Frozen symmetric 3x3 metric per interior curvature stencil.
        weighted = weights[:,None,None]*metric[1:-1]
        diagonal = np.zeros((len(points),3,3))
        diagonal[:-2] += weighted*a[:,None,None]**2
        diagonal[1:-1] += weighted*b[:,None,None]**2
        diagonal[2:] += weighted*c[:,None,None]**2
        upper = np.zeros((len(points)-1,3,3))
        upper[:-1] += weighted*(a*b)[:,None,None]
        upper[1:] += weighted*(b*c)[:,None,None]
        upper2 = weighted*(a*c)[:,None,None]
        curvature = a[:,None]*(points[:-2]-points[1:-1]) + c[:,None]*(points[2:]-points[1:-1])
        response = np.einsum("nij,nj->ni",weighted,curvature)
        gradient = np.zeros_like(points)
        for view, coefficient in ((gradient[:-2],a),(gradient[1:-1],b),(gradient[2:],c)):
            view += coefficient[:,None]*response
        return mass,diagonal,upper,upper2,gradient,float(np.sum(curvature*response))
    diagonal = np.zeros(len(points))
    diagonal[:-2] += weights*a*a
    diagonal[1:-1] += weights*b*b
    diagonal[2:] += weights*c*c
    upper = np.zeros(len(points)-1)
    upper[:-1] += weights*a*b
    upper[1:] += weights*b*c
    upper2 = weights*a*c
    curvature = a[:,None]*(points[:-2]-points[1:-1]) + c[:,None]*(points[2:]-points[1:-1])
    gradient = np.zeros_like(points)
    for view, coefficient in ((gradient[:-2], a), (gradient[1:-1], b), (gradient[2:], c)):
        view += (weights*coefficient)[:,None]*curvature
    return mass, diagonal, upper, upper2, gradient, float(np.sum(weights[:,None]*curvature**2))


def shape_metric(normal_projectors, cfg):
    normal = cfg.shape_weight if cfg.shape_normal_weight is None else cfg.shape_normal_weight
    tangent = cfg.shape_weight if cfg.shape_tangent_weight is None else cfg.shape_tangent_weight
    return normal*normal_projectors + tangent*(np.eye(3)-normal_projectors)


def normal_projectors(normals):
    n = np.asarray(normals,dtype=np.float64)
    n = n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
    return n[:,:,None]*n[:,None,:]


def semi_implicit_step(points, baseline, metric, force, cfg, links=None, material_lengths=None, normal_metric=None):
    """Native block-pentadiagonal bending/tether solve, then caller projects.

    With B including the anisotropic shape weights:
    [M + 2dt*gt*M*Q + 2dt*B] delta =
       dt*M*(force - 2gt*Q*(x-x_GT)) - 2dt*B*x.
    This smooths the resulting line, not displacement from the baseline.
    """
    from scipy.linalg import solve_banded
    n = len(points)
    dt = cfg.learning_rate * cfg.normal_radius_base**2
    if normal_metric is None:
        if cfg.shape_normal_weight is not None or cfg.shape_tangent_weight is not None:
            raise ValueError("anisotropic refinement requires normal projectors")
        bending_metric = np.broadcast_to(np.eye(3)*cfg.shape_weight,(n,3,3))
    else:
        bending_metric = shape_metric(normal_metric,cfg)
    mass, diagonal, upper, upper2, gradient, _ = bending_terms(points, links, bending_metric)
    if material_lengths is not None:
        edges = np.asarray(material_lengths) / 2
        if links is not None:
            edges = np.where(links, edges, 0.)
        mass = np.maximum(np.r_[edges, 0.] + np.r_[0., edges], 1e-12)
    k = dt * 2
    displacement = points - baseline
    rhs = dt * mass[:,None] * (force - 2 * cfg.gt_weight * np.einsum("nij,nj->ni", metric, displacement)) - k*gradient
    blocks = mass[:,None,None] * (np.eye(3)[None] + dt * 2 * cfg.gt_weight * metric)
    blocks = blocks + k*diagonal
    band = np.zeros((17, n * 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            columns = np.arange(n) * 3 + j
            band[8 + i - j, columns] = blocks[:, i, j]
            for offset,values in ((1,upper),(2,upper2)):
                columns = np.arange(offset,n)*3+j
                band[8-3*offset+i-j,columns] = k*values[:,i,j]
                columns = np.arange(n-offset)*3+j
                band[8+3*offset+i-j,columns] = k*values[:,j,i]
    return solve_banded((8, 8), band, rhs.reshape(-1), check_finite=False).reshape(n, 3)


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


def refine_curve(current, baseline_points, baseline_arc, normals, presence, to_patch, patch_shape, cfg, output_stride=1, *, independent_piece=False, directions=None):
    curve = np.asarray(current, dtype=np.float64).copy()
    normal_arc = curve[:, 3].copy()
    # Sheet axes are unoriented: interpolate nn^T metrics, never signed normals.
    metrics = ellipse_metric(normals, cfg).reshape(-1, 9)
    projectors = normal_projectors(normals).reshape(-1,9)
    for _ in range(cfg.iterations):
        points = curve[:, :3]
        baseline = interpolate(baseline_points, baseline_arc, curve[:, 3])
        metric = interpolate(metrics, normal_arc, curve[:, 3]).reshape(-1, 3, 3)
        force = presence_force(points, to_patch, presence, patch_shape, cfg, output_stride,
                               segment_weights=np.diff(curve[:, 3]), independent_piece=independent_piece)
        if cfg.direction_weight:
            extra, _ = direction_feedback(points,np.arange(len(points)-1),np.diff(curve[:,3]),
                lambda p,ids: to_patch(p),np.zeros(len(points),dtype=int),
                None if directions is None else np.asarray(directions)[None],output_stride,cfg)
            force += extra
        normal_metric = interpolate(projectors,normal_arc,curve[:,3]).reshape(-1,3,3)
        update = semi_implicit_step(points, baseline, metric, force, cfg, material_lengths=np.diff(curve[:,3]),normal_metric=normal_metric)
        limits = step_limits(to_patch(points), patch_shape, cfg, independent_piece, np.array([0, len(points)-1]))
        update = clamp_update(update, points, metric, limits)
        curve = resample(points + update, curve[:, 3])
    return curve


_ENV = "VILLA_FIBER_REFINEMENT_STATE"
_FROZEN_ENV = "VILLA_FIBER_REFINEMENT_FROZEN"


@contextmanager
def file_lock(path, shared=False):
    # flock is supported on both target platforms (Ubuntu/macOS).
    import fcntl
    with open(path, "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield


class GeometryStore:
    def __init__(self, root, *, read_only=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_only = bool(read_only)

    def register(self, key, points, metadata, cfg):
        path = self.root / key
        with file_lock(self.root / "catalog.lock"):
            if path.exists():
                if json.loads((path / "metadata.json").read_text()) != json.loads(json.dumps(metadata)):
                    raise ValueError("refinement geometry metadata changed for an existing source")
                return
            if self.read_only:
                raise ValueError("frozen refinement archive is missing a configured training fiber")
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
        if self.read_only:
            raise RuntimeError("frozen refinement cannot create test-patch state")
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
        if self.read_only:
            raise RuntimeError("frozen refinement geometry is read-only")
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
        if self.read_only:
            raise RuntimeError("frozen refinement reuses its immutable source archive")
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
        state_source = None
        if cfg.state_checkpoint:
            state_source = Path(cfg.state_checkpoint).expanduser()
            if not state_source.is_absolute():
                state_source = Path(config_path).resolve().parent / state_source
        elif resume:
            state_source = Path(resume)
        with tempfile.TemporaryDirectory(prefix="fiber-refinement-") as root:
            if state_source:
                import torch
                state = torch.load(state_source, map_location="cpu", weights_only=False)
                saved = state.get("refinement_state")
                if saved:
                    if cfg.state_sha256 and saved["sha256"] != cfg.state_sha256:
                        raise ValueError("configured frozen refinement SHA does not match source checkpoint")
                    GeometryStore(root).restore(state_source.parent / saved["file"], saved["sha256"])
                    if not cfg.update_enabled:
                        print(f"fiber refinement: frozen geometry restored source={state_source} "
                              f"archive={saved['file']} sha256={saved['sha256']}", flush=True)
                elif state.get("config", {}).get("refinement", {}).get("enabled"):
                    raise ValueError("refinement checkpoint is missing its fiber state")
                elif cfg.state_checkpoint:
                    raise ValueError("configured refinement state checkpoint has no fiber state")
            os.environ[_ENV] = root
            if not cfg.update_enabled:
                os.environ[_FROZEN_ENV] = "1"
            try:
                return function(config_path, **kwargs)
            finally:
                os.environ.pop(_FROZEN_ENV, None)
                os.environ.pop(_ENV, None)
    return wrapped


def active_store():
    if _ENV not in os.environ:
        raise RuntimeError("enabled refinement requires the training launcher (missing shared state)")
    return GeometryStore(os.environ[_ENV], read_only=os.environ.get(_FROZEN_ENV) == "1")


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


def direction_feedback(points, edges, material_lengths, transform, ids, maps, stride, cfg):
    """Frozen predicted base-space axes; exact endpoint derivative for that field.

    Decode sparse interpolated six-channel predictions, not interpolated signed axes.
    Full finite-difference warp Jacobian pulls tangent vectors back to base ZYX.
    Field/Jacobian are refreshed next iteration, not differentiated spatially.
    """
    from scipy.ndimage import map_coordinates
    from .direction import decode_lasagna_direction_3x2_analytic
    if maps is None:
        raise ValueError("direction refinement requires predicted direction channels")
    maps = np.asarray(maps)
    t = (np.arange(cfg.samples_per_segment)+.5)/cfg.samples_per_segment
    q = (points[edges,None]*(1-t)[None,:,None]+points[edges+1,None]*t[None,:,None]).reshape(-1,3)
    qids = np.repeat(ids[edges],len(t))
    mapped = transform(q,qids)
    coordinates = np.r_[qids[None],(mapped/stride).T]
    encoded = np.stack([map_coordinates(maps[:,c],coordinates,order=1,mode="nearest",prefilter=False)
                        for c in range(6)],axis=-1)
    axes = decode_lasagna_direction_3x2_analytic(encoded)[:,::-1].astype(np.float64)
    offsets = np.concatenate((np.eye(3),-np.eye(3)))*.5
    perturbed = transform((q[None]+offsets[:,None]).reshape(-1,3),np.tile(qids,6)).reshape(6,-1,3)
    jacobian = (perturbed[:3]-perturbed[3:]).transpose(1,2,0)
    if not np.isfinite(jacobian).all() or np.any(np.linalg.cond(jacobian)>1e8):
        raise ValueError("direction refinement encountered a singular/nonfinite patch Jacobian")
    axes = np.linalg.solve(jacobian,axes[...,None])[...,0]
    norms = np.linalg.norm(axes,axis=-1)
    valid = np.all((mapped/stride >= 0)&(mapped/stride <= np.array(maps.shape[2:])-1),axis=1)
    valid &= np.isfinite(axes).all(axis=1)&(norms>1e-10)
    axes = np.divide(axes,norms[:,None],out=np.zeros_like(axes),where=norms[:,None]>1e-10)
    axes = axes.reshape(len(edges),len(t),3)
    valid = valid.reshape(len(edges),len(t))
    displacement = points[edges+1]-points[edges]
    lengths = np.maximum(np.linalg.norm(displacement,axis=1),1e-12)
    tangent = displacement/lengths[:,None]
    cosine = np.sum(tangent[:,None]*axes,axis=-1)
    # Negative gradient with respect to the right endpoint; left is opposite.
    right_force = (2*cfg.direction_weight*material_lengths/lengths)[:,None]*np.mean(
        valid[:,:,None]*cosine[:,:,None]*(axes-cosine[:,:,None]*tangent[:,None]),axis=1)
    force = np.zeros_like(points)
    np.add.at(force,edges,-right_force)
    np.add.at(force,edges+1,right_force)
    mass = np.zeros(len(points))
    np.add.at(mass,edges,material_lengths/2)
    np.add.at(mass,edges+1,material_lengths/2)
    force /= np.maximum(mass[:,None],1e-12)
    losses = cfg.direction_weight*np.mean(valid*np.maximum(0.,1-cosine**2),axis=1)
    return force,losses


def refine_observations(observations, maps, directions=None):
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
    projectors = [normal_projectors(o.normals).reshape(-1,9) for o in observations]
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
        if cfg.direction_weight:
            extra, _ = direction_feedback(points,edges,weights,transform,ids,directions,observations[0].output_stride,cfg)
            force += extra
        normal_metric = np.concatenate([interpolate(p,o.current[:,3],c[:,3]).reshape(-1,3,3)
                                        for p,o,c in zip(projectors,observations,curves)])
        delta = semi_implicit_step(points, baseline, metric, force, cfg, links=links,
                                   material_lengths=np.diff(curve[:,3]),normal_metric=normal_metric)
        independent = np.array([o.independent_piece for o in observations], dtype=bool)[ids]
        limits = step_limits(transform(points, ids), observations[0].shape, cfg, independent, endpoint)
        delta = clamp_update(delta, points, metric, limits, tangent=points[following]-points[previous])
        updated = points+delta
        curves = [resample(updated[a:b], curve[a:b,3]) for a,b in zip(offsets[:-1], offsets[1:])]
    return curves


def refinement_losses(observations, curves, maps, directions=None):
    """Post-solve weighted losses per patch, each normalized by material length.

    Presence is 1-mean(probability), normal/tangent bending are weighted curvature
    energies, localization is the weighted ellipse displacement, direction is
    weighted sign-invariant tangent disagreement. All use material normalization.
    """
    from scipy.ndimage import map_coordinates
    if not observations:
        return []
    transform = BatchedPatchTransform(observations)
    queries, query_ids = [], []
    for i,(o,c) in enumerate(zip(observations,curves)):
        t = (np.arange(o.config.samples_per_segment)+.5)/o.config.samples_per_segment
        queries.append((c[:-1,None,:3]*(1-t)[None,:,None] + c[1:,None,:3]*t[None,:,None]).reshape(-1,3))
        query_ids.append(np.full(len(queries[-1]),i))
    ids = np.concatenate(query_ids)
    mapped = transform(np.concatenate(queries), ids) / observations[0].output_stride
    stack = np.stack(maps)
    values = map_coordinates(stack,np.concatenate((ids[None],mapped.T)),order=1,mode="nearest",prefilter=False)
    independent = np.array([o.independent_piece for o in observations])[ids]
    supported = np.all((mapped >= 0) & (mapped <= np.array(stack.shape[1:])-1),axis=1)
    values[independent & ~supported] = 0.
    losses, offset = [], 0
    for o,c,q in zip(observations,curves,queries):
        lengths = np.diff(c[:,3])
        length = float(lengths.sum())
        mass = (np.r_[lengths,0.] + np.r_[0.,lengths])/2
        presence = 1-float(np.sum(values[offset:offset+len(q)].reshape(len(lengths),-1).mean(axis=1)*lengths)/length)
        offset += len(q)
        baseline = interpolate(o.baseline[:,:3],o.baseline[:,3],c[:,3])
        metric = interpolate(ellipse_metric(o.normals,o.config).reshape(-1,9),o.current[:,3],c[:,3]).reshape(-1,3,3)
        delta = c[:,:3]-baseline
        localization = o.config.gt_weight*float(np.sum(mass*np.einsum("ni,nij,nj->n",delta,metric,delta))/length)
        projector = interpolate(normal_projectors(o.normals).reshape(-1,9),o.current[:,3],c[:,3]).reshape(-1,3,3)
        normal_weight = o.config.shape_weight if o.config.shape_normal_weight is None else o.config.shape_normal_weight
        tangent_weight = o.config.shape_weight if o.config.shape_tangent_weight is None else o.config.shape_tangent_weight
        normal_loss = normal_weight*bending_terms(c[:,:3],metric=projector)[-1]/length
        tangent_loss = tangent_weight*bending_terms(c[:,:3],metric=np.eye(3)-projector)[-1]/length
        direction_loss = 0.
        if o.config.direction_weight:
            _, edge_losses = direction_feedback(c[:,:3],np.arange(len(c)-1),lengths,transform,
                np.full(len(c),len(losses),dtype=int),directions,o.output_stride,o.config)
            direction_loss = float(np.sum(lengths*edge_losses)/length)
        losses.append((presence,normal_loss,tangent_loss,localization,direction_loss))
    return losses


def log_refinement_scalars(writer, phase, statistics, step):
    count, *sums = statistics
    if writer is None or count == 0:
        return
    means = dict(zip(REFINEMENT_LOSS_NAMES, np.asarray(sums)/count))
    means["total"] = sum(means.values())
    means["smoothing"] = means["smoothing_normal"]+means["smoothing_tangent"]
    for key, value in means.items():
        writer.add_scalar(f"refinement_{phase}/loss_{key}", float(value), step)


class RefinementObserver:
    def __init__(self, cfg):
        self.config = cfg
        self.pending = []
        self.seconds = 0.
        self.accepted = self.stale = 0

    def __call__(self, batch, output):
        if not self.config.enabled or not self.config.update_enabled:
            return
        from .model import presence_outputs, direction_outputs
        directions = None
        if self.config.direction_weight:
            branches = direction_outputs(output.detach())
            if branches.shape[1] != 1:
                raise ValueError("direction-guided refinement currently requires a single direction branch")
            directions = branches[:,0].float()
        # Retain only the small output probabilities, not the training graph.
        self.pending.append((batch.stream_indices.detach(), batch.refinement_geometry,
                             presence_outputs(output.detach()).amax(dim=1).squeeze(1).float(),directions))

    def flush(self, distributed, *, phase=None, step=None, writer=None, batch=None):
        if not self.config.enabled or not self.config.update_enabled:
            return
        import torch.distributed as dist
        started = time.perf_counter()
        proposals = []
        todo, images, directions = [], [], []
        for indices, observations, presence, predicted_directions in self.pending:
            indices = indices.cpu().tolist()
            maps = presence.cpu().numpy()
            direction_maps = predicted_directions.cpu().numpy() if predicted_directions is not None else [None]*len(indices)
            for index, observation, values, encoded in zip(indices, observations, maps, direction_maps):
                if observation is None:
                    continue
                todo.append((index, observation))
                images.append(values)
                directions.append(encoded)
        direction_maps = np.stack(directions) if directions and self.config.direction_weight else None
        afters = refine_observations([o for _,o in todo], images, direction_maps)
        losses = refinement_losses([o for _,o in todo], afters, images, direction_maps)
        for (index, observation), after, loss in zip(todo, afters, losses):
            proposals.append((index, observation.key, observation.start, observation.current, after, loss))
        self.pending.clear()
        if distributed.enabled:
            gathered = [None] * distributed.world_size if distributed.is_main else None
            dist.gather_object(proposals, gathered, dst=0)
        else:
            gathered = [proposals]
        statistics = empty_refinement_statistics()
        if distributed.is_main:
            store = active_store()
            applied_losses = []
            for _, key, start, before, after, loss in sorted((p for rows in gathered for p in rows), key=lambda p: p[0]):
                if store.commit(key, start, before, after):
                    self.accepted += 1
                    applied_losses.append(loss)
                else:
                    self.stale += 1
            if applied_losses:
                statistics = [float(len(applied_losses)), *np.sum(applied_losses,axis=0).tolist()]
            if phase is not None:
                if applied_losses:
                    averages = dict(zip(REFINEMENT_LOSS_NAMES,np.mean(applied_losses,axis=0)))
                    averages["total"] = sum(averages.values())
                    details = " ".join(f"loss_{k}={v:.7g}" for k,v in averages.items())
                else:
                    details = "losses=unavailable (no committed refinement proposals)"
                print(f"refinement_{phase} update={step} batch={batch if batch is not None else step} "
                      f"patches={len(applied_losses)} {details}",flush=True)
                log_refinement_scalars(writer,phase,statistics,step)
        # Do not let another rank start preparing a new test phase before all
        # proposals have been committed. Queue lookahead stays immutable.
        if distributed.enabled:
            dist.barrier()
        self.seconds += time.perf_counter() - started
        return statistics
