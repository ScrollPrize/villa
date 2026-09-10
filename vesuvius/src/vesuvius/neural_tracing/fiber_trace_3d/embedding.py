"""Within-patch metric supervision; no cross-sample state or collectives."""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool = False
    channels: int = 16
    pairs_at_128: int = 65536
    weight: float = 1.0
    output_stride: int = 1

    @classmethod
    def from_mapping(cls, raw):
        values = dict(raw.get("embedding", {}))
        unknown = set(values) - {"enabled", "channels", "pairs_at_128", "weight"}
        if unknown:
            raise ValueError(f"unknown embedding settings: {sorted(unknown)}")
        model = raw.get("model_3d", raw.get("model", {}))
        result = cls(**values, output_stride=int(model.get("output_stride", 1)))
        if type(result.enabled) is not bool:
            raise ValueError("embedding.enabled must be a boolean")
        if type(result.channels) is not int or result.channels < 2:
            raise ValueError("embedding.channels must be an integer >=2")
        if type(result.pairs_at_128) is not int or result.pairs_at_128 < 2 or result.pairs_at_128 % 2:
            raise ValueError("embedding.pairs_at_128 must be positive and even")
        if not math.isfinite(result.weight) or result.weight < 0 or result.output_stride < 1:
            raise ValueError("invalid embedding weight/output stride")
        if result.enabled and model.get("conditioned_decoder_enabled", False):
            raise ValueError("embeddings currently require the unconditioned U-Net")
        return result

    def pairs(self, shape):
        return 2 * max(1, round(self.pairs_at_128 * max(shape) / 128 / 2))


@dataclass(frozen=True)
class EmbeddingGeometry:
    eligible: torch.Tensor | None  # None only in shared-slot IPC descriptors
    seed: int
    pairs: int
    stride: int
    weight: float


def _nearest_by_scan(points, line):
    """Exact Euclidean segment projection, O(points) temporary space.

    Caller bounds points per block. Ties select the earliest segment, matching
    a stable GT ordering. AABB culling only removes segments farther than 80
    base voxels; those cannot participate in the exclusion zone.
    """
    points = np.asarray(points, dtype=np.float64)
    line = np.asarray(line, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] != 3 or len(line) < 2 or not np.isfinite(line).all():
        raise ValueError("embedding GT must be a finite polyline with >=2 points")
    best = np.full(len(points), np.inf)
    nearest = np.zeros_like(points)
    lower, upper = points.min(axis=0) - 80, points.max(axis=0) + 80
    starts, ends = line[:-1], line[1:]
    intersects = np.all(np.maximum(starts, ends) >= lower, axis=1)
    intersects &= np.all(np.minimum(starts, ends) <= upper, axis=1)
    for start, end in zip(starts[intersects], ends[intersects]):
        delta = end - start
        length2 = float(delta @ delta)
        if length2 == 0:
            continue
        t = np.clip(np.sum((points - start) * delta, axis=1) / length2, 0, 1)
        matched = start + t[:, None] * delta
        distance2 = np.sum((points - matched) ** 2, axis=1)
        use = distance2 < best
        nearest[use] = matched[use]
        best[use] = distance2[use]
    return nearest, best


class _PolylineIndex:
    """Exact midpoint broad phase; never assume nearest midpoint = segment.

    An unseen segment cannot be closer than the nearest unseen midpoint minus
    the maximum half-segment length. Only that bound permits an early exit.
    Long/sparse/ambiguous segments fall back to the bounded linear scan.
    """
    def __init__(self, line):
        from scipy.spatial import cKDTree
        self.line = np.asarray(line, dtype=np.float64)
        if self.line.ndim != 2 or self.line.shape[1] != 3 or len(self.line) < 2 or not np.isfinite(self.line).all():
            raise ValueError("embedding GT must be a finite polyline with >=2 points")
        self.starts = self.line[:-1]
        self.delta = np.diff(self.line, axis=0)
        self.length2 = np.sum(self.delta ** 2, axis=1)
        valid = self.length2 > 0
        self.starts, self.delta, self.length2 = self.starts[valid], self.delta[valid], self.length2[valid]
        if not len(self.starts):
            raise ValueError("embedding GT has no nonzero segment")
        self.half_length = np.sqrt(self.length2.max()) / 2
        self.tree = cKDTree(self.starts + self.delta / 2)

    def nearest(self, points):
        points = np.asarray(points, dtype=np.float64)
        result = np.zeros_like(points)
        best = np.full(len(points), np.inf)
        pending = np.arange(len(points))
        k = min(8, len(self.starts))
        while len(pending):
            distances, indices = self.tree.query(points[pending], k=k, workers=1)
            distances, indices = distances.reshape(-1, k), indices.reshape(-1, k)
            start, delta = self.starts[indices], self.delta[indices]
            t = np.clip(np.sum((points[pending, None] - start) * delta, axis=2) / self.length2[indices], 0, 1)
            matched = start + t[..., None] * delta
            distance2 = np.sum((points[pending, None] - matched) ** 2, axis=2)
            minimum = distance2.min(axis=1)
            # Stable first-segment tie break, not cKDTree's arbitrary tie order.
            first = np.where(distance2 == minimum[:, None], indices, len(self.starts)).min(axis=1)
            choice = (indices == first[:, None]).argmax(axis=1)
            result[pending] = matched[np.arange(len(pending)), choice]
            best[pending] = minimum
            if k == len(self.starts):
                break
            resolved = distances[:, -1] - self.half_length > np.sqrt(minimum) + 1e-10
            pending = pending[~resolved]
            if k >= 64:
                if len(pending):
                    result[pending], best[pending] = _nearest_by_scan(points[pending], self.line)
                break
            k = min(k * 2, len(self.starts))
        return result, best


def nearest_polyline_points(points, line):
    return _PolylineIndex(line).nearest(points)


def outside_normal_ellipse(delta_base, normals_zyx):
    """Native FiberReplayMetric formula, inclusive 20/80 base-voxel exclusion.

    Training requires valid normals, unlike replay's invalid-normal fallback.
    """
    normals = np.asarray(normals_zyx, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= 1e-12):
        raise ValueError("embedding exclusion requires finite nonzero Lasagna normals")
    normals = normals / lengths[:, None]
    delta = np.asarray(delta_base, dtype=np.float64)
    dn2 = np.sum(delta * normals, axis=1) ** 2
    dt2 = np.maximum(0, np.sum(delta * delta, axis=1) - dn2)
    return dn2 + dt2 / 16 > 400


def build_eligibility(shape, stride, to_base, line_base, sample_normals, base_shape):
    """Map output lattice through the *existing* inverse augmentation map.

    Normals are sampled at each Euclidean-nearest GT point, never at the
    candidate. Blocks bound both projection and normal-sampling temporary memory.
    """
    shape = tuple((int(v) + stride - 1) // stride for v in shape)
    flat = np.empty(math.prod(shape), dtype=bool)
    index = _PolylineIndex(line_base)
    for begin in range(0, len(flat), 4096):
        stop = min(begin + 4096, len(flat))
        indices = np.arange(begin, stop)
        points = np.column_stack(np.unravel_index(indices, shape)) * stride
        source = np.asarray(to_base(points), dtype=np.float64)
        valid = np.isfinite(source).all(axis=1) & np.all(source >= 0, axis=1)
        valid &= np.all(source <= np.asarray(base_shape) - 1, axis=1)
        if not np.isfinite(source).all():
            raise ValueError("nonfinite embedding inverse augmentation coordinates")
        nearest, distance2 = index.nearest(source)
        close = valid & (distance2 <= 80 ** 2)
        eligible = valid.copy()
        if close.any():
            eligible[close] = outside_normal_ellipse(source[close] - nearest[close], sample_normals(nearest[close]))
        flat[begin:stop] = eligible
    return torch.from_numpy(flat.reshape(shape))


def sample_trace_points(starts, ends, count, generator, shape):
    """Arc-length sampling of continuous segments clipped to the output lattice."""
    delta = ends - starts
    low = torch.zeros(3, device=starts.device)
    high = torch.tensor(shape, device=starts.device) - 1
    parallel = delta.abs() < 1e-12
    safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
    a, b = (low - starts) / safe_delta, (high - starts) / safe_delta
    enter = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(dim=1).clamp_min(0)
    leave = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(dim=1).clamp_max(1)
    valid = (leave > enter) & ~((parallel & ((starts < low) | (starts > high))).any(dim=1))
    starts, delta, enter, leave = starts[valid], delta[valid], enter[valid], leave[valid]
    lengths = delta.norm(dim=1) * (leave - enter)
    if lengths.numel() == 0 or not bool(lengths.sum() > 0):
        raise ValueError("embedding patch contains no nonzero GT trace segment")
    choices = torch.multinomial(lengths, count, replacement=True, generator=generator)
    t = enter[choices] + torch.rand(count, device=starts.device, generator=generator) * (leave - enter)[choices]
    return starts[choices] + t[:, None] * delta[choices]


def sample_embeddings(embedding, points):
    shape = torch.tensor(embedding.shape[-3:], device=points.device)
    coords = (2 * points / (shape - 1).clamp_min(1) - 1).flip(-1)
    sampled = F.grid_sample(embedding[None].float(), coords.reshape(1, 1, 1, -1, 3), align_corners=True)
    return F.normalize(sampled.reshape(embedding.shape[0], -1).T, dim=1, eps=1e-8)


def embedding_statistics(embeddings, presence, batch):
    """Separate component sums/counts; sampling never backpropagates to presence."""
    zero = embeddings.reshape(-1)[0].float() * 0
    pos_sum, neg_sum = zero, zero
    pos_count, neg_count = zero.detach(), zero.detach()
    if batch.embedding_geometry is None or len(batch.embedding_geometry) != len(embeddings):
        raise ValueError("embedding-enabled model requires loader embedding geometry/normals")
    for i, geometry in enumerate(batch.embedding_geometry):
        if geometry is None or tuple(geometry.eligible.shape) != tuple(embeddings.shape[-3:]):
            raise ValueError("missing or mismatched embedding output-grid geometry")
        # Loader stream seeds are 128-bit hashes; PyTorch accepts 64-bit seeds.
        # Match the existing value-augmentation seed conversion.
        rng = torch.Generator(device=embeddings.device).manual_seed(geometry.seed % (2**63 - 1))
        half = geometry.pairs // 2
        offset, count = int(batch.target_segment_offsets[i]), int(batch.target_segment_counts[i])
        starts = batch.target_segment_starts_zyx[offset:offset + count].float() / geometry.stride
        ends = batch.target_segment_ends_zyx[offset:offset + count].float() / geometry.stride
        points = sample_trace_points(starts, ends, half * 3, rng, embeddings.shape[-3:])
        on_trace = sample_embeddings(embeddings[i], points)
        positive = 1 - (on_trace[:half] * on_trace[half:2 * half]).sum(dim=1).clamp(-1, 1)
        pos_sum = pos_sum + positive.sum() * geometry.weight
        pos_count = pos_count + half
        weights = presence[i].detach().float().reshape(-1)
        if not bool(torch.isfinite(weights).all()) or bool(((weights < 0) | (weights > 1)).any()):
            raise ValueError("embedding sampling requires finite presence probabilities in [0,1]")
        weights = weights * geometry.eligible.to(weights.device).reshape(-1)
        if not bool(weights.sum() > 0):
            continue
        chosen = torch.multinomial(weights, half, replacement=True, generator=rng)
        # Integer output-grid negatives need no interpolation.
        negatives = F.normalize(embeddings[i].reshape(embeddings.shape[1], -1)[:, chosen].T.float(), dim=1, eps=1e-8)
        negative = (on_trace[2 * half:] * negatives).sum(dim=1).clamp_min(0)
        neg_sum = neg_sum + negative.sum() * geometry.weight
        neg_count = neg_count + half
    return pos_sum, pos_count, neg_sum, neg_count
