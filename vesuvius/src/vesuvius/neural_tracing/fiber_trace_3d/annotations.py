"""Optional line-annotation targets. No remote reads or dense target volumes."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F
from .direction import encode_lasagna_direction_3x2, projection_magnitude_weights_3x2


@dataclass(frozen=True)
class AnnotationConfig:
    normal_weight: float = 0.0
    width_weight: float = 0.0
    hv_weight: float = 0.0

    @classmethod
    def from_mapping(cls, raw):
        values = raw.get("annotations", {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown annotation settings: {sorted(unknown)}")
        result = cls(**{k: float(v) for k, v in values.items()})
        if any(not math.isfinite(v) or v < 0 for v in vars(result).values()):
            raise ValueError("annotation loss weights must be finite and nonnegative")
        return result

    @property
    def names(self):
        return tuple(k for k in ("normal", "width", "hv") if getattr(self, k + "_weight") > 0)

    @property
    def channels(self):
        return sum(6 if k == "normal" else 1 for k in self.names)


def baked_normals(fiber):
    value = fiber.metadata.get("baked_annotations")
    if value is None:
        return None
    axes = np.asarray(value.get("normals_xyz"), dtype=np.float64)
    if value.get("version") != 1 or axes.shape != fiber.line_points_xyz.shape:
        raise ValueError(f"invalid/misaligned baked normal schema: {fiber.path}")
    lengths = np.linalg.norm(axes, axis=1)
    valid = np.asarray(value.get('valid', np.ones(len(axes), dtype=bool)), dtype=bool)
    if valid.shape != (len(axes),):
        raise ValueError('misaligned baked normal validity')
    if not np.isfinite(axes).all() or np.any(np.abs(lengths[valid] - 1) > 1e-4) or np.any(lengths[~valid] != 0):
        raise ValueError(f"nonunit/nonfinite baked normals: {fiber.path}")
    # Axis ambiguity: never interpolate opposite signs through zero.
    axes = axes.copy()
    signs = np.where(np.sum(axes[1:] * axes[:-1], axis=1) < 0, -1, 1)
    axes[1:] *= np.cumprod(signs)[:, None]
    return axes[:, ::-1]


def line_targets(loader, record, params):
    """Rows: output ZYX, normal ZYX, full width, H/V. Zero normal/width and
    H/V=-1 mean unannotated, not negative targets. One sample/input voxel.
    """
    cfg = loader.config.annotations
    if not cfg.names:
        return None
    fiber = record.fiber
    points = fiber.line_points_zyx.astype(np.float64) / record.volume_spacing_base
    axes = baked_normals(fiber)
    width = float(fiber.metadata.get("width", 0)) / record.volume_spacing_base
    if width > 0 and axes is None and (cfg.normal_weight or cfg.width_weight):
        raise ValueError(f"width-annotated fiber needs baked normals: {fiber.path}")
    label = fiber.metadata.get("hv_classification", {})
    hv = {"H": 0., "V": 1.}.get(label.get("manual_tag") or label.get("automatic_tag"), -1.)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    if arc[-1] <= 0:
        return np.empty((0, 8), dtype=np.float32)
    # Restrict source segments before resampling long fibers. Retain disjoint
    # pieces separately, never join them across an out-of-patch gap.
    mapped, _ = loader._source_points_to_output_np(points, params, array_backend=True)
    shape = np.asarray(loader.config.patch_shape_zyx)
    keep = (lengths > 0) & np.all(np.maximum(mapped[:-1], mapped[1:]) >= -1, axis=1)
    keep &= np.all(np.minimum(mapped[:-1], mapped[1:]) < shape, axis=1)
    positions = []
    for i in np.flatnonzero(keep):
        distances = np.arange(math.ceil(arc[i]), arc[i+1], dtype=np.float64)
        positions.extend(distances.tolist())
    if not positions:
        return np.empty((0, 8), dtype=np.float32)
    a = np.asarray(positions)
    source = np.stack([np.interp(a, arc, points[:, k]) for k in range(3)], axis=1)
    out, valid = loader._source_points_to_output_np(source, params, array_backend=True)
    valid &= np.all((out >= 0) & (out <= shape - 1), axis=1)
    source, out, a = source[valid], out[valid], a[valid]
    normal = np.zeros_like(out)
    widths = np.zeros(len(out))
    if axes is not None and width > 0 and len(out):
        n = np.stack([np.interp(a, arc, axes[:, k]) for k in range(3)], axis=1)
        idx = np.clip(np.searchsorted(arc, a, side="right") - 1, 0, len(lengths)-1)
        annotated = (np.linalg.norm(axes[idx], axis=1) > 0) & (np.linalg.norm(axes[idx+1], axis=1) > 0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        tangent = points[idx+1] - points[idx]
        original_widths = fiber.metadata['baked_annotations'].get('widths_base')
        local_width = width if original_widths is None else np.interp(a, arc, original_widths) / record.volume_spacing_base
        normal, widths = transform_annotation_geometry(loader, params, source, n, tangent, local_width, annotated)
    return np.column_stack((out, normal, widths, np.full(len(out), hv))).astype(np.float32)


def transform_annotation_geometry(loader, params, source, normals, tangents, widths, valid):
    """Shared point/segment annotation transformation through augmentation."""
    jac = loader._source_to_output_jacobian(source, params)
    normal = np.linalg.solve(jac.transpose(0, 2, 1), normals[..., None])[..., 0]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    transverse = np.cross(tangents, normals)
    norm = np.linalg.norm(transverse, axis=1, keepdims=True)
    if np.any(norm[valid] <= 1e-8):
        raise ValueError("baked normal parallel to fiber tangent")
    transverse /= np.maximum(norm, 1e-12)
    widths = widths * np.linalg.norm((jac @ transverse[..., None])[..., 0], axis=1)
    widths[~valid], normal[~valid] = 0., 0.
    return normal, widths


def segment_endpoint_widths(loader, record, params, indices, fractions):
    """Widths of clipped segments, indexed into the original labelled polyline.

    Fractions carry clipping provenance, including the halo outside the crop.
    No nearest-neighbour reassignment between geometrically close segments.
    """
    fiber = record.fiber
    result = np.zeros((len(indices), 2), dtype=np.float32)
    axes = baked_normals(fiber)
    if axes is None or float(fiber.metadata.get('width', 0)) <= 0:
        return result
    idx = np.repeat(np.asarray(indices, dtype=np.int64), 2)
    fraction = np.asarray(fractions, dtype=np.float64).reshape(-1, 1)
    points = fiber.line_points_zyx.astype(np.float64) / record.volume_spacing_base
    source = points[idx] * (1-fraction) + points[idx+1] * fraction
    n = axes[idx] * (1-fraction) + axes[idx+1] * fraction
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    base_widths = np.asarray(fiber.metadata['baked_annotations'].get(
        'widths_base', np.full(len(axes), fiber.metadata['width'])))
    valid = ((np.linalg.norm(axes[idx], axis=1) > 0) & (np.linalg.norm(axes[idx+1], axis=1) > 0)
             & (base_widths[idx] > 0) & (base_widths[idx+1] > 0))
    width = (base_widths[idx] * (1-fraction[:,0]) + base_widths[idx+1] * fraction[:,0]) / record.volume_spacing_base
    _, width = transform_annotation_geometry(loader, params, source, n, points[idx+1]-points[idx], width, valid)
    return width.reshape(-1, 2).astype(np.float32)


def sample_baked_normals(fiber, points_zyx):
    from .embedding import _PolylineIndex
    axes = baked_normals(fiber)
    if axes is None:
        return None
    _, distance, index, fraction = _PolylineIndex(fiber.line_points_zyx).nearest(points_zyx, with_location=True)
    if np.any(np.linalg.norm(axes[index], axis=1) == 0) or np.any(np.linalg.norm(axes[index+1], axis=1) == 0):
        from .embedding import InvalidNormalSampleError
        raise InvalidNormalSampleError('invalid baked normal at selected fiber location')
    if not np.isfinite(distance).all():
        raise ValueError('cannot resolve baked normal along selected fiber')
    normal = axes[index] * (1-fraction[:, None]) + axes[index+1] * fraction[:, None]
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)


def resample_baked_metadata(fiber, old_arc, new_arc):
    """Carry annotations with refinement's persistent material coordinate."""
    axes = baked_normals(fiber)
    if axes is None:
        return fiber.metadata
    normal = np.stack([np.interp(new_arc, old_arc, axes[:, k]) for k in range(3)], axis=1)
    index = np.clip(np.searchsorted(old_arc, new_arc, side='right') - 1, 0, len(axes)-2)
    valid = (np.linalg.norm(axes[index], axis=1) > 0) & (np.linalg.norm(axes[index+1], axis=1) > 0)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    normal[~valid] = 0
    baked = {**fiber.metadata['baked_annotations'], 'normals_xyz': normal[:, ::-1].tolist(), 'valid': valid.tolist()}
    if 'widths_base' in baked:
        baked['widths_base'] = np.interp(new_arc, old_arc, baked['widths_base']).tolist()
    return {**fiber.metadata, 'baked_annotations': baked}


def transform_baked_metadata(fiber, matrix_xyz):
    axes = baked_normals(fiber)
    if axes is None:
        return fiber.metadata
    matrix = np.asarray(matrix_xyz)[:3, :3]
    normal = axes[:, ::-1]
    valid = np.linalg.norm(normal, axis=1) > 0
    mapped = np.linalg.solve(matrix.T, normal.T).T
    mapped /= np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1e-12)
    tangent = np.gradient(fiber.line_points_xyz.astype(np.float64), axis=0)
    transverse = np.cross(tangent, normal)
    length = np.linalg.norm(transverse, axis=1, keepdims=True)
    if np.any(length[valid] <= 1e-12):
        raise ValueError('annotated normal parallel to fiber during dataset transform')
    transverse /= np.maximum(length, 1e-12)
    stretch = np.linalg.norm(transverse @ matrix.T, axis=1)
    baked = fiber.metadata['baked_annotations']
    width = np.asarray(baked.get('widths_base', np.full(len(axes), fiber.metadata.get('width', 0))))
    return {**fiber.metadata, 'baked_annotations': {**baked,
        'normals_xyz': mapped.tolist(), 'valid': valid.tolist(), 'widths_base': (width * stretch).tolist()}}


def normal_tube_targets(batch):
    """Use the direction rasterizer's winning segment and exact GT falloff."""
    indices, axes = batch.direction_indices_bzyx, batch.normal_axis_sparse_zyx
    if indices is None or axes is None or batch.presence_target is None:
        raise ValueError('normal tube supervision requires materialized normal targets')
    labelled = torch.tensor([
        rows is not None and len(rows) > 0 and bool(np.any(np.linalg.norm(rows[:, 3:6], axis=1) > 0))
        for rows in (batch.annotation_targets or (None,) * len(batch.volume))
    ], device=indices.device, dtype=torch.bool)
    valid = labelled[indices[:, 0]] & (torch.linalg.vector_norm(axes, dim=1) > 0)
    indices, axes = indices[valid], axes[valid]
    falloff = batch.presence_target[indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]]
    return indices, axes.flip(1), falloff


def predicted_width(output, cfg):
    """Full width in network-input voxels; independent of output stride."""
    if 'width' not in cfg.names:
        return None
    channel = 6 if 'normal' in cfg.names else 0
    return F.softplus(output[:, channel:channel+1].float())


def width_similarity(width, reference):
    """Linear match score, zero at20% deviation or invalid reference width."""
    valid = torch.isfinite(reference) & (reference > 0) & torch.isfinite(width)
    score = (1 - (width - reference).abs() / (.2 * reference.clamp_min(1e-12))).clamp(0, 1)
    return torch.where(valid, score, torch.zeros_like(score))


def width_tube_targets(batch):
    if batch.width_target_input is None or batch.direction_indices_bzyx is None:
        raise ValueError('width tube supervision requires materialized width targets')
    indices = batch.direction_indices_bzyx
    widths = batch.width_target_input[indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]]
    valid = widths > 0
    indices, widths = indices[valid], widths[valid]
    falloff = batch.presence_target[indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]]
    return indices, widths, falloff


def annotation_counts(batch, cfg):
    counts = []
    for name in cfg.names:
        if name == 'normal' and batch.presence_target_mode == 'anisotropic_gaussian':
            counts.append(normal_tube_targets(batch)[2].sum() * 6)
            continue
        if name == 'width' and batch.presence_target_mode == 'anisotropic_gaussian':
            counts.append(width_tube_targets(batch)[2].sum())
            continue
        count = 0
        for rows in batch.annotation_targets or ():
            if rows is not None and len(rows):
                col = np.linalg.norm(rows[:, 3:6], axis=1) if name == "normal" else rows[:, 6 if name == "width" else 7]
                count += int(np.count_nonzero(col >= 0 if name == "hv" else col > 0)) * (6 if name == "normal" else 1)
        counts.append(count)
    return counts


def annotation_statistics(output, batch, cfg):
    """Elementwise sums/counts, like direction loss, with no patch balancing.
    Output is raw logits; positions use grid_sample's voxel-center convention.
    """
    result, channel = {}, 0
    for name in cfg.names:
        size = 6 if name == "normal" else 1
        total, count = output.sum() * 0., output.new_zeros((), dtype=torch.float32)
        if name == 'width' and batch.presence_target_mode == 'anisotropic_gaussian':
            indices, widths, falloff = width_tube_targets(batch)
            pred = predicted_width(output, cfg)[:, 0][tuple(indices.T)]
            total = (F.smooth_l1_loss(pred, widths, reduction='none') * falloff).sum()
            result.update(width_sum=total * cfg.width_weight, width_count=falloff.sum(),
                          width_unweighted_sum=total)
            channel += size
            continue
        if name == 'normal' and batch.presence_target_mode == 'anisotropic_gaussian':
            indices, axes, falloff = normal_tube_targets(batch)
            pred = output[:, channel:channel+6].float().sigmoid().permute(0, 2, 3, 4, 1)
            pred = pred[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
            error = (pred - encode_lasagna_direction_3x2(axes)).square()
            total = (error * projection_magnitude_weights_3x2(axes) * falloff[:, None]).sum()
            count = falloff.sum() * 6
            result.update(normal_sum=total * cfg.normal_weight, normal_count=count,
                          normal_unweighted_sum=total)
            channel += size
            continue
        for b, rows in enumerate(batch.annotation_targets or ()):
            if rows is None or not len(rows):
                continue
            target = torch.as_tensor(rows, device=output.device, dtype=torch.float32)
            mask = torch.linalg.vector_norm(target[:, 3:6], dim=1) > 0 if name == "normal" else (
                target[:, 6] > 0 if name == "width" else target[:, 7] >= 0)
            # Mask is GT-only. Empty samples carry zero weight.
            if not bool(mask.any()):
                continue
            target = target[mask]
            shape = target.new_tensor(batch.volume.shape[-3:])
            grid = (2 * (target[:, :3] + .5) / shape - 1).flip(1).reshape(1, 1, 1, -1, 3)
            field = output[b:b+1, channel:channel+size].float()
            if name == 'normal':
                field = field.sigmoid()
            pred = F.grid_sample(field, grid,
                                 align_corners=False, padding_mode="border").reshape(size, -1).T
            if name == "normal":
                encoded = encode_lasagna_direction_3x2(target[:, 3:6].flip(1))
                loss = ((pred - encoded).square() * projection_magnitude_weights_3x2(target[:, 3:6].flip(1))).sum()
            elif name == "width":
                loss = F.smooth_l1_loss(F.softplus(pred[:, 0]), target[:, 6], reduction="sum")
            else:
                loss = F.binary_cross_entropy_with_logits(pred[:, 0], target[:, 7], reduction="sum")
            total = total + loss
            count = count + target.shape[0] * size
        result[name + "_sum"], result[name + "_count"] = total * getattr(cfg, name + "_weight"), count
        result[name + '_unweighted_sum'] = total
        channel += size
    return result
