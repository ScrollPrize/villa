"""JSON-only linking policy for checkpoint winding analysis."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
import threading

import numpy as np

from point_collection import (
    PatchLinkOptions, SIDE_BEHIND, SIDE_FRONT, SampledLinkPatch,
    _resolve_between_patches_targets, link_points_to_patches, sample_patches_for_linking,
)
from spiral_helpers import fiber_collection_hv_tag


@dataclass(frozen=True)
class AnalysisLinkingConfig:
    distance: float | None = None  # resolved from the checkpoint before loading
    window_points: int = 5
    min_points: int = 1
    vertical_max_behind_fraction: float | None = None
    horizontal_max_front_fraction: float | None = None
    side_margin_voxels: float = 0.5
    selection: str = 'support_then_distance'
    fiber_stride: int = 1
    patch_stride: int = 1

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, dict):
            raise ValueError('linking must be a JSON object')
        settings = {}
        for key, setting in value.items():
            name = key.replace('-', '_')
            if name not in cls.__dataclass_fields__:
                raise ValueError(f'unknown linking option {key!r}')
            if name in settings:
                raise ValueError(f'duplicate linking option {key!r}')
            settings[name] = setting
        for key in ('window_points', 'min_points', 'fiber_stride', 'patch_stride'):
            if key in settings and (type(settings[key]) is not int or settings[key] < 1):
                raise ValueError(f'linking.{key} must be a positive integer')
        for key in ('distance', 'side_margin_voxels', 'vertical_max_behind_fraction',
                    'horizontal_max_front_fraction'):
            if key not in settings:
                continue
            val = settings[key]
            fraction = key.endswith('_fraction')
            if fraction and val is None:
                continue
            if (type(val) not in (int, float) or not math.isfinite(val)
                    or val < 0 or (key == 'distance' and val == 0)
                    or (fraction and val > 1)):
                domain = '[0, 1] or null' if fraction else ('> 0' if key == 'distance' else '>= 0')
                raise ValueError(f'linking.{key} must be a finite number {domain}')
            settings[key] = float(val)
        config = cls(**settings)
        if config.window_points % 2 != 1:
            raise ValueError('linking.window_points must be odd')
        if config.min_points > config.window_points:
            raise ValueError('linking.min_points must not exceed window_points')
        if config.selection not in ('support_then_distance', 'nearest', 'largest_area'):
            raise ValueError('linking.selection must be support_then_distance, nearest, or largest_area')
        return config

    def resolve_distance(self, checkpoint_distance):
        distance = self.distance if self.distance is not None else float(checkpoint_distance)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError('linking distance inherited from checkpoint must be finite and > 0')
        return replace(self, distance=distance)


def sampled_fiber_view(pcl, stride, forced_controls=()):
    """Keep point objects/IDs and the original chain intact; sample only queries."""
    items = sorted(pcl['points'].items(), key=lambda item: int(item[0]))
    if not items:
        return dict(pcl, points={})
    positions = set(range(0, len(items), stride)) | {0, len(items) - 1}
    controls = set(forced_controls) | {b['local_index'] for b in pcl.get('branches', ())}
    control_indices = pcl.get('control_line_indices', ())
    original_indices = {int(control_indices[c]) for c in controls if 0 <= c < len(control_indices)}
    positions.update(i for i, (_, point) in enumerate(items)
                     if point.get('orig_index') in original_indices)
    return dict(pcl, points={items[i][0]: items[i][1] for i in sorted(positions)})


class AnalysisPointLinker:
    """Host-loader callback, run before annotations and fiber components derive."""

    def __init__(self, config, inward_direction, *, min_z_fraction=0.8,
                 min_auto_certainty=0.5, annotation_options=None):
        self.config = config
        self.inward_direction = inward_direction
        self.min_z_fraction = min_z_fraction
        self.min_auto_certainty = min_auto_certainty
        self.annotation_options = annotation_options or PatchLinkOptions()
        self.counts = {'fibers': Counter(), 'annotations': Counter()}
        self._lock = threading.Lock()

    def _record(self, group):
        def record(counts):
            with self._lock:
                self.counts[group].update(counts)
        return record

    @property
    def report(self):
        return dict(settings=asdict(self.config),
                    counts={group: dict(counts) for group, counts in self.counts.items()})

    def __call__(self, patches, collections, fibers):
        cfg = self.config
        sampled_patches = sample_patches_for_linking(patches, cfg.patch_stride)
        # Also preserve the target of one-sided branch records.
        forced = {}
        for fiber in fibers.values():
            for branch in fiber.get('branches', ()):
                forced.setdefault(branch['branch_file'], set()).add(branch['branch_index'])
        sampled_fibers = {
            cid: sampled_fiber_view(pcl, cfg.fiber_stride, forced.get(pcl.get('file_basename'), ()))
            for cid, pcl in fibers.items()
        }
        regular = {cid: pcl for cid, pcl in collections.items() if cid not in fibers}
        for group, original, sampled in (('fibers', fibers, sampled_fibers),
                                          ('annotations', regular, regular)):
            self.counts[group].update(
                source_points=sum(len(p['points']) for p in original.values()),
                sampled_points=sum(len(p['points']) for p in sampled.values()),
                candidate_hits=0, undefined_side_hits=0, side_rejected_candidates=0,
                support_rejected_candidates=0, attached_points=0)

        def link(inputs, options, selection, geometry=sampled_patches):
            return link_points_to_patches(
                geometry, inputs, tolerance=cfg.distance,
                surface_index_tolerance=cfg.distance,
                general_hit_policy=selection, options=options)

        annotation_options = replace(self.annotation_options, report_counts=self._record('annotations'))
        # Resolve pairs against the original set: if coarse sampling removes
        # one/both named patches, never silently fall back to unrelated patches.
        general = {}
        for cid, pcl in regular.items():
            targets = _resolve_between_patches_targets(pcl, patches)
            if targets is None or all(pid in sampled_patches for pid in targets):
                general[cid] = pcl
            else:
                subset = {pid: sampled_patches[pid] for pid in targets if pid in sampled_patches}
                link({cid: pcl}, annotation_options, 'nearest', subset)
        if general:
            link(general, annotation_options, 'largest_area')

        rules, fractions = {}, {}
        for cid, pcl in fibers.items():
            tag = fiber_collection_hv_tag(pcl, min_z_fraction=self.min_z_fraction,
                                          min_auto_certainty=self.min_auto_certainty)
            fraction = (cfg.vertical_max_behind_fraction if tag == 'V'
                        else cfg.horizontal_max_front_fraction if tag == 'H' else None)
            if fraction is not None:
                rules[cid] = SIDE_FRONT if tag == 'V' else SIDE_BEHIND
                fractions[cid] = fraction
        options = PatchLinkOptions(
            window_points=cfg.window_points, window_min_points=cfg.min_points,
            strict_min_points=True, rank_by_support=cfg.selection == 'support_then_distance',
            side_rules=rules, max_wrong_side_fractions=fractions,
            inward_direction=self.inward_direction, side_margin=cfg.side_margin_voxels,
            report_counts=self._record('fibers'))
        if sampled_fibers:
            link(sampled_fibers, options, cfg.selection)

        attachments_by_patch = {}
        for pcl in collections.values():
            for point in pcl['points'].values():
                attachment = point.get('on_patch')
                if attachment is None:
                    continue
                geometry = sampled_patches[attachment['id']]
                if isinstance(geometry, SampledLinkPatch):
                    attachments_by_patch.setdefault(attachment['id'], []).append(attachment)
        for pid, attachments in attachments_by_patch.items():
            mapped = sampled_patches[pid].original_ij([a['ij'] for a in attachments]).tolist()
            for attachment, ij in zip(attachments, mapped):
                attachment['ij'] = ij
        for counts in self.counts.values():
            counts['unattached_sampled_points'] = counts['sampled_points'] - counts['attached_points']


def model_inward_direction(transform, dr, device, model_z_begin, model_z_end):
    """Bounded model queries; serialize device work across CPU linking workers."""
    from fit_spiral import inward_winding_direction

    lock = threading.Lock()

    def inward(zyxs):
        zyxs = np.asarray(zyxs)
        valid = np.isfinite(zyxs).all(axis=1) & (zyxs[:, 0] >= model_z_begin) & (zyxs[:, 0] < model_z_end)
        result = np.zeros(zyxs.shape, dtype=np.float64)
        if valid.any():
            with lock:
                result[valid] = inward_winding_direction(
                    transform, zyxs[valid], device=device, dr_per_winding=dr)
        return result
    return inward
