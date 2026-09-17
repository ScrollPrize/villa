"""Registered full-resolution/quarter-resolution patches and soft teacher targets."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from vesuvius.ink_detection.data.multiteacher import FlatDistillationDataset

FACTOR = 4
TEACHER_SHAPE = (64, 256, 256)
STUDENT_SHAPE = (16, 64, 64)


class FixedPreviewBase(FlatDistillationDataset):
    def locate(self, draw):
        if hasattr(self, 'preview_draws'):
            import numpy as np
            i,j=self.preview_draws[draw]
            return i,j,np.random.default_rng(draw)
        return super().locate(draw)


class ResolutionDataset(Dataset):
    """Reuse the audited split, full-depth reversal and spatial augmentation.

    Human labels are returned for evaluation only. The base loader's student
    intensity jitter is deliberately unused: both models see the same raw CT.
    """
    def __init__(self, data_config, teacher_normalization, *, validation=False):
        options = {k: data_config[k] for k in (
            'segment_depth_reversals', 'orientation_calibration_exclusions',
            'excluded_segments', 'path_roots') if k in data_config}
        self.base = FixedPreviewBase(
            data_config['manifest'], [teacher_normalization] * 2,
            validation=validation, augment=True, valid_depth_margin=1,
            student_normalization={'mode': 'none'},
            val_patches_per_scroll=data_config.get('val_patches_per_scroll', 128),
            **options)
        self.paired = None
        if data_config.get('paired_native'):
            from vesuvius.ink_detection.data.paired_native import PairedInputs
            self.base.augment = False
            self.paired = PairedInputs(data_config['paired_native'], validation)

        if data_config.get('training_preview_scroll'):
            if validation:raise ValueError('Training previews must not use validation masks')
            scroll=data_config['training_preview_scroll']
            self.base.augment=False
            if self.paired is None or self.paired.quality is None:
                raise ValueError('Training previews require paired native quality scores')
            self.paired.validation=True
            self.base.preview_draws=[(i,j) for i,r in enumerate(self.base.records) if r['scroll']==scroll
                for j,yx in enumerate(self.base.coordinates[i])
                if self.paired.quality.get(r['scroll']+'/'+r['segment'],{}).get(','.join(map(str,yx)),-1)>self.paired.recipe['minimum_correlation']]
            if not self.base.preview_draws:raise ValueError('No eligible training preview patches')

    def __len__(self):
        return len(self.base.preview_draws) if hasattr(self.base,'preview_draws') else len(self.base)

    def __getitem__(self, draw):
        sample = self.base[draw]
        result = {k: sample[k] for k in (
            'raw', 'teacher_image', 'valid_3d', 'labels_2d', 'mask_2d',
            'record_id', 'draw_id', 'reverse_depth', 'yx')}
        return self.paired.apply(result,self.base.records[int(sample['record_id'])],draw) if self.paired else result


def student_input(raw, factor=FACTOR):
    """Match released preprocessing: area reduction then /255, or native /200."""
    if tuple(raw.shape[-3:]) != TEACHER_SHAPE:
        raise ValueError(f'Expected raw teacher patch {TEACHER_SHAPE}')
    if factor not in (1,4):raise ValueError('Reduction factor must be one or four')
    return F.avg_pool3d(raw.float(), factor, factor) / 255.0 if factor>1 else raw.float()/200.0


def masked_area(probability, mask, dimensions, factor=FACTOR):
    """Keep valid-voxel means and the valid fraction of each output cell.

    The first/last teacher planes are artificial output padding. Excluding
    them avoids treating that padding as supervised non-ink. The outer low
    depth cells retain 3/4 support instead of discarding genuine predictions.
    """
    pool = F.avg_pool3d if dimensions == 3 else F.avg_pool2d
    weight = pool(mask.float(), factor, factor)
    target = pool(probability.float() * mask, factor, factor) / weight.clamp_min(1e-12)
    return target, weight


def foreground_mask(batch, records, thresholds):
    """Threshold the ORIGINAL raw CT, independently of student normalization."""
    cutoffs=batch['raw'].new_tensor([thresholds[records[i]['scroll']]
                                   for i in batch['record_id'].tolist()]).view(-1,1,1,1,1)
    return (batch['raw']>cutoffs) & batch['valid_3d'].bool()


@torch.no_grad()
def teacher_targets(output, valid, factor=FACTOR, foreground=None):
    p3=output['ink_3d_logits'].float().sigmoid()
    p2=output['ink'].float().sigmoid()
    if foreground is not None:
        # Background remains a valid NEGATIVE, not an ignored voxel. Apply
        # this before reduction, preserving subcell foreground occupancy.
        p3=p3*foreground
        p2=p2*foreground.any(2)
    volume, volume_weight = masked_area(p3, valid, 3, factor)
    projection, projection_weight = masked_area(p2, valid.any(2), 2, factor)
    return {'volume': volume, 'volume_weight': volume_weight,
            'projection': projection, 'projection_weight': projection_weight}


def weighted_mean_per_sample(value, weight):
    axes = tuple(range(1, value.ndim))
    return (value * weight).sum(axes) / weight.sum(axes).clamp_min(1)


def distillation_loss(output, targets, weight_2d=.5, weight_3d=.5):
    """Soft Bernoulli distillation; never threshold or normalize targets."""
    loss2 = weighted_mean_per_sample(F.binary_cross_entropy_with_logits(
        output['ink'].float(), targets['projection'], reduction='none'), targets['projection_weight'])
    loss3 = weighted_mean_per_sample(F.binary_cross_entropy_with_logits(
        output['ink_3d_logits'].float(), targets['volume'], reduction='none'), targets['volume_weight'])
    return (weight_2d*loss2 + weight_3d*loss3).mean(), {'bce_2d': loss2.detach(), 'bce_3d': loss3.detach()}
