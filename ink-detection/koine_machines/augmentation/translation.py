import numpy as np


NORMAL_POOLED_TRANSLATION_PROBABILITY = 0.30
NORMAL_POOLED_TRANSLATION_MIN_VOXELS = 10
NORMAL_POOLED_TRANSLATION_MAX_VOXELS = 40
NORMAL_POOLED_TRANSLATION_KEEP_FRACTION = 1.0 / 3.0
NORMAL_POOLED_TRANSLATION_MAX_AXES = 2
NORMAL_POOLED_TRANSLATION_MAX_ATTEMPTS = 24


def _collect_constrained_native_points(flat_patch, patch_zyxs, valid_mask):
    constrained_mask = np.zeros(np.asarray(valid_mask).shape, dtype=bool)
    constrained_mask |= np.asarray(flat_patch) != 0
    if not np.any(constrained_mask):
        return np.empty((0, 3), dtype=np.int64)

    patch_zyxs = np.asarray(patch_zyxs)
    constrained_mask &= np.asarray(valid_mask, dtype=bool)
    constrained_mask &= np.isfinite(patch_zyxs).all(axis=-1)
    if not np.any(constrained_mask):
        return np.empty((0, 3), dtype=np.int64)
    return patch_zyxs[constrained_mask].astype(np.int64, copy=False)


def translate_crop_bbox(crop_bbox, translation_zyx):
    starts = np.asarray(crop_bbox[:3], dtype=np.int64) + np.asarray(translation_zyx, dtype=np.int64)
    stops = np.asarray(crop_bbox[3:], dtype=np.int64) + np.asarray(translation_zyx, dtype=np.int64)
    return tuple(int(v) for v in np.concatenate([starts, stops]))


def count_points_within_crop(points_zyx, crop_bbox):
    if int(np.asarray(points_zyx).shape[0]) == 0:
        return 0
    starts = np.asarray(crop_bbox[:3], dtype=np.int64)
    stops = np.asarray(crop_bbox[3:], dtype=np.int64)
    within = np.all((points_zyx >= starts) & (points_zyx < stops), axis=1)
    return int(within.sum())


def maybe_translate_normal_pooled_crop_bbox(
    crop_bbox,
    patch_zyxs,
    valid_mask,
    supervision_flat_patch,
    probability=NORMAL_POOLED_TRANSLATION_PROBABILITY,
    min_translation=NORMAL_POOLED_TRANSLATION_MIN_VOXELS,
    max_translation=NORMAL_POOLED_TRANSLATION_MAX_VOXELS,
    min_keep_fraction=NORMAL_POOLED_TRANSLATION_KEEP_FRACTION,
    max_axes=NORMAL_POOLED_TRANSLATION_MAX_AXES,
    max_attempts=NORMAL_POOLED_TRANSLATION_MAX_ATTEMPTS,
    rng=None,
):
    rng = np.random if rng is None else rng
    if probability <= 0 or rng.random() >= probability:
        return crop_bbox

    constrained_points = _collect_constrained_native_points(supervision_flat_patch, patch_zyxs, valid_mask)
    if constrained_points.shape[0] == 0:
        return crop_bbox

    min_points_to_keep = max(1, int(np.ceil(constrained_points.shape[0] * float(min_keep_fraction))))
    axes = np.array([1, 2], dtype=np.int64)
    max_axes = max(1, min(int(max_axes), axes.size))
    min_translation = int(min_translation)
    max_translation = int(max_translation)

    for _ in range(max(1, int(max_attempts))):
        selected_axes = axes.copy()
        rng.shuffle(selected_axes)
        num_axes = 1
        if max_axes >= 2 and rng.random() < 0.5:
            num_axes = 2

        translation = np.zeros(3, dtype=np.int64)
        for axis in selected_axes[:num_axes]:
            magnitude = int(rng.randint(min_translation, max_translation + 1))
            sign = -1 if rng.random() < 0.5 else 1
            translation[int(axis)] = sign * magnitude

        candidate_bbox = translate_crop_bbox(crop_bbox, translation)
        if count_points_within_crop(constrained_points, candidate_bbox) >= min_points_to_keep:
            return candidate_bbox

    return crop_bbox
