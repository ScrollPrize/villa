import numpy as np
import torch
from contextlib import nullcontext


TTA_FLIP_COMBOS = [
    [],
    [-1],
    [-2],
    [-3],
    [-1, -2],
    [-1, -3],
    [-2, -3],
    [-1, -2, -3],
]
TTA_ROTATE3_PERMS = [
    (0, 1, 2),  # z-up
    (2, 0, 1),  # x-up
    (1, 2, 0),  # y-up
]

# Mapping from flip dim to displacement channel that must be negated:
# dim -1 (W/X) -> channel 2, dim -2 (H/Y) -> channel 1, dim -3 (D/Z) -> channel 0
FLIP_DIM_TO_CHANNEL = {-1: 2, -2: 1, -3: 0}
TTA_MERGE_METHODS = ("median", "mean", "trimmed_mean", "vector_medoid", "vector_geomedian")
TTA_TRANSFORM_MODES = ("mirror", "rotate3")


def _profile_section(profiler, name):
    if profiler is None:
        return nullcontext()
    return profiler.section(name)


def _validate_tta_merge_method(merge_method):
    method = str(merge_method).strip().lower()
    if method not in TTA_MERGE_METHODS:
        raise ValueError(
            f"Unknown --tta-merge-method '{merge_method}'. "
            f"Supported methods: {list(TTA_MERGE_METHODS)}"
        )
    return method


def _validate_tta_transform_mode(transform_mode):
    mode = str(transform_mode).strip().lower()
    if mode not in TTA_TRANSFORM_MODES:
        raise ValueError(
            f"Unknown --tta-transform '{transform_mode}'. "
            f"Supported modes: {list(TTA_TRANSFORM_MODES)}"
        )
    return mode


def _inverse_axis_perm(perm):
    inv = [0, 0, 0]
    for new_axis, old_axis in enumerate(perm):
        inv[old_axis] = new_axis
    return tuple(inv)


def _negate_flipped_displacement_channels(disp, flip_dims):
    if not flip_dims:
        return disp
    n_channels = int(disp.shape[1])
    n_vector_channels = (n_channels // 3) * 3
    if n_vector_channels <= 0:
        return disp

    sign = torch.ones((1, n_channels, 1, 1, 1), device=disp.device, dtype=disp.dtype)
    for d in flip_dims:
        base_ch = FLIP_DIM_TO_CHANNEL[d]
        sign[:, base_ch:n_vector_channels:3] = -1
    return disp * sign


def _reorder_rotated_displacement_channels(disp, inv_perm):
    if disp.shape[1] < 3:
        raise RuntimeError(
            f"Rotation TTA expects at least 3 displacement channels, got {disp.shape[1]}."
        )

    n_channels = int(disp.shape[1])
    n_vector_channels = (n_channels // 3) * 3
    disp_vectors = disp[:, :n_vector_channels].reshape(
        disp.shape[0], n_vector_channels // 3, 3, disp.shape[2], disp.shape[3], disp.shape[4]
    )
    disp_vectors = disp_vectors[:, :, list(inv_perm), :, :, :]
    disp_vectors = disp_vectors.reshape(
        disp.shape[0], n_vector_channels, disp.shape[2], disp.shape[3], disp.shape[4]
    )
    if n_vector_channels == n_channels:
        return disp_vectors
    return torch.cat([disp_vectors, disp[:, n_vector_channels:]], dim=1)


def _compute_vector_geomedian(points, max_iters=8, eps=1e-6, tol=1e-4):
    as_torch = torch.is_tensor(points)
    pts_in = points if as_torch else torch.as_tensor(points)
    if pts_in.ndim != 2:
        raise RuntimeError(f"Expected points with shape [N, C], got {tuple(pts_in.shape)}.")
    if int(pts_in.shape[0]) == 0:
        raise RuntimeError("Cannot compute geometric median of an empty point set.")

    pts = pts_in.to(dtype=torch.float64)
    if int(pts.shape[0]) == 1:
        out = pts[0]
    else:
        x = torch.median(pts, dim=0).values
        for _ in range(int(max_iters)):
            dist = torch.linalg.norm(pts - x.unsqueeze(0), dim=1).clamp_min(float(eps))
            w = 1.0 / dist
            x_new = (w.unsqueeze(-1) * pts).sum(dim=0) / w.sum().clamp_min(float(eps))
            step = torch.linalg.norm(x_new - x)
            x = x_new
            if float(step.item()) < float(tol):
                break
        out = x

    if as_torch:
        return out.to(dtype=pts_in.dtype, device=pts_in.device)
    return out.cpu().numpy()


def _aggregate_uv_points_geomedian(rows, cols, pts, h, w):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float64)
    if rows.size == 0:
        return np.full((h, w, 3), -1.0, dtype=np.float32), np.zeros((h, w), dtype=bool)

    grid_zyxs = np.full((h, w, 3), -1.0, dtype=np.float32)
    grid_valid = np.zeros((h, w), dtype=bool)

    linear = rows * int(w) + cols
    order = np.argsort(linear, kind="mergesort")
    linear_sorted = linear[order]
    pts_sorted = pts[order]

    unique_linear, starts, counts = np.unique(
        linear_sorted, return_index=True, return_counts=True
    )

    for linear_idx, start, count in zip(unique_linear, starts, counts):
        rr = int(linear_idx // int(w))
        cc = int(linear_idx % int(w))
        group_pts = pts_sorted[start:start + count]
        merged = (
            group_pts[0]
            if int(count) == 1
            else _compute_vector_geomedian(group_pts)
        )
        grid_zyxs[rr, cc] = merged.astype(np.float32, copy=False)
        grid_valid[rr, cc] = True

    return grid_zyxs, grid_valid


def _flatten_tta_disp_vectors(disp_stack):
    if disp_stack.ndim != 6:
        raise RuntimeError(
            f"Expected stacked TTA displacement [T, B, C, D, H, W], got {tuple(disp_stack.shape)}"
        )
    _, batch_size, channels, depth, height, width = disp_stack.shape
    vectors = disp_stack.permute(0, 1, 3, 4, 5, 2).reshape(disp_stack.shape[0], -1, channels)
    return vectors, (batch_size, channels, depth, height, width)


def _unflatten_tta_disp_vectors(vectors, disp_shape, out_dtype):
    batch_size, channels, depth, height, width = disp_shape
    merged = vectors.reshape(batch_size, depth, height, width, channels).permute(0, 4, 1, 2, 3)
    return merged.to(dtype=out_dtype)


def _merge_tta_vector_medoid(disp_stack):
    # Robust vector-consistent merge: pick the TTA vector minimizing total L2
    # distance to all other TTA vectors at each voxel.
    vec, disp_shape = _flatten_tta_disp_vectors(disp_stack)
    n_tta = vec.shape[0]
    vec32 = vec.float()
    n_locations = vec.shape[1]
    score = torch.zeros((n_tta, n_locations), dtype=vec32.dtype, device=vec32.device)

    for i in range(n_tta):
        diff = vec32 - vec32[i:i + 1]
        score[i] = (diff * diff).sum(dim=-1).sum(dim=0)

    medoid_idx = score.argmin(dim=0)
    vec_by_location = vec.permute(1, 0, 2)
    loc_idx = torch.arange(n_locations, device=vec.device)
    merged_vec = vec_by_location[loc_idx, medoid_idx]
    return _unflatten_tta_disp_vectors(merged_vec, disp_shape, disp_stack.dtype)


def _merge_tta_vector_geomedian(disp_stack, max_iters=8, eps=1e-6, tol=1e-4):
    # Robust vector-consistent merge via Weiszfeld iterations.
    vec, disp_shape = _flatten_tta_disp_vectors(disp_stack)
    vec32 = vec.float()
    x = torch.median(vec32, dim=0).values

    for _ in range(int(max_iters)):
        dist = torch.linalg.norm(vec32 - x.unsqueeze(0), dim=-1).clamp_min(eps)
        w = 1.0 / dist
        x_new = (w.unsqueeze(-1) * vec32).sum(dim=0) / w.sum(dim=0).clamp_min(eps).unsqueeze(-1)
        step = torch.linalg.norm(x_new - x, dim=-1).mean()
        x = x_new
        if float(step.item()) < tol:
            break

    return _unflatten_tta_disp_vectors(x, disp_shape, disp_stack.dtype)


def _merge_tta_displacements(disp_stack, merge_method):
    method = _validate_tta_merge_method(merge_method)

    if method == "mean":
        return disp_stack.mean(dim=0)
    if method == "median":
        return torch.median(disp_stack, dim=0).values
    if method == "trimmed_mean":
        if disp_stack.shape[0] <= 2:
            return disp_stack.mean(dim=0)
        sorted_disp, _ = torch.sort(disp_stack, dim=0)
        return sorted_disp[1:-1].mean(dim=0)
    if method == "vector_medoid":
        return _merge_tta_vector_medoid(disp_stack)
    if method == "vector_geomedian":
        return _merge_tta_vector_geomedian(disp_stack)
    raise RuntimeError(f"Unhandled TTA merge method '{method}'")


def _drop_tta_outlier_variants(disp_stack, thresh, min_keep=4):
    """Drop whole TTA variants whose global vector error is far from consensus."""
    if thresh is None:
        return disp_stack
    if disp_stack.ndim != 6:
        raise RuntimeError(
            f"Expected stacked TTA displacement [T, B, C, D, H, W], got {tuple(disp_stack.shape)}"
        )

    thresh = float(thresh)
    if thresh <= 0:
        return disp_stack

    n_tta = int(disp_stack.shape[0])
    if n_tta <= 2:
        return disp_stack

    min_keep = max(1, min(int(min_keep), n_tta))

    center = torch.median(disp_stack, dim=0).values
    diff = disp_stack.float() - center.unsqueeze(0).float()
    l2 = torch.linalg.norm(diff, dim=2)
    scores = l2.mean(dim=(1, 2, 3, 4))

    med = torch.median(scores)
    mad = torch.median(torch.abs(scores - med))

    spread = mad
    if float(spread.item()) <= 1e-12:
        spread = scores.std(unbiased=False)
    if float(spread.item()) <= 1e-12:
        return disp_stack

    cutoff = med + thresh * spread
    keep_mask = scores <= cutoff

    if int(keep_mask.sum().item()) < min_keep:
        keep_mask = torch.zeros_like(keep_mask)
        keep_idx = torch.topk(scores, k=min_keep, largest=False).indices
        keep_mask[keep_idx] = True

    if bool(keep_mask.all()):
        return disp_stack
    return disp_stack[keep_mask]


def run_model_tta(
    model,
    inputs,
    amp_enabled,
    amp_dtype,
    get_displacement_result,
    merge_method="vector_geomedian",
    transform_mode="mirror",
    outlier_drop_thresh=1.25,
    outlier_drop_min_keep=4,
    tta_batch_size=2,
    profiler=None,
):
    """Run TTA on a batch, returning merged displacement."""
    if inputs.ndim != 5:
        raise RuntimeError(f"Expected inputs with shape [B, C, D, H, W], got {tuple(inputs.shape)}")
    _validate_tta_merge_method(merge_method)
    mode = _validate_tta_transform_mode(transform_mode)

    batch_size = int(inputs.shape[0])
    if batch_size <= 0:
        raise RuntimeError("TTA received an empty batch.")

    n_tta = len(TTA_FLIP_COMBOS) if mode == "mirror" else len(TTA_ROTATE3_PERMS)
    if tta_batch_size is None:
        tta_batch_size = n_tta
    tta_batch_size = int(tta_batch_size)
    if tta_batch_size < 1:
        raise RuntimeError(f"tta_batch_size must be >= 1; got {tta_batch_size}.")
    tta_batch_size = min(tta_batch_size, n_tta)

    aligned_displacements = []

    for chunk_start in range(0, n_tta, tta_batch_size):
        with _profile_section(profiler, "iter_tta_build_variants"):
            tta_inputs = []
            if mode == "mirror":
                transform_chunk = TTA_FLIP_COMBOS[chunk_start:chunk_start + tta_batch_size]
                for flip_dims in transform_chunk:
                    x = inputs
                    for d in flip_dims:
                        x = x.flip(d)
                    tta_inputs.append(x)
            else:
                transform_chunk = TTA_ROTATE3_PERMS[chunk_start:chunk_start + tta_batch_size]
                for perm in transform_chunk:
                    tta_inputs.append(
                        inputs.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
                    )
            tta_inputs = torch.cat(tta_inputs, dim=0)
        with _profile_section(profiler, "iter_tta_forward_chunk"):
            disp_all = get_displacement_result(model, tta_inputs, amp_enabled, amp_dtype)
        expected_batch = batch_size * len(transform_chunk)
        if disp_all.shape[0] != expected_batch:
            raise RuntimeError(
                f"Unexpected TTA output batch size {disp_all.shape[0]} (expected {expected_batch})."
            )

        with _profile_section(profiler, "iter_tta_align_outputs"):
            for local_idx, transform in enumerate(transform_chunk):
                start = local_idx * batch_size
                end = start + batch_size
                disp = disp_all[start:end]

                if mode == "mirror":
                    flip_dims = transform
                    # Un-flip displacement outputs back to the original orientation.
                    for d in reversed(flip_dims):
                        disp = disp.flip(d)
                    # Negate displacement components along flipped spatial axes.
                    disp = _negate_flipped_displacement_channels(disp, flip_dims)
                else:
                    perm = transform
                    inv_perm = _inverse_axis_perm(perm)
                    disp = disp.permute(0, 1, 2 + inv_perm[0], 2 + inv_perm[1], 2 + inv_perm[2])
                    disp = _reorder_rotated_displacement_channels(disp, inv_perm)

                aligned_displacements.append(disp)

    with _profile_section(profiler, "iter_tta_merge"):
        disp_stack = torch.stack(aligned_displacements, dim=0)
        disp_stack = _drop_tta_outlier_variants(
            disp_stack,
            thresh=outlier_drop_thresh,
            min_keep=outlier_drop_min_keep,
        )
        return _merge_tta_displacements(disp_stack, merge_method)
