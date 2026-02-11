import torch


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

# Mapping from flip dim to displacement channel that must be negated:
# dim -1 (W/X) -> channel 2, dim -2 (H/Y) -> channel 1, dim -3 (D/Z) -> channel 0
FLIP_DIM_TO_CHANNEL = {-1: 2, -2: 1, -3: 0}
TTA_MERGE_METHODS = ("median", "mean", "trimmed_mean", "vector_medoid", "vector_geomedian")


def _validate_tta_merge_method(merge_method):
    method = str(merge_method).strip().lower()
    if method not in TTA_MERGE_METHODS:
        raise ValueError(
            f"Unknown --tta-merge-method '{merge_method}'. "
            f"Supported methods: {list(TTA_MERGE_METHODS)}"
        )
    return method


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
    outlier_drop_thresh=1.25,
    outlier_drop_min_keep=4,
):
    """Run mirroring-based TTA on a batch, returning merged displacement."""
    if inputs.ndim != 5:
        raise RuntimeError(f"Expected inputs with shape [B, C, D, H, W], got {tuple(inputs.shape)}")
    _validate_tta_merge_method(merge_method)

    batch_size = int(inputs.shape[0])
    if batch_size <= 0:
        raise RuntimeError("TTA received an empty batch.")

    # Stack all mirrored variants along the batch dimension and run one forward pass.
    tta_inputs = []
    for flip_dims in TTA_FLIP_COMBOS:
        x = inputs
        for d in flip_dims:
            x = x.flip(d)
        tta_inputs.append(x)

    tta_inputs = torch.cat(tta_inputs, dim=0)
    disp_all = get_displacement_result(model, tta_inputs, amp_enabled, amp_dtype)

    n_tta = len(TTA_FLIP_COMBOS)
    expected_batch = batch_size * n_tta
    if disp_all.shape[0] != expected_batch:
        raise RuntimeError(
            f"Unexpected TTA output batch size {disp_all.shape[0]} (expected {expected_batch})."
        )

    aligned_displacements = []

    for tta_idx, flip_dims in enumerate(TTA_FLIP_COMBOS):
        start = tta_idx * batch_size
        end = start + batch_size
        disp = disp_all[start:end]

        # Un-flip displacement outputs back to the original orientation.
        for d in reversed(flip_dims):
            disp = disp.flip(d)

        # Negate displacement components along flipped spatial axes.
        if flip_dims:
            sign = torch.ones((1, disp.shape[1], 1, 1, 1), device=disp.device, dtype=disp.dtype)
            for d in flip_dims:
                ch = FLIP_DIM_TO_CHANNEL[d]
                if ch >= disp.shape[1]:
                    raise RuntimeError(
                        f"TTA channel index {ch} out of bounds for displacement with {disp.shape[1]} channels."
                    )
                sign[:, ch] = -1
            disp = disp * sign

        aligned_displacements.append(disp)

    disp_stack = torch.stack(aligned_displacements, dim=0)
    disp_stack = _drop_tta_outlier_variants(
        disp_stack,
        thresh=outlier_drop_thresh,
        min_keep=outlier_drop_min_keep,
    )
    return _merge_tta_displacements(disp_stack, merge_method)
