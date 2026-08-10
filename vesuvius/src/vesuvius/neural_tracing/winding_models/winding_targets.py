"""Dense supervision targets, losses, and decoding for the winding model.

The dataset labels every supervised slab column (a ray-parallel line of
voxels at the model's column stride) with ordered winding crossings;
training needs dense per-sample targets along each column:

- ``phase``: relative winding coordinate, piecewise linear through the
  crossings (exactly k at crossing k). The winding sign is canonicalized
  once per slab so phase increases along the ray axis in every column,
  keeping scroll chirality out of the learning problem; the fit_spiral
  consumer knows each ray's winding direction and flips per ray. Winding
  indices are globally consistent across a slab's columns, so the whole
  phase field shares a single free offset which the shift-invariant loss
  (and the consumer's per-ray registration) absorbs.
- ``crossing``: a narrow Gaussian heatmap at the crossings whose nearest
  sample is pinned to exactly one, giving the penalty-reduced focal loss an
  exact positive set.

Negatives are only supervised where ``winding_valid`` holds: spans that may
contain unlabeled wraps must not be taught as "no crossing".
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# Below this heatmap value a sample no longer counts as crossing evidence and
# its supervision falls back to the winding-validity mask alone.
_CROSSING_SUPPORT = 0.05


def render_column_targets(
    crossing_t: np.ndarray,
    winding_indices: np.ndarray,
    *,
    ray_length: int,
    spacing: float,
    crossing_sigma_wv: float,
) -> dict[str, np.ndarray]:
    """Densify one column's ordered crossing labels along the ray axis.

    ``crossing_t`` must be strictly increasing and ``winding_indices``
    nondecreasing (the slab-global winding sign is applied by the caller);
    both need at least two entries.
    """
    crossing_t = np.asarray(crossing_t, dtype=np.float64)
    indices = np.asarray(winding_indices, dtype=np.float64)
    sample_ts = np.arange(ray_length, dtype=np.float64) * spacing

    phase = np.interp(sample_ts, crossing_t, indices)

    # Density supervision compares the model's per-segment phase increment to
    # the diff of the interp target — the exact integral of the
    # piecewise-constant 1/gap label density over segment (i-1, i] — so a
    # perfect prediction scores exactly zero. Each segment also carries the
    # narrowest labeled gap it touches so the loss can abstain where the
    # dataset's crossing merge distance makes tight-gap labels unreliable.
    density_target = np.zeros(ray_length, dtype=np.float64)
    density_target[1:] = np.diff(phase)
    gaps = np.diff(crossing_t)
    gap_index = np.searchsorted(crossing_t, sample_ts, side="right") - 1
    in_gap = (gap_index >= 0) & (gap_index < len(crossing_t) - 1)
    gap_at_sample = np.where(
        in_gap, gaps[np.clip(gap_index, 0, max(len(gaps) - 1, 0))], 0.0
    )
    density_gap = np.zeros(ray_length, dtype=np.float64)
    density_gap[1:] = np.minimum(gap_at_sample[1:], gap_at_sample[:-1])

    deviation = np.abs(sample_ts[:, None] - crossing_t[None, :]).min(axis=1)
    heatmap = np.exp(-0.5 * (deviation / crossing_sigma_wv) ** 2)
    nearest = np.clip(np.rint(crossing_t / spacing).astype(int), 0, ray_length - 1)
    heatmap[nearest] = 1.0

    # Winding labels are trustworthy between the column's first and last
    # crossing, except across gaps whose winding indices differ by more than
    # one: those spans may contain unlabeled wraps.
    winding_valid = (sample_ts >= crossing_t[0]) & (sample_ts <= crossing_t[-1])
    for gap in np.nonzero(np.abs(np.diff(indices)) > 1)[0]:
        winding_valid &= (sample_ts <= crossing_t[gap]) | (
            sample_ts >= crossing_t[gap + 1]
        )

    return {
        "phase_target": phase.astype(np.float32),
        "phase_valid": winding_valid,
        "crossing_target": heatmap.astype(np.float32),
        "crossing_valid": winding_valid | (heatmap > _CROSSING_SUPPORT),
        "density_target": density_target.astype(np.float32),
        "density_gap_wv": density_gap.astype(np.float32),
    }


def render_crossing_only_targets(
    crossing_t: np.ndarray,
    *,
    ray_length: int,
    spacing: float,
    crossing_sigma_wv: float,
) -> dict[str, np.ndarray]:
    """Densify position-only crossing labels along the ray axis.

    For crossings with unknown winding indices (e.g. auto-grown single-sheet
    patches), only the crossing head can be supervised: positives at the
    labeled positions plus the negatives inside each rendered kernel's
    support. Spans between labeled crossings may hide unlabeled wraps, so
    ``crossing_valid`` holds nowhere else and the phase and density targets
    stay fully unsupervised.
    """
    crossing_t = np.asarray(crossing_t, dtype=np.float64)
    sample_ts = np.arange(ray_length, dtype=np.float64) * spacing
    deviation = np.abs(sample_ts[:, None] - crossing_t[None, :]).min(axis=1)
    heatmap = np.exp(-0.5 * (deviation / crossing_sigma_wv) ** 2)
    nearest = np.clip(np.rint(crossing_t / spacing).astype(int), 0, ray_length - 1)
    heatmap[nearest] = 1.0
    zeros = np.zeros(ray_length, dtype=np.float32)
    return {
        "phase_target": zeros,
        "phase_valid": np.zeros(ray_length, dtype=bool),
        "crossing_target": heatmap.astype(np.float32),
        "crossing_valid": heatmap > _CROSSING_SUPPORT,
        "density_target": zeros,
        "density_gap_wv": zeros,
    }


_PADDED_KEYS = {"crossing_t": float("nan"), "crossing_indices": 0}


def collate_winding_batch(batch: list[dict]) -> dict:
    """Stack dataset samples; per-column crossing lists are padded to the
    batch's widest column (NaN positions alongside a count tensor)."""
    collated = {}
    max_crossings = max(sample["crossing_t"].shape[-1] for sample in batch)
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        pad_value = _PADDED_KEYS.get(key)
        if pad_value is not None:
            values = [
                F.pad(value, (0, max_crossings - value.shape[-1]), value=pad_value)
                for value in values
            ]
        collated[key] = torch.stack(values)
    return collated


def phase_loss(
    phase_pred: torch.Tensor,
    phase_target: torch.Tensor,
    phase_valid: torch.Tensor,
    *,
    huber_delta: float = 0.25,
) -> torch.Tensor:
    """Shift-invariant masked Huber loss on the relative winding phase.

    Prediction and target are mean-centered over each slab's valid samples
    jointly across all columns — winding indices are globally consistent
    within a slab, so a single free offset per slab also supervises the
    transverse coherence of the phase field. Slabs with fewer than two valid
    samples contribute nothing.
    """
    batch = phase_pred.shape[0]
    phase_pred = phase_pred.reshape(batch, -1)
    phase_target = phase_target.reshape(batch, -1)
    weight = phase_valid.reshape(batch, -1).to(phase_pred.dtype)
    count = weight.sum(dim=-1)
    denominator = count.clamp_min(1.0)
    pred_mean = (phase_pred * weight).sum(dim=-1) / denominator
    target_mean = (phase_target * weight).sum(dim=-1) / denominator
    residual = (phase_pred - pred_mean[:, None]) - (
        phase_target - target_mean[:, None]
    )
    per_sample = F.huber_loss(
        residual, torch.zeros_like(residual), delta=huber_delta, reduction="none"
    )
    per_slab = (per_sample * weight).sum(dim=-1) / denominator
    active = (count >= 2).to(phase_pred.dtype)
    return (per_slab * active).sum() / active.sum().clamp_min(1.0)


def crossing_loss(
    crossing_logits: torch.Tensor,
    crossing_target: torch.Tensor,
    crossing_valid: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced focal loss (CenterNet) on the masked crossing heatmap."""
    prob = torch.sigmoid(crossing_logits.float()).clamp(1e-5, 1.0 - 1e-5)
    valid = crossing_valid.to(prob.dtype)
    positive = (crossing_target >= 1.0).to(prob.dtype) * valid
    negative = (1.0 - positive) * valid
    positive_loss = -torch.log(prob) * (1.0 - prob) ** alpha * positive
    negative_loss = (
        -torch.log(1.0 - prob)
        * prob**alpha
        * (1.0 - crossing_target) ** beta
        * negative
    )
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1.0)


def density_supervision_mask(
    density_gap_wv: torch.Tensor,
    phase_valid: torch.Tensor,
    *,
    min_gap_wv: float,
) -> torch.Tensor:
    """Boolean supervision mask for each phase increment's segment (i-1, i].

    The last dimension is the ray axis; any leading column/batch layout is
    accepted. A segment is supervised when both endpoint labels are
    trustworthy and every labeled gap it touches is at least ``min_gap_wv``
    wide (the dataset's crossing merge distance makes tighter labels
    unreliable). Sample 0 has no preceding sample (its increment absorbs the
    free phase offset) and its rendered gap width is zero, so it is never
    supervised.
    """
    valid = phase_valid.bool()
    segment_valid = valid.clone()
    segment_valid[..., 0] = False
    segment_valid[..., 1:] &= valid[..., :-1]
    return segment_valid & (density_gap_wv >= float(min_gap_wv))


def density_loss(
    phase_increments: torch.Tensor,
    density_log_variance: torch.Tensor,
    density_target: torch.Tensor,
    density_gap_wv: torch.Tensor,
    phase_valid: torch.Tensor,
    *,
    min_gap_wv: float = 4.0,
    beta: float = 0.5,
    log_variance_range: tuple[float, float] = (-12.0, 6.0),
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL on per-segment phase increments.

    ``phase_increments[..., i]`` is the model's integral of winding density
    over its column's segment (i-1, i]; ``density_target[..., i]`` is the
    same integral of the piecewise-constant 1/gap label density (the diff of
    the interp phase target), so a perfect prediction scores exactly zero.
    No new phase head: the increments are the pre-cumsum ``softplus`` values
    from the monotone phase head. Each supervised column is normalized by
    its supervised-segment count, then columns average uniformly.

    The predicted log-variance turns the loss into an observation model the
    consumer can reuse: fit_spiral weights each registered increment by its
    precision ``exp(-log_variance)``. The NLL also self-attenuates label
    noise -- inflating the variance where residuals stay high costs only the
    ``log_variance`` term -- which replaces the fixed-scale Huber a
    homoscedastic loss would need. The log-variance is clamped for stability.

    ``beta`` applies the beta-NLL correction (Seitzer et al.): each sample's
    NLL is scaled by a detached ``sigma^(2 beta)``, so high-variance samples
    keep pulling on the mean (at ``beta = 0.5`` the mean's effective weight
    is ``1 / sigma`` instead of ``1 / sigma^2``) rather than being written
    off once the model inflates their variance -- and the hard samples are
    tight gaps and near-tears, exactly where the consumer needs signal.
    ``beta = 0`` recovers the plain NLL.
    """
    increments = phase_increments.float()
    log_variance = density_log_variance.float().clamp(*log_variance_range)
    target = density_target.to(increments.dtype)
    weight = density_supervision_mask(
        density_gap_wv, phase_valid, min_gap_wv=min_gap_wv
    ).to(increments.dtype)
    residual = increments - target
    per_sample = 0.5 * (
        torch.exp(-log_variance) * residual.square() + log_variance
    )
    if beta:
        per_sample = per_sample * torch.exp(beta * log_variance).detach()
    count = weight.sum(dim=-1)
    per_column = (per_sample * weight).sum(dim=-1) / count.clamp_min(1.0)
    active = (count > 0).to(increments.dtype)
    return (per_column * active).sum() / active.sum().clamp_min(1.0)


def head_consistency_loss(
    crossing_logits: torch.Tensor,
    phase_increments: torch.Tensor,
    phase_valid: torch.Tensor,
    crossing_target: torch.Tensor,
    *,
    crossing_sigma_wv: float,
    spacing: float,
    huber_delta: float = 0.5,
) -> torch.Tensor:
    """Couple the two observation heads through their running winding counts.

    Both heads count windings along a column: the crossing heatmap's
    cumulative mass divided by the per-crossing kernel mass, and the
    cumulative sum of the phase increments. Their running difference is
    compared at each supervised labeled crossing rather than only across
    the span total, because the running comparison carries position: a peak
    missing from the heatmap leaves a unit deficit at every crossing
    downstream of where it should be, so the gradient concentrates on the
    segment of the miss. (A span-total constraint reaches each logit
    through the sigmoid slope p(1-p), which is largest on existing peak
    shoulders -- it fattens the peaks it already has instead of recruiting
    missed ones.)

    Crossings are the only points where the two counts agree for a
    consistent model: between crossings the heatmap concentrates its mass
    near the crossing while the label-faithful increments spread it 1/gap
    over the whole gap, so a densely evaluated running difference carries
    an irreducible half-winding sawtooth -- which a dense loss would
    "fix" by blurring peak mass into the gaps. At a crossing both counts
    have accumulated the same whole windings plus half the boundary
    kernel, so the sawtooth vanishes. The evaluation points are the
    rendered heatmap's pinned samples (exactly 1 at each labeled
    crossing).

    The running difference is registered per column with one free shift,
    mirroring the phase loss: partial boundary kernel mass and any
    disagreement accumulated before the span appear as a constant offset,
    which the shift absorbs. The phase increments are detached: the phase
    count is the far more accurate side, so only the crossing head is
    calibrated toward it (an undetached version dragged the increments
    down toward the under-massed heatmap, tripling the winding-count
    error). Sample 0 is excluded: its "increment" is the free phase
    offset, not a density integral. Supervised per-column, columns average
    uniformly.
    """
    prob = torch.sigmoid(crossing_logits.float())
    # The phase count is the trusted side: detached so consistency only
    # calibrates the crossing head toward it, never the reverse.
    increments = phase_increments.float().detach()
    valid = phase_valid.bool() & (crossing_target >= 1.0)
    valid[..., 0] = False
    weight = valid.to(prob.dtype)

    # Per-crossing mass of the rendered heatmap: the sampled Gaussian kernel,
    # plus the expected inflation from pinning the nearest sample to one
    # (its Gaussian value averaged over a uniform subsample crossing offset).
    radius = int(math.ceil(5.0 * crossing_sigma_wv / spacing))
    kernel_mass = sum(
        math.exp(-0.5 * ((k * spacing) / crossing_sigma_wv) ** 2)
        for k in range(-radius, radius + 1)
    )
    mean_nearest = (
        crossing_sigma_wv
        * math.sqrt(2.0 * math.pi)
        * math.erf(spacing / (2.0 * crossing_sigma_wv * math.sqrt(2.0)))
        / spacing
    )
    kernel_mass += 1.0 - mean_nearest

    counted = increments.clone()
    counted[..., 0] = 0.0
    running = prob.cumsum(dim=-1) / kernel_mass - counted.cumsum(dim=-1)
    count = weight.sum(dim=-1)
    shift = (running * weight).sum(dim=-1) / count.clamp_min(1.0)
    residual = running - shift[..., None]
    per_sample = F.huber_loss(
        residual, torch.zeros_like(residual), delta=huber_delta, reduction="none"
    )
    per_column = (per_sample * weight).sum(dim=-1) / count.clamp_min(1.0)
    active = (count >= 2).to(prob.dtype)
    return (per_column * active).sum() / active.sum().clamp_min(1.0)


def crossing_distillation_loss(
    crossing_logits: torch.Tensor,
    phase: torch.Tensor,
    phase_target: torch.Tensor,
    phase_valid: torch.Tensor,
    crossing_valid: torch.Tensor,
    *,
    crossing_sigma_wv: float,
    spacing: float,
) -> torch.Tensor:
    """Distill the phase head's crossings into the heatmap between labels.

    The monotone phase field predicts winding structure over the whole slab,
    but the focal loss supervises the heatmap only where ``crossing_valid``
    holds; elsewhere the crossing head trains blind. This loss lets the
    trusted head teach the blind spans: crossing locations are read off the
    (detached) predicted phase as the sub-sample points where it passes an
    integer, rendered with the label kernel (nearest sample pinned to one),
    and taught to the crossing head with a soft binary cross-entropy
    restricted to ``~crossing_valid`` -- complementary to the focal loss, so
    the teacher never competes with real labels.

    The predicted phase carries one free offset per slab (the phase loss is
    shift-invariant), so its integer levels do not land on crossings by
    themselves; the offset is first registered against the phase target over
    the slab's supervised samples. Winding indices are globally consistent
    across a slab's columns, so this one registration places integer levels
    at crossings everywhere in the slab, including unlabeled spans. Slabs
    with fewer than two supervised samples provide no registration and are
    skipped.
    """
    logits = crossing_logits.float()
    phase = phase.float().detach()
    batch, length = phase.shape[0], phase.shape[-1]
    expand = (batch,) + (1,) * (phase.dim() - 1)

    # Register the slab's free phase offset against the labels, as the phase
    # loss and metrics do, so integer levels of the registered phase sit at
    # crossings.
    weight = phase_valid.reshape(batch, -1).to(phase.dtype)
    count = weight.sum(dim=-1)
    offset = (
        (phase.reshape(batch, -1) - phase_target.float().reshape(batch, -1))
        * weight
    ).sum(dim=-1) / count.clamp_min(1.0)
    slab_active = (count >= 2).to(phase.dtype).reshape(*expand)
    registered = phase - offset.reshape(*expand)

    # Sub-sample positions where the registered phase passes an integer.
    # Phase is monotone (softplus increments), so each segment (i-1, i]
    # crosses at most a run of consecutive integers; the first is
    # representative at kernel resolution.
    level = registered.floor()
    crossed = level[..., 1:] > level[..., :-1]
    step = (registered[..., 1:] - registered[..., :-1]).clamp_min(1e-6)
    fraction = ((level[..., :-1] + 1.0 - registered[..., :-1]) / step).clamp(
        0.0, 1.0
    )
    index = torch.arange(length, device=phase.device, dtype=phase.dtype)
    event_position = index[:-1] + fraction

    # Distance from every sample to its nearest event via two cumulative
    # scans (events are ordered along the ray, so last-before and
    # first-after bound the nearest).
    far = torch.full_like(event_position, 1e9)
    last_event = torch.cummax(
        torch.where(crossed, event_position, -far), dim=-1
    ).values
    next_event = (
        torch.cummin(torch.where(crossed, event_position, far).flip(-1), dim=-1)
        .values.flip(-1)
    )
    pad = far[..., :1]
    forward = index - torch.cat([-pad, last_event], dim=-1)
    backward = torch.cat([next_event, pad], dim=-1) - index
    distance = torch.minimum(forward, backward).clamp_min(0.0)

    pseudo = torch.exp(-0.5 * (distance * spacing / crossing_sigma_wv) ** 2)
    pseudo = torch.where(distance <= 0.5, torch.ones_like(pseudo), pseudo)

    mask = (~crossing_valid.bool()).to(logits.dtype) * slab_active
    # Subtract the soft targets' entropy so the loss is the KL divergence:
    # zero at a perfect match instead of the targets' entropy floor. The
    # entropy is constant in the logits, so gradients are unchanged.
    entropy = -(
        torch.special.xlogy(pseudo, pseudo)
        + torch.special.xlogy(1.0 - pseudo, 1.0 - pseudo)
    )
    per_sample = (
        F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
        - entropy
    )
    column_count = mask.sum(dim=-1)
    per_column = (per_sample * mask).sum(dim=-1) / column_count.clamp_min(1.0)
    active = (column_count > 0).to(logits.dtype)
    return (per_column * active).sum() / active.sum().clamp_min(1.0)


def position_only_phase_loss(
    phase: torch.Tensor,
    crossing_t: torch.Tensor,
    num_crossings: torch.Tensor,
    phase_valid: torch.Tensor,
    *,
    spacing: float,
    snap_gate: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weak phase supervision from position-only crossing labels.

    Position-only labels (auto-grown patches) mark where sheets are but not
    how many windings separate them. They still constrain the phase:
    consecutive labeled crossings along a column are distinct windings, so
    the predicted phase difference across each labeled gap must be at least
    one full winding, and -- both endpoints being crossings -- a positive
    integer. Both constraints are offset-free: differencing cancels the
    slab's free phase offset, so no registration is needed.

    Returns two separately weightable scalars:

    - ``hinge``: ``relu(1 - delta)`` per labeled gap. A hard constraint,
      nonzero only when the phase head under-counts a labeled gap; safe at
      any weight.
    - ``snap``: squared distance to the nearest integer, applied only when
      the prediction is already within ``snap_gate`` of one. A quantization
      refinement that abstains while the count is ambiguous (e.g. a
      predicted 1.5), so it cannot entrench a wrong integer.

    Supervised are columns with at least two labeled crossings and no phase
    supervision anywhere -- exactly the position-only columns; fully
    labeled columns are covered by the phase loss and contribute nothing
    here. Per-column means, columns averaging uniformly, matching the other
    losses. Predicted phase is read at the labeled positions by linear
    interpolation along the ray.
    """
    phase = phase.float()
    length = phase.shape[-1]
    positions = (crossing_t.float() / float(spacing)).nan_to_num(0.0)
    positions = positions.clamp(0.0, float(length - 1))

    lower = positions.floor().long().clamp(0, length - 2)
    frac = positions - lower.to(positions.dtype)
    at_lower = phase.gather(-1, lower)
    at_upper = phase.gather(-1, lower + 1)
    at_crossings = at_lower + (at_upper - at_lower) * frac

    delta = at_crossings[..., 1:] - at_crossings[..., :-1]
    slot = torch.arange(crossing_t.shape[-1], device=phase.device)
    pair_valid = (slot[1:] < num_crossings[..., None]).to(phase.dtype)
    column = (num_crossings >= 2) & (phase_valid.sum(dim=-1) == 0)
    pair_valid = pair_valid * column.to(phase.dtype)[..., None]

    hinge = F.relu(1.0 - delta)
    # round() carries no gradient, so the residual's gradient w.r.t. delta
    # is 1 inside the gate; the gate itself is a detached indicator.
    residual = delta - delta.round()
    snap = 0.5 * residual.square()
    snap_active = (
        (delta >= 1.0) & (residual.abs() <= snap_gate)
    ).to(phase.dtype)

    pairs = pair_valid.sum(dim=-1)
    hinge_column = (hinge * pair_valid).sum(dim=-1) / pairs.clamp_min(1.0)
    snap_column = (snap * snap_active * pair_valid).sum(dim=-1) / pairs.clamp_min(
        1.0
    )
    active = (pairs > 0).to(phase.dtype)
    denominator = active.sum().clamp_min(1.0)
    return (
        (hinge_column * active).sum() / denominator,
        (snap_column * active).sum() / denominator,
    )


def extract_peaks(
    prob: np.ndarray, *, threshold: float = 0.3, min_distance: int = 2
) -> np.ndarray:
    """Sample indices of local maxima above ``threshold``, greedy NMS."""
    order = np.argsort(prob, kind="stable")[::-1]
    suppressed = np.zeros(len(prob), dtype=bool)
    kept = []
    for index in order:
        if prob[index] < threshold:
            break
        if suppressed[index]:
            continue
        kept.append(int(index))
        suppressed[max(0, index - min_distance) : index + min_distance + 1] = True
    return np.sort(np.asarray(kept, dtype=np.int64))


def match_crossings(
    predicted_ts: np.ndarray, target_ts: np.ndarray, *, tolerance: float
) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (true pos, false pos, false neg)."""
    remaining = [float(t) for t in target_ts]
    true_positives = 0
    for t in sorted(float(t) for t in predicted_ts):
        if not remaining:
            break
        nearest = min(range(len(remaining)), key=lambda i: abs(remaining[i] - t))
        if abs(remaining[nearest] - t) <= tolerance:
            remaining.pop(nearest)
            true_positives += 1
    return (
        true_positives,
        len(predicted_ts) - true_positives,
        len(remaining),
    )
