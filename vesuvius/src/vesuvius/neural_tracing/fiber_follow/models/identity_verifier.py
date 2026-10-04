"""Memory-conditioned fiber verification (``identity_objective='verify'``).

The verifier answers, for any current-crop location: does it lie on the fiber this trace's
decision memory describes (its original fiber)? A location is queried with the current crop's
contextual feature there plus its crop position, and reads the same ``DecisionMemory`` tokens
(entry features, pose, slot) the decoder and scorer read, so its loss trains that memory pathway.

Supervision is dense. Every query position is labeled by one rule from the original fiber's
annotated curve: on it within POSITIVE_RADIUS, off it beyond NEGATIVE_RADIUS (whatever else lies
there, annotated or not), unknown in between and past an annotation end inside the crop. Query
positions: crossings at IDENTITY_PLANES (original fiber, head-axis point, neighbors; also a listwise
term), dense samples along/beside the original fiber, on neighbors, on the trace path and uniform,
and the model's own predicted path. The trace path is where "on the path" could stand in for "on
the original fiber", so the loss balances labels within on-path and off-path strata.

The dense identity field (the verifier on the whole token lattice) is computed without gradient
and may feed the decoder/scorer image tokens and the scorer/retry feedback.
"""
import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.models.decision_memory import IDENTITY_NEGATIVES
from vesuvius.neural_tracing.fiber_follow.data.state_labels import DEPARTURE_DISTANCE

IDENTITY_PLANES = (2., 8., 14.)  # forward planes (trace voxels) of the crossing candidates
IDENTITY_CANDIDATES = 2+IDENTITY_NEGATIVES  # original fiber, head axis, neighbors
HEAD_AXIS = 1
IDENTITY_SAMPLES = 512  # dense query positions per state
IDENTITY_CURVE = 256  # original-fiber curve samples kept for labeling
# Dense sample sources: along the original fiber, 3.5-8 voxels beside it, on neighbor paths,
# on the trace path (behind the head, and its axis ahead), uniform in the crop.
SAMPLE_MIX = dict(original=160, beside=96, neighbors=96, path=64, uniform=96)
POSITIVE_RADIUS = 1.5  # on the original fiber (the own-fiber radius)
NEGATIVE_RADIUS = DEPARTURE_DISTANCE  # off it
PATH_RADIUS = 1.5  # on the trace path (observed behind the head, predicted ahead)
# Loss balance: running label fractions per (off-path, on-path) stratum.
BALANCE_DECAY = .99
BALANCE_FLOOR = .02
MEMORY_ONLY_AGE = 64.  # departures older than this are invisible to the current crop and recent memory


def verification_buffers(n):
    """Zero-filled per-batch verification inputs (positions, masks and the labeling curve)."""
    import numpy as np
    from vesuvius.neural_tracing.fiber_follow.models.decision_memory import SLOTS
    shape = (n, len(IDENTITY_PLANES), IDENTITY_CANDIDATES)
    return dict(identity_candidates=np.zeros((*shape, 3), np.float32), identity_mask=np.zeros(shape, bool),
                identity_samples=np.zeros((n, IDENTITY_SAMPLES, 3), np.float32),
                identity_sample_mask=np.zeros((n, IDENTITY_SAMPLES), bool),
                identity_curve=np.zeros((n, IDENTITY_CURVE, 3), np.float32),
                identity_curve_mask=np.zeros((n, IDENTITY_CURVE), bool),
                identity_curve_ends=np.zeros((n, 2), bool), identity_synthetic=np.zeros(n, bool),
                identity_anchor_mask=np.zeros((n, SLOTS), bool), identity_departure_age=np.full(n, np.nan, np.float32))


class IdentityVerifier(nn.Module):
    def __init__(self, cfg, layers=2):
        super().__init__()
        from vesuvius.neural_tracing.fiber_follow.models.history_slabs import HistoryAttention
        h = cfg.hidden
        self.query = nn.Sequential(nn.Linear(h+3, h), nn.SiLU(), nn.Linear(h, h))
        self.attention = nn.ModuleList(HistoryAttention(h, cfg.heads) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2*h), nn.GELU(), nn.Linear(2*h, h))
                                 for _ in range(layers))
        self.norm = nn.LayerNorm(h)
        self.logit = nn.Linear(h, 1)

    def project(self, tokens, padding):
        """Per-layer memory K/V for one decision (empty memory rows read nothing)."""
        return [attention.project_memory(tokens, padding) for attention in self.attention]

    def forward(self, features, points, projected):
        """(B, N) logits for ``features`` (B, N, C) sampled at crop-local ``points`` (B, N, 3)."""
        query = self.query(torch.cat((features, (points/16.).to(features.dtype)), -1))
        for attention, mlp, kv in zip(self.attention, self.mlp, projected):
            query = attention.forward_cached(query, *kv)
            query = query+mlp(query)
        with torch.autocast(query.device.type, enabled=False):
            return self.logit(self.norm(query.float())).squeeze(-1)


def identity_field(model, deep, tokens, padding):
    """(B, 1, D, H, W) on-memory-fiber probability at every token centre, without gradient."""
    with torch.no_grad():
        features = deep.detach().flatten(2).transpose(1, 2)
        points = model.encoder.token_xyz[None].expand(len(features), -1, -1).to(features.dtype)
        projected = model.identity_verifier.project(tokens.detach(), padding)
        probability = torch.sigmoid(model.identity_verifier(features, points, projected))
    return probability.reshape(len(deep), 1, *deep.shape[-3:])


def nearest_distance(points, targets, mask):
    """(B, N) distance from each point to its nearest valid target, and that target's index."""
    distance = torch.cdist(points.float(), targets.float()).masked_fill(~mask[:, None, :], torch.inf)
    return distance.min(-1)


def curve_labels(points, curve, curve_mask, ends):
    """(label, known) for points (B, N, 3) from the original fiber's curve (B, M, 3).

    On the fiber within POSITIVE_RADIUS; off it beyond NEGATIVE_RADIUS; unknown between, nearest
    to an annotation end inside the crop (the fiber may continue unannotated) unless on it, and
    for states without a curve."""
    distance, index = nearest_distance(points, curve, curve_mask)
    last = curve_mask.sum(-1, keepdim=True)-1
    at_end = ((index == 0) & ends[:, :1]) | ((index == last) & ends[:, 1:])
    label = distance <= POSITIVE_RADIUS
    known = (label | ((distance > NEGATIVE_RADIUS) & ~at_end)) & curve_mask.any(-1, keepdim=True)
    return label, known


def path_stratum(points, hist, hmask, predicted):
    """Whether points lie on the trace path: observed behind the head, the model's own path ahead."""
    path = torch.cat((hist, predicted.to(hist.dtype)), 1)
    mask = torch.cat((hmask > 0, torch.ones(predicted.shape[:2], dtype=torch.bool, device=hist.device)), 1)
    return nearest_distance(points, path, mask)[0] <= PATH_RADIUS


def verification_targets(output, batch, exclude_synthetic=False):
    """Labels for every verifier query: candidates (B, P, K), dense samples (B, N), predictions (B, F).

    Returns flat (B, Q) tensors over the concatenated queries (label, known, on_path) plus the
    candidate labels/known reshaped for the listwise term and the head-axis metrics. ``train``
    marks the labels the loss uses: with ``exclude_synthetic`` synthetic switch states only
    evaluate (their construction differs from real switches), so they stay in the metrics."""
    b, planes, k = batch['identity_mask'].shape
    points = output['identity_points'].detach()
    exists = torch.cat((batch['identity_mask'].reshape(b, -1), batch['identity_sample_mask'],
                        torch.ones(output['identity_predicted_points'].shape[:2], dtype=torch.bool, device=points.device)), 1)
    label, known = curve_labels(points, batch['identity_curve'], batch['identity_curve_mask'], batch['identity_curve_ends'])
    known = known & exists & output['identity_support']
    on_path = path_stratum(points, batch['hist'], batch['hmask'], output['identity_predicted_points'].detach())
    count = planes*k
    train = known & ~batch['identity_synthetic'].bool()[:, None] if exclude_synthetic else known
    return dict(label=label, known=known, train=train, on_path=on_path,
                candidate_label=label[:, :count].reshape(b, planes, k), candidate_known=known[:, :count].reshape(b, planes, k),
                samples=slice(count, count+batch['identity_sample_mask'].shape[1]),
                predicted=slice(count+batch['identity_sample_mask'].shape[1], None))


def balance_weights(labels, on_path, valid, balance):
    """Per-candidate BCE weights giving each label half of its stratum, from running fractions.

    ``balance`` (2 strata x 2 labels) holds running counts; stratum 0 is off the trace path,
    1 on it. A label's weight is 0.5 over its running fraction (floored), so in expectation
    neither the path position nor its interaction with the label can lower the loss.
    """
    fraction = balance/balance.sum(-1, keepdim=True).clamp_min(1e-6)
    weight = .5/fraction.clamp_min(BALANCE_FLOOR)
    stratum, label = on_path.long(), labels.long()
    return torch.where(valid, weight[stratum, label], 0.)


def update_balance(balance, labels, on_path, valid):
    counts = torch.zeros_like(balance)
    index = on_path.long()*2+labels.long()
    counts.view(-1).scatter_add_(0, index[valid], torch.ones_like(index[valid], dtype=balance.dtype))
    return balance*BALANCE_DECAY+counts


def verification_loss(logits, targets, balance):
    """Per-state loss: balanced BCE over every labeled query, plus a listwise softmax per crossing plane.

    The listwise set holds the original fiber's crossing (index 0) and every labeled negative
    (the head axis only when negative), so after a switch the original fiber must beat the
    on-path point. Returns the per-state loss and which planes entered the listwise term.
    """
    known = targets['train']
    weights = balance_weights(targets['label'], targets['on_path'], known, balance)
    bce = F.binary_cross_entropy_with_logits(logits, targets['label'].float(), reduction='none')
    pointwise = (weights*bce).sum(-1)/known.sum(-1).clamp_min(1)
    label = targets['candidate_label']
    valid = targets['candidate_known'] & known[:, :label[0].numel()].reshape(label.shape)
    candidates = logits[:, :label[0].numel()].reshape(label.shape)
    members = valid & (~label | (torch.arange(label.shape[-1], device=label.device) == 0))
    planes = members[..., 0] & label[..., 0] & members[..., 1:].any(-1)
    masked = candidates.masked_fill(~members, -1e4)
    softmax = torch.where(planes, torch.logsumexp(masked, -1)-masked[..., 0], 0.)
    return pointwise+softmax.sum(-1)/planes.sum(-1).clamp_min(1), planes


# Head-axis groups: the trace on its original fiber, or switched recently / long ago (memory only).
VERIFY_GROUPS = ('own', 'switched_recent', 'switched_old', 'switched_old_synthetic')
# Original fiber vs head-axis point after a long-ago switch.
VERIFY_PAIRS = ('pair_old', 'pair_old_synthetic')
# Dense samples by path stratum and label, and the model's own predicted points by label.
VERIFY_DENSE = ('dense_path_on', 'dense_path_off', 'dense_away_on', 'dense_away_off', 'predicted_on', 'predicted_off')
# Memory as read, another state's memory (shuffled control) and no memory (current crop only).
VERIFY_KINDS = (('', 'identity_logits'), ('shuffled_', 'identity_shuffled_logits'), ('empty_', 'identity_empty_logits'))
VERIFY_METRICS = ('verify_states', 'verify_loss_sum', 'verify_listwise_planes', 'verify_listwise_correct',
                  *(f'verify_{prefix}{name}_{kind}' for prefix, _ in VERIFY_KINDS
                    for name in VERIFY_GROUPS+VERIFY_PAIRS+VERIFY_DENSE for kind in ('count', 'correct')))


def verification_metrics(output, batch, targets, per_state, planes):
    """Device counts for the interval log (see VERIFY_METRICS)."""
    label, known = targets['candidate_label'], targets['candidate_known']
    axis_valid, axis_label = known[..., HEAD_AXIS], label[..., HEAD_AXIS]
    old = (batch['identity_departure_age'] > MEMORY_ONLY_AGE)[:, None]
    synthetic = batch['identity_synthetic'].bool()[:, None]
    switched = axis_valid & ~axis_label
    groups = dict(own=axis_valid & axis_label, switched_recent=switched & ~old,
                  switched_old=switched & old & ~synthetic, switched_old_synthetic=switched & old & synthetic)
    pairs = dict(pair_old=switched & known[..., 0] & old & ~synthetic,
                 pair_old_synthetic=switched & known[..., 0] & old & synthetic)
    samples, predicted = targets['samples'], targets['predicted']
    flat_known, flat_label, on_path = targets['known'], targets['label'], targets['on_path']
    dense = dict(dense_path_on=(samples, on_path & flat_label), dense_path_off=(samples, on_path & ~flat_label),
                 dense_away_on=(samples, ~on_path & flat_label), dense_away_off=(samples, ~on_path & ~flat_label),
                 predicted_on=(predicted, flat_label), predicted_off=(predicted, ~flat_label))
    count = label[0].numel()
    logits = output['identity_logits'].detach()
    candidates = logits[:, :count].reshape(label.shape)
    members = known & (~label | (torch.arange(label.shape[-1], device=label.device) == 0))
    others = candidates.masked_fill(~members, -torch.inf)[..., 1:].amax(-1)
    out = dict(verify_states=flat_known.any(-1).sum(), verify_loss_sum=per_state.detach().sum(),
               verify_listwise_planes=planes.sum(), verify_listwise_correct=(planes & (candidates[..., 0] > others)).sum())
    for prefix, key in VERIFY_KINDS:
        value = output[key].detach()
        usable = output['identity_shuffled_valid'][:, None] if prefix == 'shuffled_' else torch.ones_like(old)
        axis = value[:, :count].reshape(label.shape)
        correct = (axis[..., HEAD_AXIS] > 0) == axis_label
        for name, selected in groups.items():
            out[f'verify_{prefix}{name}_count'] = (selected & usable).sum()
            out[f'verify_{prefix}{name}_correct'] = (selected & usable & correct).sum()
        for name, selected in pairs.items():
            out[f'verify_{prefix}{name}_count'] = (selected & usable).sum()
            out[f'verify_{prefix}{name}_correct'] = (selected & usable & (axis[..., 0] > axis[..., HEAD_AXIS])).sum()
        right = (value > 0) == flat_label
        for name, (part, selected) in dense.items():
            chosen = (flat_known & selected & usable)[:, part]
            out[f'verify_{prefix}{name}_count'] = chosen.sum()
            out[f'verify_{prefix}{name}_correct'] = (chosen & right[:, part]).sum()
    return out
