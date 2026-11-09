"""JEM objective arithmetic: alignment, logical-batch global collapse, structural collapse.

All reductions run in FP32 with autocast disabled. Model outputs arrive already
gathered at exact integer token correspondences; nothing here interpolates.

Conventions (see plans/JEM_impl.md §5):
- Alignment per student view and depth: 0.2 * mean visible KL + 0.8 * mean masked KL
  over overlapping valid tokens. An empty subset contributes a graph-connected zero
  and the other coefficient is not renormalized.
- Global collapse: for each depth k, entropy group g (a crop type or, for the slot
  ablation, one slot) and final student position i, the marginal over every valid
  probability vector in the logical batch (all ranks and source chunks). Positions
  are never pooled. R[k,g] = mean over positions with M>0 of
  (sum a*H(q) / M - H(mu)); R = sum_k depth_weight[k] sum_g weight[g] * R[k,g].
- The gradient of R is evaluated exactly with detached sufficient statistics: each
  rank backpropagates W/M * (C_local + S_local . stop_gradient(log mu + 1)), whose
  DDP average is the full-batch gradient. Its scalar value is not the objective.
- Structural collapse: ordered off-diagonal pairs within one student view,
  (a_i.a_j - b_i.b_j)^2 averaged uniformly, or per distance band (opt-in).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


@contextmanager
def fp32_reductions(device_type: str):
    """Disable autocast so reductions run in the tensors' (FP32) precision."""
    with torch.autocast(device_type=device_type, enabled=False):
        yield


# --------------------------------------------------------------------------- schedules

def global_collapse_weight(updates: int, warmup: int, *, start: float = 1.0, end: float = 0.1) -> float:
    """w_global(t) = start + (end - start) * clamp(t / W, 0, 1) over completed updates t."""
    if warmup <= 0:
        return float(end)
    fraction = min(max(updates / warmup, 0.0), 1.0)
    return float(start + (end - start) * fraction)


def structural_weight(updates: int, warmup: int, total: int, *, warm: float = 0.05, final: float = 1.0) -> float:
    """0 -> warm linearly over W updates, then warm -> final linearly until T."""
    if total <= warmup:
        raise ValueError(f"the structural schedule needs total updates > warmup, got {total} <= {warmup}")
    if warmup > 0 and updates <= warmup:
        return float(warm * max(updates, 0) / warmup)
    fraction = min(max((updates - warmup) / (total - warmup), 0.0), 1.0)
    return float(warm + (final - warm) * fraction)


# --------------------------------------------------------------------------- alignment

def subset_means(values: torch.Tensor, rows: torch.Tensor, masked: torch.Tensor, n_rows: int):
    """Per-row means of `values` over visible and masked entries, graph connected.

    Empty subsets give a zero mean that stays connected to `values`.
    Returns (visible_mean, masked_mean, visible_count, masked_count), each [n_rows].
    """
    key = rows * 2 + masked.long()
    sums = values.new_zeros(n_rows * 2).index_add(0, key, values)
    counts = torch.bincount(key, minlength=n_rows * 2).to(values.dtype)
    means = (sums / counts.clamp_min(1)).view(n_rows, 2)
    counts = counts.view(n_rows, 2)
    return means[:, 0], means[:, 1], counts[:, 0], counts[:, 1]


def alignment_from_means(visible_mean, masked_mean, *, visible_weight=0.2, masked_weight=0.8):
    return visible_weight * visible_mean + masked_weight * masked_mean


def kl_divergence(teacher_logits, student_log_probs, teacher_temperature):
    """KL(p_teacher || q_student) per row, both [n, K]; zero-safe via log-softmax."""
    log_p = F.log_softmax(teacher_logits.float() / teacher_temperature, dim=-1)
    return (log_p.exp() * (log_p - student_log_probs)).sum(-1)


# --------------------------------------------------------------------------- entropy

@dataclass(frozen=True)
class EntropyGroup:
    """Probability vectors sharing one marginal: a crop type or a single slot."""
    name: str
    view_type: str  # "global" or "local"
    slots: tuple[int, ...]
    weight: float


def entropy_groups(scope: str, *, n_global_slots: int, n_local_slots: int,
                   global_weight: float = 0.2, local_weight: float = 0.8) -> list[EntropyGroup]:
    if scope == "crop_type":
        return [EntropyGroup("global", "global", tuple(range(n_global_slots)), global_weight),
                EntropyGroup("local", "local", tuple(range(n_local_slots)), local_weight)]
    if scope == "slot":
        n_slots = n_global_slots + n_local_slots
        return ([EntropyGroup(f"global{s}", "global", (s,), 1.0 / n_slots) for s in range(n_global_slots)]
                + [EntropyGroup(f"local{s}", "local", (s,), 1.0 / n_slots) for s in range(n_local_slots)])
    raise ValueError(f"entropy_view_scope must be 'crop_type' or 'slot', got {scope!r}")


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """-sum q log q from log-probabilities; underflowed categories contribute zero."""
    return -(log_probs.exp() * log_probs).sum(-1)


def slot_group_index(groups: Sequence[EntropyGroup], view_type: str) -> list[int]:
    """Index of each slot's group among the groups of `view_type` (the groups partition the slots)."""
    members = [group for group in groups if group.view_type == view_type]
    n_slots = max(slot for group in members for slot in group.slots) + 1
    index = [-1] * n_slots
    for position, group in enumerate(members):
        for slot in group.slots:
            if index[slot] != -1:
                raise ValueError(f"slot {slot} of {view_type} views belongs to several entropy groups")
            index[slot] = position
    if -1 in index:
        raise ValueError(f"every {view_type} slot must belong to one entropy group, got {index}")
    return index


def accumulation_terms(log_q: torch.Tensor, valid: torch.Tensor, group: torch.Tensor, n_groups: int,
                       sources: torch.Tensor):
    """Per-group increments of the marginal statistics, with static shapes (no host syncs).

    log_q [R, N, K] FP32; valid [R, N] bool; group [R] (group index of each row); sources [R].
    Returns S [G, N, K], C [G, N], M [G, N] and D [G, N]: the number of distinct sources
    with at least one valid vector of the group at the position.
    """
    q = log_q.exp()
    h = entropy(log_q)
    member = group[None, :] == torch.arange(n_groups, device=group.device)[:, None]  # [G, R]
    a = valid.to(q.dtype)[None] * member[..., None].to(q.dtype)  # [G, R, N]
    S = torch.einsum("grn,rnk->gnk", a, q)
    C = (a * h[None]).sum(1)
    M = a.sum(1)
    # Count each source once, at its first row: does any row of that source have a valid vector?
    same = sources[:, None] == sources[None, :]  # [R, R]
    order = torch.arange(len(sources), device=sources.device)
    first = ~(same & (order[None, :] < order[:, None])).any(1)
    reached = torch.einsum("rs,gsn->grn", same.to(q.dtype), a) > 0
    D = (reached & first[None, :, None]).to(q.dtype).sum(1)
    return S, C, M, D


class EntropyStatistics:
    """Detached sufficient statistics of the logical-batch marginals.

    For every (depth, group): S[i, c] = sum a q, C[i] = sum a H(q), M[i] = sum a,
    and D[i] = number of distinct sources with at least one valid vector.
    """

    def __init__(self, groups: Sequence[EntropyGroup], depths: Sequence[int],
                 positions: Mapping[str, int], categories: int, device) -> None:
        self.groups = list(groups)
        self.depths = list(depths)
        self.categories = int(categories)
        self.S, self.C, self.M, self.D = {}, {}, {}, {}
        self.slot_group = {view_type: torch.tensor(slot_group_index(self.groups, view_type), device=device)
                           for view_type in sorted({group.view_type for group in self.groups})}
        for depth in self.depths:
            for group in self.groups:
                n = int(positions[group.view_type])
                key = (depth, group.name)
                self.S[key] = torch.zeros(n, categories, device=device, dtype=torch.float32)
                self.C[key] = torch.zeros(n, device=device, dtype=torch.float32)
                self.M[key] = torch.zeros(n, device=device, dtype=torch.float32)
                self.D[key] = torch.zeros(n, device=device, dtype=torch.float32)

    def groups_for(self, view_type: str) -> list[EntropyGroup]:
        return [group for group in self.groups if group.view_type == view_type]

    @torch.no_grad()
    def accumulate(self, depth: int, view_type: str, log_q: torch.Tensor, valid: torch.Tensor,
                   slots: torch.Tensor, sources: torch.Tensor) -> None:
        """log_q [rows, N, K] FP32; valid [rows, N] bool; slots/sources [rows] long."""
        group = self.slot_group[view_type].to(slots.device)[slots]
        self.add_terms(depth, view_type, accumulation_terms(log_q.float(), valid, group,
                                                            len(self.groups_for(view_type)), sources))

    @torch.no_grad()
    def add_terms(self, depth: int, view_type: str, terms) -> None:
        """Add accumulation_terms (stacked in groups_for(view_type) order)."""
        S, C, M, D = terms
        for index, group in enumerate(self.groups_for(view_type)):
            key = (depth, group.name)
            self.S[key] += S[index]
            self.C[key] += C[index]
            self.M[key] += M[index]
            self.D[key] += D[index]

    def all_reduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        # One flat buffer per statistic keeps the collective order identical on every rank.
        for table in (self.S, self.C, self.M, self.D):
            keys = sorted(table)
            flat = torch.cat([table[key].reshape(-1) for key in keys])
            dist.all_reduce(flat)
            offset = 0
            for key in keys:
                count = table[key].numel()
                table[key].copy_(flat[offset:offset + count].view_as(table[key]))
                offset += count

    def marginal(self, key) -> torch.Tensor:
        counts = self.M[key]
        return torch.where(counts[:, None] > 0, self.S[key] / counts.clamp_min(1)[:, None],
                           torch.full_like(self.S[key], 1.0 / self.categories))

    def present(self, key) -> torch.Tensor:
        return self.M[key] > 0

    def group_loss(self, key) -> dict[str, torch.Tensor]:
        """True R[k, g] with its marginal/conditional entropies (means over present positions)."""
        present = self.present(key)
        mu = self.marginal(key)
        marginal_entropy = -torch.special.xlogy(mu, mu).sum(-1)
        conditional = self.C[key] / self.M[key].clamp_min(1)
        values = conditional - marginal_entropy
        if bool(present.any()):
            loss = values[present].mean()
            return {"loss": loss, "marginal_entropy": marginal_entropy[present].mean(),
                    "conditional_entropy": conditional[present].mean(),
                    "min_vectors": self.M[key][present].min(), "min_sources": self.D[key][present].min(),
                    "positions": present.float().sum(),
                    "perplexity": torch.exp(-torch.special.xlogy(mu[present].mean(0), mu[present].mean(0)).sum())}
        zero = values.sum() * 0
        return {"loss": zero, "marginal_entropy": zero, "conditional_entropy": zero,
                "min_vectors": zero, "min_sources": zero, "positions": zero, "perplexity": zero}

    def coefficients(self, key, scale: float, reference=None, batch_weight: float = 1.0):
        """Per-position surrogate coefficient scale / (P_present * M), d = log mu_hat + 1, and log mu.

        mu_hat is this batch's marginal mu, or with ``reference`` (r [N, K], present [N]) the
        mixture batch_weight * mu + (1 - batch_weight) * r wherever r is present. d is zero
        where mu_hat underflowed to zero. log mu is always the batch marginal's (for the
        loss attribution of the batch objective).
        """
        present = self.present(key)
        n_present = present.float().sum().clamp_min(1)
        coef = torch.where(present, scale / (n_present * self.M[key].clamp_min(1)), torch.zeros_like(self.M[key]))
        mu = self.marginal(key)
        log_mu = _safe_log(mu)
        if reference is None:
            return coef, log_mu + 1.0, log_mu
        r, r_present = reference
        mixed = torch.where(r_present[:, None], batch_weight * mu + (1.0 - batch_weight) * r, mu)
        return coef, _safe_log(mixed) + 1.0, log_mu


def _safe_log(p: torch.Tensor) -> torch.Tensor:
    """log p, and zero where p is zero."""
    return torch.where(p > 0, p.clamp_min(torch.finfo(p.dtype).tiny).log(), torch.zeros_like(p))


def entropy_surrogate(log_q: torch.Tensor, valid: torch.Tensor, coef: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """sum_rows sum_i a coef_i (H(q_i) + q_i . d_i): exact first-order gradient, not the loss value."""
    q = log_q.exp()
    per_token = -(q * log_q).sum(-1) + (q * d).sum(-1)
    return (per_token * valid.float() * coef).sum()


def entropy_attribution(log_q: torch.Tensor, valid: torch.Tensor, coef: torch.Tensor, log_mu: torch.Tensor) -> torch.Tensor:
    """Per-row share of the true loss: sum_i a coef_i (H(q_i) + q_i . log mu_i) = -KL(q || mu).

    With coef = weight / (P_present * M) these sum over all rows of the logical batch to R.
    """
    q = log_q.exp()
    per_token = -(q * log_q).sum(-1) + (q * log_mu).sum(-1)
    return (per_token * valid.float() * coef).sum(-1)


# --------------------------------------------------------------------------- structural

def structural_uniform_contraction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """sum_{i != j} (a_i.a_j - b_i.b_j)^2 / (n (n - 1)) without forming n x n Grams.

    With D = A - B and S = A + B, A A^T - B B^T = (D S^T + S D^T) / 2, so
        sum_{i,j} = (<D^T D, S^T S> + <S^T D, (S^T D)^T>) / 2,
    and the diagonal i = j is (d_i . s_i)^2. Every term scales with |D|, avoiding the
    cancellation of ||A^T A||^2 + ||B^T B||^2 - 2 ||A^T B||^2 when the Grams nearly agree.
    """
    n = a.shape[0]
    if n < 2:
        return a.sum() * 0
    # Widths may differ: zero padding leaves each Gram matrix unchanged.
    width = max(a.shape[1], b.shape[1])
    a = F.pad(a, (0, width - a.shape[1]))
    b = F.pad(b, (0, width - b.shape[1]))
    d = a - b
    s = a + b
    dd = d.T @ d
    ss = s.T @ s
    sd = s.T @ d
    full = 0.5 * ((dd * ss).sum() + (sd * sd.T).sum())
    diagonal = (d * s).sum(-1).square().sum()
    return (full - diagonal) / (n * (n - 1))


def overlap_capacity(count: int, n: int, multiple: int = 64) -> int:
    """Compacted token capacity: `count` rounded up to `multiple` (at least one), at most n."""
    return min(n, max(1, -(-int(count) // multiple)) * multiple)


def overlap_order(overlap: torch.Tensor, capacity: int) -> torch.Tensor:
    """Token positions [R, capacity]: each row's overlapping tokens first, in token order.

    With capacity >= every row's overlap count, positions past a row's count are
    non-overlapping tokens (masked out by gathering `overlap` with the same order).
    """
    # Contiguous for every capacity, so compiled graphs see one layout.
    return torch.sort((~overlap).to(torch.uint8), dim=-1, stable=True).indices[:, :capacity].contiguous()


def structural_uniform_contraction_rows(a: torch.Tensor, b: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """structural_uniform_contraction of every row over its kept tokens, with static shapes.

    a [R, N, Ca], b [R, N, Cb], keep [R, N] bool -> [R]. Zeroing the dropped tokens in
    D = A - B and S = A + B removes them from every Gram product and the diagonal; rows
    with fewer than two kept tokens give zero.
    """
    width = max(a.shape[-1], b.shape[-1])
    a = F.pad(a, (0, width - a.shape[-1]))
    b = F.pad(b, (0, width - b.shape[-1]))
    m = keep.unsqueeze(-1).to(a.dtype)
    d = (a - b) * m
    s = (a + b) * m
    dd = d.mT @ d
    ss = s.mT @ s
    sd = s.mT @ d
    full = 0.5 * ((dd * ss).sum((-2, -1)) + (sd * sd.mT).sum((-2, -1)))
    diagonal = (d * s).sum(-1).square().sum(-1)
    n = keep.sum(-1).to(a.dtype)
    return torch.where(n >= 2, (full - diagonal) / (n * (n - 1)).clamp_min(1), torch.zeros_like(full))


def distance_metric(spacing: Sequence[float], patch: Sequence[int]) -> tuple[torch.Tensor, float]:
    """Axis weights (s_k / ell)^2 and ell for token cells of physical edge s_k = P_k * spacing_k.

    ell is the edge of the cube with one token cell's volume. Equal edges give weights
    of exactly one, so isotropic band membership is decided in exact integer units.
    """
    edges = [float(p) * float(s) for p, s in zip(patch, spacing)]
    if any(edge <= 0 for edge in edges):
        raise ValueError(f"token cell edges must be positive, got {edges}")
    if all(edge == edges[0] for edge in edges):
        return torch.ones(3, dtype=torch.float64), edges[0]
    ell = math.prod(edges) ** (1.0 / 3.0)
    return torch.tensor([(edge / ell) ** 2 for edge in edges], dtype=torch.float64), ell


def _band_ids(coords_i, coords_j, metric, edges_squared):
    """Band index (0..len(edges)) of every pair, from squared distances in units of ell."""
    delta = (coords_i[:, None, :] - coords_j[None, :, :]).to(torch.float64)
    distance = (delta.square() * metric.to(delta.device)).sum(-1)
    return torch.bucketize(distance, edges_squared.to(delta.device), right=False)


def structural_band_counts(coords: torch.Tensor, metric: torch.Tensor, edges: Sequence[float],
                           tile: int = 2048) -> torch.Tensor:
    """Ordered off-diagonal pair counts per band: near (0, e0], (e0, e1], ..., (e_last, inf)."""
    edges_squared = torch.tensor([float(edge) ** 2 for edge in edges], dtype=torch.float64)
    counts = torch.zeros(len(edges) + 1, dtype=torch.float64, device=coords.device)
    n = coords.shape[0]
    for i0 in range(0, n, tile):
        ci = coords[i0:i0 + tile]
        for j0 in range(0, n, tile):
            band = _band_ids(ci, coords[j0:j0 + tile], metric, edges_squared)
            if i0 == j0:
                band.fill_diagonal_(len(edges) + 1)
            counts += torch.bincount(band.flatten(), minlength=len(edges) + 2)[: len(edges) + 1].to(torch.float64)
    return counts


def band_pair_weights(counts: torch.Tensor, band_weights: Sequence[float]) -> torch.Tensor:
    """w'_s / |E_s| per band, renormalizing weights over nonempty bands; zeros if none remain."""
    weights = torch.as_tensor(band_weights, dtype=torch.float64, device=counts.device)
    nonempty = counts > 0
    total = (weights * nonempty).sum()
    if total <= 0:
        return torch.zeros_like(counts)
    return torch.where(nonempty, weights / total / counts.clamp_min(1), torch.zeros_like(counts))


class _TiledPairLoss(torch.autograd.Function):
    """sum_{i != j} w_ij (a_i.a_j - b_i.b_j)^2 in bounded-memory tiles with an analytical backward.

    w_ij = pair_weight[band(i, j)] when coordinates are given, else pair_weight[0].
    Only `a` receives gradients: d/da_i = sum_{j != i} 4 w_ij e_ij a_j (w symmetric).
    """

    @staticmethod
    def forward(ctx, a, b, coords, metric, edges_squared, pair_weight, tile):
        ctx.save_for_backward(a, b, coords, metric, edges_squared, pair_weight)
        ctx.tile = tile
        total = a.new_zeros((), dtype=torch.float64)
        for i0, j0, weights, error in _pair_tiles(a, b, coords, metric, edges_squared, pair_weight, tile):
            total += (weights * error.square()).sum(dtype=torch.float64)
        return total.to(a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, coords, metric, edges_squared, pair_weight = ctx.saved_tensors
        grad = torch.zeros_like(a)
        tile = ctx.tile
        for i0, j0, weights, error in _pair_tiles(a, b, coords, metric, edges_squared, pair_weight, tile):
            grad[i0:i0 + tile] += (4 * weights * error) @ a[j0:j0 + tile]
        return grad * grad_output, None, None, None, None, None, None


def _pair_tiles(a, b, coords, metric, edges_squared, pair_weight, tile):
    n = a.shape[0]
    for i0 in range(0, n, tile):
        ai, bi = a[i0:i0 + tile], b[i0:i0 + tile]
        for j0 in range(0, n, tile):
            error = ai @ a[j0:j0 + tile].T - bi @ b[j0:j0 + tile].T
            if coords is None or coords.numel() == 0:
                weights = torch.full_like(error, float(pair_weight[0]))
            else:
                band = _band_ids(coords[i0:i0 + tile], coords[j0:j0 + tile], metric, edges_squared)
                weights = pair_weight.to(error.dtype)[band.clamp_max(len(pair_weight) - 1)]
            if i0 == j0:
                weights.fill_diagonal_(0)
            yield i0, j0, weights, error


def structural_uniform_tiled(a: torch.Tensor, b: torch.Tensor, tile: int = 2048) -> torch.Tensor:
    n = a.shape[0]
    if n < 2:
        return a.sum() * 0
    weight = torch.tensor([1.0 / (n * (n - 1))], dtype=a.dtype, device=a.device)
    empty = torch.empty(0, device=a.device)
    return _TiledPairLoss.apply(a, b, None, empty, empty, weight, tile)


def structural_distance_stratified(a: torch.Tensor, b: torch.Tensor, coords: torch.Tensor, metric: torch.Tensor,
                                   edges: Sequence[float], band_weights: Sequence[float], tile: int = 2048,
                                   counts: torch.Tensor | None = None):
    """sum_s w'_s * mean over band s of e_ij^2. Returns (loss, band counts, pair weights).

    edges are in units of ell (cube edge of one token cell's volume) and increasing.
    """
    if counts is None:
        counts = structural_band_counts(coords, metric, edges, tile)
    pair_weight = band_pair_weights(counts, band_weights)
    if a.shape[0] < 2 or not bool((pair_weight > 0).any()):
        return a.sum() * 0, counts, pair_weight
    edges_squared = torch.tensor([float(edge) ** 2 for edge in edges], dtype=torch.float64, device=a.device)
    loss = _TiledPairLoss.apply(a, b, coords, metric.to(a.device), edges_squared,
                                pair_weight.to(a.dtype), tile)
    return loss, counts, pair_weight


def structural_dense_reference(a, b, pair_weights=None):
    """Dense n x n reference for tests."""
    n = a.shape[0]
    error = (a @ a.T - b @ b.T).square()
    weights = torch.ones_like(error) / max(n * (n - 1), 1) if pair_weights is None else pair_weights
    weights = weights * (1 - torch.eye(n, dtype=error.dtype, device=error.device))
    return (weights * error).sum()


def nonaffine_target(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Structural target: LayerNorm without affine parameters, then L2 normalization."""
    features = F.layer_norm(features.float(), features.shape[-1:], eps=eps)
    return F.normalize(features, dim=-1)


def flatten_depth_weights(depth_weights: Mapping[int, float]) -> list[tuple[int, float]]:
    return sorted((int(depth), float(weight)) for depth, weight in depth_weights.items())


def summed(values: Iterable[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    total = like.new_zeros(())
    for value in values:
        total = total + value
    return total
