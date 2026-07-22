"""Batched soft monotonic sequence alignment (three-state pair-HMM).

Aligns a modeled winding sequence (length ``N`` per ray) against a detected
band sequence (length ``M`` per ray) with three transitions:

- ``match``: winding ``i`` corresponds to band ``j``;
- ``missing``: advance the winding without consuming a band (a hole in the
  observations);
- ``extra``: advance the band without consuming a winding (a spurious or
  out-of-model band).

The DP is a full three-state pair-HMM forward/backward with an explicit
state-transition cost structure, so affine gap costs (separate gap-open and
gap-extend) are a parameter setting rather than a code change. Semi-global
(free end gap) boundary conditions are expressed as per-band extra costs:
the caller passes zero skip cost for bands outside the central modeled
interval (the observation sequence covers a padded ray), so leading and
trailing padding bands are consumed for free. Zero cost is attached to the
band, not to the path position - a positional free end gap would let the
last winding capture a displaced interior band while dumping its true band
into the free gap.

Banding: cells whose match is disallowed carry an effectively impossible
match potential and receive zero probability mass; the caller expresses the
absolute phase window (and any other mask) as ``+inf`` entries in
``match_cost``.

This module has no volume or transform dependencies. All tensors are plain
``torch`` tensors; sequences are padded with validity masks and valid entries
must be prefix-contiguous.
"""

import torch


# Effectively-impossible log potential. A large negative *finite* constant is
# used instead of -inf so that logsumexp/logcumsumexp gradients stay defined
# (exp(NEG - anything_reachable) underflows to exactly zero in float32, while
# -inf entries would generate 0 * nan products in the backward pass).
NEG = -1.0e30


def _lse(*terms):
    return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


def _row_scan_lse(source, open_col, cum_extend):
    """Left-to-right within-row gap scan with per-column costs.

    Returns ``out[:, j] = logsumexp_{j' <= j-1} source[:, j'] +
    open[j' + 1] + sum(extend[j' + 2 .. j])`` with ``out[:, 0] = NEG``: the
    log-sum over entering the gap state at column ``j' + 1`` and extending it
    up to column ``j``. ``open_col``/``cum_extend`` are column-indexed
    ``[B, cols]`` log potentials (``cum_extend`` is the inclusive prefix sum
    of the extend potentials).
    """
    batch, cols = source.shape
    shifted_open = torch.cat(
        [open_col[:, 1:],
         torch.full([batch, 1], NEG, device=source.device,
                    dtype=source.dtype)], dim=1)
    shifted_cum = torch.cat(
        [cum_extend[:, 1:],
         cum_extend[:, -1:]], dim=1)
    adjusted = source + shifted_open - shifted_cum
    run = torch.logcumsumexp(adjusted, dim=1)
    lead = torch.full([batch, 1], NEG, device=source.device, dtype=source.dtype)
    return torch.cat([lead, cum_extend[:, 1:] + run[:, :-1]], dim=1)


def _row_scan_max(source, open_col, cum_extend):
    """Max-product version of :func:`_row_scan_lse` with source indices."""
    batch, cols = source.shape
    shifted_open = torch.cat(
        [open_col[:, 1:],
         torch.full([batch, 1], NEG, device=source.device,
                    dtype=source.dtype)], dim=1)
    shifted_cum = torch.cat(
        [cum_extend[:, 1:], cum_extend[:, -1:]], dim=1)
    adjusted = source + shifted_open - shifted_cum
    run, run_index = torch.cummax(adjusted, dim=1)
    lead = torch.full([batch, 1], NEG, device=source.device, dtype=source.dtype)
    values = torch.cat([lead, cum_extend[:, 1:] + run[:, :-1]], dim=1)
    index = torch.cat([torch.zeros_like(run_index[:, :1]), run_index[:, :-1]],
                      dim=1)
    return values, index


def _prepared_potentials(match_cost, model_valid, band_valid, *,
                         missing_open, missing_extend, extra_open,
                         extra_extend, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    batch, n_model, n_bands = match_cost.shape
    dtype = match_cost.dtype
    device = match_cost.device
    if n_bands == 0:
        match_cost = torch.full([batch, n_model, 1], float('inf'),
                                device=device, dtype=dtype)
        band_valid = torch.zeros([batch, 1], dtype=torch.bool, device=device)
        n_bands = 1
    if n_model == 0:
        match_cost = torch.full([batch, 1, n_bands], float('inf'),
                                device=device, dtype=dtype)
        model_valid = torch.zeros([batch, 1], dtype=torch.bool, device=device)
        n_model = 1
    allowed = (torch.isfinite(match_cost)
               & model_valid[:, :, None] & band_valid[:, None, :])
    lp_match = torch.where(
        allowed, -match_cost / temperature,
        torch.full_like(match_cost, NEG))
    n_windings = model_valid.to(torch.long).sum(dim=1)
    m_bands = band_valid.to(torch.long).sum(dim=1)

    def per_band_potential(value):
        # Scalar or per-band [B, M] cost -> column-indexed [B, M + 1]
        # log potential (column j is band j - 1; column 0 is unused).
        column = torch.zeros([batch, n_bands + 1], dtype=dtype, device=device)
        column[:, 1:] = -torch.as_tensor(
            value, dtype=dtype, device=device).expand(batch, n_bands) \
            / temperature
        return column

    extra_open_col = per_band_potential(extra_open)
    extra_extend_col = per_band_potential(extra_extend)
    costs = {
        'mo': -float(missing_open) / temperature,
        'me': -float(missing_extend) / temperature,
        'eo': extra_open_col,
        'ee': extra_extend_col,
        'cum_ee': extra_extend_col.cumsum(dim=1),
    }
    return lp_match, n_windings, m_bands, costs, (batch, n_model, n_bands)


def _forward(lp_match, n_windings, costs, shape, dtype):
    batch, n_model, n_bands = shape
    device = lp_match.device
    cols = n_bands + 1
    neg_col = torch.full([batch, 1], NEG, device=device, dtype=dtype)
    row_m = torch.cat(
        [torch.zeros_like(neg_col),
         torch.full([batch, n_bands], NEG, device=device, dtype=dtype)], dim=1)
    row_x = torch.full([batch, cols], NEG, device=device, dtype=dtype)
    row_y = _row_scan_lse(row_m, costs['eo'], costs['cum_ee'])
    rows_m, rows_x, rows_y = [row_m], [row_x], [row_y]
    for i in range(1, n_model + 1):
        prev_m, prev_x, prev_y = row_m, row_x, row_y
        incoming = _lse(prev_m, prev_x, prev_y)
        row_m = torch.cat(
            [neg_col, incoming[:, :-1] + lp_match[:, i - 1, :]], dim=1)
        row_x = _lse(prev_m + costs['mo'], prev_y + costs['mo'],
                     prev_x + costs['me'])
        source = _lse(row_m, row_x)
        row_y = _row_scan_lse(source, costs['eo'], costs['cum_ee'])
        rows_m.append(row_m)
        rows_x.append(row_x)
        rows_y.append(row_y)
    return (torch.stack(rows_m, dim=1), torch.stack(rows_x, dim=1),
            torch.stack(rows_y, dim=1))


def _backward(lp_match, n_windings, m_bands, costs, shape, dtype):
    batch, n_model, n_bands = shape
    device = lp_match.device
    cols = n_bands + 1
    col_idx = torch.arange(cols, device=device)
    inject = torch.where(
        col_idx[None, :] == m_bands[:, None],
        torch.zeros([batch, cols], device=device, dtype=dtype),
        torch.full([batch, cols], NEG, device=device, dtype=dtype))
    neg_row = torch.full([batch, cols], NEG, device=device, dtype=dtype)
    rows_m = [None] * (n_model + 1)
    rows_x = [None] * (n_model + 1)
    rows_y = [None] * (n_model + 1)
    next_m, next_x = neg_row, neg_row
    for i in range(n_model, -1, -1):
        row_inject = torch.where(
            (n_windings == i)[:, None], inject, neg_row)
        if i < n_model:
            to_match = torch.cat(
                [next_m[:, 1:] + lp_match[:, i, :],
                 torch.full([batch, 1], NEG, device=device, dtype=dtype)],
                dim=1)
        else:
            to_match = neg_row
        to_x_open = next_x + costs['mo']
        to_x_extend = next_x + costs['me']
        # Right-to-left gap scan with per-band costs: exits from the extra
        # state are matches, missing transitions (paying open), or
        # termination. beta_Y[j] = lse_{j'' >= j}(exit[j''] + cum_ee[j''])
        # - cum_ee[j].
        exit_y = _lse(to_match, to_x_open, row_inject)
        adjusted = torch.flip(exit_y + costs['cum_ee'], dims=[1])
        run = torch.logcumsumexp(adjusted, dim=1)
        row_y = torch.flip(run, dims=[1]) - costs['cum_ee']
        neg_tail = torch.full([batch, 1], NEG, device=device, dtype=dtype)
        to_y = (torch.cat([costs['eo'][:, 1:], neg_tail], dim=1)
                + torch.cat([row_y[:, 1:], neg_tail], dim=1))
        row_m = _lse(to_match, to_x_open, to_y, row_inject)
        row_x = _lse(to_match, to_x_extend, to_y, row_inject)
        rows_m[i], rows_x[i], rows_y[i] = row_m, row_x, row_y
        next_m, next_x = row_m, row_x
    return (torch.stack(rows_m, dim=1), torch.stack(rows_x, dim=1),
            torch.stack(rows_y, dim=1))


def _viterbi(lp_match, n_windings, costs, shape, dtype, keep_pointers):
    batch, n_model, n_bands = shape
    device = lp_match.device
    cols = n_bands + 1
    neg_col = torch.full([batch, 1], NEG, device=device, dtype=dtype)
    row_m = torch.cat(
        [torch.zeros_like(neg_col),
         torch.full([batch, n_bands], NEG, device=device, dtype=dtype)], dim=1)
    row_x = torch.full([batch, cols], NEG, device=device, dtype=dtype)
    row_y, run_index = _row_scan_max(row_m, costs['eo'], costs['cum_ee'])
    rows = [(row_m, row_x, row_y)]
    pointers = []
    if keep_pointers:
        zero_ptr = torch.zeros([batch, cols], dtype=torch.int8, device=device)
        pointers.append((zero_ptr, zero_ptr,
                         torch.zeros([batch, cols], dtype=torch.long,
                                     device=device),
                         zero_ptr))
    for i in range(1, n_model + 1):
        prev_m, prev_x, prev_y = rows[-1]
        stacked_in = torch.stack([prev_m, prev_x, prev_y], dim=0)
        incoming, ptr_m = stacked_in.max(dim=0)
        row_m = torch.cat(
            [neg_col, incoming[:, :-1] + lp_match[:, i - 1, :]], dim=1)
        stacked_x = torch.stack(
            [prev_m + costs['mo'], prev_x + costs['me'],
             prev_y + costs['mo']], dim=0)
        row_x, ptr_x = stacked_x.max(dim=0)
        stacked_src = torch.stack([row_m, row_x], dim=0)
        source, source_arg = stacked_src.max(dim=0)
        row_y, run_index = _row_scan_max(source, costs['eo'], costs['cum_ee'])
        rows.append((row_m, row_x, row_y))
        if keep_pointers:
            ptr_m_full = torch.cat(
                [torch.zeros_like(ptr_m[:, :1]), ptr_m[:, :-1]],
                dim=1).to(torch.int8)
            pointers.append((ptr_m_full, ptr_x.to(torch.int8), run_index,
                             source_arg.to(torch.int8)))
    values = tuple(
        torch.stack([row[state] for row in rows], dim=1) for state in range(3))
    return values, pointers


def _backtrack(values, pointers, n_windings, m_bands, shape):
    batch, n_model, n_bands = shape
    device = n_windings.device
    v_m, v_x, v_y = values
    row = n_windings.clone()
    col = m_bands.clone()
    batch_idx = torch.arange(batch, device=device)
    terminal = torch.stack([
        v_m[batch_idx, row, col], v_x[batch_idx, row, col],
        v_y[batch_idx, row, col]], dim=0)
    state = terminal.argmax(dim=0)
    map_match = torch.full([batch, max(1, n_model)], -1, dtype=torch.long,
                           device=device)
    map_extra = torch.zeros([batch, max(1, n_bands)], dtype=torch.bool,
                            device=device)
    ptr_m = torch.stack([p[0] for p in pointers], dim=1)
    ptr_x = torch.stack([p[1] for p in pointers], dim=1)
    run_index = torch.stack([p[2] for p in pointers], dim=1)
    source_arg = torch.stack([p[3] for p in pointers], dim=1)
    for _ in range(n_model + n_bands + 1):
        active = (row > 0) | (col > 0)
        if not bool(active.any()):
            break
        is_m = active & (state == 0) & (row > 0) & (col > 0)
        is_x = active & (state == 1) & (row > 0)
        is_y = active & (state == 2) & (col > 0)
        stuck = active & ~(is_m | is_x | is_y)
        if bool(stuck.any()):
            # Degenerate all-impossible cells (empty rays); terminate them.
            row = torch.where(stuck, torch.zeros_like(row), row)
            col = torch.where(stuck, torch.zeros_like(col), col)
        if bool(is_m.any()):
            map_match[is_m, (row[is_m] - 1).clamp(min=0)] = col[is_m] - 1
        if bool(is_y.any()):
            map_extra[is_y, (col[is_y] - 1).clamp(min=0)] = True
        next_state = state.clone()
        cell_ptr_m = ptr_m[batch_idx, row, col].to(torch.long)
        cell_ptr_x = ptr_x[batch_idx, row, col].to(torch.long)
        # ptr_x source order was (M, X, Y) -> codes 0, 1, 2 already.
        cell_run = run_index[batch_idx, row, col]
        # The extra state at (i, j) either extends the extra run from
        # (i, j - 1) or opens from the best of match/missing at the column
        # where the run started - so the source state is read at that column.
        opens = cell_run == (col - 1).clamp(min=0)
        cell_source = source_arg[batch_idx, row, cell_run].to(torch.long)
        y_pred = torch.where(opens, cell_source, torch.full_like(state, 2))
        next_state = torch.where(is_m, cell_ptr_m, next_state)
        next_state = torch.where(is_x, cell_ptr_x, next_state)
        next_state = torch.where(is_y, y_pred, next_state)
        row = torch.where(is_m | is_x, row - 1, row)
        col = torch.where(is_m | is_y, col - 1, col)
        state = next_state
    return map_match, map_extra


def soft_align_sequences(match_cost, model_valid, band_valid, *,
                         missing_open, missing_extend, extra_open,
                         extra_extend, temperature,
                         compute_map_path=False):
    """Forward/backward soft alignment of model windings to detected bands.

    Args:
        match_cost: ``[B, N, M]`` differentiable geometric match cost; ``+inf``
            (or non-finite) entries are banded out of the recursion.
        model_valid / band_valid: prefix-contiguous validity masks.
        missing_open/missing_extend: cost per missing winding (model skip);
            tie them for a length-constant gap cost.
        extra_open/extra_extend: cost per extra band (observation skip),
            scalar or per-band ``[B, M]``; pass zero for padding bands
            outside the central interval to realize free end gaps.
        temperature: softmin temperature (log-space); lower is harder.
        compute_map_path: also return the MAP alignment path.

    Returns a dict of per-ray tensors: ``log_partition``, ``match_posterior``
    ``[B, N, M]``, ``missing_posterior`` ``[B, N]``, ``extra_posterior`` /
    ``interior_extra_posterior`` ``[B, M]``, ``winding_entropy`` and
    ``top2_margin`` ``[B, N]``, ``ray_entropy``, ``map_score``,
    ``map_clearance``, expected match/missing/extra counts, and (optionally)
    ``map_match`` (band index per winding, ``-1`` = missing) plus
    ``map_extra``.
    """
    lp_match, n_windings, m_bands, costs, shape = _prepared_potentials(
        match_cost, model_valid, band_valid, missing_open=missing_open,
        missing_extend=missing_extend, extra_open=extra_open,
        extra_extend=extra_extend, temperature=temperature)
    batch, n_model, n_bands = shape
    dtype = lp_match.dtype
    device = lp_match.device
    alpha_m, alpha_x, alpha_y = _forward(
        lp_match, n_windings, costs, shape, dtype)
    beta_m, beta_x, beta_y = _backward(
        lp_match, n_windings, m_bands, costs, shape, dtype)
    batch_idx = torch.arange(batch, device=device)
    log_partition = _lse(
        alpha_m[batch_idx, n_windings, m_bands],
        alpha_x[batch_idx, n_windings, m_bands],
        alpha_y[batch_idx, n_windings, m_bands])
    norm = log_partition[:, None, None]
    match_posterior = torch.exp(
        alpha_m[:, 1:, 1:] + beta_m[:, 1:, 1:] - norm)
    missing_posterior = torch.exp(
        alpha_x[:, 1:, :] + beta_x[:, 1:, :] - norm).sum(dim=2)
    extra_cells = torch.exp(alpha_y[:, :, 1:] + beta_y[:, :, 1:] - norm)
    extra_posterior = extra_cells.sum(dim=1)
    row_idx = torch.arange(n_model + 1, device=device)
    interior_row = (row_idx[None, :] > 0) & (
        row_idx[None, :] < n_windings[:, None])
    interior_extra_posterior = (
        extra_cells * interior_row[:, :, None]).sum(dim=1)

    model_valid_f = (torch.arange(n_model, device=device)[None, :]
                     < n_windings[:, None]).to(dtype)
    band_valid_f = (torch.arange(n_bands, device=device)[None, :]
                    < m_bands[:, None]).to(dtype)
    match_posterior = match_posterior * model_valid_f[:, :, None] \
        * band_valid_f[:, None, :]
    missing_posterior = missing_posterior * model_valid_f
    extra_posterior = extra_posterior * band_valid_f
    interior_extra_posterior = interior_extra_posterior * band_valid_f

    categorical = torch.cat(
        [match_posterior, missing_posterior[:, :, None]], dim=2)
    entropy_terms = -categorical * torch.log(categorical.clamp(min=1e-12))
    winding_entropy = torch.where(
        model_valid_f.bool(), entropy_terms.sum(dim=2),
        torch.zeros_like(missing_posterior))
    n_valid = n_windings.to(dtype).clamp(min=1.0)
    ray_entropy = winding_entropy.sum(dim=1) / n_valid
    top2 = categorical.topk(min(2, categorical.shape[2]), dim=2).values
    if top2.shape[2] > 1:
        top2_margin = top2[:, :, 0] - top2[:, :, 1]
    else:
        top2_margin = top2[:, :, 0]
    top2_margin = torch.where(
        model_valid_f.bool(), top2_margin, torch.zeros_like(top2_margin))

    with torch.no_grad():
        values, pointers = _viterbi(
            lp_match.detach(), n_windings, costs, shape, dtype,
            keep_pointers=compute_map_path)
        v_m, v_x, v_y = values
        map_score = torch.stack([
            v_m[batch_idx, n_windings, m_bands],
            v_x[batch_idx, n_windings, m_bands],
            v_y[batch_idx, n_windings, m_bands]], dim=0).max(dim=0).values
        # Clearance of the MAP path against the sum of all other paths,
        # in log-probability units (large = decisive alignment).
        rest = log_partition.detach() + torch.log1p(
            -torch.exp((map_score - log_partition.detach()).clamp(max=-1e-7)))
        map_clearance = map_score - rest

    result = {
        'log_partition': log_partition,
        'match_posterior': match_posterior,
        'missing_posterior': missing_posterior,
        'extra_posterior': extra_posterior,
        'interior_extra_posterior': interior_extra_posterior,
        'winding_entropy': winding_entropy,
        'ray_entropy': ray_entropy,
        'top2_margin': top2_margin,
        'map_score': map_score,
        'map_clearance': map_clearance,
        'expected_matches': match_posterior.sum(dim=(1, 2)),
        'expected_missing': missing_posterior.sum(dim=1),
        'expected_extra': interior_extra_posterior.sum(dim=1),
        'num_windings': n_windings,
        'num_bands': m_bands,
    }
    if compute_map_path:
        with torch.no_grad():
            map_match, map_extra = _backtrack(
                values, pointers, n_windings, m_bands, shape)
        model_pad = torch.arange(map_match.shape[1], device=device)[None, :]
        result['map_match'] = torch.where(
            model_pad < n_windings[:, None], map_match,
            torch.full_like(map_match, -1))
        band_pad = torch.arange(map_extra.shape[1], device=device)[None, :]
        result['map_extra'] = map_extra & (band_pad < m_bands[:, None])
    return result
