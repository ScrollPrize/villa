"""Three-way merge for VC3D fiber annotation JSON files.

Pure functions over parsed JSON dicts — no I/O, no S3 knowledge. vc_sync
imports this to auto-resolve conflicts where both machines changed a fiber
file since their last common synced version (the "base").

Merge semantics per section:

- control_points: classic diff3. Each side's point list is aligned to the
  base by position (1e-6 tolerance, LCS). Base points matched in BOTH
  alignments are anchors; between consecutive anchors, a region changed on
  only one side takes that side's points, identical changes collapse, and
  a region changed differently on both sides is a conflict (this subsumes
  delete-vs-modify and both-extending-the-same-end).
- branches (cross-fiber links): set semantics keyed by
  (branch_file, local position, linked position). Additions from either
  side survive; a link present in base and untouched on one side but gone
  from the other was deleted; a link modified on one side and deleted on
  the other is kept (an approval must never be silently discarded);
  pending=False wins over pending=True for the same link. Local
  control_point_index is re-resolved against the merged points.
- line_points: derived data (the optimizer fits them to the volume inside
  VC3D, which this module cannot run). If only one side changed geometry,
  its line is taken wholesale; if both changed disjoint regions, the line
  is spliced piecewise at anchor control points; if splicing fails the
  local line is kept and a "needs_reoptimization" tag is added so VC3D can
  re-fit later.
- tags: base-aware set merge. Other scalars follow the side with the newer
  generation; generation becomes max(local, remote) + 1.

A merge either succeeds cleanly or reports conflicts; it never guesses on
overlapping edits. Callers keep pre-merge copies of every input.
"""

import copy
import math

POS_TOL = 1.0e-6

# Keys this module computes; everything else passes through from the
# newer-generation side.
_MANAGED_KEYS = ('control_points', 'line_points', 'branches', 'tags', 'generation')

REOPTIMIZE_TAG = 'needs_reoptimization'


def is_fiber_doc(doc):
    return (isinstance(doc, dict) and
            doc.get('type') == 'vc3d_fiber' and
            isinstance(doc.get('control_points'), list) and
            isinstance(doc.get('line_points'), list))


def pos_eq(a, b, tol=POS_TOL):
    try:
        return (len(a) == 3 and len(b) == 3 and
                all(math.isfinite(x) and math.isfinite(y) and abs(x - y) <= tol
                    for x, y in zip(a, b)))
    except TypeError:
        return False


def qpos(p):
    """Quantized position usable as a dict key (matches POS_TOL scale)."""
    return tuple(round(float(x), 6) for x in p)


def _seq_eq(a, b):
    return len(a) == len(b) and all(pos_eq(x, y) for x, y in zip(a, b))


def _dist2(a, b):
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _lcs_matches(base, side):
    """Longest common subsequence of positions; returns [(base_i, side_j)]."""
    n, m = len(base), len(side)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row = dp[i]
        below = dp[i + 1]
        for j in range(m - 1, -1, -1):
            if pos_eq(base[i], side[j]):
                row[j] = below[j + 1] + 1
            else:
                row[j] = below[j] if below[j] >= row[j + 1] else row[j + 1]
    matches = []
    i = j = 0
    while i < n and j < m:
        if pos_eq(base[i], side[j]):
            matches.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return matches


def merge_control_points(base, local, remote):
    """Diff3 over point sequences.

    Returns (merged, regions, conflicts). regions is one entry per
    inter-anchor gap: {'owner': none|local|remote|both|conflict,
    'local': (lo, hi), 'remote': (lo, hi)} with half-open side index
    ranges (exclusive of the surrounding anchors).
    """
    match_local = _lcs_matches(base, local)
    match_remote = _lcs_matches(base, remote)
    map_local = dict(match_local)
    map_remote = dict(match_remote)
    anchors = [bi for bi, _ in match_local if bi in map_remote]

    bounds = ([(-1, -1, -1)] +
              [(a, map_local[a], map_remote[a]) for a in anchors] +
              [(len(base), len(local), len(remote))])

    merged = []
    regions = []
    conflicts = []
    for k in range(len(bounds) - 1):
        b0, l0, r0 = bounds[k]
        b1, l1, r1 = bounds[k + 1]
        base_seg = base[b0 + 1:b1]
        local_seg = local[l0 + 1:l1]
        remote_seg = remote[r0 + 1:r1]
        local_changed = not _seq_eq(local_seg, base_seg)
        remote_changed = not _seq_eq(remote_seg, base_seg)

        if local_changed and remote_changed:
            if _seq_eq(local_seg, remote_seg):
                owner, seg = 'both', local_seg
            else:
                owner, seg = 'conflict', base_seg
                conflicts.append(
                    "control_points: both sides changed base points "
                    f"{b0 + 1}..{max(b0 + 1, b1 - 1)} differently "
                    f"(local {len(local_seg)} pts, remote {len(remote_seg)} pts)")
        elif local_changed:
            owner, seg = 'local', local_seg
        elif remote_changed:
            owner, seg = 'remote', remote_seg
        else:
            owner, seg = 'none', base_seg

        merged.extend(seg)
        regions.append({'owner': owner,
                        'local': (l0 + 1, l1),
                        'remote': (r0 + 1, r1)})
        if b1 < len(base):
            # Anchor point itself: all three agree within tolerance.
            merged.append(local[l1])

    return merged, regions, conflicts


def _branch_key(branch):
    return (branch.get('branch_file', ''),
            qpos(branch.get('control_point_position', (0, 0, 0))),
            qpos(branch.get('branch_control_point_position', (0, 0, 0))))


def _branches_of(doc):
    branches = doc.get('branches', [])
    return branches if isinstance(branches, list) else []


def _branch_signature(branch):
    """What counts as a meaningful edit to a link (indices and directions
    are derived/re-resolved elsewhere; pending is the review state)."""
    return (bool(branch.get('pending', False)),
            _branch_key(branch))


def merge_branches(base_doc, local_doc, remote_doc, merged_cps, prefer_local):
    base_by_key = {}
    for branch in _branches_of(base_doc):
        if isinstance(branch, dict):
            base_by_key[_branch_key(branch)] = branch
    local_entries = [b for b in _branches_of(local_doc) if isinstance(b, dict)]
    remote_entries = [b for b in _branches_of(remote_doc) if isinstance(b, dict)]
    local_by_key = {_branch_key(b): b for b in local_entries}
    remote_by_key = {_branch_key(b): b for b in remote_entries}

    ordered_keys = [_branch_key(b) for b in local_entries]
    ordered_keys += [_branch_key(b) for b in remote_entries
                     if _branch_key(b) not in local_by_key]

    merged = []
    notes = []
    stats = {'links_kept': 0, 'links_added_local': 0, 'links_added_remote': 0,
             'links_deleted': 0, 'links_approved': 0}
    seen = set()
    for key in ordered_keys:
        if key in seen:
            continue
        seen.add(key)
        in_local = key in local_by_key
        in_remote = key in remote_by_key
        in_base = key in base_by_key

        if in_local and in_remote:
            local_entry = local_by_key[key]
            remote_entry = remote_by_key[key]
            local_pending = bool(local_entry.get('pending', False))
            remote_pending = bool(remote_entry.get('pending', False))
            if local_pending != remote_pending:
                # Approval (pending -> absent/False) always wins.
                chosen = remote_entry if local_pending else local_entry
                stats['links_approved'] += 1
            else:
                chosen = local_entry if prefer_local else remote_entry
            merged.append(copy.deepcopy(chosen))
            stats['links_kept'] += 1
            continue

        side_entry = local_by_key.get(key) or remote_by_key.get(key)
        side_name = 'local' if in_local else 'remote'
        if not in_base:
            merged.append(copy.deepcopy(side_entry))
            stats['links_added_%s' % side_name] += 1
            continue
        # Present in base, gone from the other side: deletion unless this
        # side meaningfully modified it (approving a link beats deleting it).
        if _branch_signature(side_entry) == _branch_signature(base_by_key[key]):
            stats['links_deleted'] += 1
            notes.append(f"link to {key[0]} deleted on "
                         f"{'remote' if in_local else 'local'}")
            continue
        merged.append(copy.deepcopy(side_entry))
        notes.append(f"kept link to {key[0]} modified on {side_name} "
                     "but deleted on the other side")
        stats['links_kept'] += 1

    # Local anchor indices may have shifted; re-resolve them against the
    # merged control points. (VC3D's loader also resolves by position, so
    # this is belt-and-braces plus a guard for anchors deleted by a
    # winning geometry hunk.)
    surviving = []
    for branch in merged:
        position = branch.get('control_point_position')
        index = None
        best = POS_TOL * POS_TOL
        for i, cp in enumerate(merged_cps):
            d2 = _dist2(cp, position)
            if d2 <= best:
                best = d2
                index = i
        if index is None:
            notes.append(f"dropped link to {branch.get('branch_file', '?')}: "
                         "its control point was removed by the merged geometry")
            stats['links_deleted'] += 1
            continue
        branch['control_point_index'] = index
        surviving.append(branch)
    return surviving, notes, stats


def merge_tags(base_doc, local_doc, remote_doc):
    base = list(base_doc.get('tags', []) or [])
    local = list(local_doc.get('tags', []) or [])
    remote = list(remote_doc.get('tags', []) or [])
    base_set, local_set, remote_set = set(base), set(local), set(remote)

    def keep(tag):
        return ((tag in local_set and tag in remote_set) or
                (tag in local_set and tag not in base_set) or
                (tag in remote_set and tag not in base_set))

    merged = [t for t in base if keep(t)]
    merged += [t for t in local if keep(t) and t not in merged]
    merged += [t for t in remote if keep(t) and t not in merged]
    return merged


def merge_line_points(local_doc, remote_doc, regions, anchor_positions):
    """Piecewise line splice. Returns (line_points, note) — note is None on
    the clean paths and a human-readable reason when falling back to the
    local line (caller adds the reoptimize tag)."""
    local_line = local_doc.get('line_points', [])
    remote_line = remote_doc.get('line_points', [])
    owners = [region['owner'] for region in regions]
    remote_owns = 'remote' in owners
    local_owns = 'local' in owners or 'both' in owners

    if not remote_owns:
        return list(local_line), None
    if not local_owns:
        return list(remote_line), None

    def anchor_line_indices(line):
        indices = []
        start = 0
        for position in anchor_positions:
            best_index = None
            best_d2 = math.inf
            for i in range(start, len(line)):
                d2 = _dist2(line[i], position)
                if d2 < best_d2:
                    best_d2 = d2
                    best_index = i
            if best_index is None:
                return None
            indices.append(best_index)
            start = best_index + 1
        return indices

    local_idx = anchor_line_indices(local_line)
    remote_idx = anchor_line_indices(remote_line)
    if local_idx is None or remote_idx is None:
        return list(local_line), "line splice failed (non-monotone anchors)"

    merged = []
    for k, region in enumerate(regions):
        owner = region['owner']
        line = remote_line if owner == 'remote' else local_line
        idx = remote_idx if owner == 'remote' else local_idx
        lo = 0 if k == 0 else idx[k - 1]
        hi = len(line) if k == len(regions) - 1 else idx[k]
        if hi < lo:
            return list(local_line), "line splice failed (crossed segments)"
        merged.extend(line[lo:hi])
    return merged, None


def merge_fibers(base, local, remote):
    """Three-way merge. Returns
    {'ok': bool, 'merged': dict|None, 'conflicts': [str], 'notes': [str],
     'stats': {...}}."""
    result = {'ok': False, 'merged': None, 'conflicts': [], 'notes': [],
              'stats': {}}

    for name, doc in (('base', base), ('local', local), ('remote', remote)):
        if not is_fiber_doc(doc):
            result['conflicts'].append(f"{name} version is not a vc3d_fiber document")
            return result
    for field in ('type', 'version', 'filename'):
        values = {str(doc.get(field)) for doc in (base, local, remote)}
        if len(values) > 1:
            result['conflicts'].append(
                f"'{field}' differs between versions: {sorted(values)}")
            return result

    # Short circuits: only one side truly changed, or both converged.
    if local == remote or remote == base:
        result.update(ok=True, merged=copy.deepcopy(local),
                      notes=(["remote side unchanged; kept local"]
                             if remote == base and local != remote else
                             ["both sides identical"]))
        return result
    if local == base:
        result.update(ok=True, merged=copy.deepcopy(remote),
                      notes=["local side unchanged; took remote"])
        return result

    merged_cps, regions, cp_conflicts = merge_control_points(
        base['control_points'], local['control_points'],
        remote['control_points'])
    if cp_conflicts:
        result['conflicts'] = cp_conflicts
        return result

    generation_local = int(local.get('generation', 1) or 1)
    generation_remote = int(remote.get('generation', 1) or 1)
    prefer_local = generation_local >= generation_remote
    newer = local if prefer_local else remote

    merged_branches, branch_notes, branch_stats = merge_branches(
        base, local, remote, merged_cps, prefer_local)

    anchor_positions = []
    for k, region in enumerate(regions[:-1]):
        # The anchor after region k is the point at local index region[k]['local'][1]
        anchor_positions.append(local['control_points'][region['local'][1]])

    geometry_merged = any(region['owner'] in ('local', 'remote', 'both')
                          for region in regions)
    line_points, line_note = merge_line_points(local, remote, regions,
                                               anchor_positions)

    merged = copy.deepcopy(newer)
    merged['control_points'] = [list(p) for p in merged_cps]
    merged['line_points'] = [list(p) for p in line_points]
    merged['branches'] = merged_branches
    merged['tags'] = merge_tags(base, local, remote)
    merged['generation'] = max(generation_local, generation_remote) + 1
    if geometry_merged or line_note:
        if REOPTIMIZE_TAG not in merged['tags']:
            merged['tags'].append(REOPTIMIZE_TAG)

    notes = list(branch_notes)
    if line_note:
        notes.append(line_note + "; kept local line, tagged for reoptimization")

    stats = dict(branch_stats)
    stats['cp_regions_local'] = sum(1 for r in regions if r['owner'] == 'local')
    stats['cp_regions_remote'] = sum(1 for r in regions if r['owner'] == 'remote')
    stats['geometry_merged'] = geometry_merged

    result.update(ok=True, merged=merged, notes=notes, stats=stats)
    return result


def summarize(result):
    if not result['ok']:
        return 'conflict: ' + '; '.join(result['conflicts'])
    stats = result.get('stats', {})
    if not stats:
        return '; '.join(result.get('notes', [])) or 'merged'
    parts = []
    if stats.get('cp_regions_local') or stats.get('cp_regions_remote'):
        parts.append("control points: %d local + %d remote region(s)" %
                     (stats.get('cp_regions_local', 0),
                      stats.get('cp_regions_remote', 0)))
    added = stats.get('links_added_local', 0) + stats.get('links_added_remote', 0)
    if added:
        parts.append(f"links +{added}")
    if stats.get('links_deleted'):
        parts.append(f"links -{stats['links_deleted']}")
    if stats.get('links_approved'):
        parts.append(f"{stats['links_approved']} link approval(s) applied")
    if stats.get('geometry_merged'):
        parts.append("geometry merged (tagged for reoptimization)")
    return ', '.join(parts) if parts else 'merged (metadata only)'
