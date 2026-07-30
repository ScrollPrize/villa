"""Three-way merge for VC3D fiber annotation JSON files.

Pure functions over parsed JSON dicts — no I/O, no S3 knowledge. vc_sync
imports this to auto-resolve conflicts where both machines changed a fiber
file since their last common synced version (the "base").

Design contract: merge output must load cleanly in an UNMODIFIED VC3D,
whose loader is strict and destructive (offering to delete inconsistent
branch links from disk, or dropping whole fibers from the session). The
invariants, from LineAnnotationController.cpp / Atlas.cpp:

- C1: every control point must appear in line_points, in order, within
  1e-8 (validateFiberInputControlPoints); violating this makes the whole
  file unloadable. Consequence: control_points and line_points must always
  be written as a consistent pair from the same source.
- C2: cross-fiber link reciprocity is index-exact across files, plus
  positions (1e-6) and directions.
- C3: stored link directions must match tangents recomputed from
  line_points within ~0.26 deg (branchDirectionsCompatible, 1e-5 on
  |cos theta|).

Merge semantics:

- Same geometry on both sides (the incident case — divergent link edits):
  geometry and derived fields come wholesale from the newer-generation
  side; branches merge with set semantics (additions from both sides
  survive, base-aware deletions, approvals beat both pending and
  deletion); tags merge base-aware. Loader-safe because every branch
  entry was written by VC3D against this exact geometry.
- Geometry changed on ONE side (including line-only re-optimizations):
  that side's control_points/line_points/hv are taken wholesale (C1 by
  construction); links from the other side are re-anchored by position.
- Geometry changed on BOTH sides in disjoint regions: control_points
  merge via diff3 (position-aligned against the base; overlapping edits
  conflict). The optimizer cannot run here, so line_points is set to the
  merged control points verbatim (C1 holds exactly) and the merged fiber
  is tagged `needs_reoptimization`; VC3D offers to re-fit the line on
  load.
- A link whose anchor control point was deleted by the merged geometry
  makes the whole merge a conflict: links are never silently dropped.
- After any merge that involves links, the caller must run
  refresh_pair_links() against each peer file (C2/C3): vc_sync mirrors
  the merged fiber's link decisions into the peers and uploads them.

A merge either succeeds cleanly or reports conflicts; it never guesses on
overlapping edits. Callers keep pre-merge copies of every input.
"""

import copy
import json
import math

POS_TOL = 1.0e-6

# Matches finiteDirection's epsilon guard: a vector this short has no
# usable direction.
_DIR_EPS = 1.0e-12

REOPTIMIZE_TAG = 'needs_reoptimization'
# ^ consumed by VC3D's load-time re-optimization prompt; keep the literal
#   in sync with kNeedsReoptimizationTag in LineAnnotationController.cpp.


def _cp_position(value):
    return value.get('position') if isinstance(value, dict) else value


def _finite_point(p):
    """Strictly a 3-vector of finite numbers — no bools, no numeric
    strings. Anything looser either crashes the arithmetic below or writes
    values the loader throws on."""
    try:
        return (len(p) == 3 and
                all(isinstance(x, (int, float)) and not isinstance(x, bool) and
                    math.isfinite(x)
                    for x in p))
    except TypeError:
        return False


_SEGMENT_KEYS_V1 = {
    'optimizer', 'metadata_version', 'tracer_version', 'normal_manifest',
    'fiber_manifest', 'trace_to_base_scale',
    'max_endpoint_error_base_voxels', 'config',
}
_SEGMENT_KEYS_V2 = {
    'optimizer', 'metadata_version', 'tracer_version', 'outcome',
    'normal_manifest', 'fiber_manifest', 'trace_to_base_scale',
    'meeting_error_base_voxels', 'meeting_error_ratio', 'meeting_source',
    'failure_code', 'failure_detail', 'config',
}
_CONFIG_KEYS_COMMON = {
    'step_voxels', 'cone_angle_degrees', 'cone_angle_step_degrees',
    'cone_grid_size', 'beam_width', 'beam_prune_distance_voxels',
    'beam_lookahead_steps', 'smoothness_weight',
    'smoothness_normal_weight', 'smoothness_tangent_weight',
    'smoothness_free_angle_degrees', 'cumulative_smoothness_steps',
    'cumulative_smoothness_tangent_weight', 'initial_free_angle_degrees',
    'max_step_factor',
    'endpoint_accept_threshold_base_voxels',
}
_CONFIG_KEYS_V1 = _CONFIG_KEYS_COMMON | {'fusion_gap_factor'}
_CONFIG_KEYS_V2 = _CONFIG_KEYS_COMMON | {'meeting_accept_max_error_ratio'}


def _valid_segment(segment):
    if not isinstance(segment, dict):
        return False
    version = (segment.get('metadata_version'), segment.get('tracer_version'))
    if version == (1, 1):
        segment_keys = _SEGMENT_KEYS_V1
        config_keys = _CONFIG_KEYS_V1
    elif version == (2, 2):
        segment_keys = _SEGMENT_KEYS_V2
        config_keys = _CONFIG_KEYS_V2
    else:
        return False
    if (set(segment) != segment_keys or
            segment.get('optimizer') != 'native_fiber_trace3d' or
            not isinstance(segment.get('normal_manifest'), str) or
            not segment['normal_manifest'] or
            not isinstance(segment.get('fiber_manifest'), str) or
            not segment['fiber_manifest']):
        return False
    config = segment.get('config')
    numeric = [segment.get('trace_to_base_scale')]
    if not isinstance(config, dict) or set(config) != config_keys:
        return False
    numeric.extend(config.values())
    if not all(isinstance(value, (int, float)) and
               not isinstance(value, bool) and math.isfinite(value)
               for value in numeric):
        return False
    if version == (1, 1):
        error = segment.get('max_endpoint_error_base_voxels')
        return (isinstance(error, (int, float)) and not isinstance(error, bool)
                and math.isfinite(error) and error >= 0)
    outcome = segment.get('outcome')
    error = segment.get('meeting_error_base_voxels')
    ratio = segment.get('meeting_error_ratio')
    strings = [segment.get('meeting_source'), segment.get('failure_code'),
               segment.get('failure_detail')]
    if (outcome not in {'accepted_native', 'lasagna_fallback'} or
            not all(isinstance(value, str) for value in strings) or
            (error is None) != (ratio is None)):
        return False
    if error is not None and not (
            isinstance(error, (int, float)) and not isinstance(error, bool) and
            math.isfinite(error) and error >= 0 and
            isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and
            math.isfinite(ratio) and 0 <= ratio <= 1):
        return False
    if not 0 <= config['meeting_accept_max_error_ratio'] <= 1:
        return False
    if outcome == 'accepted_native':
        return (error is not None and bool(segment['meeting_source']) and
                not segment['failure_code'] and not segment['failure_detail'])
    return bool(segment['failure_code'])


def is_fiber_doc(doc):
    """Structural AND content validity for every field this module
    computes over. Malformed (possibly remote-controlled) input must be
    rejected here so callers degrade to manual resolution instead of
    crashing mid-merge."""
    if not (isinstance(doc, dict) and doc.get('type') == 'vc3d_fiber'):
        return False
    version = doc.get('version', 1)
    if version not in (1, 2):
        return False
    line_points = doc.get('line_points')
    if not isinstance(line_points, list) or not all(_finite_point(p)
                                                    for p in line_points):
        return False
    control_points = doc.get('control_points')
    if not isinstance(control_points, list):
        return False
    if version == 1:
        if not all(_finite_point(p) for p in control_points):
            return False
    else:
        for index, cp in enumerate(control_points):
            if (not isinstance(cp, dict) or
                    not set(cp) <= {'position', 'segment_to_next'} or
                    not _finite_point(cp.get('position'))):
                return False
            segment = cp.get('segment_to_next')
            if segment is not None:
                if index + 1 == len(control_points) or not _valid_segment(segment):
                    return False
    tags = doc.get('tags', [])
    if not (isinstance(tags, list) and
            all(isinstance(tag, str) for tag in tags)):
        return False  # tags: null is unloadable ("tags must be an array")
    generation = doc.get('generation', 1)
    if generation is not None and (isinstance(generation, bool) or
                                   not isinstance(generation, (int, float)) or
                                   not math.isfinite(generation)):
        return False
    return True


def pos_eq(a, b, tol=POS_TOL):
    """Port of the loader's pointsApproximatelyEqual: a squared-EUCLIDEAN
    ball of radius `tol`. A per-axis box would accept positions up to
    sqrt(3)*tol apart that the loader rejects — the exact class of
    inconsistency the destructive repair prompt punishes."""
    try:
        a = _cp_position(a)
        b = _cp_position(b)
        if len(a) != 3 or len(b) != 3:
            return False
        d2 = 0.0
        for x, y in zip(a, b):
            fx = float(x)
            fy = float(y)
            if not (math.isfinite(fx) and math.isfinite(fy)):
                return False
            d2 += (fx - fy) ** 2
        return d2 <= tol * tol
    except (TypeError, ValueError):
        return False


def _seq_eq(a, b):
    return len(a) == len(b) and all(pos_eq(x, y) for x, y in zip(a, b))


def _dist2(a, b):
    return sum((float(x) - float(y)) ** 2
               for x, y in zip(_cp_position(a), _cp_position(b)))


def _normalized(v):
    """Unit vector, or None when the input is non-finite or ~zero
    (mirrors normalizedOrZero + finiteDirection)."""
    if not _finite_point(v):
        return None
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= _DIR_EPS:
        return None
    return [float(x) * (1.0 / n) for x in v]


def directions_compatible(a, b, tol=1.0e-5):
    """Port of branchDirectionsCompatible: sign-agnostic, 1e-5 on
    |cos theta| (~0.26 deg)."""
    na = _normalized(a)
    nb = _normalized(b)
    if na is None or nb is None:
        return False
    dot = sum(x * y for x, y in zip(na, nb))
    return abs(abs(dot) - 1.0) <= tol


# --- exact ports of VC3D's endpoint tangent computation -------------------
# VC3D validates stored link directions against tangents it recomputes from
# line_points; producing directions with the same algorithm on the same
# data guarantees agreement well inside the 0.26 deg tolerance.

def nearest_line_point_index(line, point):
    """Port of fiber_slice::nearestLinePointIndex: first-wins argmin of
    squared distance, skipping non-finite points."""
    best_index = 0
    best_d2 = math.inf
    for i, p in enumerate(line):
        if not _finite_point(p):
            continue
        d2 = _dist2(p, point)
        if d2 < best_d2:
            best_d2 = d2
            best_index = i
    return best_index


def line_tangent_at(line, index):
    """Port of tangentAtLinePosition for an integer position: segment
    tangent at `index`, unit length, with VC3D's {1,0,0} degenerate
    fallback."""
    n = len(line)
    lower = max(0, min(int(index), n - 1))
    upper = min(lower + 1, n - 1)
    if lower == upper and lower > 0:
        lower -= 1
    tangent = _normalized([float(line[upper][k]) - float(line[lower][k])
                           for k in range(3)])
    return tangent if tangent is not None else [1.0, 0.0, 0.0]


def endpoint_tangent(line, point):
    """Port of endpointTangentFromLinePoints: tangent of the line segment
    nearest to `point`, or None when the line has fewer than 2 points."""
    if len(line) < 2 or not _finite_point(point):
        return None
    return line_tangent_at(line, nearest_line_point_index(line, point))


def _nearly(a, b, tol=1.0e-12):
    try:
        return (len(a) == 3 and len(b) == 3 and
                all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b)))
    except (TypeError, ValueError):
        return False


def _snapped_direction(stored, tangent):
    """The value a stored direction field must hold: (+-)tangent.

    The loader's checks are sign-agnostic, but the stored sign encodes the
    link's sense in VC3D, so it is preserved. Snapping to +-tangent
    (rather than leave-if-within-tolerance) is what makes reciprocal PAIRS
    pass the loader's stored-vs-stored comparison: two fields each within
    tolerance of the true tangent can still be 2x tolerance apart from
    each other. The accept window is ~1e-12 per component rather than
    byte-equality so a VC3D build whose float contraction differs from
    CPython by an ulp doesn't cause every refresh to rewrite (and
    re-upload) every direction; pairwise that window is still 7 orders of
    magnitude inside the loader's 1e-5 tolerance."""
    negated = [-x for x in tangent]
    if _nearly(stored, tangent) or _nearly(stored, negated):
        return stored
    normalized = _normalized(stored)
    if (normalized is not None and
            sum(x * y for x, y in zip(normalized, tangent)) < 0.0):
        return negated
    return tangent


def resolve_cp_index(control_points, position):
    """Index of the control point pos_eq-equal to `position` (nearest when
    several qualify), or None. pos_eq is the loader's own Euclidean
    predicate, so anything resolvable here is acceptable there."""
    best_index = None
    best_d2 = math.inf
    for i, cp in enumerate(control_points):
        if pos_eq(cp, position):
            d2 = _dist2(cp, position)
            if d2 < best_d2:
                best_d2 = d2
                best_index = i
    return best_index


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


def _branches_of(doc):
    branches = doc.get('branches', [])
    return branches if isinstance(branches, list) else []


def _branch_target(entry):
    """Link target normalized the way the loader normalizes it: the bare
    filename (fs::path(...).filename()). Comparing raw strings would treat
    a legacy 'fibers/f.json' entry as a different peer than 'f.json' and
    skip its consistency fixes."""
    name = entry.get('branch_file')
    if not isinstance(name, str):
        return None
    basename = name.rsplit('/', 1)[-1]
    return basename or None


def _structured_branch(branch):
    """A branch entry this module understands well enough to merge.

    Entries carrying the obsolete `link_direction` key are deliberately
    opaque: the loader strips them at parse time, so treating one as
    decided truth would cement a reciprocal the loader then tears down."""
    return (isinstance(branch, dict) and
            'link_direction' not in branch and
            _branch_target(branch) is not None and
            _finite_point(branch.get('control_point_position')) and
            _finite_point(branch.get('branch_control_point_position')))


def split_branches(doc):
    """(structured, opaque) partition of a document's branch entries.

    Opaque entries (non-objects, missing or malformed fields) are data
    this module cannot interpret; they are preserved as-is and merged
    only by whole-value comparison — never silently dropped."""
    structured = []
    opaque = []
    for branch in _branches_of(doc):
        (structured if _structured_branch(branch) else opaque).append(branch)
    return structured, opaque


def links_to(doc, peer_name):
    """Structured branch entries of `doc` that point at `peer_name`
    (compared as basenames, like the loader)."""
    return [entry for entry in _branches_of(doc)
            if _structured_branch(entry) and _branch_target(entry) == peer_name]


def links_to_any(doc):
    """All structured branch entries of a document."""
    return [entry for entry in _branches_of(doc) if _structured_branch(entry)]


def _canon_opaque(entries):
    return sorted(json.dumps(entry, sort_keys=True) for entry in entries)


def merge_opaque_branches(base_opaque, local_opaque, remote_opaque):
    """Whole-value base-aware merge of uninterpretable branch entries.
    Returns (merged_entries, conflict_message_or_None).

    Note the guarantee is only "the MERGE never drops them": an unmodified
    VC3D still strips structurally unparseable entries (and rewrites the
    file) when it next loads it — exactly as it would have without any
    merge."""
    canon_base = _canon_opaque(base_opaque)
    canon_local = _canon_opaque(local_opaque)
    canon_remote = _canon_opaque(remote_opaque)
    if canon_local == canon_remote:
        return copy.deepcopy(local_opaque), None
    if canon_local == canon_base:
        return copy.deepcopy(remote_opaque), None
    if canon_remote == canon_base:
        return copy.deepcopy(local_opaque), None
    return [], ("branches: structurally unparseable entries differ between "
                "local and remote; refusing to merge them")


def _same_link(a, b):
    """Tolerant link identity: same target file (as basenames) and both
    endpoint positions within POS_TOL — the same geometric predicate used
    for control-point alignment."""
    return (_branch_target(a) == _branch_target(b) and
            pos_eq(a.get('control_point_position'),
                   b.get('control_point_position')) and
            pos_eq(a.get('branch_control_point_position'),
                   b.get('branch_control_point_position')))


def _find_link(entries, branch, used):
    for i, entry in enumerate(entries):
        if i not in used and _same_link(entry, branch):
            return i
    return None


def _link_modified(entry, base_entry):
    """The review state is the meaningful mutable field on a link; indices
    and directions are derived and re-resolved elsewhere."""
    return (bool(entry.get('pending', False)) !=
            bool(base_entry.get('pending', False)))


def merge_branches(base_doc, local_doc, remote_doc, prefer_local):
    """Base-aware set merge of structured link entries. Additions from
    either side survive; untouched-here-but-gone-there means deletion;
    approving a link beats deleting it; pending=False wins over
    pending=True for the same link. Local anchor indices are NOT touched
    here — the caller re-anchors entries against the merged geometry."""
    base_entries, _ = split_branches(base_doc)
    local_entries, _ = split_branches(local_doc)
    remote_entries, _ = split_branches(remote_doc)

    merged = []
    notes = []
    stats = {'links_kept': 0, 'links_added_local': 0, 'links_added_remote': 0,
             'links_deleted': 0, 'links_approved': 0}
    used_remote = set()
    used_base = set()

    def base_match(branch):
        index = _find_link(base_entries, branch, used_base)
        if index is None:
            return None
        used_base.add(index)
        return base_entries[index]

    def merge_one_sided(entry, side_name):
        base_entry = base_match(entry)
        if base_entry is None:
            # Note: a link whose anchor was MOVED on this side (beyond
            # tolerance) also lands here — its identity no longer matches
            # the base, so it reads as an addition and survives even if the
            # other side deleted the original. Kept deliberately: a moved
            # link is indistinguishable from a re-created one.
            merged.append(copy.deepcopy(entry))
            stats['links_added_%s' % side_name] += 1
            return
        # Present in base, gone from the other side: deletion unless this
        # side meaningfully modified it (approving a link beats deleting it).
        if not _link_modified(entry, base_entry):
            stats['links_deleted'] += 1
            notes.append(f"link to {entry.get('branch_file', '?')} deleted on "
                         f"{'remote' if side_name == 'local' else 'local'}")
            return
        merged.append(copy.deepcopy(entry))
        notes.append(f"kept link to {entry.get('branch_file', '?')} modified "
                     f"on {side_name} but deleted on the other side")
        stats['links_kept'] += 1

    for local_entry in local_entries:
        remote_index = _find_link(remote_entries, local_entry, used_remote)
        if remote_index is None:
            merge_one_sided(local_entry, 'local')
            continue
        used_remote.add(remote_index)
        base_entry = base_match(local_entry)
        remote_entry = remote_entries[remote_index]
        local_pending = bool(local_entry.get('pending', False))
        remote_pending = bool(remote_entry.get('pending', False))
        if local_pending != remote_pending:
            if base_entry is not None:
                # The side that changed the review state relative to the
                # base wins: approving beats an untouched pending copy, and
                # deliberately re-flagging beats an untouched approval.
                base_pending = bool(base_entry.get('pending', False))
                chosen = (local_entry if local_pending != base_pending
                          else remote_entry)
            else:
                # No base entry to arbitrate: prefer the approved state.
                chosen = remote_entry if local_pending else local_entry
            if bool(chosen.get('pending', False)):
                notes.append(f"link to {_branch_target(chosen) or '?'} "
                             "re-flagged as pending")
            else:
                stats['links_approved'] += 1
        else:
            chosen = local_entry if prefer_local else remote_entry
        merged.append(copy.deepcopy(chosen))
        stats['links_kept'] += 1

    for i, remote_entry in enumerate(remote_entries):
        if i not in used_remote:
            merge_one_sided(remote_entry, 'remote')

    return merged, notes, stats


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


def _rebind_local_anchors(entries, control_points, line_points):
    """Re-anchor link entries against merged geometry: index by position,
    position snapped to the exact control point, local direction from the
    line. Returns (entries, conflict_message_or_None). Far-side fields are
    left alone — refresh_pair_links() fixes them against the peer file."""
    if entries and len(line_points) < 2:
        return entries, ("links present but the merged line has fewer than "
                         "2 points; cannot derive endpoint directions")
    for entry in entries:
        index = resolve_cp_index(control_points, entry['control_point_position'])
        if index is None:
            # Never silently drop a link: geometry that removed a link's
            # anchor makes the whole merge a manual conflict.
            return entries, (f"link to {entry.get('branch_file', '?')} anchors "
                             "at a control point absent from the merged "
                             "geometry")
        entry['control_point_index'] = index
        entry['control_point_position'] = [float(x)
                                           for x in _cp_position(control_points[index])]
        tangent = endpoint_tangent(line_points, entry['control_point_position'])
        entry['control_point_direction'] = _snapped_direction(
            entry.get('control_point_direction'), tangent)
    return entries, None


def _manual_hv_tag(doc):
    hv = doc.get('hv_classification')
    return hv.get('manual_tag') if isinstance(hv, dict) else None


def _merge_manual_hv_tag(base, local, remote, merged, notes):
    """Three-way merge of hv_classification.manual_tag — the one field
    inside hv_classification that is a USER decision, not derived (the
    loader round-trips it independently of the recomputed scores). The
    rest of hv_classification stays with the carrier. Returns a conflict
    message or None."""
    base_tag = _manual_hv_tag(base)
    local_tag = _manual_hv_tag(local)
    remote_tag = _manual_hv_tag(remote)
    if local_tag == remote_tag:
        chosen = local_tag
    elif remote_tag == base_tag:
        chosen = local_tag
    elif local_tag == base_tag:
        chosen = remote_tag
    else:
        return (f"hv manual_tag changed differently on both sides "
                f"({local_tag!r} vs {remote_tag!r})")
    if chosen == _manual_hv_tag(merged):
        return None
    hv = merged.get('hv_classification')
    if not isinstance(hv, dict):
        hv = {}
        merged['hv_classification'] = hv
    if chosen is None:
        hv.pop('manual_tag', None)
    else:
        hv['manual_tag'] = chosen
    notes.append(f"kept manual hv tag {chosen!r}")
    return None


def merge_fibers(base, local, remote):
    """Three-way merge. Returns
    {'ok': bool, 'merged': dict|None, 'conflicts': [str], 'notes': [str],
     'stats': {...}, 'peer_files': [str]}.

    A non-empty peer_files means the caller MUST run refresh_pair_links()
    for each listed peer and persist both sides together, or the merged
    file violates VC3D's cross-file reciprocity checks."""
    result = {'ok': False, 'merged': None, 'conflicts': [], 'notes': [],
              'stats': {}, 'peer_files': []}

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

    # Short circuits: only one side truly changed, or both converged. The
    # winning content is a file VC3D itself wrote — normally consistent
    # with its peers, since VC3D writes both sides of a link in lockstep.
    # peer_files is still reported so the caller's consistency pass can
    # verify (and heal) that invariant; for genuinely consistent pairs the
    # pass is a no-op with zero rewrites.
    def short_circuit_peers(doc):
        return sorted({_branch_target(entry) for entry in links_to_any(doc)} |
                      {_branch_target(entry) for entry in links_to_any(base)})

    if local == remote or remote == base:
        result.update(ok=True, merged=copy.deepcopy(local),
                      peer_files=short_circuit_peers(local),
                      notes=(["remote side unchanged; kept local"]
                             if remote == base and local != remote else
                             ["both sides identical"]))
        return result
    if local == base:
        result.update(ok=True, merged=copy.deepcopy(remote),
                      peer_files=short_circuit_peers(remote),
                      notes=["local side unchanged; took remote"])
        return result

    # Branch entries this module cannot interpret are preserved by
    # whole-value comparison; a clean merge must never discard input.
    _, base_opaque = split_branches(base)
    _, local_opaque = split_branches(local)
    _, remote_opaque = split_branches(remote)
    opaque_branches, opaque_conflict = merge_opaque_branches(
        base_opaque, local_opaque, remote_opaque)
    if opaque_conflict:
        result['conflicts'] = [opaque_conflict]
        return result

    generation_local = int(local.get('generation', 1) or 1)
    generation_remote = int(remote.get('generation', 1) or 1)
    prefer_local = generation_local >= generation_remote
    newer = local if prefer_local else remote

    merged_branches, branch_notes, branch_stats = merge_branches(
        base, local, remote, prefer_local)

    geometry_same = (_seq_eq(local['control_points'], remote['control_points'])
                     and _seq_eq(local['line_points'], remote['line_points']))
    notes = list(branch_notes)
    stats = dict(branch_stats)
    reoptimize = False

    if geometry_same:
        # The incident case: divergent link/metadata edits over identical
        # geometry. Every entry was VC3D-written against this geometry, so
        # no re-anchoring is needed.
        carrier = newer
        stats['geometry_merged'] = False
    else:
        # In the one-sided branch below only regions/conflicts are used —
        # the composite merged_cps mixes floats from all three docs, and a
        # loader-safe file must take control_points AND line_points from
        # ONE written pair.
        merged_cps, regions, cp_conflicts = merge_control_points(
            base['control_points'], local['control_points'],
            remote['control_points'])
        if cp_conflicts:
            result['conflicts'] = cp_conflicts
            return result

        owners = {region['owner'] for region in regions} - {'none'}
        base_line = base['line_points']
        local_line_changed = not _seq_eq(local['line_points'], base_line)
        remote_line_changed = not _seq_eq(remote['line_points'], base_line)

        if owners <= {'local', 'both'} or owners <= {'remote', 'both'}:
            # Geometry effectively changed on one side (or identically on
            # both): take that side's control_points AND line_points as the
            # written pair (C1 by construction). With no owned regions at
            # all, geometry differs only in line_points — pick the side
            # whose line actually moved.
            if owners & {'local', 'remote'}:
                side = 'local' if 'local' in owners else 'remote'
            elif local_line_changed != remote_line_changed:
                side = 'local' if local_line_changed else 'remote'
            else:
                side = 'local' if prefer_local else 'remote'
            carrier = local if side == 'local' else remote
            # The discarded side may have re-optimized its line without
            # moving control points; that refit cannot be carried, so ask
            # VC3D to redo it. Never dropped silently.
            other = remote if side == 'local' else local
            other_line_changed = (remote_line_changed if side == 'local'
                                  else local_line_changed)
            if (other_line_changed and
                    not _seq_eq(other['line_points'], carrier['line_points'])):
                reoptimize = True
                notes.append(f"{'remote' if side == 'local' else 'local'} "
                             f"re-optimized the line; kept the {side} line "
                             "and tagged for reoptimization")
        else:
            # Disjoint edits on both sides. The real line can only be
            # produced by VC3D's optimizer (needs the volume), so write the
            # merged control points AS the line — trivially satisfying the
            # loader's control-points-on-line invariant — and tag for
            # re-optimization.
            normalized_cps = []
            for cp in merged_cps:
                position = [float(x) for x in _cp_position(cp)]
                normalized_cps.append(
                    {'position': position} if isinstance(cp, dict) else position)
            merged_cps = normalized_cps
            carrier = {'control_points': merged_cps,
                       'line_points': [[float(x) for x in _cp_position(cp)]
                                       for cp in merged_cps]}
            reoptimize = True
            notes.append("geometry merged from both sides; line set to the "
                         "control-point polyline pending reoptimization in "
                         "VC3D")

        merged_branches, anchor_conflict = _rebind_local_anchors(
            merged_branches, carrier['control_points'], carrier['line_points'])
        if anchor_conflict:
            result['conflicts'] = [anchor_conflict]
            return result

        stats['cp_regions_local'] = sum(1 for r in regions if r['owner'] == 'local')
        stats['cp_regions_remote'] = sum(1 for r in regions if r['owner'] == 'remote')
        stats['geometry_merged'] = bool(owners)

    merged = copy.deepcopy(newer)
    merged['control_points'] = copy.deepcopy(carrier['control_points'])
    merged['line_points'] = copy.deepcopy(carrier['line_points'])
    if 'hv_classification' in carrier:
        merged['hv_classification'] = copy.deepcopy(carrier['hv_classification'])
    elif not geometry_same and 'hv_classification' in merged:
        # Synthetic-geometry path: the carried classification is stale for
        # the merged control points; VC3D recomputes it (one benign rewrite
        # on load, corrected permanently by the reoptimization save).
        notes.append("hv_classification is stale for the merged geometry; "
                     "VC3D recomputes it on load")
    merged['branches'] = merged_branches + opaque_branches
    if opaque_branches:
        noun = 'entry' if len(opaque_branches) == 1 else 'entries'
        notes.append(f"{len(opaque_branches)} unparseable branch "
                     f"{noun} carried through unchanged")
    merged['tags'] = merge_tags(base, local, remote)
    if reoptimize and REOPTIMIZE_TAG not in merged['tags']:
        merged['tags'].append(REOPTIMIZE_TAG)
    merged['generation'] = max(generation_local, generation_remote) + 1
    manual_tag_conflict = _merge_manual_hv_tag(base, local, remote, merged,
                                               notes)
    if manual_tag_conflict:
        result['conflicts'] = [manual_tag_conflict]
        return result

    stats['reoptimize'] = reoptimize
    result['peer_files'] = sorted(
        {_branch_target(entry) for entry in links_to_any(merged)} |
        {_branch_target(entry) for entry in links_to_any(base)})
    result.update(ok=True, merged=merged, notes=notes, stats=stats)
    return result


def refresh_pair_links(a_doc, b_doc, a_name, b_name, base_doc=None):
    """Make the A<->B cross-fiber link pair consistent, treating A's
    entries as the decided truth (A was just auto-merged).

    Pure: returns {'ok', 'a_doc', 'b_doc', 'a_changed', 'b_changed',
    'notes', 'conflicts'} over deep copies. Fields already satisfying the
    loader's predicates are left byte-identical, so an in-sync pair yields
    zero changes (and no re-upload). On any unresolvable anchor the caller
    must treat A's merge as a manual conflict — nothing is guessed.

    base_doc gates deletion mirroring: without it, a reciprocal of a link
    A no longer carries is LEFT IN PLACE (e.g. after a moved anchor, the
    peer keeps the stale entry alongside the new one — loader-visible).
    Callers that merged A must always pass A's base.
    """
    a = copy.deepcopy(a_doc)
    b = copy.deepcopy(b_doc)
    out = {'ok': False, 'a_doc': a, 'b_doc': b,
           'a_changed': False, 'b_changed': False, 'notes': [], 'conflicts': []}
    if not is_fiber_doc(a) or not is_fiber_doc(b):
        out['conflicts'].append(f"{a_name} or {b_name} is not a vc3d_fiber document")
        return out

    a_cps, a_line = a['control_points'], a['line_points']
    b_cps, b_line = b['control_points'], b['line_points']
    a_entries = links_to(a, b_name)
    if a_entries and (len(a_line) < 2 or len(b_line) < 2):
        out['conflicts'].append(
            f"cannot derive endpoint directions between {a_name} and {b_name}")
        return out

    def set_field(entry, key, value, doc_flag):
        if entry.get(key) != value:
            entry[key] = value
            out[doc_flag] = True

    # Positions and directions are snapped BYTE-EXACT (to the control
    # point's value / to +-recomputed tangent), never merely
    # within-tolerance: the loader also compares stored-A against stored-B
    # (reciprocity), and two fields each within tolerance of the truth can
    # be 2x tolerance apart from each other. VC3D-written fields are
    # already exact, so consistent pairs still see zero changes.
    def snap_position(entry, key, point, doc_flag):
        set_field(entry, key, [float(x) for x in _cp_position(point)], doc_flag)

    def snap_direction(entry, key, tangent, doc_flag):
        set_field(entry, key, _snapped_direction(entry.get(key), tangent),
                  doc_flag)

    def snap_pending(entry, desired, doc_flag):
        if bool(entry.get('pending', False)) == desired:
            return
        if desired:
            entry['pending'] = True
        else:
            entry.pop('pending', None)
        out[doc_flag] = True

    used_reciprocals = []
    for entry in a_entries:
        local_index = resolve_cp_index(a_cps, entry['control_point_position'])
        far_index = resolve_cp_index(b_cps, entry['branch_control_point_position'])
        if local_index is None or far_index is None:
            out['conflicts'].append(
                f"link {a_name} -> {b_name}: anchor control point not found in "
                f"{a_name if local_index is None else b_name}")
            continue
        pa = a_cps[local_index]
        pb = b_cps[far_index]
        da = endpoint_tangent(a_line, pa)
        db = endpoint_tangent(b_line, pb)

        set_field(entry, 'control_point_index', local_index, 'a_changed')
        snap_position(entry, 'control_point_position', pa, 'a_changed')
        snap_direction(entry, 'control_point_direction', da, 'a_changed')
        set_field(entry, 'branch_control_point_index', far_index, 'a_changed')
        snap_position(entry, 'branch_control_point_position', pb, 'a_changed')
        snap_direction(entry, 'branch_control_point_direction', db, 'a_changed')

        # One reciprocal per entry: two pos_eq-identical A entries must not
        # both claim the same B entry (leaving B one reciprocal short).
        reciprocal = next(
            (r for r in links_to(b, a_name)
             if not any(r is used for used in used_reciprocals) and
             pos_eq(r.get('control_point_position'), pb) and
             pos_eq(r.get('branch_control_point_position'), pa)),
            None)
        if reciprocal is not None:
            used_reciprocals.append(reciprocal)
        if reciprocal is None:
            reciprocal = {
                'control_point_index': far_index,
                # Runtime id; VC3D rebinds it from branch_file at load.
                'branch_fiber_id': 0,
                'branch_control_point_index': local_index,
                'branch_file': a_name,
                'control_point_direction': db,
                'branch_control_point_direction': da,
                'control_point_position': [float(x) for x in pb],
                'branch_control_point_position': [float(x) for x in pa],
            }
            if entry.get('pending', False):
                reciprocal['pending'] = True
            b['branches'] = _branches_of(b) + [reciprocal]
            out['b_changed'] = True
            out['notes'].append(f"restored reciprocal link {b_name} -> {a_name}")
        else:
            set_field(reciprocal, 'control_point_index', far_index, 'b_changed')
            snap_position(reciprocal, 'control_point_position', pb, 'b_changed')
            snap_direction(reciprocal, 'control_point_direction', db, 'b_changed')
            set_field(reciprocal, 'branch_control_point_index', local_index,
                      'b_changed')
            snap_position(reciprocal, 'branch_control_point_position', pa,
                          'b_changed')
            snap_direction(reciprocal, 'branch_control_point_direction', da,
                           'b_changed')
            # Review state stays in lockstep on both refs, as VC3D keeps it.
            snap_pending(reciprocal, bool(entry.get('pending', False)),
                         'b_changed')

    # Pairs present in the base but absent from merged A were deleted by
    # the merge: mirror the deletion. Unrelated B->A entries (pairs A never
    # tracked) are left untouched.
    if base_doc is not None and is_fiber_doc(base_doc):
        for base_entry in links_to(base_doc, b_name):
            if any(_same_link(base_entry, entry) for entry in a_entries):
                continue
            survivors = []
            removed = 0
            for candidate in _branches_of(b):
                if (_structured_branch(candidate) and
                        _branch_target(candidate) == a_name and
                        pos_eq(candidate.get('control_point_position'),
                               base_entry.get('branch_control_point_position')) and
                        pos_eq(candidate.get('branch_control_point_position'),
                               base_entry.get('control_point_position'))):
                    removed += 1
                    continue
                survivors.append(candidate)
            if removed:
                b['branches'] = survivors
                out['b_changed'] = True
                out['notes'].append(
                    f"removed reciprocal of deleted link in {b_name}")

    if out['b_changed']:
        # A rewritten peer is a newer version; keep the generation
        # monotonic so later 3-way merges pick the right "newer" side
        # (VC3D bumps on every save too).
        b['generation'] = int(b.get('generation', 1) or 1) + 1

    out['ok'] = not out['conflicts']
    return out


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
    if stats.get('reoptimize'):
        parts.append("geometry merged (tagged for reoptimization)")
    elif stats.get('geometry_merged'):
        parts.append("geometry merged")
    if result.get('peer_files'):
        parts.append("link consistency over %d peer file(s)" %
                     len(result['peer_files']))
    return ', '.join(parts) if parts else 'merged (metadata only)'
