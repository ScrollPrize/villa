"""Independent mini-port of VC3D's load-time fiber validation.

Extracted verbatim from tests/test_fiber_merge.py so non-test tooling
(fiber_migrate_v1_to_v3.py) can run the same acceptance check. This is
deliberately a SEPARATE implementation from fiber_merge's predicates: it
mirrors the C++ loader (LineAnnotationController.cpp,
FiberSliceGeometry.cpp, Atlas.cpp) so a fiber_merge primitive that drifts
from the loader makes checks FAIL instead of drifting with it.
"""
import math

_REQUIRED_BRANCH_KEYS = (
    'control_point_index', 'branch_control_point_index',
    'control_point_direction', 'branch_control_point_direction',
    'control_point_position', 'branch_control_point_position', 'branch_file')


def _l_finite(p):
    return (isinstance(p, (list, tuple)) and len(p) == 3 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and
                math.isfinite(x) for x in p))


def _l_cp_position(value):
    return value.get('position') if isinstance(value, dict) else value


def _l_pos_eq(a, b, tol=1.0e-6):
    """pointsApproximatelyEqual: EUCLIDEAN ball."""
    if not _l_finite(a) or not _l_finite(b):
        return False
    return sum((x - y) ** 2 for x, y in zip(a, b)) <= tol * tol


def _l_finite_direction(v):
    """finiteDirection: finite and norm > 1e-12."""
    return _l_finite(v) and math.sqrt(sum(x * x for x in v)) > 1.0e-12


def _l_dirs_compatible(a, b, tol=1.0e-5):
    """branchDirectionsCompatible: sign-agnostic, 1e-5 on |cos|."""
    if not _l_finite_direction(a) or not _l_finite_direction(b):
        return False
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    dot = sum((x / na) * (y / nb) for x, y in zip(a, b))
    return abs(abs(dot) - 1.0) <= tol


def _l_tangent(line, point):
    """nearestLinePointIndex + tangentAtLinePosition."""
    best, best_d2 = 0, float('inf')
    for i, p in enumerate(line):
        if not _l_finite(p):
            continue
        d2 = sum((x - y) ** 2 for x, y in zip(p, point))
        if d2 < best_d2:
            best_d2, best = d2, i
    n = len(line)
    lower = max(0, min(best, n - 1))
    upper = min(lower + 1, n - 1)
    if lower == upper and lower > 0:
        lower -= 1
    delta = [line[upper][k] - line[lower][k] for k in range(3)]
    norm = math.sqrt(sum(x * x for x in delta))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return [1.0, 0.0, 0.0]
    return [x / norm for x in delta]


def _l_basename(name):
    return name.rsplit('/', 1)[-1] if isinstance(name, str) else ''


def loader_issues(docs_by_name):
    """Independent mini-port of VC3D's load-time fiber validation. Returns
    a list of issue strings; [] means an unmodified VC3D loads the set
    cleanly and rewrites nothing destructive."""
    issues = []
    for name, doc in docs_by_name.items():
        cps = [_l_cp_position(cp) for cp in doc['control_points']]
        line = doc['line_points']
        for label, points in (('control point', cps), ('line point', line)):
            for k, point in enumerate(points):
                if not _l_finite(point):
                    issues.append(f"{name}: non-finite {label} {k} (fatal)")
        # C1: control points must be an ordered subset of line_points
        # within Euclidean 1e-8 (validateFiberInputControlPoints) — fatal.
        li = 0
        for k, point in enumerate(cps):
            while li < len(line) and not _l_pos_eq(line[li], point, tol=1.0e-8):
                li += 1
            if li >= len(line):
                issues.append(f"{name}: control point {k} not on line (fatal)")
                break
            li += 1
        for entry in doc.get('branches', []):
            # Parse-time stripping: the loader deletes (and rewrites the
            # file without) entries it cannot parse — surfaced as issues so
            # merge output never relies on that destruction.
            if (not isinstance(entry, dict) or
                    'link_direction' in entry or
                    any(key not in entry for key in _REQUIRED_BRANCH_KEYS) or
                    not _l_basename(entry['branch_file'])):
                issues.append(f"{name}: branch entry stripped at parse time "
                              "(file rewritten)")
                continue
            if (not _l_finite_direction(entry['control_point_direction']) or
                    not _l_finite_direction(
                        entry['branch_control_point_direction'])):
                issues.append(f"{name}: non-finite branch direction")
                continue
            i = entry['control_point_index']
            if not (isinstance(i, int) and 0 <= i < len(cps)):
                issues.append(f"{name}: local CP index out of range")
                continue
            if not _l_pos_eq(cps[i], entry['control_point_position']):
                issues.append(f"{name}: local CP position mismatch")
                continue
            if len(line) >= 2:
                tangent = _l_tangent(line, entry['control_point_position'])
                if not _l_dirs_compatible(entry['control_point_direction'],
                                          tangent):
                    issues.append(f"{name}: branch endpoint direction mismatch")
                    continue
            target = docs_by_name.get(_l_basename(entry['branch_file']))
            if target is None:
                issues.append(f"{name}: missing linked fiber "
                              f"{entry['branch_file']}")
                continue
            tcps = [_l_cp_position(cp) for cp in target['control_points']]
            tline = target['line_points']
            j = entry['branch_control_point_index']
            if not (isinstance(j, int) and 0 <= j < len(tcps)):
                issues.append(f"{name}: linked CP index out of range")
                continue
            if not _l_pos_eq(tcps[j], entry['branch_control_point_position']):
                issues.append(f"{name}: linked CP position mismatch")
                continue
            if len(tline) >= 2:
                tangent = _l_tangent(tline,
                                     entry['branch_control_point_position'])
                if not _l_dirs_compatible(
                        entry['branch_control_point_direction'], tangent):
                    issues.append(f"{name}: linked endpoint direction mismatch")
                    continue
            # C2: index-exact reciprocity, plus positions and directions
            # compared STORED against STORED.
            reciprocal = any(
                _l_basename(c.get('branch_file')) == name and
                c.get('control_point_index') == j and
                c.get('branch_control_point_index') == i and
                _l_pos_eq(c.get('control_point_position'),
                          entry['branch_control_point_position']) and
                _l_pos_eq(c.get('branch_control_point_position'),
                          entry['control_point_position']) and
                _l_dirs_compatible(c.get('control_point_direction'),
                                   entry['branch_control_point_direction']) and
                _l_dirs_compatible(c.get('branch_control_point_direction'),
                                   entry['control_point_direction'])
                for c in target.get('branches', []) if isinstance(c, dict))
            if not reciprocal:
                issues.append(f"{name}: missing reciprocal branch in "
                              f"{entry['branch_file']}")
    return issues
