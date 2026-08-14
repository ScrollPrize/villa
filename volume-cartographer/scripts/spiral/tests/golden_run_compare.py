"""Shared logic for the golden-run harness: flattening driver results into
leaf paths, calibrating tolerance bands from repeated baseline runs, and
comparing a fresh run against the calibrated golden file.

A driver result (see tests/golden_run_driver.py) is flattened into
{path: leaf} where path is a JSON-encoded list of keys/indices. Leaves are
classified during calibration:
  - identical across all baseline runs -> asserted exactly;
  - numeric and varying -> asserted within a band derived from the observed
    run-to-run spread plus margin (GPU kernels are not deterministic enough
    for bit-exact assertions);
  - non-numeric and varying -> dropped, recorded under "ignored" so the
    golden file documents what is not covered.
Paths matching VOLATILE_PATTERNS (timings) are never asserted.
"""

import json
import math
import re

VOLATILE_PATTERNS = [
    r'_seconds',
    r'_timing',
    r'per_sec',
]

# Numeric leaves under these prefixes are stochastic by policy (GPU-order
# dependent), even when a small calibration sample happens to agree exactly:
# loss traces and satisfaction metrics. Everything else (config round-trip,
# checkpoint structure, RNG-state hashes) must match exactly.
STOCHASTIC_PREFIXES = (
    '["metrics"',
    '["satisfied"',
)

# Per-entry satisfaction values are not asserted at all. They are quantized
# in coarse cells, so one borderline cell moves satisfied_area by hundreds,
# and there are ~1600 of them: banding each one still leaves a high chance
# that some run trips a single band, which makes the gate flaky without
# making it more sensitive. The regression signal for satisfaction lives in
# the satisfied_aggregates sums (whose spread is an order of magnitude
# tighter because the per-entry noise averages out) and in the exact entry
# identities and counts, both of which are still asserted.
PER_ENTRY_SATISFIED_RE = re.compile(
    r'^\["satisfied", "(patches|pcls|unverified_patches)", "\d+", '
    r'"(fraction|satisfied_area|satisfied_points)"')
PER_ENTRY_SATISFIED_REASON = (
    'per-entry satisfaction value: quantized and individually noisy; '
    'covered by satisfied_aggregates')

# Margin beyond the observed spread: bands are centre +/- (SPREAD_MARGIN *
# half-spread + REL_MARGIN * |centre| + ABS_MARGIN). Calibrated from a small
# number of runs, so the observed spread underestimates the true one; leaves
# with zero observed spread get the wider ZERO_SPREAD_* margins since their
# agreement was coincidental.
SPREAD_MARGIN = 3.0
REL_MARGIN = 0.05
ABS_MARGIN = 1e-6
ZERO_SPREAD_REL_MARGIN = 0.15
ZERO_SPREAD_ABS_MARGIN = 1.0


def flatten(value, prefix=()):
    """Flatten nested dicts/lists to {json_path: leaf}."""
    items = {}
    if isinstance(value, dict):
        for key, child in value.items():
            items.update(flatten(child, prefix + (str(key),)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            items.update(flatten(child, prefix + (str(index),)))
    else:
        items[json.dumps(list(prefix))] = value
    return items


def is_volatile(path):
    return any(re.search(pattern, path) for pattern in VOLATILE_PATTERNS)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def calibrate(results, spec):
    """Build the golden file from several unmodified-code driver results."""
    flattened = [flatten(result) for result in results]
    all_paths = set().union(*[set(f) for f in flattened])
    shared_paths = set(flattened[0]).intersection(*[set(f) for f in flattened[1:]])
    unstable_presence = sorted(all_paths - shared_paths)

    exact = {}
    bands = {}
    ignored = []
    for path in sorted(shared_paths):
        if is_volatile(path):
            ignored.append({'path': path, 'reason': 'volatile'})
            continue
        if PER_ENTRY_SATISFIED_RE.match(path):
            ignored.append({'path': path, 'reason': PER_ENTRY_SATISFIED_REASON})
            continue
        values = [f[path] for f in flattened]
        numeric = all(_is_number(value) and math.isfinite(value) for value in values)
        stochastic = numeric and path.startswith(STOCHASTIC_PREFIXES)
        if not stochastic and all(value == values[0] for value in values[1:]):
            exact[path] = values[0]
        elif stochastic:
            low, high = min(values), max(values)
            centre = (low + high) / 2
            if low == high:
                margin = (ZERO_SPREAD_REL_MARGIN * abs(centre)
                          + ZERO_SPREAD_ABS_MARGIN)
            else:
                margin = (SPREAD_MARGIN * (high - low) / 2
                          + REL_MARGIN * abs(centre) + ABS_MARGIN)
            bands[path] = [low - margin, high + margin]
        elif numeric:
            # Non-stochastic numeric leaf that varies across baseline runs:
            # band it from the observed spread rather than failing.
            low, high = min(values), max(values)
            centre = (low + high) / 2
            margin = (SPREAD_MARGIN * (high - low) / 2
                      + REL_MARGIN * abs(centre) + ABS_MARGIN)
            bands[path] = [low - margin, high + margin]
        else:
            ignored.append({'path': path, 'reason': 'varying non-numeric',
                            'values': [repr(v) for v in values]})
    for path in unstable_presence:
        ignored.append({'path': path, 'reason': 'not present in every run'})

    return {
        'spec': spec,
        'num_calibration_runs': len(results),
        'exact': exact,
        'bands': bands,
        'ignored': ignored,
    }


def compare(golden, result):
    """Compare one fresh driver result against the golden file.

    Returns a list of human-readable failure strings (empty = pass).
    """
    flattened = flatten(result)
    ignored_paths = {entry['path'] for entry in golden.get('ignored', [])}
    failures = []

    for path, expected in golden['exact'].items():
        if path not in flattened:
            failures.append(f'missing path {path} (expected exact {expected!r})')
        elif flattened[path] != expected:
            failures.append(
                f'{path}: expected exactly {expected!r}, got {flattened[path]!r}')

    for path, (low, high) in golden['bands'].items():
        if path not in flattened:
            failures.append(f'missing path {path} (expected in [{low}, {high}])')
            continue
        value = flattened[path]
        if not (_is_number(value) and math.isfinite(value)):
            failures.append(f'{path}: expected a finite number, got {value!r}')
        elif not (low <= value <= high):
            failures.append(f'{path}: {value} outside band [{low}, {high}]')

    known = set(golden['exact']) | set(golden['bands']) | ignored_paths
    unexpected = [path for path in flattened
                  if path not in known and not is_volatile(path)]
    for path in sorted(unexpected):
        failures.append(f'unexpected new path {path} = {flattened[path]!r}')

    return failures
