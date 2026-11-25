"""Cut a winding solve's most violated crossings, then find its components again.

A wgraph winding run (fiber_solve_inputs) solves its fibers' windings from
measured crossings, each stating the winding step ``t`` (turns) between two
fibers; one badly violated crossing can tie a mislabelled group into the
component the fit holds to absolute windings. With
``fiber_constraint_cut_turns`` positive, read_winding_solve therefore:

1. rebuilds the run's crossings with the winding stage's own builder,
   vc_object_store ``PackedWindingGeometry.prepare_winding_graph``
   (read_measurements, build_crossings): the run store's measurements, the
   run's fibers and families, the signs of the normal-sign run it pins, its
   recorded weights, and the whole fibers' seam transport around the
   manifest's umbilicus;
2. scores each crossing by ``|W_b - W_a - t|`` under the run's
   ``selected_winding`` W (for an integer step, the solver's two weighted
   integer arcs for ``t`` have exactly this residual);
3. cuts those violated by more than the threshold and finds the components
   of the rest (cut_components), every fiber of the run a node.

The solve's components are the starting truth. A crossing between two of
them is one the run's crossing filter removed and is dropped. Constraints
the store does not hold (fixed_object groups, scaffold_reference, manual
links) are not rebuilt, so where the readable crossings leave a solve
component in pieces the missing links count as satisfied: after the cut each
piece's largest kept part stays joined to the largest kept part of the
component's main (largest) piece. With nothing cut the result is the solve's
partition; a fiber leaves its solve component only where cut crossings
disconnect it.
"""
import hashlib

import numpy as np

# The violation histogram's bounds, in turns.
CUT_HISTOGRAM_TURNS = (1, 2, 4, 8, 16)
# Native workers for the measurement read (its transient memory grows with
# them: about 3.3 GB with 4, 4 GB with 16 on PHerc0332's 6.6 M crossings) and
# the graph build.
WORKERS = 8
# The run store's input row naming the run's normal-sign run.
INPUT_ROLE = 'input_0'
SIGN_STAGE = 'normal_sign'
# Recorded run settings the rebuild reads.
WEIGHT_SETTINGS = ('weight_parallel', 'weight_perpendicular', 'normal_dot_weight_power',
                   'winding_distance_damped')


def cut_enabled(cut_turns):
    """Whether ``fiber_constraint_cut_turns`` cuts: None or non-positive
    keeps the solve's components."""
    return cut_turns is not None and float(cut_turns) > 0


def canonical_labels(labels):
    """``labels`` renumbered 0, 1, ... in the order of each label's first fiber."""
    labels = np.asarray(labels)
    if not len(labels):
        return np.zeros(0, dtype=np.int64)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind='stable')] = np.arange(len(first))
    return rank[inverse.reshape(-1)]


def partition_sha256(labels):
    """A digest of the partition ``labels`` states, whatever its label values."""
    return hashlib.sha256(canonical_labels(labels).astype('<i8').tobytes()).hexdigest()


def _components(count, a, b):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(count, count))
    return canonical_labels(connected_components(graph, directed=False)[1])


def _first_per_group(group, key, value):
    """Per group (0..G-1, every one present), ``value`` at the largest ``key``
    (the smallest ``value`` on a tie)."""
    order = np.lexsort((value, -key, group))
    first = np.flatnonzero(np.r_[True, np.diff(group[order]) != 0])
    return value[order][first]


def cut_components(solve_component, a, b, violation, cut_turns):
    """The components left after cutting the crossings ``a``, ``b`` (fiber
    indices) violated by more than ``cut_turns`` (``violation`` in turns),
    starting from the run's ``solve_component`` per fiber (module step 3).
    Returns the canonical labels int64 [F] and the cut's log line."""
    solve = canonical_labels(solve_component)
    count = len(solve)
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    violation = np.asarray(violation, dtype=np.float64)
    inside = solve[a] == solve[b]
    cut = inside & (violation > float(cut_turns))
    uncut = _components(count, a[inside], b[inside])
    component = kept = _components(count, a[inside & ~cut], b[inside & ~cut])
    joined_pieces = joined_fibers = 0
    if count:
        # Each uncut piece's anchor: the kept part holding most of its fibers;
        # every piece's anchor joins that of its solve component's main piece.
        pairs, sizes = np.unique(np.stack([uncut, kept], 1), axis=0, return_counts=True)
        anchor = _first_per_group(pairs[:, 0], sizes, pairs[:, 1])
        piece_solve = np.zeros(len(anchor), dtype=np.int64)
        piece_solve[uncut] = solve
        piece_size = np.bincount(uncut)
        main_piece = _first_per_group(piece_solve, piece_size, np.arange(len(anchor)))
        joins = _components(int(kept.max()) + 1, anchor, anchor[main_piece[piece_solve]])
        component = canonical_labels(joins[kept])
        joined_pieces = len(anchor) - len(main_piece)
        joined_fibers = count - int(piece_size[main_piece].sum())
    solve_sizes = np.bincount(solve, minlength=1)
    sizes = np.bincount(component, minlength=1)
    left = int(((solve == solve_sizes.argmax()) & (component != sizes.argmax())).sum())
    between = int((~inside).sum())
    histogram = ', '.join(f'> {bound:g}: {int((inside & (violation > bound)).sum()):,}'
                          for bound in CUT_HISTOGRAM_TURNS)
    line = (f'fiber constraints: violation |W_b - W_a - t| {histogram} turns; '
            f'cut {int(cut.sum()):,} above {float(cut_turns):g} turns'
            + (f', dropped {between:,} joining two solve components' if between else '')
            + f'; components {len(np.unique(solve)):,} -> {len(np.unique(component)):,}, '
            f'largest {int(solve_sizes.max()):,} -> {int(sizes.max()):,} fibers, '
            f'{left:,} left the largest'
            + (f'; {joined_pieces:,} pieces ({joined_fibers:,} fibers) the read crossings leave '
               'apart kept joined' if joined_pieces else ''))
    return component, line


def _uuid_keys(uuids):
    return np.ascontiguousarray(uuids, dtype=np.uint8).view('S16').reshape(-1)


def _align(source_uuids, wanted_uuids):
    """Index into ``source_uuids`` (UUID-ordered) of every wanted UUID, -1
    where absent."""
    source, wanted = _uuid_keys(source_uuids), _uuid_keys(wanted_uuids)
    order = np.argsort(source, kind='stable')
    at = np.minimum(np.searchsorted(source[order], wanted), max(len(source) - 1, 0))
    found = len(source) > 0
    hit = (source[order][at] == wanted) if found else np.zeros(len(wanted), dtype=bool)
    return np.where(hit, order[at] if found else -1, -1).astype(np.int64)


def _input_run(reader, run_uuid):
    """The run ``run_uuid``'s pinned input run (its ``input_0`` row)."""
    import uuid
    from vc_object_store import fiber_graph_solve_table
    from fiber_solve_inputs import WindingSolveError
    table = fiber_graph_solve_table('inputs')
    names = [column['name'] for column in table['columns']]
    wanted = uuid.UUID(run_uuid).bytes
    rows = [dict(zip(names, row)) for row in reader.extension_rows(table['name'], [wanted])]
    rows = [row for row in rows if row['role'] == INPUT_ROLE and row['input_run_uuid'] is not None]
    if len(rows) != 1:
        raise WindingSolveError(f'run {run_uuid} pins no single normal-sign run')
    return str(uuid.UUID(bytes=bytes(rows[0]['input_run_uuid'])))


# The measurement columns the graph build reads.
MEASUREMENT_COLUMNS = ('endpoints', 'arcs', 'distance', 'parallel', 'normal_a', 'normal_b',
                       'connector', 'valid_a', 'valid_b', 'valid_connector')


def read_measurements(reader, run_uuid, settings, fiber_uuid, volume_uuid):
    """The measured crossings of the run store's lineage and the run's
    normal signs, for build_crossings (module step 1).

    ``reader`` is open on the run store; ``settings`` the run's recorded
    settings; ``fiber_uuid`` its fibers (the run's rows, UUID order), read
    in ``volume_uuid``'s voxels. Refuses (WindingSolveError) a run whose
    crossings cannot be rebuilt."""
    from fiber_solve_inputs import WindingSolveError, _run_settings

    missing = [key for key in WEIGHT_SETTINGS if key not in settings]
    if missing:
        raise WindingSolveError(
            f'run {run_uuid} records no {", ".join(missing)}: its crossings cannot be rebuilt; '
            'set fiber_constraint_cut_turns to null to use its components as they are')
    if not settings.get('normal_sign_from_run', True):
        raise WindingSolveError(
            f'run {run_uuid} was solved without a normal-sign run (manual links only): it has '
            'no measured crossings to cut; set fiber_constraint_cut_turns to null')
    sign_run = _input_run(reader, run_uuid)
    stage, _ = _run_settings(reader, sign_run)
    if stage != SIGN_STAGE:
        raise WindingSolveError(f'run {run_uuid}\'s input run {sign_run} is a {stage!r} run, '
                                f'not a {SIGN_STAGE} run')
    signs = reader.fiber_result_columns(sign_run)
    at = _align(signs['fiber_uuid'], fiber_uuid)
    column = list(signs['columns']).index('normal_sign')
    if (at < 0).any() or not signs['valid'][at[at >= 0], column].all():
        raise WindingSolveError(f'the normal-sign run {sign_run} gives no sign to some of '
                                f'run {run_uuid}\'s fibers')
    normal_sign = np.ascontiguousarray(signs['values'][at, column], dtype=np.int64)
    del signs
    try:
        graph = reader.fiber_graph_columns(WORKERS, False)
    except ValueError as error:
        raise WindingSolveError(
            f'run {run_uuid}: its crossings cannot be rebuilt ({error}); set '
            'fiber_constraint_cut_turns to null to use its components as they are') from error
    if str(graph['volume']) != str(volume_uuid):
        raise WindingSolveError(
            f'the crossings are measured in the volume {graph["volume"]!r}, the fibers read '
            f'in {volume_uuid!r}: their anchors would not match')
    measurements = {name: graph[name] for name in MEASUREMENT_COLUMNS}
    measurements.update(graph_index=_align(fiber_uuid, graph['fiber_uuid']),
                        normal_sign=normal_sign, run_uuid=run_uuid, settings=dict(settings))
    return measurements


def build_crossings(measurements, family, geometry, axis_xyz, handedness):
    """The crossings the run solved from, rebuilt (module step 1) from
    read_measurements' ``measurements`` (released here), the run's fiber
    ``family`` and the fibers' packed ``geometry`` (offsets, arc, xyz, turns),
    vc_object_store.pack_winding_geometry around ``axis_xyz`` with
    ``handedness``. Returns ``a``, ``b`` int64 (fiber indices), ``target``
    float64 (turns) and the log lines (the counts, what is not rebuilt)."""
    from vc_object_store import PackedWindingGeometry
    from fiber_solve_inputs import SEAM_ANGLE, WindingSolveError
    graph = dict(measurements)
    measurements.clear()
    run_uuid, settings = graph['run_uuid'], graph['settings']
    offsets, arc, xyz, turns = geometry
    axis = np.ascontiguousarray(axis_xyz, dtype=np.float64)
    native = PackedWindingGeometry(offsets, arc, xyz, turns, axis, SEAM_ANGLE, int(handedness))
    flag = lambda name: np.asarray(graph[name]).view(bool)  # noqa: E731
    endpoints, arcs = graph['endpoints'], graph['arcs']
    enclosures = float(settings.get('weight_enclosure', 0.0)) > 0
    try:
        built = native.prepare_winding_graph(
            np.ascontiguousarray(endpoints[:, 0]), np.ascontiguousarray(endpoints[:, 1]),
            graph['graph_index'], np.ascontiguousarray(family, dtype=np.int64),
            graph['normal_sign'],
            np.ascontiguousarray(arcs[:, 0]), np.ascontiguousarray(arcs[:, 1]),
            np.asarray(graph['distance']), flag('parallel'),
            graph['normal_a'], graph['normal_b'], graph['connector'],
            flag('valid_a'), flag('valid_b'), flag('valid_connector'),
            parallel_weight=float(settings['weight_parallel']),
            perpendicular_weight=float(settings['weight_perpendicular']),
            dot_weight_power=float(settings['normal_dot_weight_power']),
            damp_by_winding_distance=bool(settings['winding_distance_damped']),
            enclosures=enclosures, event_targets=True, workers=WORKERS)
    except ValueError as error:
        raise WindingSolveError(f'run {run_uuid}: its crossings cannot be rebuilt: {error}')
    del graph, endpoints, arcs, native
    a = np.asarray(built['event_a'], dtype=np.int64)
    b = np.asarray(built['event_b'], dtype=np.int64)
    target = np.asarray(built['event_target'], dtype=np.float64)
    if enclosures:
        # Exact H -> V relations: one more constraint each, with an integer
        # target, between fibers a crossing already joins.
        a = np.concatenate([a, np.asarray(built['enclosure_a'], dtype=np.int64)])
        b = np.concatenate([b, np.asarray(built['enclosure_b'], dtype=np.int64)])
        target = np.concatenate([target, np.asarray(built['enclosure_target'], dtype=np.float64)])
    read, used = int(built['read']), int(built['used'])
    del built
    recorded = int(settings.get('automatic_used', -1))
    filtered = int(settings.get('automatic_crossing_filtered', 0))
    lines = [f'fiber constraints: {used:,} usable crossings of {read:,} measured '
             f'(the run recorded {recorded:,})']
    caveats = []
    if filtered:
        caveats.append(f'its crossing filter removed {filtered:,} crossings this build cannot '
                       f'identify, so {used - recorded:,} extra ones are rebuilt and scored')
    unread = [name for name, present in (
        ('fixed_object groups', settings.get('fixed_object')),
        ('scaffold_reference constraints', settings.get('scaffold_reference')),
        ('manual links', float(settings.get('weight_manual', 0.0)) > 0)) if present]
    if unread:
        caveats.append(f'its {", ".join(unread)} are not in the store: what only they join is '
                       'kept joined, never cut')
    if caveats:
        lines.append('fiber constraints: NOT FULLY REPRODUCED: the run: ' + '; '.join(caveats))
    if not filtered and recorded >= 0 and used != recorded:
        lines.append(f'fiber constraints: WARNING: {used:,} usable crossings rebuilt, the run '
                     f'recorded {recorded:,}: the rebuild does not select the run\'s crossings')
    return a, b, target, lines


def violation(selected_winding, a, b, target):
    """Each crossing's violation under the solve, |W_b - W_a - t| in turns."""
    W = np.asarray(selected_winding, dtype=np.float64)
    return np.abs(W[b] - W[a] - np.asarray(target, dtype=np.float64))
