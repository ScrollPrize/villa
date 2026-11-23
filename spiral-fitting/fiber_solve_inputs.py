"""Fibers and one fiber winding solve (wgraph), read as fit inputs.

The input manifest's ``winding_solve`` block names a fiber store, the store
holding the solve's results (``run_store``, a result overlay whose lineage
includes the fiber store) and the winding run in it (``run_uuid``). The run
gives every fiber it kept its H/V family, connected component (CC) and
``selected_winding`` W, the winding index at the fiber's first vertex.

At startup (read_winding_solve), for every fiber with a run row:

- ``w``, the continuous winding at each vertex, is computed on the whole fiber
  (fiber_winding_coordinate) around the manifest's umbilicus, with the
  handedness the scroll's ``spiral_outward_sense`` fixes;
- the vertices inside the fit's z window are kept, as one or more pieces per
  fiber, decimated along each piece to ``fiber_min_point_spacing``.

Once the model exists, WindingSolve.set_seam_offset measures ``c``, the
offset between the solve's seam and Spiral's theta = 0 under the step-0
transform, and sets ``w_s = w - c`` at the kept points. Nothing reads these
inputs as a loss yet.
"""
import dataclasses
import json
import math
import uuid

import numpy as np

# The seam is the half-plane from the umbilicus towards this XY ray (-y), as
# vc_object_store's WindingSeam defaults it.
SEAM_ANGLE = -math.pi / 2
# Load guard: a vertex closer than this to the axis in XY, or a step between
# consecutive vertices of more than this many turns, refuses the solve. The
# unwrapped angle counts the seam crossings exactly while a step sweeps less
# than half a turn, which a straight segment does around a straight axis;
# the margin covers an axis knot inside a segment. (PHerc0332's largest step
# is 0.32 turns, one fiber passing 2.6 voxels from the axis; then 0.25.)
AXIS_DISTANCE_MIN = 1e-6
STEP_TURNS_MAX = 0.4
# c's spread check: the 99th percentile of the circular deviation from c.
SEAM_OFFSET_SPREAD_MAX = 0.05
# Run fibers read per geometry call: bounds the startup's peak memory.
READ_CHUNK_FIBERS = 16384
# The vc_object_store table of winding-solve runs and its winding stage.
RUNS_TABLE_ROLE = 'runs'
WINDING_STAGE = 'winding'
RESULT_COLUMNS = ('family', 'enabled', 'normal_sign', 'selected_winding', 'component')
# The run's family codes (FiberGraphSolveExtension: H=0, V=1, broken=2).
FAMILY_H, FAMILY_V = 0, 1


class WindingSolveError(ValueError):
    """The winding solve cannot be read as a fit input."""


def handedness_for(spiral_outward_sense):
    """The solve's winding handedness for a scroll's outward sense: -1 for
    ACW, +1 for CW (both derive from the same catalog flags)."""
    if spiral_outward_sense == 'ACW':
        return -1
    if spiral_outward_sense == 'CW':
        return 1
    raise ValueError(f'spiral_outward_sense must be CW or ACW, not {spiral_outward_sense!r}')


def fiber_winding_coordinate(xyz, W, axis_xyz, handedness):
    """Continuous winding w at every vertex of one fiber, ordered as stored.

    W is the solve's selected_winding, the winding at the first vertex. Must
    match WindingSeam (vc_object_store geometry/FiberWinding.hpp) in axis
    (linear in z, clamped at its ends), seam ray (-pi/2) and handedness sign;
    test_fiber_winding_coordinate checks it. Equivalent to

        order = np.argsort(axis_xyz[:, 2])
        cx = np.interp(xyz[:, 2], axis_xyz[order, 2], axis_xyz[order, 0])
        cy = np.interp(xyz[:, 2], axis_xyz[order, 2], axis_xyz[order, 1])
        alpha = np.unwrap(np.arctan2(xyz[:, 1] - cy, xyz[:, 0] - cx))
        u = handedness * (alpha + np.pi / 2) / (2 * np.pi)   # turns past the seam ray
        return W + np.mod(u[0], 1.0) + (u - u[0])

    after the load guard (packed_winding_coordinate).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    return packed_winding_coordinate(
        np.array([0, len(xyz)], dtype=np.int64), xyz, np.array([W]), axis_xyz, handedness)


def packed_winding_coordinate(offsets, xyz, W, axis_xyz, handedness, name=None):
    """fiber_winding_coordinate for many fibers at once: fiber i is
    ``xyz[offsets[i]:offsets[i + 1]]`` with first-vertex winding ``W[i]``.

    The load guard refuses (WindingSolveError, naming the fiber by
    ``name(i)``, default its index) a non-finite vertex, a vertex within
    AXIS_DISTANCE_MIN of the axis in XY, or a step of more than
    STEP_TURNS_MAX turns between consecutive vertices: there the unwrapped
    angle would not be the seam's crossing count.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float64)
    W = np.asarray(W)
    counts = np.diff(offsets)
    first = offsets[:-1]
    if handedness not in (1, -1):
        raise ValueError('winding handedness must be +1 or -1')
    if len(counts) != len(W) or (counts < 1).any() or offsets[-1] != len(xyz):
        raise ValueError('invalid packed fiber offsets')
    name = name or str
    owner = np.repeat(np.arange(len(counts)), counts)

    def refuse(points, what):
        fiber = int(owner[np.flatnonzero(points)[0]])
        raise WindingSolveError(f'fiber {name(fiber)}: {what}')

    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        refuse(~finite, 'a vertex is not finite')
    axis = np.asarray(axis_xyz, dtype=np.float64)
    order = np.argsort(axis[:, 2], kind='stable')
    rx = xyz[:, 0] - np.interp(xyz[:, 2], axis[order, 2], axis[order, 0])
    ry = xyz[:, 1] - np.interp(xyz[:, 2], axis[order, 2], axis[order, 1])
    near = np.hypot(rx, ry) < AXIS_DISTANCE_MIN
    if near.any():
        refuse(near, f'a vertex lies within {AXIS_DISTANCE_MIN:g} of the umbilicus in XY')
    alpha = np.arctan2(ry, rx)
    # np.unwrap's step: the angle change mapped into [-pi, pi], +pi kept.
    raw = np.diff(alpha)
    step = np.mod(raw + np.pi, 2 * np.pi) - np.pi
    step[(step == -np.pi) & (raw > 0)] = np.pi
    step = np.concatenate([[0.0], step])
    step[first] = 0.0
    too_far = np.abs(step) / (2 * np.pi) > STEP_TURNS_MAX
    if too_far.any():
        refuse(too_far, f'consecutive vertices are more than {STEP_TURNS_MAX} turns apart '
                        'around the umbilicus')
    # Unwrapped angle relative to each fiber's first vertex, per fiber.
    total = np.cumsum(step)
    relative = total - np.repeat(total[first], counts)
    u_first = handedness * (alpha[first] + np.pi / 2) / (2 * np.pi)
    return (np.repeat(W.astype(np.float64) + np.mod(u_first, 1.0), counts)
            + handedness * relative / (2 * np.pi))


def decimate_pieces(zyx, piece_offsets, spacing):
    """Indices of the points kept from consecutive pieces ``zyx[piece_offsets[k]:
    piece_offsets[k + 1]]``: each piece's first point, then greedily the
    next point at least ``spacing`` further along the piece (arc length)
    than the last kept one, and its last point. ``spacing`` <= 0 keeps all."""
    piece_offsets = np.asarray(piece_offsets, dtype=np.int64)
    if spacing <= 0 or len(zyx) == 0:
        return np.arange(len(zyx))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(np.asarray(zyx, dtype=np.float64), axis=0), axis=1))])
    current, last = piece_offsets[:-1].copy(), piece_offsets[1:] - 1
    kept = [current.copy()]
    active = current < last
    while active.any():
        at = current[active]
        following = np.minimum(np.searchsorted(arc, arc[at] + spacing, side='left'),
                               last[active])
        current[active] = following
        kept.append(following)
        active = current < last
    return np.unique(np.concatenate(kept))


def circular_median(turns):
    """``(c, deviation)``: a median of ``turns`` on the circle of one turn,
    in [0, 1), and each value's signed deviation from it in [-0.5, 0.5)."""
    angles = 2 * np.pi * np.asarray(turns, dtype=np.float64)
    centre = math.atan2(np.sin(angles).mean(), np.cos(angles).mean()) / (2 * np.pi)
    around = np.mod(turns - centre + 0.5, 1.0) - 0.5
    c = float(np.mod(centre + np.median(around), 1.0))
    return c, np.mod(turns - c + 0.5, 1.0) - 0.5


def undeformed_theta(model, zyx, chunk=200_000):
    """Wrapped Spiral angle theta_0 (float32 semantics, as float64) of fit-frame
    ZYX points under the model's step-0 slice-to-spiral transform, which
    does not depend on the trained parameters (ported from the frozen
    winding_convention.undeformed_phi)."""
    import torch
    from sample_spiral import get_theta
    zyx = np.asarray(zyx, dtype=np.float32)
    out = np.empty(len(zyx), dtype=np.float64)
    transform = model.get_slice_to_spiral_transform(truncate_at_step=0)
    with torch.inference_mode():
        for lo in range(0, len(zyx), chunk):
            spiral = transform(torch.from_numpy(zyx[lo:lo + chunk]).to(model.device))
            out[lo:lo + chunk] = get_theta(spiral[:, 1:])[0].double().cpu().numpy()
    return out


@dataclasses.dataclass
class WindingSolve:
    """One winding solve's fibers, as the fit reads them.

    Per fiber, F of them in the run's fiber-UUID order (the store's fibers
    with an enabled run row; fibers without a row or with a disabled one
    are dropped):
    ``fiber_uuid`` uint8 [F, 16], ``family`` int8 [F] (FAMILY_H / FAMILY_V),
    ``selected_winding`` int64 [F] (W, at the first vertex),
    ``component`` int64 [F] (the run's CC id), ``fiber_cc`` int64 [F] (its
    index into the per-CC arrays), and ``w_min`` / ``w_max`` float64 [F], the
    extremes of w over the whole fiber (the undeformed sheet labels n0 of
    its points lie in [floor(w_min - c), floor(w_max - c)]).

    Per CC, C of them in component-id order: ``cc_component`` int64 [C],
    ``cc_fiber_count`` int64 [C]; ``largest_cc`` is the index of the CC with
    the most fibers.

    Kept points, P of them, grouped by fiber in fiber order and along each
    fiber in stored order, all inside the z window: ``zyx`` float32 [P, 3]
    (fit frame), ``w`` float64 [P], ``point_fiber`` int64 [P]; pieces
    (maximal runs of consecutive in-window vertices of one fiber, after
    decimation) are ``piece_offsets`` int64 [K + 1] into the points and
    ``piece_fiber`` int64 [K].

    ``c``, ``c_spread_p99`` and ``w_s`` (= w - c, float64 [P]) are None until
    set_seam_offset.
    """
    fibers_path: str
    run_store_path: str
    run_uuid: str
    # The fiber store's position the run consumed (read there) and the run
    # store's source position.
    fiber_store_position: dict
    run_store_position: dict
    handedness: int
    store_fiber_count: int
    # Run rows with enabled = 0, dropped.
    disabled_count: int
    fiber_uuid: np.ndarray
    family: np.ndarray
    selected_winding: np.ndarray
    component: np.ndarray
    fiber_cc: np.ndarray
    w_min: np.ndarray
    w_max: np.ndarray
    cc_component: np.ndarray
    cc_fiber_count: np.ndarray
    largest_cc: int
    z_window: tuple
    min_point_spacing: float
    window_point_count: int
    max_step_turns: float
    zyx: np.ndarray
    w: np.ndarray
    point_fiber: np.ndarray
    piece_offsets: np.ndarray
    piece_fiber: np.ndarray
    c: float = None
    c_spread_p99: float = None
    w_s: np.ndarray = None

    def fiber_name(self, index):
        return str(uuid.UUID(bytes=self.fiber_uuid[index].tobytes()))

    def report(self):
        """The startup log lines."""
        kept_fibers = np.unique(self.point_fiber)
        in_largest = int((self.fiber_cc[kept_fibers] == self.largest_cc).sum())
        without_row = self.store_fiber_count - len(self.fiber_uuid) - self.disabled_count
        z0, z1 = self.z_window
        return [
            f'winding solve: run {self.run_uuid} (handedness {self.handedness:+d}) of '
            f'{self.store_fiber_count:,} fibers: {len(self.fiber_uuid):,} in the run; '
            f'dropped {without_row:,} without a run row and {self.disabled_count:,} '
            f'disabled; '
            f'{int((self.family == FAMILY_H).sum()):,} H, '
            f'{int((self.family == FAMILY_V).sum()):,} V',
            f'winding solve: {len(self.cc_component):,} components, largest '
            f'{int(self.cc_fiber_count[self.largest_cc]):,} fibers, '
            f'{int((self.cc_fiber_count == 1).sum()):,} single fibers; '
            f'largest step of w between consecutive vertices {self.max_step_turns:.3f} turns',
            f'winding solve: z window [{z0:g}, {z1:g}): {len(kept_fibers):,} fibers '
            f'({in_largest:,} in the largest component), {len(self.piece_fiber):,} pieces, '
            f'{len(self.w):,} of {self.window_point_count:,} points kept at spacing '
            f'{self.min_point_spacing:g}']

    def set_seam_offset(self, theta0):
        """Measure c, the circular median over the kept points of
        phi_seam - theta_0 / 2pi (phi_seam = w mod 1), and set w_s = w - c.
        ``theta0`` is Spiral's wrapped angle at the kept points under the
        step-0 transform (undeformed_theta). Refuses the solve when the 99th
        percentile of the deviation from c exceeds SEAM_OFFSET_SPREAD_MAX
        turns (a wrong handedness or fiber scale). Returns the log line."""
        if not len(self.w):
            self.c, self.c_spread_p99, self.w_s = None, None, self.w.copy()
            return 'winding solve: no fiber point in the z window; c not measured'
        c, deviation = circular_median(
            np.mod(self.w, 1.0) - np.asarray(theta0, dtype=np.float64) / (2 * np.pi))
        spread = float(np.percentile(np.abs(deviation), 99))
        if spread > SEAM_OFFSET_SPREAD_MAX:
            raise WindingSolveError(
                f'the solve\'s seam is not at a constant angle from Spiral\'s theta = 0: '
                f'c = {c:.3f} turns, but the 99th percentile of the deviation from it is '
                f'{spread:.3f} turns (at most {SEAM_OFFSET_SPREAD_MAX}); check the '
                'handedness, the umbilicus and the fiber store\'s scale')
        self.c, self.c_spread_p99, self.w_s = c, spread, self.w - c
        return (f'winding solve: c = {c:.3f} turns (seam vs Spiral theta = 0), '
                f'99th-percentile deviation {spread:.4f} turns over {len(self.w):,} points')


def _run_settings(reader, run_uuid):
    """``(stage, settings)`` of the run's row in the runs table."""
    from vc_object_store import fiber_graph_solve_table
    table = fiber_graph_solve_table(RUNS_TABLE_ROLE)
    names = [column['name'] for column in table['columns']]
    wanted = uuid.UUID(run_uuid).bytes
    for row in reader.extension_rows(table['name']):
        record = dict(zip(names, row))
        if bytes(record['run_uuid']) == wanted:
            return record['stage'], json.loads(record['settings'] or '{}')
    raise WindingSolveError(f'the run store has no run {run_uuid}')


def _consumed_position(run_reader, fiber_source, fibers_path):
    """The fiber store's position the run store's lineage pins."""
    same_store = [member for member in run_reader.members
                  if member['store_uuid'] == fiber_source['store_uuid']]
    for member in same_store:
        if member['generation'] == fiber_source['generation']:
            return {'store': member['store_uuid'], 'generation': member['generation'],
                    'commit_sequence': member['commit_sequence']}
    if same_store:
        raise WindingSolveError(
            f'the run store was solved from another generation of the fiber store '
            f'{fibers_path} (store {fiber_source["store_uuid"]}): its lineage names '
            f'generation {same_store[0]["generation"]}, the store is at generation '
            f'{fiber_source["generation"]}')
    raise WindingSolveError(
        f'the run store was not solved from the fiber store {fibers_path} '
        f'(store {fiber_source["store_uuid"]}): its lineage does not include it')


def read_winding_solve(*, fibers_path, run_store_path, run_uuid, volume_uuid,
                       fit_scale, axis_xyz, spiral_outward_sense, z_window,
                       min_point_spacing, chunk_fibers=READ_CHUNK_FIBERS):
    """Read one winding solve (WindingSolve) for a fit.

    ``axis_xyz`` is the umbilicus as (N, 3) XYZ in ``volume_uuid``'s own voxels;
    the fibers are read in the same voxels (the store maps their declared
    frames), so their angles around it are the solve's. Kept points are
    converted to fit voxels (``fit_scale`` volume voxels each) and kept
    inside ``z_window`` = [z0, z1) of fit z.
    """
    from vc_object_store import CollectionReader, ObjectStore
    from object_store_inputs import ObjectFrame

    handedness = handedness_for(spiral_outward_sense)
    run_uuid = str(uuid.UUID(run_uuid))
    with CollectionReader(run_store_path) as run_reader:
        stage, settings = _run_settings(run_reader, run_uuid)
        if stage != WINDING_STAGE:
            raise WindingSolveError(f'run {run_uuid} is a {stage!r} run, not a winding run')
        recorded = settings.get('handedness')
        if recorded is None:
            raise WindingSolveError(f'run {run_uuid} records no handedness')
        if recorded != handedness:
            raise WindingSolveError(
                f'run {run_uuid} was solved with handedness {recorded}, but the scroll '
                f'({spiral_outward_sense}) has handedness {handedness:+d}')
        run_position = dict(run_reader.source)
        with CollectionReader(fibers_path) as fiber_reader:
            fiber_source = dict(fiber_reader.source)
        position = _consumed_position(run_reader, fiber_source, fibers_path)
        columns = run_reader.fiber_result_columns(run_uuid)
    values, valid = columns['values'], columns['valid'].astype(bool)
    index = {name: list(columns['columns']).index(name) for name in RESULT_COLUMNS}
    # Disabled rows (enabled = 0) are a legal result: their fibers are dropped.
    enabled = ~(valid[:, index['enabled']] & (values[:, index['enabled']] == 0))
    disabled_count = int((~enabled).sum())
    values, valid = values[enabled], valid[enabled]
    required = [index[name] for name in ('family', 'enabled', 'selected_winding', 'component')]
    fiber_uuid = np.ascontiguousarray(columns['fiber_uuid'][enabled])
    name = lambda i: str(uuid.UUID(bytes=fiber_uuid[i].tobytes()))  # noqa: E731
    unusable = ~valid[:, required].all(axis=1)
    unusable |= values[:, index['enabled']] != 1
    unusable |= ~np.isin(values[:, index['family']], (FAMILY_H, FAMILY_V))
    if unusable.any():
        raise WindingSolveError(
            f'run {run_uuid}: {int(unusable.sum())} fiber rows not disabled are not enabled '
            f'H/V fibers with a winding and a component (e.g. fiber {name(np.flatnonzero(unusable)[0])})')
    family = values[:, index['family']].astype(np.int8)
    selected_winding = values[:, index['selected_winding']].astype(np.int64)
    component = values[:, index['component']].astype(np.int64)
    cc_component, fiber_cc, cc_fiber_count = np.unique(
        component, return_inverse=True, return_counts=True)

    with ObjectStore.open(str(fibers_path), read_only=True, position=position) as store:
        store_fiber_count = int(store.object_count('fiber'))
    frame = ObjectFrame(volume_uuid, 1.0, 1.0 / fit_scale)
    z0, z1 = (float(value) for value in z_window)
    w_min = np.empty(len(fiber_uuid))
    w_max = np.empty(len(fiber_uuid))
    max_step, window_points = 0.0, 0
    kept_zyx, kept_w, kept_fiber, piece_lengths, piece_fiber = [], [], [], [], []
    with CollectionReader(fibers_path, position=position) as reader:
        for begin in range(0, len(fiber_uuid), chunk_fibers):
            end = min(begin + chunk_fibers, len(fiber_uuid))
            read = reader.geometry_array_columns(
                fiber_uuid[begin:end], 'line_points', object_kind='fiber',
                volume=volume_uuid, expected_position=position)
            if not np.array_equal(read['object_uuid'], fiber_uuid[begin:end]):
                raise WindingSolveError('the fiber store returned fibers out of the run\'s order')
            geometry = read['geometry']
            offsets = np.asarray(geometry.offsets, dtype=np.int64)
            xyz = geometry.values
            del read, geometry
            counts = np.diff(offsets)
            first = offsets[:-1]
            w = packed_winding_coordinate(
                offsets, xyz, selected_winding[begin:end], axis_xyz, handedness,
                name=lambda i: name(begin + i))
            w_min[begin:end] = np.minimum.reduceat(w, first)
            w_max[begin:end] = np.maximum.reduceat(w, first)
            steps = np.abs(np.diff(w))
            steps[first[1:] - 1] = 0.0
            max_step = max(max_step, float(steps.max(initial=0.0)))
            zyx = frame.xyz_to_fit_zyx(xyz)
            del xyz
            inside = (zyx[:, 0] >= z0) & (zyx[:, 0] < z1)
            window_points += int(inside.sum())
            starts = np.zeros(len(inside), dtype=bool)
            starts[first] = True
            starts |= ~np.concatenate([[False], inside[:-1]])
            starts &= inside
            points = np.flatnonzero(inside)
            owner = np.repeat(np.arange(begin, end), counts)[points]
            offsets = np.append(np.flatnonzero(starts[points]), len(points))
            keep = decimate_pieces(zyx[points], offsets, min_point_spacing)
            piece = np.searchsorted(offsets, keep, side='right') - 1
            points = points[keep]
            kept_zyx.append(zyx[points])
            kept_w.append(w[points])
            kept_fiber.append(owner[keep])
            pieces, lengths = np.unique(piece, return_counts=True)
            piece_lengths.append(lengths)
            piece_fiber.append(owner[keep][np.searchsorted(piece, pieces)])
            del zyx, w

    def joined(parts, dtype, shape=(0,)):
        return np.concatenate(parts).astype(dtype) if parts else np.empty(shape, dtype)

    return WindingSolve(
        fibers_path=str(fibers_path), run_store_path=str(run_store_path), run_uuid=run_uuid,
        fiber_store_position=position, run_store_position=run_position,
        handedness=handedness, store_fiber_count=store_fiber_count,
        disabled_count=disabled_count,
        fiber_uuid=fiber_uuid, family=family, selected_winding=selected_winding,
        component=component, fiber_cc=fiber_cc.astype(np.int64),
        w_min=w_min, w_max=w_max, cc_component=cc_component.astype(np.int64),
        cc_fiber_count=cc_fiber_count.astype(np.int64),
        largest_cc=int(np.argmax(cc_fiber_count)) if len(cc_fiber_count) else -1,
        z_window=(z0, z1), min_point_spacing=float(min_point_spacing),
        window_point_count=window_points, max_step_turns=max_step,
        zyx=joined(kept_zyx, np.float32, (0, 3)), w=joined(kept_w, np.float64),
        point_fiber=joined(kept_fiber, np.int64),
        piece_offsets=np.concatenate([[0], np.cumsum(joined(piece_lengths, np.int64))]),
        piece_fiber=joined(piece_fiber, np.int64))
