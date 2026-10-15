import logging
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable, Optional
from pathlib import Path

import build123d as bd
import meshlib.mrmeshnumpy as mn
import meshlib.mrmeshpy as mm
import numpy as np
import trimesh

from . import alignment
from . import curved_divider_wall

logger = logging.getLogger(__name__)


def show_meshlib(*meshes: mm.Mesh, show_axis: bool = True):
    trimesh_meshes = []
    for mesh in meshes:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "mesh.stl"
            mm.saveMesh(mesh, temp_path)
            tri_mesh = trimesh.load(temp_path)
            trimesh_meshes.append(tri_mesh)
    axis = trimesh.creation.axis(origin_size=10)
    if show_axis:
        trimesh_meshes.append(axis)
    scene = trimesh.Scene(trimesh_meshes)
    return scene.show(resolution=(1200, 600))


def count_vertices(mesh: mm.Mesh):
    return mn.getNumpyVerts(mesh).shape[0]


def _copy_meshlib(mesh: mm.Mesh):
    copy = mm.Mesh()
    copy.addPart(mesh)
    return copy


def load_mesh(mesh_file: str) -> mm.Mesh:
    return mm.loadMesh(Path(mesh_file))


def affine_rotation(angle_rad: float) -> mm.AffineXf3f:
    """Affine rotation about Z axis."""
    rotation_mat = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad), 0],
            [-np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )
    rotation_mat_mm = mm.AffineXf3f(
        mm.Matrix3f(*[mm.Vector3f(*d) for d in rotation_mat]),
        mm.Vector3f(0, 0, 0),
    )
    return rotation_mat_mm


def mesh_smooth_denoise(mesh: mm.Mesh, gamma: float = 20) -> mm.Mesh:
    """Smooth a mesh with a denoise operation.

    Denoises a mesh per the paper 'Mesh Denoising via a Novel Mumford-Shah Framework'.

    Args:
        gamma: Smoothing amount.
    """
    logger.info(f"Smoothing mesh with denoise, gamma = {gamma}")
    mesh_smooth = _copy_meshlib(mesh)
    settings = mm.DenoiseViaNormalsSettings()
    settings.gamma = gamma
    mm.meshDenoiseViaNormals(mesh_smooth)
    return mesh_smooth


def mesh_smooth_shrink_expand(
    mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.4,
    amount_mm: float = 2,
    shrink_first: bool = True,
):
    """Smooth a mesh with a shrink/expand operation.

    Shrinking first smooths convexities, expanding first smooths concavities.
    """
    voxel_size = (
        mesh.computeBoundingBox().diagonal() * voxel_size_diagonal_percent / 100
    )

    mesh_smooth = _copy_meshlib(mesh)
    logger.info(f"Smoothing mesh with a shrink/expand offset of {amount_mm} mm")
    params = mm.OffsetParameters()
    params.voxelSize = voxel_size
    mesh_smooth = mm.offsetMesh(
        mesh_smooth,
        offset=-amount_mm if shrink_first else amount_mm,
        params=params,  # type: ignore
    )
    mesh_smooth = mm.offsetMesh(
        mesh_smooth,
        offset=amount_mm if shrink_first else -amount_mm,
        params=params,  # type: ignore
    )
    return mesh_smooth


def get_principal_components(mesh: mm.Mesh) -> np.ndarray:
    """Get principal components of mesh as a 3x3 array."""
    vertices = mn.getNumpyVerts(mesh)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    cov_matrix = np.cov(centered_vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices].T
    return principal_components


def rotate_about_2nd_principal(mesh: mm.Mesh, rotation_rad: float) -> mm.Mesh:
    """Rotate about second principal.

    Rotate the mesh about the cylinder (Z) axis so that the wide direction of its
    cross-section, the principal component of the vertices projected onto the XY
    plane, lies along X (the split between the case halves), then by a further
    rotation_rad.
    """
    vertices_xy = mn.getNumpyVerts(mesh)[:, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(vertices_xy, rowvar=False))
    wide = eigenvectors[:, np.argmax(eigenvalues)]
    rotated_mesh = _copy_meshlib(mesh)

    # affine_rotation(a) rotates by -a, so this takes the wide direction onto X.
    first_rot = np.arctan2(wide[1], wide[0])
    rotated_mesh.transform(affine_rotation(first_rot + rotation_rad))
    return rotated_mesh


@dataclass
class ScrollMesh:
    """Scroll mesh processing options.

    Attributes:
        mesh_path: Path to scroll mesh.
        rotation_from_2nd_principal_rad. Rotation from the second principal vector
        about the cylinder axis. Defaults to 0.
        voxel_size_diagonal_percent: Size of voxels as a percentage of the diagonal
        size. Defaults to 0.4.
        simplify_max_error_diagonal_percent: Maximum simplification error as a
        percentage of the diagonal size. Defaults to 1.
        target_scale_diagonal_mm: Target diagonal size of the mesh. Defaults to None.
        rotation_callback: Rotation callback that accepts a mesh and returns a mesh
        smoothing_callback: Smoothing callback that accepts a mesh and returns a mesh
        smoothing_unite_with_original: Unite with the original after smoothing to ensure
        mesh does not get smaller.
        cylinder_axis: Cylinder axis direction in input mesh coordinates, e.g. along
        the umbilicus. Defaults to None, which fits the axis giving the smallest
        enclosing cylinder.
        parting_z_margin_mm: How far beyond the lining, along the axis, the parting
        surface between the halves may leave the S-shaped divider; no more than the
        case's upper and lower margins, where the caps (split on the divider) begin.
        Defaults to 2.5.
        parting_max_slope_deg: Steepest the parting surface may be in returning to the
        divider at those ends, so the halves do not slide along each other there.
        Defaults to 50.
        rotation_search: After rotation_callback, turn the mesh about the cylinder axis
        to where the split leaves the scroll least room (rotation_minimising_fill), so
        the parting surface bends least. Defaults to True.
        rotation_search_depth_tolerance_mm: Only consider rotations whose pocket depth
        is within this of the shallowest. Defaults to 5.
        rotation_search_min_gain: Keep the rotation_callback's split unless the search
        cuts the room by at least this fraction of the lined scroll's volume. Defaults
        to 0.01.

    """

    mesh_path: str
    cylinder_axis_tol: float = 0.01
    voxel_size_diagonal_percent: float = 0.4
    simplify_max_error_diagonal_percent: float = 2
    lining_offset_mm: float = 2
    wall_thickness_mm: float = 2
    target_scale_diagonal_mm: Optional[float] = None
    rotation_callback: Optional[Callable[[mm.Mesh], mm.Mesh]] = partial(
        rotate_about_2nd_principal, rotation_rad=0
    )
    smoothing_callback: Optional[Callable[[mm.Mesh], mm.Mesh]] = None
    smoothing_unite_with_original: bool = True
    vertical_offset: Optional[float] = None
    cylinder_axis: Optional[tuple[float, float, float]] = None
    parting_z_margin_mm: float = 2.5
    parting_max_slope_deg: float = 50
    rotation_search: bool = True
    rotation_search_depth_tolerance_mm: float = 5
    rotation_search_min_gain: float = 0.01


def _lined_grid(
    mesh: mm.Mesh, lining_offset_mm: float, pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    """Voxels of a pitch-sized grid within lining_offset_mm of the mesh's interior, as a
    boolean array indexed [x, y, z], and the position of voxel [0, 0, 0]'s centre."""
    box = mesh.computeBoundingBox()
    pad = lining_offset_mm + pitch
    params = mm.MeshToDistanceVolumeParams()
    params.vol.origin = mm.Vector3f(box.min.x - pad, box.min.y - pad, box.min.z - pad)
    params.vol.voxelSize = mm.Vector3f(pitch, pitch, pitch)
    size = box.size()
    params.vol.dimensions = mm.Vector3i(
        *[int(np.ceil((s + 2 * pad) / pitch)) + 1 for s in (size.x, size.y, size.z)]
    )
    params.dist.signMode = mm.SignDetectionMode.HoleWindingRule
    volume = mn.getNumpy3Darray(mm.meshToDistanceVolume(mm.MeshPart(mesh), params))
    origin = np.array([params.vol.origin.x, params.vol.origin.y, params.vol.origin.z])
    return volume < lining_offset_mm, origin + pitch / 2


def split_fill_and_depth(
    mesh: mm.Mesh,
    angle_rad: float,
    lining_outer_radius: float,
    lining_offset_mm: float,
    wall_thickness_mm: float,
    pitch: float,
) -> tuple[float, float, float]:
    """Fill and pocket depth of the case split, with the mesh rotated by angle_rad
    (anticlockwise) about Z, and the lined mesh's volume, all from a pitch-sized grid.

    Each half's cavity is pullable along Y, so in every (x, z) column it spans from the
    divider to the lined mesh's farthest voxel on that side. The fill, in mm^3, is the
    cavity beyond the lined mesh: the room the scroll has to move in, before the
    protrusions that take it up. The depth is the farthest lined voxel from the divider
    over both halves.
    """
    rotated = _copy_meshlib(mesh)
    # affine_rotation(a) rotates by -a.
    rotated.transform(affine_rotation(-angle_rad))
    lined, centre = _lined_grid(rotated, lining_offset_mm, pitch)
    x = centre[0] + pitch * np.arange(lined.shape[0])
    y = centre[1] + pitch * np.arange(lined.shape[1])
    # Distance of each voxel from the divider, along Y, for each column [x, :, z].
    beyond = (
        y[None, :] - curved_divider_wall.divider_y(x, lining_outer_radius, wall_thickness_mm)[:, None]
    )[:, :, None]
    fill, depth = 0.0, 0.0
    for side in (lined & (beyond > 0), lined & (beyond <= 0)):
        farthest = np.where(side, np.abs(beyond), 0).max(axis=1)
        count = side.sum(axis=1)
        fill += ((farthest / pitch + 0.5)[count > 0] - count[count > 0]).clip(min=0).sum() * pitch**3
        depth = max(depth, farthest.max() + pitch / 2)
    return fill, depth, lined.sum() * pitch**3


def rotation_minimising_fill(
    mesh: mm.Mesh,
    lining_outer_radius: float,
    lining_offset_mm: float,
    wall_thickness_mm: float,
    depth_tolerance_mm: float,
    min_gain: float,
    pitch: float = 0.5,
) -> float:
    """Rotation about Z (radians, anticlockwise) giving the least fill (see
    split_fill_and_depth) among those whose pocket depth is within depth_tolerance_mm
    of the shallowest, searched every 10 degrees and then every degree around the best.
    Zero unless that cuts the fill by at least min_gain times the lined mesh's volume.

    The divider is symmetric under a half turn, so angles in (-90, 90] degrees cover it;
    beyond those, a half turn would only swap which half holds which side of the scroll.
    """
    evaluated = {}

    def evaluate(degrees):
        for d in degrees:
            d = (d + 89) % 180 - 89
            if d not in evaluated:
                evaluated[d] = split_fill_and_depth(
                    mesh,
                    np.radians(d),
                    lining_outer_radius,
                    lining_offset_mm,
                    wall_thickness_mm,
                    pitch,
                )

    def best():
        shallowest = min(depth for _, depth, _ in evaluated.values())
        ok = [d for d, (_, depth, _) in evaluated.items() if depth <= shallowest + depth_tolerance_mm]
        return min(ok, key=lambda d: evaluated[d][0])

    evaluate(range(-80, 91, 10))
    evaluate(range(best() - 5, best() + 6))
    pick = best()
    fill_0, depth_0, lined_volume = evaluated[0]
    fill, depth, _ = evaluated[pick]
    logger.info(
        f"Split fill {fill_0 / 1000:.2f} cm3 (depth {depth_0:.1f} mm) at the current "
        f"rotation, {fill / 1000:.2f} cm3 (depth {depth:.1f} mm) at {pick} deg"
    )
    if fill_0 - fill < min_gain * lined_volume:
        return 0.0
    return float(np.radians(pick))


def _parting_surface(
    mesh: mm.Mesh,
    lining_offset_mm: float,
    lining_outer_radius: float,
    wall_thickness_mm: float,
    z_margin_mm: float,
    max_slope_deg: float,
    pitch: float,
    iterations: int = 20000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height (y) of the surface parting the case halves over a grid of (x, z).

    Across the lining in x, and to z_margin_mm beyond it along the axis, the surface
    may leave the S-shaped divider. In every column the lined scroll occupies, it
    passes through the scroll's inner half (at least a quarter of its extent in from its
    lowest and highest points), as near the divider as that allows, smoothed over
    neighbouring columns: so each half meets the scroll on its side however the scroll
    lies, and where the scroll lies off the divider the surface leaves it at its edge,
    where the inner half narrows to a line, as a mould's parting line follows an
    object's outline, rather than folding back along one face of it. Elsewhere it
    departs from the divider as smoothly as it can: the departure is harmonic, zero at
    the region's ends along the axis (where the caps, split on the divider, begin) and
    free at its sides, found by red-black successive over-relaxation. Then it is made no steeper than
    max_slope_deg anywhere, so the halves do not slide along each other as they part:
    where the scroll would need it steeper (mostly towards the ends along the axis), it
    lies wholly in one half for a little way.

    Returns the grid's x and z coordinates and the heights, indexed [x, z].
    """
    lined, centre = _lined_grid(mesh, lining_offset_mm, pitch)
    occupied = lined.any(axis=1)
    first = lined.argmax(axis=1)
    last = lined.shape[1] - 1 - lined[:, ::-1, :].argmax(axis=1)
    lowest = centre[1] + pitch * (first - 0.5)
    highest = centre[1] + pitch * (last + 0.5)

    x_end = lining_outer_radius
    box = mesh.computeBoundingBox()
    z_reach = lining_offset_mm + wall_thickness_mm + z_margin_mm
    z_lo, z_hi = box.min.z - z_reach, box.max.z + z_reach
    # Nodes on the lined grid's columns, between nodes on the region's edges.
    kx = np.arange(np.floor((-x_end - centre[0]) / pitch) + 1, np.ceil((x_end - centre[0]) / pitch))
    kz = np.arange(np.floor((z_lo - centre[2]) / pitch) + 1, np.ceil((z_hi - centre[2]) / pitch))
    xs = np.concatenate([[-x_end], centre[0] + pitch * kx, [x_end]])
    zs = np.concatenate([[z_lo], centre[2] + pitch * kz, [z_hi]])
    ix = np.concatenate([[-1], kx, [-1]]).astype(int)
    iz = np.concatenate([[-1], kz, [-1]]).astype(int)
    inside = (ix[:, None] >= 0) & (ix[:, None] < lined.shape[0]) & (iz[None, :] >= 0) & (iz[None, :] < lined.shape[2])
    jx, jz = np.clip(ix, 0, lined.shape[0] - 1), np.clip(iz, 0, lined.shape[2] - 1)
    bound = inside & occupied[jx][:, jz]
    bound[:, [0, -1]] = False
    lo, hi = lowest[jx][:, jz], highest[jx][:, jz]

    divider = curved_divider_wall.divider_y(xs, lining_outer_radius, wall_thickness_mm)[:, None]
    # Within the scroll's inner half, smoothed over a millimetre or so of neighbouring
    # columns (its extent at the outline is noisy), and kept within it.
    quarter = (hi - lo) / 4
    middle = np.where(bound, np.clip(divider, lo + quarter, hi - quarter), 0)
    for _ in range(int(np.ceil(1 / pitch)) * 2):
        total = np.zeros_like(middle)
        count = np.zeros_like(middle)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                total += np.roll(np.roll(np.where(bound, middle, 0), di, 0), dj, 1)
                count += np.roll(np.roll(bound, di, 0), dj, 1)
        middle = np.where(bound, total / np.maximum(count, 1), 0)
    middle = np.clip(middle, lo, hi)

    departure = np.where(bound, middle - divider, 0)
    free = ~bound
    free[:, [0, -1]] = False

    n = max(len(xs), len(zs))
    omega = 2 / (1 + np.sin(np.pi / n))
    parity = (np.arange(len(xs))[:, None] + np.arange(len(zs))[None, :]) % 2
    d = departure
    for _ in range(iterations):
        change = 0.0
        for colour in (0, 1):
            interior = d[1:-1, 1:-1]
            mean = 0.25 * (d[:-2, 1:-1] + d[2:, 1:-1] + d[1:-1, :-2] + d[1:-1, 2:])
            moving = (parity[1:-1, 1:-1] == colour) & free[1:-1, 1:-1]
            step = np.where(moving, omega * (mean - interior), 0)
            change = max(change, np.abs(step).max())
            d[1:-1, 1:-1] = interior + step
            d[0, 1:-1] = np.where(free[0, 1:-1], d[1, 1:-1], d[0, 1:-1])
            d[-1, 1:-1] = np.where(free[-1, 1:-1], d[-2, 1:-1], d[-1, 1:-1])
        if change < 1e-4:
            break

    # Smoothed across the scroll's outline, where the pinned and free parts meet at a
    # kink, over a millimetre or so, kept within the scroll and on the divider at the
    # region's ends.
    for _ in range(int(np.ceil(1 / pitch)) * 2):
        padded = np.pad(d, 1, mode="edge")
        d = sum(padded[1 + di : 1 + di + d.shape[0], 1 + dj : 1 + dj + d.shape[1]] for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9
        d = np.where(bound, np.clip(d, lo - divider, hi - divider), d)
        d[:, [0, -1]] = 0

    # No steeper than max_slope_deg between neighbours: alternately limit each point's
    # departure to its neighbours' plus the rise that slope allows, along each axis,
    # which only ever brings the surface back towards the divider.
    slope = np.tan(np.radians(max_slope_deg))
    rise_x = slope * np.diff(xs)[:, None]
    rise_z = slope * np.diff(zs)[None, :]

    def limit(target, neighbour, rise):
        return np.sign(target) * np.minimum(np.abs(target), np.abs(neighbour) + rise)

    for _ in range(iterations):
        before = d.copy()
        for j in range(1, d.shape[1]):
            d[:, j] = limit(d[:, j], d[:, j - 1], rise_z[:, j - 1])
        for j in range(d.shape[1] - 2, -1, -1):
            d[:, j] = limit(d[:, j], d[:, j + 1], rise_z[:, j])
        for i in range(1, d.shape[0]):
            d[i] = limit(d[i], d[i - 1], rise_x[i - 1])
        for i in range(d.shape[0] - 2, -1, -1):
            d[i] = limit(d[i], d[i + 1], rise_x[i])
        if np.abs(d - before).max() < 1e-6:
            break
    return xs, zs, divider + d


def _heightfield_solid(
    xs: np.ndarray, zs: np.ndarray, bottom: np.ndarray, top: np.ndarray
) -> mm.Mesh:
    """Closed mesh of the region bottom < y < top over the grid xs x zs, the heights
    indexed [x, z]."""
    nx, nz = len(xs), len(zs)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    verts = np.concatenate(
        [np.stack([X, bottom, Z], -1).reshape(-1, 3), np.stack([X, top, Z], -1).reshape(-1, 3)]
    )
    n = nx * nz
    idx = np.arange(n).reshape(nx, nz)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = [
        np.stack([a, b, c], 1), np.stack([a, c, d], 1),  # bottom, facing -y
        np.stack([a, c, b], 1) + n, np.stack([a, d, c], 1) + n,  # top, facing +y
    ]

    def wall(edge, outward_sign):
        p, q = edge[:-1], edge[1:]
        tris = [np.stack([p, q, q + n], 1), np.stack([p, q + n, p + n], 1)]
        return [t[:, ::-1] for t in tris] if outward_sign < 0 else tris

    faces += wall(idx[:, 0], -1) + wall(idx[:, -1], 1) + wall(idx[0, :], 1) + wall(idx[-1, :], -1)
    solid = mn.meshFromFacesVerts(np.concatenate(faces).astype(np.int32), verts.astype(np.float32))
    if solid.volume() < 0:
        raise RuntimeError("Heightfield solid came out inside out")
    return solid


def _exact_boolean(a: mm.Mesh, b: mm.Mesh, op: mm.BooleanOperation, voxel_size: float):
    """Exact boolean, falling back to a voxel boolean where it fails (as it can on
    self-intersecting voxel-derived meshes)."""
    result = mm.boolean(a, b, op)
    if result.valid():
        return result.mesh
    logger.warning(f"Exact boolean failed ({result.errorString}); using voxels")
    voxel_op = {
        mm.BooleanOperation.Union: mm.voxelBooleanUnite,
        mm.BooleanOperation.DifferenceAB: mm.voxelBooleanSubtract,
    }[op]
    return voxel_op(a, b, voxel_size)


def assert_one_component(mesh: mm.Mesh):
    components = mm.getAllComponents(mesh)
    assert len(components) == 1


def build_lining(
    mesh_params: ScrollMesh,
    *,
    _debug=False,
) -> tuple[mm.Mesh, mm.Mesh, mm.Mesh, mm.Mesh, mm.Mesh, float, float]:
    """Build lining given a scroll mesh.

    Args:
        mesh_params: ScrollMesh parameters.

    Returns:
        Lining meshes for the positive and negative side, cavity meshes for the
        positive and negative side, aligned scroll mesh, cylinder radius, and height.
    """
    # TODO(akoen): support different upper and lower margin
    # NOTE(akoen): This method seems to work for all meshes. However, some things that
    # may someday cause trouble: tunnels, holes, etc.
    # Look at Meshlib GitHub "heal", "tunnel"

    mesh_scroll = load_mesh(mesh_params.mesh_path)

    # Get the largest connected component
    assert_one_component(mesh_scroll)

    # Optionally scale mesh
    if mesh_params.target_scale_diagonal_mm:
        scale = (
            mesh_params.target_scale_diagonal_mm
            / mesh_scroll.computeBoundingBox().diagonal()
        )

        logger.info("Scaling mesh by factor %.2f", scale)
        scale_mat = mm.AffineXf3f.linear(mm.Matrix3f.scale(scale))
        mesh_scroll.transform(scale_mat)
    elif mesh_scroll.computeBoundingBox().diagonal() < 5:
        logger.warning("Mesh is very small. Have you forgotten to scale?")

    voxel_size = (
        mesh_scroll.computeBoundingBox().diagonal()
        * mesh_params.voxel_size_diagonal_percent
        / 100
    )

    logger.debug(f"Initial mesh: {count_vertices(mesh_scroll)} vertices")

    # Simplify mesh
    decimate_settings = mm.DecimateSettings()
    decimate_settings.maxError = mesh_params.simplify_max_error_diagonal_percent / 100
    decimate_settings.packMesh = True
    mm.decimateMesh(mesh_scroll, decimate_settings)

    logger.debug(f"Simplified mesh: {count_vertices(mesh_scroll)} vertices")

    # Smooth mesh
    if (sc := mesh_params.smoothing_callback) is not None:
        logger.info("Smoothing mesh")
        mesh_scroll_smoothed = sc(mesh_scroll)
        if mesh_params.smoothing_unite_with_original:
            logger.info("Uniting smoothed mesh with original")
            mesh_scroll = mm.voxelBooleanUnite(
                mesh_scroll, mesh_scroll_smoothed, voxel_size
            )
        else:
            mesh_scroll = mesh_scroll_smoothed

    # Fit minimum bounding cylinder
    logger.info("Aligning mesh")
    points = np.array([(point.x, point.y, point.z) for point in mesh_scroll.points])
    rotation_mat, translation_mat, radius, height = alignment.fit_cylinder(
        points,
        mesh_params.cylinder_axis_tol,
        _debug=_debug,
        axis=mesh_params.cylinder_axis,
    )

    A = mm.Matrix3f(*[mm.Vector3f(*d) for d in rotation_mat])
    mesh_scroll.transform(mm.AffineXf3f(A, mm.Vector3f(*translation_mat)))

    # Optimize Rotation
    if mesh_params.rotation_callback is not None:
        mesh_scroll = mesh_params.rotation_callback(mesh_scroll)

    # Turn the split to where the case leaves the scroll least room
    if mesh_params.rotation_search:
        angle = rotation_minimising_fill(
            mesh_scroll,
            radius + mesh_params.lining_offset_mm + mesh_params.wall_thickness_mm,
            mesh_params.lining_offset_mm,
            mesh_params.wall_thickness_mm,
            mesh_params.rotation_search_depth_tolerance_mm,
            mesh_params.rotation_search_min_gain,
        )
        if angle:
            logger.info(f"Rotating mesh by {np.degrees(angle):.0f} deg about the axis")
            # affine_rotation(a) rotates by -a.
            mesh_scroll.transform(affine_rotation(-angle))

    # Optionally translate mesh
    if mesh_params.vertical_offset is not None:
        logger.info("Translating mesh by %.2f", mesh_params.vertical_offset)
        translate_mat = mm.AffineXf3f.translation(
            mm.Vector3f(0, 0, mesh_params.vertical_offset)
        )
        mesh_scroll.transform(translate_mat)

    # Offset
    logger.info("Offsetting mesh")
    params = mm.OffsetParameters()
    params.voxelSize = voxel_size
    offset_mesh = mm.offsetMesh(mesh_scroll, offset=mesh_params.lining_offset_mm, params=params)  # type: ignore

    logger.debug(f"Offset mesh: {count_vertices(offset_mesh)} vertices")

    # Split in two
    logger.info("Splitting mesh")
    hole_edges_pos = mm.UndirectedEdgeBitSet()
    hole_edges_neg = mm.UndirectedEdgeBitSet()

    # The region of the positive half, bounded by the parting surface: the S-shaped
    # divider (as build_divider_solid, which splits the caps) away from the scroll,
    # bent to pass through it.
    lining_outer_radius = radius + mesh_params.lining_offset_mm + mesh_params.wall_thickness_mm
    xs, zs, parting = _parting_surface(
        mesh_scroll,
        mesh_params.lining_offset_mm,
        lining_outer_radius,
        mesh_params.wall_thickness_mm,
        mesh_params.parting_z_margin_mm,
        mesh_params.parting_max_slope_deg,
        min(voxel_size, 0.5),
    )
    extent = 112.5  # build_divider_solid's reach for build_lining's case_max_radius
    big_xs = np.concatenate([[-extent], xs, [extent]])
    big_zs = np.concatenate([[-100], zs, [900]])
    divider = curved_divider_wall.divider_y(big_xs, lining_outer_radius, mesh_params.wall_thickness_mm)
    big_parting = np.repeat(divider[:, None], len(big_zs), axis=1)
    big_parting[1:-1, 1:-1] = parting
    divider_piece_mesh = _heightfield_solid(
        big_xs, big_zs, big_parting, np.full_like(big_parting, extent)
    )

    # Create positive mesh
    split_mesh_pos = _copy_meshlib(offset_mesh)
    # mm.trimWithPlane(split_mesh_pos, cut_plane, outCutEdges=hole_edges_pos)
    split_mesh_pos = mm.boolean(
        split_mesh_pos, divider_piece_mesh, mm.BooleanOperation.Intersection
    ).mesh

    # Create negative mesh
    split_mesh_neg = _copy_meshlib(offset_mesh)
    # mm.trimWithPlane(split_mesh_neg, -cut_plane, outCutEdges=hole_edges_neg)
    split_mesh_neg = mm.boolean(
        split_mesh_neg, divider_piece_mesh, mm.BooleanOperation.DifferenceAB
    ).mesh

    logger.debug(f"Split mesh pos: {count_vertices(split_mesh_pos)} vertices")

    # Remove overhangs
    logger.info("Removing overhangs")
    cavity_mesh_pos = _copy_meshlib(split_mesh_pos)
    cavity_mesh_neg = _copy_meshlib(split_mesh_neg)
    extrusion_dir = mm.Vector3f(0, 1, 0)
    extrusion_distance = 0
    mm.fixUndercuts(cavity_mesh_pos, extrusion_dir, voxel_size, extrusion_distance)
    mm.fixUndercuts(cavity_mesh_neg, -extrusion_dir, voxel_size, extrusion_distance)

    logger.debug(f"Cavity_mesh_pos: {count_vertices(cavity_mesh_pos)} vertices")

    # Build mesh
    logger.info("Building lining")

    # Each half's lining and divider wall are one shell, wall_thickness_mm thick, around
    # its cavity and a thin sheet on the parting surface, cut along the surface: the
    # wall follows the surface at the lining's thickness, and meets the lining without a
    # seam. The wall replaces the case's wall on the S-shaped divider, and the posts at
    # its ends there, which are removed with the cavities; the sheet stops the wall
    # thickness short of the lining's outer radius, as those posts did, and reaches a
    # millimetre beyond the surface's grid along the axis, where the surface is back on
    # the divider, to fuse with the case's wall beyond.
    w = mesh_params.wall_thickness_mm
    sheet_xs = xs[np.abs(xs) <= lining_outer_radius - w]
    sheet_zs = np.concatenate([[zs[0] - 1], zs, [zs[-1] + 1]])
    sheet_parting = np.concatenate([parting[:, :1], parting, parting[:, -1:]], axis=1)[np.abs(xs) <= lining_outer_radius - w]
    half_thickness = max(voxel_size, 0.25)
    sheet = _heightfield_solid(
        sheet_xs, sheet_zs, sheet_parting - half_thickness, sheet_parting + half_thickness
    )
    cavity_mesh_pos_offset = mm.offsetMesh(mm.voxelBooleanUnite(cavity_mesh_pos, sheet, voxel_size), offset=w, params=params)  # type: ignore
    cavity_mesh_neg_offset = mm.offsetMesh(mm.voxelBooleanUnite(cavity_mesh_neg, sheet, voxel_size), offset=w, params=params)  # type: ignore
    # Generously: over the lining the case has nothing else near the divider. A
    # millimetre wider than the parting surface's grid, past the end posts' outer edge,
    # and a wall thickness more than the old wall each way, past its swept ends.
    flat_xs = np.concatenate([[xs[0] - 1], xs, [xs[-1] + 1]])
    flat = np.repeat(
        curved_divider_wall.divider_y(flat_xs, lining_outer_radius, w)[:, None], len(zs), axis=1
    )
    flat_thickness = w * np.sqrt(1 + np.gradient(flat, flat_xs, axis=0) ** 2) + w
    old_wall_pos = _heightfield_solid(flat_xs, zs, flat - w, flat + flat_thickness)
    old_wall_neg = _heightfield_solid(flat_xs, zs, flat - flat_thickness, flat + w)

    # mm.trimWithPlane(cavity_mesh_pos_offset, cut_plane, outCutEdges=hole_edges_pos)
    # mm.trimWithPlane(cavity_mesh_neg_offset, -cut_plane, outCutEdges=hole_edges_neg)
    cavity_mesh_pos_offset = mm.boolean(
        cavity_mesh_pos_offset, divider_piece_mesh, mm.BooleanOperation.Intersection
    ).mesh
    cavity_mesh_neg_offset = mm.boolean(
        cavity_mesh_neg_offset, divider_piece_mesh, mm.BooleanOperation.DifferenceAB
    ).mesh

    fill_hole_params = mm.FillHoleParams()
    edges_pos = mm.std_vector_Id_EdgeTag(hole_edges_pos)
    edges_neg = mm.std_vector_Id_EdgeTag(hole_edges_neg)
    mm.fillHoles(cavity_mesh_pos_offset, edges_pos, fill_hole_params)
    mm.fillHoles(cavity_mesh_neg_offset, edges_neg, fill_hole_params)

    lining_mesh_pos = mm.voxelBooleanSubtract(
        cavity_mesh_pos_offset, cavity_mesh_pos, voxel_size
    )
    lining_mesh_neg = mm.voxelBooleanSubtract(
        cavity_mesh_neg_offset, cavity_mesh_neg, voxel_size
    )

    unite = partial(_exact_boolean, op=mm.BooleanOperation.Union, voxel_size=voxel_size)
    cavity_mesh_pos = unite(cavity_mesh_pos, old_wall_pos)
    cavity_mesh_neg = unite(cavity_mesh_neg, old_wall_neg)

    logger.debug(f"Lining mesh pos: {count_vertices(lining_mesh_pos)} vertices")

    return (
        lining_mesh_pos,
        lining_mesh_neg,
        cavity_mesh_pos,
        cavity_mesh_neg,
        mesh_scroll,
        radius,
        height,
    )


def combine_brep_case_lining(
    case: bd.Solid,
    cavity_mesh: mm.Mesh,
    lining_mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.01,
):
    """Combine a BRep case mesh with a lining."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "case.stl"
        bd.export_stl(case, str(temp_path))
        case_mesh = load_mesh(str(temp_path))
        return combine_case_lining(
            case_mesh, cavity_mesh, lining_mesh, voxel_size_diagonal_percent
        )


def brep_to_mesh(brep: bd.Solid) -> mm.Mesh:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "brep.stl"
        bd.export_stl(brep.solids()[0], str(temp_path))
        return load_mesh(str(temp_path))


def combine_case_lining(
    case_mesh: mm.Mesh,
    cavity_mesh: mm.Mesh,
    lining_mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.4,
):
    "Combine a case mesh with a lining."
    voxel_size = (
        lining_mesh.computeBoundingBox().diagonal() * voxel_size_diagonal_percent / 100
    )

    combined_mesh = mm.boolean(case_mesh, cavity_mesh, mm.BooleanOperation.DifferenceAB)
    combined_mesh = mm.boolean(
        combined_mesh.mesh, lining_mesh, mm.BooleanOperation.Union
    )
    return combined_mesh.mesh
