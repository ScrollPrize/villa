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
        protrusion_min_gap_mm: Where the scroll lies wholly on one side of the divider
        with a gap beyond this to it, the other half protrudes past the divider to meet
        the scroll. Defaults to 1.
        protrusion_clearance_mm: Clearance between such a protrusion and the lining of
        the half it slides into. Defaults to 1.
        rotation_search: After rotation_callback, turn the mesh about the cylinder axis
        to where the split leaves the scroll least room (rotation_minimising_fill), so
        the protrusions are smallest. Defaults to True.
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
    protrusion_min_gap_mm: float = 1
    protrusion_clearance_mm: float = 1
    rotation_search: bool = True
    rotation_search_depth_tolerance_mm: float = 5
    rotation_search_min_gain: float = 0.01


def _sweep(mesh: mm.Mesh, toward: mm.Vector3f, voxel_size: float, length: float):
    """Mesh swept along `toward` by `length` beyond its extent: each column along that
    axis is filled from the mesh's far side onwards."""
    swept = _copy_meshlib(mesh)
    mm.fixUndercuts(swept, -toward, voxel_size, length)
    return swept


def _translated(mesh: mm.Mesh, y: float):
    moved = _copy_meshlib(mesh)
    moved.transform(mm.AffineXf3f.translation(mm.Vector3f(0, y, 0)))
    return moved


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

    The divider is symmetric under a half turn, so angles in [0, 180) degrees cover it.
    """
    evaluated = {}

    def evaluate(degrees):
        for d in degrees:
            d %= 180
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

    evaluate(range(0, 180, 10))
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


def _protrusion(
    offset_mesh: mm.Mesh,
    divider_piece_mesh: mm.Mesh,
    *,
    below: bool,
    lining_outer_radius: float,
    wall_thickness_mm: float,
    min_gap_mm: float,
    clearance_mm: float,
    voxel_size: float,
) -> tuple[Optional[mm.Mesh], Optional[mm.Mesh]]:
    """Protrusion of one case half past the divider, to meet the scroll where it lies
    wholly on the other side.

    With below=True this is the negative half rising (along +Y) into the region between
    the divider and the underside of the offset scroll, in the columns (along Y) where
    that gap exceeds min_gap_mm; with below=False, the positive half descending onto
    the scroll's top. The protrusion reaches from wall_thickness_mm on its own side of
    the divider (joining its half's divider wall) to the offset scroll, and is kept
    clearance_mm (plus a voxel for discretisation) inside the scroll's footprint, so
    that it slides past the other half's lining around the scroll.

    Returns the protrusion and the region to clear from the other half, or (None, None)
    where the scroll crosses the divider (within min_gap_mm) in every column.
    """
    y = mm.Vector3f(0, 1, 0)
    reach = offset_mesh.computeBoundingBox().diagonal()
    down = _sweep(offset_mesh, -y, voxel_size, reach)
    up = _sweep(offset_mesh, y, voxel_size, reach)
    # The gap between the scroll and the divider (or beyond it) in each column, and its
    # part beyond the divider shifted by `shift` towards the scroll.
    if below:
        gap = mm.voxelBooleanSubtract(down, up, voxel_size)
    else:
        gap = mm.voxelBooleanSubtract(up, down, voxel_size)
    # Within the lining; the divider piece, cropped to a little beyond that so that it
    # still covers it when shifted, ends a fixed distance past the divider, which the
    # sweeps may otherwise reach. Booleans between voxel-derived meshes are voxel based,
    # being robust to their self-intersections.
    bounds = offset_mesh.computeBoundingBox()
    lo = mm.Vector3f(bounds.min.x - 1, -lining_outer_radius, bounds.min.z - 1)
    size = mm.Vector3f(bounds.size().x + 2, 2 * lining_outer_radius, bounds.size().z + 2)
    gap = mm.voxelBooleanIntersect(gap, mm.makeCube(size, lo), voxel_size)
    margin = 2 * (wall_thickness_mm + min_gap_mm)
    crop = mm.boolean(
        divider_piece_mesh,
        mm.makeCube(
            mm.Vector3f(size.x, size.y + 2 * margin, size.z),
            mm.Vector3f(lo.x, lo.y - margin, lo.z),
        ),
        mm.BooleanOperation.Intersection,
    )
    if not crop.valid():
        raise RuntimeError(f"Cropping the divider failed: {crop.errorString}")
    combine = mm.voxelBooleanIntersect if below else mm.voxelBooleanSubtract
    sign = 1 if below else -1
    inside_far = lambda shift: combine(
        gap, _translated(crop.mesh, sign * shift), voxel_size
    )

    # Columns where the gap beyond the divider exceeds min_gap_mm, opened to drop
    # features narrower than twice the clearance.
    deep = inside_far(min_gap_mm)
    if deep.volume() < voxel_size**3:
        return None, None
    columns = _sweep(_sweep(deep, y, voxel_size, reach), -y, voxel_size, reach)
    params = mm.OffsetParameters()
    params.voxelSize = voxel_size
    columns = mm.offsetMesh(columns, offset=-clearance_mm, params=params)  # type: ignore
    if columns.volume() < voxel_size**3:
        return None, None
    columns = mm.offsetMesh(columns, offset=clearance_mm, params=params)  # type: ignore

    # The scroll's footprint, shrunk by the sliding clearance.
    footprint = mm.voxelBooleanUnite(down, up, voxel_size)
    footprint = mm.offsetMesh(footprint, offset=-(clearance_mm + voxel_size), params=params)  # type: ignore

    mesa = inside_far(-wall_thickness_mm)
    mesa = mm.voxelBooleanIntersect(mesa, columns, voxel_size)
    mesa = mm.voxelBooleanIntersect(mesa, footprint, voxel_size)
    if mesa.volume() < voxel_size**3:
        return None, None

    # Clear the other half from these columns (its divider wall, any floor under the
    # scroll), widened by the clearance; it is bounded by the footprint, so the lining
    # around the scroll stays.
    widened = mm.offsetMesh(columns, offset=clearance_mm + voxel_size, params=params)  # type: ignore
    cut = mm.voxelBooleanIntersect(inside_far(0), widened, voxel_size)
    return mesa, cut


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

    # Get divider piece
    divider_piece = (
        curved_divider_wall.build_divider_solid(
            radius + mesh_params.lining_offset_mm + mesh_params.wall_thickness_mm,
            112.5 / 2,
            mesh_params.wall_thickness_mm,
        )
        .part.move(bd.Location((0, 0, -100)))
        .solid()
    )
    divider_piece_mesh = brep_to_mesh(divider_piece)

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
    cavity_mesh_pos_offset = mm.offsetMesh(cavity_mesh_pos, offset=mesh_params.wall_thickness_mm, params=params)  # type: ignore
    cavity_mesh_neg_offset = mm.offsetMesh(cavity_mesh_neg, offset=mesh_params.wall_thickness_mm, params=params)  # type: ignore

    # mm.trimWithPlane(cavity_mesh_pos_offset, cut_plane, outCutEdges=hole_edges_pos)
    # mm.trimWithPlane(cavity_mesh_neg_offset, -cut_plane, outCutEdges=hole_edges_neg)
    cavity_mesh_pos_offset = mm.boolean(
        cavity_mesh_pos_offset, divider_piece_mesh, mm.BooleanOperation.Intersection
    ).mesh
    cavity_mesh_neg_offset = mm.boolean(
        cavity_mesh_neg_offset, divider_piece_mesh, mm.BooleanOperation.DifferenceAB
    ).mesh

    # Where the scroll lies wholly on one side of the divider, let the other half rise
    # past the divider to meet it rather than leaving a gap the scroll can move in.
    for mesa_side, cut_side in (("neg", "pos"), ("pos", "neg")):
        mesa, cut = _protrusion(
            offset_mesh,
            divider_piece_mesh,
            below=mesa_side == "neg",
            lining_outer_radius=radius
            + mesh_params.lining_offset_mm
            + mesh_params.wall_thickness_mm,
            wall_thickness_mm=mesh_params.wall_thickness_mm,
            min_gap_mm=mesh_params.protrusion_min_gap_mm,
            clearance_mm=mesh_params.protrusion_clearance_mm,
            voxel_size=voxel_size,
        )
        if mesa is None:
            continue
        logger.info(f"Adding protrusion of the {mesa_side} half past the divider")
        # Exact booleans keep the linings' clean cut along the divider.
        unite = partial(_exact_boolean, op=mm.BooleanOperation.Union, voxel_size=voxel_size)
        subtract = partial(
            _exact_boolean, op=mm.BooleanOperation.DifferenceAB, voxel_size=voxel_size
        )
        if cut_side == "pos":
            cavity_mesh_pos = unite(cavity_mesh_pos, cut)
            cavity_mesh_pos_offset = subtract(cavity_mesh_pos_offset, cut)
            cavity_mesh_neg_offset = unite(cavity_mesh_neg_offset, mesa)
        else:
            cavity_mesh_neg = unite(cavity_mesh_neg, cut)
            cavity_mesh_neg_offset = subtract(cavity_mesh_neg_offset, cut)
            cavity_mesh_pos_offset = unite(cavity_mesh_pos_offset, mesa)

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
