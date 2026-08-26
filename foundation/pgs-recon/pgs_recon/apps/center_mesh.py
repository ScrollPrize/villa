import argparse
import random
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from vtkmodules.vtkCommonCore import mutable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolygon
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkTriangleMeshPointNormals
from vtkmodules.vtkFiltersGeneral import vtkOBBTree, vtkTransformPolyDataFilter

import pgs_recon.utils.educelab as el
from pgs_recon.utils import wavefront as wobj


# convert a list of vertices and polygons (output from a WavefrontOBJ) to a
# vtkPolyData. Similar to wavefront.mesh_to_polydata(), but pid determines
# which polygon property to use for cell creation:
# 0 = vertex
# 1 = uv coordinate
# 2 = vertex normal
def mesh_from_obj_props(verts, polys, pid):
    polydata = vtkPolyData()

    # vertices
    pts = vtkPoints()
    for v in verts:
        pts.InsertNextPoint(v)
    polydata.SetPoints(pts)

    # faces
    cells = vtkCellArray()
    for p in polys:
        poly = vtkPolygon()
        for v in p:
            poly.GetPointIds().InsertNextId(v[pid])
        cells.InsertNextCell(poly)
    polydata.SetPolys(cells)
    return polydata


# calculate a 3D point on a triangle from a barycentric coordinate and the
# triangle's three vertices
def interpolate_on_tri(bary, a, b, c):
    x = bary[0] * a[0] + bary[1] * b[0] + bary[2] * c[0]
    y = bary[0] * a[1] + bary[1] * b[1] + bary[2] * c[1]
    z = bary[0] * a[2] + bary[1] * b[2] + bary[2] * c[2]
    return np.array([x, y, z])


# given a UV coordinate, find its corresponding 3D point on the mesh
# uv_tree: vtkOBBTree built with a meshed version of the UV map
# uv_pt: the UV coordinate to lookup
# mesh: a mesh of type WavefrontOBJ
def lookup_uv_to_3d(uv_tree, uv_pt, mesh):
    # Inputs
    p1 = [uv_pt[0], uv_pt[1], -1.0]
    p2 = [uv_pt[0], uv_pt[1], 1.0]
    tolerance = 0.001

    # Outputs
    t = mutable(0)
    x = [0.0, 0.0, 0.0]
    params = [0.0, 0.0, 0.0]
    s_id = mutable(0)
    c_id = mutable(0)
    res = uv_tree.IntersectWithLine(p1, p2, tolerance, t, x, params, s_id, c_id)

    if res > 0:
        bary = [1 - params[0] - params[1], params[0], params[1]]
        a, b, c = [mesh.vertices[c[0]] for c in mesh.polygons[c_id]]
        x = interpolate_on_tri(bary, a, b, c)
        return x
    else:
        return None

# Default align_vector tolerance
_ALIGN_ATOL = 2e-3

# calculate rotation matrix that aligns vector a to b
# If a == b, return identity
# If -a == b, return 180-degree rotation around c
def align_vector(a, b, c, atol=_ALIGN_ATOL):
    # Identity initial rotation
    r = np.eye(3)

    # First, check for anti-parallel vectors
    if np.isclose(a.dot(b), -1., atol=atol):
        d = np.copysign((1., 1., 1.), np.abs(c) - 1)
        r = np.diagflat(d)

    # Next, return early for parallel vectors
    a = r @ a
    if np.isclose(a.dot(b), 1., atol=atol):
        return r

    # Finally, refine the rotation with a special form of Rodriguez rotation
    # https://math.stackexchange.com/a/476311
    v = np.cross(a, b)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    rod = np.eye(3) + vx + np.dot(vx, vx) / (1 + c)
    return rod @ r


# Find the closest vector to v in vs
# If result is anti-parallel to v, it will be flipped
def find_closest_vector(v, vs):
    dots = [np.dot(el.unit_vec(v), el.unit_vec(e)) for e in vs]
    edge_idx = np.argmax(np.abs(dots))
    v = el.unit_vec(vs[edge_idx])
    if dots[edge_idx] < 0:
        v *= -1
    return v


# Orient and center the mesh using its oriented bounding box and surface normals
def bounding_box_calibration(polydata, max_edge, mid_edge, max_dir, mid_dir,
                             flip_max, flip_mid):
    # Get basis vectors
    max_target = basis_vectors[max_dir]
    mid_target = basis_vectors[mid_dir]
    min_target = 1 - (max_target + mid_target)

    # Align max edge to max target
    max_target = flip_max * max_target
    r = align_vector(el.unit_vec(max_edge), max_target, mid_target)

    # Rotate mid edge by prev rot, align to mid target, and update rot
    mid_target = flip_mid * mid_target
    r = align_vector(r @ el.unit_vec(mid_edge), mid_target, max_target) @ r

    # Sample N normals
    num_pts = polydata.GetNumberOfPoints()
    num_samples = min(num_pts, 1000)
    n_idxs = random.sample(range(polydata.GetNumberOfPoints()), num_samples)
    normals = []
    nrml_data = polydata.GetPointData().GetArray('Normals')
    for idx in n_idxs:
        normals.append(nrml_data.GetTuple(idx))

    # Update normals to their new rotated orientation
    normals = np.array(normals)
    normals = (r @ normals.T).T

    # See how many samples are in the direction of min_target
    s = 0
    for n in normals:
        v = np.dot(n, min_target)
        s += 1 if v > 0 else 0

    # If <= 40%, then rotate 180 degrees
    if s <= num_samples * 0.4:
        r = align_vector(-min_target, min_target, mid_target) @ r

    # Scale is identity for this method
    scale = np.eye(4, dtype=np.float32)

    return scale, r


# automatically scale and orient a mesh by detecting the EL sample square
# assumes the texture image has been reordered
def sample_square_calibration(mesh, img, edges):
    # Defaults
    scale = np.eye(4, dtype=np.float32)
    r = np.eye(3, dtype=np.float32)

    # Detect board
    print('Detecting EduceLab sample square...')
    detected, boards, ppcm, kp_ids, kp_pixels, flip, rotate = el.detect_sample_square(
        img)

    # Nothing detected
    if not detected:
        print('Not detected.')
        return detected, scale, r, None

    # Report detection results
    num_markers = sum((b.marker_cnt for b in boards))
    num_boards = sum((b.board_cnt for b in boards))
    print(f'Detected:\n'
          f' - Markers: {num_markers}\n'
          f' - Board corners: {num_boards}\n'
          f' - Texture resolution (pixels/cm): {ppcm}')

    # Flip the image and UV map if needed
    # OpenCV flip codes: 0 == vertical, 1 == horizontal
    if flip is not None:
        print('Flipping image...')
        img = cv2.flip(img, flip)
        for idx, uv in enumerate(mesh.texcoords):
            u = 1. - uv[0] if flip == 1 else uv[0]
            v = 1. - uv[1] if flip == 0 else uv[1]
            mesh.texcoords[idx] = [u, v]

    # Rotate image
    if rotate is not None:
        msg = ['90°', '180°', '270°']
        print(f'Rotating image {msg[rotate]}...')
        img = cv2.rotate(img, rotate)
        for idx, uv in enumerate(mesh.texcoords):
            mesh.texcoords[idx] = el.rotate_kp(uv, (1., 1.), rotate).tolist()

    # Convert UV map to polydata
    print('Computing UV lookup tree...')
    uvs = [[uv[0], uv[1], 0.] for uv in mesh.texcoords]
    uv_tree = vtkOBBTree()
    uv_tree.SetDataSet(mesh_from_obj_props(uvs, mesh.polygons, pid=1))
    uv_tree.BuildLocator()

    # ORIENTATION
    if num_markers > 0:
        print('Calculating orientation...')
        right_samples = []
        down_samples = []
        for b in boards:
            for m in b.marker_corners:
                pts = m[0] / [img.shape[1] - 1, img.shape[0] - 1]
                pts = [lookup_uv_to_3d(uv_tree, uv, mesh) for uv in pts]
                if not np.all(pts):
                    continue
                right_samples.append(el.unit_vec(pts[1] - pts[0]))
                right_samples.append(el.unit_vec(pts[2] - pts[3]))
                down_samples.append(el.unit_vec(pts[3] - pts[0]))
                down_samples.append(el.unit_vec(pts[2] - pts[1]))

        # Make sure we detected the UVs correctly
        if len(right_samples) == 0 and len(down_samples) == 0:
            print('Error: Found orientation markers but the corners don\'t '
                  'lie in the UV map. Cannot calculate orientation.')
        else:
            # Rotation targets
            right_target = basis_vectors['x']
            down_target = -basis_vectors['y']

            # Align right edge to X axis
            right = np.mean(right_samples, axis=0)
            if edges is not None:
                right = find_closest_vector(right, edges)
            r = align_vector(right, right_target, down_target)

            # Align down edge to Y axis
            down = np.mean(down_samples, axis=0)
            if edges is not None:
                down = find_closest_vector(down, edges)
            r = align_vector(r @ down, down_target, right_target) @ r
    else:
        print('Warning: No markers detected. Cannot calculate orientation.')

    # SCALE
    print('Calculating scale...')
    # Convert pixels to 3D points
    kp_pos = []
    for p in kp_pixels:
        uv = p / [img.shape[1] - 1, img.shape[0] - 1]
        res = lookup_uv_to_3d(uv_tree, uv, mesh)
        # Stop processing this box if no intersection
        if res is None:
            break
        else:
            kp_pos.append(res)
    scale_samples = []
    distances_expected = []
    distances_measured = []
    for ids, pts in zip(combinations(kp_ids, r=2),
                        combinations(kp_pos, r=2)):
        dist_cm = el.kp_dist(ids[0], ids[1])
        dist_3d = np.linalg.norm(pts[1] - pts[0])
        distances_expected.append(dist_cm)
        distances_measured.append(dist_3d)
        scale_samples.append(dist_cm / dist_3d)
    scale_factor = np.mean(scale_samples)
    scale = np.eye(4, dtype=np.float32)
    np.fill_diagonal(scale[0:3, 0:3], scale_factor)

    # Measure error
    errors = []
    for expected, measured in zip(distances_expected, distances_measured):
        errors.append(np.abs(measured * scale_factor - expected))

    # Report results
    print(f'Scale factor: {scale_factor:.7f}')
    print(f'Absolute error (cm):')
    print(f' - Max: {np.max(errors):.7f}')
    print(f' - Min: {np.min(errors):.7f}')
    print(f' - Mean: {np.mean(errors):.7f}')
    print(f' - Median: {np.median(errors):.7f}')

    return detected, scale, r, img if flip is not None or rotate is not None else None


basis_vectors = {
    'x': np.array((1, 0, 0)),
    'y': np.array((0, 1, 0)),
    'z': np.array((0, 0, 1))
}


def main():
    # Parser arguments
    parser = argparse.ArgumentParser('pgs-center')
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input OBJ file')
    parser.add_argument('--output-file', '-o', required=True,
                        help='Output OBJ file')
    parser.add_argument('--sample-square-calibration',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='If enabled, attempt to detect the EduceLab sample'
                             'square for automatic mesh scaling and '
                             'orientation.')
    parser.add_argument('--load-transform',
                        help='If provided, load a centering transform from the '
                             'given file path.')
    parser.add_argument('--save-transform',
                        help='If provided, save the centering transform to the '
                             'given file path.')

    ss_opts = parser.add_argument_group('sample square calibration options')
    ss_opts.add_argument('--use-marker-dirs', action='store_true',
                         help='By default, sample square calibration will '
                              'reorient the mesh using the bounding box edges '
                              'which most closely align to the right and down '
                              'directions calculated from the detected sample '
                              'square. When this flag is provided, it will '
                              'instead use the mean directions calculated from '
                              'the sample square markers. These are often less '
                              'globally accurate than the bounding box edges.')
    ss_opts.add_argument('--parallel-atol', type=float,
                         help='Tolerance for detecting parallel vectors when '
                              'calculating vector-to-vector rotation.')

    bb_opts = parser.add_argument_group('bounding box calibration options')
    bb_opts.add_argument('--max-dir', type=str.lower, choices=['x', 'y', 'z'],
                         default='x',
                         help='Axis to which the largest bounding box edge is '
                              'mapped')
    bb_opts.add_argument('--mid-dir', type=str.lower, choices=['x', 'y', 'z'],
                         default='y',
                         help='Axis to which the 2nd largest bounding box edge '
                              'is mapped')
    bb_opts.add_argument('--flip-max', action='store_const', const=-1,
                         default=1,
                         help='If provided, invert the axis to which the '
                              'largest bounding box edge is mapped')
    bb_opts.add_argument('--flip-mid', action='store_const', const=-1,
                         default=1,
                         help='If provided, invert the axis to which the 2nd '
                              'largest bounding box edge is mapped')
    args = parser.parse_args()

    if args.max_dir == args.mid_dir:
        print('Error: --max-dir and --mid-dir must be different values')
        sys.exit(1)

    # load obj
    print('Loading mesh...')
    mesh = wobj.load_obj(args.input_file)
    poly_data = wobj.mesh_to_polydata(mesh)

    # Find path to a texture image
    texture_path = None
    img = None
    if args.sample_square_calibration:
        for mtl_lib in mesh.mtllibs:
            mtl_path = (mesh.path.parent / mtl_lib).resolve()
            mtl = wobj.load_mtllib(mtl_path)
            for m in mtl.values():
                if 'map_Kd' in m.keys():
                    texture_path = mtl_path.parent / m['map_Kd']
                    break

        # Load texture image
        if texture_path is None:
            print('Warning: Mesh does not have texture map. '
                  'Skipping sample square calibration.')
            args.sample_square_calibration = False
        else:
            print('Loading texture image...')
            img = cv2.imread(str(texture_path))

    # Calculate normals
    if poly_data.GetPointData().HasArray('Normals') == 0:
        print('Generating normals...')
        ngen = vtkTriangleMeshPointNormals()
        ngen.SetInputData(poly_data)
        ngen.Update()
        poly_data = ngen.GetOutput()

    # overwrite default parallel vec tolerance
    # better to pass this as a var all the way through to usage, but I'm lazy
    if args.parallel_atol:
        global _ALIGN_ATOL
        _ALIGN_ATOL = args.parallel_atol

    # load or generate transform
    new_img = None
    if args.load_transform is not None:
        print('Loading transform from file...')
        tfm_mat = np.load(args.load_transform)
    else:
        # Setup identity transforms
        trans = np.eye(4, dtype=np.float32)
        rot = np.eye(4, dtype=np.float32)
        scale = np.eye(4, dtype=np.float32)

        # calculate OBB
        print('Computing OBB...')
        obb_tree = vtkOBBTree()
        corner = np.array([0., 0., 0.])
        max_edge = np.array([0., 0., 0.])
        mid_edge = np.array([0., 0., 0.])
        min_edge = np.array([0., 0., 0.])
        sizes = np.array([0., 0., 0.])
        obb_tree.ComputeOBB(poly_data, corner, max_edge, mid_edge, min_edge,
                            sizes)

        # Center mesh on origin
        trans[0:3, 3] = -(corner + 0.5 * (max_edge + mid_edge + min_edge))

        # Calculate scale and orientation from sample square
        detected = False
        if args.sample_square_calibration:
            edges = None
            if not args.use_marker_dirs:
                edges = (max_edge, mid_edge, min_edge)
            detected, scale, rot[0:3, 0:3], new_img = sample_square_calibration(
                mesh, img, edges)

        # Fallback to bounding box method if sample square disabled/failed
        if not args.sample_square_calibration or not detected:
            print('Starting bounding box calibration...')
            scale, rot[0:3, 0:3] = bounding_box_calibration(poly_data,
                                                            max_edge,
                                                            mid_edge,
                                                            args.max_dir,
                                                            args.mid_dir,
                                                            args.flip_max,
                                                            args.flip_mid)

        # Setup transform matrix
        tfm_mat = scale @ rot @ trans

    # transform polydata
    print('Transforming mesh...')
    tfm = vtkTransform()
    tfm.SetMatrix(tfm_mat.flatten().tolist())
    transformer = vtkTransformPolyDataFilter()
    transformer.SetInputData(poly_data)
    transformer.SetTransform(tfm)
    transformer.Update()

    # save transform
    if args.save_transform is not None:
        print('Saving transform to file...')
        np.save(args.save_transform, tfm_mat, allow_pickle=False)

    # convert back to WavefrontOBJ
    print('Preparing output obj file...')
    mesh_tfm = wobj.polydata_to_mesh(transformer.GetOutput(), src_mesh=mesh)

    # collect replacement textures
    textures = None
    if new_img is not None:
        textures = {texture_path.name: new_img}

    # save new obj
    print('Saving mesh...')
    output_file = Path(args.output_file)
    wobj.save_obj(mesh_tfm, output_file, _textures=textures)


if __name__ == '__main__':
    main()
