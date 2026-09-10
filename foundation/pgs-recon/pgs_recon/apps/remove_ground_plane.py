import argparse

import pgs_recon.utils.wavefront as wobj
from pgs_recon.utils import geometry as geom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input mesh file')
    parser.add_argument('--output-file', '-o', required=True,
                        help='Output mesh file')
    parser.add_argument('--scale', type=float,
                        help='Scale mesh before processing')
    parser.add_argument('--distance-threshold', type=float, default=0.1,
                        help='During ground plane estimation, points are '
                             'considered inliers if their distance from the '
                             'plane is less than the distance threshold. This '
                             'should be tuned based on the point density.')
    args = parser.parse_args()

    # Load the mesh
    print('Loading mesh...')
    obj = wobj.load_obj(args.input_file)
    mesh = geom.wavefront_to_mesh(obj)

    if args.scale is not None:
        print('Scaling mesh...')
        mesh.vertices *= args.scale

    print('Fitting plane...')
    _, plane_inliers = geom.segment_plane(mesh,
                                          dist_threshold=args.distance_threshold)
    geom.remove_vertices_by_index(mesh, plane_inliers)

    print('Removing small connected components...')
    geom.keep_largest_connected_component(mesh, filter_vertices=True)

    print('Saving mesh...')
    obj = geom.mesh_to_wavefront(mesh, obj)
    wobj.save_obj(obj, args.output_file)


if __name__ == '__main__':
    main()
