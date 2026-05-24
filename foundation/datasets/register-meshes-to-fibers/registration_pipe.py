# Giorgio Angelotti 2025
# python registration_pipe.py --mesh_root example_data/meshes --tif_root example_data/tifs --cube_label_root example_data/labels --output_root registered-meshes
import argparse
import os
import numpy as np
import open3d as o3d
import torch

# ---------------------------
# Import functions from extract-skeleton module
from extract_skeleton_tif import extract_skeleton_from_tif
# ---------------------------
# Import functions from registration module
from registration import MeshSurface, optimize_all_registration, assign_skeletons_to_mesh

# Global device (as defined in registration module)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_FIBER_TYPES = {"hz", "vt"}
MESH_NAME_FORMAT = "<label>_<hz|vt>.obj"


def simplify_mesh_o3d(mesh_o3d: o3d.geometry.TriangleMesh, target_triangles: int) -> o3d.geometry.TriangleMesh:
    """
    Simplify an Open3D mesh to the specified target number of triangles.
    """
    simplified = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_non_manifold_edges()
    return simplified


def load_mesh_from_file(file: str) -> list:
    """
    Loads a mesh from the given .obj file and converts it into a MeshSurface instance.
    Returns a list containing a single MeshSurface.
    """
    mesh_o3d = o3d.io.read_triangle_mesh(file)
    if not mesh_o3d.has_vertices():
        return []
    # Optionally, you could simplify the mesh here.
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    # Convert face connectivity to an edge list.
    edges_set = set()
    for face in faces:
        e1 = tuple(sorted((face[0], face[1])))
        e2 = tuple(sorted((face[1], face[2])))
        e3 = tuple(sorted((face[2], face[0])))
        edges_set.add(e1)
        edges_set.add(e2)
        edges_set.add(e3)
    edges = [list(e) for e in edges_set]
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    # Create a MeshSurface instance (its __post_init__ computes neighbor info).
    mesh_surface = MeshSurface(vertices=vertices_tensor, edges=edges,
                               skeleton_curves=[], vertices_np=vertices)
    return [mesh_surface]


def simplify_curve(curve: np.ndarray, target_points: int) -> np.ndarray:
    """
    Simplify a skeleton curve by uniformly sampling target_points along the curve.
    """
    if curve.shape[0] <= target_points:
        return curve
    indices = np.linspace(0, curve.shape[0] - 1, target_points, dtype=int)
    return curve[indices]


def prepare_skeleton_curves(
    curves_dict: dict,
    curve_type: str,
    origin: np.ndarray,
    skel_axis: str,
    target_points: int,
) -> dict:
    """
    Select and normalize skeleton curves before registration.
    """
    prepared = {curve_type: []}
    for curve in curves_dict.get(curve_type, []):
        new_curve = np.asarray(curve, dtype=np.float32) - origin
        if skel_axis == "zyx":
            new_curve = new_curve[:, [2, 1, 0]]
        if target_points > 0:
            new_curve = simplify_curve(new_curve, target_points)
        prepared[curve_type].append(new_curve)
    return prepared


def parse_mesh_label_and_fiber(mesh_file: str) -> tuple[int, str]:
    filename = os.path.basename(mesh_file)
    base, ext = os.path.splitext(filename)
    parts = base.split("_")
    if ext.lower() != ".obj" or len(parts) != 2:
        raise ValueError(f"mesh filename must match '{MESH_NAME_FORMAT}', got {filename!r}")
    try:
        label = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"mesh filename must match '{MESH_NAME_FORMAT}', got {filename!r}") from exc
    fiber_type = parts[1]
    if fiber_type not in VALID_FIBER_TYPES:
        raise ValueError(f"mesh filename must match '{MESH_NAME_FORMAT}', got {filename!r}")
    return label, fiber_type


def curve_type_for_fiber(fiber_type: str, requested_curve_type: str) -> str:
    if fiber_type not in VALID_FIBER_TYPES:
        raise ValueError("fiber_type must be 'hz' or 'vt'")
    if requested_curve_type != "auto":
        return requested_curve_type
    if fiber_type == "vt":
        return "vertical"
    return "horizontal"


def unified_pipeline(args):
    """
    Pipeline for processing a single mesh. Expects that:
      - args.mesh is the path to the mesh file,
      - args.tif is the path to the fiber inference TIFF,
      - args.cube_label is the path to the cube mask TIFF.
    
    It extracts skeleton curves (via kimimaro), assigns the curves to the mesh, runs registration,
    and finally visualizes the registered mesh with the skeleton curves in one combined window.
    """
    mesh_files = list(getattr(args, "meshes", None) or [args.mesh])
    print(f"Loading {len(mesh_files)} mesh(es) ...")
    meshes = []
    for mesh_file in mesh_files:
        loaded_meshes = load_mesh_from_file(mesh_file)
        for mesh in loaded_meshes:
            mesh._source_mesh_file = mesh_file
        meshes.extend(loaded_meshes)
    if not meshes:
        print("No mesh was loaded. Exiting.")
        return

    curves_dict = {"vertical": [], "horizontal": []}
    origin = np.array([float(x) for x in args.skel_origin.split(",")])
    skeleton_axis_order = getattr(args, "skeleton_axis_order", "zyx")
    for mesh_file in mesh_files:
        # Parse fiber type and label number from mesh filename (assumes format: "<label>_<fiber_type>.obj")
        label_number, fiber_type = parse_mesh_label_and_fiber(mesh_file)
        mesh_curve_type = curve_type_for_fiber(fiber_type, args.curve_type)
        print(f"Extracting skeleton curves from {args.tif} for {mesh_file} ...")
        extracted_curves = extract_skeleton_from_tif(args.tif, args.cube_label,
                                                     z_threshold=1. / np.sqrt(2),
                                                     label=label_number,
                                                     fiber_type=fiber_type,
                                                     axis_order=skeleton_axis_order)
        prepared_curves = prepare_skeleton_curves(
            extracted_curves,
            mesh_curve_type,
            origin,
            args.skel_axis,
            args.target_skel_points,
        )
        mesh_curves = prepared_curves.get(mesh_curve_type, [])
        curves_dict[mesh_curve_type].extend(mesh_curves)
        for mesh in meshes:
            if getattr(mesh, "_source_mesh_file", None) == mesh_file:
                mesh.skeleton_curves = mesh_curves
        if not mesh_curves:
            print(f"No {mesh_curve_type} skeleton curves matched {mesh_file}; skipping this mesh.")
    n_vert = len(curves_dict.get("vertical", []))
    n_horz = len(curves_dict.get("horizontal", []))
    print(f"Skeleton extraction complete. Extracted {n_vert} vertical and {n_horz} horizontal curve(s).")

    global_skel_curves = curves_dict["vertical"] + curves_dict["horizontal"]
    print("Final number of skeleton curves for registration: {}".format(len(global_skel_curves)))
    
    # Check if no curves were extracted and skip registration if so.
    if len(global_skel_curves) == 0:
        print("No skeleton curves were extracted; skipping registration.")
        return

    # Assign skeleton curves to the mesh.
    active_meshes = [mesh for mesh in meshes if getattr(mesh, "skeleton_curves", None)]
    if not active_meshes:
        print("No meshes had matching skeleton curves; skipping registration.")
        return

    for mesh in active_meshes:
        skel_pts = assign_skeletons_to_mesh(mesh.vertices_np, mesh.skeleton_curves, thresh_z=0.1).to(device)
        mesh.skeleton_points = skel_pts

    # (Optional) Visualize initial mesh and skeleton curves.
    initial_meshes_geom = []
    color_mesh = [0, 0, 1]
    for mesh in active_meshes:
        init_pcd = o3d.geometry.PointCloud()
        init_pcd.points = o3d.utility.Vector3dVector(mesh.vertices_np)
        init_pcd.paint_uniform_color(color_mesh)
        initial_meshes_geom.append(init_pcd)

    line_sets = []
    color_map = {"vertical": [1, 0, 0], "horizontal": [0, 0, 1]}
    for label, curves in curves_dict.items():
        for curve in curves:
            num_pts = curve.shape[0]
            edges = [[j, j+1] for j in range(num_pts - 1)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(curve)
            ls.lines = o3d.utility.Vector2iVector(edges)
            ls.colors = o3d.utility.Vector3dVector([color_map[label] for _ in range(len(edges))])
            line_sets.append(ls)
    if args.visualize:
        combined_initial = initial_meshes_geom + line_sets
        o3d.visualization.draw_geometries(combined_initial, window_name="Initial Mesh + Skeleton Curves")

    # Run registration optimization.
    print("Running registration optimization...")
    registered_vertices_list = optimize_all_registration(active_meshes, global_skel_curves,
                                                          num_iters=args.num_iters, lr=args.lr,
                                                          lambda_data=args.lambda_data, lambda_disp=args.lambda_disp,
                                                          lambda_elastic=args.lambda_elastic, lambda_self=args.lambda_self,
                                                          lambda_lap=args.lambda_lap, lambda_arap=args.lambda_arap, lambda_sdf=args.lambda_sdf,
                                                          tau=args.tau, delta=args.delta, beta=args.beta, tau_sdf=args.tau_sdf,
                                                          lambda_inter=args.lambda_inter, delta_inter=args.delta_inter, beta_inter=args.beta_inter,
                                                          batch_size=args.batch_size)
    # Final visualization: combine the registered mesh and the skeleton curves.
    registered_meshes_geom = []
    for mesh in active_meshes:
        reg_vertices = (mesh.vertices + mesh.displacement).detach().cpu().numpy()
        reg_pcd = o3d.geometry.PointCloud()
        reg_pcd.points = o3d.utility.Vector3dVector(reg_vertices)
        reg_pcd.paint_uniform_color(color_mesh)
        registered_meshes_geom.append(reg_pcd)
    final_geoms = registered_meshes_geom + line_sets
    if args.visualize:
        o3d.visualization.draw_geometries(final_geoms, window_name="Registered Mesh + Skeleton Curves")

    # Return the updated meshes so that their displacement is taken into account.
    return active_meshes


def process_one_mesh(mesh_file: str, tif_file: str, cube_label_file: str, output_root: str, global_args):
    """
    Process a single mesh: run the registration pipeline (using unified_pipeline),
    and save the registered mesh to an output folder that mimics the mesh tree structure.
    """
    print("Processing mesh:", mesh_file)
    # Prepare the temporary args object.
    class TempArgs:
        pass
    temp = TempArgs()
    temp.mesh = mesh_file
    temp.tif = tif_file
    temp.cube_label = cube_label_file
    # Copy registration parameters from global_args.
    temp.num_iters = global_args.num_iters
    temp.lr = global_args.lr
    temp.lambda_data = global_args.lambda_data
    temp.lambda_disp = global_args.lambda_disp
    temp.lambda_elastic = global_args.lambda_elastic
    temp.lambda_self = global_args.lambda_self
    temp.lambda_lap = global_args.lambda_lap
    temp.lambda_arap = global_args.lambda_arap
    temp.lambda_sdf = global_args.lambda_sdf
    temp.batch_size = global_args.batch_size
    temp.tau = global_args.tau
    temp.delta = global_args.delta
    temp.beta = global_args.beta
    temp.tau_sdf = global_args.tau_sdf
    temp.lambda_inter = global_args.lambda_inter
    temp.delta_inter = global_args.delta_inter
    temp.beta_inter = global_args.beta_inter
    temp.target_triangles = global_args.target_triangles
    temp.skel_origin = global_args.skel_origin
    temp.skel_axis = global_args.skel_axis
    temp.skeleton_axis_order = getattr(global_args, "skeleton_axis_order", "zyx")
    temp.curve_type = global_args.curve_type
    temp.target_skel_points = global_args.target_skel_points
    # Parse fiber type and label number from the mesh filename.
    temp.skeleton_label, temp.fiber_type = parse_mesh_label_and_fiber(mesh_file)
    temp.visualize = global_args.visualize

    # Run the registration pipeline and get the updated meshes.
    registered_meshes = unified_pipeline(temp)
    if registered_meshes is None or len(registered_meshes) == 0:
        print("Registration failed for", mesh_file)
        return

    # Use the updated mesh with displacement applied.
    mesh_surface = registered_meshes[0]
    reg_vertices = (mesh_surface.vertices + mesh_surface.displacement).detach().cpu().numpy()

    # Re-read the original mesh (to obtain faces) and update its vertices.
    original_mesh = o3d.io.read_triangle_mesh(mesh_file)
    original_mesh.vertices = o3d.utility.Vector3dVector(reg_vertices)
    # Determine the output file path.
    rel_path = os.path.relpath(mesh_file, global_args.mesh_root)
    out_file = os.path.join(output_root, os.path.splitext(rel_path)[0] + "_registered.obj")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("Saving registered mesh to:", out_file)
    o3d.io.write_triangle_mesh(out_file, original_mesh)


def process_mesh_group(mesh_files: list[str], tif_file: str, cube_label_file: str, output_root: str, global_args):
    """
    Process meshes that share one TIFF/cube-label context in a single optimization call.
    """
    if not mesh_files:
        return
    print("Processing mesh group:", mesh_files)

    class TempArgs:
        pass
    temp = TempArgs()
    temp.meshes = list(mesh_files)
    temp.tif = tif_file
    temp.cube_label = cube_label_file
    temp.num_iters = global_args.num_iters
    temp.lr = global_args.lr
    temp.lambda_data = global_args.lambda_data
    temp.lambda_disp = global_args.lambda_disp
    temp.lambda_elastic = global_args.lambda_elastic
    temp.lambda_self = global_args.lambda_self
    temp.lambda_lap = global_args.lambda_lap
    temp.lambda_arap = global_args.lambda_arap
    temp.lambda_sdf = global_args.lambda_sdf
    temp.batch_size = global_args.batch_size
    temp.tau = global_args.tau
    temp.delta = global_args.delta
    temp.beta = global_args.beta
    temp.tau_sdf = global_args.tau_sdf
    temp.lambda_inter = global_args.lambda_inter
    temp.delta_inter = global_args.delta_inter
    temp.beta_inter = global_args.beta_inter
    temp.target_triangles = global_args.target_triangles
    temp.skel_origin = global_args.skel_origin
    temp.skel_axis = global_args.skel_axis
    temp.skeleton_axis_order = getattr(global_args, "skeleton_axis_order", "zyx")
    temp.curve_type = global_args.curve_type
    temp.target_skel_points = global_args.target_skel_points
    temp.visualize = global_args.visualize

    registered_meshes = unified_pipeline(temp)
    if registered_meshes is None or len(registered_meshes) == 0:
        print("Registration failed for mesh group", mesh_files)
        return

    for mesh_file, mesh_surface in zip(mesh_files, registered_meshes):
        mesh_file = getattr(mesh_surface, "_source_mesh_file", mesh_file)
        reg_vertices = (mesh_surface.vertices + mesh_surface.displacement).detach().cpu().numpy()
        original_mesh = o3d.io.read_triangle_mesh(mesh_file)
        original_mesh.vertices = o3d.utility.Vector3dVector(reg_vertices)
        rel_path = os.path.relpath(mesh_file, global_args.mesh_root)
        out_file = os.path.join(output_root, os.path.splitext(rel_path)[0] + "_registered.obj")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        print("Saving registered mesh to:", out_file)
        o3d.io.write_triangle_mesh(out_file, original_mesh)



def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: register meshes to skeleton fibers extracted from fiber inference TIFFs. "
                    "Input arguments are root folders; the output tree will mimic the mesh root folder structure."
    )
    parser.add_argument("--mesh_root", type=str, required=True,
                        help="Root folder containing mesh subfolders (e.g., 'steroids').")
    parser.add_argument("--tif_root", type=str, required=True,
                        help="Root folder containing fiber inference TIFFs (files named like <subfolder>.tif).")
    parser.add_argument("--cube_label_root", type=str, required=True,
                        help="Root folder containing cube label masks (subfolder structure with <subfolder>_mask.tif).")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output folder where registered meshes will be saved (mimicking the mesh folder tree).")
    parser.add_argument("--visualize", action="store_true", help="Visualize intermediate and final results.")
    # Registration hyperparameters.
    parser.add_argument("--num_iters", type=int, default=10000, help="Number of registration iterations.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for registration.")
    parser.add_argument("--lambda_data", type=float, default=1.0, help="Data term weight.")
    parser.add_argument("--lambda_disp", type=float, default=1e-3, help="Displacement term weight.")
    parser.add_argument("--lambda_elastic", type=float, default=1.0, help="Elasticity term weight.")
    parser.add_argument("--lambda_self", type=float, default=1e-1, help="Self-intersection term weight.")
    parser.add_argument("--lambda_lap", type=float, default=5e-1, help="Laplacian term weight.")
    parser.add_argument("--lambda_arap", type=float, default=2.0, help="ARAP term weight.")
    parser.add_argument("--lambda_sdf", type=float, default=1.0, help="SDF term weight.")
    parser.add_argument("--batch_size", type=int, default=40960*2, help="Batch size for registration.")
    parser.add_argument("--tau", type=float, default=0.01, help="Tau parameter for registration.")
    parser.add_argument("--delta", type=float, default=0.05, help="Delta parameter for registration.")
    parser.add_argument("--beta", type=float, default=10.0, help="Beta parameter for registration.")
    parser.add_argument("--tau_sdf", type=float, default=0.01, help="Tau SDF parameter for registration.")
    parser.add_argument("--lambda_inter", type=float, default=0.1, help="Inter-mesh intersection term weight.")
    parser.add_argument("--delta_inter", type=float, default=0.05, help="Delta parameter for inter-mesh intersection.")
    parser.add_argument("--beta_inter", type=float, default=10.0, help="Beta parameter for inter-mesh intersection.")
    parser.add_argument("--target_triangles", type=int, default=1000,
                        help="Reserved for topology-aware mesh decimation; not applied by the current pipeline.")
    # Skeleton processing arguments.
    parser.add_argument("--skel_origin", type=str, default="0,0,0",
                        help="Origin offset for skeleton curves as comma-separated values, e.g., '0,0,0'.")
    parser.add_argument("--skel_axis", type=str, choices=["xyz", "zyx"], default="zyx",
                        help="Axis order for skeleton curves before registration. The default 'zyx' matches the skeleton extraction default and converts curves to mesh-style 'xyz'.")
    parser.add_argument("--curve_type", type=str, choices=["auto", "vertical", "horizontal"], default="auto",
                        help="Type of skeleton curves to use for registration. 'auto' maps hz meshes to horizontal and vt meshes to vertical.")
    parser.add_argument("--skeleton_axis_order", type=str, choices=["zyx", "xyz"], default="zyx",
                        help="Coordinate order used for PCA classification during skeleton extraction.")
    parser.add_argument("--target_skel_points", type=int, default=512,
                        help="Target number of points for skeleton curve simplification.")
    # These two are needed for the current extraction routine.
    parser.add_argument("--cube_label", type=str, help="(Temporary) cube label file; not used when processing in batch.")
    parser.add_argument("--skeleton_label", type=int, default=1,
                        help="(Temporary) skeleton label; will be parsed from mesh filename in batch mode.")
    
    args = parser.parse_args()
    if args.target_triangles != 1000:
        print("--target_triangles is reserved for a future topology-aware decimation pass and is not applied.")

    # Traverse the mesh root folder recursively and process .obj files by cube context.
    for root, dirs, files in os.walk(args.mesh_root):
        group_mesh_files = []
        for file in files:
            if file.lower().endswith(".obj"):
                group_mesh_files.append(os.path.join(root, file))
        if not group_mesh_files:
            continue

        group_mesh_files.sort()
        subfolder = os.path.basename(root)
        tif_file = os.path.join(args.tif_root, subfolder + ".tif")
        cube_label_file = os.path.join(args.cube_label_root, subfolder, subfolder + "_mask.tif")
        if not os.path.isfile(tif_file):
            print(f"Skipping {root}: corresponding TIF file not found: {tif_file}")
            continue
        if not os.path.isfile(cube_label_file):
            print(f"Skipping {root}: corresponding cube label not found: {cube_label_file}")
            continue
        process_mesh_group(group_mesh_files, tif_file, cube_label_file, args.output_root, args)
    
if __name__ == "__main__":
    main()
