# Vesuvius Challenge Team 2024
# integrated from ThaumatoAnakalyptor
# Integration of https://github.com/giorgioangel/slim-flatboi

import os
import open3d as o3d
import numpy as np
from math import sqrt, cos, sin, radians
from tqdm import tqdm
import igl
from PIL import Image
import cv2
import multiprocessing
from scipy.stats import pearsonr

Image.MAX_IMAGE_PIXELS = None


def print_array_to_file(array, file_path):
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each element of the array to the file
        for element in array:
            file.write(str(element) + "\n")


def triangle_area_multiprocessing(args):
    v1, v2, v3, area_cutoff = args
    return triangle_area(v1, v2, v3) > area_cutoff


def triangle_area(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


class Flatboi:
    input_obj: str
    output_obj: str
    max_iter: int

    def __init__(
        self, obj_path: str, max_iter: int, output_obj: str = None, um: float = 7.91
    ):
        self.stretch_factor = 1000.0
        self.input_obj = obj_path
        if output_obj is not None:
            self.output_obj = output_obj
        else:
            self.output_obj = obj_path.replace(".obj", "_flatboi.obj")
        self.max_iter = max_iter
        self.read_mesh(obj_path)
        self.filter_mesh()

    def read_mesh(self, obj_path):
        self.mesh = o3d.io.read_triangle_mesh(obj_path)
        self.vertices = (
            np.asarray(self.mesh.vertices, dtype=np.float64) / self.stretch_factor
        )
        if len(np.asarray(self.mesh.vertex_normals)) > 0:
            self.vertex_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float64)
        self.triangles = np.asarray(self.mesh.triangles, dtype=np.int64)
        if len(np.asarray(self.mesh.triangle_uvs)) > 0:
            self.original_uvs = np.asarray(self.mesh.triangle_uvs, dtype=np.float64)
        else:
            self.original_uvs = None

    def filter_mesh(self, area_cutoff=0.0000001):
        # Filtering out any triangles with 0 area
        len_before = len(self.triangles)

        args = [
            (self.vertices[t[0]], self.vertices[t[1]], self.vertices[t[2]], area_cutoff)
            for t in self.triangles
        ]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            filter_list = list(
                tqdm(
                    pool.imap(triangle_area_multiprocessing, args),
                    total=len(args),
                    desc="Filtering Triangles",
                )
            )

        # Filter original uvs if they exist
        if self.original_uvs is not None:
            self.original_uvs = np.array(self.original_uvs).reshape((-1, 3, 2))
            if len(self.original_uvs) == len(self.triangles):
                self.original_uvs = np.array(
                    [
                        self.original_uvs[i]
                        for i in range(len(self.triangles))
                        if filter_list[i]
                    ]
                )
                self.original_uvs = np.array(self.original_uvs).reshape((-1, 2))
        # Filter triangles
        self.triangles = np.array(
            [self.triangles[i] for i in range(len(self.triangles)) if filter_list[i]]
        )
        print(
            f"Filtered out {len_before - len(self.triangles)} triangles with 0 area of total {len_before} triangles."
        )

    def generate_boundary(self):
        res = igl.boundary_loop(self.triangles)
        print("Generated Boundary")
        return res

    def harmonic_ic(self):
        bnd = self.generate_boundary()
        bnd_uv = igl.map_vertices_to_circle(self.vertices, bnd)
        # harmonic map to unit circle, this will be the initial condition
        uv = igl.harmonic(self.vertices, self.triangles, bnd, bnd_uv, 1)
        return bnd, bnd_uv, uv

    def arap_ic(self):
        bnd = self.generate_boundary()
        bnd_uv = igl.map_vertices_to_circle(self.vertices, bnd)
        uv = igl.harmonic(self.vertices, self.triangles, bnd, bnd_uv, 1)
        arap = igl.ARAP(self.vertices, self.triangles, 2, np.zeros(0))
        print("ARAP")
        for i in tqdm(range(10), desc="ARAP"):
            uv = arap.solve(np.zeros((0, 0)), uv)
        uva = arap.solve(np.zeros((0, 0)), uv)

        bc = np.zeros((bnd.shape[0], 2), dtype=np.float64)
        for i in tqdm(range(bnd.shape[0])):
            bc[i] = uva[bnd[i]]

        return bnd, bc, uva

    def original_ic(self):
        input_directory = os.path.dirname(self.input_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.input_obj))
        tif_path = os.path.join(input_directory, f"{base_file_name}_0.png")

        # Check if the .mtl file exists
        if not os.path.exists(tif_path):
            raise FileNotFoundError("No .tif file found.")

        with Image.open(tif_path) as img:
            width, height = img.size

        uv = np.zeros((self.vertices.shape[0], 2), dtype=np.float64)
        uvs = self.original_uvs.reshape(
            (self.triangles.shape[0], self.triangles.shape[1], 2)
        )
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                uv[self.triangles[t, v]] = uvs[t, v]

        # Multiply uv coordinates by image dimensions
        uv[:, 0] *= width
        uv[:, 1] *= height

        bnd = self.generate_boundary()
        bnd_uv = np.zeros((bnd.shape[0], 2), dtype=np.float64)

        for i in range(bnd.shape[0]):
            bnd_uv[i] = uv[bnd[i]]

        return bnd, bnd_uv, uv

    def ordered_ic(self):
        uv = np.zeros((self.vertices.shape[0], 2), dtype=np.float64)
        uvs = self.original_uvs.reshape(
            (self.triangles.shape[0], self.triangles.shape[1], 2)
        )
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                uv[self.triangles[t, v]] = uvs[t, v]

        bnd = np.array([[0]])
        bnd_uv = np.zeros((bnd.shape[0], 2), dtype=np.float64)
        for i in range(bnd.shape[0]):
            bnd_uv[i] = uv[bnd[i]]

        return np.zeros((0, 1), dtype=np.int32), np.zeros((0, 2), dtype=np.float64), uv

    def orient_uvs(self, vertices):
        # Assert that no NaNs or Infs are present
        assert not np.any(
            np.isnan(vertices)
        ), f"There are {np.sum(np.isnan(vertices))} NaNs in the vertices."
        assert not np.any(
            np.isinf(vertices)
        ), f"There are {np.sum(np.isinf(vertices))} Infs in the vertices."
        print("Orienting UVs...")
        # Rotate vertices and calculate the needed area
        vertices[:, 0] = 1.0 - vertices[:, 0]
        u_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        v_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        u_longer_v = u_range > v_range
        u_return = vertices[:, 0]
        v_return = vertices[:, 1]

        area_return = u_range * v_range
        for angle in range(-70, 70, 5):
            u_prime = vertices[:, 0] * np.cos(np.deg2rad(angle)) - vertices[
                :, 1
            ] * np.sin(np.deg2rad(angle))
            v_prime = vertices[:, 0] * np.sin(np.deg2rad(angle)) + vertices[
                :, 1
            ] * np.cos(np.deg2rad(angle))
            u_prime_range = np.max(u_prime) - np.min(u_prime)
            v_prime_range = np.max(v_prime) - np.min(v_prime)
            if u_prime_range < v_prime_range and u_longer_v:
                continue
            elif u_prime_range > v_prime_range and not u_longer_v:
                continue
            area = u_prime_range * v_prime_range
            if area < area_return:
                u_return = u_prime
                v_return = v_prime
                area_return = area

        slim_uvs = np.stack((u_return, v_return), axis=-1)
        return slim_uvs

    def slim_optimization(self, slim, old_uvs, iterations=None):
        if iterations is None:
            iterations = self.max_iter
        energies = np.zeros(iterations + 1, dtype=np.float64)
        energies[0] = slim.energy()

        threshold = 1e-5
        converged = False
        iteration = 0
        while iteration < iterations:
            print(iteration)
            temp_energy = slim.energy()
            slim.solve(1)
            new_energy = slim.energy()
            energies[iteration + 1] = new_energy
            iteration += 1
            print(f"{temp_energy:.3f} {new_energy:.3f}")
            if (
                new_energy >= float("inf")
                or new_energy == float("nan")
                or np.isnan(new_energy)
                or np.isinf(new_energy)
            ):
                converged = False
                break
            elif new_energy < temp_energy:
                if abs(new_energy - temp_energy) < threshold:
                    converged = True
                    break
                else:
                    converged = True
            elif new_energy == temp_energy:
                converged = True
                break
            else:
                converged = False
                break

        if (
            converged
            and not np.any(np.isnan(slim.vertices()))
            and not np.any(np.isinf(slim.vertices()))
        ):
            print("Converg(ed/ing)")
            slim_uvs = slim.vertices()
        else:
            print("Not Converged")
            slim_uvs = old_uvs

        slim_uvs = self.orient_uvs(slim_uvs)
        slim_uvs = slim_uvs.astype(np.float64)

        return slim_uvs, energies

    def slim(self, initial_condition="original"):
        def print_errors(slim_uvs):
            l2, linf, area_error = self.stretch_metrics(slim_uvs)
            print(
                f"Stretch metrics L2: {l2:.5f}, Linf: {linf:.5f}, Area Error: {area_error:.5f}",
                end="\n",
            )

        if (
            initial_condition == "original" or initial_condition == "ordered"
        ) and self.original_uvs is None:
            print(
                f"No original UVs available. Cannot use initial condition '{initial_condition}'. Using 'arap' instead."
            )
            initial_condition = "arap"

        if initial_condition == "original":
            print("Using Cylindrical Unwrapping UV Condition")
            bnd, bnd_uv, uv = self.original_ic()
            l2, linf, area_error = self.stretch_metrics(uv)
            print(
                f"Stretch metrics ABF L2: {l2:.5f}, Linf: {linf:.5f}, Area Error: {area_error:.5f}",
                end="\n",
            )
        elif initial_condition == "arap":
            # generating arap boundary, boundary uvs, and uvs
            print("Using ARAP Condition")
            bnd, bnd_uv, uv = self.arap_ic()
        elif initial_condition == "harmonic":
            # generating harmonic boundary, boundary uvs, and uvs
            print("Using Harmonic Condition")
            bnd, bnd_uv, uv = self.harmonic_ic()
        elif initial_condition == "ordered":
            print("Using Ordered Condition")
            bnd, bnd_uv, uv = self.ordered_ic()

        self.vertices = self.vertices.astype(np.float64)
        self.triangles = self.triangles.astype(np.int64)
        uv = uv.astype(np.float64)
        bnd = bnd.astype(np.int64)
        bnd_uv = bnd_uv.astype(np.float64)

        uv = self.orient_uvs(
            uv
        )  # Enables the UVs to be oriented correctly for the slim optimization

        energies = []

        # initializing SLIM with Symmetric Dirichlet Distortion Energy (isometric)
        print("Symmetric Dirichlet Distortion Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=uv,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(
            slim, uv, iterations=30
        )  # 30 iterations
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("Log ARAP Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_LOG_ARAP,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("Symmetric Dirichlet Distortion Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("Conformal Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_CONFORMAL,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("Log ARAP Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_LOG_ARAP,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("ARAP Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_ARAP,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("Symmetric Dirichlet Distortion Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs, iterations=30)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        print("Exponential Symmetric Dirichlet Distortion Energy")
        slim = igl.SLIM(
            self.vertices,
            self.triangles,
            v_init=slim_uvs,
            b=bnd,
            bc=bnd_uv,
            energy_type=igl.SLIM_ENERGY_TYPE_EXP_SYMMETRIC_DIRICHLET,
            soft_penalty=0,
        )
        slim_uvs, energies_ = self.slim_optimization(slim, slim_uvs, iterations=10)
        energies.extend(list(energies_))
        print_errors(slim_uvs)

        # rescale slim uvs
        slim_uvs = slim_uvs * self.stretch_factor

        return slim_uvs, np.array(energies)

    @staticmethod
    def normalize(uv):
        uv_min = np.min(uv, axis=0)
        uv_max = np.max(uv, axis=0)
        new_uv = (uv - uv_min) / (uv_max - uv_min)
        return new_uv

    def save_img(self, uv):
        output_directory = os.path.dirname(self.output_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.output_obj))
        image_path = os.path.join(output_directory, f"{base_file_name}_0.png")
        min_x, min_y = np.min(uv, axis=0)
        shifted_coords = uv - np.array([min_x, min_y])
        max_x, max_y = np.max(shifted_coords, axis=0)
        # Create a mask image of the determined size
        image_size = (int(round(max_y)) + 1, int(round(max_x)) + 1)
        print(f"Image size: {image_size}")

        mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
        triangles = [[shifted_coords[t_i] for t_i in t] for t in self.triangles]
        for triangle in triangles:
            triangle = np.array(triangle).astype(np.int32)
            try:
                cv2.fillPoly(mask, [triangle], 255)
            except:
                pass
        mask = mask[::-1, :]
        cv2.imwrite(image_path, mask)

    def save_mtl(self):
        output_directory = os.path.dirname(self.output_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.output_obj))

        new_file_path = os.path.join(output_directory, f"{base_file_name}.mtl")
        content = f"""# Material file generated by ThaumatoAnakalyptor
        newmtl default
        Ka 1.0 1.0 1.0
        Kd 1.0 1.0 1.0
        Ks 0.0 0.0 0.0
        illum 2
        d 1.0
        map_Kd {base_file_name}_0.png
        """

        with open(new_file_path, "w") as file:
            file.write(content)

    def save_obj(self, uv):
        output_directory = os.path.dirname(self.output_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.output_obj))
        obj_path = os.path.join(output_directory, f"{base_file_name}.obj")
        normalized_uv = self.normalize(uv)
        slim_uvs = np.zeros((self.triangles.shape[0], 3, 2), dtype=np.float64)
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                slim_uvs[t, v, :] = normalized_uv[self.triangles[t, v]]
        slim_uvs = slim_uvs.reshape(-1, 2)
        self.mesh.triangles = o3d.utility.Vector3iVector(
            np.array(self.triangles).astype(np.int32)
        )
        self.mesh.triangle_uvs = o3d.utility.Vector2dVector(slim_uvs)
        o3d.io.write_triangle_mesh(obj_path, self.mesh)

    def stretch_triangle(self, triangle_3d, triangle_2d):
        q1, q2, q3 = triangle_3d

        s1, t1 = triangle_2d[0]
        s2, t2 = triangle_2d[1]
        s3, t3 = triangle_2d[2]

        A = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2  # 2d area
        if A != 0:
            Ss = (q1 * (t2 - t3) + q2 * (t3 - t1) + q3 * (t1 - t2)) / (2 * A)
            St = (q1 * (s3 - s2) + q2 * (s1 - s3) + q3 * (s2 - s1)) / (2 * A)
        else:
            Ss = 0
            St = 0

        a = np.dot(Ss, Ss)
        b = np.dot(Ss, St)
        c = np.dot(St, St)

        G = sqrt(((a + c) + sqrt((a - c) ** 2 + 4 * b**2)) / 2)

        L2 = sqrt((a + c) / 2)

        ab = np.linalg.norm(q2 - q1)
        bc = np.linalg.norm(q3 - q2)
        ca = np.linalg.norm(q1 - q3)
        s = (ab + bc + ca) / 2
        area = sqrt(s * (s - ab) * (s - bc) * (s - ca))  # 3d area

        return L2, G, area, abs(A)

    def align_uv_map(self, uv, volume_axis, rotate_angle):
        # Randomly sample 500 points or all points, whichever is smaller
        num_points = min(500, uv.shape[0])
        sampled_indices = np.random.choice(uv.shape[0], num_points, replace=False)
        sampled_uv = uv[sampled_indices]

        # Initialize variables to store the best angle and highest correlation
        best_angle = 0
        highest_correlation = -1

        # Iterate over trial angles (every half degree)
        for angle in range(0, 360, 1):
            # Rotate the UV map by the trial angle
            theta = radians(angle)
            rotation_matrix = np.array(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
            )
            rotated_uv = np.dot(sampled_uv, rotation_matrix.T)

            # Measure the Pearson correlation between -V and the specified volume axis
            negative_v = -rotated_uv[:, 1]
            correlation, _ = pearsonr(negative_v, volume_axis[sampled_indices])

            # Update the best angle if the current correlation is higher
            if correlation > highest_correlation:
                highest_correlation = correlation
                best_angle = angle

        # Adjust the best angle by adding 180 degrees
        best_angle = (best_angle + rotate_angle) % 360

        # Rotate the entire UV map by the best angle
        theta = radians(best_angle)
        rotation_matrix = np.array(
            [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        )
        rotated_uv = np.dot(uv, rotation_matrix.T)

        print(
            f"Best alignment angle: {best_angle} degrees with correlation: {highest_correlation:.3f}"
        )

        return rotated_uv, best_angle, highest_correlation

    def stretch_metrics(self, uv):
        if len(uv.shape) == 2:
            temp = uv.copy()
            uv = np.zeros((self.triangles.shape[0], 3, 2), dtype=np.float64)
            for t in range(self.triangles.shape[0]):
                for v in range(self.triangles.shape[1]):
                    uv[t, v, :] = temp[self.triangles[t, v]]

        linf_all = np.zeros(self.triangles.shape[0])
        area_all = np.zeros(self.triangles.shape[0])
        area2d_all = np.zeros(self.triangles.shape[0])
        per_triangle_area = np.zeros(self.triangles.shape[0])

        nominator = 0
        for t in range(self.triangles.shape[0]):
            t3d = [
                self.vertices[self.triangles[t, i]]
                for i in range(self.triangles.shape[1])
            ]
            t2d = [uv[t, i] for i in range(self.triangles.shape[1])]

            l2, linf, area, area2d = self.stretch_triangle(t3d, t2d)

            linf_all[t] = linf
            area_all[t] = area
            area2d_all[t] = area2d
            nominator += (l2**2) * area_all[t]

        l2_mesh = sqrt(nominator / np.sum(area_all))
        linf_mesh = np.max(linf_all)

        alpha = area_all / np.sum(area_all)
        beta = area2d_all / np.sum(area2d_all)

        for t in range(self.triangles.shape[0]):
            if alpha[t] > beta[t]:
                per_triangle_area[t] = 1 - beta[t] / alpha[t]
            else:
                per_triangle_area[t] = 1 - alpha[t] / beta[t]

        return l2_mesh, linf_mesh, np.mean(per_triangle_area)


def main():
    path = f""

    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Add UV coordinates using the flatboi script for SLIM to a ThaumatoAnakalyptor papyrus surface mesh (.obj). output mesh has additional "_flatboi.obj" in name.'
    )
    parser.add_argument("--path", type=str, help="Path of .obj Mesh", default=path)
    parser.add_argument("--iter", type=int, help="Max number of iterations.")
    parser.add_argument(
        "--ic",
        type=str,
        help="Initial condition for SLIM. Options: original, arap, harmonic",
        default="arap",
    )
    parser.add_argument(
        "--axis",
        type=str,
        help="Volume axis for alignment. Options: x, y, z",
        default="z",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        help="Angle to add to the best alignment angle. Default is 180 degrees.",
        default=180,
    )

    # Take arguments back over
    args = parser.parse_args()
    path = args.path

    # Check if the input file exists and is a .obj file
    if not os.path.exists(path):
        print(f"Error: The file '{path}' does not exist.")
        exit(1)

    if not path.lower().endswith(".obj"):
        print(f"Error: The file '{path}' is not a .obj file.")
        exit(1)

    assert args.iter > 0, "Max number of iterations should be positive."

    # Get the directory of the input file
    input_directory = os.path.dirname(path)
    # Filename for the energies file
    energies_file = os.path.join(input_directory, "energies_flatboi.txt")

    print(f"Adding UV coordinates to mesh {path}")

    flatboi = Flatboi(path, args.iter)
    harmonic_uvs, harmonic_energies = flatboi.slim(initial_condition=args.ic)

    # Align the UV map
    volume_axis = flatboi.vertices[:, {"x": 0, "y": 1, "z": 2}[args.axis.lower()]]
    aligned_uvs, best_angle, highest_correlation = flatboi.align_uv_map(
        harmonic_uvs, volume_axis, args.rotate
    )
    print(
        f"Best alignment angle: {best_angle} degrees with correlation: {highest_correlation:.3f}"
    )

    flatboi.save_img(aligned_uvs)
    flatboi.save_obj(aligned_uvs)
    print_array_to_file(harmonic_energies, energies_file)
    flatboi.save_mtl()


if __name__ == "__main__":
    main()
