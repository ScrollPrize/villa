import igl
import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from math import sqrt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timezone

Image.MAX_IMAGE_PIXELS = None

# --- W&B minimal integration -------------------------------------------------
class WBLogger:
    """
    Thin wrapper so you can call:
        logger = WBLogger(obj_path, n_iters)
        logger.log_initial_stretch(stretch_init)
        logger.log_energy(sym_dirichlet, it)
        logger.log_final_stretch(stretch_final)
        logger.finish()
    If wandb isn't installed/available, this becomes a no-op.
    """
    def __init__(self, obj_path, n_iters):
        self.enabled = False
        try:
            import wandb  # noqa: F401
            self.wandb = wandb
        except Exception:
            self.wandb = None
            return

        run_name = self._derive_names(obj_path)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        project = "flattening"
        entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_ORG")
        print(f"Entity {entity}")
        try:
            self.run = self.wandb.init(
                project=project,
                entity=entity,            # set via env: WANDB_ENTITY=your-org
                name=f"{run_name}_{ts}",
                config={
                    "obj_path": str(obj_path),
                    "iterations": int(n_iters),
                },
            )
            # make 'iter' the step metric for energy curves
            self.wandb.define_metric("iter")
            self.wandb.define_metric("symmetric_dirichlet", step_metric="iter")
            self.wandb.define_metric("stretch/*", step_metric="iter")
            self.enabled = True
        except Exception:
            # If init fails, silently disable logging
            self.enabled = False

    @staticmethod
    def _derive_names(obj_path):
        """
        Run:     Volume + OBJ filename stem.
        Fallbacks: parent folder name -> 'flatboi'.
        """
        p = Path(obj_path).resolve()
        volume = None
        for parent in p.parents:
            if parent.name.endswith(".volpkg"):
                volume = parent.stem   # strip '.volpkg'
                break
        if not volume:
            volume = p.parent.name or "flatboi"

        run_name = volume + "_" + p.stem
        return run_name

    def log_dict(self, d: dict):
        if self.enabled and d:
            try:
                self.wandb.log(d)
            except Exception:
                pass

    def log_energy(self, value, step):
        if self.enabled and value is not None:
            try:
                self.wandb.log({"symmetric_dirichlet": float(value), "iter": int(step)})
            except Exception:
                pass

    def finish(self):
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception:
                pass
# ---------------------------------------------------------------------------

def print_array_to_file(array, file_path):
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write each element of the array to the file
        for element in array:
            file.write(str(element) + '\n')

class Flatboi:
    input_obj: str
    max_iter: int
    def __init__(self, obj_path: str, max_iter: int):
        self.input_obj = obj_path
        self.max_iter = max_iter
        self.read_mesh()

    def read_mesh(self):
        self.mesh = o3d.io.read_triangle_mesh(self.input_obj)
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(self.mesh.triangles, dtype=np.int32)
        self.vc3d_uvs = np.asarray(self.mesh.triangle_uvs, dtype=np.float64) # read VC3D flattening

    def generate_boundary(self):
        return igl.boundary_loop(self.triangles)
    
    def harmonic_ic(self):
        bnd = self.generate_boundary()
        bnd_uv = igl.map_vertices_to_circle(self.vertices, bnd)
        uv = igl.harmonic(self.vertices, self.triangles, bnd, bnd_uv, 1)
        return bnd, bnd_uv, uv
    
    def original_ic(self):
        uv = np.zeros((self.vertices.shape[0], 2), dtype=np.float64)
        uvs = self.vc3d_uvs.reshape((self.triangles.shape[0], self.triangles.shape[1], 2))
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                uv[self.triangles[t,v]] = uvs[t,v]
        
        bnd = self.generate_boundary()
        bnd_uv = np.zeros((bnd.shape[0], 2), dtype=np.float64)

        for i in range(bnd.shape[0]):
            bnd_uv[i] = uv[bnd[i]]

        return bnd, bnd_uv, uv
    
    def slim(self, initial_condition='original', logger=None):
        if initial_condition == 'original':
            bnd, bnd_uv, uv = self.original_ic()
            l2_mean, l2_median, linf, area_err = self.stretch_metrics(uv)
            print(f"Starting metrics from VC3D flattening -- "
                  f"L2(mean): {l2_mean:.5f}, L2(median): {l2_median:.5f}, "
                  f"Linf: {linf:.5f}, Area Error: {area_err:.5f}")
            # log at iter=0 so curves start at the initial condition
            if logger:
                logger.log_dict({
                    "iter": 0,
                    "stretch/l2_mean": float(l2_mean),
                    "stretch/l2_median": float(l2_median),
                    "stretch/linf": float(linf),
                    "stretch/area_error": float(area_err),
                })

        elif initial_condition == 'harmonic':
            bnd, bnd_uv, uv = self.harmonic_ic()

        slim = igl.SLIM(self.vertices, self.triangles, v_init=uv, b=bnd, bc=bnd_uv, energy_type=igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, soft_penalty=0)

        energies = np.empty(self.max_iter + 1, dtype=float)
        energies[0] = slim.energy()
        # log initial energy (iter=0)
        if logger:
            logger.log_energy(energies[0], step=0)

        pbar      = tqdm(range(self.max_iter), desc="Optimisation")

        for i in pbar:
            slim.solve(1)

            energy = slim.energy()
            energies[i + 1] = energy

            # put the current energy in the bar’s tail
            # {:.3e} → scientific notation; tweak as you like
            pbar.set_postfix(E=f"{energy:.3e}")
            if logger:
                logger.log_energy(energy, step=i+1)
                # log stretch every 20 iterations
                if ((i + 1) % 20) == 0:
                    l2m, l2med, linf_v, area_v = self.stretch_metrics(slim.vertices())
                    logger.log_dict({
                        "iter": i + 1,
                        "stretch/l2_mean": float(l2m),
                        "stretch/l2_median": float(l2med),
                        "stretch/linf": float(linf_v),
                        "stretch/area_error": float(area_v),
                    })
        # ensure we also log the final stretch if not already on a 20-step boundary
        if (self.max_iter % 20) != 0:
            l2_mean_f, l2_median_f, linf_f, area_err_f = self.stretch_metrics(slim.vertices())
            if logger:
                logger.log_dict({
                    "iter": self.max_iter,
                    "stretch/l2_mean": float(l2_mean_f),
                    "stretch/l2_median": float(l2_median_f),
                    "stretch/linf": float(linf_f),
                    "stretch/area_error": float(area_err_f),
                })
        else:
            # already logged at the last iteration due to %20==0
            l2_mean_f, l2_median_f, linf_f, area_err_f = self.stretch_metrics(slim.vertices())

        print(f"Stretch metrics SLIM from {initial_condition} -- "
              f"L2(mean): {l2_mean_f:.5f}, L2(median): {l2_median_f:.5f}, "
              f"Linf: {linf_f:.5f}, Area Error: {area_err_f:.5f}")
        
        return slim.vertices(), energies
    
    @staticmethod
    def shift(uv):
        uv_min = np.min(uv, axis=0)
        new_uv = (uv - uv_min)
        return new_uv
    
    def save_obj(self, uv):
        input_directory = os.path.dirname(self.input_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.input_obj))
        obj_path = os.path.join(input_directory, f"{base_file_name}_flatboi.obj")
        shifted_uv = self.shift(uv)
        slim_uvs = np.zeros((self.triangles.shape[0],3,2), dtype=np.float64)
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                slim_uvs[t,v,:] = shifted_uv[self.triangles[t,v]]
        slim_uvs = slim_uvs.reshape(-1,2)
        self.mesh.triangle_uvs = o3d.utility.Vector2dVector(slim_uvs)
        o3d.io.write_triangle_mesh(obj_path, self.mesh)

    def stretch_triangle(self, triangle_3d, triangle_2d):
        q1, q2, q3 = triangle_3d

        s1, t1 = triangle_2d[0]
        s2, t2 = triangle_2d[1]
        s3, t3 = triangle_2d[2]

        A = ((s2-s1)*(t3-t1)-(s3-s1)*(t2-t1))/2 # 2d area
        Ss = (q1*(t2-t3)+q2*(t3-t1)+q3*(t1-t2))/(2*A)
        St = (q1*(s3-s2)+q2*(s1-s3)+q3*(s2-s1))/(2*A)
        a = np.dot(Ss,Ss)
        b = np.dot(Ss,St)
        c = np.dot(St,St)

        G = sqrt(((a+c)+sqrt((a-c)**2+4*b**2))/2)

        L2 = sqrt((a+c)/2)

        ab = np.linalg.norm(q2-q1)
        bc = np.linalg.norm(q3-q2)
        ca = np.linalg.norm(q1-q3)
        s = (ab+bc+ca)/2
        area = sqrt(s*(s-ab)*(s-bc)*(s-ca)) # 3d area

        
        return L2, G, area, abs(A)
    
    @staticmethod
    def weighted_median(data, weights):
        # Sort data and weights by the data values
        sorted_indices = np.argsort(data)
        sorted_data = np.array(data)[sorted_indices]
        sorted_weights = np.array(weights)[sorted_indices]

        # Compute the cumulative sum of weights
        cumsum_sorted_weights = np.cumsum(sorted_weights)
        
        # Get the sum of weights
        total_weight = cumsum_sorted_weights[-1]
        
        # Find the index where the cumulative sum of weights equals or exceeds half the total weight
        half_weight_index = np.where(cumsum_sorted_weights >= total_weight / 2)[0][0]
        
        # Return the corresponding data value
        return sorted_data[half_weight_index]

    def stretch_metrics(self, uv):
        if len(uv.shape) == 2:
            temp = uv.copy()
            uv = np.zeros((self.triangles.shape[0],3,2), dtype=np.float64)
            for t in range(self.triangles.shape[0]):
                for v in range(self.triangles.shape[1]):
                    uv[t,v,:] = temp[self.triangles[t,v]]

        l2_all = np.zeros(self.triangles.shape[0])
        linf_all = np.zeros(self.triangles.shape[0])
        area_all = np.zeros(self.triangles.shape[0])
        area2d_all = np.zeros(self.triangles.shape[0])
        per_triangle_area = np.zeros(self.triangles.shape[0])

        nominator = 0
        for t in range(self.triangles.shape[0]):
            t3d = [self.vertices[self.triangles[t,i]] for i in range(self.triangles.shape[1])]
            t2d = [uv[t,i] for i in range(self.triangles.shape[1])]

            l2, linf, area, area2d = self.stretch_triangle(t3d, t2d)

            linf_all[t] = linf
            area_all[t] = area
            area2d_all[t] = area2d
            l2_all[t] = l2**2
            nominator += l2_all[t]*area_all[t]
            
        l2_mesh_mean = sqrt( nominator / np.sum(area_all))
        l2_mesh_median = self.weighted_median(l2_all, area_all)
        linf_mesh = np.max(linf_all)

        alpha = area_all/np.sum(area_all)
        beta = area2d_all/np.sum(area2d_all)

        for t in range(self.triangles.shape[0]):
            if alpha[t] > beta[t]:
                per_triangle_area[t] = 1 - beta[t]/alpha[t]
            else:
                per_triangle_area[t] = 1 - alpha[t]/beta[t]
        
        area_err = float(np.mean(per_triangle_area))
        return l2_mesh_mean, l2_mesh_median, linf_mesh, area_err
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reflatten .obj")
    parser.add_argument('input', type=str, help='Path to .obj to reflatten.')
    parser.add_argument('iter', type=int, help='Max number of iterations.')

    args = parser.parse_args()

    # Check if the input file exists and is a .obj file
    if not os.path.exists(args.input):
        print(f"Error: The file '{args.input}' does not exist.")
        exit(1)

    if not args.input.lower().endswith('.obj'):
        print(f"Error: The file '{args.input}' is not a .obj file.")
        exit(1)

    assert args.iter > 0, "Max number of iterations should be positive."

    # Get the directory of the input file
    input_directory = os.path.dirname(args.input)
    # Filename for the energies file
    energies_file = os.path.join(input_directory, 'energies_flatboi.txt')
    try:
        # Init W&B logger (no-op if wandb not installed or init fails)
        _logger = WBLogger(args.input, args.iter)

        flatboi = Flatboi(args.input, args.iter)
        original_uvs, original_energies = flatboi.slim(initial_condition='original', logger=_logger)
        print(f"Symmetric Dirichlet Energy per iter: {original_energies}")

        # Optionally persist energies to a file near the mesh
        input_directory = os.path.dirname(args.input)
        base_file_name, _ = os.path.splitext(os.path.basename(args.input))
        energies_file = os.path.join(input_directory, f"{base_file_name}_energies_flatboi.txt")
        np.savetxt(energies_file, original_energies, fmt="%.10g")

        flatboi.save_obj(original_uvs)

        _logger.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


