import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QComboBox, QPushButton, QLabel, QProgressDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QApplication, QGraphicsPathItem
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject, pyqtSlot, Qt, QEvent, QPointF
from PyQt5.QtGui import QImage, QPainter, QPainterPath, QPen, QColor
import pyqtgraph as pg
import pyqtgraph.exporters
import cv2
import h5py
import zarr
import pickle
import time
import os
import ast
import sys
from tqdm import tqdm
# remove the lib first
try:
    del sys.modules['scroll_graph_util']
except KeyError:
    print("Module 'scroll_graph_util' not found in sys.modules; continuing.")
from scroll_graph_util import compute_mean_windings_precomputed, load_xyz_from_file, umbilicus_xy_at_z

########################################
# Utility Functions
########################################

def load_graph_pkl(graph_pkl_path, use_h5=False):
    """
    Load or build a slimmed ScrollGraph, retaining only 'sample_points' and 'centroid' per node.
    If use_h5=True, creates/loads an even smaller '_tiny.pkl' with only keys and centroids.
    If a '<base>_small.pkl' or '<base>_tiny.pkl' exists alongside the original, load from that.
    Otherwise, unpickle the full graph once, trim and downcast nodes, save a small pickle, and return it.
    """
    import scroll_graph_util
    base, ext = os.path.splitext(graph_pkl_path)
    
    if use_h5:
        # For H5 usage, we only need node keys and centroids
        tiny_pkl = base + '_tiny' + ext
        # Load pre-tiny if available
        if os.path.exists(tiny_pkl):
            print(f"[TinyLoad] Loading tiny graph from {tiny_pkl}", file=sys.stderr)
            nodes = pickle.load(open(tiny_pkl, 'rb'))
            graph = scroll_graph_util.ScrollGraph(0, None)
            graph.nodes = nodes
            print(f"[TinyLoad] Loaded tiny graph with {len(nodes)} nodes", file=sys.stderr)
            return graph
        # Else build tiny graph
        print(f"[TinyLoad] Building tiny graph from {graph_pkl_path}", file=sys.stderr)
        class TinyScrollGraph(scroll_graph_util.ScrollGraph):
            def __setstate__(self, state):
                nodes = state.get('nodes', {}) or {}
                total = len(nodes)
                print(f"[TinyLoad] Trimming {total} nodes to keys+centroids only...", file=sys.stderr)
                tiny_nodes = {}
                for key, data in tqdm(nodes.items(), desc="[TinyLoad] Trimming nodes", total=total, file=sys.stderr):
                    cent = data.get('centroid')
                    cent16 = np.asarray(cent, dtype=np.float16) if cent is not None else None
                    tiny_nodes[key] = {'centroid': cent16}
                self.nodes = tiny_nodes
                print(f"[TinyLoad] Tiny graph ready with {len(tiny_nodes)} nodes", file=sys.stderr)
        class TinyUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'scroll_graph_util' and name in ('ScrollGraph', 'Graph'):
                    return TinyScrollGraph
                return super().find_class(module, name)
        with open(graph_pkl_path, 'rb') as f:
            print(f"[TinyLoad] Unpickling full graph...", file=sys.stderr)
            unp = TinyUnpickler(f)
            tiny_graph = unp.load()
            print(f"[TinyLoad] Full graph unpickled and trimmed to tiny.", file=sys.stderr)
        # Save tiny nodes dict
        try:
            with open(tiny_pkl, 'wb') as fout:
                pickle.dump(tiny_graph.nodes, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[TinyLoad] Saved tiny graph nodes to {tiny_pkl}", file=sys.stderr)
        except Exception as e:
            print(f"[TinyLoad] Warning: could not save tiny pickle: {e}", file=sys.stderr)
        return tiny_graph
    else:
        # Original small pkl logic for non-H5 usage
        small_pkl = base + '_small' + ext
        # Load pre-slimmed if available
        if os.path.exists(small_pkl):
            print(f"[SlimLoad] Loading slimmed graph from {small_pkl}", file=sys.stderr)
            nodes = pickle.load(open(small_pkl, 'rb'))
            graph = scroll_graph_util.ScrollGraph(0, None)
            graph.nodes = nodes
            print(f"[SlimLoad] Loaded slim graph with {len(nodes)} nodes", file=sys.stderr)
            return graph
        # Else build slim graph
        print(f"[SlimLoad] Building slimmed graph from {graph_pkl_path}", file=sys.stderr)
        class SlimScrollGraph(scroll_graph_util.ScrollGraph):
            def __setstate__(self, state):
                nodes = state.get('nodes', {}) or {}
                total = len(nodes)
                print(f"[SlimLoad] Trimming {total} nodes...", file=sys.stderr)
                slim_nodes = {}
                for key, data in tqdm(nodes.items(), desc="[SlimLoad] Trimming nodes", total=total, file=sys.stderr):
                    sp = data.get('sample_points')
                    sp16 = np.asarray(sp, dtype=np.float16) if sp is not None else None
                    cent = data.get('centroid')
                    cent16 = np.asarray(cent, dtype=np.float16) if cent is not None else None
                    slim_nodes[key] = {'sample_points': sp16, 'centroid': cent16}
                self.nodes = slim_nodes
                print(f"[SlimLoad] Slim graph ready with {len(slim_nodes)} nodes", file=sys.stderr)
        class SlimUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'scroll_graph_util' and name in ('ScrollGraph', 'Graph'):
                    return SlimScrollGraph
                return super().find_class(module, name)
        with open(graph_pkl_path, 'rb') as f:
            print(f"[SlimLoad] Unpickling full graph...", file=sys.stderr)
            unp = SlimUnpickler(f)
            slim_graph = unp.load()
            print(f"[SlimLoad] Full graph unpickled and trimmed.", file=sys.stderr)
        # Save slim nodes dict
        try:
            with open(small_pkl, 'wb') as fout:
                pickle.dump(slim_graph.nodes, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[SlimLoad] Saved slim graph nodes to {small_pkl}", file=sys.stderr)
        except Exception as e:
            print(f"[SlimLoad] Warning: could not save slim pickle: {e}", file=sys.stderr)
        return slim_graph

########################################
# Helper Functions
########################################

def brush_for_winding(w, r, g, b):
    """
    Given a winding value w (assumed numeric), return a pyqtgraph brush
    using a three-color scheme: winding 1 -> red, 2 -> green, 3 -> blue,
    4 -> red, etc. Handles negative values properly.
    """
    # Round to nearest integer.
    w_int = int(round(w / 360.0))
    # Handle negative values
    mod = w_int % 3
    # Python's modulo handles negatives differently, so we adjust
    if mod < 0:
        mod += 3
    if mod == 1:
        return g   # green
    elif mod == 2:
        return b   # blue
    else:  # mod == 0
        return r   # red
    
# def vectorized_brush_for_winding(w_array, brush_red, brush_green, brush_blue):
#     """
#     Vectorized version of brush_for_winding.
#     Given an array of winding values, compute the integer rounded value,
#     then assign brushes based on mod 3:
#     mod == 1 -> brush_green
#     mod == 2 -> brush_blue
#     mod == 0 -> brush_red
#     """
#     # Round w/360 to the nearest integer.
#     w_int = np.rint(w_array / 360.0).astype(np.int64)
#     mod = w_int % 3
#     # Create an empty array of objects (brushes).
#     result = np.empty(mod.shape, dtype=object)
#     result[mod == 1] = brush_green
#     result[mod == 2] = brush_blue
#     result[mod == 0] = brush_red
#     return result

def vectorized_brush_for_winding(w_array, brush_red, brush_green, brush_blue):
    """
    Computes a QBrush for each winding value in w_array by interpolating
    along a manually defined gradient that cycles from red -> green -> blue -> red.
    The gradient is divided into 12 equal intervals (each corresponding to 30°).
    
    Parameters:
      w_array: numpy array of winding values in degrees.
      brush_red, brush_green, brush_blue: ignored in this implementation.
    
    Returns:
      A list of QBrush objects with the same shape as w_array.
    """
    # Define a palette of 12 colors for angles 0, 30, 60, ..., 330 degrees.
    brushes = []
    steps = 36
    d_angle = 360.0 / steps
    for i in range(steps):
        angle = i * d_angle  # in degrees
        if angle < 120:
            # Transition: Red (255, 0, 0) to Green (0, 255, 0)
            f = angle / 120.0  # f increases from 0 to 1
            r = 255 * (1 - f)
            g = 255 * f
            b = 0
        elif angle < 240:
            # Transition: Green (0, 255, 0) to Blue (0, 0, 255)
            f = (angle - 120) / 120.0
            r = 0
            g = 255 * (1 - f)
            b = 255 * f
        else:
            # Transition: Blue (0, 0, 255) to Red (255, 0, 0)
            f = (angle - 240) / 120.0
            r = 255 * f
            g = 0
            b = 255 * (1 - f)
        brushes.append(pg.mkBrush(r, g, b))
    
    # Map winding values (in degrees) to a normalized index in [0, steps).
    w_int = np.rint(w_array / (3 * d_angle)).astype(np.int64)
    mod = w_int % steps
    # Handle negative values
    mod = np.where(mod < 0, mod + steps, mod)
    # Create an empty array of objects (brushes).
    result = np.empty(mod.shape, dtype=object)
    for i in range(steps):
        result[mod == i] = brushes[i]
    
    return result

def save_high_res_widget(widget, save_path, fixed_width=5000):
    # Get the widget's current size.
    original_size = widget.size()
    orig_width = original_size.width()
    orig_height = original_size.height()
    
    # Compute scale factor based on the desired fixed width.
    scale_factor = fixed_width / orig_width
    new_width = fixed_width
    new_height = int(orig_height * scale_factor)
    
    # Create a QImage with the fixed width and proportional height.
    image = QImage(new_width, new_height, QImage.Format_ARGB32)
    image.fill(Qt.white)  # Fill background with white (or any color you prefer)
    
    # Render the widget into the QImage using QPainter.
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(scale_factor, scale_factor)
    widget.render(painter)
    painter.end()
    
    image.save(save_path)

########################################
# Worker Classes
########################################

# --- XY Loader Worker ---
class OmeZarrLoaderWorker(QThread):
    slice_loaded = pyqtSignal(np.ndarray)

    def __init__(self, ome_zarr_path, z_index, resolution):
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.z_index = z_index
        self.resolution = resolution

    def run(self):
        try:
            store = zarr.open(self.ome_zarr_path, mode='r')
            # Load at selected resolution level
            dset = store[str(self.resolution)]
            # Calculate the correct z-index for this pyramid level
            try:
                res = int(self.resolution)
            except Exception:
                res = 0
            if res > 0:
                # assume each level downscales by 2**res
                z_idx = int(self.z_index / (2 ** res))
            else:
                z_idx = int(self.z_index)
            # Clamp to valid range
            z_dim = dset.shape[0]
            if z_idx < 0:
                z_idx = 0
            elif z_idx >= z_dim:
                z_idx = z_dim - 1
            # Extract the slice
            image_slice = dset[z_idx]
        except Exception as e:
            print(f"Error loading z slice at index {getattr(self, 'z_index', None)}: {e}")
            image_slice = np.zeros((512,512,3), dtype=np.uint8)
        # Emit the loaded (or placeholder) slice
        self.slice_loaded.emit(image_slice)

# --- XZ Loader Worker using Umbilicus Data ---
class OmeZarrXZLoaderWorker(QThread):
    xz_slice_loaded = pyqtSignal(np.ndarray)

    def __init__(self, ome_zarr_path, resolution, finit_center_value, umbilicus_path):
        """
        :param ome_zarr_path: Path to the OME-Zarr store.
        :param resolution: Pyramid level in the Zarr store (integer string key).
        :param finit_center_value: Rotation angle (in degrees) for the XZ view.
        :param umbilicus_path: Path to the umbilicus .txt file.
        """
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.resolution = resolution
        self.finit_center_value = finit_center_value
        self.umbilicus_path = umbilicus_path

    def run(self):
        try:
            store = zarr.open(self.ome_zarr_path, mode='r')
            # Load at selected resolution level
            dset = store[str(self.resolution)]
            z_dim, y_dim, x_dim = dset.shape  # shape: (Z, Y, X)

            # Determine downscaling factor for this level
            try:
                level = int(self.resolution)
            except Exception:
                level = 0
            factor = 2 ** level if level > 0 else 1

            # Load umbilicus data and map into this resolution's coordinate space
            if self.umbilicus_path is not None and os.path.exists(self.umbilicus_path):
                print(f"Loading umbilicus data from {self.umbilicus_path}")
                raw_data = load_xyz_from_file(self.umbilicus_path) - 500
                umbilicus_data = raw_data / factor
            else:
                # estimated center in (y,z,x) order
                umbilicus_data = np.array([[y_dim / 2, 0, x_dim / 2]])
                print(f"Umbilicus path {self.umbilicus_path} not found; using estimated umbilicus data.")
            # Build per-slice centers via interpolation
            centers = []
            for z_val in range(z_dim):
                pos = umbilicus_xy_at_z(umbilicus_data, z_val)
                centers.append(pos)
            centers = np.array(centers)  # shape: (z_dim, 2); column 0: x, column 1: y

            L = max(x_dim, y_dim) / 2.0
            line_positions = np.arange(-L, L+1)
            angle_rad = np.deg2rad(self.finit_center_value)

            xs = centers[:,0][:, None] + line_positions[None, :] * np.cos(angle_rad)
            ys = centers[:,1][:, None] + line_positions[None, :] * np.sin(angle_rad)
            xs_int = np.rint(xs).astype(int)
            ys_int = np.rint(ys).astype(int)
            xs_int = np.clip(xs_int, 0, x_dim - 1)
            ys_int = np.clip(ys_int, 0, y_dim - 1)

            z_idx = np.arange(z_dim)[:, None]
            xz_image = dset[z_idx, ys_int, xs_int]  # shape: (z_dim, num_samples)
            # xz_image = xz_image.T  # shape: (num_samples, z_dim)
        except Exception as e:
            print(f"Error loading XZ slice with finit center {self.finit_center_value}: {e}")
            xz_image = np.zeros((512,512), dtype=np.uint8)
        self.xz_slice_loaded.emit(xz_image)

# --- Persistent Overlay Worker ---
class PersistentScrollGraphWorker(QObject):
    # This worker now emits a tuple for XY overlay and XZ overlay respectively.
    overlay_points_computed = pyqtSignal(object)
    overlay_points_xz_computed = pyqtSignal(object)
    overlay_labels_computed = pyqtSignal(object)
    overlay_labels_computed_xz = pyqtSignal(object)

    def __init__(self, graph_pkl_path, umbilicus_data, unlabeled, h5_path=None, parent=None):
        super().__init__(parent)
        # Determine if we should use the tiny version based on h5_path
        use_h5 = h5_path is not None and h5_path.endswith(".h5")
        # Load the scroll graph (from pickle) once.
        self.scroll_graph = load_graph_pkl(graph_pkl_path, use_h5=use_h5)
        self.umbilicus_data = umbilicus_data
        self.UNLABELED = unlabeled
        self.overlay_point_nodes_indices = None
        self.overlay_point_nodes_indices_xz = None        
        self.inverse_indices = None
        self.inverse_indices_xz = None        
        self.close_mask = None
        self.close_mask_xz = None

    @pyqtSlot(np.ndarray, np.ndarray)
    def compute_labels(self, windings, windings_computed):
        # Compute the labels for the XY view.
        self.compute_labels_xy(windings, windings_computed)
        # Compute the labels for the XZ view.
        self.compute_labels_xz(windings, windings_computed)

    def compute_labels_xy(self, windings, windings_computed):
        try:
            # Get close nodes labels.
            close_windings = windings[self.close_mask]
            close_computed = windings_computed[self.close_mask]
            print(f"[Worker compute_labels_xy] Processing {len(close_windings)} close node windings")
            # Transfer to per-point windings with self.overlay_point_nodes_indices.
            overlay_windings = close_windings[self.overlay_point_nodes_indices]
            overlay_windings_computed = close_computed[self.overlay_point_nodes_indices]
            # Get per-point labels.
            overlay_windings, overlay_windings_computed = compute_mean_windings_precomputed(
                self.inverse_indices, overlay_windings, overlay_windings_computed, self.UNLABELED
            )
            print(f"[Worker compute_labels_xy] Computed mean windings for {len(overlay_windings)} unique points")
            self.overlay_labels_computed.emit((overlay_windings, overlay_windings_computed))
        except Exception as e:
            print("Error in PersistentScrollGraphWorker:", e)
            self.overlay_labels_computed.emit((np.empty((0, 1)), np.empty((0, 1))))

    def compute_labels_xz(self, windings, windings_computed):
        try:
            # Get close nodes labels.
            close_windings = windings[self.close_mask_xz]
            close_computed = windings_computed[self.close_mask_xz]
            # Transfer to per-point windings with self.overlay_point_nodes_indices.
            overlay_windings = close_windings[self.overlay_point_nodes_indices_xz]
            overlay_windings_computed = close_computed[self.overlay_point_nodes_indices_xz]
            # Get per-point labels.
            overlay_windings, overlay_windings_computed = compute_mean_windings_precomputed(
                self.inverse_indices_xz, overlay_windings, overlay_windings_computed, self.UNLABELED
            )
            self.overlay_labels_computed_xz.emit((overlay_windings, overlay_windings_computed))
        except Exception as e:
            print("Error in PersistentScrollGraphWorker compute_labels_xz:", e)
            self.overlay_labels_computed_xz.emit((np.empty((0, 1)), np.empty((0, 1))))
    

    @pyqtSlot(int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int)
    def compute_overlay(self, z_index, h5_path, labels, computed_labels, f_init, undeleted_nodes_indices, block_size):
        try:
            # Call get_points_XY to compute the XY overlay.
            overlay_points, self.overlay_point_nodes_indices, overlay_windings, overlay_windings_computed, \
            self.inverse_indices, winding, winding_computed, self.close_mask = self.scroll_graph.get_points_XY(
                z_index, h5_path, labels, computed_labels, f_init, undeleted_nodes_indices, self.UNLABELED, block_size
            )
            self.overlay_points_computed.emit(
                (overlay_points, overlay_windings, overlay_windings_computed, winding, winding_computed)
            )
        except Exception as e:
            print("Error in PersistentScrollGraphWorker compute overlay:", e)
            self.overlay_points_computed.emit(
                (np.empty((0, 3)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)))
            )

    @pyqtSlot(float, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int)
    def compute_overlay_xz(self, f_target, h5_path, labels, computed_labels, f_init, undeleted_nodes_indices, block_size):
        try:
            # Call get_points_XZ to compute the overlay for the XZ view.
            overlay_points, self.overlay_point_nodes_indices_xz, overlay_windings, overlay_windings_computed, \
            self.inverse_indices_xz, winding, winding_computed, self.close_mask_xz = self.scroll_graph.get_points_XZ(
                f_target, self.umbilicus_data, h5_path, labels, computed_labels, f_init, undeleted_nodes_indices, self.UNLABELED, block_size
            )
            self.overlay_points_xz_computed.emit(
                (overlay_points, overlay_windings, overlay_windings_computed, winding, winding_computed)
            )
        except Exception as e:
            print("Error in PersistentScrollGraphWorker compute_overlay_xz:", e)
            self.overlay_points_xz_computed.emit(
                (np.empty((0, 3)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)))
            )

########################################
# Main OME-Zarr View Window
########################################

class OmeZarrViewWindow(QMainWindow):
    # Signals to request overlay computation.
    overlay_request = pyqtSignal(int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int)
    overlay_request_xz = pyqtSignal(float, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int)
    label_request = pyqtSignal(np.ndarray, np.ndarray)
    # Signal to update labels in gui_main
    labels_updated_signal = pyqtSignal(dict, str)  # (node_updates, view_type)

    def __init__(self, graph_labels, solver, experiment_path, ome_zarr_path,
                 graph_pkl_path, h5_path, umbilicus_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OME-Zarr Views")
        self.setAttribute(Qt.WA_DeleteOnClose)

        os.makedirs("GraphLabelerViews", exist_ok=True)

        self.UNLABELED = -9999
        self.graph_labels = graph_labels
        self.solver = solver
        self.experiment_path = experiment_path
        # defer Zarr pyramid loading until resolution is selected
        self.ome_zarr_path = ome_zarr_path
        # metadata (dims, L) will be loaded upon resolution change
        self.z_dim = self.y_dim = self.x_dim = None
        self.L = None
        self.graph_pkl_path = graph_pkl_path
        self.h5_path = h5_path
        self.umbilicus_path = umbilicus_path
        # Compute umbilicus_data
        if self.umbilicus_path and os.path.exists(self.umbilicus_path):
            self.umbilicus_data = load_xyz_from_file(self.umbilicus_path) - 500
        else:
            self.umbilicus_data = np.array([[self.y_dim / 2, 0, self.x_dim / 2]])
        self.winding, overlay_point_nodes_indices = None, None
        self.f_init = np.array(self.solver.get_positions())[:, 1]
        self.undeleted_nodes_indices = np.array(self.solver.get_undeleted_indices())
        self.labels = np.ones(len(self.undeleted_nodes_indices)) * self.UNLABELED
        self.computed_labels = np.ones(len(self.undeleted_nodes_indices)) * self.UNLABELED
        self.red_brush = pg.mkBrush(255, 0, 0)
        self.green_brush = pg.mkBrush(0, 255, 0)
        self.blue_brush = pg.mkBrush(0, 0, 255)
        self.white_brush = pg.mkBrush(255, 255, 255)
        self.calc_brush_red   = pg.mkBrush(255, 50, 0, 100)
        self.calc_brush_green = pg.mkBrush(0, 255, 50, 100)
        self.calc_brush_blue  = pg.mkBrush(50, 0, 255, 100)
        
        # Per-point label storage for direct labeling in the view
        self.point_labels_xy = {}  # Dict mapping point index to label
        self.point_labels_xz = {}  # Dict mapping point index to label
        
        # Current label for painting
        self.current_label = 0
        
        # For pipette mode
        self.pipette_mode = False
        
        # For stroke-based painting
        self._stroke_backup = None
        self._current_path_xy = None
        self._current_path_xz = None
        
        # Drawing radius for brush
        self.drawing_radius = 10.0
        
        # For XY view updates.
        self.pending_z_center = None
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._trigger_z_slice_update)
        self.loader_worker_running = False  # Flag to indicate an update is in progress.

        self.debounce_timer_labels = QTimer(self)
        self.debounce_timer_labels.setSingleShot(True)
        self.debounce_timer_labels.timeout.connect(self._trigger_overlay_labels_update)
        
        # For XZ view updates.
        self.pending_finit_center = None
        self.finit_debounce_timer = QTimer(self)
        self.finit_debounce_timer.setSingleShot(True)
        self.finit_debounce_timer.timeout.connect(self._trigger_xz_slice_update)
        
        # Containers for storing last computed overlay windings for XZ view.
        self.last_overlay_windings_xz = None
        self.last_overlay_windings_computed_xz = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        # --- Settings bar: resolution dropdown ---
        settings_layout = QHBoxLayout()
        main_layout.addLayout(settings_layout)
        settings_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combobox = QComboBox()
        # populate integer-named pyramid levels from Zarr store
        try:
            store = zarr.open(self.ome_zarr_path, mode='r')
            levels = sorted(int(k) for k in store.array_keys() if k.isdigit())
        except Exception:
            levels = []
        self.available_resolutions = levels
        for lvl in levels:
            self.resolution_combobox.addItem(str(lvl), lvl)
        if levels:
            self.current_resolution = levels[0]
            self.resolution_combobox.setCurrentIndex(0)
        self.resolution_combobox.currentIndexChanged.connect(self.on_resolution_changed)
        settings_layout.addWidget(self.resolution_combobox)
        
        # Add label spinbox
        settings_layout.addStretch()
        settings_layout.addWidget(QLabel("Label:"))
        self.label_spinbox = QSpinBox()
        self.label_spinbox.setRange(-10000, 1000)
        self.label_spinbox.setValue(0)
        self.label_spinbox.valueChanged.connect(self.on_label_changed)
        settings_layout.addWidget(self.label_spinbox)
        
        # Add pipette mode indicator
        self.pipette_label = QLabel("Pipette: OFF")
        settings_layout.addWidget(self.pipette_label)
        
        # Add erase button
        self.erase_button = QPushButton("Erase Label")
        self.erase_button.clicked.connect(lambda: self.label_spinbox.setValue(self.UNLABELED))
        settings_layout.addWidget(self.erase_button)
        
        # Add drawing radius control
        settings_layout.addWidget(QLabel("Radius:"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setRange(1.0, 50.0)
        self.radius_spinbox.setValue(self.drawing_radius)
        self.radius_spinbox.setDecimals(1)
        self.radius_spinbox.valueChanged.connect(self.update_drawing_radius)
        settings_layout.addWidget(self.radius_spinbox)
        
        self.labels_status = QLabel("Custom labels: 0")
        settings_layout.addWidget(self.labels_status)
        
        # Add gradient coloring checkbox
        self.gradient_coloring_checkbox = QCheckBox("Gradient Coloring")
        self.gradient_coloring_checkbox.setChecked(True)
        self.gradient_coloring_checkbox.toggled.connect(self.update_views)
        settings_layout.addWidget(self.gradient_coloring_checkbox)
        
        # Add apply labels button
        self.apply_labels_button = QPushButton("Apply Labels to Graph")
        self.apply_labels_button.clicked.connect(self.apply_labels_to_graph)
        settings_layout.addWidget(self.apply_labels_button)
        
        # Load initial resolution
        self.on_resolution_changed(0)
        
        views_layout = QHBoxLayout()
        main_layout.addLayout(views_layout)
        
        # --- XY View ---
        self.xy_view = pg.ImageView()
        self.xy_view.ui.roiBtn.hide()
        self.xy_view.ui.menuBtn.hide()
        views_layout.addWidget(self.xy_view)
        
        # --- XZ View ---
        self.xz_view = pg.ImageView()
        self.xz_view.ui.roiBtn.hide()
        self.xz_view.ui.menuBtn.hide()
        views_layout.addWidget(self.xz_view)
        
        self.load_placeholder_images()
        
        # Create overlay items for stroke preview
        self._overlay = QGraphicsPathItem()
        pen = QPen(QColor(50, 150, 255, 150))
        pen.setWidthF(self.drawing_radius * 2)
        pen.setCapStyle(Qt.RoundCap)
        self._overlay.setPen(pen)
        self._overlay.setZValue(1000)
        self.xy_view.addItem(self._overlay)
        
        self._overlay_xz = QGraphicsPathItem()
        pen_xz = QPen(QColor(50, 150, 255, 150))
        pen_xz.setWidthF(self.drawing_radius * 2)
        pen_xz.setCapStyle(Qt.RoundCap)
        self._overlay_xz.setPen(pen_xz)
        self._overlay_xz.setZValue(1000)
        self.xz_view.addItem(self._overlay_xz)
        
        # Install event filters for labeling functionality
        self.xy_view.getView().scene().installEventFilter(self)
        self.xz_view.getView().scene().installEventFilter(self)
        
        # Umbilicus dot overlay.
        self.umbilicus_dot = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush('r'), size=20)
        self.xy_view.addItem(self.umbilicus_dot)

        # Setup persistent overlay worker.
        self.overlay_thread = QThread()
        self.persistent_overlay_worker = PersistentScrollGraphWorker(self.graph_pkl_path, self.umbilicus_data, self.UNLABELED, self.h5_path)
        self.persistent_overlay_worker.moveToThread(self.overlay_thread)
        self.overlay_thread.start()
        self.persistent_overlay_worker.overlay_points_computed.connect(self.on_overlay_points_computed)
        self.overlay_request.connect(self.persistent_overlay_worker.compute_overlay)
        # Connect the new XZ overlay signals.
        self.overlay_request_xz.connect(self.persistent_overlay_worker.compute_overlay_xz)
        self.persistent_overlay_worker.overlay_points_xz_computed.connect(self.on_overlay_points_xz_computed)
        self.persistent_overlay_worker.overlay_labels_computed.connect(self.on_overlay_labels_computed)
        self.persistent_overlay_worker.overlay_labels_computed_xz.connect(self.on_overlay_labels_computed_xz)
        self.label_request.connect(self.persistent_overlay_worker.compute_labels)

        # We'll store the last computed overlay windings to support dummy color updates.
        self.last_overlay_windings = None
        self.last_overlay_windings_computed = None

    def load_placeholder_images(self):
        red_image = np.zeros((512,512,3), dtype=np.uint8)
        red_image[..., 0] = 255
        self.xy_view.setImage(red_image)
        self.xz_view.setImage(red_image)
    
    # --- XY view update ---
    def update_z_slice_center(self, z_center_value):
        self.pending_z_center = z_center_value
        self.debounce_timer.stop()
        self.debounce_timer.start(5000)
    
    def _trigger_z_slice_update(self):
        print("Triggering z slice update.")
        if self.loader_worker_running:
            print("Update already in progress; waiting for it to finish.")
            return
        if self.pending_z_center is None:
            return
        z_index = int(self.pending_z_center * 4 - 500)
        self.current_z_index = z_index  # Store for later use.
        
        # Clear point labels for the new slice
        self.point_labels_xy.clear()
        
        self.loader_worker_running = True
        print(f"Loading OME-Zarr XY slice at index {z_index} (from z slice center {self.pending_z_center})")
        # Launch loader worker at selected resolution
        self.loader_worker = OmeZarrLoaderWorker(self.ome_zarr_path, z_index, self.current_resolution)
        self.loader_worker.slice_loaded.connect(self.on_slice_loaded)
        self.loader_worker.start()

        try:
            pos = umbilicus_xy_at_z(self.umbilicus_data, self.current_z_index)
            scale = 2**(self.current_resolution)
            pos /= scale
            # pos is in (x, y) order.
            self.umbilicus_dot.setData([pos[1]], [pos[0]])
        except Exception as e:
            print("Error updating umbilicus dot:", e)

        self.computed_labels = np.array(self.solver.get_labels())
        # self.labels[:] = self.UNLABELED
        mask_labels = np.array(self.solver.get_gt())
        self.labels[mask_labels] = self.computed_labels[mask_labels]

        self.on_overlay_points_updated(self.labels, self.computed_labels)
        self.loader_worker_running = False
        new_z_index = int(self.pending_z_center * 4 - 500)
        if new_z_index != self.current_z_index:
            print("Slider value changed during update; scheduling a new update.")
            self.debounce_timer.start(5000)

    def on_overlay_points_updated(self, labels, computed_labels):
        try:
            # print(f"Shapes of labels and computed labels: {labels.shape}, {computed_labels.shape}, undeleted: {self.undeleted_nodes_indices.shape}")
            self.overlay_request.emit(int(self.current_z_index), self.h5_path, labels, computed_labels,
                                      self.f_init, self.undeleted_nodes_indices, 50)
        except Exception as e:
            print("Error requesting overlay points:", e)
    
    def on_slice_loaded(self, image_slice):
        self.xy_view.setImage(image_slice)
        # exporting data of image view object
        current_timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Ensure the widget is fully rendered
        self.xy_view.export(os.path.join("GraphLabelerViews", f"xy_view_{current_timestamp}.png"))
    
    def get_brushes(self):
        # Create boolean masks based on valid winding conditions.
        brush_mask = np.abs(self.last_overlay_windings // 360 - self.UNLABELED) > 2
        brush_mask_computed = np.abs(self.last_overlay_windings_computed // 360 - self.UNLABELED) > 2

        print(f"[get_brushes] Valid windings: {np.sum(brush_mask)} / {len(brush_mask)} ({np.sum(brush_mask) / len(brush_mask) * 100:.2f}%)")
        print(f"[get_brushes] Valid computed windings: {np.sum(brush_mask_computed)} / {len(brush_mask_computed)} ({np.sum(brush_mask_computed) / len(brush_mask_computed) * 100:.2f}%)")
        print(f"[get_brushes] Custom labels in point_labels_xy: {len(self.point_labels_xy)}")

        # Flatten the arrays to work with one-dimensional data.
        overlay_windings_flat = self.last_overlay_windings.flatten()
        overlay_windings_computed_flat = self.last_overlay_windings_computed.flatten()

        # Create an empty array for the final brushes.
        result = np.empty(overlay_windings_flat.shape, dtype=object)
        
        # Check if gradient checkbox exists and is checked
        use_gradient = hasattr(self, 'gradient_coloring_checkbox') and self.gradient_coloring_checkbox.isChecked()
        
        # First, check for custom labels in point_labels_xy
        has_custom_label = np.zeros(len(result), dtype=bool)
        for idx, label in self.point_labels_xy.items():
            if idx < len(result):
                # Convert label to winding angle for brush selection
                winding_angle = label * 360.0
                if use_gradient:
                    custom_brush = vectorized_brush_for_winding(np.array([winding_angle]), self.red_brush, self.green_brush, self.blue_brush)[0]
                else:
                    custom_brush = brush_for_winding(winding_angle, self.red_brush, self.green_brush, self.blue_brush)
                result[idx] = custom_brush
                has_custom_label[idx] = True
        
        # For points without custom labels, use the winding values which include gui_main labels
        # Compute the brush arrays based on checkbox state
        if use_gradient:
            brushes_primary = vectorized_brush_for_winding(overlay_windings_flat, self.red_brush, self.green_brush, self.blue_brush)
            brushes_computed = vectorized_brush_for_winding(overlay_windings_computed_flat, self.calc_brush_red, self.calc_brush_green, self.calc_brush_blue)
        else:
            # Use simple 3-color cycling
            brushes_primary = np.empty(overlay_windings_flat.shape, dtype=object)
            brushes_computed = np.empty(overlay_windings_computed_flat.shape, dtype=object)
            for i in range(len(overlay_windings_flat)):
                brushes_primary[i] = brush_for_winding(overlay_windings_flat[i], self.red_brush, self.green_brush, self.blue_brush)
                brushes_computed[i] = brush_for_winding(overlay_windings_computed_flat[i], self.calc_brush_red, self.calc_brush_green, self.calc_brush_blue)
        
        # Apply brushes based on masks
        mask_no_custom = ~has_custom_label
        
        # Priority order:
        # 1. Custom labels (already applied above)
        # 2. GUI main labels (brush_mask = True)
        # 3. Computed labels (brush_mask = False, brush_mask_computed = True)
        # 4. White brush for truly unlabeled
        
        # Apply primary brushes where we have valid labels from gui_main
        result[mask_no_custom & brush_mask] = brushes_primary[mask_no_custom & brush_mask]
        
        # Apply computed brushes where we don't have gui_main labels but do have computed labels
        condition_computed = np.logical_and(~brush_mask, brush_mask_computed) & mask_no_custom
        result[condition_computed] = brushes_computed[condition_computed]
        
        # For truly unlabeled points, use white brush
        result[mask_no_custom & ~(brush_mask | brush_mask_computed)] = self.white_brush

        return result.tolist()

    def on_overlay_points_computed(self, overlay_data):
        # overlay_data is a tuple: (overlay_points, overlay_windings, overlay_windings_computed, winding, winding_computed)
        overlay_points, overlay_windings, overlay_windings_computed, self.winding, self.winding_computed  = overlay_data
        print(f"Overlay points computed: {overlay_points.shape} unique points found.")
        self.last_overlay_windings = overlay_windings
        self.last_overlay_windings_computed = overlay_windings_computed
        brushes = self.get_brushes()
        # Scale to proper ome zarr index
        scale = 2**(self.current_resolution)
        overlay_points /= scale
        # Assume overlay_points columns: [z, x, y, ...] for XY view.
        x_coords = overlay_points[:, 1]
        y_coords = overlay_points[:, 2]
        
        if hasattr(self, "overlay_scatter"):
            print("Updating existing overlay scatter plot with new colors.")
            self.overlay_scatter.setData(x=x_coords, y=y_coords, brush=brushes)
        else:
            print("Creating new overlay scatter plot with custom colors.")
            self.overlay_scatter = pg.ScatterPlotItem(
                x=x_coords,
                y=y_coords,
                pen=pg.mkPen(None),
                brush=brushes,
                size=2
            )
            self.xy_view.addItem(self.overlay_scatter)
        # exporting data of image view object
        current_timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.xy_view.export(os.path.join("GraphLabelerViews", f"points_xy_view_{current_timestamp}.png"))
        save_high_res_widget(self.xy_view, os.path.join("GraphLabelerViews", f"pixmap_points_xy_view_{current_timestamp}.png"))
        print("OME-Zarr XY view updated with new slice.")
    
    # --- XZ Overlay Helpers ---
    def get_brushes_xz(self):
        # Similar to get_brushes(), but operating on the XZ-specific arrays.
        brush_mask = np.abs(self.last_overlay_windings_xz // 360 - self.UNLABELED) > 2
        brush_mask_computed = np.abs(self.last_overlay_windings_computed_xz // 360 - self.UNLABELED) > 2

        print(f"[get_brushes_xz] Valid windings: {np.sum(brush_mask)} / {len(brush_mask)} ({np.sum(brush_mask) / len(brush_mask) * 100:.2f}%)")
        print(f"[get_brushes_xz] Valid computed windings: {np.sum(brush_mask_computed)} / {len(brush_mask_computed)} ({np.sum(brush_mask_computed) / len(brush_mask_computed) * 100:.2f}%)")
        print(f"[get_brushes_xz] Custom labels in point_labels_xz: {len(self.point_labels_xz)}")

        overlay_windings_flat = self.last_overlay_windings_xz.flatten()
        overlay_windings_computed_flat = self.last_overlay_windings_computed_xz.flatten()

        result = np.empty(overlay_windings_flat.shape, dtype=object)
        
        # Check if gradient checkbox exists and is checked
        use_gradient = hasattr(self, 'gradient_coloring_checkbox') and self.gradient_coloring_checkbox.isChecked()
        
        # First, check for custom labels in point_labels_xz
        has_custom_label = np.zeros(len(result), dtype=bool)
        for idx, label in self.point_labels_xz.items():
            if idx < len(result):
                # Convert label to winding angle for brush selection
                winding_angle = label * 360.0
                if use_gradient:
                    custom_brush = vectorized_brush_for_winding(np.array([winding_angle]), self.red_brush, self.green_brush, self.blue_brush)[0]
                else:
                    custom_brush = brush_for_winding(winding_angle, self.red_brush, self.green_brush, self.blue_brush)
                result[idx] = custom_brush
                has_custom_label[idx] = True
        
        # For points without custom labels, use the winding values which include gui_main labels
        # Compute the brush arrays based on checkbox state
        if use_gradient:
            brushes_primary = vectorized_brush_for_winding(overlay_windings_flat, self.red_brush, self.green_brush, self.blue_brush)
            brushes_computed = vectorized_brush_for_winding(overlay_windings_computed_flat, self.calc_brush_red, self.calc_brush_green, self.calc_brush_blue)
        else:
            # Use simple 3-color cycling
            brushes_primary = np.empty(overlay_windings_flat.shape, dtype=object)
            brushes_computed = np.empty(overlay_windings_computed_flat.shape, dtype=object)
            for i in range(len(overlay_windings_flat)):
                brushes_primary[i] = brush_for_winding(overlay_windings_flat[i], self.red_brush, self.green_brush, self.blue_brush)
                brushes_computed[i] = brush_for_winding(overlay_windings_computed_flat[i], self.calc_brush_red, self.calc_brush_green, self.calc_brush_blue)
        
        # Apply brushes based on masks
        mask_no_custom = ~has_custom_label
        
        # Priority order:
        # 1. Custom labels (already applied above)
        # 2. GUI main labels (brush_mask = True)  
        # 3. Computed labels (brush_mask = False, brush_mask_computed = True)
        # 4. White brush for truly unlabeled
        
        # Apply primary brushes where we have valid labels from gui_main
        result[mask_no_custom & brush_mask] = brushes_primary[mask_no_custom & brush_mask]
        
        # Apply computed brushes where we don't have gui_main labels but do have computed labels
        condition_computed = np.logical_and(~brush_mask, brush_mask_computed) & mask_no_custom
        result[condition_computed] = brushes_computed[condition_computed]
        
        # For truly unlabeled points, use white brush
        result[mask_no_custom & ~(brush_mask | brush_mask_computed)] = self.white_brush

        return result.tolist()

    def on_overlay_points_updated_xz(self, labels, computed_labels):
        try:
            self.overlay_request_xz.emit(float(self.pending_finit_center), self.h5_path, labels, computed_labels,
                                          self.f_init, self.undeleted_nodes_indices, 50)
        except Exception as e:
            print("Error requesting overlay points for XZ view:", e)

    def on_overlay_points_xz_computed(self, overlay_data):
        # overlay_data is a tuple: (overlay_points, overlay_windings, overlay_windings_computed, winding, winding_computed)
        overlay_points, overlay_windings, overlay_windings_computed, winding, winding_computed = overlay_data
        print(f"Overlay XZ points computed: {overlay_points.shape} unique points found.")
        self.last_overlay_windings_xz = overlay_windings
        self.last_overlay_windings_computed_xz = overlay_windings_computed
        brushes = self.get_brushes_xz()

        # Scale to proper ome zarr index
        scale = 2**(self.current_resolution)
        overlay_points /= scale
        # Assume overlay_points for the XZ view: first column is the sample (x) coordinate and second column is z.
        x_coords = overlay_points[:, 0]
        z_coords = self.L - overlay_points[:, 1]
        
        if hasattr(self, "overlay_scatter_xz"):
            print("Updating existing XZ overlay scatter plot with new colors.")
            self.overlay_scatter_xz.setData(x=x_coords, y=z_coords, brush=brushes)
        else:
            print("Creating new XZ overlay scatter plot with custom colors.")
            self.overlay_scatter_xz = pg.ScatterPlotItem(
                x=x_coords,
                y=z_coords,
                pen=pg.mkPen(None),
                brush=brushes,
                size=2
            )
            self.xz_view.addItem(self.overlay_scatter_xz)
        # exporting data of image view object
        current_timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Ensure the widget is fully rendered
        self.xz_view.export(os.path.join("GraphLabelerViews", f"points_xz_view_{current_timestamp}.png"))
        save_high_res_widget(self.xz_view, os.path.join("GraphLabelerViews", f"pixmap_points_xz_view_{current_timestamp}.png"))
    
    # --- Color Update Trigger ---
    def update_overlay_labels(self, labels, computed_labels):
        print(f"[ome_zarr_view] update_overlay_labels called with {len(labels)} labels, {np.sum(np.abs(labels - self.UNLABELED) > 2)} labeled points")
        self.labels = labels
        self.computed_labels = computed_labels
        self.debounce_timer_labels.stop()
        self.debounce_timer_labels.start(100)  # Reduced from 5000ms to 100ms for more responsive updates

    def _trigger_overlay_labels_update(self):
        windings = self.labels * 360.0 + self.f_init
        windings_computed = self.computed_labels * 360.0 + self.f_init
        try:
            labeled_mask = np.abs(self.labels - self.UNLABELED) > 2
            print(f"[_trigger_overlay_labels_update] Requesting update with {np.sum(labeled_mask)} labeled points")
            if np.sum(labeled_mask) > 0:
                sample_labels = self.labels[labeled_mask][:5]  # Show first 5 labeled points
                sample_windings = windings[labeled_mask][:5]
                print(f"[_trigger_overlay_labels_update] Sample labels: {sample_labels} -> windings: {sample_windings}")
            self.label_request.emit(windings, windings_computed)
        except Exception as e:
            print("Error requesting overlay labels:", e)

    def on_overlay_labels_computed(self, data):
        if not hasattr(self, "overlay_scatter"):
            print("[on_overlay_labels_computed] No overlay_scatter, returning early")
            return
        overlay_windings, overlay_windings_computed = data
        print(f"[on_overlay_labels_computed] Received windings for {len(overlay_windings)} points")
        self.last_overlay_windings = overlay_windings
        self.last_overlay_windings_computed = overlay_windings_computed
        brushes = self.get_brushes()
        self.overlay_scatter.setBrush(brushes)
        print(f"[on_overlay_labels_computed] Updated scatter plot with {len(brushes)} brushes")

    def on_overlay_labels_computed_xz(self, data):
        if not hasattr(self, "overlay_scatter_xz"):
            return
        overlay_windings, overlay_windings_computed = data
        print("Overlay labels computed for XZ view.")
        self.last_overlay_windings_xz = overlay_windings
        self.last_overlay_windings_computed_xz = overlay_windings_computed
        brushes = self.get_brushes_xz()
        self.overlay_scatter_xz.setBrush(brushes)
    
    # --- XZ view update ---
    def update_finit_center(self, finit_center_value):
        self.pending_finit_center = finit_center_value
        self.finit_debounce_timer.stop()
        self.finit_debounce_timer.start(1000)
    
    def _trigger_xz_slice_update(self):
        if self.pending_finit_center is None:
            return
        print(f"Loading OME-Zarr XZ slice with f init center {self.pending_finit_center}, type of {type(self.pending_finit_center)}")
        
        # Clear point labels for the new slice
        self.point_labels_xz.clear()
        
        # Launch XZ loader worker at selected resolution
        self.xz_loader_worker = OmeZarrXZLoaderWorker(self.ome_zarr_path, self.current_resolution,
                                                    self.pending_finit_center, self.umbilicus_path)
        self.xz_loader_worker.xz_slice_loaded.connect(self.on_xz_slice_loaded)
        self.xz_loader_worker.start()
        # When updating the XZ view, request the XZ overlay.
        self.on_overlay_points_updated_xz(self.labels, self.computed_labels)
        
    def on_resolution_changed(self, index):
        """
        Handler for resolution dropdown changes: load metadata and update views.
        """
        level = self.resolution_combobox.itemData(index)
        self.current_resolution = level
        # Load Zarr metadata for this resolution
        try:
            store = zarr.open(self.ome_zarr_path, mode='r')
            dset = store[str(level)]
            self.z_dim, self.y_dim, self.x_dim = dset.shape
            self.L = max(self.x_dim, self.y_dim) / 2.0
        except Exception as e:
            QMessageBox.warning(self, "Resolution Error",
                                f"Could not load resolution {level}: {e}")
            return
        # Update overlay worker if exists
        if hasattr(self, 'persistent_overlay_worker'):
            self.persistent_overlay_worker.umbilicus_data = self.umbilicus_data
        # Refresh current slices
        if hasattr(self, 'pending_z_center') and self.pending_z_center is not None:
            self._trigger_z_slice_update()
        if hasattr(self, 'pending_finit_center') and self.pending_finit_center is not None:
            self._trigger_xz_slice_update()
    
    def on_xz_slice_loaded(self, xz_image):
        self.xz_view.setImage(xz_image)
        # exporting data of image view object
        current_timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.xz_view.export(os.path.join("GraphLabelerViews", f"xz_view_{current_timestamp}.png"))
        print("OME-Zarr XZ view updated with new slice.")
    
    def on_label_changed(self, value):
        """Handle label spinbox value change."""
        self.current_label = value
        print(f"Current label set to: {value}")
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_P:
            # Toggle pipette mode
            self.pipette_mode = not self.pipette_mode
            if self.pipette_mode:
                self.pipette_label.setText("Pipette: ON")
                self.pipette_label.setStyleSheet("color: green;")
                print("Pipette mode enabled - click a point to sample its label")
            else:
                self.pipette_label.setText("Pipette: OFF")
                self.pipette_label.setStyleSheet("")
                print("Pipette mode disabled")
        else:
            super().keyPressEvent(event)
    
    def eventFilter(self, source, event):
        """Filter events for pipette functionality and stroke-based painting."""
        if event.type() == QEvent.GraphicsSceneMousePress:
            # Check which view was clicked
            if source == self.xy_view.getView().scene():
                if self.pipette_mode:
                    self.handle_pipette_click_xy(event)
                else:
                    self.start_stroke_xy(event)
                return True
            elif source == self.xz_view.getView().scene():
                if self.pipette_mode:
                    self.handle_pipette_click_xz(event)
                else:
                    self.start_stroke_xz(event)
                return True
        elif event.type() == QEvent.GraphicsSceneMouseMove:
            if source == self.xy_view.getView().scene() and self._current_path_xy is not None:
                self.extend_stroke_xy(event)
                return True
            elif source == self.xz_view.getView().scene() and self._current_path_xz is not None:
                self.extend_stroke_xz(event)
                return True
        elif event.type() == QEvent.GraphicsSceneMouseRelease:
            if source == self.xy_view.getView().scene() and self._current_path_xy is not None:
                self.finish_stroke_xy(event)
                return True
            elif source == self.xz_view.getView().scene() and self._current_path_xz is not None:
                self.finish_stroke_xz(event)
                return True
        return super().eventFilter(source, event)
    
    def handle_pipette_click_xy(self, event):
        """Handle pipette click in XY view."""
        if not hasattr(self, 'overlay_scatter'):
            return
        
        # Get mouse position in data coordinates
        pos = self.xy_view.getView().mapSceneToView(event.scenePos())
        mouse_x, mouse_y = pos.x(), pos.y()
        
        # Get scatter plot data
        data = self.overlay_scatter.data
        if data is None or len(data) == 0:
            return
        
        x_coords = data['x']
        y_coords = data['y']
        
        # Find nearest point
        distances = np.sqrt((x_coords - mouse_x)**2 + (y_coords - mouse_y)**2)
        nearest_idx = np.argmin(distances)
        
        # Check if point has a custom label first
        if nearest_idx in self.point_labels_xy and False:
            sampled_label = self.point_labels_xy[nearest_idx]
            print(f"Sampled custom label: {sampled_label}")
        else:
            # Sample from the winding value
            if hasattr(self, 'last_overlay_windings') and self.last_overlay_windings is not None:
                winding = self.last_overlay_windings.flatten()[nearest_idx]
                sampled_label = int(round(winding / 360.0))
                print(f"Sampled winding label: {sampled_label}")
            else:
                print("No winding data available")
                return
        
        # Update the spinbox
        self.label_spinbox.setValue(sampled_label)
        
        # Disable pipette mode after sampling
        self.pipette_mode = False
        self.pipette_label.setText("Pipette: OFF")
        self.pipette_label.setStyleSheet("")
    
    def handle_pipette_click_xz(self, event):
        """Handle pipette click in XZ view."""
        if not hasattr(self, 'overlay_scatter_xz'):
            return
        
        # Get mouse position in data coordinates
        pos = self.xz_view.getView().mapSceneToView(event.scenePos())
        mouse_x, mouse_y = pos.x(), pos.y()
        
        # Get scatter plot data
        data = self.overlay_scatter_xz.data
        if data is None or len(data) == 0:
            return
        
        x_coords = data['x']
        y_coords = data['y']
        
        # Find nearest point
        distances = np.sqrt((x_coords - mouse_x)**2 + (y_coords - mouse_y)**2)
        nearest_idx = np.argmin(distances)
        
        # Check if point has a custom label first
        if nearest_idx in self.point_labels_xz and False:
            sampled_label = self.point_labels_xz[nearest_idx]
            print(f"Sampled custom label: {sampled_label}")
        else:
            # Sample from the winding value
            if hasattr(self, 'last_overlay_windings_xz') and self.last_overlay_windings_xz is not None:
                winding = self.last_overlay_windings_xz.flatten()[nearest_idx]
                sampled_label = int(round(winding / 360.0))
                print(f"Sampled winding label: {sampled_label}")
            else:
                print("No winding data available")
                return
        
        # Update the spinbox
        self.label_spinbox.setValue(sampled_label)
        
        # Disable pipette mode after sampling
        self.pipette_mode = False
        self.pipette_label.setText("Pipette: OFF")
        self.pipette_label.setStyleSheet("")
    
    def update_drawing_radius(self, value):
        self.drawing_radius = value
        # Update overlay pen widths
        pen = self._overlay.pen()
        pen.setWidthF(self.drawing_radius * 2)
        self._overlay.setPen(pen)
        
        pen_xz = self._overlay_xz.pen()
        pen_xz.setWidthF(self.drawing_radius * 2)
        self._overlay_xz.setPen(pen_xz)
        print(f"Drawing radius updated to: {self.drawing_radius}")
    
    def _current_label_qcolor(self):
        """Return a semi-transparent QColor matching the current label."""
        lab = self.label_spinbox.value()
        alpha = 150
        
        if lab == self.UNLABELED:
            # Draw erase strokes in dark gray/black
            return QColor(50, 50, 50, alpha)
        
        # Use gradient coloring for consistency with the displayed points
        # Map label to angle (label * 360 degrees)
        angle = (lab * 360.0) % 360
        if angle < 0:
            angle += 360
            
        if angle < 120:
            # Transition: Red to Green
            f = angle / 120.0
            r = int(255 * (1 - f))
            g = int(255 * f)
            b = 0
        elif angle < 240:
            # Transition: Green to Blue
            f = (angle - 120) / 120.0
            r = 0
            g = int(255 * (1 - f))
            b = int(255 * f)
        else:
            # Transition: Blue to Red
            f = (angle - 240) / 120.0
            r = int(255 * f)
            g = 0
            b = int(255 * (1 - f))
            
        return QColor(r, g, b, alpha)
    
    def start_stroke_xy(self, event):
        """Start a brush stroke in XY view."""
        if self._stroke_backup is None:
            self._stroke_backup = self.point_labels_xy.copy()
        
        pos = self.xy_view.getView().mapSceneToView(event.scenePos())
        self._current_path_xy = QPainterPath(QPointF(pos.x(), pos.y()))
        
        pen = self._overlay.pen()
        pen.setColor(self._current_label_qcolor())
        self._overlay.setPen(pen)
        self._overlay.setPath(self._current_path_xy)
        self._overlay.setVisible(True)
    
    def extend_stroke_xy(self, event):
        """Extend the brush stroke in XY view."""
        pos = self.xy_view.getView().mapSceneToView(event.scenePos())
        self._current_path_xy.lineTo(QPointF(pos.x(), pos.y()))
        self._overlay.setPath(self._current_path_xy)
    
    def finish_stroke_xy(self, event):
        """Finish the brush stroke and apply labels in XY view."""
        # Extract path points
        pts = [(self._current_path_xy.elementAt(i).x,
                self._current_path_xy.elementAt(i).y)
               for i in range(self._current_path_xy.elementCount())]
        
        # Apply labels along the path
        self._apply_brush_path_to_labels_xy(pts)
        
        # Clear the overlay
        self._stroke_backup = None
        self._current_path_xy = None
        self._overlay.setPath(QPainterPath())
        self._overlay.setVisible(False)
    
    def start_stroke_xz(self, event):
        """Start a brush stroke in XZ view."""
        if self._stroke_backup is None:
            self._stroke_backup = self.point_labels_xz.copy()
        
        pos = self.xz_view.getView().mapSceneToView(event.scenePos())
        self._current_path_xz = QPainterPath(QPointF(pos.x(), pos.y()))
        
        pen = self._overlay_xz.pen()
        pen.setColor(self._current_label_qcolor())
        self._overlay_xz.setPen(pen)
        self._overlay_xz.setPath(self._current_path_xz)
        self._overlay_xz.setVisible(True)
    
    def extend_stroke_xz(self, event):
        """Extend the brush stroke in XZ view."""
        pos = self.xz_view.getView().mapSceneToView(event.scenePos())
        self._current_path_xz.lineTo(QPointF(pos.x(), pos.y()))
        self._overlay_xz.setPath(self._current_path_xz)
    
    def finish_stroke_xz(self, event):
        """Finish the brush stroke and apply labels in XZ view."""
        # Extract path points
        pts = [(self._current_path_xz.elementAt(i).x,
                self._current_path_xz.elementAt(i).y)
               for i in range(self._current_path_xz.elementCount())]
        
        # Apply labels along the path
        self._apply_brush_path_to_labels_xz(pts)
        
        # Clear the overlay
        self._stroke_backup = None
        self._current_path_xz = None
        self._overlay_xz.setPath(QPainterPath())
        self._overlay_xz.setVisible(False)
    
    def _interpolate_path(self, path, r):
        """Densify a list of (x,y) points so no segment exceeds half the radius r."""
        if not path:
            return []
        step = r * 0.5
        interp = []
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            interp.append((x0, y0))
            dx = x1 - x0
            dy = y1 - y0
            dist = np.hypot(dx, dy)
            if dist > step:
                n = int(dist / step)
                for k in range(1, n + 1):
                    t = k / (n + 1)
                    interp.append((x0 + t * dx, y0 + t * dy))
        interp.append(path[-1])
        return interp
    
    def _apply_brush_path_to_labels_xy(self, data_path):
        """Apply labels along the brush path in XY view."""
        if not hasattr(self, 'overlay_scatter'):
            return
        
        # Get scatter plot data
        data = self.overlay_scatter.data
        if data is None or len(data) == 0:
            return
        
        # Densify the path
        r = self.drawing_radius
        data_path = self._interpolate_path(data_path, r)
        
        current_label = self.label_spinbox.value()
        x_coords = data['x']
        y_coords = data['y']
        
        # Find points within radius of the path
        hit = set()
        for px, py in data_path:
            distances = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
            close_indices = np.where(distances <= r)[0]
            hit.update(close_indices)
        
        # Apply label to hit points
        for idx in hit:
            if current_label == self.UNLABELED:
                # Erase label
                if idx in self.point_labels_xy:
                    del self.point_labels_xy[idx]
            else:
                # Set label
                self.point_labels_xy[idx] = current_label
        
        # Update display
        brushes = self.get_brushes()
        self.overlay_scatter.setBrush(brushes)
        self.update_labels_status()
    
    def _apply_brush_path_to_labels_xz(self, data_path):
        """Apply labels along the brush path in XZ view."""
        if not hasattr(self, 'overlay_scatter_xz'):
            return
        
        # Get scatter plot data
        data = self.overlay_scatter_xz.data
        if data is None or len(data) == 0:
            return
        
        # Densify the path
        r = self.drawing_radius
        data_path = self._interpolate_path(data_path, r)
        
        current_label = self.label_spinbox.value()
        x_coords = data['x']
        y_coords = data['y']
        
        # Find points within radius of the path
        hit = set()
        for px, py in data_path:
            distances = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
            close_indices = np.where(distances <= r)[0]
            hit.update(close_indices)
        
        # Apply label to hit points
        for idx in hit:
            if current_label == self.UNLABELED:
                # Erase label
                if idx in self.point_labels_xz:
                    del self.point_labels_xz[idx]
            else:
                # Set label
                self.point_labels_xz[idx] = current_label
        
        # Update display
        brushes = self.get_brushes_xz()
        self.overlay_scatter_xz.setBrush(brushes)
        self.update_labels_status()

    def update_labels_status(self):
        """Update the status label with the count of custom labels."""
        total_labels = len(self.point_labels_xy) + len(self.point_labels_xz)
        self.labels_status.setText(f"Custom labels: {total_labels}")

    def clear_custom_labels(self):
        """Clear all custom point labels."""
        self.point_labels_xy.clear()
        self.point_labels_xz.clear()
        self.update_labels_status()
        # Refresh the display
        if hasattr(self, 'overlay_scatter'):
            brushes = self.get_brushes()
            self.overlay_scatter.setBrush(brushes)
        if hasattr(self, 'overlay_scatter_xz'):
            brushes = self.get_brushes_xz()
            self.overlay_scatter_xz.setBrush(brushes)
        print("Cleared all custom labels")

    def closeEvent(self, event):
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        if self.finit_debounce_timer.isActive():
            self.finit_debounce_timer.stop()
        if hasattr(self, "overlay_thread") and self.overlay_thread.isRunning():
            self.overlay_thread.quit()
            self.overlay_thread.wait()
        print("OME-Zarr View Window is closing.")
        super().closeEvent(event)

    def update_views(self):
        """Trigger view updates when gradient checkbox is toggled."""
        # Force brush update on both views
        if hasattr(self, 'overlay_scatter'):
            brushes = self.get_brushes()
            self.overlay_scatter.setBrush(brushes)
        if hasattr(self, 'overlay_scatter_xz'):
            brushes = self.get_brushes_xz()
            self.overlay_scatter_xz.setBrush(brushes)

    def apply_labels_to_graph(self):
        """Apply point labels back to graph nodes based on voting criteria."""
        if not hasattr(self, 'persistent_overlay_worker'):
            QMessageBox.warning(self, "Error", "Overlay worker not initialized")
            return
            
        # Debug: Check if we have the necessary mappings
        print(f"\nDEBUG: Starting apply_labels_to_graph")
        print(f"XY labels: {len(self.point_labels_xy)} points labeled")
        print(f"XZ labels: {len(self.point_labels_xz)} points labeled")
        
        # Process XY and XZ views separately
        xy_node_labels = {}
        xz_node_labels = {}
        
        # Process XY labels
        if hasattr(self.persistent_overlay_worker, 'overlay_point_nodes_indices') and self.persistent_overlay_worker.overlay_point_nodes_indices is not None:
            print(f"\nDEBUG XY: overlay_point_nodes_indices length: {len(self.persistent_overlay_worker.overlay_point_nodes_indices)}")
            
            # Check if we have inverse_indices for XY
            if hasattr(self.persistent_overlay_worker, 'inverse_indices') and self.persistent_overlay_worker.inverse_indices is not None:
                print(f"DEBUG XY: inverse_indices length: {len(self.persistent_overlay_worker.inverse_indices)}")
                print(f"DEBUG XY: inverse_indices unique values: {len(np.unique(self.persistent_overlay_worker.inverse_indices))}")
                
                # Map displayed points to original points, then to nodes
                xy_labels_by_node = {}
                debug_count = 0
                for display_idx, label in self.point_labels_xy.items():
                    # Find all original points that map to this display point
                    original_indices = np.where(self.persistent_overlay_worker.inverse_indices == display_idx)[0]
                    if debug_count < 5:  # Only show first 5 for debugging
                        print(f"DEBUG XY: Display point {display_idx} (label={label}) maps to {len(original_indices)} original points")
                        debug_count += 1
                    
                    for orig_idx in original_indices:
                        if orig_idx < len(self.persistent_overlay_worker.overlay_point_nodes_indices):
                            node_idx = self.persistent_overlay_worker.overlay_point_nodes_indices[orig_idx]
                            if node_idx not in xy_labels_by_node:
                                xy_labels_by_node[node_idx] = []
                            xy_labels_by_node[node_idx].append(label)
                
                print(f"DEBUG XY: Total labeled display points: {len(self.point_labels_xy)}")
                print(f"DEBUG XY: These expand to label assignments across {sum(len(labels) for labels in xy_labels_by_node.values())} original points")
            else:
                print("WARNING: No inverse_indices for XY view - using direct mapping")
                # Fallback to direct mapping
                xy_labels_by_node = {}
                for point_idx, label in self.point_labels_xy.items():
                    if point_idx < len(self.persistent_overlay_worker.overlay_point_nodes_indices):
                        node_idx = self.persistent_overlay_worker.overlay_point_nodes_indices[point_idx]
                        if node_idx not in xy_labels_by_node:
                            xy_labels_by_node[node_idx] = []
                        xy_labels_by_node[node_idx].append(label)
            
            print(f"DEBUG XY: Labels grouped into {len(xy_labels_by_node)} nodes")
            
            # Count total XY points per node
            xy_total_points = {}
            for node_idx in self.persistent_overlay_worker.overlay_point_nodes_indices:
                xy_total_points[node_idx] = xy_total_points.get(node_idx, 0) + 1
            
            print(f"DEBUG XY: Total points counted for {len(xy_total_points)} nodes")
            
            # Apply voting for XY
            debug_node_count = 0
            for node_idx, labels in xy_labels_by_node.items():
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Find most common label
                max_label = max(label_counts.items(), key=lambda x: x[1])
                label, count = max_label
                total = xy_total_points.get(node_idx, 0)
                
                if debug_node_count < 10:  # Only show first 10 nodes
                    print(f"DEBUG XY: Node {node_idx} - {count} labeled points out of {total} total ({count/total*100:.1f}% if total > 0)")
                    debug_node_count += 1
                
                if count >= 3 and total > 0:
                    percentage = count / total
                    if percentage >= 0.5:
                        xy_node_labels[node_idx] = (label, percentage, count)
                        if debug_node_count <= 10:
                            print(f"  -> ACCEPTED for XY view")
                    else:
                        if debug_node_count <= 10:
                            print(f"  -> REJECTED: percentage {percentage:.1%} < 50%")
                else:
                    if debug_node_count <= 10:
                        print(f"  -> REJECTED: count {count} < 3 or total {total} = 0")
            
            if len(xy_labels_by_node) > 10:
                print(f"DEBUG XY: ... and {len(xy_labels_by_node) - 10} more nodes")
            print(f"DEBUG XY: Total nodes with accepted labels: {len(xy_node_labels)}")
        else:
            print("WARNING: No overlay_point_nodes_indices for XY view")
                        
        # Process XZ labels
        if hasattr(self.persistent_overlay_worker, 'overlay_point_nodes_indices_xz') and self.persistent_overlay_worker.overlay_point_nodes_indices_xz is not None:
            print(f"\nDEBUG XZ: overlay_point_nodes_indices_xz length: {len(self.persistent_overlay_worker.overlay_point_nodes_indices_xz)}")
            
            # Check if we have inverse_indices for XZ
            if hasattr(self.persistent_overlay_worker, 'inverse_indices_xz') and self.persistent_overlay_worker.inverse_indices_xz is not None:
                print(f"DEBUG XZ: inverse_indices_xz length: {len(self.persistent_overlay_worker.inverse_indices_xz)}")
                print(f"DEBUG XZ: inverse_indices_xz unique values: {len(np.unique(self.persistent_overlay_worker.inverse_indices_xz))}")
                
                # Map displayed points to original points, then to nodes
                xz_labels_by_node = {}
                debug_count = 0
                for display_idx, label in self.point_labels_xz.items():
                    # Find all original points that map to this display point
                    original_indices = np.where(self.persistent_overlay_worker.inverse_indices_xz == display_idx)[0]
                    if debug_count < 5:  # Only show first 5 for debugging
                        print(f"DEBUG XZ: Display point {display_idx} (label={label}) maps to {len(original_indices)} original points")
                        debug_count += 1
                    
                    for orig_idx in original_indices:
                        if orig_idx < len(self.persistent_overlay_worker.overlay_point_nodes_indices_xz):
                            node_idx = self.persistent_overlay_worker.overlay_point_nodes_indices_xz[orig_idx]
                            if node_idx not in xz_labels_by_node:
                                xz_labels_by_node[node_idx] = []
                            xz_labels_by_node[node_idx].append(label)
                
                print(f"DEBUG XZ: Total labeled display points: {len(self.point_labels_xz)}")
                print(f"DEBUG XZ: These expand to label assignments across {sum(len(labels) for labels in xz_labels_by_node.values())} original points")
            else:
                print("WARNING: No inverse_indices_xz for XZ view - using direct mapping")
                # Fallback to direct mapping
                xz_labels_by_node = {}
                for point_idx, label in self.point_labels_xz.items():
                    if point_idx < len(self.persistent_overlay_worker.overlay_point_nodes_indices_xz):
                        node_idx = self.persistent_overlay_worker.overlay_point_nodes_indices_xz[point_idx]
                        if node_idx not in xz_labels_by_node:
                            xz_labels_by_node[node_idx] = []
                        xz_labels_by_node[node_idx].append(label)
            
            # Count total XZ points per node
            xz_total_points = {}
            for node_idx in self.persistent_overlay_worker.overlay_point_nodes_indices_xz:
                xz_total_points[node_idx] = xz_total_points.get(node_idx, 0) + 1
            
            # Apply voting for XZ
            for node_idx, labels in xz_labels_by_node.items():
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Find most common label
                max_label = max(label_counts.items(), key=lambda x: x[1])
                label, count = max_label
                total = xz_total_points.get(node_idx, 0)
                
                if count >= 3 and total > 0:
                    percentage = count / total
                    if percentage >= 0.5:
                        xz_node_labels[node_idx] = (label, percentage, count)
        
        # Combine results with disambiguation
        node_updates_xy = {}  # Track XY-specific updates
        node_updates_xz = {}  # Track XZ-specific updates
        all_nodes = set(xy_node_labels.keys()) | set(xz_node_labels.keys())
        
        for node_idx in all_nodes:
            xy_data = xy_node_labels.get(node_idx)
            xz_data = xz_node_labels.get(node_idx)
            
            if xy_data and xz_data:
                # Both views have valid labels
                xy_label, xy_pct, xy_count = xy_data
                xz_label, xz_pct, xz_count = xz_data
                
                if xy_label == xz_label:
                    # Same label in both views - assign to XY view by default
                    node_updates_xy[node_idx] = xy_label
                else:
                    # Different labels - choose the one with higher percentage
                    if xy_pct > xz_pct:
                        node_updates_xy[node_idx] = xy_label
                        print(f"Node {node_idx}: XY label {xy_label} ({xy_pct:.1%}) wins over XZ label {xz_label} ({xz_pct:.1%})")
                    elif xz_pct > xy_pct:
                        node_updates_xz[node_idx] = xz_label
                        print(f"Node {node_idx}: XZ label {xz_label} ({xz_pct:.1%}) wins over XY label {xy_label} ({xy_pct:.1%})")
                    else:
                        # Same percentage - use the one with more points
                        if xy_count >= xz_count:
                            node_updates_xy[node_idx] = xy_label
                        else:
                            node_updates_xz[node_idx] = xz_label
            elif xy_data:
                # Only XY has valid label
                node_updates_xy[node_idx] = xy_data[0]
            elif xz_data:
                # Only XZ has valid label
                node_updates_xz[node_idx] = xz_data[0]
                
        total_updates = len(node_updates_xy) + len(node_updates_xz)
        if total_updates > 0:
            # Emit separate signals for XY and XZ updates
            if node_updates_xy:
                self.labels_updated_signal.emit(node_updates_xy, "XY")
                print(f"Emitted {len(node_updates_xy)} XY view updates")
            if node_updates_xz:
                self.labels_updated_signal.emit(node_updates_xz, "XZ")
                print(f"Emitted {len(node_updates_xz)} XZ view updates")
            
            # Create detailed message
            xy_count = len(node_updates_xy)
            xz_count = len(node_updates_xz)
            
            msg = f"Updated {total_updates} nodes:\n"
            msg += f"- {xy_count} from XY view\n"
            msg += f"- {xz_count} from XZ view"
            
            QMessageBox.information(self, "Labels Applied", msg)
        else:
            QMessageBox.information(self, "No Updates", 
                                    "No nodes met the criteria (≥3 points and ≥50% agreement)\nCheck console for debug information")
