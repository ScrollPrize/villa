import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QComboBox, QPushButton, QLabel, QProgressDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QApplication
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject, pyqtSlot, Qt, QEvent, QPointF
from PyQt5.QtGui import QImage, QPainter
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
# remove the lib first
try:
    del sys.modules['scroll_graph_util']
except KeyError:
    print("Module 'scroll_graph_util' not found in sys.modules; continuing.")
from scroll_graph_util import compute_mean_windings_precomputed, load_xyz_from_file, umbilicus_xy_at_z

########################################
# Utility Functions
########################################

def load_graph_pkl(graph_pkl_path):
    with open(graph_pkl_path, 'rb') as f:
        graph = pickle.load(f)
    return graph   

########################################
# Helper Functions
########################################

def brush_for_winding(w, r, g, b):
    """
    Given a winding value w (assumed numeric), return a pyqtgraph brush
    using a three-color scheme: winding 1 -> red, 2 -> green, 3 -> blue,
    4 -> red, etc.
    """
    # Round to nearest integer.
    w_int = int(round(w / 360.0))
    mod = w_int % 3
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
    
    # Map winding values (in degrees) to a normalized index in [0, 12).
    w_int = np.rint(w_array / (3 * d_angle)).astype(np.int64)
    mod = w_int % steps
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

    def __init__(self, ome_zarr_path, z_index):
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.z_index = z_index

    def run(self):
        try:
            store = zarr.open(self.ome_zarr_path, mode='r')
            image_slice = store['0'][self.z_index]
        except Exception as e:
            print(f"Error loading z slice at index {self.z_index}: {e}")
            image_slice = np.zeros((512,512,3), dtype=np.uint8)
        self.slice_loaded.emit(image_slice)

# --- XZ Loader Worker using Umbilicus Data ---
class OmeZarrXZLoaderWorker(QThread):
    xz_slice_loaded = pyqtSignal(np.ndarray)

    def __init__(self, ome_zarr_path, finit_center_value, umbilicus_path):
        """
        :param ome_zarr_path: Path to the OME-Zarr store.
        :param finit_center_value: Rotation angle (in degrees) for the XZ view.
        :param umbilicus_path: Path to the umbilicus .txt file.
        """
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.finit_center_value = finit_center_value
        self.umbilicus_path = umbilicus_path

    def run(self):
        try:
            store = zarr.open(self.ome_zarr_path, mode='r')
            dset = store['0']
            z_dim, y_dim, x_dim = dset.shape  # shape: (Z, Y, X)

            # Load umbilicus data and subtract 500.
            umbilicus_data = load_xyz_from_file(self.umbilicus_path) - 500
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

    def __init__(self, graph_pkl_path, umbilicus_path, unlabeled, parent=None):
        super().__init__(parent)
        # Load the scroll graph (from pickle) once.
        self.scroll_graph = load_graph_pkl(graph_pkl_path)
        self.umbilicus_data = load_xyz_from_file(umbilicus_path) - 500
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
            # Transfer to per-point windings with self.overlay_point_nodes_indices.
            overlay_windings = close_windings[self.overlay_point_nodes_indices]
            overlay_windings_computed = close_computed[self.overlay_point_nodes_indices]
            # Get per-point labels.
            overlay_windings, overlay_windings_computed = compute_mean_windings_precomputed(
                self.inverse_indices, overlay_windings, overlay_windings_computed, self.UNLABELED
            )
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
            print("Error in PersistentScrollGraphWorker:", e)
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
        self.ome_zarr_path = ome_zarr_path
        store = zarr.open(self.ome_zarr_path, mode='r')
        dset = store['0']
        z_dim, y_dim, x_dim = dset.shape  # shape: (Z, Y, X)
        self.L = max(x_dim, y_dim) / 2.0
        self.graph_pkl_path = graph_pkl_path
        self.h5_path = h5_path
        self.umbilicus_path = umbilicus_path
        self.umbilicus_data = load_xyz_from_file(self.umbilicus_path) - 500
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
        
        # Umbilicus dot overlay.
        self.umbilicus_dot = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush('r'), size=20)
        self.xy_view.addItem(self.umbilicus_dot)

        # Setup persistent overlay worker.
        self.overlay_thread = QThread()
        self.persistent_overlay_worker = PersistentScrollGraphWorker(self.graph_pkl_path, self.umbilicus_path, self.UNLABELED)
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
        self.loader_worker_running = True
        print(f"Loading OME-Zarr XY slice at index {z_index} (from z slice center {self.pending_z_center})")
        self.loader_worker = OmeZarrLoaderWorker(self.ome_zarr_path, z_index)
        self.loader_worker.slice_loaded.connect(self.on_slice_loaded)
        self.loader_worker.start()

        try:
            pos = umbilicus_xy_at_z(self.umbilicus_data, self.current_z_index)
            # pos is in (x, y) order.
            self.umbilicus_dot.setData([pos[1]], [pos[0]])
        except Exception as e:
            print("Error updating umbilicus dot:", e)

        self.computed_labels = np.array(self.solver.get_labels())
        self.labels[:] = self.UNLABELED
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

        print(f"Percent valid windings: {np.sum(brush_mask) / len(brush_mask) * 100:.2f}%")
        print(f"Percent valid computed windings: {np.sum(brush_mask_computed) / len(brush_mask_computed) * 100:.2f}%")

        # Flatten the arrays to work with one-dimensional data.
        overlay_windings_flat = self.last_overlay_windings.flatten()
        overlay_windings_computed_flat = self.last_overlay_windings_computed.flatten()

        # Compute the brush arrays in a vectorized fashion.
        brushes_primary = vectorized_brush_for_winding(overlay_windings_flat, self.red_brush, self.green_brush, self.blue_brush)
        brushes_computed = vectorized_brush_for_winding(overlay_windings_computed_flat, self.calc_brush_red, self.calc_brush_green, self.calc_brush_blue)

        # Create an empty array for the final brushes.
        result = np.empty(overlay_windings_flat.shape, dtype=object)
        # Where the primary mask is true, assign primary brushes.
        result[brush_mask] = brushes_primary[brush_mask]
        # Where the primary mask is false but the computed mask is true, assign computed brushes.
        condition_computed = np.logical_and(~brush_mask, brush_mask_computed)
        result[condition_computed] = brushes_computed[condition_computed]
        # For all other cases, use the white brush.
        result[~(brush_mask | brush_mask_computed)] = self.white_brush

        return result.tolist()

    def on_overlay_points_computed(self, overlay_data):
        # overlay_data is a tuple: (overlay_points, overlay_windings, overlay_windings_computed, winding, winding_computed)
        overlay_points, overlay_windings, overlay_windings_computed, self.winding, self.winding_computed  = overlay_data
        print(f"Overlay points computed: {overlay_points.shape} unique points found.")
        self.last_overlay_windings = overlay_windings
        self.last_overlay_windings_computed = overlay_windings_computed
        brushes = self.get_brushes()
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

        print(f"XZ view: Percent valid windings: {np.sum(brush_mask) / len(brush_mask) * 100:.2f}%")
        print(f"XZ view: Percent valid computed windings: {np.sum(brush_mask_computed) / len(brush_mask_computed) * 100:.2f}%")

        overlay_windings_flat = self.last_overlay_windings_xz.flatten()
        overlay_windings_computed_flat = self.last_overlay_windings_computed_xz.flatten()

        brushes_primary = vectorized_brush_for_winding(overlay_windings_flat, self.red_brush, self.green_brush, self.blue_brush)
        brushes_computed = vectorized_brush_for_winding(overlay_windings_computed_flat, self.calc_brush_red, self.calc_brush_green, self.calc_brush_blue)

        result = np.empty(overlay_windings_flat.shape, dtype=object)
        result[brush_mask] = brushes_primary[brush_mask]
        condition_computed = np.logical_and(~brush_mask, brush_mask_computed)
        result[condition_computed] = brushes_computed[condition_computed]
        result[~(brush_mask | brush_mask_computed)] = self.white_brush

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
        self.labels = labels
        self.computed_labels = computed_labels
        self.debounce_timer_labels.stop()
        self.debounce_timer_labels.start(5000)

    def _trigger_overlay_labels_update(self):
        windings = self.labels * 360.0 + self.f_init
        windings_computed = self.computed_labels * 360.0 + self.f_init
        try:
            print("Requesting overlay labels update.")
            self.label_request.emit(windings, windings_computed)
        except Exception as e:
            print("Error requesting overlay labels:", e)

    def on_overlay_labels_computed(self, data):
        if not hasattr(self, "overlay_scatter"):
            return
        overlay_windings, overlay_windings_computed = data
        print("Overlay labels computed.")
        self.last_overlay_windings = overlay_windings
        self.last_overlay_windings_computed = overlay_windings_computed
        brushes = self.get_brushes()
        self.overlay_scatter.setBrush(brushes)

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
        self.xz_loader_worker = OmeZarrXZLoaderWorker(self.ome_zarr_path, self.pending_finit_center, self.umbilicus_path)
        self.xz_loader_worker.xz_slice_loaded.connect(self.on_xz_slice_loaded)
        self.xz_loader_worker.start()
        # When updating the XZ view, request the XZ overlay.
        self.on_overlay_points_updated_xz(self.labels, self.computed_labels)
    
    def on_xz_slice_loaded(self, xz_image):
        self.xz_view.setImage(xz_image)
        # exporting data of image view object
        current_timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.xz_view.export(os.path.join("GraphLabelerViews", f"xz_view_{current_timestamp}.png"))
        print("OME-Zarr XZ view updated with new slice.")
    
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
