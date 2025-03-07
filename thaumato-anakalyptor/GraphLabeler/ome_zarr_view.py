import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QComboBox, QPushButton, QLabel, QProgressDialog, QSpinBox, QDoubleSpinBox, QCheckBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject, pyqtSlot, Qt, QEvent, QPointF
import pyqtgraph as pg
from scipy.interpolate import interp1d
import cv2
import h5py
import pickle
import time
import os
import ast

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

def load_xyz_from_file(filename='umbilicus.txt'):
    """
    Load a file with comma-separated xyz coordinates into a 2D numpy array.
    """
    return np.loadtxt(filename, delimiter=',')

def umbilicus_xy_at_z(points_array, z_val):
    """
    Given umbilicus data in the order (y, z, x), interpolate using the z values
    to obtain the corresponding (x, y) coordinate for the current z.
    
    :param points_array: A 2D numpy array of shape (n, 3) with columns (y, z, x).
    :param z_val: The z value at which to interpolate.
    :return: A 1D numpy array [x, y] for the rotation center.
    """
    Y = points_array[:, 0]
    Z = points_array[:, 1]
    X = points_array[:, 2]
    fy = interp1d(Z, Y, kind='linear', fill_value="extrapolate")
    fx = interp1d(Z, X, kind='linear', fill_value="extrapolate")
    return np.array([fx(z_val), fy(z_val)])

def brush_for_winding(w):
    """
    Given a winding value w (assumed numeric), return a pyqtgraph brush
    using a three-color scheme: winding 1 -> red, 2 -> green, 3 -> blue,
    4 -> red, etc.
    """
    # Round to nearest integer.
    w_int = int(round(w / 360.0))
    mod = w_int % 3
    if mod == 1:
        return pg.mkBrush(255, 0, 0)   # red
    elif mod == 2:
        return pg.mkBrush(0, 255, 0)   # green
    else:  # mod == 0
        return pg.mkBrush(0, 0, 255)   # blue

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
        import zarr
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
        import zarr
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
            xz_image = xz_image.T  # shape: (num_samples, z_dim)
        except Exception as e:
            print(f"Error loading XZ slice with finit center {self.finit_center_value}: {e}")
            xz_image = np.zeros((512,512), dtype=np.uint8)
        self.xz_slice_loaded.emit(xz_image)

# --- Persistent Overlay Worker ---
class PersistentScrollGraphWorker(QObject):
    # This worker now emits a tuple: (overlay_points, overlay_windings)
    overlay_points_computed = pyqtSignal(object)

    def __init__(self, graph_pkl_path, parent=None):
        super().__init__(parent)
        # Load the scroll graph (from pickle) once.
        self.scroll_graph = load_graph_pkl(graph_pkl_path)

    @pyqtSlot(int, str, np.ndarray, np.ndarray, np.ndarray, int)
    def compute_overlay(self, z_index, h5_path, labels, f_init, undeleted_nodes_indices, block_size):
        try:
            # Assume get_points_XY returns a tuple: (points, windings)
            overlay_points, overlay_windings = self.scroll_graph.get_points_XY(
                z_index, h5_path, labels, f_init, undeleted_nodes_indices, block_size
            )
            self.overlay_points_computed.emit((overlay_points, overlay_windings))
        except Exception as e:
            print("Error in PersistentScrollGraphWorker:", e)
            self.overlay_points_computed.emit((np.empty((0, 3)), np.empty((0, 1))))

########################################
# Main OME-Zarr View Window
########################################

class OmeZarrViewWindow(QMainWindow):
    # Signal to request overlay computation.
    overlay_request = pyqtSignal(int, str, np.ndarray, np.ndarray, np.ndarray, int)

    def __init__(self, graph_labels, solver, experiment_path, ome_zarr_path,
                 graph_pkl_path, h5_path, umbilicus_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OME-Zarr Views")
        # Ensure the window is deleted on close.
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        self.graph_labels = graph_labels
        self.solver = solver
        self.experiment_path = experiment_path
        self.ome_zarr_path = ome_zarr_path
        self.graph_pkl_path = graph_pkl_path
        self.h5_path = h5_path
        self.umbilicus_path = umbilicus_path
        self.umbilicus_data = load_xyz_from_file(self.umbilicus_path) - 500
        
        # For XY view updates.
        self.pending_z_center = None
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        # Set debounce to 5000 ms (5 seconds)
        self.debounce_timer.timeout.connect(self._trigger_z_slice_update)
        self.loader_worker_running = False  # Flag to indicate an update is in progress.
        
        # For XZ view updates.
        self.pending_finit_center = None
        self.finit_debounce_timer = QTimer(self)
        self.finit_debounce_timer.setSingleShot(True)
        self.finit_debounce_timer.timeout.connect(self._trigger_xz_slice_update)
        
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
        self.persistent_overlay_worker = PersistentScrollGraphWorker(self.graph_pkl_path)
        self.persistent_overlay_worker.moveToThread(self.overlay_thread)
        self.overlay_thread.start()
        self.persistent_overlay_worker.overlay_points_computed.connect(self.on_overlay_points_computed)
        self.overlay_request.connect(self.persistent_overlay_worker.compute_overlay)
        
        # We'll store the last computed overlay windings to support dummy color updates.
        self.last_overlay_windings = None

    def load_placeholder_images(self):
        red_image = np.zeros((512,512,3), dtype=np.uint8)
        red_image[..., 0] = 255
        self.xy_view.setImage(red_image)
        self.xz_view.setImage(red_image)
    
    # --- XY view update ---
    def update_z_slice_center(self, z_center_value):
        self.pending_z_center = z_center_value
        # Stop and restart the debounce timer (5 seconds)
        self.debounce_timer.stop()
        self.debounce_timer.start(5000)
    
    def _trigger_z_slice_update(self):
        print("Triggering z slice update.")
        # Only start if no update is currently running.
        if self.loader_worker_running:
            print("Update already in progress; waiting for it to finish.")
            return
        
        if self.pending_z_center is None:
            return
        
        # Compute z_index from the slider value.
        z_index = int(self.pending_z_center * 4 - 500)
        self.current_z_index = z_index  # Store for later use.
        self.loader_worker_running = True
        print(f"Loading OME-Zarr XY slice at index {z_index} (from z slice center {self.pending_z_center})")
        self.loader_worker = OmeZarrLoaderWorker(self.ome_zarr_path, z_index)
        self.loader_worker.slice_loaded.connect(self.on_slice_loaded)
        self.loader_worker.start()
    
    def on_slice_loaded(self, image_slice):
        self.xy_view.setImage(image_slice)
        print("OME-Zarr XY view updated with new slice.")
        try:
            pos = umbilicus_xy_at_z(self.umbilicus_data, self.current_z_index)
            # pos is in (x, y) order.
            self.umbilicus_dot.setData([pos[1]], [pos[0]])
        except Exception as e:
            print("Error updating umbilicus dot:", e)
        
        # Request overlay points from the persistent overlay worker.
        try:
            undeleted_nodes_indices = np.array(self.solver.get_undeleted_indices())
            labels = np.array(self.solver.get_labels())
            f_init = np.array(self.solver.get_positions())[:, 1]
            self.overlay_request.emit(self.current_z_index, self.h5_path, labels, f_init, undeleted_nodes_indices, 50)
        except Exception as e:
            print("Error requesting overlay points:", e)
        
        # Mark the current update as finished.
        self.loader_worker_running = False
        # If the slider value has changed during processing, schedule a new update.
        new_z_index = int(self.pending_z_center * 4 - 500)
        if new_z_index != self.current_z_index:
            print("Slider value changed during update; scheduling a new update.")
            self.debounce_timer.start(5000)
    
    def on_overlay_points_computed(self, overlay_data):
        # overlay_data is a tuple: (overlay_points, overlay_windings)
        overlay_points, overlay_windings = overlay_data
        print(f"Overlay points computed: {overlay_points.shape} unique points found.")
        # Store the windings for dummy color update.
        self.last_overlay_windings = overlay_windings

        # Compute brushes for each winding using the 3-color mapping.
        brushes = []
        # Assume overlay_windings is an array of shape (N, 1) or (N,)
        for w in overlay_windings.flatten():
            brushes.append(brush_for_winding(w))
        
        # Extract x and y coordinates.
        # Here we assume overlay_points columns: [z, x, y, ...]
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
    
    # --- Dummy Color Update Trigger ---
    def dummy_update_overlay_colors(self):
        """
        When labels change, this dummy update will update the overlay colors by
        adding an offset of 1 to the winding value before mapping.
        """
        if hasattr(self, "overlay_scatter") and self.last_overlay_windings is not None:
            new_brushes = []
            for w in self.last_overlay_windings.flatten():
                new_w = w + 1  # add offset of 1
                new_brushes.append(brush_for_winding(new_w))
            print("Dummy updating overlay colors.")
            self.overlay_scatter.setBrush(new_brushes)
    
    # --- XZ view update ---
    def update_finit_center(self, finit_center_value):
        self.pending_finit_center = finit_center_value
        self.finit_debounce_timer.stop()
        self.finit_debounce_timer.start(1000)
    
    def _trigger_xz_slice_update(self):
        if self.pending_finit_center is None:
            return
        print(f"Loading OME-Zarr XZ slice with f init center {self.pending_finit_center}")
        self.xz_loader_worker = OmeZarrXZLoaderWorker(self.ome_zarr_path, self.pending_finit_center, self.umbilicus_path)
        self.xz_loader_worker.xz_slice_loaded.connect(self.on_xz_slice_loaded)
        self.xz_loader_worker.start()
    
    def on_xz_slice_loaded(self, xz_image):
        self.xz_view.setImage(xz_image)
        print("OME-Zarr XZ view updated with new slice.")
    
    def closeEvent(self, event):
        # Stop any active timers.
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        if self.finit_debounce_timer.isActive():
            self.finit_debounce_timer.stop()
        # Stop persistent worker thread.
        if hasattr(self, "overlay_thread") and self.overlay_thread.isRunning():
            self.overlay_thread.quit()
            self.overlay_thread.wait()
        print("OME-Zarr View Window is closing.")
        super().closeEvent(event)
