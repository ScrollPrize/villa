import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import pyqtgraph as pg
from tqdm import tqdm
from scipy.interpolate import interp1d
import cv2

# --- Helper Functions ---

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
            # The umbilicus data is in order (y, z, x).
            # For each z slice, we interpolate using z as the independent variable.
            centers = []  # will hold (x, y) for each z
            for z_val in range(z_dim):
                pos = umbilicus_xy_at_z(umbilicus_data, z_val)
                centers.append(pos)
            centers = np.array(centers)  # shape: (z_dim, 2); column 0: x, column 1: y

            # Choose a half-length for the sampling line.
            L = max(x_dim, y_dim) / 2.0
            line_positions = np.arange(-L, L+1)
            angle_rad = np.deg2rad(self.finit_center_value)

            # For each z slice, compute sampling coordinates relative to the umbilicus center.
            # centers[:,0] is x, centers[:,1] is y.
            xs = centers[:,0][:, None] + line_positions[None, :] * np.cos(angle_rad)
            ys = centers[:,1][:, None] + line_positions[None, :] * np.sin(angle_rad)
            xs_int = np.rint(xs).astype(int)
            ys_int = np.rint(ys).astype(int)
            xs_int = np.clip(xs_int, 0, x_dim - 1)
            ys_int = np.clip(ys_int, 0, y_dim - 1)

            # Create a column vector for z indices.
            z_idx = np.arange(z_dim)[:, None]
            xz_image = dset[z_idx, ys_int, xs_int]  # shape: (z_dim, num_samples)
            # Flip vertically so that z layer 0 appears at the bottom.
            xz_image = np.flipud(xz_image)
        except Exception as e:
            print(f"Error loading XZ slice with finit center {self.finit_center_value}: {e}")
            xz_image = np.zeros((512,512), dtype=np.uint8)
        self.xz_slice_loaded.emit(xz_image)

# --- Main OME-Zarr View Window ---
class OmeZarrViewWindow(QMainWindow):
    def __init__(self, graph_labels, solver_interface, experiment_path, ome_zarr_path, umbilicus_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OME-Zarr Views")
        
        self.graph_labels = graph_labels
        self.solver_interface = solver_interface
        self.experiment_path = experiment_path
        self.ome_zarr_path = ome_zarr_path
        self.umbilicus_path = umbilicus_path
        
        # For XY view updates.
        self.pending_z_center = None
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._trigger_z_slice_update)
        
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
        
        # Add a scatter plot item for the umbilicus position in the XY view.
        self.umbilicus_dot = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush('r'), size=20)
        self.xy_view.addItem(self.umbilicus_dot)
    
    def load_placeholder_images(self):
        red_image = np.zeros((512,512,3), dtype=np.uint8)
        red_image[..., 0] = 255
        self.xy_view.setImage(red_image)
        self.xz_view.setImage(red_image)
    
    # --- XY view update ---
    def update_z_slice_center(self, z_center_value):
        self.pending_z_center = z_center_value
        self.debounce_timer.start(1000)
    
    def _trigger_z_slice_update(self):
        if self.pending_z_center is None:
            return
        # Compute z_index from the slider value.
        z_index = int(self.pending_z_center * 4 - 500)
        self.current_z_index = z_index  # Store for later use.
        print(f"Loading OME-Zarr XY slice at index {z_index} (from z slice center {self.pending_z_center})")
        self.loader_worker = OmeZarrLoaderWorker(self.ome_zarr_path, z_index)
        self.loader_worker.slice_loaded.connect(self.on_slice_loaded)
        self.loader_worker.start()
    
    def on_slice_loaded(self, image_slice):
        print("OME-Zarr XY view updated with new slice.")
        # Draw the umbilicus dot on the XY view.
        try:
            umbilicus_data = load_xyz_from_file(self.umbilicus_path) - 500
            # Interpolate the umbilicus XY (from umbilicus data, which is (y, z, x)) for the current z_index.
            pos = umbilicus_xy_at_z(umbilicus_data, self.current_z_index)
            # cv2 draw circle: center, radius, color, thickness
            image_slice = cv2.circle(image_slice, (int(pos[0]), int(pos[1])), 15, (255, 0, 0), -1)
            
        except Exception as e:
            print("Error updating umbilicus dot:", e)
        self.xy_view.setImage(image_slice)
    
    # --- XZ view update ---
    def update_finit_center(self, finit_center_value):
        self.pending_finit_center = finit_center_value
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
