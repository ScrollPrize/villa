import sys, os, ast, numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QAction, QMessageBox, QInputDialog, QGraphicsEllipseItem, QFileDialog,
    QProgressDialog, QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QComboBox
)
from PyQt5.QtCore import Qt, QEvent, QPointF
import pyqtgraph as pg
from scipy.spatial import cKDTree
import time
from tqdm import tqdm

# --------------------------------------------------
# Importing the custom graph problem library.
# --------------------------------------------------
sys.path.append('../ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

from config import load_config, save_config
from utils import vectorized_point_to_polyline_distance
from widgets import create_sync_slider_spinbox

# --------------------------------------------------
# Main: Create GUI.
# --------------------------------------------------
class PointCloudLabeler(QMainWindow):
    def __init__(self, point_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Graph Labeler")
        
        # Load configuration (or fall back to defaults)
        self.config = load_config("config_labeling_gui.json")
        self.graph_path = self.config.get("graph_path", "/media/julian/2/Scroll5/scroll5_complete_surface_points_zarrtest/1352_3600_5005/graph.bin")
        self.default_experiment = self.config.get("default_experiment", "denominator3-rotated")
        self.ome_zarr_path = self.config.get("ome_zarr_path", None)
        self.graph_pkl_path = self.config.get("graph_pkl_path", None)
        self.h5_path = self.config.get("h5_path", None)
        self.umbilicus_path = self.config.get("umbilicus_path", None)


        # Initialize solver using SolverInterface if no external point data is provided.
        if point_data is None:
            self.solver = graph_problem_gpu_py.Solver(self.graph_path)
            gt_path = os.path.join("../experiments", self.default_experiment,
                                   "checkpoints", "checkpoint_graph_solver_connected_2.bin")
            if not os.path.exists(gt_path):
                gt_path = os.path.join("../experiments", self.default_experiment,
                                       "checkpoints", "checkpoint_graph_f_star_final.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path)
            else:
                print("Default graph file not found; continuing without loading.")
            point_data = self.solver.get_positions()
        else:
            self.solver = None
        self.seed_node = None # master node, this is the fixed node, to which all other nodes are fixed in f_star to
        self.recompute = True

        # Global variables and state.
        self.scaleFactor = 100
        self.s_pressed = False
        self.original_drawing_mode = True
        self.pipette_mode = False
        self.calc_drawing_mode = False
        self.UNLABELED = -9999
        self.undo_stack = []
        self.redo_stack = []
        self._stroke_backup = None
        
        # Spline storage.
        self.spline_items = []
        self.spline_segments = {}
        
        # Create menu.
        self._create_menu()
        
        # Data and labels.
        self.points = np.array(point_data)
        self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        self.calculated_labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        
        # Display parameters.
        self.point_size = 3
        self.max_display = 200000
        self.f_star_min, self.f_star_max = float(np.min(self.points[:, 0])), float(np.max(self.points[:, 0]))
        self.f_init_min, self.f_init_max = -180.0, 180.0
        self.z_min, self.z_max = float(np.min(self.points[:, 2])), float(np.max(self.points[:, 2]))
        
        # Create KD-trees.
        self.kdtree_xy = cKDTree(self.points[:, [0, 1]])
        self.kdtree_xz = cKDTree(self.points[:, [0, 2]])
        
        # Pre-created brushes.
        self.brush_black = pg.mkBrush(0, 0, 0)
        self.brush_red   = pg.mkBrush(255, 0, 0)
        self.brush_green = pg.mkBrush(0, 255, 0)
        self.brush_blue  = pg.mkBrush(0, 0, 255)
        self.calc_brush_black = pg.mkBrush(0, 0, 0, 100)
        self.calc_brush_red   = pg.mkBrush(255, 0, 0, 100)
        self.calc_brush_green = pg.mkBrush(0, 255, 0, 100)
        self.calc_brush_blue  = pg.mkBrush(0, 0, 255, 100)
        
        # Guide lines and indicators.
        self.line_finit_neg = pg.InfiniteLine(pos=-180, angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_pos = pg.InfiniteLine(pos=180, angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_center = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_upper  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_lower  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_center = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_upper  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_lower  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        # Orange indicator (for XZ shear) originally belonged to XZ view;
        # now we switch it to the XY view.
        self.xz_shear_indicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('orange', width=1, style=Qt.DashLine))
        # Purple indicator (for XY horizontal shear) is now switched to the XZ view.
        self.xy_horizontal_indicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
        
        # Cursor circle.
        self.cursor_circle = QGraphicsEllipseItem(0, 0, 0, 0)
        self.cursor_circle.setPen(pg.mkPen('cyan', width=1, style=Qt.DashLine))
        self.cursor_circle.setVisible(False)
        
        # Layout setup.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Upper view area.
        views_columns_layout = QHBoxLayout()
        main_layout.addLayout(views_columns_layout)
        
        # Left (XY) view.
        left_column = QVBoxLayout()
        views_columns_layout.addLayout(left_column)
        self.xy_plot = pg.PlotWidget()
        self.xy_plot.setBackground('w')
        self.xy_plot.setLabel('bottom', 'f_star')
        self.xy_plot.setLabel('left', 'f_init')
        self.xy_plot.setYRange(-270, 270)
        self.xy_plot.setMouseEnabled(x=True, y=True)
        left_column.addWidget(self.xy_plot)
        self.xy_plot.addItem(self.cursor_circle)
        xy_controls = QHBoxLayout()
        self.z_center_widget, self.z_center_slider, self.z_center_spinbox = create_sync_slider_spinbox(
            "Z slice center:", self.z_min, self.z_max, (self.z_min + self.z_max) / 2, self.scaleFactor, self.z_slice_center_changed)
        self.z_thickness_widget, self.z_thickness_slider, self.z_thickness_spinbox = create_sync_slider_spinbox(
            "Z slice thickness:", 0.01, self.z_max - self.z_min, (self.z_max - self.z_min) * 0.1, self.scaleFactor, self.update_views)
        xy_controls.addWidget(self.z_center_widget)
        xy_controls.addWidget(self.z_thickness_widget)
        left_column.addLayout(xy_controls)
        # Add two new shear controls for the XY view:
        # Vertical shear (rotating around the f_star axis)
        xy_vertical_shear_layout = QHBoxLayout()
        self.xy_vertical_shear_widget, self.xy_vertical_shear_slider, self.xy_vertical_shear_spinbox = create_sync_slider_spinbox(
            "XY Vertical Shear (°):", -90.0, 90.0, 0.0, self.scaleFactor, self.update_views, decimals=1)
        xy_vertical_shear_layout.addWidget(self.xy_vertical_shear_widget)
        left_column.addLayout(xy_vertical_shear_layout)
        # Horizontal shear (rotating around the f_init axis)
        xy_horizontal_shear_layout = QHBoxLayout()
        self.xy_horizontal_shear_widget, self.xy_horizontal_shear_slider, self.xy_horizontal_shear_spinbox = create_sync_slider_spinbox(
            "XY Horizontal Shear (°):", -90.0, 90.0, 0.0, self.scaleFactor, self.update_views, decimals=1)
        xy_horizontal_shear_layout.addWidget(self.xy_horizontal_shear_widget)
        left_column.addLayout(xy_horizontal_shear_layout)
        self.apply_calc_xy_button = QPushButton("Apply Updated Labels to XY")
        self.apply_calc_xy_button.clicked.connect(self.apply_calculated_labels_xy)
        left_column.addWidget(self.apply_calc_xy_button)
        
        # Right (XZ) view.
        right_column = QVBoxLayout()
        views_columns_layout.addLayout(right_column)
        self.xz_plot = pg.PlotWidget()
        self.xz_plot.setBackground('w')
        self.xz_plot.setLabel('bottom', 'f_star')
        self.xz_plot.setLabel('left', 'Z')
        self.xz_plot.setMouseEnabled(x=True, y=True)
        right_column.addWidget(self.xz_plot)
        xz_controls = QHBoxLayout()
        self.finit_center_widget, self.finit_center_slider, self.finit_center_spinbox = create_sync_slider_spinbox(
            "f init center:", float(np.min(self.points[:, 1])), float(np.max(self.points[:, 1])),
            (np.min(self.points[:, 1]) + np.max(self.points[:, 1])) / 2, self.scaleFactor, self.f_init_center_changed)
        self.finit_thickness_widget, self.finit_thickness_slider, self.finit_thickness_spinbox = create_sync_slider_spinbox(
            "f init thickness:", 0.01, float(np.max(self.points[:, 1]) - np.min(self.points[:, 1])), 5.0, self.scaleFactor, self.update_views)
        xz_controls.addWidget(self.finit_center_widget)
        xz_controls.addWidget(self.finit_thickness_widget)
        right_column.addLayout(xz_controls)
        # XZ shear control (unchanged functionality)
        xz_shear_layout = QHBoxLayout()
        self.xz_shear_widget, self.xz_shear_slider, self.xz_shear_spinbox = create_sync_slider_spinbox(
            "XZ Shear (°):", -90.0, 90.0, 0.0, self.scaleFactor, self.update_views, decimals=1)
        xz_shear_layout.addWidget(self.xz_shear_widget)
        right_column.addLayout(xz_shear_layout)
        self.apply_calc_xz_button = QPushButton("Apply Updated Labels to XZ")
        self.apply_calc_xz_button.clicked.connect(self.apply_calculated_labels_xz)
        right_column.addWidget(self.apply_calc_xz_button)
        
        # --------------------------------------------------------------------
        # Top Controls Row: Spline and Label Update Controls.
        # --------------------------------------------------------------------
        top_controls_layout = QHBoxLayout()
        self.update_labels_button = QPushButton("Update Labels")
        self.update_labels_button.clicked.connect(self.update_labels)
        top_controls_layout.addWidget(self.update_labels_button)

        # --- Solver selection dropdown ---
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["F*", "F*2", "F*3", "F*4", "Winding Number", "Union", "Random", "Set Labels"])
        top_controls_layout.addWidget(QLabel("Select Solver:"))
        top_controls_layout.addWidget(self.solver_combo)

        self.use_z_range_checkbox = QCheckBox("Solve in Z range")
        self.use_z_range_checkbox.setChecked(False)
        top_controls_layout.addWidget(self.use_z_range_checkbox)

        solve_iterations_layout = QHBoxLayout()
        self.solve_iterations_spinbox = QSpinBox()
        self.solve_iterations_spinbox.setRange(10, 1000000)
        self.solve_iterations_spinbox.setValue(15000)
        solve_iterations_layout.addWidget(QLabel("Solver Iterations:"))
        solve_iterations_layout.addWidget(self.solve_iterations_spinbox)
        top_controls_layout.addLayout(solve_iterations_layout)
        
        spline_min_layout = QHBoxLayout()
        self.spline_min_points_spinbox = QSpinBox()
        self.spline_min_points_spinbox.setRange(10, 10000)
        self.spline_min_points_spinbox.setValue(100)
        spline_min_layout.addWidget(QLabel("Min points for spline:"))
        spline_min_layout.addWidget(self.spline_min_points_spinbox)
        top_controls_layout.addLayout(spline_min_layout)
        
        self.update_spline_button = QPushButton("Update Spline")
        self.update_spline_button.clicked.connect(self.update_winding_splines)
        top_controls_layout.addWidget(self.update_spline_button)
        
        self.clear_splines_button = QPushButton("Clear Splines")
        self.clear_splines_button.clicked.connect(self.clear_splines)
        top_controls_layout.addWidget(self.clear_splines_button)
        
        self.disregard_label0_checkbox = QCheckBox("Disregard label 0")
        self.disregard_label0_checkbox.setChecked(True)
        top_controls_layout.addWidget(self.disregard_label0_checkbox)
        
        self.assign_line_labels_button = QPushButton("Assign Line Labels")
        self.assign_line_labels_button.clicked.connect(self.assign_line_labels)
        top_controls_layout.addWidget(self.assign_line_labels_button)
        
        line_dist_layout = QHBoxLayout()
        self.line_distance_threshold_spinbox = QSpinBox()
        self.line_distance_threshold_spinbox.setRange(1, 100)
        self.line_distance_threshold_spinbox.setValue(4)
        line_dist_layout.addWidget(QLabel("Line dist thresh:"))
        line_dist_layout.addWidget(self.line_distance_threshold_spinbox)
        top_controls_layout.addLayout(line_dist_layout)
        
        effective_range_layout = QHBoxLayout()
        self.assign_min_spinbox = QSpinBox()
        self.assign_min_spinbox.setRange(-1000, 1000)
        self.assign_min_spinbox.setValue(-1000)
        self.assign_max_spinbox = QSpinBox()
        self.assign_max_spinbox.setRange(-1000, 1000)
        self.assign_max_spinbox.setValue(1000)
        effective_range_layout.addWidget(QLabel("Spline winding range min:"))
        effective_range_layout.addWidget(self.assign_min_spinbox)
        effective_range_layout.addWidget(QLabel("max:"))
        effective_range_layout.addWidget(self.assign_max_spinbox)
        top_controls_layout.addLayout(effective_range_layout)
        
        main_layout.addLayout(top_controls_layout)
        
        # --------------------------------------------------------------------
        # Bottom Controls Row: Common Drawing and File Controls.
        # --------------------------------------------------------------------
        common_controls_layout = QHBoxLayout()
        self.radius_widget, self.radius_slider, self.radius_spinbox = create_sync_slider_spinbox(
            "Drawing radius:", 0.1, 20.0, 3.5, self.scaleFactor, self.update_views, decimals=1)
        common_controls_layout.addWidget(self.radius_widget)
        
        max_disp_layout = QHBoxLayout()
        max_disp_label = QLabel("Max Display Points:")
        self.max_display_spinbox = QSpinBox()
        self.max_display_spinbox.setRange(1000, 1000000)
        self.max_display_spinbox.setValue(self.max_display)
        max_disp_layout.addWidget(max_disp_label)
        max_disp_layout.addWidget(self.max_display_spinbox)
        common_controls_layout.addLayout(max_disp_layout)
        self.max_display_spinbox.valueChanged.connect(self.update_max_display)
        
        self.drawing_mode_checkbox = QCheckBox("Drawing Mode")
        self.drawing_mode_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.drawing_mode_checkbox)
        self.drawing_mode_checkbox.toggled.connect(self.update_drawing_mode)
        
        self.show_guides_checkbox = QCheckBox("Show guides")
        self.show_guides_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.show_guides_checkbox)
        self.show_guides_checkbox.toggled.connect(self.update_guides)
        
        self.pipette_button = QPushButton("Pipette")
        self.pipette_button.clicked.connect(self.activate_pipette)
        common_controls_layout.addWidget(self.pipette_button)
        
        label_save_layout = QHBoxLayout()
        self.label_spinbox = QSpinBox()
        self.label_spinbox.setRange(-10000, 1000)
        self.label_spinbox.setValue(1)
        label_save_layout.addWidget(QLabel("Label:"))
        label_save_layout.addWidget(self.label_spinbox)
        common_controls_layout.addLayout(label_save_layout)

        self.erase_button = QPushButton("Erase Label")
        # set label to unlabeled if erase button is checked
        self.erase_button.clicked.connect(lambda: self.label_spinbox.setValue(self.UNLABELED))
        common_controls_layout.addWidget(self.erase_button)
        
        self.calc_draw_button = QPushButton("Updated Labels Draw Mode: Off")
        self.calc_draw_button.setCheckable(True)
        self.calc_draw_button.clicked.connect(self.toggle_calc_draw_mode)
        common_controls_layout.addWidget(self.calc_draw_button)
        
        self.apply_all_calc_button = QPushButton("Apply All Updated Labels")
        self.apply_all_calc_button.clicked.connect(self.apply_all_calculated_labels)
        common_controls_layout.addWidget(self.apply_all_calc_button)
        
        self.clear_calc_button = QPushButton("Clear Updated Labels")
        self.clear_calc_button.clicked.connect(self.clear_calculated_labels)
        common_controls_layout.addWidget(self.clear_calc_button)
        
        self.save_graph_button = QPushButton("Save Labeled Graph")
        self.save_graph_button.clicked.connect(self.save_graph)
        common_controls_layout.addWidget(self.save_graph_button)
        
        main_layout.addLayout(common_controls_layout)
        
        # Create scatter items for displaying points.
        self.xy_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xz_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xy_plot.addItem(self.xy_scatter)
        self.xz_plot.addItem(self.xz_scatter)
        self.xy_calc_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xz_calc_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xy_plot.addItem(self.xy_calc_scatter)
        self.xz_plot.addItem(self.xz_calc_scatter)
        
        self.xy_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        self.xz_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        
        self._enable_pencil(self.xy_plot, self.xy_scatter, view_name='xy')
        self._enable_pencil(self.xz_plot, self.xz_scatter, view_name='xz')
        
        self.xy_plot.scene().installEventFilter(self)
        
        self.update_guides()
        self.update_views()

    def z_slice_center_changed(self):
        if hasattr(self, 'ome_zarr_window') and self.ome_zarr_window is not None:
            self.ome_zarr_window.update_z_slice_center(self.z_center_spinbox.value())
        self.update_views()

    def f_init_center_changed(self):
        if hasattr(self, 'ome_zarr_window') and self.ome_zarr_window is not None:
            self.ome_zarr_window.update_finit_center(self.finit_center_spinbox.value())
        self.update_views()

    # Instead, call save_config on close.
    def closeEvent(self, event):
        self.config["graph_path"] = self.graph_path
        self.config["default_experiment"] = self.default_experiment
        if self.ome_zarr_path:
            self.config["ome_zarr_path"] = self.ome_zarr_path  # Save the OME-Zarr path
        if self.graph_pkl_path:
            self.config["graph_pkl_path"] = self.graph_pkl_path
        if self.h5_path:
            self.config["h5_path"] = self.h5_path
        if self.umbilicus_path:
            self.config["umbilicus_path"] = self.umbilicus_path
        save_config(self.config, "config_labeling_gui.json")
        event.accept()
    
    # --------------------------------------------------
    # Event filter for cursor circle.
    # --------------------------------------------------
    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseMove and source is self.xy_plot.scene():
            if self.drawing_mode_checkbox.isChecked():
                pos = event.scenePos()
                dataPos = self.xy_plot.plotItem.vb.mapSceneToView(pos)
                r = self.radius_spinbox.value()
                self.cursor_circle.setRect(dataPos.x() - r, dataPos.y() - r, 2 * r, 2 * r)
                self.cursor_circle.setVisible(True)
            else:
                self.cursor_circle.setVisible(False)
            return False
        return super(PointCloudLabeler, self).eventFilter(source, event)
    
    # --------------------------------------------------
    # Setup menu.
    # --------------------------------------------------
    def _create_menu(self):
        menu_bar = self.menuBar()
        data_menu = menu_bar.addMenu("Data")
        load_action = QAction("Load Data", self)
        load_action.triggered.connect(self.load_data)
        data_menu.addAction(load_action)
        save_labels_action = QAction("Save Labels", self)
        save_labels_action.triggered.connect(self.save_labels_to_path)
        data_menu.addAction(save_labels_action)
        load_labels_action = QAction("Load Labels", self)
        load_labels_action.triggered.connect(self.load_labels_from_path)
        data_menu.addAction(load_labels_action)
        
        set_ome_zarr_action = QAction("Set OME-Zarr Path", self)
        set_ome_zarr_action.triggered.connect(self.set_ome_zarr_path)
        data_menu.addAction(set_ome_zarr_action)

        set_graph_pkl_action = QAction("Set Graph.pkl Path", self)
        set_graph_pkl_action.triggered.connect(self.set_graph_pkl_path)
        data_menu.addAction(set_graph_pkl_action)

        set_h5_action = QAction("Set H5 Path", self)
        set_h5_action.triggered.connect(self.set_h5_path)
        data_menu.addAction(set_h5_action)

        set_umbilicus_action = QAction("Set Umbilicus Path", self)
        set_umbilicus_action.triggered.connect(self.set_umbilicus_path)
        data_menu.addAction(set_umbilicus_action)
        
        # Existing "View" menu for OME-Zarr window…
        view_menu = menu_bar.addMenu("View")
        ome_zarr_action = QAction("Open OME-Zarr View", self)
        ome_zarr_action.triggered.connect(self.open_ome_zarr_view_window)
        view_menu.addAction(ome_zarr_action)
        
        help_menu = menu_bar.addMenu("Help")
        usage_action = QAction("Usage", self)
        usage_action.triggered.connect(self.show_help)
        help_menu.addAction(usage_action)

    def set_ome_zarr_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select OME-Zarr Directory", self.ome_zarr_path or os.getcwd())
        if directory:
            self.ome_zarr_path = directory
            self.config["ome_zarr_path"] = directory
            save_config(self.config, "config_labeling_gui.json")

    def set_graph_pkl_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Graph .pkl File", self.graph_pkl_path or os.getcwd(), "Pickle Files (*.pkl);;All Files (*)")
        if fname:
            self.graph_pkl_path = fname
            self.config["graph_pkl_path"] = fname
            save_config(self.config, "config_labeling_gui.json")

    def set_h5_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select H5 File", self.h5_path or os.getcwd(), "HDF5 Files (*.h5);;All Files (*)")
        if fname:
            self.h5_path = fname
            self.config["h5_path"] = fname
            save_config(self.config, "config_labeling_gui.json")

    def set_umbilicus_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Umbilicus .txt File", self.umbilicus_path or os.getcwd(), "Text Files (*.txt);;All Files (*)")
        if fname:
            self.umbilicus_path = fname
            self.config["umbilicus_path"] = fname
            save_config(self.config, "config_labeling_gui.json")

    def open_ome_zarr_view_window(self):
        try:
            del sys.modules['ome_zarr_view']
        except KeyError:
            print("Module 'ome_zarr_view' not found in sys.modules; continuing.")
        try:
            del OmeZarrViewWindow
        except NameError:
            print("Name 'OmeZarrViewWindow' not found; continuing")
        from ome_zarr_view import OmeZarrViewWindow
        self.ome_zarr_window = OmeZarrViewWindow(
            graph_labels=self.labels,
            solver=self.solver,
            experiment_path=self.default_experiment,
            ome_zarr_path=self.ome_zarr_path,
            graph_pkl_path=self.graph_pkl_path,
            h5_path=self.h5_path,
            umbilicus_path=self.umbilicus_path
        )
        # Connect the destroyed signal so that when the window is closed,
        # we automatically set our pointer to None.
        self.ome_zarr_window.destroyed.connect(self.on_ome_zarr_view_destroyed)
        self.ome_zarr_window.show()

    def on_ome_zarr_view_destroyed(self):
        print("OME-Zarr view window has been destroyed; setting pointer to None.")
        self.ome_zarr_window = None
    
    def load_data(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Data")
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        bin_path_lineedit = QLineEdit()
        bin_path_lineedit.setText(os.path.dirname(self.graph_path))
        browse_button = QPushButton("Browse...")
        browse_button.setToolTip("Click to choose the bin folder")
        h_layout = QHBoxLayout()
        h_layout.addWidget(bin_path_lineedit)
        h_layout.addWidget(browse_button)
        form_layout.addRow("Bin Folder:", h_layout)
        exp_lineedit = QLineEdit()
        exp_lineedit.setText(self.default_experiment)
        form_layout.addRow("Experiment name:", exp_lineedit)
        layout.addLayout(form_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        browse_button.clicked.connect(lambda: self.browse_for_directory(bin_path_lineedit))
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        if dialog.exec_() == QDialog.Accepted:
            selected_dir = bin_path_lineedit.text().strip()
            exp_name = exp_lineedit.text().strip()
            self.default_experiment = exp_name
            if not selected_dir or not exp_name:
                return
            bin_file_path = os.path.join(selected_dir, "graph.bin")
            if not os.path.exists(bin_file_path):
                QMessageBox.warning(self, "Load Data", f"File {bin_file_path} does not exist.")
                return
            self.graph_path = bin_file_path
            self.solver = graph_problem_gpu_py.Solver(self.graph_path)
            gt_path = os.path.join("../experiments", exp_name, "checkpoints", "checkpoint_graph_solver_connected_2.bin")
            if not os.path.exists(gt_path):
                gt_path = os.path.join("../experiments", exp_name, "checkpoints", "checkpoint_graph_f_star_final.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path)
            else:
                QMessageBox.warning(self, "Load Data", f"Graph file not found at {gt_path}")

            self.update_positions(update_labels=True)

    def update_positions(self, update_labels=False, update_slide_ranges=True):
        self.recompute = True
        new_points = np.array(self.solver.get_positions())
        print(f"Points shape previous vs new: {self.points.shape} vs {new_points.shape}")
        self.points = np.array(new_points)
        if update_labels:
            self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
            self.calculated_labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        self.kdtree_xy = cKDTree(self.points[:, [0, 1]])
        self.kdtree_xz = cKDTree(self.points[:, [0, 2]])
        if update_slide_ranges:
            self.update_slider_ranges()
        self.update_views()
    
    def browse_for_directory(self, lineedit):
        directory = QFileDialog.getExistingDirectory(self, "Select Bin Folder", lineedit.text() or os.getcwd())
        if directory:
            lineedit.setText(directory)
    
    def save_labels_to_path(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Labels", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            with open(fname, "w") as f:
                f.write(str(self.labels.tolist()))
    
    def load_labels_from_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Labels", os.path.join("../experiments", self.default_experiment),
                                                 "Text Files (*.txt);;All Files (*)")
        if fname:
            with open(fname, "r") as f:
                data = f.read()
            try:
                new_labels = np.array(ast.literal_eval(data), dtype=np.int32)
            except Exception as e:
                QMessageBox.warning(self, "Load Labels", f"Error reading file: {e}")
                return
            if len(new_labels) == len(self.labels):
                self.labels = new_labels
                self.update_views()
            else:
                QMessageBox.warning(self, "Load Labels", "Loaded labels length does not match current data.")
    
    def show_help(self):
        help_text = (
            "Graph Labeler Usage Instructions\n"
            "================================\n\n"
            "1. Views:\n"
            "   • XY View (left): f_star (horizontal) vs. f_init (vertical).\n"
            "   • XZ View (right): f_star (horizontal) vs. Z (vertical).\n\n"
            "2. Slice Controls:\n"
            "   • Adjust Z slice (XY) and f init slice (XZ) using the sliders.\n\n"
            "3. Shear Controls:\n"
            "   • XZ Shear: Applies shear to the XZ view (x_new = f_star + shear*(finit - center)).\n"
            "     (Orange indicator now appears in the XY view.)\n"
            "   • XY Vertical Shear: In the XY view, shifts f_init based on (z - z_center).\n"
            "   • XY Horizontal Shear: In the XY view, shifts f_star based on (z - z_center).\n"
            "     (Purple indicator now appears in the XZ view.)\n\n"
            "4. Drawing:\n"
            "   • Use the mouse to manually label points. The active label is set in the 'Label' spinbox.\n\n"
            "5. Updated Label Tools:\n"
            "   • Update Labels, Clear Updated Labels, Apply Updated Labels to XY/XZ.\n\n"
            "6. Spline Tools (Top Row):\n"
            "   • Update Labels, Min points for spline, Update/Clear Spline, etc.\n\n"
            "7. Common Controls (Bottom Row):\n"
            "   • Drawing radius, Max Display Points, Drawing Mode, Pipette, Label selection, etc.\n\n"
            "8. Key Shortcuts:\n"
            "   • P: Pipette, U: Toggle Updated Labels Draw Mode, Ctrl+Z: Undo, Ctrl+Y: Redo.\n\n"
            "9. Saving:\n"
            "   • Use the Data menu to load data or save/load labels, and to save the final labeled graph.\n"
        )
        QMessageBox.information(self, "Usage Instructions", help_text)
    
    def update_slider_ranges(self):
        self.z_min = float(np.min(self.points[:, 2]))
        self.z_max = float(np.max(self.points[:, 2]))
        self.z_center_slider.setMinimum(int(self.z_min * self.scaleFactor))
        self.z_center_slider.setMaximum(int(self.z_max * self.scaleFactor))
        self.z_center_spinbox.setMinimum(self.z_min)
        self.z_center_spinbox.setMaximum(self.z_max)
        center_val = (self.z_min + self.z_max) / 2
        self.z_center_slider.setValue(int(center_val * self.scaleFactor))
        self.z_center_spinbox.setValue(center_val)
        z_range = self.z_max - self.z_min
        self.z_thickness_slider.setMinimum(int(0.01 * self.scaleFactor))
        self.z_thickness_slider.setMaximum(int(z_range * self.scaleFactor))
        self.z_thickness_spinbox.setMinimum(0.01)
        self.z_thickness_spinbox.setMaximum(z_range)
        thickness_val = z_range * 0.1
        self.z_thickness_slider.setValue(int(thickness_val * self.scaleFactor))
        self.z_thickness_spinbox.setValue(thickness_val)
    
    def update_max_display(self, val):
        self.max_display = val
        self.update_views()
    
    def update_drawing_mode(self, checked):
        if checked:
            self.xy_scatter.setAcceptedMouseButtons(Qt.LeftButton)
            self.xz_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        else:
            self.xy_scatter.setAcceptedMouseButtons(Qt.NoButton)
            self.xz_scatter.setAcceptedMouseButtons(Qt.NoButton)
    
    def displayed_label(self, i, y):
        lab = self.labels[i]
        if abs(lab - self.UNLABELED) < 2:
            lab = self.calculated_labels[i]
        if y < -180:
            return lab + 1
        elif y > 180:
            return lab - 1
        else:
            return lab
    
    def get_nearby_indices_xy(self, x, y, r):
        if y > 180:
            effective_y = y - 360
        elif y < -180:
            effective_y = y + 360
        else:
            effective_y = y
        return np.asarray(self.kdtree_xy.query_ball_point([x, effective_y], r=r), dtype=np.int32)
    
    def update_guides(self):
        # For the XY view:
        if self.show_guides_checkbox.isChecked():
            if self.line_finit_neg.scene() is None:
                self.xy_plot.addItem(self.line_finit_neg)
            if self.line_finit_pos.scene() is None:
                self.xy_plot.addItem(self.line_finit_pos)
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
            upper = finit_center + finit_thickness / 2
            lower = finit_center - finit_thickness / 2
            self.line_finit_center.setPos(finit_center)
            self.line_finit_upper.setPos(upper)
            self.line_finit_lower.setPos(lower)
            if self.line_finit_center.scene() is None:
                self.xy_plot.addItem(self.line_finit_center)
            if self.line_finit_upper.scene() is None:
                self.xy_plot.addItem(self.line_finit_upper)
            if self.line_finit_lower.scene() is None:
                self.xy_plot.addItem(self.line_finit_lower)
            # Add the orange indicator only if not added already.
            self.xz_shear_indicator.setAngle(self.xz_shear_spinbox.value())
            center_f_star = (self.f_star_min + self.f_star_max) / 2
            center_f_init = (self.f_init_min + self.f_init_max) / 2
            self.xz_shear_indicator.setPos(QPointF(center_f_star, center_f_init))
            if self.xz_shear_indicator.scene() is None:
                self.xy_plot.addItem(self.xz_shear_indicator)
        else:
            # Remove guide items if guides are turned off.
            for item in [self.line_finit_neg, self.line_finit_pos, self.line_finit_center,
                        self.line_finit_upper, self.line_finit_lower, self.xz_shear_indicator]:
                if item.scene() is not None:
                    self.xy_plot.removeItem(item)
        
        # For the XZ view:
        if self.show_guides_checkbox.isChecked():
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value() / self.scaleFactor
            self.line_z_center.setPos(z_center)
            self.line_z_upper.setPos(z_center + z_thickness / 2)
            self.line_z_lower.setPos(z_center - z_thickness / 2)
            for item in [self.line_z_center, self.line_z_upper, self.line_z_lower]:
                if item.scene() is None:
                    self.xz_plot.addItem(item)
            self.xy_horizontal_indicator.setAngle(self.xy_horizontal_shear_spinbox.value())
            center_f_star = (self.f_star_min + self.f_star_max) / 2
            center_z = (self.z_min + self.z_max) / 2
            self.xy_horizontal_indicator.setPos(QPointF(center_f_star, center_z))
            if self.xy_horizontal_indicator.scene() is None:
                self.xz_plot.addItem(self.xy_horizontal_indicator)
        else:
            for item in [self.line_z_center, self.line_z_upper, self.line_z_lower, self.xy_horizontal_indicator]:
                if item.scene() is not None:
                    self.xz_plot.removeItem(item)
    
    def downsample_points(self, pts, labels, calc_labels=None, max_display=1):
        n = pts.shape[0]
        if n > max_display:
            indices = np.linspace(0, n - 1, max_display, dtype=int)
            if calc_labels is not None:
                return pts[indices], labels[indices], calc_labels[indices]
            else:
                return pts[indices], labels[indices]
        else:
            if calc_labels is not None:
                return pts, labels, calc_labels
            else:
                return pts, labels
    
    def get_brushes_from_labels(self, labels_array):
        brushes = np.empty(labels_array.shape[0], dtype=object)
        mask_unlabeled = (labels_array == self.UNLABELED) | (labels_array == self.UNLABELED + 1) | (labels_array == self.UNLABELED - 1)
        brushes[mask_unlabeled] = self.brush_black
        mask_valid = ~mask_unlabeled
        valid_labels = labels_array[mask_valid]
        mod = valid_labels % 3
        valid_indices = np.where(mask_valid)[0]
        for i, idx in enumerate(valid_indices):
            if mod[i] == 0:
                brushes[idx] = self.brush_red
            elif mod[i] == 1:
                brushes[idx] = self.brush_green
            elif mod[i] == 2:
                brushes[idx] = self.brush_blue
        return brushes
    
    def _enable_pencil(self, plot_widget, scatter_item, view_name='xy'):
        scatter_item.mousePressEvent = lambda ev: self._on_mouse_press(ev, plot_widget, view_name)
        scatter_item.mouseMoveEvent = lambda ev: self._on_mouse_drag(ev, plot_widget, view_name)
        scatter_item.mouseReleaseEvent = lambda ev: self._on_mouse_release(ev, plot_widget, view_name)
    
    def _on_mouse_press(self, ev, plot_widget, view_name):
        if view_name == 'xy' and self.pipette_mode:
            ev.accept()
            self.pick_label_at(ev, plot_widget)
            self.pipette_mode = False
            return
        if view_name == 'xz' and self.pipette_mode:
            ev.accept()
            self.pick_label_at_xz(ev, plot_widget)
            self.pipette_mode = False
            return
        if (ev.button() == Qt.LeftButton and self.drawing_mode_checkbox.isChecked() and not self.s_pressed):
            ev.accept()
            if self._stroke_backup is None:
                self._stroke_backup = self.labels.copy()
            if self.calc_drawing_mode:
                self._paint_points_calculated(ev, plot_widget, view_name)
            else:
                self._paint_points(ev, plot_widget, view_name)
        else:
            ev.ignore()
    
    def _on_mouse_drag(self, ev, plot_widget, view_name):
        if (ev.buttons() & Qt.LeftButton and self.drawing_mode_checkbox.isChecked() and not self.s_pressed):
            ev.accept()
            if self.calc_drawing_mode:
                self._paint_points_calculated(ev, plot_widget, view_name)
            else:
                self._paint_points(ev, plot_widget, view_name)
        else:
            ev.ignore()
    
    def _on_mouse_release(self, ev, plot_widget, view_name):
        if ev.button() == Qt.LeftButton and self._stroke_backup is not None:
            self.undo_stack.append(self._stroke_backup)
            self.redo_stack = []
            self._stroke_backup = None
        ev.accept()
    
    def _paint_points(self, ev, plot_widget, view_name):
        start_time = time.time()
        current_time = time.time()
        # Get the current mouse position in view coordinates
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        current_label = self.label_spinbox.value()
        r = self.radius_spinbox.value()

        # If we have a previous mouse position less than 0.5 seconds old,
        # create 10 intermediate steps along the line.
        if hasattr(self, 'last_mouse_time') and (current_time - self.last_mouse_time < 0.5):
            last_x, last_y = self.last_mouse_point
            distance = np.sqrt((x_m - last_x) ** 2 + (y_m - last_y) ** 2)
            if distance > 40:
                xs = [x_m]
                ys = [y_m]
            else:
                steps = 10
                xs = np.linspace(last_x, x_m, steps)
                ys = np.linspace(last_y, y_m, steps)
        else:
            xs = [x_m]
            ys = [y_m]

        # For each step along the interpolated path, update labels near that point.
        for x, y in zip(xs, ys):
            if view_name == 'xy':
                # Adjust the label based on the y coordinate.
                if y > 180:
                    update_label = current_label + 1
                elif y < -180:
                    update_label = current_label - 1
                else:
                    update_label = current_label
                # Ensure that the update_label does not accidentally flip to an invalid state.
                if update_label in [self.UNLABELED + 1, self.UNLABELED - 1]:
                    update_label = self.UNLABELED

                # Get nearby indices in the xy plane.
                indices = np.array(self.get_nearby_indices_xy(x, y, r))
                z_center = self.z_center_spinbox.value()
                z_thickness = self.z_thickness_slider.value() / self.scaleFactor
                z_min_val = z_center - z_thickness / 2
                z_max_val = z_center + z_thickness / 2
                mask = (self.points[indices, 2] >= z_min_val) & (self.points[indices, 2] <= z_max_val)
                indices = indices[mask]

            elif view_name == 'xz':
                # For the xz view, use the KDTree to find nearby points.
                indices = np.asarray(self.kdtree_xz.query_ball_point([x, y], r=r))
                finit_center = self.finit_center_spinbox.value()
                finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
                finit_min_val = finit_center - finit_thickness / 2
                finit_max_val = finit_center + finit_thickness / 2
                mask = (self.points[indices, 1] >= finit_min_val) & (self.points[indices, 1] <= finit_max_val)
                indices = indices[mask]
                update_label = current_label
            else:
                indices = np.array([])

            # Update the labels for the points that were found.
            if indices.size:
                self.labels[indices] = update_label

        # Save the current mouse position and time for the next event.
        self.last_mouse_point = (x_m, y_m)
        self.last_mouse_time = current_time

        end_time = time.time()  # finished updating labels
        self.update_views()
        end_time2 = time.time()  # finished updating views
        # print(f"Time to update labels: {end_time - start_time:.4f} s, "
        #     f"Time to update views: {end_time2 - end_time:.4f} s")

    def _paint_points_calculated(self, ev, plot_widget, view_name):
        start_time = time.time()
        current_time = time.time()
        # Get the current mouse position in view coordinates.
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        r = self.radius_spinbox.value()

        # If a previous mouse event occurred within 0.5 sec, interpolate 10 steps.
        if hasattr(self, 'last_mouse_time') and (current_time - self.last_mouse_time < 0.5):
            last_x, last_y = self.last_mouse_point
            distance = np.sqrt((x_m - last_x) ** 2 + (y_m - last_y) ** 2)
            if distance > 40:
                xs = [x_m]
                ys = [y_m]
            else:
                steps = 10
                xs = np.linspace(last_x, x_m, steps)
                ys = np.linspace(last_y, y_m, steps)
        else:
            xs = [x_m]
            ys = [y_m]

        # For each intermediate position, update the calculated labels.
        for x, y in zip(xs, ys):
            if view_name == 'xy':
                indices = np.array(self.get_nearby_indices_xy(x, y, r))
                z_center = self.z_center_spinbox.value()
                z_thickness = self.z_thickness_slider.value() / self.scaleFactor
                z_min_val = z_center - z_thickness / 2
                z_max_val = z_center + z_thickness / 2
                mask = (self.points[indices, 2] >= z_min_val) & (self.points[indices, 2] <= z_max_val)
                indices = indices[mask]
                for i in indices:
                    if self.labels[i] == self.UNLABELED and self.calculated_labels[i] != self.UNLABELED:
                        self.labels[i] = self.calculated_labels[i]
            elif view_name == 'xz':
                indices = np.asarray(self.kdtree_xz.query_ball_point([x, y], r=r))
                finit_center = self.finit_center_spinbox.value()
                finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
                finit_min_val = finit_center - finit_thickness / 2
                finit_max_val = finit_center + finit_thickness / 2
                mask = (self.points[indices, 1] >= finit_min_val) & (self.points[indices, 1] <= finit_max_val)
                indices = indices[mask]
                for i in indices:
                    if self.labels[i] == self.UNLABELED and self.calculated_labels[i] != self.UNLABELED:
                        self.labels[i] = self.calculated_labels[i]

        # Save the current mouse position and time for the next event.
        self.last_mouse_point = (x_m, y_m)
        self.last_mouse_time = current_time

        end_time = time.time()  # Finished updating labels.
        self.update_views()
        end_time2 = time.time()  # Finished updating views.
        # print(f"Time to update calc labels: {end_time - start_time:.4f} s, Time to update views: {end_time2 - end_time:.4f} s")

    def update_views(self):
        t0 = time.time()
        # ----- Compute z-slice values -----
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        t1 = time.time()
        # print("Step 1 (z-slice):", t1 - t0, "s")
        
        # ----- Process XY view visible points -----
        mask_xy = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        pts_xy = self.points[mask_xy]
        labels_xy = self.labels[mask_xy]
        calc_labels_xy = self.calculated_labels[mask_xy]
        t2 = time.time()
        # print("Step 2 (XY: mask & extract pts/labels):", t2 - t1, "s")
        
        # ----- Adjust wrapping: top -----
        mask_top = pts_xy[:, 1] < -90
        pts_top = pts_xy[mask_top].copy()
        if pts_top.size:
            pts_top[:, 1] += 360
        labels_top = labels_xy[mask_top] - 1
        calc_labels_top = calc_labels_xy[mask_top] - 1
        t3 = time.time()
        # print("Step 3 (XY: top wrap adjustment):", t3 - t2, "s")
        
        # ----- Adjust wrapping: bottom -----
        mask_bottom = pts_xy[:, 1] > 90
        pts_bottom = pts_xy[mask_bottom].copy()
        if pts_bottom.size:
            pts_bottom[:, 1] -= 360
        labels_bottom = labels_xy[mask_bottom] + 1
        calc_labels_bottom = calc_labels_xy[mask_bottom] + 1
        t4 = time.time()
        # print("Step 4 (XY: bottom wrap adjustment):", t4 - t3, "s")
        
        # ----- Combine and downsample XY arrays -----
        pts_combined = np.concatenate([pts_xy, pts_top, pts_bottom], axis=0)
        labels_combined = np.concatenate([labels_xy, labels_top, labels_bottom], axis=0)
        calc_labels_combined = np.concatenate([calc_labels_xy, calc_labels_top, calc_labels_bottom], axis=0)
        pts_combined, labels_combined, calc_labels_combined = self.downsample_points(
            pts_combined, labels_combined, calc_labels_combined, self.max_display)
        t5 = time.time()
        # print("Step 5 (XY: combine & downsample):", t5 - t4, "s")
        
        # ----- Compute new brushes for XY view -----
        new_brushes_xy = self.get_brushes_from_labels(labels_combined)
        t6 = time.time()
        # print("XY Step 6 (brushes):", t6 - t5, "s")
        
        # ----- Build geometry for XY view (shear transforms) -----
        shear_vertical = self.xy_vertical_shear_spinbox.value()
        vertical_factor = np.tan(np.radians(shear_vertical))
        shear_horizontal = self.xy_horizontal_shear_spinbox.value()
        horizontal_factor = np.tan(np.radians(shear_horizontal))
        new_f_init = pts_combined[:, 1] + vertical_factor * (pts_combined[:, 2] - z_center)
        new_f_star = pts_combined[:, 0] + horizontal_factor * (pts_combined[:, 2] - z_center)
        new_xy_geometry = {'x': new_f_star, 'y': new_f_init}
        t6b = time.time()
        # print("XY Step 6b (shear geometry):", t6b - t6, "s")
        
        # ----- Caching for XY view geometry & brushes -----
        xy_key = (z_center,
                self.z_thickness_slider.value(),
                self.xy_vertical_shear_spinbox.value(),
                self.xy_horizontal_shear_spinbox.value(),
                pts_combined.shape[0])
        if self.recompute or not hasattr(self, "_cached_xy_key") or self._cached_xy_key != xy_key:
            # Geometry changed; perform full update.
            self.xy_scatter.setData(x=new_xy_geometry['x'], y=new_xy_geometry['y'],
                                    size=self.point_size, pen=None, brush=new_brushes_xy)
            self._cached_xy_geometry = new_xy_geometry
            self._cached_xy_brushes = new_brushes_xy.copy()
            self._cached_xy_key = xy_key
            # print("XY: Full setData update")
        else:
            t6c = time.time()
            # Geometry unchanged; update brushes only if necessary.
            changed_mask = np.array([new_brushes_xy[i] != self._cached_xy_brushes[i]
                                    for i in range(len(new_brushes_xy))])
            # print(f"XY: Brushes size: {new_brushes_xy.size}, changed mask size: {changed_mask.size}, nr changed: {np.sum(changed_mask)}")
            t6d = time.time()
            # print("XY: changed mask time:", t6d - t6c, "s")
            if np.any(changed_mask):
                self.xy_scatter.setBrush(new_brushes_xy)
                self._cached_xy_brushes[changed_mask] = new_brushes_xy[changed_mask]
                # print("XY: Partial brush update")
            t6e = time.time()
            # print("XY: Brush update time:", t6e - t6d, "s")
        t7 = time.time()
        # print("XY Caching update:", t7 - t6b, "s")
        
        # ----- Caching for XY calc labels -----
        mask_calc = (np.abs(labels_combined - self.UNLABELED) <= 1) & \
                    (np.abs(calc_labels_combined - self.UNLABELED) > 1)
        pts_combined_calc = pts_combined[mask_calc]
        calc_labels_combined_calc = calc_labels_combined[mask_calc]
        new_brushes_calc_xy = np.empty(calc_labels_combined_calc.shape[0], dtype=object)
        for i, lab in enumerate(calc_labels_combined_calc):
            mod = lab % 3
            if mod == 0:
                new_brushes_calc_xy[i] = self.calc_brush_red
            elif mod == 1:
                new_brushes_calc_xy[i] = self.calc_brush_green
            elif mod == 2:
                new_brushes_calc_xy[i] = self.calc_brush_blue
        calc_xy_key = (z_center, self.z_thickness_slider.value(), pts_combined_calc.shape[0])
        if self.recompute or not hasattr(self, "_cached_calc_xy_key") or self._cached_calc_xy_key != calc_xy_key:
            self.xy_calc_scatter.setData(x=pts_combined_calc[:, 0] if pts_combined_calc.size else [],
                                        y=pts_combined_calc[:, 1] if pts_combined_calc.size else [],
                                        size=self.point_size, pen=None, brush=new_brushes_calc_xy)
            self._cached_calc_xy_geometry = {'x': pts_combined_calc[:, 0] if pts_combined_calc.size else [],
                                            'y': pts_combined_calc[:, 1] if pts_combined_calc.size else []}
            self._cached_calc_xy_brushes = new_brushes_calc_xy.copy()
            self._cached_calc_xy_key = calc_xy_key
            # print("XY Calc: Full setData update")
        else:
            changed_mask_calc = np.array([new_brushes_calc_xy[i] != self._cached_calc_xy_brushes[i]
                                        for i in range(len(new_brushes_calc_xy))])
            # print(f"XY Calc: Brushes size: {new_brushes_calc_xy.size}, changed mask size: {changed_mask_calc.size}, nr changed: {np.sum(changed_mask)}")
            if np.any(changed_mask_calc):
                self.xy_calc_scatter.setBrush(new_brushes_calc_xy)
                self._cached_calc_xy_brushes[changed_mask_calc] = new_brushes_calc_xy[changed_mask_calc]
                # print("XY Calc: Partial brush update")
        t7b = time.time()
        # print("XY Calc Caching update:", t7b - t7, "s")
        
        # ----- Process XZ view -----
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
        finit_min_val = finit_center - finit_thickness / 2
        finit_max_val = finit_center + finit_thickness / 2
        mask_xz = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        pts_xz = self.points[mask_xz]
        labels_xz = self.labels[mask_xz]
        shear_angle_deg_xz = self.xz_shear_spinbox.value()
        xz_shear_factor = np.tan(np.radians(shear_angle_deg_xz))
        if pts_xz.size:
            new_x_xz = pts_xz[:, 0] + xz_shear_factor * (pts_xz[:, 1] - finit_center)
        else:
            new_x_xz = pts_xz[:, 0]
        pts_xz_display = pts_xz.copy()
        pts_xz_display[:, 0] = new_x_xz
        pts_xz_display, labels_xz = self.downsample_points(pts_xz_display, labels=labels_xz, max_display=self.max_display)
        new_xz_geometry = {'x': pts_xz_display[:, 0], 'y': pts_xz_display[:, 2]}
        new_brushes_xz = self.get_brushes_from_labels(labels_xz)
        t8 = time.time()
        # print("XZ Step 8 (processing & downsampling):", t8 - t7b, "s")
        
        # ----- Caching for XZ view geometry & brushes -----
        xz_key = (finit_center,
                self.finit_thickness_slider.value(),
                self.xz_shear_spinbox.value(),
                pts_xz_display.shape[0])
        if self.recompute or not hasattr(self, "_cached_xz_key") or self._cached_xz_key != xz_key:
            self.xz_scatter.setData(x=new_xz_geometry['x'], y=new_xz_geometry['y'],
                                    size=self.point_size, pen=None, brush=new_brushes_xz)
            self._cached_xz_geometry = new_xz_geometry
            self._cached_xz_brushes = new_brushes_xz.copy()
            self._cached_xz_key = xz_key
            # print("XZ: Full setData update")
        else:
            t8b = time.time()
            changed_mask_xz = np.array([new_brushes_xz[i] != self._cached_xz_brushes[i]
                                        for i in range(len(new_brushes_xz))])
            # print(f"XZ: Brushes size: {new_brushes_xz.size}, changed mask size: {changed_mask_xz.size}")
            t8c = time.time()
            # print("XZ: changed mask time:", t8c - t8b, "s")
            if np.any(changed_mask_xz):
                self.xz_scatter.setBrush(new_brushes_xz)
                self._cached_xz_brushes[changed_mask_xz] = new_brushes_xz[changed_mask_xz]
                # print("XZ: Partial brush update")
            t8d = time.time()
            # print("XZ: Brush update time:", t8d - t8c, "s")
        t9 = time.time()
        # print("XZ Caching update:", t9 - t8, "s")
        
        # ----- Caching for XZ calc labels -----
        mask_xz_calc = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        pts_xz_full = self.points[mask_xz_calc]
        manual_labels_xz = self.labels[mask_xz_calc]
        calc_labels_xz_full = self.calculated_labels[mask_xz_calc]
        valid_calc_mask_xz = (manual_labels_xz == self.UNLABELED) & (calc_labels_xz_full != self.UNLABELED)
        pts_calc_xz = pts_xz_full[valid_calc_mask_xz]
        labels_calc_xz = calc_labels_xz_full[valid_calc_mask_xz]
        pts_calc_xz_display = pts_calc_xz.copy()
        if pts_calc_xz.size:
            pts_calc_xz_display[:, 0] = pts_calc_xz_display[:, 0] + xz_shear_factor * (pts_calc_xz_display[:, 1] - finit_center)
        new_brushes_calc_xz = np.empty(labels_calc_xz.shape[0], dtype=object)
        for i, lab in enumerate(labels_calc_xz):
            mod = lab % 3
            if mod == 0:
                new_brushes_calc_xz[i] = self.calc_brush_red
            elif mod == 1:
                new_brushes_calc_xz[i] = self.calc_brush_green
            elif mod == 2:
                new_brushes_calc_xz[i] = self.calc_brush_blue
        calc_xz_key = (finit_center, self.finit_thickness_slider.value(), pts_calc_xz_display.shape[0])
        if self.recompute or not hasattr(self, "_cached_calc_xz_key") or self._cached_calc_xz_key != calc_xz_key:
            self.xz_calc_scatter.setData(x=pts_calc_xz_display[:, 0] if pts_calc_xz_display.size else [],
                                        y=pts_calc_xz_display[:, 2] if pts_calc_xz_display.size else [],
                                        size=self.point_size, pen=None, brush=new_brushes_calc_xz)
            self._cached_calc_xz_geometry = {'x': pts_calc_xz_display[:, 0] if pts_calc_xz_display.size else [],
                                            'y': pts_calc_xz_display[:, 2] if pts_calc_xz_display.size else []}
            self._cached_calc_xz_brushes = new_brushes_calc_xz.copy()
            self._cached_calc_xz_key = calc_xz_key
            # print("XZ Calc: Full setData update")
        else:
            changed_mask_calc_xz = np.array([new_brushes_calc_xz[i] != self._cached_calc_xz_brushes[i]
                                            for i in range(len(new_brushes_calc_xz))])
            # print(f"XZ Calc: Brushes size: {new_brushes_calc_xz.size}, changed mask size: {changed_mask_calc_xz.size}")
            if np.any(changed_mask_calc_xz):
                self.xz_calc_scatter.setBrush(new_brushes_calc_xz)
                self._cached_calc_xz_brushes[changed_mask_calc_xz] = new_brushes_calc_xz[changed_mask_calc_xz]
                # print("XZ Calc: Partial brush update")
        t10 = time.time()
        # print("XZ Calc Caching update:", t10 - t9, "s")
        
        # ----- Update guides -----
        self.update_guides()
        t11 = time.time()

        if hasattr(self, 'ome_zarr_window') and self.ome_zarr_window is not None:
            self.ome_zarr_window.update_overlay_labels(self.labels, self.calculated_labels)
        # print("Step 16 (update_guides):", t11 - t10, "s")
        
        # print(f"Total update_views time: {t11 - t0:.4f} s")
        self.recompute = False
    
    def update_winding_splines(self):
        for item in self.spline_items:
            self.xy_plot.removeItem(item)
        self.spline_items = []
        self.spline_segments = {}
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        mask = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        pts = self.points[mask]
        labels_xy = self.labels[mask]
        calc_labels_xy = self.calculated_labels[mask]
        f_init_adjusted = np.empty_like(pts[:, 1])
        effective_labels = np.empty_like(labels_xy)
        for i in range(len(pts)):
            if pts[i, 1] < -180:
                f_init_adjusted[i] = pts[i, 1] + 360
            elif pts[i, 1] > 180:
                f_init_adjusted[i] = pts[i, 1] - 360
            else:
                f_init_adjusted[i] = pts[i, 1]
            base_label = labels_xy[i] if labels_xy[i] != self.UNLABELED else calc_labels_xy[i]
            effective_labels[i] = base_label if base_label != self.UNLABELED else self.UNLABELED
            if base_label != self.UNLABELED:
                if pts[i, 1] < -180:
                    effective_labels[i] = base_label - 1
                elif pts[i, 1] > 180:
                    effective_labels[i] = base_label + 1
                else:
                    effective_labels[i] = base_label
        valid = effective_labels != self.UNLABELED
        if self.disregard_label0_checkbox.isChecked():
            valid = valid & (effective_labels != 0)
        if not np.any(valid):
            return
        f_init_valid = f_init_adjusted[valid]
        f_star_valid = pts[valid, 0]
        eff_labels_valid = effective_labels[valid]
        threshold = self.spline_min_points_spinbox.value()
        step = 5
        unique_labels = np.unique(eff_labels_valid)
        temp_segments = {}
        for ul in unique_labels:
            if ul == 0:
                if len(np.where(eff_labels_valid == 0)[0]) < 2:
                    continue
            else:
                if len(np.where(eff_labels_valid == ul)[0]) < threshold:
                    continue
            indices = np.where(eff_labels_valid == ul)[0]
            x_label = f_init_valid[indices]
            y_label = f_star_valid[indices]
            grid_min = np.floor(x_label.min())
            grid_max = np.ceil(x_label.max())
            grid = np.arange(grid_min, grid_max + 1, step)
            fitted_values = np.empty_like(grid, dtype=float)
            valid_mask = np.zeros_like(grid, dtype=bool)
            for i, g in enumerate(grid):
                window = np.where(np.abs(x_label - g) <= step)[0]
                if window.size > 0:
                    if window.size >= 2:
                        coeffs = np.polyfit(x_label[window], y_label[window], 1)
                        fitted = np.polyval(coeffs, g)
                    else:
                        fitted = y_label[window[0]]
                    if -5000 <= fitted <= 5000:
                        fitted_values[i] = fitted
                        valid_mask[i] = True
                    else:
                        valid_mask[i] = False
                else:
                    valid_mask[i] = False
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size == 0:
                continue
            segments = []
            current_segment = [valid_indices[0]]
            for idx in valid_indices[1:]:
                if idx == current_segment[-1] + 1:
                    if abs(fitted_values[idx] - fitted_values[current_segment[-1]]) < 20:
                        current_segment.append(idx)
                    else:
                        if len(current_segment) >= 2:
                            segments.append(current_segment)
                        current_segment = [idx]
                else:
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = [idx]
            if len(current_segment) >= 2:
                segments.append(current_segment)
            if not segments:
                continue
            temp_segments[ul] = (grid, fitted_values, segments)
        sorted_ul = sorted(temp_segments.keys())
        final_segments = {}
        for i, ul in enumerate(sorted_ul):
            grid, fitted_values, segments = temp_segments[ul]
            new_segments = []
            if i < len(sorted_ul) - 1:
                neighbor_ul = sorted_ul[i + 1]
                n_grid, n_fitted, n_segments = temp_segments[neighbor_ul]
                neighbor_points = []
                for seg in n_segments:
                    neighbor_points.append(np.column_stack((n_fitted[seg], n_grid[seg])))
                if neighbor_points:
                    neighbor_poly = np.vstack(neighbor_points)
                    neighbor_poly = neighbor_poly[np.argsort(neighbor_poly[:, 1])]
                else:
                    neighbor_poly = None
            else:
                neighbor_poly = None
            for seg in segments:
                seg_grid = grid[seg]
                seg_current = fitted_values[seg]
                valid_idx = []
                for j in range(len(seg_grid)):
                    valid_pt = True
                    if neighbor_poly is not None:
                        if seg_grid[j] >= neighbor_poly[:, 1].min() and seg_grid[j] <= neighbor_poly[:, 1].max():
                            neighbor_val = np.interp(seg_grid[j], neighbor_poly[:, 1], neighbor_poly[:, 0])
                            if seg_current[j] >= neighbor_val:
                                valid_pt = False
                    if valid_pt:
                        valid_idx.append(j)
                if valid_idx:
                    valid_idx = np.array(valid_idx)
                    grouped = []
                    current = [valid_idx[0]]
                    for k in valid_idx[1:]:
                        if k == current[-1] + 1:
                            current.append(k)
                        else:
                            if len(current) >= 2:
                                grouped.append(current)
                            current = [k]
                    if len(current) >= 2:
                        grouped.append(current)
                    for group in grouped:
                        new_seg = np.array(seg)[np.array(group)]
                        new_segments.append(new_seg)
            if new_segments:
                final_segments[ul] = (grid, fitted_values, new_segments)
        for ul, (grid, fitted_values, segments) in final_segments.items():
            self.spline_segments[ul] = []
            mod = int(ul) % 3
            if mod == 0:
                color = (255, 0, 0)
            elif mod == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            pen = pg.mkPen(color=color, width=2)
            for seg in segments:
                seg_grid = grid[seg]
                seg_fitted = fitted_values[seg]
                if len(seg_grid) >= 2:
                    spline_item = pg.PlotDataItem(x=seg_fitted, y=seg_grid, pen=pen)
                    self.xy_plot.addItem(spline_item)
                    self.spline_items.append(spline_item)
                    polyline = np.column_stack((seg_fitted, seg_grid))
                    self.spline_segments.setdefault(ul, []).append(polyline)
    
    def assign_line_labels(self):
        thresh = self.line_distance_threshold_spinbox.value()
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        mask = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        pts = self.points[mask]
        current_labels = self.labels[mask]
        calc_labels = self.calculated_labels[mask]
        f_init = pts[:, 1]
        f_init_adjusted = np.where(f_init < -180, f_init + 360, np.where(f_init > 180, f_init - 360, f_init))
        base_label = np.where(current_labels != self.UNLABELED, current_labels, calc_labels)
        effective_labels = np.where(np.abs(base_label - self.UNLABELED) < 2, self.UNLABELED,
                                    np.where(f_init < -180, base_label - 1,
                                             np.where(f_init > 180, base_label + 1, base_label)))
        if self.disregard_label0_checkbox.isChecked():
            effective_labels[effective_labels == 0] = self.UNLABELED
        assign_min = self.assign_min_spinbox.value()
        assign_max = self.assign_max_spinbox.value()
        range_mask = (effective_labels >= assign_min) & (effective_labels <= assign_max)
        if not np.any(range_mask):
            print("No points within the specified spline winding range.")
            return
        pts = pts[range_mask]
        current_labels = current_labels[range_mask]
        calc_labels = calc_labels[range_mask]
        f_init_adjusted = f_init_adjusted[range_mask]
        effective_labels = effective_labels[range_mask]
        global_indices = np.where(mask)[0][range_mask]
        valid_idx = np.where(effective_labels != self.UNLABELED)[0]
        total_points = len(valid_idx)
        progress = QProgressDialog("Assigning line labels...", "Cancel", 0, total_points, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        assign_count = 0
        update_interval = max(total_points // 100, 1)
        for j, idx in enumerate(valid_idx):
            ul = effective_labels[idx]
            if ul not in self.spline_segments:
                if j % update_interval == 0:
                    progress.setValue(j)
                if progress.wasCanceled():
                    break
                continue
            pt = np.array([pts[idx, 0], f_init_adjusted[idx]])
            d_self = np.min([vectorized_point_to_polyline_distance(pt, seg) for seg in self.spline_segments[ul]])
            d_minus = np.inf
            if (ul - 1) in self.spline_segments:
                d_minus = np.min([vectorized_point_to_polyline_distance(pt, seg) for seg in self.spline_segments[ul - 1]])
            d_plus = np.inf
            if (ul + 1) in self.spline_segments:
                d_plus = np.min([vectorized_point_to_polyline_distance(pt, seg) for seg in self.spline_segments[ul + 1]])
            if d_self < thresh and d_self < d_minus and d_self < d_plus and calc_labels[idx] != self.UNLABELED:
                global_idx = global_indices[idx]
                self.labels[global_idx] = calc_labels[idx]
                assign_count += 1
            if j % update_interval == 0:
                progress.setValue(j)
            if progress.wasCanceled():
                break
        progress.close()
        self.update_views()
        print(f"Assigned line labels to {assign_count} points (threshold: {thresh}).")
    
    def pick_label_at(self, ev, plot_widget):
        dataPos = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x = dataPos.x()
        y = dataPos.y()
        r = self.radius_spinbox.value()
        indices = self.get_nearby_indices_xy(x, y, r)
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        indices = [i for i in indices if self.points[i, 2] >= z_min_val and self.points[i, 2] <= z_max_val]
        indices = np.array(indices)
        if indices.size == 0:
            return
        disp_labels = np.array([self.displayed_label(i, y) for i in indices])
        disp_labels = disp_labels[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED + 1) &
                                  (disp_labels != self.UNLABELED - 1)]
        if disp_labels.size == 0:
            return
        vals, counts = np.unique(disp_labels, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])
        self.label_spinbox.setValue(mode_label)
    
    def pick_label_at_xz(self, ev, plot_widget):
        dataPos = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x = dataPos.x()
        z = dataPos.y()
        r = self.radius_spinbox.value()
        indices = self.kdtree_xz.query_ball_point([x, z], r=r)
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
        finit_min_val = finit_center - finit_thickness / 2
        finit_max_val = finit_center + finit_thickness / 2
        indices = [i for i in indices if self.points[i, 1] >= finit_min_val and self.points[i, 1] <= finit_max_val]
        indices = np.array(indices)
        if indices.size == 0:
            return
        disp_labels = self.labels[indices]
        disp_labels = disp_labels[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED + 1) &
                                  (disp_labels != self.UNLABELED - 1)]
        if disp_labels.size == 0:
            return
        vals, counts = np.unique(disp_labels, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])
        self.label_spinbox.setValue(mode_label)
    
    def activate_pipette(self):
        self.pipette_mode = True
    
    def update_ground_truth(self):
        ground_truth = self.labels.tolist()
        with open("ground_truth.txt", "w") as f:
            f.write(str(ground_truth))
    
    # Example of using the solver interface when updating labels:
    def update_labels(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            # if not np.any(gt):
            #     return
            self.solver.set_labels(self.labels, gt)
            selected_solver = self.solver_combo.currentText() if hasattr(self, "solver_combo") else "F*"
            if selected_solver == "Winding Number":
                self.solver.solve_winding_number(num_iterations=500, i_round=-3, seed_node=-1,
                                                 other_block_factor=15.0, side_fix_nr=-1, display=False)
            elif "F*" in selected_solver:
                if self.use_z_range_checkbox.isChecked():
                    print("Using z-range")
                    undeleted = self.solver.get_undeleted_indices()
                    z_center = self.z_center_spinbox.value()
                    z_thickness = self.z_thickness_slider.value() / self.scaleFactor
                    z_min_val = 4 * (z_center - z_thickness / 2) - 500.0
                    z_max_val = 4 * (z_center + z_thickness / 2) - 500.0
                    deleted_mask_previous = self.solver.set_z_range(z_min_val, z_max_val)
                    self.seed_node = None
                else:
                    deleted_mask_previous = np.zeros_like(self.labels, dtype=bool)
                if self.seed_node is None:
                    assert len(deleted_mask_previous) == len(self.labels), "Deleted mask shape mismatch"
                    # try to set a seed node. first labeled point found.
                    for i, label in tqdm(enumerate(self.labels), desc="Finding seed node"):
                        if i == 0: # do not use "no-seed" index as seed node
                            continue
                        if abs(label - self.UNLABELED) > 2:
                            if deleted_mask_previous[i]:
                                continue
                            offset = np.logical_not(deleted_mask_previous[:i]).sum()
                            self.seed_node = offset
                            break
                if self.seed_node is not None:
                    self.solver.set_labeled_edges(self.seed_node)
                    self.solver.set_f_star(self.seed_node)
                    # self.solver.fix_good_edges()
                if selected_solver == "F*":
                    self.solver.solve_f_star_with_labels(num_iterations=15000, seed_node=self.seed_node if self.seed_node else 0, spring_constant=1.1, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=15000, seed_node=self.seed_node if self.seed_node else 0, spring_constant=1.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=3.0, other_block_factor=0.75, lr=7.0, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=2.0, other_block_factor=0.2, lr=4.0, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=30000, spring_constant=1.0, other_block_factor=0.2, lr=4.0, error_cutoff=-1.0, display=True)

                    # self.solver.solve_f_star_with_labels(num_iterations=75000, seed_node=self.seed_node if self.seed_node else 0, other_block_factor=0.2, lr=4.0, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=75000, seed_node=self.seed_node if self.seed_node else 0, other_block_factor=0.3, lr=7.0, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=35000, seed_node=self.seed_node, other_block_factor=0.25, lr=3.0, error_cutoff=0.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=15000, other_block_factor=0.750, lr=3.0, error_cutoff=0.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=7500, other_block_factor=0.50, lr=5.0, error_cutoff=1080.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=7500, other_block_factor=0.50, lr=5.0, error_cutoff=720.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=7500, other_block_factor=0.50, lr=5.0, error_cutoff=360.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=7500, other_block_factor=0.50, lr=5.0, error_cutoff=180.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=7500, other_block_factor=0.50, lr=5.0, error_cutoff=90.0, display=True)
                elif selected_solver == "F*2":
                    # self.solver.solve_f_star_with_labels(num_iterations=20000, spring_constant=1.0, other_block_factor=0.03, lr=0.10, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=int(self.solve_iterations_spinbox.value()), seed_node=self.seed_node if self.seed_node else 0, spring_constant=1.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=35000, seed_node=self.seed_node if self.seed_node else 0, other_block_factor=1.0, lr=1.0, error_cutoff=0.0, display=True)
                elif selected_solver == "F*3":
                    self.solver.solve_f_star_with_labels(num_iterations=15000, seed_node=self.seed_node if self.seed_node else 0, spring_constant=4.0, other_block_factor=0.5, lr=0.25, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=15000, seed_node=self.seed_node if self.seed_node else 0, spring_constant=2.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=30000, seed_node=self.seed_node if self.seed_node else 0, spring_constant=1.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=30000, seed_node=self.seed_node if self.seed_node else 0, spring_constant=0.5, other_block_factor=0.02, lr=0.080, error_cutoff=-1.0, display=True)
                elif selected_solver == "F*4":
                    self.solver.solve_f_star(num_iterations=int(3 * self.solve_iterations_spinbox.value() / 4), spring_constant=1.0, o=0.0, step_sigma=36000000.0, i_round=6, visualize=True)
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value() / 4), spring_constant=1.0, o=0.0, step_sigma=360.0, i_round=6, visualize=True)

                if self.use_z_range_checkbox.isChecked():
                    print("Resetting z-range")
                    self.solver.set_undeleted_indices(undeleted)
                    self.seed_node = None

                # Update positions
                self.update_positions(update_slide_ranges=False)
                return
            elif selected_solver == "Union":
                self.solver.solve_union()
            elif selected_solver == "Random":
                self.solver.solve_random(num_iterations=5000, i_round=-3, display=True)
            elif selected_solver == "Set Labels":
                print("Setting Labels and no solving.")
            else:
                self.solver.solve_winding_number(num_iterations=500, i_round=-3, seed_node=-1,
                                                 other_block_factor=15.0, side_fix_nr=-1, display=False)
            calculated_labels = self.solver.get_labels()
            self.calculated_labels = np.array(calculated_labels)
            self.update_views()
    
    def save_graph(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            self.solver.set_labels(self.labels, gt)
            # delete unasigned points
            label_indices = np.array(self.solver.get_undeleted_indices())
            mask_labeled = np.abs(self.labels - self.UNLABELED) > 2
            labeled_indices = list(label_indices[mask_labeled])
            self.solver.set_undeleted_indices(labeled_indices)
            print(f"Deleted Unlabeled Points: {len(label_indices) - len(labeled_indices)} of {len(label_indices)}. Saving graph...")
            # Save graph as output_graph.bin for meshing
            graph_solved_path = self.graph_path.replace("graph.bin", "output_graph.bin")
            self.solver.save_solution(graph_solved_path)
    
    def undo(self):
        if self.undo_stack:
            prev_state = self.undo_stack.pop()
            self.redo_stack.append(self.labels.copy())
            self.labels = prev_state.copy()
            self.update_views()
    
    def redo(self):
        if self.redo_stack:
            next_state = self.redo_stack.pop()
            self.undo_stack.append(self.labels.copy())
            self.labels = next_state.copy()
            self.update_views()
    
    def save_labels(self):
        labeled_data = np.hstack([self.points, self.labels.reshape(-1, 1)])
        np.save("labeled_pointcloud.npy", labeled_data)
    
    def toggle_calc_draw_mode(self):
        self.calc_drawing_mode = self.calc_draw_button.isChecked()
        if self.calc_drawing_mode:
            self.calc_draw_button.setText("Update Labels Draw Mode: On")
        else:
            self.calc_draw_button.setText("Update Labels Draw Mode: Off")
    
    def apply_all_calculated_labels(self):
        mask = (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED)
        if np.any(mask):
            self.labels[mask] = self.calculated_labels[mask]
            self.update_views()
    
    def clear_calculated_labels(self):
        self.calculated_labels[:] = self.UNLABELED
        self.update_views()
    
    def apply_calculated_labels_xy(self):
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        mask = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        indices = np.where(mask & (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED))[0]
        if indices.size:
            self.labels[indices] = self.calculated_labels[indices]
            self.update_views()
    
    def apply_calculated_labels_xz(self):
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
        finit_min_val = finit_center - finit_thickness / 2
        finit_max_val = finit_center + finit_thickness / 2
        mask = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        indices = np.where(mask & (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED))[0]
        if indices.size:
            self.labels[indices] = self.calculated_labels[indices]
            self.update_views()
    
    def clear_splines(self):
        for item in self.spline_items:
            self.xy_plot.removeItem(item)
        self.spline_items = []
        self.spline_segments = {}
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and not self.s_pressed:
            self.s_pressed = True
            self.original_drawing_mode = self.drawing_mode_checkbox.isChecked()
            self.drawing_mode_checkbox.setChecked(False)
            self.xy_scatter.setAcceptedMouseButtons(Qt.NoButton)
            self.xz_scatter.setAcceptedMouseButtons(Qt.NoButton)
            event.accept()
        elif event.key() == Qt.Key_U:
            self.calc_draw_button.setChecked(not self.calc_drawing_mode)
            self.toggle_calc_draw_mode()
            time.sleep(0.1)
            event.accept()
        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undo()
            event.accept()
        elif event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier):
            self.redo()
            event.accept()
        elif event.key() == Qt.Key_P:
            self.activate_pipette()
            event.accept()
        # up arrow
        elif event.key() == Qt.Key_Up:
            self.label_spinbox.setValue(self.label_spinbox.value() + 1)
            event.accept()
        # down arrow
        elif event.key() == Qt.Key_Down:
            self.label_spinbox.setValue(self.label_spinbox.value() - 1)
            event.accept()
        else:
            super(PointCloudLabeler, self).keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_S and self.s_pressed:
            self.s_pressed = False
            self.drawing_mode_checkbox.setChecked(self.original_drawing_mode)
            if self.original_drawing_mode:
                self.xy_scatter.setAcceptedMouseButtons(Qt.LeftButton)
                self.xz_scatter.setAcceptedMouseButtons(Qt.LeftButton)
            else:
                self.xy_scatter.setAcceptedMouseButtons(Qt.NoButton)
                self.xz_scatter.setAcceptedMouseButtons(Qt.NoButton)
            event.accept()
        else:
            super(PointCloudLabeler, self).keyReleaseEvent(event)
