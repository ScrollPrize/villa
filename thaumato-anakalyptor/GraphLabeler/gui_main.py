import sys, os, ast, numpy as np
import math
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import (
    QSplitter, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QAction, QMessageBox, QInputDialog, QGraphicsEllipseItem, QFileDialog,
    QProgressDialog, QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QComboBox,
    QTextBrowser, QGraphicsPathItem, QLayout
)
from PyQt5.QtCore import Qt, QEvent, QPointF, QTimer
from PyQt5.QtGui  import QPainterPath, QPen, QColor
import pyqtgraph as pg
from scipy.spatial import cKDTree
from tqdm import tqdm
from collections import Counter
import re

# --------------------------------------------------
# Importing the custom graph problem library.
# --------------------------------------------------
sys.path.append('../ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

from config import load_config, save_config
from utils import vectorized_point_to_polyline_distance, build_temporary_group_edges, filter_point_normals
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
        self.graph_path = self.config.get("graph_path", "")
        self.default_experiment = self.config.get("default_experiment", "")
        self.graph_version = self.config.get("graph_version", 3)
        self.ome_zarr_path = self.config.get("ome_zarr_path", None)
        self.graph_pkl_path = self.config.get("graph_pkl_path", None)
        self.h5_path = self.config.get("h5_path", None)
        self.umbilicus_path = self.config.get("umbilicus_path", None)

        # Initialize solver using SolverInterface if no external point data is provided.
        if point_data is None and self.default_experiment != "" and self.graph_path != "":
            self.solver = graph_problem_gpu_py.Solver(self.graph_path)
            gt_path = os.path.join("../experiments", self.default_experiment,
                                   "checkpoints", "checkpoint_graph_tugging.bin")
            if not os.path.exists(gt_path):
                gt_path = os.path.join("../experiments", self.default_experiment,
                                       "checkpoints", "checkpoint_graph_f_star_final.bin")
                # Keep the configured version even for the fallback file
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path, self.graph_version)
            else:
                print("Default graph file not found; continuing without loading.")
            point_data = self.solver.get_positions()
        else:
            self.solver = None
            point_data = np.zeros((0, 3), dtype=np.float32)
        point_data = np.array(point_data, dtype=np.float32)
        if len(point_data.shape) != 2:
            print(f"Error: point_data should be a 2D array, but got shape {point_data.shape}")
            point_data = np.zeros((0, 3), dtype=np.float32)
        self.seed_node = None # master node, this is the fixed node, to which all other nodes are fixed in f_star to
        self.recompute = True
        self.stop_slab_computation = False  # Initialize the stop flag.

        # Global variables and state.
        self.scaleFactor = 100
        self.s_pressed = False
        self.original_drawing_mode = True
        self.pipette_mode = False
        self.group_pipette_mode = False
        self.calc_drawing_mode = False
        self.UNLABELED = -9999
        # Load teflon label number from config (persistent) or use default 200
        self.teflon_label = self.config.get("teflon_label", 200)
        self.undo_stack = []
        self.redo_stack = []
        self._stroke_backup = None
        self.gt_labels = None
        self.calculating = False
        
        # Autosave functionality
        self.last_saved_labels = None
        self.last_saved_group = None
        self.last_saved_streaks = None
        self.last_manual_save_path = None
        self.last_load_path = None
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.perform_autosave)
        self.autosave_timer.start(10 * 60 * 1000)  # 10 minutes in milliseconds
        
        # Spline storage.
        self.spline_items = []
        self.spline_segments = {}
        
        # Create menu.
        self._create_menu()
        
        # Data and labels.
        self.hide_labels = False
        self.hide_estimated_colors = False
        self.show_original_points = False
        self.points = np.array(point_data)
        self.original_points = np.array(point_data)
        self.group = np.zeros(len(self.points), dtype=np.int32)
        self.active_group = 0
        self.last_group_update = 0
        self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        self.calculated_labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        # Streak flags: each point can be part of a streak (True) or not (False)
        self.streaks = np.zeros(len(self.points), dtype=bool)
        # Streak painting mode off by default
        self.streak_mode = False
        
        # Display parameters.
        self.point_size = 2
        self.max_display = 1400
        if self.points.shape[0] > 0:
            self.f_star_min, self.f_star_max = float(np.min(self.points[:, 0])), float(np.max(self.points[:, 0]))
            self.z_min, self.z_max = float(np.min(self.points[:, 2])), float(np.max(self.points[:, 2]))
        else:
            self.f_star_min, self.f_star_max = -1.0, 1.0
            self.z_min, self.z_max = -1.0, 1.0
        self.f_init_min, self.f_init_max = -180.0, 180.0
        
        # Create KD-trees.
        self.kdtree_xy = cKDTree(self.points[:, [0, 1]])
        self.kdtree_xz = cKDTree(self.points[:, [0, 2]])
        self.original_kdtree_xy = self.kdtree_xy
        self.original_kdtree_xz = self.kdtree_xz
        self.computed_kdtree_xy = self.kdtree_xy
        self.computed_kdtree_xz = self.kdtree_xz
        
        # Pre-created brushes.
        self.brush_black = pg.mkBrush(0, 0, 0)
        self.brush_red_active   = pg.mkBrush(255, 0, 0)
        self.brush_green_active = pg.mkBrush(0, 255, 0)
        self.brush_brown_active  = pg.mkBrush(218,165,32)        
        self.brush_red_inactive   = pg.mkBrush(205, 0, 200, 180)
        self.brush_green_inactive = pg.mkBrush(0, 205, 200, 180)
        self.brush_brown_inactive  = pg.mkBrush(168,125,152, 180)
        self.calc_brush_black = pg.mkBrush(0, 0, 0, 80)
        self.calc_brush_red   = pg.mkBrush(255, 0, 0, 80)
        self.calc_brush_green = pg.mkBrush(0, 255, 0, 80)
        self.calc_brown_blue  = pg.mkBrush(218,165,32, 80)
        self.transparent_brush = pg.mkBrush(0, 0, 0, 0)
        # Additional brushes for 4/5 color modes
        self.brush_blue_active = pg.mkBrush(0, 0, 255)
        self.brush_blue_inactive = pg.mkBrush(0, 0, 205, 180)
        self.brush_yellow_active = pg.mkBrush(255, 255, 0)
        self.brush_yellow_inactive = pg.mkBrush(205, 205, 0, 180)
        # Lists for dynamic color selection
        self.active_brushes = [
            self.brush_red_active,
            self.brush_green_active,
            self.brush_brown_active,
            self.brush_blue_active,
            self.brush_yellow_active]
        self.inactive_brushes = [
            self.brush_red_inactive,
            self.brush_green_inactive,
            self.brush_brown_inactive,
            self.brush_blue_inactive,
            self.brush_yellow_inactive]
        # Load number of colors from config (default 3)
        self.num_colors = self.config.get("num_colors", 3)

        cornflower_blue = (100, 149, 237)  # Cornflower Blue
        medium_orchid   = (186, 85, 211)   # Medium Orchid (a purple shade)
        turquoise       = (64, 224, 208)   # Turquoise

        # Generate brushes for values 0.0, 0.1, 0.2, ... 2.9 (i.e. 30 steps)
        self.unlabeled_brushes = []
        for i in range(30):  # i = 0..29
            val = i * 0.1  # our discrete value in [0,3)
            seg = int(val)   # segment 0, 1, or 2
            frac = val - seg
            if seg == 0:
                start_color, end_color = cornflower_blue, medium_orchid
            elif seg == 1:
                start_color, end_color = medium_orchid, turquoise
            elif seg == 2:
                # For cyclic interpolation, wrap from turquoise back to cornflower_blue.
                start_color, end_color = turquoise, cornflower_blue
            # Interpolate per channel.
            interp_color = tuple(
                int((1 - frac) * s + frac * e) for s, e in zip(start_color, end_color)
            )
            brush = pg.mkBrush(*interp_color)
            self.unlabeled_brushes.append(brush)
        self.unlabeled_brushes = np.array(self.unlabeled_brushes)
        
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
        self.cursor_circle_xz = QGraphicsEllipseItem(0, 0, 0, 0)
        self.cursor_circle_xz.setPen(pg.mkPen('cyan', width=1, style=Qt.DashLine))
        self.cursor_circle_xz.setVisible(False)
        
        # Layout setup.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Set up the main vertical layout without enforcing its children as the window's minimum size
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSizeConstraint(QLayout.SetNoConstraint)

        # Upper view area.
        splitter = QSplitter(Qt.Horizontal)
        
        # Left (XY) view.
        left_widget = QWidget()
        left_column = QVBoxLayout(left_widget)
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
        xy_selections = QHBoxLayout()
        self.z_selection_widget, self.z_selection_slider, self.z_selection_spinbox = create_sync_slider_spinbox(
            "Z selection center:", self.z_min, self.z_max, (self.z_min + self.z_max) / 2, self.scaleFactor, self.update_views)
        self.z_selection_thickness_widget, self.z_selection_thickness_slider, self.z_selection_thickness_spinbox = create_sync_slider_spinbox(
            "Z selection thickness:", 0.0, self.z_max - self.z_min, 0.0, self.scaleFactor, self.update_views)
        xy_selections.addWidget(self.z_selection_widget)
        xy_selections.addWidget(self.z_selection_thickness_widget)
        left_column.addLayout(xy_selections)
        self.line_fstar_min = pg.InfiniteLine(pos=self.f_star_min, angle=90, 
                                        pen=pg.mkPen('blue', width=1, style=Qt.DashLine))
        self.line_fstar_max = pg.InfiniteLine(pos=self.f_star_max, angle=90, 
                                                pen=pg.mkPen('blue', width=1, style=Qt.DashLine))

        # Create slider/spinbox controls for f* min and max; using the f* data range as both min and max of the slider.
        self.fstar_min_widget, self.fstar_min_slider, self.fstar_min_spinbox = create_sync_slider_spinbox(
            "F* Min:", self.f_star_min, self.f_star_max, self.f_star_min, self.scaleFactor, self.update_fstar_indicators, decimals=1)
        self.fstar_max_widget, self.fstar_max_slider, self.fstar_max_spinbox = create_sync_slider_spinbox(
            "F* Max:", self.f_star_min, self.f_star_max, self.f_star_max, self.scaleFactor, self.update_fstar_indicators, decimals=1)

        left_column.addWidget(self.fstar_min_widget)
        left_column.addWidget(self.fstar_max_widget)
        # Use f* range checkbox
        self.use_fstar_range_checkbox = QCheckBox("Use F* range")
        self.use_fstar_range_checkbox.setChecked(False)
        left_column.addWidget(self.use_fstar_range_checkbox)
        self.use_fstar_range_checkbox.toggled.connect(self.update_fstar_indicators)
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
        right_widget = QWidget()
        right_column = QVBoxLayout(right_widget)
        self.xz_plot = pg.PlotWidget()
        self.xz_plot.setBackground('w')
        self.xz_plot.setLabel('bottom', 'f_star')
        self.xz_plot.setLabel('left', 'Z')
        self.xz_plot.setMouseEnabled(x=True, y=True)
        right_column.addWidget(self.xz_plot)
        self.xz_plot.addItem(self.cursor_circle_xz)
        xz_controls = QHBoxLayout()
        self.finit_center_widget, self.finit_center_slider, self.finit_center_spinbox = create_sync_slider_spinbox(
            "f init center:", -180.0, 180.0,
            0.0, self.scaleFactor, self.f_init_center_changed)
        self.finit_thickness_widget, self.finit_thickness_slider, self.finit_thickness_spinbox = create_sync_slider_spinbox(
            "f init thickness:", 0.01, 360.0, 5.0, self.scaleFactor, self.update_views)
        xz_controls.addWidget(self.finit_center_widget)
        xz_controls.addWidget(self.finit_thickness_widget)
        right_column.addLayout(xz_controls)
        xz_selections = QHBoxLayout()
        self.finit_selection_widget, self.finit_selection_slider, self.finit_selection_spinbox = create_sync_slider_spinbox(
            "f selection center:", -180.0, 180.0,
            0.0, self.scaleFactor, self.update_views)
        self.f_selection_thickness_widget, self.f_selection_thickness_slider, self.f_selection_thickness_spinbox = create_sync_slider_spinbox(
            "f selection thickness:", 0.0, 360.0, 0.0, self.scaleFactor, self.update_views)
        xz_selections.addWidget(self.finit_selection_widget)
        xz_selections.addWidget(self.f_selection_thickness_widget)
        right_column.addLayout(xz_selections)
        # XZ shear control (unchanged functionality)
        xz_shear_layout = QHBoxLayout()
        self.xz_shear_widget, self.xz_shear_slider, self.xz_shear_spinbox = create_sync_slider_spinbox(
            "XZ Shear (°):", -90.0, 90.0, 0.0, self.scaleFactor, self.update_views, decimals=1)
        xz_shear_layout.addWidget(self.xz_shear_widget)
        right_column.addLayout(xz_shear_layout)
        self.apply_calc_xz_button = QPushButton("Apply Updated Labels to XZ")
        self.apply_calc_xz_button.clicked.connect(self.apply_calculated_labels_xz)
        right_column.addWidget(self.apply_calc_xz_button)

        # Add both widgets to the splitter.
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Add the splitter to the main layout.
        main_layout.addWidget(splitter)
        
        # --------------------------------------------------------------------
        # Top Controls Row: Spline and Label Update Controls.
        # --------------------------------------------------------------------

        top_controls_layout = QHBoxLayout()

        self.update_labels_button = QPushButton("Update Labels")
        self.update_labels_button.clicked.connect(lambda: self.update_labels())
        top_controls_layout.addWidget(self.update_labels_button)

        # --- Solver selection dropdown ---
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["F*", "Linear", "F*3", "F*4", "Ripple", "Smooth", "Ripple Smooth Combined", "Tugging", "F*Slab", "Winding Number", "Union", "Random", "Create Good Edges", "Set Labels"])
        self.solver_combo.setCurrentIndex(6) # Default to "Ripple Smooth Combined"
        top_controls_layout.addWidget(QLabel("Select Solver:"))
        top_controls_layout.addWidget(self.solver_combo)

        self.use_z_range_checkbox = QCheckBox("Solve in Z range")
        self.use_z_range_checkbox.setChecked(True)
        top_controls_layout.addWidget(self.use_z_range_checkbox)

        solve_iterations_layout = QHBoxLayout()
        self.solve_iterations_spinbox = QSpinBox()
        self.solve_iterations_spinbox.setRange(10, 1000000)
        self.solve_iterations_spinbox.setValue(15000)
        solve_iterations_layout.addWidget(QLabel("Solver Iterations:"))
        solve_iterations_layout.addWidget(self.solve_iterations_spinbox)
        top_controls_layout.addLayout(solve_iterations_layout)

        solve_other_block_factor_layout = QHBoxLayout()
        self.solve_other_block_factor_spinbox = QDoubleSpinBox()
        self.solve_other_block_factor_spinbox.setRange(0, 100)
        self.solve_other_block_factor_spinbox.setDecimals(2)
        self.solve_other_block_factor_spinbox.setSingleStep(1.0)
        self.solve_other_block_factor_spinbox.setValue(0.10)
        solve_other_block_factor_layout.addWidget(QLabel("Same Sheet Factor:"))
        solve_other_block_factor_layout.addWidget(self.solve_other_block_factor_spinbox)
        top_controls_layout.addLayout(solve_other_block_factor_layout)
        
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

        self.assign_line_labels_all_button = QPushButton("Assign Line Labels All")
        self.assign_line_labels_all_button.clicked.connect(self.assign_line_labels_all)
        top_controls_layout.addWidget(self.assign_line_labels_all_button)
        
        line_dist_layout = QHBoxLayout()
        self.line_distance_threshold_spinbox = QSpinBox()
        self.line_distance_threshold_spinbox.setRange(1, 100)
        self.line_distance_threshold_spinbox.setValue(4)
        line_dist_layout.addWidget(QLabel("Line dist thresh:"))
        line_dist_layout.addWidget(self.line_distance_threshold_spinbox)
        top_controls_layout.addLayout(line_dist_layout)
        main_layout.addLayout(top_controls_layout)
        
        range_controls_layout = QHBoxLayout()
        effective_range_layout = QHBoxLayout()
        self.assign_min_spinbox = QSpinBox()
        self.assign_min_spinbox.setRange(-10000, 10000)
        self.assign_min_spinbox.setValue(-1000)
        self.assign_max_spinbox = QSpinBox()
        self.assign_max_spinbox.setRange(-10000, 10000)
        self.assign_max_spinbox.setValue(1000)
        effective_range_layout.addWidget(QLabel("Spline winding range min:"))
        effective_range_layout.addWidget(self.assign_min_spinbox)
        effective_range_layout.addWidget(QLabel("max:"))
        effective_range_layout.addWidget(self.assign_max_spinbox)
        range_controls_layout.addLayout(effective_range_layout)

        self.outside_checkbox = QCheckBox("Outside")
        self.outside_checkbox.setChecked(False)
        range_controls_layout.addWidget(self.outside_checkbox)

        self.deleted_range_button = QPushButton("<-- Delete Range")
        self.deleted_range_button.clicked.connect(self.delete_range)
        range_controls_layout.addWidget(self.deleted_range_button)

        self.split_groups_button = QPushButton("<-- Split Range into Groups")
        self.split_groups_button.clicked.connect(self.split_groups_range)
        range_controls_layout.addWidget(self.split_groups_button)
        # Streak Mode toggle: enables painting boolean streaks on points
        self.streak_mode_checkbox = QCheckBox("Streak Mode")
        self.streak_mode_checkbox.setChecked(False)
        range_controls_layout.addWidget(self.streak_mode_checkbox)
        self.streak_mode_checkbox.toggled.connect(self.toggle_streak_mode)
        main_layout.addLayout(range_controls_layout)
        
        # --------------------------------------------------------------------
        # Bottom Controls Row: Common Drawing and File Controls.
        # --------------------------------------------------------------------
        common_controls_layout = QHBoxLayout()
        self.radius_widget, self.radius_slider, self.radius_spinbox = create_sync_slider_spinbox(
            "Drawing radius:", 0.1, 20.0, 3.5, 1.0, self.update_views, decimals=1)
        common_controls_layout.addWidget(self.radius_widget)

        solve_other_block_r_layout = QHBoxLayout()
        self.solve_other_block_r_spinbox = QDoubleSpinBox()
        self.solve_other_block_r_spinbox.setRange(0, 100)
        self.solve_other_block_r_spinbox.setDecimals(2)
        self.solve_other_block_r_spinbox.setSingleStep(1.0)
        self.solve_other_block_r_spinbox.setValue(10)
        solve_other_block_r_layout.addWidget(QLabel("Good Edges R:"))
        solve_other_block_r_layout.addWidget(self.solve_other_block_r_spinbox)
        common_controls_layout.addLayout(solve_other_block_r_layout)
        
        max_disp_layout = QHBoxLayout()
        max_disp_label = QLabel("Max Display Points:")
        self.max_display_spinbox = QSpinBox()
        self.max_display_spinbox.setRange(1, 10000000)
        self.max_display_spinbox.setValue(self.max_display)
        max_disp_layout.addWidget(max_disp_label)
        max_disp_layout.addWidget(self.max_display_spinbox)
        common_controls_layout.addLayout(max_disp_layout)
        self.max_display_spinbox.valueChanged.connect(self.update_max_display)

        self.local_downsample_checkbox = QCheckBox("Local Downsample")
        self.local_downsample_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.local_downsample_checkbox)
        
        self.drawing_mode_checkbox = QCheckBox("Drawing Mode")
        self.drawing_mode_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.drawing_mode_checkbox)
        self.drawing_mode_checkbox.toggled.connect(self.update_drawing_mode)
        
        self.show_guides_checkbox = QCheckBox("Show guides")
        self.show_guides_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.show_guides_checkbox)
        self.show_guides_checkbox.toggled.connect(self.update_guides)

        # Button to toggle GT label display
        self.show_gt_labels_button = QPushButton("Show GT Labels")
        self.show_gt_labels_button.setCheckable(True)
        self.show_gt_labels_button.clicked.connect(self.toggle_gt_labels)
        common_controls_layout.addWidget(self.show_gt_labels_button)

        self.gt_to_group_button = QPushButton("GT to Group")
        self.gt_to_group_button.clicked.connect(self.gt_labels_to_group)
        common_controls_layout.addWidget(self.gt_to_group_button)
        
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
        # Color mode buttons
        self.colors4_button = QPushButton("4 Colors")
        self.colors4_button.setCheckable(True)
        self.colors5_button = QPushButton("5 Colors")
        self.colors5_button.setCheckable(True)
        common_controls_layout.addWidget(self.colors4_button)
        common_controls_layout.addWidget(self.colors5_button)
        self.colors4_button.clicked.connect(lambda checked: self._on_colors_toggled(4, checked))
        self.colors5_button.clicked.connect(lambda checked: self._on_colors_toggled(5, checked))
        # Restore color mode from config
        if self.num_colors == 4:
            self.colors4_button.setChecked(True)
        elif self.num_colors == 5:
            self.colors5_button.setChecked(True)
        
        main_layout.addLayout(common_controls_layout)


        # --------------------------------------------------
        # Cellar Layout
        # --------------------------------------------------
        cellar_controls_layout = QHBoxLayout()


        self.slab_compute_button = QPushButton("Compute Slabs")
        self.slab_compute_button.clicked.connect(self.run_slab_computation)
        cellar_controls_layout.addWidget(self.slab_compute_button)

        self.stop_slab_button = QPushButton("Stop Slab Computation")
        self.stop_slab_button.clicked.connect(self.set_stop_slab_flag)
        cellar_controls_layout.addWidget(self.stop_slab_button)

        self.skip_initial_checkbox = QCheckBox("Skip Initial")
        self.skip_initial_checkbox.setChecked(False)
        cellar_controls_layout.addWidget(self.skip_initial_checkbox)

        slabs_range_layout = QHBoxLayout()
        self.slabs_min_spinbox = QSpinBox()
        self.slabs_min_spinbox.setRange(-100000, 100000)
        self.slabs_min_spinbox.setValue(-100000)
        self.slabs_max_spinbox = QSpinBox()
        self.slabs_max_spinbox.setRange(-100000, 100000)
        self.slabs_max_spinbox.setValue(100000)
        slabs_range_layout.addWidget(QLabel("Slabs Z range min:"))
        slabs_range_layout.addWidget(self.slabs_min_spinbox)
        slabs_range_layout.addWidget(QLabel("max:"))
        slabs_range_layout.addWidget(self.slabs_max_spinbox)
        cellar_controls_layout.addLayout(slabs_range_layout)
        # slab thickness
        slab_thickness_layout = QHBoxLayout()
        self.slab_thickness_spinbox = QSpinBox()
        self.slab_thickness_spinbox.setRange(1, 100000)
        self.slab_thickness_spinbox.setValue(100)
        slab_thickness_layout.addWidget(QLabel("Slab thickness:"))
        slab_thickness_layout.addWidget(self.slab_thickness_spinbox)
        cellar_controls_layout.addLayout(slab_thickness_layout)

        self.reset_points_button = QPushButton("Reset Points")
        self.reset_points_button.clicked.connect(self.reset_points)
        cellar_controls_layout.addWidget(self.reset_points_button)

        self.teflon_spinbox = QSpinBox()
        self.teflon_spinbox.setRange(-10000, 10000)
        self.teflon_spinbox.setValue(self.teflon_label)
        self.teflon_spinbox.valueChanged.connect(self.update_teflon)
        cellar_controls_layout.addWidget(QLabel("Teflon Label:"))
        cellar_controls_layout.addWidget(self.teflon_spinbox)

        self.hide_teflon_checkbox = QCheckBox("Hide Teflon")
        self.hide_teflon_checkbox.setChecked(False)
        cellar_controls_layout.addWidget(self.hide_teflon_checkbox)
        self.hide_teflon_checkbox.toggled.connect(self.update_views)

        self.filter_teflon_button = QPushButton("Filter Teflon")
        self.filter_teflon_button.clicked.connect(self.filter_teflon)
        cellar_controls_layout.addWidget(self.filter_teflon_button)

        self.filter_angle_spinbox = QSpinBox()
        self.filter_angle_spinbox.setRange(0, 90)
        self.filter_angle_spinbox.setValue(80)
        cellar_controls_layout.addWidget(QLabel("Filter Angle:"))
        cellar_controls_layout.addWidget(self.filter_angle_spinbox)

        self.filter_distance_spinbox = QSpinBox()
        self.filter_distance_spinbox.setRange(0, 100)
        self.filter_distance_spinbox.setValue(5)
        cellar_controls_layout.addWidget(QLabel("Normal Estimation Distance:"))
        cellar_controls_layout.addWidget(self.filter_distance_spinbox)

        self.reset_teflon_button = QPushButton("Reset Teflon")
        self.reset_teflon_button.clicked.connect(self.reset_teflon)
        cellar_controls_layout.addWidget(self.reset_teflon_button)

        # Show Max Group (just display, no control)
        self.max_group_label = QLabel("Max Group: 0")
        cellar_controls_layout.addWidget(self.max_group_label)

        group_save_layout = QHBoxLayout()
        self.group_spinbox = QSpinBox()
        self.group_spinbox.setRange(0, 1)
        self.group_spinbox.setValue(0)
        self.group_spinbox.valueChanged.connect(self.update_group)
        group_save_layout.addWidget(QLabel("Group:"))
        group_save_layout.addWidget(self.group_spinbox)
        self.group_offset_spinbox = QSpinBox()
        self.group_offset_spinbox.setRange(-1000, 1000)
        self.group_offset_spinbox.setValue(0)
        group_save_layout.addWidget(QLabel("Group Offset:"))
        group_save_layout.addWidget(self.group_offset_spinbox)
        self.offset_group_button = QPushButton("Apply Group Offset")
        self.offset_group_button.clicked.connect(self.offset_group)
        group_save_layout.addWidget(self.offset_group_button)
        self.merge_group_spinbox = QSpinBox()
        self.merge_group_spinbox.setRange(-1000, 1000)
        self.merge_group_spinbox.setValue(0)
        group_save_layout.addWidget(QLabel("Merge Group Index:"))
        group_save_layout.addWidget(self.merge_group_spinbox)
        self.merge_group_button = QPushButton("Merge Group")
        self.merge_group_button.clicked.connect(self.merge_group)
        group_save_layout.addWidget(self.merge_group_button)
        cellar_controls_layout.addLayout(group_save_layout)

        main_layout.addLayout(cellar_controls_layout)
        
        # Create scatter items for displaying points.
        self.xy_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xz_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xy_plot.addItem(self.xy_scatter)
        self._overlay = QGraphicsPathItem()
        # semi‑transparent blue (alpha=150), with round caps
        pen = QPen(QColor(50, 150, 255, 150))
        pen.setWidthF(self.radius_spinbox.value() * 2)  # diameter = 2*radius
        pen.setCapStyle(Qt.RoundCap)
        self._overlay.setPen(pen)
        self._overlay.setZValue(1000)  # draw on top of everything
        self.xy_plot.addItem(self._overlay)

        # keep the overlay line in sync if the user changes the brush radius
        self.radius_spinbox.valueChanged.connect(self._update_overlay_pen)
        self.radius_slider.valueChanged.connect(self._update_overlay_pen)

        self.xy_plot.addItem(self._overlay)
        self.xz_plot.addItem(self.xz_scatter)
        self._overlay_xz = QGraphicsPathItem()
        pen_xz = QPen(QColor(50,150,255,150))
        pen_xz.setWidthF(self.radius_spinbox.value()*2)
        pen_xz.setCapStyle(Qt.RoundCap)
        self._overlay_xz.setPen(pen_xz)
        self._overlay_xz.setZValue(1000)
        self.xz_plot.addItem(self._overlay_xz)
        self.radius_spinbox.valueChanged.connect(self._update_overlay_xz)
        self.radius_slider.valueChanged.connect(self._update_overlay_xz)

        self.xy_calc_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xz_calc_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xy_plot.addItem(self.xy_calc_scatter)
        self.xz_plot.addItem(self.xz_calc_scatter)
        
        self.xy_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        self.xz_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        
        self._enable_pencil(self.xy_plot, self.xy_scatter, view_name='xy')
        self._enable_pencil(self.xz_plot, self.xz_scatter, view_name='xz')
        
        self.xy_plot.scene().installEventFilter(self)
        self.xz_plot.scene().installEventFilter(self)
        self.xy_plot.viewport().installEventFilter(self)
        self.xz_plot.viewport().installEventFilter(self)
        
        self.update_slider_ranges()

        # Restore Z slice settings from config
        if "z_slice_center" in self.config:
            self.z_center_spinbox.setValue(self.config["z_slice_center"])
        if "z_slice_thickness" in self.config:
            self.z_thickness_spinbox.setValue(self.config["z_slice_thickness"])
        # Restore f_init slice settings from config
        if "f_init_center" in self.config:
            self.finit_center_spinbox.setValue(self.config["f_init_center"])
        if "f_init_thickness" in self.config:
            self.finit_thickness_spinbox.setValue(self.config["f_init_thickness"])

        self.update_guides()
        self.update_views()

        # Enforce a minimum width on interactive widgets (excluding spinboxes) so they remain clickable
        MIN_WIDTH = 20
        for cls in (QPushButton, QLabel, QSlider, QCheckBox, QComboBox):
            for w in self.findChildren(cls):
                w.setMinimumWidth(MIN_WIDTH)
        # Add a QSizeGrip in the status bar so users can resize via the bottom-right corner
        self.statusBar().setSizeGripEnabled(True) # Enable default double grip

    def z_slice_center_changed(self):
        if hasattr(self, 'ome_zarr_window') and self.ome_zarr_window is not None:
            self.ome_zarr_window.update_z_slice_center(self.z_center_spinbox.value())
        self.update_views()

    def update_fstar_indicators(self):
        # Update the positions of the f* indicator lines
        self.line_fstar_min.setPos(self.fstar_min_spinbox.value())
        self.line_fstar_max.setPos(self.fstar_max_spinbox.value())
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
        # Save slice and color settings
        self.config["z_slice_center"] = self.z_center_spinbox.value()
        self.config["z_slice_thickness"] = self.z_thickness_spinbox.value()
        self.config["f_init_center"] = self.finit_center_spinbox.value()
        self.config["f_init_thickness"] = self.finit_thickness_spinbox.value()
        self.config["num_colors"] = self.num_colors
        # Persist teflon label number
        self.config["teflon_label"] = self.teflon_label
        self.config["graph_version"] = self.graph_version
        save_config(self.config, "config_labeling_gui.json")
        event.accept()
    
    # --------------------------------------------------
    # Event filter for cursor circle.
    # --------------------------------------------------
    def eventFilter(self, source, event):
        # 1) catch the mouse back/forward button presses
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.XButton1:    # "Back" button
                self.label_spinbox.setValue(self.label_spinbox.value() - 1)
                return True
            elif event.button() == Qt.XButton2:  # "Forward" button
                self.label_spinbox.setValue(self.label_spinbox.value() + 1)
                return True
                
        if event.type() == QEvent.MouseMove and self.drawing_mode_checkbox.isChecked():
            # XY
            if source is self.xy_plot.scene():
                plot, circle = self.xy_plot, self.cursor_circle
            # XZ
            elif source is self.xz_plot.scene():
                plot, circle = self.xz_plot, self.cursor_circle_xz
            else:
                return super().eventFilter(source, event)

            pos = event.scenePos()
            dataPos = plot.plotItem.vb.mapSceneToView(pos)
            r = self.radius_spinbox.value()
            circle.setRect(dataPos.x() - r, dataPos.y() - r, 2*r, 2*r)
            circle.setVisible(True)
            return False

        # if they moved off either view, hide both
        if event.type() in (QEvent.Leave, QEvent.HoverLeave):
            self.cursor_circle.setVisible(False)
            self.cursor_circle_xz.setVisible(False)

        return super().eventFilter(source, event)
    
    # --------------------------------------------------
    # Setup menu.
    # --------------------------------------------------
    def _create_menu(self):
        menu_bar = self.menuBar()

        ### Deactivating until complete implementation ###
        # project_menu = menu_bar.addMenu("Project")
        # load_action = QAction("Load Project", self)
        # load_action.triggered.connect(self.load_project)
        # project_menu.addAction(load_action)
        # save_project_action = QAction("Save Project", self)
        # save_project_action.triggered.connect(self.save_project_to_path)
        # project_menu.addAction(save_project_action)

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
        reset_labels_action = QAction("Reset Labels", self)
        reset_labels_action.triggered.connect(self.reset_labels)
        data_menu.addAction(reset_labels_action)

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
        # Connect the labels updated signal
        self.ome_zarr_window.labels_updated_signal.connect(self.on_ome_zarr_labels_updated)
        self.ome_zarr_window.show()

    def on_ome_zarr_view_destroyed(self):
        print("OME-Zarr view window has been destroyed; setting pointer to None.")
        self.ome_zarr_window = None

    def load_project(self):
        # Choose path to load config from
        fname, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Jason Files (*.json);;All Files (*)")
        if fname:
            load_config(fname)
            self.graph_path = self.config.get("graph_path", self.graph_path)
            self.default_experiment = self.config.get("default_experiment", self.default_experiment)
            self.graph_version = self.config.get("graph_version", self.graph_version)
            self.ome_zarr_path = self.config.get("ome_zarr_path", self.ome_zarr_path)
            self.graph_pkl_path = self.config.get("graph_pkl_path", self.graph_pkl_path)
            self.h5_path = self.config.get("h5_path", self.h5_path)
            self.umbilicus_path = self.config.get("umbilicus_path", self.umbilicus_path)
            if self.graph_path:
                self.solver = graph_problem_gpu_py.Solver(self.graph_path)
                gt_path = os.path.join("../experiments", self.default_experiment, "checkpoints", "checkpoint_graph_tugging.bin")
                version = self.graph_version
                if not os.path.exists(gt_path):
                    gt_path = os.path.join("../experiments", self.default_experiment, "checkpoints", "checkpoint_graph_f_star_final.bin")
                    # Keep the configured version even for the fallback file
                if os.path.exists(gt_path):
                    self.solver.load_graph(gt_path, version)
                else:
                    QMessageBox.warning(self, "Load Data", f"Graph file not found at {gt_path}")
                self.update_positions(update_labels=True)
                self.update_slider_ranges()
                self.update_views()
            else:
                QMessageBox.warning(self, "Load Project", "Graph path not found in config.")
                return

    def save_project_to_path(self):
        # save config to path
        fname, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Jason Files (*.json);;All Files (*)")
        if fname:
            self.config["graph_path"] = self.graph_path
            self.config["default_experiment"] = self.default_experiment
            self.config["graph_version"] = self.graph_version
            if self.ome_zarr_path:
                self.config["ome_zarr_path"] = self.ome_zarr_path
            if self.graph_pkl_path:
                self.config["graph_pkl_path"] = self.graph_pkl_path
            if self.h5_path:
                self.config["h5_path"] = self.h5_path
            if self.umbilicus_path:
                self.config["umbilicus_path"] = self.umbilicus_path
            # Save slice and color settings
            self.config["z_slice_center"] = self.z_center_spinbox.value()
            self.config["z_slice_thickness"] = self.z_thickness_spinbox.value()
            self.config["f_init_center"] = self.finit_center_spinbox.value()
            self.config["f_init_thickness"] = self.finit_thickness_spinbox.value()
            self.config["num_colors"] = self.num_colors
            # Persist teflon label number
            self.config["teflon_label"] = self.teflon_label
            save_config(self.config, fname)

    def load_data(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Data")
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        bin_path_lineedit = QLineEdit()
        bin_path_lineedit.setText(self.graph_path)
        browse_button = QPushButton("Browse...")
        browse_button.setToolTip("Click to choose the bin folder")
        h_layout = QHBoxLayout()
        h_layout.addWidget(bin_path_lineedit)
        h_layout.addWidget(browse_button)
        form_layout.addRow("Bin Folder:", h_layout)
        exp_lineedit = QLineEdit()
        exp_lineedit.setText(self.default_experiment)
        form_layout.addRow("Experiment name:", exp_lineedit)
        
        # Add graph version selector
        version_spinbox = QSpinBox()
        version_spinbox.setRange(0, 10)
        version_spinbox.setValue(self.graph_version)
        version_spinbox.setToolTip("Graph version to load (default: 3)")
        form_layout.addRow("Graph Version:", version_spinbox)
        
        layout.addLayout(form_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        browse_button.clicked.connect(lambda: self.browse_for_bin(bin_path_lineedit, exp_lineedit))
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        if dialog.exec_() == QDialog.Accepted:
            selected_dir = bin_path_lineedit.text().strip()
            exp_name = exp_lineedit.text().strip()
            selected_version = version_spinbox.value()
            self.default_experiment = exp_name
            self.graph_version = selected_version
            self.config["graph_version"] = self.graph_version
            if not selected_dir or not exp_name:
                return
            bin_file_path = selected_dir # os.path.join(selected_dir, "graph.bin")
            if not os.path.exists(bin_file_path):
                QMessageBox.warning(self, "Load Data", f"File {bin_file_path} does not exist.")
                return
            self.graph_path = bin_file_path
            self.solver = graph_problem_gpu_py.Solver(self.graph_path)
            gt_path = os.path.join("../experiments", exp_name, "checkpoints", "checkpoint_graph_tugging.bin")
            if not os.path.exists(gt_path):
                gt_path = os.path.join("../experiments", exp_name, "checkpoints", "checkpoint_graph_f_star_final.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path, selected_version)
            else:
                QMessageBox.warning(self, "Load Data", f"Graph file not found at {gt_path}")

            self.update_positions(update_labels=True)

    def update_positions(self, update_labels=False, update_slide_ranges=True):
        self.recompute = True
        new_points = np.array(self.solver.get_positions())
        print(f"Points shape previous vs new: {self.points.shape} vs {new_points.shape}")
        self.points = np.array(new_points)
        if update_labels:
            self.original_points = np.array(self.points)
            self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
            self.calculated_labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
            self.group = np.zeros(len(self.points), dtype=np.int32)

        if self.show_original_points:
            self.toggle_original_points()
        self.kdtree_xy = cKDTree(self.points[:, [0, 1]])
        self.kdtree_xz = cKDTree(self.points[:, [0, 2]])
        if update_slide_ranges:
            self.update_slider_ranges()
        self.update_views()
        print(f"Done updating positions.")
    
    def browse_for_bin(self, lineedit, exp_lineedit):
        graph_path, _ = QFileDialog.getOpenFileName(self, "Select Graph Bin", lineedit.text() or os.getcwd(), "Graph Files (*.bin);;All Files (*)")
        if graph_path:
            lineedit.setText(graph_path)
            # if exp_lineedit empty, set it to the directory name
            if not exp_lineedit.text():
                exp_name = os.path.basename(os.path.dirname(graph_path))
                exp_lineedit.setText(exp_name)
    
    def save_labels_to_path(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Labels", os.path.join("../experiments", self.default_experiment), "Text Files (*.txt);;All Files (*)")
        if not fname.endswith(".txt"):
            fname += ".txt"
        if fname:
            self._save_labels_to_path(fname)
            # Track this as the last manual save path for autosave reference
            self.last_manual_save_path = fname
            # Update the last saved state
            self._update_last_saved_state()
    
    def _save_labels_to_path(self, fname):
        if fname:
            with open(fname, "w") as f:
                # write labels, group, and streak flags to file
                payload = {
                    "labels": self.labels.tolist(),
                    "group": self.group.tolist(),
                    "streaks": self.streaks.tolist()
                }
                data = json.dumps(payload)
                f.write(data)
            print(f"Labels saved to {fname}")
    
    def load_labels_from_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Labels", os.path.join("../experiments", self.default_experiment),
                                                 "Text Files (*.txt);;All Files (*)")
        if fname:
            with open(fname, "r") as f:
                data = json.load(f)
            try:
                new_labels = np.array(data["labels"], dtype=np.int32)
                new_groups = np.array(data["group"], dtype=np.int32)
                # Load streaks if present, else default to False and warn
                if "streaks" in data:
                    new_streaks = np.array(data["streaks"], dtype=bool)
                else:
                    QMessageBox.warning(self, "Load Labels",
                                        "Old labels file: no streak data found; all streaks set to False.")
                    new_streaks = np.zeros(len(new_labels), dtype=bool)
            except Exception as e:
                # Fallback to old label save format (no groups or streaks)
                print("Falling back to old label format.")
                with open(fname, "r") as f:
                    data = f.read()
                try:
                    new_labels = np.array(ast.literal_eval(data), dtype=np.int32)
                    new_groups = np.zeros(len(new_labels), dtype=np.int32)
                    new_streaks = np.zeros(len(new_labels), dtype=bool)
                except Exception as e:
                    QMessageBox.warning(self, "Load Labels", f"Error reading file: {e}")
                    return
            if len(new_labels) == len(self.labels):
                self.labels = new_labels
                self.group = new_groups
                self.streaks = new_streaks
                # Track this load path for autosave reference
                self.last_load_path = fname
                # Update the last saved state to reflect what was just loaded
                self._update_last_saved_state()
                self.update_views()
            else:
                QMessageBox.warning(self, "Load Labels", "Loaded labels length does not match current data.")

    def reset_labels(self):
        self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        self.group = np.zeros(len(self.points), dtype=np.int32)
        self.update_views()
    
    def show_help(self):
        help_html = """
        <html>
        <head>
        <style>
            body {
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 11pt;
            }
            h2 {
            font-size: 14pt;
            margin: 0 0 0.4em 0;
            }
            h3 {
            font-size: 12pt;
            margin: 1em 0 0.4em 0;
            }
            ul {
            margin-top: 0;
            margin-bottom: 1em;
            padding-left: 1.2em;
            }
            li {
            margin-bottom: 0.5em;
            }
            .shortcut {
            font-weight: bold;
            color: #2222AA;
            }
            .section-title {
            font-weight: bold;
            color: #AA2222;
            }
        </style>
        </head>
        <body>

        <h2>Graph Labeler Usage Instructions</h2>

        <h3 class="section-title">1. Views</h3>
        <ul>
            <li><strong>XY View (left):</strong> f_star (horizontal) vs. f_init (vertical).</li>
            <li><strong>XZ View (right):</strong> f_star (horizontal) vs. Z (vertical).</li>
        </ul>

        <h3 class="section-title">2. Slice Controls</h3>
        <ul>
            <li>Use the Z-slice sliders (left view) and f_init-slice sliders (right view) to focus on a "slab."</li>
            <li>The 'Thickness' sliders define how tall each slab is (in Z or in f_init).</li>
            <li>The 'Z selection' and 'f selection' controls further isolate areas if needed.</li>
        </ul>

        <h3 class="section-title">3. Shear Controls</h3>
        <ul>
            <li><strong>XZ Shear (right):</strong> Shears the XZ view horizontally.</li>
            <li><strong>XY Vertical Shear (left):</strong> Shifts f_init in the XY view based on Z.</li>
            <li><strong>XY Horizontal Shear (left):</strong> Shifts f_star in the XY view based on Z.</li>
            <li>Helps visually separate or untangle overlapping sheets.</li>
        </ul>

        <h3 class="section-title">4. Drawing & Labeling</h3>
        <ul>
            <li>Left-click and drag on the view to label points (with the label in the bottom spinbox).</li>
            <li>"Drawing radius" controls the brush size for labeling.</li>
            <li>"Erase Label" sets the label spinbox to the "unlabeled" code, so you can drag to remove labels.</li>
            <li>The "Group" spinbox (and offset/merge tools) let you manage multi-layer labeling.</li>
        </ul>

        <h3 class="section-title">5. Updated Label Tools</h3>
        <ul>
            <li>After running a solver ("Update Labels"), some points get an "updated" (calculated) label.</li>
            <li><span class="shortcut">U</span> toggles "Update Labels Draw Mode": paint only those updated labels onto unlabeled points.</li>
            <li>"Apply All Updated Labels" replaces every unlabeled point with its calculated label.</li>
            <li>"Apply Updated Labels to XY/XZ" applies only to the visible slab in that view.</li>
            <li>"Clear Updated Labels" removes them entirely.</li>
        </ul>

        <h3 class="section-title">6. Spline Tools (Top Row)</h3>
        <ul>
            <li>"Update Spline" fits polynomial lines to labeled winding sheets (for big structures like spiral/winding shapes).</li>
            <li>"Assign Line Labels" snaps nearby unlabeled points to those fitted lines if they're close.</li>
            <li>"Clear Splines" removes any drawn splines.</li>
        </ul>

        <h3 class="section-title">7. Common Controls (Bottom Row)</h3>
        <ul>
            <li><strong>Drawing radius</strong> = brush size for labeling.</li>
            <li><strong>Max Display Points</strong> limits the visible points (speeds up display).</li>
            <li><strong>Drawing Mode</strong> toggles whether left-click labeling is active.</li>
            <li><strong>Pipette</strong> (or press <span class="shortcut">P</span>) picks up a label under the cursor.</li>
            <li><strong>Save Labeled Graph</strong> writes a new graph file with unlabeled points removed.</li>
        </ul>

        <h3 class="section-title">8. Additional Tools & Info</h3>
        <ul>
            <li><strong>Filter Teflon / Reset Teflon</strong>: specialized for Teflon-labeled regions.</li>
            <li><strong>Compute Slabs</strong>: advanced auto-label in multiple Z-slab steps.</li>
            <li><strong>Reset Points</strong> reverts all data to the original positions.</li>
        </ul>

        <h3 class="section-title">9. Key Shortcuts</h3>
        <ul>
            <li><span class="shortcut">S (hold)</span>: Temporarily disable drawing (release to restore).</li>
            <li><span class="shortcut">P</span>: Pipette a label.</li>
            <li><span class="shortcut">G</span>: Pipette a group & label.</li>
            <li><span class="shortcut">U</span>: Toggle updated-labels paint mode.</li>
            <li><span class="shortcut">Ctrl+Z</span>: Undo.</li>
            <li><span class="shortcut">Ctrl+Y</span>: Redo.</li>
            <li><span class="shortcut">Space or O</span>: Toggle original vs. computed point positions (view-only).</li>
            <li><span class="shortcut">Up/Down Arrow</span>: Increase/decrease the label spinbox.</li>
            <li><span class="shortcut">C</span>: Hide/show <em>all</em> labeled points.</li>
            <li><span class="shortcut">L</span>: Hide/show the estimated color shading for unlabeled points.</li>
        </ul>

        <h3 class="section-title">10. Resetting Points</h3>
        <ul>
            <li><strong>Reset Points</strong> (bottom row) truly reverts the dataset to the original 3D coords.</li>
            <li><strong>Space or O</strong> only toggles the <em>view</em> between original vs. computed—does not permanently move them.</li>
        </ul>

        <h3 class="section-title">11. Saving & Loading</h3>
        <ul>
            <li>Use the "Data" menu to load data (the .bin file), or load/save .txt label files.</li>
            <li>"Reset Labels" discards all labeling, making every point unlabeled again.</li>
            <li>"Save Labeled Graph" exports the final labeled graph, removing any unlabeled nodes.</li>
        </ul>

        </body>
        </html>
        """

        # Create a custom QDialog so we can show a scrollable text browser
        dialog = QDialog(self)
        dialog.setWindowTitle("Usage Instructions")
        
        layout = QVBoxLayout(dialog)
        text_browser = QTextBrowser(dialog)
        text_browser.setHtml(help_html)
        text_browser.setOpenExternalLinks(False)
        text_browser.setReadOnly(True)

        layout.addWidget(text_browser)
        # Set an initial size for the dialog; the user can still resize
        dialog.resize(700, 500)
        
        dialog.exec_()

    def update_slider_ranges(self):
        if self.points is not None and len(self.points) > 0:
            self.z_min = float(np.min(self.points[:, 2]))
            self.z_max = float(np.max(self.points[:, 2]))
        else:
            self.z_min = 0.0
            self.z_max = 1.0
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
        thickness_val = min(thickness_val, 100.0)
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
    
    def _interpolate_path(self, path, r):
        """
        Densify a list of (x,y) points so no segment exceeds half the radius r.
        """
        if not path:
            return []
        step = r * 0.5
        interp = []
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            interp.append((x0, y0))
            dx = x1 - x0; dy = y1 - y0
            dist = math.hypot(dx, dy)
            if dist > step:
                n = int(dist / step)
                for k in range(1, n + 1):
                    t = k / (n + 1)
                    interp.append((x0 + t * dx, y0 + t * dy))
        interp.append(path[-1])
        return interp
    
    def update_guides(self):
        # ---------------------------
        # For the XY view (Main guides):
        # ---------------------------
        if self.show_guides_checkbox.isChecked():
            # Add existing finit guidelines.
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
            # Orange indicator (for XZ shear) shown in XY view.
            self.xz_shear_indicator.setAngle(self.xz_shear_spinbox.value())
            center_f_star = (self.f_star_min + self.f_star_max) / 2
            center_f_init = (self.f_init_min + self.f_init_max) / 2
            self.xz_shear_indicator.setPos(QPointF(center_f_star, center_f_init))
            if self.xz_shear_indicator.scene() is None:
                self.xy_plot.addItem(self.xz_shear_indicator)

            # ---------------------------
            # Additional f-selection guides for XY view:
            # (These are vertical purple lines marking the selection center and the boundaries.)
            f_sel_thickness = self.f_selection_thickness_slider.value() / self.scaleFactor
            if f_sel_thickness:
                # The f selection center is taken from the finit_selection_spinbox.
                f_sel_center = self.finit_selection_spinbox.value()
                left_sel = f_sel_center - f_sel_thickness
                right_sel = f_sel_center + f_sel_thickness
                if not hasattr(self, 'f_selection_left_line'):
                    self.f_selection_left_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
                if not hasattr(self, 'f_selection_right_line'):
                    self.f_selection_right_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
                if not hasattr(self, 'f_selection_center_line'):
                    self.f_selection_center_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
                self.f_selection_left_line.setPos(left_sel)
                self.f_selection_right_line.setPos(right_sel)
                self.f_selection_center_line.setPos(f_sel_center)
                if self.f_selection_left_line.scene() is None:
                    self.xy_plot.addItem(self.f_selection_left_line)
                if self.f_selection_right_line.scene() is None:
                    self.xy_plot.addItem(self.f_selection_right_line)
                if self.f_selection_center_line.scene() is None:
                    self.xy_plot.addItem(self.f_selection_center_line)
            else:
                # Remove f selection guides if thickness is zero.
                for attr in ['f_selection_left_line', 'f_selection_right_line', 'f_selection_center_line']:
                    if hasattr(self, attr):
                        line_item = getattr(self, attr)
                        if line_item.scene() is not None:
                            self.xy_plot.removeItem(line_item)
        else:
            # Remove all XY guide items when guides are turned off.
            for item in [self.line_finit_neg, self.line_finit_pos, self.line_finit_center,
                        self.line_finit_upper, self.line_finit_lower, self.xz_shear_indicator,
                        getattr(self, 'f_selection_left_line', None),
                        getattr(self, 'f_selection_right_line', None),
                        getattr(self, 'f_selection_center_line', None)]:
                if item is not None and item.scene() is not None:
                    self.xy_plot.removeItem(item)

        # ---------------------------
        # f* Range Guides for XY view:
        # ---------------------------
        if self.use_fstar_range_checkbox.isChecked():
            if self.line_fstar_min.scene() is None:
                self.xy_plot.addItem(self.line_fstar_min)
            if self.line_fstar_max.scene() is None:
                self.xy_plot.addItem(self.line_fstar_max)
        else:
            for item in [self.line_fstar_min, self.line_fstar_max]:
                if item.scene() is not None:
                    self.xy_plot.removeItem(item)

        # ---------------------------
        # For the XZ view (Main guides):
        # ---------------------------
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

            # ---------------------------
            # Additional z-selection guides for XZ view:
            # (These are horizontal purple lines marking the selection center and the boundaries.)
            z_sel_thickness = self.z_selection_thickness_slider.value() / self.scaleFactor
            if z_sel_thickness:
                z_sel_center = self.z_selection_spinbox.value()
                top_sel = z_sel_center + z_sel_thickness / 2
                bottom_sel = z_sel_center - z_sel_thickness / 2
                if not hasattr(self, 'z_selection_top_line'):
                    self.z_selection_top_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
                if not hasattr(self, 'z_selection_bottom_line'):
                    self.z_selection_bottom_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
                if not hasattr(self, 'z_selection_center_line'):
                    self.z_selection_center_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('purple', width=1, style=Qt.DashLine))
                self.z_selection_top_line.setPos(top_sel)
                self.z_selection_bottom_line.setPos(bottom_sel)
                self.z_selection_center_line.setPos(z_sel_center)
                if self.z_selection_top_line.scene() is None:
                    self.xz_plot.addItem(self.z_selection_top_line)
                if self.z_selection_bottom_line.scene() is None:
                    self.xz_plot.addItem(self.z_selection_bottom_line)
                if self.z_selection_center_line.scene() is None:
                    self.xz_plot.addItem(self.z_selection_center_line)
            else:
                for attr in ['z_selection_top_line', 'z_selection_bottom_line', 'z_selection_center_line']:
                    if hasattr(self, attr):
                        line_item = getattr(self, attr)
                        if line_item.scene() is not None:
                            self.xz_plot.removeItem(line_item)
        else:
            for item in [self.line_z_center, self.line_z_upper, self.line_z_lower,
                        self.xy_horizontal_indicator,
                        getattr(self, 'z_selection_top_line', None),
                        getattr(self, 'z_selection_bottom_line', None),
                        getattr(self, 'z_selection_center_line', None)]:
                if item is not None and item.scene() is not None:
                    self.xz_plot.removeItem(item)

    def update_teflon(self, val):
        self.teflon_label = val
        self.update_views()

    def filter_teflon(self):
        # filter pointcloud with normal
        angle_thresh = self.filter_angle_spinbox.value()
        distance_thresh = self.filter_distance_spinbox.value()
        filter_mask = filter_point_normals(self.points * np.array([1.0, 0.2, 0.1]), np.array([1.0, 0.0, 0.0]), angle_thresh, radius=distance_thresh) # squeeze y and z to make the single lines/sheets more connected
        unlabeled_mask = np.abs(self.labels - self.UNLABELED) < 2
        mask_edge = np.logical_and(self.points[:, 1] > -165, self.points[:, 1] < 165) # hacky way to not filter out nodes that are close to the wrap-around (not points in cartesian -> wrongly assigning 90 deg angle, therefore need to not filter out)
        mask = np.logical_and(filter_mask, unlabeled_mask)
        mask = np.logical_and(mask, mask_edge)
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []

        # assign to group 0 and label teflon
        self.group[mask] = 0
        self.labels[mask] = self.teflon_label
        self.update_views()
        self.update_groups_data()

    def reset_teflon(self):
        # reset teflon to unlabeled, group 0
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []

        mask = self.labels == self.teflon_label
        self.labels[mask] = self.UNLABELED
        self.group[mask] = 0
        self.update_views()

    def toggle_gt_labels(self):
        if self.show_gt_labels_button.isChecked():
            gt_f_stars, indices = self.solver.get_gt_f_star()
            gt_f_stars = np.array(gt_f_stars)
            indices = np.array(indices)
            print(f"Dtype of gt_f_stars: {gt_f_stars.dtype}")
            print(f"Length of gt_f_stars: {len(gt_f_stars)}")
            print(f"Dtype of indices: {indices.dtype}")
            print(f"Length of indices: {len(indices)}")
            indices = indices.astype(np.int32)

            self.gt_labels = np.zeros(len(self.points), dtype=np.int32) + self.UNLABELED * 360.0
            self.gt_labels[indices] = np.array(gt_f_stars)
            self.update_views()
        else:
            self.gt_labels = None
            self.update_views()

    def gt_labels_to_group(self):
        gt_f_stars, indices = self.solver.get_gt_f_star()
        indices = np.array(indices, dtype=np.int32)
        gt_wrap_nrs = np.array(gt_f_stars) / 360.0
        gt_wrap_nrs = np.round(gt_wrap_nrs).astype(np.int32)
        if len(gt_wrap_nrs) == 0:
            print("No GT labels found.")
            return
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []
        new_group_nr = np.max(self.group) + 1
        self.group[indices] = new_group_nr
        self.labels[indices] = gt_wrap_nrs

    def update_group(self, val):
        self.active_group = val
        self.update_views()

    def update_groups_data(self):
        # Do only once every n_seconds
        n_seconds = 5
        if time.time() - self.last_group_update < n_seconds:
            return
        self.last_group_update = time.time()

        self.max_group = np.max(self.group)
        self.max_group_label.setText(f"Max Group: {self.max_group}")
        self.group_spinbox.setMaximum(self.max_group + 1)

    def offset_group(self):
        offset = int(self.group_offset_spinbox.value())
        if self.active_group == 0:
            return
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []

        mask_group = self.group == self.active_group
        mask_labeled = np.abs(self.labels - self.UNLABELED) > 2
        mask_group_labeled = np.logical_and(mask_group, mask_labeled)
        self.labels[mask_group_labeled] += offset # offset only the labeled points
        # Set the offset group back to 0.
        self.group_offset_spinbox.setValue(0)
        self.update_views()

    def merge_group(self):
        if self.active_group == 0:
            return
        
        merge_group = int(self.merge_group_spinbox.value())
        if merge_group == self.active_group:
            return
        
        # Open Ask Dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Merge Group")
        msg.setInformativeText(f"Merge the active group ({self.active_group}) with the {merge_group} group?")
        msg.setWindowTitle("Merge Group")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        ret = msg.exec_()
        if ret == QMessageBox.No:
            print("Merge Group cancelled.")
            return
        print(f"Merging group {self.active_group} with {merge_group} group.")
        
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []
        
        self.group[self.group == self.active_group] = merge_group
        mask_higher_groups = self.group > self.active_group
        self.group[mask_higher_groups] -= 1
        if merge_group > self.active_group:
            self.active_group = merge_group - 1
        else:
            self.active_group = merge_group
        self.merge_group_spinbox.setValue(0)
        self.group_spinbox.setValue(self.active_group)

        self.update_views()

    def _update_overlay_pen(self, radius):
        """When the radius slider changes, update the overlay pen thickness."""
        pen = self._overlay.pen()
        pen.setWidthF(radius * 2)
        self._overlay.setPen(pen)

    def _update_overlay_xz(self, radius):
        pen = self._overlay_xz.pen()
        pen.setWidthF(radius * 2)
        self._overlay_xz.setPen(pen)
    
    def downsample_points(self, pts, max_display=1, axis=None):
        if self.local_downsample_checkbox.isChecked():
            indices = self.subsample_points_local(pts, max_display=max_display, axis=axis)
        else:  
            indices = self.global_subsample_points(pts, max_display=max_display)
        return indices

    def global_subsample_points(self, pts, max_display=1):
        n = pts.shape[0]
        if n > max_display:
            indices = np.linspace(0, n - 1, max_display, dtype=int)
            return indices
        else:
            return np.arange(n)
            
    def subsample_points_local(self, points, max_display=1, grid_size=50, axis=None):
        """
        Subsample points in 2D such that each grid cell (of size grid_size)
        contains at most max_per_cell points.

        Parameters:
            points (np.ndarray): An (N, 2) array of 2D points.
            grid_size (float): The size of each grid cell.
            max_per_cell (int): Maximum number of points to keep per cell.

        Returns:
            np.ndarray: The subsampled points.
        """
        max_per_cell = max_display
        # Compute the lower bound of the grid and cell indices for each point.
        if len(points) == 0:
            return np.array([], dtype=int)
        min_xy = points.min(axis=0)
        grid_size = np.array([grid_size, grid_size, 7*grid_size])
        cell_indices = ((points - min_xy) // grid_size).astype(int)
        if axis is not None:
            if axis == 'xy':
                cell_indices = cell_indices[:, :2]
            elif axis == 'xz':
                cell_indices = cell_indices[:, [0, 2]]
            else:
                raise ValueError("Invalid axis. Use 'xy' or 'xz'.")
        
        # Use np.ravel_multi_index to compute a unique cell ID for each point.
        max_indices = cell_indices.max(axis=0) + 1
        cell_ids = np.ravel_multi_index(cell_indices.T, dims=tuple(max_indices))
        
        # Sort all points by their cell_id.
        order = np.argsort(cell_ids)
        sorted_cell_ids = cell_ids[order]
        
        # Get the start index and counts for each unique cell.
        unique_cells, first_index, counts = np.unique(sorted_cell_ids, return_index=True, return_counts=True)
        
        # For each unique cell, randomly select up to max_per_cell indices.
        selected_indices = []
        for idx, count in zip(first_index, counts):
            # indices of points in the current cell
            cell_group = order[idx:idx+count]
            if count > max_display:
                indices = np.linspace(0, count-1, max_display, dtype=int)
            else:
                indices = np.arange(count)
            # Keep only up to max_per_cell points
            selected_indices.append(cell_group[indices])
        
        # Concatenate selected indices and return the subsampled points.
        keep_indices = np.concatenate(selected_indices)
        return keep_indices
    
    def get_brushes_from_labels(self, labels_array, group_array, points_array):
        brushes = np.empty(labels_array.shape[0], dtype=object)
        if self.hide_labels:
            brushes[:] = self.brush_black
            return brushes

        # Identify the unlabeled points.
        mask_unlabeled = (
            (labels_array == self.UNLABELED)
        | (labels_array == self.UNLABELED + 1)
        | (labels_array == self.UNLABELED - 1)
        )

        if self.hide_estimated_colors:
            brushes[mask_unlabeled] = self.brush_black
        else:
            # For each unlabeled point, compute its color using its original x coordinate.
            x_vals = points_array[mask_unlabeled, 0]
            y_vals = points_array[mask_unlabeled, 1]
            # Compute the color value using the mapping: ((x * 20) - y) / 360, wrapped mod 3.
            color_vals = (((x_vals * 20) - y_vals) / 360) % 3
            # Round to one decimal place.
            rounded_vals = np.round(color_vals, 1)
            # Wrap around any values equal or exceeding 3.0.
            rounded_vals[rounded_vals >= 3.0] = 0.0
            # Each 0.1 increment corresponds to an index.
            indices = np.array(np.round(rounded_vals * 10), dtype=int)
            # Sanity-check: force any out-of-bound indices to 0.
            indices = np.where((indices < 0) | (indices >= len(self.unlabeled_brushes)), 0, indices)
            # Assign the precomputed brushes using vectorized indexing.
            brushes[mask_unlabeled] = self.unlabeled_brushes[indices]

        # For "labeled" points, continue with your existing logic.
        mask_valid = ~mask_unlabeled
        valid_indices = np.where(mask_valid)[0]
        if valid_indices.size:
            valid_labels = labels_array[mask_valid]
            valid_groups = group_array[mask_valid]
            mask_active_group = valid_groups == self.active_group
            # Cycle labels over current number of colors
            mod = valid_labels % self.num_colors
            for j, idx in enumerate(valid_indices):
                # Teflon hiding logic
                if valid_groups[j] == 0 and (abs(valid_labels[j] - self.teflon_label) < 2) and self.hide_teflon_checkbox.isChecked():
                    brushes[idx] = self.transparent_brush
                else:
                    if mask_active_group[j]:
                        brushes[idx] = self.active_brushes[mod[j]]
                    else:
                        brushes[idx] = self.inactive_brushes[mod[j]]
        return brushes
    
    def update_brushes_gt(self, brushes, gt_f_stars):
        for i, label in enumerate(gt_f_stars):
            if abs(label/360.0 - self.UNLABELED) > 2:
                # New brush, use color gradient r g b for each 360 deg one step
                # 0 -> red, 360 -> green, 720 -> blue
                label_mod = label % 1080
                if label_mod < 360:
                    rest = label_mod / 360
                    color = (255 * (1 - rest), 255 * rest, 0)
                elif label_mod < 720:
                    rest = (label_mod - 360) / 360
                    color = (0, 255 * (1 - rest), 255 * rest)
                else:
                    rest = (label_mod - 720) / 360
                    color = (255 * rest, 0, 255 * (1 - rest))
                brushes[i] = pg.mkBrush(color[0], color[1], color[2], 150)
        return brushes
    
    def _on_colors_toggled(self, num, checked):
        """
        Handler for 4/5 color toggle buttons. Sets num_colors and updates views.
        """
        if checked:
            self.num_colors = num
            # Ensure mutual exclusivity
            if num == 4:
                self.colors5_button.setChecked(False)
            elif num == 5:
                self.colors4_button.setChecked(False)
        else:
            # Revert to default 3 colors
            self.num_colors = 3
        # Save to config
        self.config["num_colors"] = self.num_colors
        # save_config(self.config, "config_labeling_gui.json")
        self.update_views()
    
    def _enable_pencil(self, plot_widget, scatter_item, view_name='xy'):
        scatter_item.mousePressEvent = lambda ev: self._on_mouse_press(ev, plot_widget, view_name)
        scatter_item.mouseMoveEvent = lambda ev: self._on_mouse_drag(ev, plot_widget, view_name)
        scatter_item.mouseReleaseEvent = lambda ev: self._on_mouse_release(ev, plot_widget, view_name)

    def _current_label_qcolor(self):
        """Return a semi‐transparent QColor matching the current spin‑box label,
        but draw UNLABELED in the "unlabeled" color (black here)."""
        lab = self.label_spinbox.value()
        alpha = 150
        # In streak mode, use black for label 0 (clear) and color for label 1
        if getattr(self, 'streak_mode', False):
            if lab == 1:
                # color matching active brush for label 1
                base = self.active_brushes[1].color()
                return QColor(base.red(), base.green(), base.blue(), alpha)
            else:
                return QColor(0, 0, 0, alpha)

        if lab == self.UNLABELED:
            # draw erase strokes in your unlabeled color
            return QColor(0, 0, 0, alpha)

        # otherwise pick brush based on current number of colors (label % num_colors)
        idx = lab % self.num_colors
        # select the corresponding active brush from the configured list
        brush = self.active_brushes[idx]
        base = brush.color()
        return QColor(base.red(), base.green(), base.blue(), alpha)
    
    def _on_mouse_press(self, ev, plot_widget, view_name):
        # pipette / group‑pipette remain unchanged:
        if view_name == 'xy' and self.pipette_mode:
            ev.accept()
            self.pick_label_at(ev, plot_widget)
            self.pipette_mode = False
            return
        if view_name == 'xy' and self.group_pipette_mode:
            ev.accept()
            self.pick_group_at(ev, plot_widget)
            self.group_pipette_mode = False
            return
        if view_name == 'xz' and self.pipette_mode:
            ev.accept()
            self.pick_label_at_xz(ev, plot_widget)
            self.pipette_mode = False
            return
        if view_name == 'xz' and self.group_pipette_mode:
            ev.accept()
            self.pick_group_at_xz(ev, plot_widget)
            self.group_pipette_mode = False
            return

        # ---- start a brush stroke in XY ----
        if plot_widget is self.xy_plot \
        and ev.button() == Qt.LeftButton \
        and self.drawing_mode_checkbox.isChecked() \
        and not self.s_pressed:
            ev.accept()
            # backup for undo
            if self._stroke_backup is None:
                self._stroke_backup = (self.labels.copy(), self.group.copy())
            pt = self.xy_plot.plotItem.vb.mapSceneToView(ev.scenePos())
            self._current_path_xy = QPainterPath(QPointF(pt.x(), pt.y()))
            pen = self._overlay.pen()
            pen.setColor(self._current_label_qcolor())
            self._overlay.setPen(pen)
            self._overlay.setPath(self._current_path_xy)
            self._overlay.setVisible(True)
            return

        # ---- start a brush stroke in XZ ----
        if plot_widget is self.xz_plot \
        and ev.button() == Qt.LeftButton \
        and self.drawing_mode_checkbox.isChecked() \
        and not self.s_pressed:
            ev.accept()
            if self._stroke_backup is None:
                self._stroke_backup = (self.labels.copy(), self.group.copy())
            pt = self.xz_plot.plotItem.vb.mapSceneToView(ev.scenePos())
            self._current_path_xz = QPainterPath(QPointF(pt.x(), pt.y()))
            pen = self._overlay_xz.pen()
            pen.setColor(self._current_label_qcolor())
            self._overlay_xz.setPen(pen)
            self._overlay_xz.setPath(self._current_path_xz)
            self._overlay_xz.setVisible(True)
            return

        ev.ignore()
    
    def _on_mouse_drag(self, ev, plot_widget, view_name):
        # ---- extend XY stroke ----
        if plot_widget is self.xy_plot \
        and (ev.buttons() & Qt.LeftButton) \
        and self.drawing_mode_checkbox.isChecked() \
        and not self.s_pressed:
            ev.accept()
            pt = self.xy_plot.plotItem.vb.mapSceneToView(ev.scenePos())
            self._current_path_xy.lineTo(QPointF(pt.x(), pt.y()))
            self._overlay.setPath(self._current_path_xy)
            return

        # ---- extend XZ stroke ----
        if plot_widget is self.xz_plot \
        and (ev.buttons() & Qt.LeftButton) \
        and self.drawing_mode_checkbox.isChecked() \
        and not self.s_pressed:
            ev.accept()
            pt = self.xz_plot.plotItem.vb.mapSceneToView(ev.scenePos())
            self._current_path_xz.lineTo(QPointF(pt.x(), pt.y()))
            self._overlay_xz.setPath(self._current_path_xz)
            return

        ev.ignore()
    
    def _on_mouse_release(self, ev, plot_widget, view_name):
        # ---- finish XY stroke (label or streak) ----
        if plot_widget is self.xy_plot \
        and ev.button() == Qt.LeftButton \
        and self._stroke_backup is not None:
            ev.accept()
            pts = [(self._current_path_xy.elementAt(i).x,
                    self._current_path_xy.elementAt(i).y)
                for i in range(self._current_path_xy.elementCount())]
            if self.streak_mode:
                self._apply_brush_path_to_streaks(pts)
            else:
                self._apply_brush_path_to_labels(pts)
            # no undo for streak mode currently
            if not self.streak_mode:
                self.undo_stack.append(self._stroke_backup)
                self.redo_stack.clear()
            self._stroke_backup = None
            self._overlay.setPath(QPainterPath())
            self._overlay.setVisible(False)
            return

        # ---- finish XZ stroke (label or streak) ----
        if plot_widget is self.xz_plot \
        and ev.button() == Qt.LeftButton \
        and self._stroke_backup is not None:
            ev.accept()
            pts = [(self._current_path_xz.elementAt(i).x,
                    self._current_path_xz.elementAt(i).y)
                for i in range(self._current_path_xz.elementCount())]
            if self.streak_mode:
                self._apply_brush_path_to_streaks_xz(pts)
            else:
                self._apply_brush_path_to_labels_xz(pts)
            if not self.streak_mode:
                self.undo_stack.append(self._stroke_backup)
                self.redo_stack.clear()
            self._stroke_backup = None
            self._overlay_xz.setPath(QPainterPath())
            self._overlay_xz.setVisible(False)
            return

        ev.accept()  # eat all other releases so the view doesn't pan
    
    def _apply_brush_path_to_labels(self, data_path):
        """
        Apply the brush stroke (given as a list of data‐space (x,y) points)
        to update self.labels and self.group, preserving wrap‑around logic.
        """
        r    = self.radius_spinbox.value()
        # densify stroke points so gaps are smaller than brush radius/2
        data_path = self._interpolate_path(data_path, r)
        zc   = self.z_center_spinbox.value()
        half = self.z_thickness_slider.value() / self.scaleFactor / 2
        grp  = self.active_group
        base = self.label_spinbox.value()

        # 1) Collect for each point index the list of wrap offsets (+1, 0, -1)
        hit_offsets = {}   # idx -> list of offsets
        for x, y in data_path:
            off = 1 if y > 180 else (-1 if y < -180 else 0)
            for i in self.get_nearby_indices_xy(x, y, r):
                # only if inside current Z slab
                if (zc - half) <= self.points[i, 2] <= (zc + half):
                    hit_offsets.setdefault(i, []).append(off)

        if not hit_offsets:
            return

        # 2) For each point, pick the most common offset and apply
        for i, offs in hit_offsets.items():
            most_common_off = Counter(offs).most_common(1)[0][0]
            new_lab = base + most_common_off
            # prevent accidental UNLABELED±1
            if new_lab in (self.UNLABELED + 1, self.UNLABELED - 1):
                new_lab = self.UNLABELED

            self.labels[i] = new_lab
            self.group[i]  = grp

        # 3) Repaint
        self.update_views()

    def _apply_brush_path_to_labels_xz(self, data_path):
        """
        Apply the brush stroke (given as a list of data‐space (f_star, Z) points)
        to update self.labels and self.group, respecting the current f_init slab.
        """
        """
        Apply the brush stroke in the XZ view to update labels and group, using display-space
        coordinates that include the current XZ shear transform.
        """
        # brush radius in display-space units
        r = self.radius_spinbox.value()
        # densify stroke points so gaps are smaller than brush radius/2
        data_path = self._interpolate_path(data_path, r)
        # f_init slab center and half-thickness
        fc = self.finit_center_spinbox.value()
        half = self.finit_thickness_slider.value() / self.scaleFactor / 2
        grp = self.active_group
        base = self.label_spinbox.value()

        # shear factor for XZ view
        shear_factor = np.tan(np.radians(self.xz_shear_spinbox.value()))
        # select indices within the f_init slab
        slab_mask = (self.points[:, 1] >= (fc - half)) & (self.points[:, 1] <= (fc + half))
        slab_indices = np.nonzero(slab_mask)[0]
        if slab_indices.size == 0:
            return
        # compute display-space coordinates for slab points
        if self.show_original_points:
            pts = self.original_points[slab_indices]
        else:
            pts = self.points[slab_indices]
        x_disp = pts[:, 0] + shear_factor * (pts[:, 1] - fc)
        z_disp = pts[:, 2]
        coords = np.vstack([x_disp, z_disp]).T
        # build a temporary KD-tree on display coords
        tree = cKDTree(coords)

        hit = set()
        for xf, z in data_path:
            # find slab points near the stroke path in display space
            rel_idxs = tree.query_ball_point([xf, z], r=r)
            for j in rel_idxs:
                hit.add(slab_indices[j])

        if not hit:
            return

        # assign label and group to all hit points
        for i in hit:
            self.labels[i] = base
            self.group[i] = grp

        self.update_views()
    
    def _apply_brush_path_to_streaks(self, data_path):
        """
        Apply brush stroke in XY view to set boolean streak flags based on current spinbox (0/1).
        """
        r = self.radius_spinbox.value()
        data_path = self._interpolate_path(data_path, r)
        zc = self.z_center_spinbox.value()
        half = self.z_thickness_slider.value() / self.scaleFactor / 2
        hit = set()
        for x, y in data_path:
            for i in self.get_nearby_indices_xy(x, y, r):
                if (zc - half) <= self.points[i, 2] <= (zc + half):
                    hit.add(i)
        base = self.label_spinbox.value()
        for i in hit:
            self.streaks[i] = (base == 1)
        self.update_views()

    def _apply_brush_path_to_streaks_xz(self, data_path):
        """
        Apply brush stroke in XZ view to set boolean streak flags based on current spinbox (0/1).
        """
        r = self.radius_spinbox.value()
        data_path = self._interpolate_path(data_path, r)
        fc = self.finit_center_spinbox.value()
        half = self.finit_thickness_slider.value() / self.scaleFactor / 2
        shear = np.tan(np.radians(self.xz_shear_spinbox.value()))
        slab_mask = (self.points[:, 1] >= (fc - half)) & (self.points[:, 1] <= (fc + half))
        slab_idx = np.nonzero(slab_mask)[0]
        if slab_idx.size == 0:
            return
        if self.show_original_points:
            pts = self.original_points[slab_idx]
        else:
            pts = self.points[slab_idx]
        coords = np.vstack([pts[:, 0] + shear * (pts[:, 1] - fc), pts[:, 2]]).T
        tree = cKDTree(coords)
        hit = set()
        for xf, z in data_path:
            rel = tree.query_ball_point([xf, z], r)
            for j in rel:
                hit.add(slab_idx[j])
        base = self.label_spinbox.value()
        for i in hit:
            self.streaks[i] = (base == 1)
        self.update_views()

    def _update_views_streak(self):
        """
        Repaint views showing only streak flags (black for False, label-1 color for True).
        """
        # XY view
        z_center = self.z_center_spinbox.value()
        z_thick = self.z_thickness_slider.value() / self.scaleFactor
        z_min = z_center - z_thick / 2
        z_max = z_center + z_thick / 2
        mask_xy = (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max)
        pts_xy = self.original_points[mask_xy] if self.show_original_points else self.points[mask_xy]
        streaks_xy = self.streaks[mask_xy]
        # wrap top/bottom
        mask_top = pts_xy[:, 1] < -90
        pts_top = pts_xy[mask_top].copy(); pts_top[:, 1] += 360
        s_top = streaks_xy[mask_top]
        mask_bot = pts_xy[:, 1] > 90
        pts_bot = pts_xy[mask_bot].copy(); pts_bot[:, 1] -= 360
        s_bot = streaks_xy[mask_bot]
        pts_all = np.concatenate([pts_xy, pts_top, pts_bot], axis=0)
        streaks_all = np.concatenate([streaks_xy, s_top, s_bot], axis=0)
        # shear transforms
        vert = np.tan(np.radians(self.xy_vertical_shear_spinbox.value()))
        horz = np.tan(np.radians(self.xy_horizontal_shear_spinbox.value()))
        f_inits = pts_all[:, 1] + vert * (pts_all[:, 2] - z_center)
        f_stars = pts_all[:, 0] + horz * (pts_all[:, 2] - z_center)
        # downsample
        keep = self.downsample_points(pts_all, self.max_display, axis='xy')
        xs = f_stars[keep]; ys = f_inits[keep]
        sts = streaks_all[keep]
        # brushes
        brushes = [self.brush_black if not s else self.active_brushes[1] for s in sts]
        self.xy_scatter.setData(x=xs, y=ys, size=self.point_size, pen=None, brush=brushes)
        # hide calculated labels and overlays
        self.xy_calc_scatter.setData([], [], size=self.point_size, pen=None, brush=[])
        self._overlay.setVisible(False)
        # XZ view
        f_center = self.finit_center_spinbox.value()
        f_thick = self.finit_thickness_slider.value() / self.scaleFactor
        f_min = f_center - f_thick / 2
        f_max = f_center + f_thick / 2
        mask_xz = (self.points[:, 1] >= f_min) & (self.points[:, 1] <= f_max)
        pts_xz = self.original_points[mask_xz] if self.show_original_points else self.points[mask_xz]
        streaks_xz = self.streaks[mask_xz]
        shear = np.tan(np.radians(self.xz_shear_spinbox.value()))
        disp = pts_xz.copy(); disp[:, 0] = pts_xz[:, 0] + shear * (pts_xz[:, 1] - f_center)
        keep_xz = self.downsample_points(disp, self.max_display, axis='xz')
        xs_xz = disp[:, 0][keep_xz]; ys_xz = disp[:, 2][keep_xz]
        sts_xz = streaks_xz[keep_xz]
        brushes_xz = [self.brush_black if not s else self.active_brushes[1] for s in sts_xz]
        self.xz_scatter.setData(x=xs_xz, y=ys_xz, size=self.point_size, pen=None, brush=brushes_xz)
        self.xz_calc_scatter.setData([], [], size=self.point_size, pen=None, brush=[])
        self._overlay_xz.setVisible(False)
        # update guides/groups
        self.update_guides()
        self.update_groups_data()
        self.recompute = False

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
                if self.hide_teflon_checkbox.isChecked():
                    mask = mask & np.logical_or(self.labels[indices] != self.teflon_label, self.group[indices] != 0)
                indices = indices[mask]

            elif view_name == 'xz':
                # For the xz view, use the KDTree to find nearby points.
                indices = np.asarray(self.kdtree_xz.query_ball_point([x, y], r=r))
                finit_center = self.finit_center_spinbox.value()
                finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
                finit_min_val = finit_center - finit_thickness / 2
                finit_max_val = finit_center + finit_thickness / 2
                mask = (self.points[indices, 1] >= finit_min_val) & (self.points[indices, 1] <= finit_max_val)
                if self.hide_teflon_checkbox.isChecked():
                    mask = mask & np.logical_or(self.labels[indices] != self.teflon_label, self.group[indices] != 0)
                indices = indices[mask]
                update_label = current_label
            else:
                indices = np.array([])

            # Update the labels for the points that were found.
            if indices.size:
                self.labels[indices] = update_label
                self.group[indices] = self.active_group

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
                        self.group[i] = self.active_group

        # Save the current mouse position and time for the next event.
        self.last_mouse_point = (x_m, y_m)
        self.last_mouse_time = current_time

        end_time = time.time()  # Finished updating labels.
        self.update_views()
        end_time2 = time.time()  # Finished updating views.
        # print(f"Time to update calc labels: {end_time - start_time:.4f} s, Time to update views: {end_time2 - end_time:.4f} s")

    def update_views(self, val=None):
        if not hasattr(self, 'xy_scatter'):
            return
        # In streak mode, only display boolean streaks
        if self.streak_mode:
            self._update_views_streak()
            return
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
        if self.show_original_points:
            pts_xy = self.original_points[mask_xy]
        else:
            pts_xy = self.points[mask_xy]
        original_pts_xy = self.original_points[mask_xy]
        labels_xy = self.labels[mask_xy]
        group_xy = self.group[mask_xy]
        calc_labels_xy = self.calculated_labels[mask_xy]
        t2 = time.time()
        # print("Step 2 (XY: mask & extract pts/labels):", t2 - t1, "s")
        
        # ----- Adjust wrapping: top -----
        mask_top = pts_xy[:, 1] < -90
        pts_top = pts_xy[mask_top].copy()
        if pts_top.size:
            pts_top[:, 1] += 360
        original_pts_top = original_pts_xy[mask_top].copy()
        if original_pts_top.size:
            original_pts_top[:, 1] += 360
        labels_top = labels_xy[mask_top] - 1
        group_top = group_xy[mask_top]
        calc_labels_top = calc_labels_xy[mask_top] - 1
        t3 = time.time()
        # print("Step 3 (XY: top wrap adjustment):", t3 - t2, "s")
        
        # ----- Adjust wrapping: bottom -----
        mask_bottom = pts_xy[:, 1] > 90
        pts_bottom = pts_xy[mask_bottom].copy()
        if pts_bottom.size:
            pts_bottom[:, 1] -= 360
        original_pts_bottom = original_pts_xy[mask_bottom].copy()
        if original_pts_bottom.size:
            original_pts_bottom[:, 1] -= 360
        labels_bottom = labels_xy[mask_bottom] + 1
        group_bottom = group_xy[mask_bottom]
        calc_labels_bottom = calc_labels_xy[mask_bottom] + 1
        t4 = time.time()
        # print("Step 4 (XY: bottom wrap adjustment):", t4 - t3, "s")
        
        # ----- Combine and downsample XY arrays -----
        pts_combined = np.concatenate([pts_xy, pts_top, pts_bottom], axis=0)
        original_pts_combined = np.concatenate([original_pts_xy, original_pts_top, original_pts_bottom], axis=0)
        labels_combined = np.concatenate([labels_xy, labels_top, labels_bottom], axis=0)
        group_combined = np.concatenate([group_xy, group_top, group_bottom], axis=0)
        calc_labels_combined = np.concatenate([calc_labels_xy, calc_labels_top, calc_labels_bottom], axis=0)
        keep_indices_xy = self.downsample_points(pts_combined, self.max_display, axis='xy')
        # Downsample all data
        pts_combined, original_pts_combined, labels_combined, group_combined, calc_labels_combined = pts_combined[keep_indices_xy], original_pts_combined[keep_indices_xy], labels_combined[keep_indices_xy], group_combined[keep_indices_xy], calc_labels_combined[keep_indices_xy]
        if self.gt_labels is not None:
            gt_labels_xy = self.gt_labels[mask_xy]
            gt_labels_top = gt_labels_xy[mask_top] - 1
            gt_labels_bottom = gt_labels_xy[mask_bottom] + 1
            gt_labels_combined = np.concatenate([gt_labels_xy, gt_labels_top, gt_labels_bottom], axis=0)
            gt_labels_combined = gt_labels_combined[keep_indices_xy]
        t5 = time.time()
        # print("Step 5 (XY: combine & downsample):", t5 - t4, "s")
        
        # ----- Compute new brushes for XY view -----
        new_brushes_xy = self.get_brushes_from_labels(labels_combined, group_combined, original_pts_combined)
        if self.gt_labels is not None:
            new_brushes_xy = self.update_brushes_gt(new_brushes_xy, gt_labels_combined)
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
                new_brushes_calc_xy[i] = self.calc_brown_blue
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
            # print(f"XY Calc: Brushes size: {new_brushes_calc_xy.size}, changed mask size: {changed_mask_calc.size}, nr changed: {np.sum(changed_mask_calc)}")
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
        if self.show_original_points:
            pts_xz = self.original_points[mask_xz]
        else:
            pts_xz = self.points[mask_xz]
        original_pts_xz = self.original_points[mask_xz]
        labels_xz = self.labels[mask_xz]
        group_xz = self.group[mask_xz]
        shear_angle_deg_xz = self.xz_shear_spinbox.value()
        xz_shear_factor = np.tan(np.radians(shear_angle_deg_xz))
        if pts_xz.size:
            new_x_xz = pts_xz[:, 0] + xz_shear_factor * (pts_xz[:, 1] - finit_center)
        else:
            new_x_xz = pts_xz[:, 0]
        pts_xz_display = pts_xz.copy()
        pts_xz_display[:, 0] = new_x_xz
        keep_indices_xz = self.downsample_points(pts_xz_display, max_display=self.max_display, axis='xz')
        pts_xz_display, original_pts_xz, labels_xz, group_xz = pts_xz_display[keep_indices_xz], original_pts_xz[keep_indices_xz], labels_xz[keep_indices_xz], group_xz[keep_indices_xz] # Downsample all data
        if self.gt_labels is not None:
            gt_labels_xz = self.gt_labels[mask_xz]
            gt_labels_xz = gt_labels_xz[keep_indices_xz]
        new_xz_geometry = {'x': pts_xz_display[:, 0], 'y': pts_xz_display[:, 2]}
        new_brushes_xz = self.get_brushes_from_labels(labels_xz, group_xz, original_pts_xz)
        if self.gt_labels is not None:
            new_brushes_xz = self.update_brushes_gt(new_brushes_xz, gt_labels_xz)
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
                new_brushes_calc_xz[i] = self.calc_brown_blue
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
        self.update_groups_data()
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
        mask = np.logical_and(mask, self.group == 0) # only consider group 0 points
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
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []
        
        thresh = self.line_distance_threshold_spinbox.value()
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        mask = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        mask = np.logical_and(mask, self.group == 0) # only consider group 0 points
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
                self.group[global_idx] = 0 # Use GT group when assigning line labels
                assign_count += 1
            if j % update_interval == 0:
                progress.setValue(j)
            if progress.wasCanceled():
                break
        progress.close()
        self.update_views()
        print(f"Assigned line labels to {assign_count} points (threshold: {thresh}).")

    def assign_line_labels_all(self):
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []

        # Get current parameters.
        thresh = self.line_distance_threshold_spinbox.value()
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        assign_min = self.assign_min_spinbox.value()
        assign_max = self.assign_max_spinbox.value()

        # Get the points in the current z-slab.
        mask = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        mask = np.logical_and(mask, self.group == 0) # only consider group 0 points
        pts = self.points[mask]
        global_indices = np.where(mask)[0]
        if pts.shape[0] == 0:
            return

        # Only work on candidate points that are not already "labeled" (UNLABELED).
        cand_mask = np.abs(self.labels[mask] - self.UNLABELED) < 2
        if not np.any(cand_mask):
            return
        cand_globals = global_indices[cand_mask]
        # For distance computations we only need the x,y coordinates.
        candidate_pts = pts[cand_mask, :2]  # shape (m, 2)
        m = candidate_pts.shape[0]

        # Get sorted list of winding labels (ul) available in spline_segments in the allowed range.
        labels_to_check = sorted([ul for ul in self.spline_segments if assign_min <= ul <= assign_max])
        L = len(labels_to_check)
        if L == 0:
            return

        # --- Helper: Compute distance from many points to a set of line segments.
        # Each polyline is broken into segments (p0 -> p1) and we compute the minimum distance.
        def point_to_segments_distance(points, segments):
            # points: (m,2); segments: (n,2,2)
            p0 = segments[:, 0, :]  # shape (n,2)
            p1 = segments[:, 1, :]  # shape (n,2)
            v = p1 - p0             # shape (n,2)
            # Compute vector from p0 to every point: shape (m, n, 2)
            diff = points[:, np.newaxis, :] - p0[np.newaxis, :, :]
            # Compute projection factor t along each segment.
            dot_val = np.sum(diff * v, axis=2)  # shape (m, n)
            norm_v2 = np.sum(v * v, axis=1)       # shape (n,)
            t = dot_val / (norm_v2[np.newaxis, :] + 1e-8)
            t = np.clip(t, 0, 1)
            proj = p0[np.newaxis, :, :] + t[..., np.newaxis] * v[np.newaxis, :, :]
            dist = np.linalg.norm(points[:, np.newaxis, :] - proj, axis=2)  # shape (m, n)
            return np.min(dist, axis=1)  # shape (m,)

        progress = QProgressDialog("Assigning all line labels...", "Cancel", 0, L, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        update_interval = max(L // 100, 1)
        
        # For each label in labels_to_check, compute the minimum distance from every candidate point.
        # We assume each spline_segments[ul] is a list of polyline arrays.
        D = np.empty((L, m))
        for i, ul in enumerate(labels_to_check):
            if i % update_interval == 0:
                progress.setValue(i)
            if progress.wasCanceled():
                return
            d_vals = np.full(m, np.inf)
            for polyline in self.spline_segments[ul]:
                # Skip segments that are too short.
                if polyline.shape[0] < 2:
                    continue
                # Build an array of segments from the polyline.
                segs = np.stack([polyline[:-1], polyline[1:]], axis=1)  # shape (n_seg, 2, 2)
                d_seg = point_to_segments_distance(candidate_pts, segs)
                d_vals = np.minimum(d_vals, d_seg)
            D[i, :] = d_vals

        # Now determine for each candidate point whether the distance for a given label is below threshold
        # and lower than the distances for its immediate neighbors.
        cond = (D < thresh)
        for k in range(L):
            if k > 0:
                cond[k, :] &= (D[k, :] < D[k - 1, :])
            if k < L - 1:
                cond[k, :] &= (D[k, :] < D[k + 1, :])

        # For each candidate point, choose the first (lowest-index in sorted order) label that satisfies the condition.
        assigned_idx = np.full(m, -1, dtype=int)
        for k in range(L):
            to_assign = (assigned_idx == -1) & cond[k, :]
            assigned_idx[to_assign] = k

        # Update the labels (and groups) for candidate points where an assignment was found.
        assigned = assigned_idx != -1
        for idx, k in zip(cand_globals[assigned], assigned_idx[assigned]):
            # Get the label to assign.
            self.labels[idx] = labels_to_check[k]
            self.group[idx] = 0  # GT group
            assert self.points[idx, 2] >= z_min_val and self.points[idx, 2] <= z_max_val, f"Point {idx} out of range: {self.points[idx, 2]} not in [{z_min_val}, {z_max_val}]"

        assign_count = np.sum(assigned)
        progress.close()
        self.recompute = True
        self.update_views()
        print(f"Assigned all line labels to {assign_count} of {m} points (threshold: {thresh}).")

    def delete_range(self):
        self._split_range(make_group=False)

    def split_groups_range(self):
        self._split_range(make_group=True)

    def _split_range(self, make_group):
        # delete points within the specified range
        z_center = self.z_selection_spinbox.value()
        z_thickness = self.z_selection_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        mask = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        f_init_center = self.finit_selection_spinbox.value()
        f_init_thickness = self.f_selection_thickness_slider.value() / self.scaleFactor
        f_init_min_val = f_init_center - f_init_thickness
        f_init_max_val = f_init_center + f_init_thickness
        mask = np.logical_and(mask, np.logical_and(self.points[:, 1] >= f_init_min_val, self.points[:, 1] <= f_init_max_val))
        if self.outside_checkbox.isChecked():
            # Delete points outside the range
            mask = np.logical_not(mask)
        range_min = self.assign_min_spinbox.value()
        range_max = self.assign_max_spinbox.value()
        mask = np.logical_and(mask, np.logical_and(self.labels >= range_min, self.labels <= range_max))
        mask = np.logical_and(mask, self.group == self.active_group) # Only delete points in the active group

        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []

        if not make_group:
            self.labels[mask] = self.UNLABELED
            self.group[mask] = 0
        else:
            new_group = self.group.max() + 1
            self.group[mask] = new_group
            # Activate the new group
            self.group_spinbox.setValue(new_group)

        self.recompute = True
        self.update_views()
    
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

    def pick_group_at(self, ev, plot_widget):
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
        disp_group = np.array([self.group[i] for i in indices])

        disp_group = disp_group[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED + 1) &
                                  (disp_labels != self.UNLABELED - 1)]
        disp_labels = disp_labels[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED + 1) &
                                  (disp_labels != self.UNLABELED - 1)]
        
        if disp_group.size == 0:
            return
        vals, counts = np.unique(disp_group, return_counts=True)
        mode_group = int(vals[np.argmax(counts)])
        vals, counts = np.unique(disp_labels, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])

        self.group_spinbox.setValue(mode_group)
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

    def pick_group_at_xz(self, ev, plot_widget):
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
        disp_group = self.group[indices]
        disp_group = disp_group[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED + 1) &
                                  (disp_labels != self.UNLABELED - 1)]
        disp_labels = disp_labels[(disp_labels != self.UNLABELED) &
                                    (disp_labels != self.UNLABELED + 1) &
                                    (disp_labels != self.UNLABELED - 1)]
        if disp_group.size == 0:
            return
        vals, counts = np.unique(disp_group, return_counts=True)
        mode_group = int(vals[np.argmax(counts)])
        vals, counts = np.unique(disp_labels, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])
        self.group_spinbox.setValue(mode_group)
        self.label_spinbox.setValue(mode_label)
    
    def activate_pipette(self):
        self.pipette_mode = True

    def activate_group_pipette(self):
        self.group_pipette_mode = True

    def reset_points(self):
        self.points = self.original_points.copy()
        self.solver.set_positions(list(20*self.points[:,0]))
        self.recompute = True
        self.kdtree_xy = cKDTree(self.points[:, [0, 1]])
        self.kdtree_xz = cKDTree(self.points[:, [0, 2]])
        self.update_slider_ranges()
        self.update_views()

    def run_slab_computation(self):
        """
        Computes labels over slabs in z. It starts at the current slab (using the current z_center
        and thickness) and saves the result. If the current thickness is less than 200, it resets the
        thickness to 200 and recomputes. Then it moves upward (and then downward) in 100-unit steps
        for the z_center, computing and saving labels at each step. At the end, a popup displays the total
        computation time.

        For each slab, the procedure is as follows:
        1. Set the line thickness assignment to 3.
        2. For two iterations:
            a. Set the solver mode to "F*3" and call update_labels().
            b. Clear splines and calculated labels.
            c. Update winding splines.
            d. Switch the solver mode to "Winding Number" and call update_labels() to compute the winding number.
            e. Restore the previous solver mode.
            f. Call assign_line_labels().
            g. Clear splines and calculated labels.
        3. Ensure the z-range checkbox is enabled during the computation.
        4. Update views and save the labels for each slab.
        """
        # Pop-Up "Are you sure?" dialog
        reply = QMessageBox.question(self, "Slab Computation",
                                    "This will take a while. Are you sure you want to continue?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        # Reset the stop flag at the beginning of a new computation
        self.stop_slab_computation = False
        start_time = time.time()
        initial_z_center = self.z_center_spinbox.value()
        initial_thickness = self.z_thickness_spinbox.value() / self.scaleFactor
        desired_thickness = float(self.slab_thickness_spinbox.value())
        step = desired_thickness / 2

        def compute_current_slab(update_splines=False, extra_z_range=None, seed_node=None):
            # Check if the computation should be stopped
            if not self.stop_slab_computation:
                # Set the line thickness assignment to 3.
                # self.line_distance_threshold_spinbox.setValue(3)
                # Ensure z-range is enabled.
                original_z_range = self.use_z_range_checkbox.isChecked()
                self.use_z_range_checkbox.setChecked(True)
                original_disregard_label0 = self.disregard_label0_checkbox.isChecked()
                self.disregard_label0_checkbox.setChecked(False)
                for _ in range(2):
                    if seed_node is not None:
                        self.solver.set_f_star(seed_node)
                    # --- Run F*5 update via update_labels ---
                    if hasattr(self, "solver_combo"):
                        prev_solver = self.solver_combo.currentText()
                        self.solver_combo.setCurrentText("F*Slab")
                    self.update_labels(extra_z_range=extra_z_range)
                    # Use same seed node for a continuous graph 

                    # Check if the computation should be stopped
                    if self.stop_slab_computation:
                        break

                    # # --- Run F*2 update via update_labels ---
                    # if hasattr(self, "solver_combo"):
                    #     prev_solver = self.solver_combo.currentText()
                    #     self.solver_combo.setCurrentText("F*2")
                    # self.update_labels()
                    # # --- Run winding number update via update_labels ---
                    # if hasattr(self, "solver_combo"):
                    #     self.solver_combo.setCurrentText("Winding Number")
                    # self.update_labels()
                    # Restore the previous solver mode.
                    if hasattr(self, "solver_combo"):
                        self.solver_combo.setCurrentText(prev_solver)
                    if update_splines:
                        # Clear splines and calculated labels.
                        self.clear_splines()
                        self.clear_calculated_labels()
                        # Update winding splines.
                        self.update_winding_splines()
                        self.update_views()
                    # Assign line labels.
                    self.assign_line_labels()
                    # self.assign_line_labels_all()
            # Restore the original z-range checkbox state.
            self.use_z_range_checkbox.setChecked(original_z_range)
            self.disregard_label0_checkbox.setChecked(original_disregard_label0)
            # Update views.
            self.recompute = True
            self.update_views()

        # Compute for the initial slab.
        if not self.skip_initial_checkbox.isChecked():
            compute_current_slab(update_splines=True)
        else:
            print("Skipping initial slab computation.")
            # --- Run F*5 update via update_labels ---
            if hasattr(self, "solver_combo"):
                prev_solver = self.solver_combo.currentText()
                self.solver_combo.setCurrentText("Set Labels")
                self.update_labels()
                self.solver.set_f_star(self.find_seed_node()) # make labeled nodes straight
                # Update positions
                self.update_positions(update_slide_ranges=False)
                # Restore the previous solver mode.
                self.solver_combo.setCurrentText(prev_solver)
                # Clear splines and calculated labels.
                self.clear_splines()
                self.clear_calculated_labels()
                # Update winding splines.
                self.update_winding_splines()
                # Update views.
                self.recompute = True
                self.update_views()
        
        base_path = os.path.join("../experiments", self.default_experiment)
        slab_filename = os.path.join(base_path, "slabs", "slab_initial.txt")
        #make dir
        os.makedirs(os.path.dirname(slab_filename), exist_ok=True)
        self._save_labels_to_path(slab_filename)

        # get seed node
        seed_node = self.find_seed_node(extra_z_range=(initial_z_center - initial_thickness / 2, initial_z_center + initial_thickness / 2))

        # If the current slab thickness is less than 200, set thickness to 200 and recompute.
        self.z_thickness_spinbox.setValue(desired_thickness)
        self.z_thickness_slider.setValue(int(desired_thickness * self.scaleFactor))
        if initial_thickness < desired_thickness/self.scaleFactor:
            self.recompute = True
            self.update_views()
            compute_current_slab(seed_node=seed_node)
            slab_filename = os.path.join(base_path, "slabs", "slab_thickness_initial.txt")
            self._save_labels_to_path(slab_filename)

        # adjust for initial computation steps
        done_steps = max(0, np.floor(initial_thickness * self.scaleFactor / (2 * step)) - 1)
        # Iterate upward from the original z_center.
        current_z_center = initial_z_center + done_steps * step
        z_max = min(self.z_max, float(self.slabs_max_spinbox.value()))
        z_min = max(self.z_min, float(self.slabs_min_spinbox.value()))
        while current_z_center + desired_thickness / 2 < z_max:
            if self.stop_slab_computation:
                print("Slab computation stopped during upward iteration.")
                break
            extra_z_range = (current_z_center - desired_thickness / 2, current_z_center + desired_thickness / 2)
            current_z_center += step
            self.z_center_spinbox.setValue(current_z_center)
            self.z_center_slider.setValue(int(current_z_center * self.scaleFactor))
            self.recompute = True
            self.update_views()
            compute_current_slab(extra_z_range=extra_z_range, seed_node=seed_node)
            slab_filename = os.path.join(base_path, "slabs", f"slab_up_{int(current_z_center)}.txt")
            self._save_labels_to_path(slab_filename)

        if self.stop_slab_computation:
            QMessageBox.information(self, "Slab Computation", "Slab computation was stopped by the user.")
            return

        # Iterate downward from the original z_center.
        current_z_center = initial_z_center - done_steps * step
        while current_z_center - desired_thickness / 2 > z_min:
            if self.stop_slab_computation:
                print("Slab computation stopped during downward iteration.")
                break
            extra_z_range = (current_z_center - desired_thickness / 2, current_z_center + desired_thickness / 2)
            current_z_center -= step
            self.z_center_spinbox.setValue(current_z_center)
            self.z_center_slider.setValue(int(current_z_center * self.scaleFactor))
            self.recompute = True
            self.update_views()
            compute_current_slab(extra_z_range=extra_z_range, seed_node=seed_node)
            slab_filename = os.path.join(base_path, "slabs", f"slab_down_{int(current_z_center)}.txt")
            self._save_labels_to_path(slab_filename)

        if self.stop_slab_computation:
            QMessageBox.information(self, "Slab Computation", "Slab computation was stopped by the user.")
            return

        total_time = time.time() - start_time
        QMessageBox.information(self, "Slab Computation",
                                f"Slab computation completed in {total_time:.2f} seconds.")
        
    def set_stop_slab_flag(self):
        self.stop_slab_computation = True
        print("Stop slab computation flag set.")
        
    def set_labels(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            if self.hide_teflon_checkbox.isChecked():
                gt_0 = np.logical_and(gt, np.abs(self.labels - self.teflon_label) > 2)
            else:
                gt_0 = gt
            gt_0 = np.logical_and(gt_0, self.group == 0) # only consider group 0 points
            self.solver.set_labels(list(self.labels), list(gt_0))
        return gt
    
    def find_seed_node(self, deleted_mask_previous=None, extra_z_range=None):
        if deleted_mask_previous is None:
            undeleted = self.solver.get_undeleted_indices()
            deleted_mask_previous = self.solver.get_deleted_mask(undeleted)
        assert len(deleted_mask_previous) == len(self.labels), "Deleted mask shape mismatch"
        mask_valid = np.abs(self.labels - self.UNLABELED) > 2
        mask_group_0 = self.group == 0
        mask_non_teflon = np.abs(self.labels - self.teflon_label) > 2
        mask_non_deleted = np.logical_not(deleted_mask_previous)
        mask_not_first_index = np.arange(len(self.labels)) != 0
        
        mask = np.logical_and(mask_valid, mask_group_0)
        mask = np.logical_and(mask, mask_non_teflon)
        mask = np.logical_and(mask, mask_non_deleted)
        mask = np.logical_and(mask, mask_not_first_index)
        if extra_z_range is not None:
            print(f"Using extra z-range: {extra_z_range}")
            mask_z_range = np.logical_and(self.points[:, 2] >= extra_z_range[0], self.points[:, 2] <= extra_z_range[1])
            mask = np.logical_and(mask, mask_z_range)
        first_valid = np.where(mask)[0]
        if first_valid.size == 0:
            print("No seed node found.")
            seed_node = None
        else:
            offset = np.logical_not(deleted_mask_previous[:first_valid[0]]).sum()
            seed_node = offset
            assert mask[first_valid[0]], f"Seed node {seed_node} at {first_valid[0]} is not valid"
            first_valid = first_valid[0]
        print(f"Seed node found at index {seed_node} at {first_valid}. With label {self.labels[first_valid]}")
        return seed_node
    
    # Example of using the solver interface when updating labels:
    def update_labels(self, extra_z_range=None):
        if self.calculating:
            print("Already calculating, skipping update_labels")
            return
        self.calculating = True
        undeleted = self.solver.get_undeleted_indices()
        other_block_factor=float(self.solve_other_block_factor_spinbox.value())
        if self.solver is not None:
            # Deactivate edges within same streak blocks before updating labels
            try:
                self.solver.deactivate_same_block_edges([bool(s) for s in self.streaks])
            except Exception:
                # Solver may not support streak deactivation; ignore if unavailable
                pass
            self.solver.remove_temporary_edges()
            gt = self.set_labels()

            groups = np.unique(self.group)
            group_metadata= []
            for group in groups:
                if group == 0:
                    continue
                gt_g = np.logical_and(gt, self.group == group)
                print(f"Sum of gt_g: {np.sum(gt_g)} in group {group}")
                self.solver.fix_good_edges(list(self.labels), list(gt_g))
                source, target, winding_number_difference = build_temporary_group_edges(self.solver, list(self.labels), list(gt_g))
                group_metadata.append((source, target, winding_number_difference))

            if self.hide_teflon_checkbox.isChecked():
                print("Hiding teflon during solver update")
                self.solver.delete_nodes(list(np.array(undeleted)[np.abs(self.labels - self.teflon_label) < 2]))
            if self.use_z_range_checkbox.isChecked():
                print("Using z-range")
                z_center = self.z_center_spinbox.value()
                z_thickness = self.z_thickness_slider.value() / self.scaleFactor
                z_min_val = 4 * (z_center - z_thickness / 2) - 500.0
                z_max_val = 4 * (z_center + z_thickness / 2) - 500.0
                self.solver.set_z_range(z_min_val, z_max_val)
                self.seed_node = None
            if self.use_fstar_range_checkbox.isChecked():
                print("Using f*-range")
                self.solver.set_f_star_range(f_star_min=float(20*self.fstar_min_spinbox.value()), f_star_max=float(20*self.fstar_max_spinbox.value()))
                self.seed_node = None
            selected_solver = self.solver_combo.currentText() if hasattr(self, "solver_combo") else "F*"
            deleted_mask_previous = self.solver.get_deleted_mask(undeleted)
            if selected_solver == "Winding Number":
                self.solver.solve_winding_number(num_iterations=500, i_round=-3, seed_node=-1,
                                                 other_block_factor=15.0, side_fix_nr=-1, display=False)
            elif "F*" in selected_solver or selected_solver in ["Linear", "Ripple", "Smooth", "Ripple Smooth Combined", "Tugging", "Create Good Edges"]:
                if self.seed_node is None:
                    self.seed_node = self.find_seed_node(deleted_mask_previous=deleted_mask_previous, extra_z_range=extra_z_range)
                if self.seed_node is not None:
                    # self.solver.set_labeled_edges(self.seed_node)
                    self.solver.set_f_star(self.seed_node)
                if selected_solver == "F*":
                    self.solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=1.1, other_block_factor=other_block_factor, lr=0.05, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=1.0, other_block_factor=other_block_factor, lr=0.05, error_cutoff=-1.0, display=True)
                elif selected_solver == "Linear":
                    self.solver.solve_f_star_with_labels(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, other_block_factor=other_block_factor, lr=0.05, error_cutoff=-1.0, display=True)
                elif selected_solver == "F*3":
                    self.solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=4.0, other_block_factor=0.5, lr=0.25, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=2.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                    self.solver.solve_f_star_with_labels(num_iterations=30000, spring_constant=1.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=True)
                elif selected_solver == "F*4":
                    self.solver.solve_f_star(num_iterations=int(2 * self.solve_iterations_spinbox.value() / 4), spring_constant=1.0, o=0.0, step_sigma=36000000.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True)
                    self.solver.solve_f_star(num_iterations=int(2 * self.solve_iterations_spinbox.value() / 4), spring_constant=1.0, o=0.0, step_sigma=360.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True)
                elif selected_solver == "Ripple":
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, o=0.0, step_sigma=36000000.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True)
                elif selected_solver == "Smooth":
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, o=0.0, step_sigma=360.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True)
                elif selected_solver == "Ripple Smooth Combined":
                    self.solver.solve_f_star(num_iterations=int(3 * self.solve_iterations_spinbox.value() / 4), spring_constant=1.0, o=0.0, step_sigma=36000000.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True)
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value() / 4), spring_constant=1.0, o=0.0, step_sigma=360.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True)
                elif selected_solver == "F*Slab":
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, o=0.0, step_sigma=36000000.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True, adjust_median=False, blow_away=True)
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value()//2), spring_constant=1.0, o=0.0, step_sigma=36000000.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True, adjust_median=False, blow_away=False)
                    self.solver.solve_f_star(num_iterations=int(self.solve_iterations_spinbox.value()//2), spring_constant=1.0, o=0.0, step_sigma=360.0, teflon_winding_nr=self.teflon_label, i_round=6, visualize=True, adjust_median=False)
                    self.solver.solve_f_star_with_labels(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, other_block_factor=other_block_factor, lr=0.25, error_cutoff=-1.0, display=True)
                    # self.solver.solve_f_star_with_labels(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, other_block_factor=other_block_factor, lr=0.05, error_cutoff=-1.0, display=True)
                    self.solver.solve_winding_number(num_iterations=500, i_round=-3, seed_node=-1, other_block_factor=15.0, side_fix_nr=-1, display=False)
                elif selected_solver == "Create Good Edges":
                    self.solver.label_good_neighbors(r=self.solve_other_block_r_spinbox.value(), delta_neg=180.0, delta_top=10.0, delta_perfect=5.0, min_neighbors=5)
                elif selected_solver == "Tugging":
                    self.solver.solve_tugging(num_iterations=int(self.solve_iterations_spinbox.value()), spring_constant=1.0, step_sigma=520.0, o=0.0, i_round=2, visualize=True, distribute=0.3, diff_step=0.00000550)

                if self.use_z_range_checkbox.isChecked() or self.use_fstar_range_checkbox.isChecked() or self.hide_teflon_checkbox.isChecked():
                    print(f"Resetting z-range, length: {len(undeleted)}")
                    self.solver.set_undeleted_indices(undeleted)
                    self.seed_node = None
                if selected_solver == "F*Slab":
                    calculated_labels = self.solver.get_labels()
                    self.calculated_labels = np.array(calculated_labels)
                    # print("Reset the z undeleted indices")
                # Update positions
                self.update_positions(update_slide_ranges=False)

                # # compare position to group metadata, DEBUG
                # for source, target, winding_number_difference in group_metadata:
                #     for i, (s, t, wnr) in enumerate(zip(source, target, winding_number_difference)):
                #         wnr_d = (self.points[t, 0] * 20 - self.points[s, 0] * 20) - wnr * 360
                #         print(f"Group {self.group[t]}: {wnr_d:.2f}° difference between {s} and {t}")

                self.calculating = False
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
            
            self.solver.set_undeleted_indices(undeleted)
            if self.use_z_range_checkbox.isChecked() or self.use_fstar_range_checkbox.isChecked():
                print("Resetting z-range")
                self.seed_node = None
            calculated_labels = self.solver.get_labels()
            self.calculated_labels = np.array(calculated_labels)

            self.update_views()
            self.calculating = False
    
    def save_graph(self):
        if self.solver is not None:
            gt_0 = np.abs(self.labels - self.UNLABELED) > 2 # remove unlabeled
            gt_1 = np.logical_and(gt_0, np.abs(self.labels - self.teflon_label) > 2) # remove teflon
            self.solver.set_labels(self.labels, gt_1)
            # delete unasigned points
            label_indices = np.array(self.solver.get_undeleted_indices())
            labeled_indices = list(label_indices[gt_1])
            self.solver.set_undeleted_indices(labeled_indices)
            print(f"Deleted Unlabeled Points: {len(label_indices) - len(labeled_indices)} of {len(label_indices)}. Total {gt_1.shape[0] - np.sum(gt_1)} deleted points with {np.sum(gt_0)} unlabeled points and {np.sum(np.abs(self.labels - self.teflon_label) > 2)} teflon points. Saving graph...")
            # Save graph as output_graph.bin for meshing
            graph_solved_path = self.graph_path.replace("graph.bin", "output_graph.bin")
            self.solver.save_solution(graph_solved_path)
    
    def undo(self):
        if self.undo_stack:
            prev_state, prev_group = self.undo_stack.pop()
            self.redo_stack.append((self.labels.copy(), self.group.copy()))
            self.labels = prev_state.copy()
            self.group = prev_group.copy()
            self.update_views()
    
    def redo(self):
        if self.redo_stack:
            next_state, next_group = self.redo_stack.pop()
            self.undo_stack.append((self.labels.copy(), self.group.copy()))
            self.labels = next_state.copy()
            self.group = next_group.copy()
            self.update_views()

    def update_undo_redo(self):
        # Clear out undo stack if it is larger than 50
        max_len = 50
        if len(self.undo_stack) > max_len:
            self.undo_stack = self.undo_stack[-max_len:]

    def toggle_original_points(self):
        self.show_original_points = not self.show_original_points
        print(f"Toggling original points: {self.show_original_points}")
        self.recompute = True
        if self.show_original_points:
            self.computed_kdtree_xy = self.kdtree_xy
            self.computed_kdtree_xz = self.kdtree_xz
            self.kdtree_xy = self.original_kdtree_xy
            self.kdtree_xz = self.original_kdtree_xz
        else:
            self.kdtree_xy = self.computed_kdtree_xy
            self.kdtree_xz = self.computed_kdtree_xz
        self.update_views()
    
    def toggle_calc_draw_mode(self):
        self.calc_drawing_mode = self.calc_draw_button.isChecked()
        if self.calc_drawing_mode:
            self.calc_draw_button.setText("Update Labels Draw Mode: On")
        else:
            self.calc_draw_button.setText("Update Labels Draw Mode: Off")
    
    def toggle_streak_mode(self, checked):
        """
        Toggle streak drawing mode: when on, brush strokes will toggle the boolean streak
        flag on points instead of assigning numeric labels.  When enabled, set spinbox to 1.
        """
        self.streak_mode = checked
        # In streak mode, use label 1 to set streak=True; label 0 clears streak
        if checked:
            self.label_spinbox.setValue(1)
        # Update views immediately when toggling streak mode
        self.update_views()

    def apply_all_calculated_labels(self):
        mask = (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED)
        if np.any(mask):
            # Track undo/redo
            self.undo_stack.append((self.labels.copy(), self.group.copy()))
            self.redo_stack = []

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
            # Track undo/redo
            self.undo_stack.append((self.labels.copy(), self.group.copy()))
            self.redo_stack = []

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
            # Track undo/redo
            self.undo_stack.append((self.labels.copy(), self.group.copy()))
            self.redo_stack = []

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
        elif event.key() == Qt.Key_G:
            self.activate_group_pipette()
            event.accept()
        # space or O
        elif event.key() == Qt.Key_Space or event.key() == Qt.Key_O:
            self.toggle_original_points()
            event.accept()
        # up arrow
        elif event.key() == Qt.Key_Up:
            self.label_spinbox.setValue(self.label_spinbox.value() + 1)
            event.accept()
        # down arrow
        elif event.key() == Qt.Key_Down:
            self.label_spinbox.setValue(self.label_spinbox.value() - 1)
            event.accept()
        elif event.key() == Qt.Key_C:
            self.hide_labels = not self.hide_labels
            self.update_views()
            event.accept()
        elif event.key() == Qt.Key_L:
            self.hide_estimated_colors = not self.hide_estimated_colors
            self.update_views()
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

    def perform_autosave(self):
        """Perform autosave if conditions are met."""
        # Only autosave if we have a reference path (either from manual save or load)
        reference_path = self.last_manual_save_path or self.last_load_path
        if not reference_path:
            return
            
        # Check if labels have changed since last save
        if self._labels_have_changed():
            try:
                autosave_path = self.get_autosave_path(reference_path)
                if autosave_path:
                    # Ensure autosave directory exists
                    autosave_dir = os.path.dirname(autosave_path)
                    os.makedirs(autosave_dir, exist_ok=True)
                    self._save_labels_to_path(autosave_path)
                    # Update last saved state
                    self._update_last_saved_state()
                    print(f"Autosaved to {autosave_path}")
                    
                    # Clean up old autosave files, keeping only the 30 most recent
                    self._cleanup_old_autosaves(autosave_dir)
            except Exception as e:
                print(f"Autosave failed: {e}")

    def _cleanup_old_autosaves(self, autosave_dir, max_files=30):
        """Remove old autosave files, keeping only the most recent ones."""
        try:
            # Safety check: only clean up if we're actually in an autosave directory
            if not os.path.basename(autosave_dir) == "autosave":
                print(f"Safety check failed: {autosave_dir} is not an autosave directory")
                return
                
            # Additional safety check: ensure the directory exists and is readable
            if not os.path.exists(autosave_dir) or not os.path.isdir(autosave_dir):
                print(f"Autosave directory does not exist or is not a directory: {autosave_dir}")
                return
            
            # Find all autosave files in the directory
            autosave_files = []
            for filename in os.listdir(autosave_dir):
                # Additional safety: ensure file is actually an autosave file
                if filename.startswith("autosave_") and filename.endswith(".txt"):
                    file_path = os.path.join(autosave_dir, filename)
                    
                    # Safety check: ensure it's a file and not a directory
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Extract timestamp from filename to sort by age
                    match = re.match(r'autosave_(\d{8}_\d{6})_.*\.txt$', filename)
                    if match:
                        timestamp_str = match.group(1)
                        try:
                            # Parse timestamp
                            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            autosave_files.append((timestamp, file_path, filename))
                        except ValueError:
                            # If timestamp parsing fails, use file modification time
                            mtime = os.path.getmtime(file_path)
                            timestamp = datetime.fromtimestamp(mtime)
                            autosave_files.append((timestamp, file_path, filename))
            
            # Sort by timestamp (oldest first)
            autosave_files.sort(key=lambda x: x[0])
            
            # Remove old files if we have more than max_files
            if len(autosave_files) > max_files:
                files_to_remove = autosave_files[:-max_files]  # Keep the last max_files
                removed_count = 0
                for timestamp, file_path, filename in files_to_remove:
                    try:
                        # Final safety check: ensure the file is still in the autosave directory
                        if os.path.dirname(file_path) == autosave_dir and filename.startswith("autosave_"):
                            os.remove(file_path)
                            print(f"Removed old autosave: {filename}")
                            removed_count += 1
                        else:
                            print(f"Safety check failed for file: {filename}")
                    except OSError as e:
                        print(f"Failed to remove old autosave {filename}: {e}")
                
                if removed_count > 0:
                    print(f"Cleaned up {removed_count} old autosave files, keeping {max_files} most recent")
                
        except Exception as e:
            print(f"Failed to cleanup old autosaves: {e}")
    
    def _labels_have_changed(self):
        """Check if current labels differ from last saved state."""
        if self.last_saved_labels is None:
            # No previous save state, consider it changed if we have labels
            return True
            
        # Compare current state with last saved state
        labels_changed = not np.array_equal(self.labels, self.last_saved_labels)
        group_changed = not np.array_equal(self.group, self.last_saved_group)
        streaks_changed = not np.array_equal(self.streaks, self.last_saved_streaks)
        
        return labels_changed or group_changed or streaks_changed
    
    def _update_last_saved_state(self):
        """Update the last saved state to current state."""
        self.last_saved_labels = self.labels.copy()
        self.last_saved_group = self.group.copy()
        self.last_saved_streaks = self.streaks.copy()

    def get_autosave_path(self, reference_path):
        """Generate autosave path based on reference path."""
        if not reference_path:
            return None
            
        # Get directory and filename
        reference_dir = os.path.dirname(reference_path)
        reference_filename = os.path.basename(reference_path)
        
        # Check if the reference file is already an autosave file
        if "autosave_" in reference_filename and "autosave" in reference_dir:
            # If it's already an autosave file, use the same autosave directory
            autosave_dir = reference_dir
            # Extract the original filename (remove the autosave prefix and timestamp)
            if reference_filename.startswith("autosave_"):
                # Format is: autosave_YYYYMMDD_HHMMSS_original_filename.ext
                # Split and take everything after the timestamp parts
                parts = reference_filename.split("_", 3)  # Split into max 4 parts
                if len(parts) >= 4:
                    original_filename = parts[3]  # Everything after autosave_YYYYMMDD_HHMMSS_
                else:
                    # Fallback: try to find the original filename after any timestamp pattern
                    # Look for pattern autosave_########_######_filename
                    import re
                    match = re.match(r'autosave_\d{8}_\d{6}_(.+)', reference_filename)
                    if match:
                        original_filename = match.group(1)
                    else:
                        original_filename = reference_filename
            else:
                original_filename = reference_filename
        else:
            # Create autosave directory
            autosave_dir = os.path.join(reference_dir, "autosave")
            original_filename = reference_filename
            
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create autosave filename
        name_without_ext = os.path.splitext(original_filename)[0]
        ext = os.path.splitext(original_filename)[1]
        autosave_filename = f"autosave_{timestamp}_{name_without_ext}{ext}"
        
        return os.path.join(autosave_dir, autosave_filename)

    def on_ome_zarr_labels_updated(self, node_updates, view_type):
        """Handle label updates from OME-Zarr view."""
        print(f"Received label updates for {len(node_updates)} nodes from OME-Zarr view ({view_type})")
        
        # Debug: Check coordinate spaces
        print(f"Debug: len(self.points)={len(self.points)}, len(self.labels)={len(self.labels)}")
        if hasattr(self, 'solver'):
            undeleted = self.solver.get_undeleted_indices()
            print(f"Debug: len(undeleted)={len(undeleted)}")
        
        # Debug: Check node_updates indices range
        if node_updates:
            min_idx = min(node_updates.keys())
            max_idx = max(node_updates.keys())
            print(f"Debug: node_updates indices range: {min_idx} to {max_idx}")
        
        # Track undo/redo
        self.undo_stack.append((self.labels.copy(), self.group.copy()))
        self.redo_stack = []
        
        # Get the drawing radius
        radius = self.radius_spinbox.value()
        
        # Apply the label updates with radius expansion using view-specific logic
        updated_count = 0
        total_affected_nodes = 0
        
        for solver_node_idx, new_label in node_updates.items():
            # Direct indexing - no mapping needed since arrays are in undeleted space
            if not (0 <= solver_node_idx < len(self.labels)):
                print(f"Warning: node index {solver_node_idx} out of range")
                continue
                
            # Get the position of the updated node
            node_pos = self.points[solver_node_idx]
            
            # Use view-specific logic based on which view the updates came from
            if view_type == "XY":
                # Use XY view logic similar to _apply_brush_path_to_labels
                affected_indices = self._get_nearby_indices_xy_with_radius(node_pos, radius)
            elif view_type == "XZ":
                # Use XZ view logic similar to _apply_brush_path_to_labels_xz
                affected_indices = self._get_nearby_indices_xz_with_radius(node_pos, radius)
            else:
                print(f"Warning: unknown view type {view_type}, using XY logic as fallback")
                affected_indices = self._get_nearby_indices_xy_with_radius(node_pos, radius)
            
            # Apply the label to all affected nodes
            if affected_indices:
                # Handle wrap-around logic based on view type
                for idx in affected_indices:
                    if view_type == "XY":
                        # Handle f_init wrap-around for XY view
                        point_f_init = self.points[idx, 1]
                        node_f_init = node_pos[1]
                        
                        # Calculate the effective label considering wrap-around
                        effective_label = new_label
                        if point_f_init > 180 and node_f_init < -90:
                            # Point is in upper wrap, adjust label
                            effective_label = new_label + 1
                        elif point_f_init < -180 and node_f_init > 90:
                            # Point is in lower wrap, adjust label
                            effective_label = new_label - 1
                        
                        # Prevent accidental UNLABELED±1
                        if effective_label in (self.UNLABELED + 1, self.UNLABELED - 1):
                            effective_label = self.UNLABELED
                    else:
                        # For XZ view, no wrap-around adjustment needed
                        effective_label = new_label
                    
                    self.labels[idx] = effective_label
                    self.group[idx] = self.active_group
                
                total_affected_nodes += len(affected_indices)
                updated_count += 1
                print(f"Updated node {solver_node_idx} to label {new_label} ({view_type} view), affecting {len(affected_indices)} nearby nodes within radius {radius}")
            else:
                # Just update the single node if no nearby nodes found
                self.labels[solver_node_idx] = new_label
                self.group[solver_node_idx] = self.active_group
                updated_count += 1
                total_affected_nodes += 1
                print(f"Updated node {solver_node_idx} to label {new_label} ({view_type} view) - no nearby nodes within radius")
        
        # Update the views to reflect the changes
        self.update_views()
        
        # Also update the OME-Zarr view if it's still open
        if hasattr(self, 'ome_zarr_window') and self.ome_zarr_window is not None:
            self.ome_zarr_window.update_overlay_labels(self.labels, self.calculated_labels)
        
        print(f"Applied {updated_count} label updates from OME-Zarr view ({view_type}), affecting {total_affected_nodes} total nodes with radius {radius}")

    def _get_nearby_indices_xy_with_radius(self, node_pos, radius):
        """Get nearby node indices using the same XY logic as mouse drawing."""
        # Use the exact same function that mouse drawing uses
        x, y = node_pos[0], node_pos[1]  # f_star, f_init
        indices = self.get_nearby_indices_xy(x, y, radius)
        
        # Apply the same Z filtering as mouse drawing
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        
        # Filter indices to only those in the Z slab
        filtered_indices = [i for i in indices if z_min_val <= self.points[i, 2] <= z_max_val]
        return filtered_indices

    def _get_nearby_indices_xz_with_radius(self, node_pos, radius):
        """Get nearby node indices using the same XZ logic as mouse drawing."""
        # Use the exact same logic as _apply_brush_path_to_labels_xz
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
        finit_min_val = finit_center - finit_thickness / 2
        finit_max_val = finit_center + finit_thickness / 2
        
        # Same slab filtering as mouse drawing
        slab_mask = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        slab_indices = np.nonzero(slab_mask)[0]
        if slab_indices.size == 0:
            return []
        
        # Same shear and coordinate logic as mouse drawing
        shear_factor = np.tan(np.radians(self.xz_shear_spinbox.value()))
        if self.show_original_points:
            pts = self.original_points[slab_indices]
        else:
            pts = self.points[slab_indices]
        
        x_disp = pts[:, 0] + shear_factor * (pts[:, 1] - finit_center)
        z_disp = pts[:, 2]
        coords = np.vstack([x_disp, z_disp]).T
        
        # Same temporary KD-tree logic as mouse drawing
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        
        node_x_disp = node_pos[0] + shear_factor * (node_pos[1] - finit_center)
        node_z_disp = node_pos[2]
        
        rel_idxs = tree.query_ball_point([node_x_disp, node_z_disp], r=radius)
        return [slab_indices[j] for j in rel_idxs]
