import sys, os, ast, numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QAction, QMessageBox, QInputDialog, QGraphicsEllipseItem, QFileDialog,
    QProgressDialog, QDialog, QFormLayout, QDialogButtonBox, QLineEdit
)
from PyQt5.QtCore import Qt, QEvent, QPointF
import pyqtgraph as pg
from scipy.spatial import cKDTree
import time

# --------------------------------------------------
# Importing the custom graph problem library.
# --------------------------------------------------
sys.path.append('ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

def point_to_polyline_distance(point, polyline):
    """
    Compute the minimum distance from a point (x,y) to a polyline.
    The polyline is a 2D numpy array representing connected points.
    """
    min_dist = np.inf
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        v = p2 - p1
        w = point - p1
        if np.all(v == 0):
            dist = np.linalg.norm(w)
        else:
            t = np.dot(w, v) / np.dot(v, v)
            if t < 0:
                dist = np.linalg.norm(point - p1)
            elif t > 1:
                dist = np.linalg.norm(point - p2)
            else:
                proj = p1 + t * v
                dist = np.linalg.norm(point - proj)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def vectorized_point_to_polyline_distance(point, polyline):
    """
    Compute the minimum distance from a point (2,) to a polyline.
    Uses vectorized operations.
    """
    p1 = polyline[:-1]
    p2 = polyline[1:]
    v = p2 - p1
    w = point - p1
    dot_wv = np.einsum('ij,ij->i', w, v)
    dot_vv = np.einsum('ij,ij->i', v, v)
    t = np.divide(dot_wv, dot_vv, out=np.zeros_like(dot_wv), where=dot_vv != 0)
    t = np.clip(t, 0, 1)
    proj = p1 + (t[:, None] * v)
    dists = np.linalg.norm(point - proj, axis=1)
    return np.min(dists)

# --------------------------------------------------
# Main: Create GUI.
# --------------------------------------------------
class PointCloudLabeler(QMainWindow):
    def __init__(self, point_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Graph Labeler")
        
        # Default data settings
        self.graph_path = "/media/julian/2/Scroll5/scroll5_complete_surface_points_zarrtest/1352_3600_5005/graph.bin"
        self.default_z_min = 3000
        self.default_z_max = 4000
        self.default_experiment = "denominator3-rotated"
        
        # Initialize solver if no external point data is provided.
        if point_data is None:
            self.solver = graph_problem_gpu_py.Solver(
                self.graph_path, z_min=self.default_z_min, z_max=self.default_z_max)
            gt_path = os.path.join("experiments", self.default_experiment,
                                   "checkpoints", "checkpoint_graph_solver_connected_2.bin")
            if not os.path.exists(gt_path):
                gt_path = os.path.join("experiments", self.default_experiment,
                                       "checkpoints", "checkpoint_graph_f_star_final.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path)
            else:
                print("Default graph file not found; continuing without loading.")
            point_data = self.solver.get_positions()
        else:
            self.solver = None
        
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
        
        # Spline storage (keys: effective winding number, values: list of polylines)
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
        self.z_center_widget, self.z_center_slider, self.z_center_spinbox = self.create_sync_slider_spinbox(
            "Z slice center:", self.z_min, self.z_max, (self.z_min + self.z_max) / 2)
        self.z_thickness_widget, self.z_thickness_slider, self.z_thickness_spinbox = self.create_sync_slider_spinbox(
            "Z slice thickness:", 0.01, self.z_max - self.z_min, (self.z_max - self.z_min) * 0.1)
        xy_controls.addWidget(self.z_center_widget)
        xy_controls.addWidget(self.z_thickness_widget)
        left_column.addLayout(xy_controls)
        # Add two new shear controls for the XY view:
        # Vertical shear (rotating around the f_star axis)
        xy_vertical_shear_layout = QHBoxLayout()
        self.xy_vertical_shear_widget, self.xy_vertical_shear_slider, self.xy_vertical_shear_spinbox = self.create_sync_slider_spinbox(
            "XY Vertical Shear (°):", -90.0, 90.0, 0.0, decimals=1)
        xy_vertical_shear_layout.addWidget(self.xy_vertical_shear_widget)
        left_column.addLayout(xy_vertical_shear_layout)
        # Horizontal shear (rotating around the f_init axis)
        xy_horizontal_shear_layout = QHBoxLayout()
        self.xy_horizontal_shear_widget, self.xy_horizontal_shear_slider, self.xy_horizontal_shear_spinbox = self.create_sync_slider_spinbox(
            "XY Horizontal Shear (°):", -90.0, 90.0, 0.0, decimals=1)
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
        self.finit_center_widget, self.finit_center_slider, self.finit_center_spinbox = self.create_sync_slider_spinbox(
            "f init center:", float(np.min(self.points[:, 1])), float(np.max(self.points[:, 1])),
            (np.min(self.points[:, 1]) + np.max(self.points[:, 1])) / 2)
        self.finit_thickness_widget, self.finit_thickness_slider, self.finit_thickness_spinbox = self.create_sync_slider_spinbox(
            "f init thickness:", 0.01, float(np.max(self.points[:, 1]) - np.min(self.points[:, 1])), 5.0)
        xz_controls.addWidget(self.finit_center_widget)
        xz_controls.addWidget(self.finit_thickness_widget)
        right_column.addLayout(xz_controls)
        # XZ shear control (unchanged functionality)
        xz_shear_layout = QHBoxLayout()
        self.xz_shear_widget, self.xz_shear_slider, self.xz_shear_spinbox = self.create_sync_slider_spinbox(
            "XZ Shear (°):", -90.0, 90.0, 0.0, decimals=1)
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
        self.radius_widget, self.radius_slider, self.radius_spinbox = self.create_sync_slider_spinbox(
            "Drawing radius:", 1.0, 20.0, 5, decimals=0)
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
        self.label_spinbox.setRange(-1000, 1000)
        self.label_spinbox.setValue(1)
        label_save_layout.addWidget(QLabel("Label:"))
        label_save_layout.addWidget(self.label_spinbox)
        common_controls_layout.addLayout(label_save_layout)
        
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
        
        help_menu = menu_bar.addMenu("Help")
        usage_action = QAction("Usage", self)
        usage_action.triggered.connect(self.show_help)
        help_menu.addAction(usage_action)
    
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
            self.solver = graph_problem_gpu_py.Solver(self.graph_path,
                                                    z_min=self.default_z_min,
                                                    z_max=self.default_z_max)
            gt_path = os.path.join("experiments", exp_name, "checkpoints", "checkpoint_graph_solver_connected_2.bin")
            if not os.path.exists(gt_path):
                gt_path = os.path.join("experiments", exp_name, "checkpoints", "checkpoint_graph_f_star_final.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path)
            else:
                QMessageBox.warning(self, "Load Data", f"Graph file not found at {gt_path}")
            new_points = self.solver.get_positions()
            self.points = np.array(new_points)
            self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
            self.calculated_labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
            self.kdtree_xy = cKDTree(self.points[:, [0, 1]])
            self.kdtree_xz = cKDTree(self.points[:, [0, 2]])
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
        fname, _ = QFileDialog.getOpenFileName(self, "Load Labels", os.path.join("experiments", self.default_experiment),
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
    
    def create_sync_slider_spinbox(self, label_text, min_val, max_val, default_val, decimals=2):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * self.scaleFactor))
        slider.setMaximum(int(max_val * self.scaleFactor))
        slider.setValue(int(default_val * self.scaleFactor))
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(decimals)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(1.0)
        slider.valueChanged.connect(lambda val: (spinbox.blockSignals(True), spinbox.setValue(val / self.scaleFactor),
                                                  spinbox.blockSignals(False), self.update_views()))
        spinbox.valueChanged.connect(lambda val: (slider.blockSignals(True), slider.setValue(int(val * self.scaleFactor)),
                                                   slider.blockSignals(False), self.update_views()))
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        return container, slider, spinbox
    
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
        return self.kdtree_xy.query_ball_point([x, effective_y], r=r)
    
    def update_guides(self):
        # For the XY view:
        if self.show_guides_checkbox.isChecked():
            try:
                self.xy_plot.addItem(self.line_finit_neg)
                self.xy_plot.addItem(self.line_finit_pos)
            except Exception:
                pass
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
            upper = finit_center + finit_thickness / 2
            lower = finit_center - finit_thickness / 2
            self.line_finit_center.setPos(finit_center)
            self.line_finit_upper.setPos(upper)
            self.line_finit_lower.setPos(lower)
            try:
                self.xy_plot.addItem(self.line_finit_center)
                self.xy_plot.addItem(self.line_finit_upper)
                self.xy_plot.addItem(self.line_finit_lower)
            except Exception:
                pass
            # Add the orange indicator (original XZ shear) to the XY view.
            self.xz_shear_indicator.setAngle(self.xz_shear_spinbox.value())
            center_f_star = (self.f_star_min + self.f_star_max) / 2
            center_f_init = (self.f_init_min + self.f_init_max) / 2
            self.xz_shear_indicator.setPos(QPointF(center_f_star, center_f_init))
            try:
                self.xy_plot.addItem(self.xz_shear_indicator)
            except Exception:
                pass
        else:
            try:
                self.xy_plot.removeItem(self.line_finit_neg)
                self.xy_plot.removeItem(self.line_finit_pos)
                self.xy_plot.removeItem(self.line_finit_center)
                self.xy_plot.removeItem(self.line_finit_upper)
                self.xy_plot.removeItem(self.line_finit_lower)
                self.xy_plot.removeItem(self.xz_shear_indicator)
            except Exception:
                pass
        
        # For the XZ view:
        if self.show_guides_checkbox.isChecked():
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value() / self.scaleFactor
            self.line_z_center.setPos(z_center)
            self.line_z_upper.setPos(z_center + z_thickness / 2)
            self.line_z_lower.setPos(z_center - z_thickness / 2)
            try:
                self.xz_plot.addItem(self.line_z_center)
                self.xz_plot.addItem(self.line_z_upper)
                self.xz_plot.addItem(self.line_z_lower)
            except Exception:
                pass
            # Add the purple indicator (new XY horizontal shear) to the XZ view.
            self.xy_horizontal_indicator.setAngle(self.xy_horizontal_shear_spinbox.value())
            center_f_star = (self.f_star_min + self.f_star_max) / 2
            center_z = (self.z_min + self.z_max) / 2
            self.xy_horizontal_indicator.setPos(QPointF(center_f_star, center_z))
            try:
                self.xz_plot.addItem(self.xy_horizontal_indicator)
            except Exception:
                pass
        else:
            try:
                self.xz_plot.removeItem(self.line_z_center)
                self.xz_plot.removeItem(self.line_z_upper)
                self.xz_plot.removeItem(self.line_z_lower)
                self.xz_plot.removeItem(self.xy_horizontal_indicator)
            except Exception:
                pass
    
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
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        current_label = self.label_spinbox.value()
        r = self.radius_spinbox.value()
        if view_name == 'xy':
            if y_m > 180:
                update_label = current_label + 1
            elif y_m < -180:
                update_label = current_label - 1
            else:
                update_label = current_label
            if update_label in [self.UNLABELED + 1, self.UNLABELED - 1]:
                update_label = self.UNLABELED
            indices = self.get_nearby_indices_xy(x_m, y_m, r)
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value() / self.scaleFactor
            z_min_val = z_center - z_thickness / 2
            z_max_val = z_center + z_thickness / 2
            indices = [i for i in indices if self.points[i, 2] >= z_min_val and self.points[i, 2] <= z_max_val]
        elif view_name == 'xz':
            indices = self.kdtree_xz.query_ball_point([x_m, y_m], r=r)
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
            finit_min_val = finit_center - finit_thickness / 2
            finit_max_val = finit_center + finit_thickness / 2
            indices = [i for i in indices if self.points[i, 1] >= finit_min_val and self.points[i, 1] <= finit_max_val]
            update_label = current_label
        else:
            indices = []
        indices = np.array(indices)
        if indices.size:
            self.labels[indices] = update_label
        self.update_views()
    
    def _paint_points_calculated(self, ev, plot_widget, view_name):
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        r = self.radius_spinbox.value()
        if view_name == 'xy':
            indices = self.get_nearby_indices_xy(x_m, y_m, r)
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value() / self.scaleFactor
            z_min_val = z_center - z_thickness / 2
            z_max_val = z_center + z_thickness / 2
            indices = [i for i in indices if self.points[i, 2] >= z_min_val and self.points[i, 2] <= z_max_val]
            for i in indices:
                if self.labels[i] == self.UNLABELED and self.calculated_labels[i] != self.UNLABELED:
                    self.labels[i] = self.calculated_labels[i]
        elif view_name == 'xz':
            indices = self.kdtree_xz.query_ball_point([x_m, y_m], r=r)
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
            finit_min_val = finit_center - finit_thickness / 2
            finit_max_val = finit_center + finit_thickness / 2
            indices = [i for i in indices if self.points[i, 1] >= finit_min_val and self.points[i, 1] <= finit_max_val]
            for i in indices:
                if self.labels[i] == self.UNLABELED and self.calculated_labels[i] != self.UNLABELED:
                    self.labels[i] = self.calculated_labels[i]
        self.update_views()
    
    def update_views(self):
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness / 2
        z_max_val = z_center + z_thickness / 2
        mask_xy = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        pts_xy = self.points[mask_xy]
        labels_xy = self.labels[mask_xy]
        calc_labels_xy = self.calculated_labels[mask_xy]
        mask_top = pts_xy[:, 1] < -90
        pts_top = pts_xy[mask_top].copy()
        if pts_top.size:
            pts_top[:, 1] += 360
        labels_top = labels_xy[mask_top] - 1
        calc_labels_top = calc_labels_xy[mask_top] - 1
        mask_bottom = pts_xy[:, 1] > 90
        pts_bottom = pts_xy[mask_bottom].copy()
        if pts_bottom.size:
            pts_bottom[:, 1] -= 360
        labels_bottom = labels_xy[mask_bottom] + 1
        calc_labels_bottom = calc_labels_xy[mask_bottom] + 1
        pts_combined = np.concatenate([pts_xy, pts_top, pts_bottom], axis=0)
        labels_combined = np.concatenate([labels_xy, labels_top, labels_bottom], axis=0)
        calc_labels_combined = np.concatenate([calc_labels_xy, calc_labels_top, calc_labels_bottom], axis=0)
        pts_combined, labels_combined, calc_labels_combined = self.downsample_points(
            pts_combined, labels_combined, calc_labels_combined, self.max_display)
        brushes_xy = self.get_brushes_from_labels(labels_combined)
        # Apply two shear transformations for the XY view:
        # Vertical shear (shifts f_init)
        shear_vertical = self.xy_vertical_shear_spinbox.value()
        vertical_factor = np.tan(np.radians(shear_vertical))
        # Horizontal shear (shifts f_star)
        shear_horizontal = self.xy_horizontal_shear_spinbox.value()
        horizontal_factor = np.tan(np.radians(shear_horizontal))
        new_f_init = pts_combined[:, 1] + vertical_factor * (pts_combined[:, 2] - z_center)
        new_f_star = pts_combined[:, 0] + horizontal_factor * (pts_combined[:, 2] - z_center)
        self.xy_scatter.setData(x=new_f_star, y=new_f_init,
                                size=self.point_size, pen=None, brush=brushes_xy)
        
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
        finit_min_val = finit_center - finit_thickness / 2
        finit_max_val = finit_center + finit_thickness / 2
        mask_xz = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        pts_xz = self.points[mask_xz]
        labels_xz = self.labels[mask_xz]
        shear_angle_deg_xz = self.xz_shear_spinbox.value()
        shear_factor = np.tan(np.radians(shear_angle_deg_xz))
        if pts_xz.size:
            x_new = pts_xz[:, 0] + shear_factor * (pts_xz[:, 1] - finit_center)
        else:
            x_new = pts_xz[:, 0]
        pts_xz_display = pts_xz.copy()
        pts_xz_display[:, 0] = x_new
        pts_xz_display, labels_xz = self.downsample_points(pts_xz_display, labels=labels_xz, max_display=self.max_display)
        brushes_xz = self.get_brushes_from_labels(labels_xz)
        self.xz_scatter.setData(x=pts_xz_display[:, 0], y=pts_xz_display[:, 2],
                                size=self.point_size, pen=None, brush=brushes_xz)
        
        mask_calc = (np.abs(labels_combined - self.UNLABELED) <= 1) & (np.abs(calc_labels_combined - self.UNLABELED) > 1)
        pts_combined_calc = pts_combined[mask_calc]
        labels_combined_calc = calc_labels_combined[mask_calc]
        brushes_calc = np.empty(labels_combined_calc.shape[0], dtype=object)
        for i, lab in enumerate(labels_combined_calc):
            mod = lab % 3
            if mod == 0:
                brushes_calc[i] = self.calc_brush_red
            elif mod == 1:
                brushes_calc[i] = self.calc_brush_green
            elif mod == 2:
                brushes_calc[i] = self.calc_brush_blue
        self.xy_calc_scatter.setData(x=pts_combined_calc[:, 0] if pts_combined_calc.size else [],
                                     y=pts_combined_calc[:, 1] if pts_combined_calc.size else [],
                                     size=self.point_size, pen=None, brush=brushes_calc)
        
        mask_xz = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        pts_xz_full = self.points[mask_xz]
        manual_labels_xz = self.labels[mask_xz]
        calc_labels_xz = self.calculated_labels[mask_xz]
        valid_calc_mask_xz = (manual_labels_xz == self.UNLABELED) & (calc_labels_xz != self.UNLABELED)
        pts_calc_xz = pts_xz_full[valid_calc_mask_xz]
        labels_calc_xz = calc_labels_xz[valid_calc_mask_xz]
        pts_calc_xz_display = pts_calc_xz.copy()
        if pts_calc_xz.size:
            pts_calc_xz_display[:, 0] = pts_calc_xz_display[:, 0] + shear_factor * (pts_calc_xz_display[:, 1] - finit_center)
        brushes_calc_xz = np.empty(labels_calc_xz.shape[0], dtype=object)
        for i, lab in enumerate(labels_calc_xz):
            mod = lab % 3
            if mod == 0:
                brushes_calc_xz[i] = self.calc_brush_red
            elif mod == 1:
                brushes_calc_xz[i] = self.calc_brush_green
            elif mod == 2:
                brushes_calc_xz[i] = self.calc_brush_blue
        self.xz_calc_scatter.setData(x=pts_calc_xz_display[:, 0] if pts_calc_xz_display.size else [],
                                     y=pts_calc_xz_display[:, 2] if pts_calc_xz_display.size else [],
                                     size=self.point_size, pen=None, brush=brushes_calc_xz)
        
        self.update_guides()
    
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
    
    def update_labels(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            # if not np.any(gt):
            #     return
            self.solver.set_labels(self.labels, gt)
            self.solver.solve_winding_number(num_iterations=500, i_round=-3, seed_node=-1,
                                             other_block_factor=15.0, side_fix_nr=-1, display=False)
            # self.solver.solve_union()
            calculated_labels = self.solver.get_labels()
            self.calculated_labels = np.array(calculated_labels)
            self.update_views()
    
    def save_graph(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            self.solver.set_labels(self.labels, gt)
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

# --------------------------------------------------
# Main: Create GUI.
# --------------------------------------------------
def main():
    app = QApplication(sys.argv)
    gui = PointCloudLabeler()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
