import sys, os, ast, numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QAction, QMessageBox, QInputDialog, QGraphicsEllipseItem, QFileDialog
)
from PyQt5.QtCore import Qt, QEvent
import pyqtgraph as pg
from scipy.spatial import cKDTree

# --------------------------------------------------
# Import your custom library.
# --------------------------------------------------
sys.path.append('ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

# --------------------------------------------------
# Graph Labeler GUI with updated pipette, solver, and calculated label tools.
# --------------------------------------------------
class PointCloudLabeler(QMainWindow):
    def __init__(self, point_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Graph Labeler")
        
        # --- Default settings for data (used when loading via the Data menu) ---
        self.graph_path = "/media/julian/2/Scroll5/scroll5_complete_surface_points_zarrtest/1352_3600_5005/graph.bin"
        self.default_z_min = 3000
        self.default_z_max = 4000
        self.default_experiment = "denominator3-rotated"
        
        # --- Initialize solver if no point data is provided ---
        if point_data is None:
            self.solver = graph_problem_gpu_py.Solver(self.graph_path,
                                                       z_min=self.default_z_min,
                                                       z_max=self.default_z_max)
            gt_path = os.path.join("experiments", self.default_experiment,
                                   "checkpoints", "checkpoint_graph_solver_connected_2.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path)
            else:
                print("Default graph file not found; continuing without loading.")
            point_data = self.solver.get_positions()
        else:
            self.solver = None
        
        # --- Global Variables ---
        self.scaleFactor = 100  # for converting float values to slider integers
        self.s_pressed = False
        self.original_drawing_mode = True
        self.pipette_mode = False  # when True, the next left-click (in either view) will pick a label
        self.calc_drawing_mode = False  # when True, drawing strokes will transfer calculated labels
        
        self.UNLABELED = -9999  # special value for unlabeled points
        
        self.undo_stack = []
        self.redo_stack = []
        self._stroke_backup = None
        
        # --- Create Menus ---
        self._create_menu()
        
        # --- Data and Labels ---
        self.points = np.array(point_data)  # shape: (N,3)
        self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        self.calculated_labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        
        # --- Display Parameters ---
        self.point_size = 3
        self.max_display = 200000  # increased 4×
        
        # Coordinate naming.
        self.f_star_min, self.f_star_max = float(np.min(self.points[:,0])), float(np.max(self.points[:,0]))
        self.f_init_min, self.f_init_max = -180.0, 180.0  
        self.z_min, self.z_max = float(np.min(self.points[:,2])), float(np.max(self.points[:,2]))
        
        # --- KD-trees ---
        self.kdtree_xy = cKDTree(self.points[:, [0,1]])
        self.kdtree_xz = cKDTree(self.points[:, [0,2]])
        
        # --- Pre-created Brushes for manual labels ---
        self.brush_black = pg.mkBrush(0, 0, 0)
        self.brush_red   = pg.mkBrush(255, 0, 0)
        self.brush_green = pg.mkBrush(0, 255, 0)
        self.brush_blue  = pg.mkBrush(0, 0, 255)
        # --- Brushes for calculated labels (faded) ---
        self.calc_brush_black = pg.mkBrush(0, 0, 0, 100)
        self.calc_brush_red   = pg.mkBrush(255, 0, 0, 100)
        self.calc_brush_green = pg.mkBrush(0, 255, 0, 100)
        self.calc_brush_blue  = pg.mkBrush(0, 0, 255, 100)
        
        # --- Guide Items ---
        self.line_finit_neg = pg.InfiniteLine(pos=-180, angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_pos = pg.InfiniteLine(pos=180, angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_center = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_upper  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_lower  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_center = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_upper  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_lower  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.shear_indicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('orange', width=1, style=Qt.DashLine))
        
        # --- Cursor Circle ---
        self.cursor_circle = QGraphicsEllipseItem(0, 0, 0, 0)
        self.cursor_circle.setPen(pg.mkPen('cyan', width=1, style=Qt.DashLine))
        self.cursor_circle.setVisible(False)
        
        # --- Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Horizontal layout for two view columns.
        views_columns_layout = QHBoxLayout()
        main_layout.addLayout(views_columns_layout)
        
        # Left Column: XY view and its slice controls.
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
        # XY slice controls (Z slice):
        xy_controls = QHBoxLayout()
        self.z_center_widget, self.z_center_slider, self.z_center_spinbox = self.create_sync_slider_spinbox(
            "Z slice center:", self.z_min, self.z_max, (self.z_min+self.z_max)/2)
        self.z_thickness_widget, self.z_thickness_slider, self.z_thickness_spinbox = self.create_sync_slider_spinbox(
            "Z slice thickness:", 0.01, self.z_max-self.z_min, (self.z_max-self.z_min)*0.1)
        xy_controls.addWidget(self.z_center_widget)
        xy_controls.addWidget(self.z_thickness_widget)
        left_column.addLayout(xy_controls)
        # --- Button to apply calculated labels in current XY slice ---
        self.apply_calc_xy_button = QPushButton("Apply Updated Labels to XY")
        self.apply_calc_xy_button.clicked.connect(self.apply_calculated_labels_xy)
        left_column.addWidget(self.apply_calc_xy_button)
        
        # Right Column: XZ view and its slice controls (f_init slice).
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
            "f init center:", float(np.min(self.points[:,1])), float(np.max(self.points[:,1])),
            (np.min(self.points[:,1])+np.max(self.points[:,1]))/2)
        self.finit_thickness_widget, self.finit_thickness_slider, self.finit_thickness_spinbox = self.create_sync_slider_spinbox(
            "f init thickness:", 0.01, float(np.max(self.points[:,1]) - np.min(self.points[:,1])), 5.0)
        xz_controls.addWidget(self.finit_center_widget)
        xz_controls.addWidget(self.finit_thickness_widget)
        right_column.addLayout(xz_controls)
        # --- Button to apply calculated labels in current XZ slice ---
        self.apply_calc_xz_button = QPushButton("Apply Updated Labels to XZ")
        self.apply_calc_xz_button.clicked.connect(self.apply_calculated_labels_xz)
        right_column.addWidget(self.apply_calc_xz_button)
        
        # Common Controls row.
        common_controls_layout = QHBoxLayout()
        main_layout.addLayout(common_controls_layout)
        self.radius_widget, self.radius_slider, self.radius_spinbox = self.create_sync_slider_spinbox(
            "Drawing radius:", 1.0, 20.0, 7.5, decimals=0)
        common_controls_layout.addWidget(self.radius_widget)
        self.shear_widget, self.shear_slider, self.shear_spinbox = self.create_sync_slider_spinbox(
            "Shear (°):", -90.0, 90.0, 0.0, decimals=1)
        common_controls_layout.addWidget(self.shear_widget)
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
        # --- Buttons for updated label tools ---
        self.update_labels_button = QPushButton("Update Labels")
        self.update_labels_button.clicked.connect(self.update_labels)
        common_controls_layout.addWidget(self.update_labels_button)
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
        
        # Create scatter items.
        self.xy_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xz_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xy_plot.addItem(self.xy_scatter)
        self.xz_plot.addItem(self.xz_scatter)
        # New scatter items for calculated labels (overlay)
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
    
    # --------------------------
    # Event filter for updating the cursor circle.
    # --------------------------
    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseMove and source is self.xy_plot.scene():
            if self.drawing_mode_checkbox.isChecked():
                pos = event.scenePos()
                dataPos = self.xy_plot.plotItem.vb.mapSceneToView(pos)
                r = self.radius_spinbox.value()
                self.cursor_circle.setRect(dataPos.x()-r, dataPos.y()-r, 2*r, 2*r)
                self.cursor_circle.setVisible(True)
            else:
                self.cursor_circle.setVisible(False)
            return False
        return super(PointCloudLabeler, self).eventFilter(source, event)
    
    # --------------------------
    # Menu creation and Data loading.
    # --------------------------
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
        self.graph_path, ok = QInputDialog.getText(self, "Load Data", "Enter path to graph.bin:", text=self.graph_path)
        if not ok or not self.graph_path:
            return
        exp_name, ok = QInputDialog.getText(self, "Load Data", "Enter experiment name:", text=self.default_experiment)
        if not ok or not exp_name:
            return
        if os.path.exists(self.graph_path):
            self.solver = graph_problem_gpu_py.Solver(self.graph_path,
                                                       z_min=self.default_z_min,
                                                       z_max=self.default_z_max)
            gt_path = os.path.join("experiments", exp_name, "checkpoints", "checkpoint_graph_solver_connected_2.bin")
            if os.path.exists(gt_path):
                self.solver.load_graph(gt_path)
            else:
                QMessageBox.warning(self, "Load Data", f"Graph file not found at {gt_path}")
            new_points = self.solver.get_positions()
            self.points = np.array(new_points)
            self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)  # discard old labels
            self.kdtree_xy = cKDTree(self.points[:, [0,1]])
            self.kdtree_xz = cKDTree(self.points[:, [0,2]])
            self.update_views()
            print(f"Loaded graph from {self.graph_path} for experiment {exp_name}")
        else:
            QMessageBox.warning(self, "Load Data", f"File {self.graph_path} does not exist.")
    
    def save_labels_to_path(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Labels", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            with open(fname, "w") as f:
                f.write(str(self.labels.tolist()))
            print(f"Labels saved to {fname}")
    
    def load_labels_from_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Labels", os.path.join("experiments", self.default_experiment), "Text Files (*.txt);;All Files (*)")
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
                print(f"Labels loaded from {fname}")
            else:
                QMessageBox.warning(self, "Load Labels", "Loaded labels length does not match current data.")
    
    def show_help(self):
        help_text = (
            "Usage Instructions:\n\n"
            "1. Views:\n"
            "   - Left (XY) view: horizontal axis = f_star; vertical axis = f_init (angular).\n"
            "     Grey dashed lines at f_init = -180 and 180 show the primary range.\n"
            "     Grey dashed lines (from the right column) indicate the current f init slice (center and thickness).\n"
            "     The shear indicator (orange dashed line) shows the shear angle (in degrees) used in the XZ view.\n"
            "   - Right (XZ) view: horizontal axis = f_star; vertical axis = Z.\n\n"
            "2. Slice Controls:\n"
            "   - Left column: Adjust Z slice (center and thickness) to filter points for the XY view.\n"
            "   - Right column: Adjust f init center and thickness to filter points for the XZ view.\n\n"
            "3. Common Controls:\n"
            "   - Drawing radius – a circle around the cursor shows the affected area.\n"
            "   - Shear: in the XZ view, f_star' = f_star + tan(shear°)*(f_init - f init center).\n"
            "   - Max Display Points for performance.\n"
            "   - Toggle Drawing Mode; press and hold 'S' to temporarily disable drawing mode.\n"
            "   - Toggle 'Show guides' to show/hide overlay indicator lines.\n"
            "   - Pipette: click the Pipette button (or press 'P') and then click in either view to pick a label.\n"
            "   - Update Labels: generates labels via the solver.\n"
            "   - Save Labeled Graph: Saves the labeled graph as solved graph for further meshing.\n\n"
            "4. Drawing (in the XY view):\n"
            "   - Manual drawing applies the label from the Label spinbox.\n"
            "   - If drawing in the virtual regions (f_init > 180 or < -180), the label is adjusted (+1 or -1).\n\n"
            "5. Updated Label Tools:\n"
            "   - Updated Labels Overlay: Points that are unlabeled but have a solver-computed label are shown in faded colors.\n"
            "   - Updated Labels Draw Mode: Toggle this mode (via the 'Updated Labels Draw Mode' button) to allow drawing strokes that transfer\n"
            "     the updated label (if available) onto points.\n"
            "   - Clear Updated Labels: Clears all calculated labels.\n"
            "   - Apply Updated Labels to XY/XZ: Under each view, these buttons copy all calculated labels (in the current slice)\n"
            "     over to the manual labels for points that are still unlabeled.\n\n"
            "6. Undo/Redo:\n"
            "   - Press Ctrl+Z to undo; Ctrl+Y to redo.\n\n"
            "7. Saving:\n"
            "   - Use the Data menu to save or load labels.\n"
        )
        QMessageBox.information(self, "Tool Usage", help_text)
    
    # --------------------------
    # Synchronized slider and spinbox helper.
    # --------------------------
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
        slider.valueChanged.connect(lambda val: (spinbox.blockSignals(True), spinbox.setValue(val / self.scaleFactor), spinbox.blockSignals(False), self.update_views()))
        spinbox.valueChanged.connect(lambda val: (slider.blockSignals(True), slider.setValue(int(val * self.scaleFactor)), slider.blockSignals(False), self.update_views()))
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        return container, slider, spinbox
    
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
    
    # --------------------------
    # Helper: return the displayed label for point index i using given y.
    # --------------------------
    def displayed_label(self, i, y):
        lab = self.labels[i]
        if y < -180:
            return lab + 1
        elif y > 180:
            return lab - 1
        else:
            return lab
    
    # --------------------------
    # Helper: return nearby indices in the XY view.
    # --------------------------
    def get_nearby_indices_xy(self, x, y, r):
        if y > 180:
            effective_y = y - 360
        elif y < -180:
            effective_y = y + 360
        else:
            effective_y = y
        return self.kdtree_xy.query_ball_point([x, effective_y], r=r)
    
    # --------------------------
    # Guide overlays.
    # --------------------------
    def update_guides(self):
        if self.show_guides_checkbox.isChecked():
            try:
                self.xy_plot.addItem(self.line_finit_neg)
                self.xy_plot.addItem(self.line_finit_pos)
            except Exception:
                pass
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value()/self.scaleFactor
            upper = finit_center + finit_thickness/2
            lower = finit_center - finit_thickness/2
            self.line_finit_center.setPos(finit_center)
            self.line_finit_upper.setPos(upper)
            self.line_finit_lower.setPos(lower)
            try:
                self.xy_plot.addItem(self.line_finit_center)
                self.xy_plot.addItem(self.line_finit_upper)
                self.xy_plot.addItem(self.line_finit_lower)
            except Exception:
                pass
            angle = self.shear_spinbox.value()
            self.shear_indicator.setAngle(angle)
            try:
                self.xy_plot.addItem(self.shear_indicator)
            except Exception:
                pass
        else:
            try:
                self.xy_plot.removeItem(self.line_finit_neg)
                self.xy_plot.removeItem(self.line_finit_pos)
                self.xy_plot.removeItem(self.line_finit_center)
                self.xy_plot.removeItem(self.line_finit_upper)
                self.xy_plot.removeItem(self.line_finit_lower)
                self.xy_plot.removeItem(self.shear_indicator)
            except Exception:
                pass
        
        if self.show_guides_checkbox.isChecked():
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value()/self.scaleFactor
            self.line_z_center.setPos(z_center)
            self.line_z_upper.setPos(z_center + z_thickness/2)
            self.line_z_lower.setPos(z_center - z_thickness/2)
            try:
                self.xz_plot.addItem(self.line_z_center)
                self.xz_plot.addItem(self.line_z_upper)
                self.xz_plot.addItem(self.line_z_lower)
            except Exception:
                pass
        else:
            try:
                self.xz_plot.removeItem(self.line_z_center)
                self.xz_plot.removeItem(self.line_z_upper)
                self.xz_plot.removeItem(self.line_z_lower)
            except Exception:
                pass
    
    # --------------------------
    # Downsampling helper.
    # --------------------------
    def downsample_points(self, pts, labels, calc_labels=None, max_display=1):
        n = pts.shape[0]
        if n > max_display:
            indices = np.linspace(0, n-1, max_display, dtype=int)
            if calc_labels is not None:
                return pts[indices], labels[indices], calc_labels[indices]
            else:
                return pts[indices], labels[indices]
        else:
            if calc_labels is not None:
                return pts, labels, calc_labels
            else:
                return pts, labels
        
    # --------------------------
    # Brush lookup for manual labels.
    # --------------------------
    def get_brushes_from_labels(self, labels_array):
        brushes = np.empty(labels_array.shape[0], dtype=object)
        mask_unlabeled = (labels_array == self.UNLABELED) | (labels_array == self.UNLABELED+1) | (labels_array == self.UNLABELED-1)
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
    
    # --------------------------
    # Enable pencil tool.
    # --------------------------
    def _enable_pencil(self, plot_widget, scatter_item, view_name='xy'):
        scatter_item.mousePressEvent = lambda ev: self._on_mouse_press(ev, plot_widget, view_name)
        scatter_item.mouseMoveEvent  = lambda ev: self._on_mouse_drag(ev, plot_widget, view_name)
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
    
    # --------------------------
    # Drawing: update manual labels.
    # --------------------------
    def _paint_points(self, ev, plot_widget, view_name):
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        current_label = self.label_spinbox.value()
        r = self.radius_spinbox.value()
        if view_name == 'xy':
            if y_m > 180:
                effective_y = y_m - 360
                update_label = current_label + 1
            elif y_m < -180:
                effective_y = y_m + 360
                update_label = current_label - 1
            else:
                effective_y = y_m
                update_label = current_label
            if update_label == self.UNLABELED+1 or update_label == self.UNLABELED-1:
                update_label = self.UNLABELED
            indices = self.get_nearby_indices_xy(x_m, y_m, r)
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value()/self.scaleFactor
            z_min_val = z_center - z_thickness/2
            z_max_val = z_center + z_thickness/2
            indices = [i for i in indices if self.points[i,2] >= z_min_val and self.points[i,2] <= z_max_val]
        elif view_name == 'xz':
            indices = self.kdtree_xz.query_ball_point([x_m, y_m], r=r)
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value()/self.scaleFactor
            finit_min_val = finit_center - finit_thickness/2
            finit_max_val = finit_center + finit_thickness/2
            indices = [i for i in indices if self.points[i,1] >= finit_min_val and self.points[i,1] <= finit_max_val]
            update_label = current_label
        else:
            indices = []
        indices = np.array(indices)
        if indices.size:
            self.labels[indices] = update_label
        self.update_views()
    
    # --------------------------
    # Calculated Drawing: transfer calculated labels.
    # --------------------------
    def _paint_points_calculated(self, ev, plot_widget, view_name):
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        r = self.radius_spinbox.value()
        if view_name == 'xy':
            indices = self.get_nearby_indices_xy(x_m, y_m, r)
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value()/self.scaleFactor
            z_min_val = z_center - z_thickness/2
            z_max_val = z_center + z_thickness/2
            indices = [i for i in indices if self.points[i,2] >= z_min_val and self.points[i,2] <= z_max_val]
            for i in indices:
                # Only update if manual label is unlabeled and a calculated label exists.
                if self.labels[i] == self.UNLABELED and self.calculated_labels[i] != self.UNLABELED:
                    if y_m > 180:
                        self.labels[i] = self.calculated_labels[i] + 1
                    elif y_m < -180:
                        self.labels[i] = self.calculated_labels[i] - 1
                    else:
                        self.labels[i] = self.calculated_labels[i]
        elif view_name == 'xz':
            indices = self.kdtree_xz.query_ball_point([x_m, y_m], r=r)
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value()/self.scaleFactor
            finit_min_val = finit_center - finit_thickness/2
            finit_max_val = finit_center + finit_thickness/2
            indices = [i for i in indices if self.points[i,1] >= finit_min_val and self.points[i,1] <= finit_max_val]
            for i in indices:
                if self.labels[i] == self.UNLABELED and self.calculated_labels[i] != self.UNLABELED:
                    self.labels[i] = self.calculated_labels[i]
        self.update_views()
    
    # --------------------------
    # Update the views.
    # --------------------------
    def update_views(self):
        # --- XY View: Filter by Z slice (manual labels) ---
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value()/self.scaleFactor
        z_min_val = z_center - z_thickness/2
        z_max_val = z_center + z_thickness/2
        mask_xy = (self.points[:,2] >= z_min_val) & (self.points[:,2] <= z_max_val)
        pts_xy = self.points[mask_xy]
        labels_xy = self.labels[mask_xy]
        calc_labels_xy = self.calculated_labels[mask_xy]
        mask_top = pts_xy[:,1] < -90
        pts_top = pts_xy[mask_top].copy()
        if pts_top.size:
            pts_top[:,1] += 360
        labels_top = labels_xy[mask_top] - 1
        calc_labels_top = calc_labels_xy[mask_top] - 1
        mask_bottom = pts_xy[:,1] > 90
        pts_bottom = pts_xy[mask_bottom].copy()
        if pts_bottom.size:
            pts_bottom[:,1] -= 360
        labels_bottom = labels_xy[mask_bottom] + 1
        calc_labels_bottom = calc_labels_xy[mask_bottom] + 1
        pts_combined = np.concatenate([pts_xy, pts_top, pts_bottom], axis=0)
        labels_combined = np.concatenate([labels_xy, labels_top, labels_bottom], axis=0)
        calc_labels_combined = np.concatenate([calc_labels_xy, calc_labels_top, calc_labels_bottom], axis=0)
        pts_combined, labels_combined, calc_labels_combined = self.downsample_points(pts_combined, labels_combined, calc_labels_combined, self.max_display)
        brushes_xy = self.get_brushes_from_labels(labels_combined)
        self.xy_scatter.setData(x=pts_combined[:,0], y=pts_combined[:,1],
                                size=self.point_size, pen=None, brush=brushes_xy)
        
        # --- XZ View: Filter by f_init slice (manual labels) ---
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value()/self.scaleFactor
        finit_min_val = finit_center - finit_thickness/2
        finit_max_val = finit_center + finit_thickness/2
        mask_xz = (self.points[:,1] >= finit_min_val) & (self.points[:,1] <= finit_max_val)
        pts_xz = self.points[mask_xz]
        labels_xz = self.labels[mask_xz]
        shear_angle_deg = self.shear_spinbox.value()
        shear_factor = np.tan(np.radians(shear_angle_deg))
        if pts_xz.size:
            x_new = pts_xz[:,0] + shear_factor * (pts_xz[:,1] - finit_center)
        else:
            x_new = pts_xz[:,0]
        pts_xz_display = pts_xz.copy()
        pts_xz_display[:,0] = x_new
        pts_xz_display, labels_xz = self.downsample_points(pts_xz_display, labels=labels_xz, max_display=self.max_display)
        brushes_xz = self.get_brushes_from_labels(labels_xz)
        self.xz_scatter.setData(x=pts_xz_display[:,0], y=pts_xz_display[:,2],
                                size=self.point_size, pen=None, brush=brushes_xz)
        
        # --- Overlay: Calculated Labels in XY view ---
        # Only display calculated label if manual label is UNLABELED and a calculated label exists.
        mask_calc = (np.abs(labels_combined - self.UNLABELED) <= 1) & (np.abs(calc_labels_combined - self.UNLABELED) > 1)
        pts_combined_calc = pts_combined[mask_calc]
        labels_combined_calc = calc_labels_combined[mask_calc]
        # Create brushes for calculated labels (faded) based on mod.
        brushes_calc = np.empty(labels_combined_calc.shape[0], dtype=object)
        for i, lab in enumerate(labels_combined_calc):
            mod = lab % 3
            if mod == 0:
                brushes_calc[i] = self.calc_brush_red
            elif mod == 1:
                brushes_calc[i] = self.calc_brush_green
            elif mod == 2:
                brushes_calc[i] = self.calc_brush_blue
        self.xy_calc_scatter.setData(x=pts_combined_calc[:,0] if pts_combined_calc.size else [],
                                     y=pts_combined_calc[:,1] if pts_combined_calc.size else [],
                                     size=self.point_size, pen=None, brush=brushes_calc)
        
        # --- Overlay: Calculated Labels in XZ view ---
        mask_xz = (self.points[:,1] >= finit_min_val) & (self.points[:,1] <= finit_max_val)
        pts_xz_full = self.points[mask_xz]
        manual_labels_xz = self.labels[mask_xz]
        calc_labels_xz = self.calculated_labels[mask_xz]
        valid_calc_mask_xz = (manual_labels_xz == self.UNLABELED) & (calc_labels_xz != self.UNLABELED)
        pts_calc_xz = pts_xz_full[valid_calc_mask_xz]
        labels_calc_xz = calc_labels_xz[valid_calc_mask_xz]
        pts_calc_xz_display = pts_calc_xz.copy()
        if pts_calc_xz.size:
            pts_calc_xz_display[:,0] = pts_calc_xz_display[:,0] + shear_factor * (pts_calc_xz_display[:,1] - finit_center)
        brushes_calc_xz = np.empty(labels_calc_xz.shape[0], dtype=object)
        for i, lab in enumerate(labels_calc_xz):
            mod = lab % 3
            if mod == 0:
                brushes_calc_xz[i] = self.calc_brush_red
            elif mod == 1:
                brushes_calc_xz[i] = self.calc_brush_green
            elif mod == 2:
                brushes_calc_xz[i] = self.calc_brush_blue
        self.xz_calc_scatter.setData(x=pts_calc_xz_display[:,0] if pts_calc_xz_display.size else [],
                                     y=pts_calc_xz_display[:,2] if pts_calc_xz_display.size else [],
                                     size=self.point_size, pen=None, brush=brushes_calc_xz)
        
        self.update_guides()
    
    # --------------------------
    # Pipette: pick label at XY click.
    # --------------------------
    def pick_label_at(self, ev, plot_widget):
        dataPos = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x = dataPos.x()
        y = dataPos.y()
        r = self.radius_spinbox.value()
        indices = self.get_nearby_indices_xy(x, y, r)
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value()/self.scaleFactor
        z_min_val = z_center - z_thickness/2
        z_max_val = z_center + z_thickness/2
        indices = [i for i in indices if self.points[i,2] >= z_min_val and self.points[i,2] <= z_max_val]
        indices = np.array(indices)
        if indices.size == 0:
            print("Pipette: no points found")
            return
        disp_labels = np.array([self.displayed_label(i, y) for i in indices])
        disp_labels = disp_labels[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED+1) &
                                  (disp_labels != self.UNLABELED-1)]
        if disp_labels.size == 0:
            print("Pipette: no labeled points found")
            return
        vals, counts = np.unique(disp_labels, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])
        print(f"Pipette (XY): picked label {mode_label} from {disp_labels.size} points")
        self.label_spinbox.setValue(mode_label)
    
    # --------------------------
    # Pipette: pick label at XZ click.
    # --------------------------
    def pick_label_at_xz(self, ev, plot_widget):
        dataPos = plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x = dataPos.x()
        z = dataPos.y()
        r = self.radius_spinbox.value()
        indices = self.kdtree_xz.query_ball_point([x, z], r=r)
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value()/self.scaleFactor
        finit_min_val = finit_center - finit_thickness/2
        finit_max_val = finit_center + finit_thickness/2
        indices = [i for i in indices if self.points[i,1] >= finit_min_val and self.points[i,1] <= finit_max_val]
        indices = np.array(indices)
        if indices.size == 0:
            print("Pipette (XZ): no points found")
            return
        disp_labels = self.labels[indices]
        disp_labels = disp_labels[(disp_labels != self.UNLABELED) &
                                  (disp_labels != self.UNLABELED+1) &
                                  (disp_labels != self.UNLABELED-1)]
        if disp_labels.size == 0:
            print("Pipette (XZ): no labeled points found")
            return
        vals, counts = np.unique(disp_labels, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])
        print(f"Pipette (XZ): picked label {mode_label} from {disp_labels.size} points")
        self.label_spinbox.setValue(mode_label)
    
    # --------------------------
    # Activate pipette mode.
    # --------------------------
    def activate_pipette(self):
        self.pipette_mode = True
        print("Pipette mode activated. Click in the XY or XZ view to pick a label.")
    
    # --------------------------
    # Update Ground Truth.
    # --------------------------
    def update_ground_truth(self):
        ground_truth = self.labels.tolist()
        with open("ground_truth.txt", "w") as f:
            f.write(str(ground_truth))
        print("Ground truth updated. Saved to ground_truth.txt")
    
    # --------------------------
    # Update Labels.
    # --------------------------
    def update_labels(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            any_gt = np.any(gt)
            if not any_gt:
                print("No ground truth labels found.")
                return
            self.solver.set_labels(self.labels, gt)  # if applicable
            print(f"Running winding number solver")
            self.solver.solve_winding_number(num_iterations=500, i_round=-3, seed_node=-1, other_block_factor=15.0, side_fix_nr=-1, display=False)
            calculated_labels = self.solver.get_labels()
            self.calculated_labels = np.array(calculated_labels)
            self.update_views()
            print("Labels updated from solver.")
    
    # --------------------------
    # Save Labeled Graph.
    # --------------------------
    def save_graph(self):
        if self.solver is not None:
            gt = np.abs((self.labels - self.UNLABELED) > 2)
            self.solver.set_labels(self.labels, gt)  # if applicable
            graph_solved_path = self.graph_path.replace("graph.bin", "output_graph.bin")
            self.solver.save_solution(graph_solved_path)
            print("Saved Labeled Graph.")
    
    # --------------------------
    # Undo/Redo functionality.
    # --------------------------
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
    
    # --------------------------
    # Save Labels.
    # --------------------------
    def save_labels(self):
        labeled_data = np.hstack([self.points, self.labels.reshape(-1,1)])
        np.save("labeled_pointcloud.npy", labeled_data)
        print("Labels saved to labeled_pointcloud.npy")
    
    # --------------------------
    # Toggle Calculated Draw Mode.
    # --------------------------
    def toggle_calc_draw_mode(self):
        self.calc_drawing_mode = self.calc_draw_button.isChecked()
        if self.calc_drawing_mode:
            self.calc_draw_button.setText("Update Labels Draw Mode: On")
        else:
            self.calc_draw_button.setText("Update Labels Draw Mode: Off")
    
    # --------------------------
    # Apply All Calculated Labels.
    # --------------------------
    def apply_all_calculated_labels(self):
        mask = (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED)
        if np.any(mask):
            self.labels[mask] = self.calculated_labels[mask]
            self.update_views()
            print(f"Applied calculated labels to {np.sum(mask)} points.")
        else:
            print("No applicable calculated labels found.")

    # --------------------------
    # Clear Calculated Labels.
    # --------------------------
    def clear_calculated_labels(self):
        self.calculated_labels[:] = self.UNLABELED
        self.update_views()
        print("Calculated labels cleared.")
    
    # --------------------------
    # Apply Calculated Labels from XY slice to manual labels.
    # --------------------------
    def apply_calculated_labels_xy(self):
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value()/self.scaleFactor
        z_min_val = z_center - z_thickness/2
        z_max_val = z_center + z_thickness/2
        mask = (self.points[:,2] >= z_min_val) & (self.points[:,2] <= z_max_val)
        # Only apply where manual label is UNLABELED and a calculated label exists.
        indices = np.where(mask & (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED))[0]
        if indices.size:
            self.labels[indices] = self.calculated_labels[indices]
            self.update_views()
            print(f"Applied calculated labels to {indices.size} points in XY slice.")
        else:
            print("No applicable calculated labels found in current XY slice.")
    
    # --------------------------
    # Apply Calculated Labels from XZ slice to manual labels.
    # --------------------------
    def apply_calculated_labels_xz(self):
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value()/self.scaleFactor
        finit_min_val = finit_center - finit_thickness/2
        finit_max_val = finit_center + finit_thickness/2
        mask = (self.points[:,1] >= finit_min_val) & (self.points[:,1] <= finit_max_val)
        indices = np.where(mask & (self.labels == self.UNLABELED) & (self.calculated_labels != self.UNLABELED))[0]
        if indices.size:
            self.labels[indices] = self.calculated_labels[indices]
            self.update_views()
            print(f"Applied calculated labels to {indices.size} points in XZ slice.")
        else:
            print("No applicable calculated labels found in current XZ slice.")
    
    # --------------------------
    # Key events.
    # --------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and not self.s_pressed:
            self.s_pressed = True
            self.original_drawing_mode = self.drawing_mode_checkbox.isChecked()
            self.drawing_mode_checkbox.setChecked(False)
            self.xy_scatter.setAcceptedMouseButtons(Qt.NoButton)
            self.xz_scatter.setAcceptedMouseButtons(Qt.NoButton)
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
# Main: Create GUI without initial data; user loads data via the Data menu.
# --------------------------------------------------
def main():
    app = QApplication(sys.argv)
    gui = PointCloudLabeler()  # no initial point_data; load via Data menu
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
