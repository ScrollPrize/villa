import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QAction, QMessageBox, QGraphicsEllipseItem
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
# Labeling GUI with Undo/Redo, guide overlays, shear (in degrees),
# pipette tool, and a cursor circle indicator.
# --------------------------------------------------
class PointCloudLabeler(QMainWindow):
    def __init__(self, point_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimized Point Cloud Labeler")
        
        # --- Essential Global Variables ---
        self.scaleFactor = 100  # for converting float values to slider integer steps
        self.s_pressed = False
        self.original_drawing_mode = True  # to remember the Drawing Mode checkbox state
        
        # Special constant for unlabeled points.
        self.UNLABELED = -9999
        
        # Undo/Redo stacks (each entry is a copy of the labels array)
        self.undo_stack = []
        self.redo_stack = []
        self._stroke_backup = None  # backup at beginning of a drawing stroke
        
        # --- Create Menu and Help ---
        self._create_menu()
        
        # --- Data and Labels ---
        # Assume the custom library returns points in order: [f_star, f_init, Z]
        self.points = np.array(point_data)  # shape: (N,3)
        self.labels = np.full(len(self.points), self.UNLABELED, dtype=np.int32)
        
        # --- Display Parameters ---
        self.point_size = 3
        self.max_display = 50000  # default maximum points to display
        
        # Coordinate naming:
        # f_star is the first coordinate.
        # f_init (angular) is the second coordinate; its real range is [-180, 180].
        # In the XY view, we extend f_init (vertical) to [-270, 270] for duplicate display.
        self.f_star_min, self.f_star_max = float(np.min(self.points[:, 0])), float(np.max(self.points[:, 0]))
        self.f_init_min, self.f_init_max = -180.0, 180.0  
        self.z_min, self.z_max = float(np.min(self.points[:, 2])), float(np.max(self.points[:, 2]))
        
        # --- KD-trees ---
        self.kdtree_xy = cKDTree(self.points[:, [0, 1]])  # for XY view (f_star, f_init)
        self.kdtree_xz = cKDTree(self.points[:, [0, 2]])  # for XZ view (f_star, Z)
        
        # --- Pre-created Brushes ---
        self.brush_black = pg.mkBrush(0, 0, 0)
        self.brush_red   = pg.mkBrush(255, 0, 0)
        self.brush_green = pg.mkBrush(0, 255, 0)
        self.brush_blue  = pg.mkBrush(0, 0, 255)
        
        # --- Guide Items ---
        # In the XY view, grey dashed lines at f_init = -180 and 180.
        self.line_finit_neg = pg.InfiniteLine(pos=-180, angle=0,
                                              pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_pos = pg.InfiniteLine(pos=180, angle=0,
                                              pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        # f_init slice indicators: now also grey (same as other lines).
        self.line_finit_center = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_upper  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_finit_lower  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        # In the XZ view, grey dashed lines for the Z slice.
        self.line_z_center = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_upper  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        self.line_z_lower  = pg.InfiniteLine(angle=0, pen=pg.mkPen('grey', width=1, style=Qt.DashLine))
        # Shear indicator: now drawn in orange.
        self.shear_indicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('orange', width=1, style=Qt.DashLine))
        
        # --- Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Horizontal layout for two view columns.
        views_columns_layout = QHBoxLayout()
        main_layout.addLayout(views_columns_layout)
        
        # Left Column: XY view and its slice controls (filtered by Z).
        left_column = QVBoxLayout()
        views_columns_layout.addLayout(left_column)
        self.xy_plot = pg.PlotWidget()
        self.xy_plot.setBackground('w')
        self.xy_plot.setLabel('bottom', 'f_star')
        self.xy_plot.setLabel('left', 'f_init')
        self.xy_plot.setYRange(-270, 270)
        self.xy_plot.setMouseEnabled(x=True, y=True)
        left_column.addWidget(self.xy_plot)
        # Add a circle indicator for the drawing radius.
        self.cursor_circle = pg.QtGui.QGraphicsEllipseItem(0, 0, 0, 0)
        self.cursor_circle.setPen(pg.mkPen('cyan', width=1, style=Qt.DashLine))
        self.xy_plot.addItem(self.cursor_circle)
        # XY view slice controls: for filtering by Z.
        xy_controls = QHBoxLayout()
        self.z_center_widget, self.z_center_slider, self.z_center_spinbox = self.create_sync_slider_spinbox(
            "Z slice center:", self.z_min, self.z_max, (self.z_min+self.z_max)/2)
        self.z_thickness_widget, self.z_thickness_slider, self.z_thickness_spinbox = self.create_sync_slider_spinbox(
            "Z slice thickness:", 0.01, self.z_max-self.z_min, (self.z_max-self.z_min)*0.1)
        xy_controls.addWidget(self.z_center_widget)
        xy_controls.addWidget(self.z_thickness_widget)
        left_column.addLayout(xy_controls)
        
        # Right Column: XZ view and its slice controls (filtering by f_init).
        right_column = QVBoxLayout()
        views_columns_layout.addLayout(right_column)
        self.xz_plot = pg.PlotWidget()
        self.xz_plot.setBackground('w')
        self.xz_plot.setLabel('bottom', 'f_star')
        self.xz_plot.setLabel('left', 'Z')
        self.xz_plot.setMouseEnabled(x=True, y=True)
        right_column.addWidget(self.xz_plot)
        # XZ view slice controls: renamed to "f init center" and "f init thickness"
        self.finit_center_widget, self.finit_center_slider, self.finit_center_spinbox = self.create_sync_slider_spinbox(
            "f init center:", float(np.min(self.points[:,1])), float(np.max(self.points[:,1])),
            (np.min(self.points[:,1])+np.max(self.points[:,1]))/2)
        self.finit_thickness_widget, self.finit_thickness_slider, self.finit_thickness_spinbox = self.create_sync_slider_spinbox(
            "f init thickness:", 0.01, float(np.max(self.points[:,1]) - np.min(self.points[:,1])), 5.0)
        xz_controls = QHBoxLayout()
        xz_controls.addWidget(self.finit_center_widget)
        xz_controls.addWidget(self.finit_thickness_widget)
        right_column.addLayout(xz_controls)
        
        # Common Controls row:
        common_controls_layout = QHBoxLayout()
        main_layout.addLayout(common_controls_layout)
        # Drawing radius (1–20, default 10)
        self.radius_widget, self.radius_slider, self.radius_spinbox = self.create_sync_slider_spinbox(
            "Drawing radius:", 1.0, 20.0, 10.0, decimals=0)
        common_controls_layout.addWidget(self.radius_widget)
        # Shear: now in degrees from -90 to 90.
        self.shear_widget, self.shear_slider, self.shear_spinbox = self.create_sync_slider_spinbox(
            "Shear (°):", -90.0, 90.0, 0.0, decimals=1)
        common_controls_layout.addWidget(self.shear_widget)
        # Max Display Points:
        max_disp_layout = QHBoxLayout()
        max_disp_label = QLabel("Max Display Points:")
        self.max_display_spinbox = QSpinBox()
        self.max_display_spinbox.setRange(1000, 1000000)
        self.max_display_spinbox.setValue(self.max_display)
        max_disp_layout.addWidget(max_disp_label)
        max_disp_layout.addWidget(self.max_display_spinbox)
        common_controls_layout.addLayout(max_disp_layout)
        self.max_display_spinbox.valueChanged.connect(self.update_max_display)
        # Drawing Mode toggle:
        self.drawing_mode_checkbox = QCheckBox("Drawing Mode")
        self.drawing_mode_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.drawing_mode_checkbox)
        self.drawing_mode_checkbox.toggled.connect(self.update_drawing_mode)
        # Show Guides toggle:
        self.show_guides_checkbox = QCheckBox("Show guides")
        self.show_guides_checkbox.setChecked(True)
        common_controls_layout.addWidget(self.show_guides_checkbox)
        self.show_guides_checkbox.toggled.connect(self.update_guides)
        # Pipette button:
        self.pipette_button = QPushButton("Pipette")
        self.pipette_button.clicked.connect(self.pick_label)
        common_controls_layout.addWidget(self.pipette_button)
        # Label selection and Save button:
        label_save_layout = QHBoxLayout()
        self.label_spinbox = QSpinBox()
        self.label_spinbox.setRange(-1000, 1000)  # all integers allowed; 0 is valid
        self.label_spinbox.setValue(1)
        label_save_layout.addWidget(QLabel("Label:"))
        label_save_layout.addWidget(self.label_spinbox)
        self.save_button = QPushButton("Save Labels")
        self.save_button.clicked.connect(self.save_labels)
        label_save_layout.addWidget(self.save_button)
        common_controls_layout.addLayout(label_save_layout)
        
        # Create scatter items and add them.
        self.xy_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xz_scatter = pg.ScatterPlotItem(size=self.point_size, pen=None)
        self.xy_plot.addItem(self.xy_scatter)
        self.xz_plot.addItem(self.xz_scatter)
        self.xy_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        self.xz_scatter.setAcceptedMouseButtons(Qt.LeftButton)
        
        # Enable pencil (drawing) tool on both views.
        self._enable_pencil(self.xy_plot, self.xy_scatter, view_name='xy')
        self._enable_pencil(self.xz_plot, self.xz_scatter, view_name='xz')
        
        # Install event filter on the XY view's scene to update the cursor circle.
        self.xy_plot.scene().installEventFilter(self)
        
        # Initially update guides and views.
        self.update_guides()
        self.update_views()
    
    # --------------------------
    # Event filter for cursor circle in the XY view.
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
    # Menu creation and Help.
    # --------------------------
    def _create_menu(self):
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        usage_action = QAction("Usage", self)
        usage_action.triggered.connect(self.show_help)
        help_menu.addAction(usage_action)
    
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
            "   - Drawing radius (range 1–20, default 10).\n"
            "   - Shear (in degrees, range -90 to 90): in the XZ view, f_star' = f_star + tan(shear°)*(f_init - f init center).\n"
            "     The shear indicator in the XY view shows this angle (in orange).\n"
            "   - Max Display Points for performance.\n"
            "   - Toggle Drawing Mode; press and hold 'S' to temporarily disable drawing mode.\n"
            "   - Toggle 'Show guides' to show/hide overlay indicator lines.\n"
            "   - Pipette: click the Pipette button to sample the labels within the drawing radius at the current cursor position\n"
            "     (only non‑black points are considered) and update the current label to the most common value.\n\n"
            "4. Drawing (in the XY view):\n"
            "   - If you draw in the real region (f_init between -180 and 180), the underlying point gets the chosen label.\n"
            "   - If you draw in the virtual region (f_init > 180), the underlying real point gets label = (chosen label) + 1,\n"
            "     while its duplicate displays the chosen label.\n"
            "   - If you draw in the virtual region (f_init < -180), the underlying real point gets label = (chosen label) - 1,\n"
            "     while its duplicate displays the chosen label.\n"
            "   - Points with label equal to UNLABELED or UNLABELED±1 display as black.\n\n"
            "5. Undo/Redo:\n"
            "   - Press Ctrl+Z to undo the last drawing stroke; Ctrl+Y to redo.\n\n"
            "6. Saving:\n"
            "   - Click 'Save Labels' to export the labeled point cloud as a .npy file.\n"
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
        spinbox.setSingleStep((max_val - min_val) / 100.0)
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
    # Guide overlays.
    # --------------------------
    def update_guides(self):
        # --- In the XY view ---
        if self.show_guides_checkbox.isChecked():
            try:
                self.xy_plot.addItem(self.line_finit_neg)
                self.xy_plot.addItem(self.line_finit_pos)
            except Exception:
                pass
            # Update f init slice indicators based on the "f init center" and "f init thickness" controls.
            finit_center = self.finit_center_spinbox.value()
            finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
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
            # Update shear indicator in the XY view: use the shear angle from the slider.
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
        
        # --- In the XZ view ---
        if self.show_guides_checkbox.isChecked():
            z_center = self.z_center_spinbox.value()
            z_thickness = self.z_thickness_slider.value() / self.scaleFactor
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
    def downsample_points(self, pts, labels, max_display):
        n = pts.shape[0]
        if n > max_display:
            indices = np.linspace(0, n - 1, max_display, dtype=int)
            return pts[indices], labels[indices]
        else:
            return pts, labels
    
    # --------------------------
    # Brush lookup.
    # --------------------------
    def get_brushes_from_labels(self, labels_array):
        # For the XY view:
        # - If a point's stored label is UNLABELED or UNLABELED±1, display black.
        # - Otherwise, use label mod 3:
        #     0 -> red, 1 -> green, 2 -> blue.
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
    # Enable pencil tool on a PlotWidget.
    # --------------------------
    def _enable_pencil(self, plot_widget, scatter_item, view_name='xy'):
        scatter_item.mousePressEvent = lambda ev: self._on_mouse_press(ev, plot_widget, view_name)
        scatter_item.mouseMoveEvent  = lambda ev: self._on_mouse_drag(ev, plot_widget, view_name)
        scatter_item.mouseReleaseEvent = lambda ev: self._on_mouse_release(ev, plot_widget, view_name)
    
    def _on_mouse_press(self, ev, plot_widget, view_name):
        if (ev.button() == Qt.LeftButton and 
            self.drawing_mode_checkbox.isChecked() and 
            not self.s_pressed):
            ev.accept()
            if self._stroke_backup is None:
                self._stroke_backup = self.labels.copy()
            self._paint_points(ev, plot_widget, view_name)
        else:
            ev.ignore()
    
    def _on_mouse_drag(self, ev, plot_widget, view_name):
        if (ev.buttons() & Qt.LeftButton and 
            self.drawing_mode_checkbox.isChecked() and 
            not self.s_pressed):
            ev.accept()
            self._paint_points(ev, plot_widget, view_name)
        else:
            ev.ignore()
    
    def _on_mouse_release(self, ev, plot_widget, view_name):
        if ev.button() == Qt.LeftButton and self._stroke_backup is not None:
            self.undo_stack.append(self._stroke_backup)
            self.redo_stack = []  # clear redo stack
            self._stroke_backup = None
        ev.accept()
    
    # --------------------------
    # Drawing: update labels.
    # In the XY view, if the drawn f_init (vertical) is outside [-180,180],
    # update the underlying real point with an offset.
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
            # Ensure that if update_label equals UNLABELED±1, it becomes UNLABELED.
            if update_label == self.UNLABELED+1 or update_label == self.UNLABELED-1:
                update_label = self.UNLABELED
            indices = self.kdtree_xy.query_ball_point([x_m, effective_y], r=r)
        elif view_name == 'xz':
            indices = self.kdtree_xz.query_ball_point([x_m, y_m], r=r)
        else:
            indices = []
        if indices:
            self.labels[indices] = update_label
        self.update_views()
    
    # --------------------------
    # Update the views.
    # --------------------------
    def update_views(self):
        # --- XY View: Filter by Z slice ---
        z_center = self.z_center_spinbox.value()
        z_thickness = self.z_thickness_slider.value() / self.scaleFactor
        z_min_val = z_center - z_thickness/2
        z_max_val = z_center + z_thickness/2
        mask_xy = (self.points[:, 2] >= z_min_val) & (self.points[:, 2] <= z_max_val)
        pts_xy = self.points[mask_xy]
        labels_xy = self.labels[mask_xy]
        # In the XY view: x = f_star; y = f_init.
        # Create duplicate points for wrap-around:
        mask_top = pts_xy[:, 1] < -90
        pts_top = pts_xy[mask_top].copy()
        if pts_top.size:
            pts_top[:, 1] += 360
        labels_top = labels_xy[mask_top] - 1  # When duplicating, adjust displayed label
        mask_bottom = pts_xy[:, 1] > 90
        pts_bottom = pts_xy[mask_bottom].copy()
        if pts_bottom.size:
            pts_bottom[:, 1] -= 360
        labels_bottom = labels_xy[mask_bottom] + 1  # adjust displayed label
        pts_combined = np.concatenate([pts_xy, pts_top, pts_bottom], axis=0)
        labels_combined = np.concatenate([labels_xy, labels_top, labels_bottom], axis=0)
        pts_combined, labels_combined = self.downsample_points(pts_combined, labels_combined, self.max_display)
        brushes_xy = self.get_brushes_from_labels(labels_combined)
        self.xy_scatter.setData(x=pts_combined[:, 0], y=pts_combined[:, 1],
                                size=self.point_size, pen=None, brush=brushes_xy)
        
        # --- XZ View: Filter by f_init slice (from right–column controls) ---
        finit_center = self.finit_center_spinbox.value()
        finit_thickness = self.finit_thickness_slider.value() / self.scaleFactor
        finit_min_val = finit_center - finit_thickness/2
        finit_max_val = finit_center + finit_thickness/2
        mask_xz = (self.points[:, 1] >= finit_min_val) & (self.points[:, 1] <= finit_max_val)
        pts_xz = self.points[mask_xz]
        labels_xz = self.labels[mask_xz]
        # In the XZ view, apply shear transformation:
        # Shear slider now gives an angle in degrees; convert to radians then compute tan(angle).
        shear_angle_deg = self.shear_spinbox.value()
        shear_factor = np.tan(np.radians(shear_angle_deg))
        if pts_xz.size:
            x_new = pts_xz[:, 0] + shear_factor * (pts_xz[:, 1] - finit_center)
        else:
            x_new = pts_xz[:, 0]
        pts_xz_display = pts_xz.copy()
        pts_xz_display[:, 0] = x_new
        pts_xz_display, labels_xz = self.downsample_points(pts_xz_display, labels_xz, self.max_display)
        brushes_xz = self.get_brushes_from_labels(labels_xz)
        self.xz_scatter.setData(x=pts_xz_display[:, 0], y=pts_xz_display[:, 2],
                                size=self.point_size, pen=None, brush=brushes_xz)
        
        # Update guide overlays.
        self.update_guides()
    
    # --------------------------
    # Pipette tool: when the Pipette button is pressed, sample the labels within the drawing radius
    # from the current cursor position in the XY view and set the label spinbox to the most common label.
    # --------------------------
    def pick_label(self):
        # Get the current global cursor position and map to the XY view.
        pos_global = pg.QtGui.QCursor.pos()
        view = self.xy_plot.scene().views()[0]
        pos_widget = view.mapFromGlobal(pos_global)
        scenePos = self.xy_plot.plotItem.vb.mapToScene(pos_widget)
        dataPos = self.xy_plot.plotItem.vb.mapSceneToView(scenePos)
        x = dataPos.x()
        y = dataPos.y()
        # Use the same drawing radius.
        r = self.radius_spinbox.value()
        # Wrap the y coordinate for querying:
        if y > 180:
            effective_y = y - 360
        elif y < -180:
            effective_y = y + 360
        else:
            effective_y = y
        indices = self.kdtree_xy.query_ball_point([x, effective_y], r=r)
        if len(indices) == 0:
            return
        picked = self.labels[indices]
        # Only consider labeled points (ignore UNLABELED and UNLABELED±1).
        picked = picked[(picked != self.UNLABELED) & (picked != self.UNLABELED+1) & (picked != self.UNLABELED-1)]
        if len(picked) == 0:
            return
        # Pick the mode (most frequent label).
        vals, counts = np.unique(picked, return_counts=True)
        mode_label = int(vals[np.argmax(counts)])
        self.label_spinbox.setValue(mode_label)
    
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
    # Save labeled point cloud.
    # --------------------------
    def save_labels(self):
        labeled_data = np.hstack([self.points, self.labels.reshape(-1, 1)])
        np.save("labeled_pointcloud.npy", labeled_data)
        print("Labels saved to labeled_pointcloud.npy")
    
    # --------------------------
    # Key events: S to temporarily disable drawing; Ctrl+Z/Ctrl+Y for undo/redo.
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
        elif event.key() == Qt.Key_P:  # Alternatively, allow pipette via 'P'
            self.pick_label()
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
    
    # --------------------------
    # Event filter to update the cursor circle in the XY view.
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

# --------------------------------------------------
# Main: Load graph and point cloud using your custom library.
# --------------------------------------------------
def main():
    app = QApplication(sys.argv)
    
    solver = graph_problem_gpu_py.Solver(
        "/media/julian/2/Scroll5/scroll5_complete_surface_points_zarrtest/1352_3600_5005/graph.bin",
        z_min=3000, z_max=4000
    )
    experiment_name = "denominator3-rotated"
    intermediate_ring_solution_save_path2 = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_solver_connected_2.bin"
    solver.load_graph(intermediate_ring_solution_save_path2)
    point_data = solver.get_positions()
    
    gui = PointCloudLabeler(point_data)
    gui.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
