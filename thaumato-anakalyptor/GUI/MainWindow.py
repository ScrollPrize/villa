### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QMainWindow, QAction, QSplitter, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QFrame,
                             QFileDialog, QLineEdit, QCheckBox, QMessageBox, QStyle, QVBoxLayout, QScrollArea, QHBoxLayout, QGraphicsScene, QGraphicsView)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QKeyEvent, QIcon
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItemGroup

import tifffile

# Custom GUI Elements
from .CollapsibleBox import CollapsibleBox
from .UmbilicusWindow import UmbilicusWindow
from .ConfigWindow import ConfigWindow

import os

from tqdm import tqdm
import time
import json
import threading
import multiprocessing
# multiprocessing.set_start_method('spawn')
import subprocess

import signal
import numpy as np
import open3d as o3d
import zarr 
import tarfile
import py7zr
import tempfile

from ThaumatoAnakalyptor.sheet_to_mesh import umbilicus_xy_at_z

class GraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)

    def wheelEvent(self, event):
        if not event.modifiers() == Qt.ControlModifier:
            # send to parent if Ctrl is not pressed
            super().wheelEvent(event)
            return
        # Ctrl + Wheel to zoom
        factor = 1.1  # Zoom factor
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)


class ThaumatoAnakalyptor(QMainWindow):
    # Signals for computation completion
    gridCellsComputationDone = pyqtSignal()
    pointcloudComputationDone = pyqtSignal()
    instancesComputationDone = pyqtSignal()
    stitchSheetComputationDone = pyqtSignal()
    meshingComputationDone = pyqtSignal()
    swapVolumeComputationDone = pyqtSignal()
    ppmComputationDone = pyqtSignal()
    texturingComputationDone = pyqtSignal()
    inkDetectionComputationDone = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.process = None
        self.process_grid_cells = None
        self.process_pointcloud = None
        self.process_instances = None
        self.process_stitching = None
        self.process_meshing = None
        self.process_swap_volume = None
        self.process_ppm = None
        self.process_texturing = None
        self.process_ink_detection = None
        
        self.isSelectingStartingPoint = False
        self.display_points = False
        self.display_instances_points = False
        self.points = []
        self.points_pc = None
        self.points_instances_pc = None
        self.loadConfig()
        self.initUI()
        # set icon
        icon = QIcon("GUI/ThaumatoAnakalyptor.png")
        self.setWindowIcon(icon)
        # Set minimum window size
        self.setMinimumSize(800, 600)

    def initUI(self):
        self.setWindowTitle('ThaumatoAnakalyptor')

        # Menu Bar
        menubar = self.menuBar()
        config = QAction('Config', self)
        helpMenu = QAction('Help', self)
        helpMenu.triggered.connect(self.showHelp)  # Connect the triggered signal to showHelp method
        menubar.addAction(config)
        menubar.addAction(helpMenu)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left Panel for Image
        left_panel = QLabel()
        left_panel.setFrameStyle(QFrame.StyledPanel)

        # Check and load TIFF files
        self.tifImages = self.loadImages(self.Config.get("original_2d_tiffs", ""))
        self.zarr_volume = None
        self.currentTifIndex = 3100
        self.image_array = None
        self.old_index = -1

        # Setup left panel with QGraphicsView
        self.tifScene = QGraphicsScene(self)
        self.tifView = GraphicsView(self.tifScene)
        self.tifView.setRenderHint(QPainter.Antialiasing)
        self.tifView.mousePressEvent = self.onTifMousePress

        left_panel_layout = QVBoxLayout()
        left_panel_layout.addWidget(self.tifView)
        self.setupTifNavigation(left_panel_layout)

        left_panel.setLayout(left_panel_layout)

        # Load the first TIFF image
        self.loadSlice(self.currentTifIndex)
        self.tifView.fitInView(self.tifScene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # Right Panel setup with a Scroll Area
        right_panel_scroll_area = QScrollArea()  # Create a QScrollArea
        right_panel_scroll_area.setWidgetResizable(True)  # Make the scroll area resizable
        right_panel = QWidget()  # This will be the scrollable content
        right_layout = QVBoxLayout(right_panel)  # Use right_panel as the parent for the layout
        right_layout.setAlignment(Qt.AlignTop)  # Align sections at the top
        right_panel_scroll_area.setWidget(right_panel)  # Set the widget you want to scroll

        # Collapsible Sections
        self.addVolumeProcessing(right_layout)
        self.addMeshGeneration(right_layout)
        self.addRendering(right_layout)
        self.addInkDetection(right_layout)

        right_panel.setLayout(right_layout)

        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel_scroll_area)
        # Set splitter ratio
        splitter.setSizes([400, 200])

        # Trigger click on Config
        config.triggered.connect(self.openConfigWindow)


    def loadImages(self, path):
        if path and os.path.exists(path):
            if path.endswith(".zarr"):
                # ome-zarr volume
                self.zarr_volume = zarr.open(path)
                self.zarr_volume = self.zarr_volume[0]
                return [*range(self.zarr_volume.shape[0])]
            else:
                return sorted([f for f in os.listdir(path) if f.endswith('.tif')])
        return []

    def setupTifNavigation(self, layout):
        navigationLayout = QHBoxLayout()

        # Add stretch to align content to the right
        navigationLayout.addStretch()

        # Step Size
        navigationLayout.addWidget(QLabel("Step Size:"))
        self.stepSizeBox = QLineEdit("100")
        self.stepSizeBox.setFixedWidth(50)  # Adjust width as needed
        navigationLayout.addWidget(self.stepSizeBox)

        # Index
        navigationLayout.addWidget(QLabel("Index:"))
        self.indexBox = QLineEdit("0")
        self.indexBox.setFixedWidth(50)  # Adjust width as needed
        self.indexBox.returnPressed.connect(self.jumpToTifIndex)
        navigationLayout.addWidget(self.indexBox)

        # Checkbox Display Points
        self.displayPointsCheckbox = QCheckBox("Display Points")
        self.displayPointsCheckbox.stateChanged.connect(self.toggleDisplayPoints)
        navigationLayout.addWidget(self.displayPointsCheckbox)

        # Checkbox Display Instances Points
        self.displayInstancesPointsCheckbox = QCheckBox("Display Instances Points")
        self.displayInstancesPointsCheckbox.stateChanged.connect(self.toggleDisplayInstancesPoints)
        navigationLayout.addWidget(self.displayInstancesPointsCheckbox)

        # Add the navigation layout to the main layout
        layout.addLayout(navigationLayout)

    def loadSlice(self, index):
        umbilicus_path = os.path.join(self.Config.get("downsampled_2d_tiffs", ""), "umbilicus.txt")
        if os.path.exists(umbilicus_path):
            self.points = []
            if os.path.exists(umbilicus_path):
                with open(umbilicus_path, "r") as file:
                    for line in file:
                        y, z, x = map(int, line.strip().split(', '))
                        self.points.append((x-500, y-500, z-500))

                self.points = np.array(self.points)

        if 0 <= index < len(self.tifImages):
            if self.old_index != index:
                volume = self.Config.get("original_2d_tiffs", None)
                if volume is not None and volume.endswith(".zarr"):
                    image_array = self.loadZarrImage(index)
                else:
                    image_array = self.loadTifImage(index)
                self.image_array = image_array
            else:
                image_array = self.image_array


            # print(f"Loaded image {imagePath} at index {index} with shape {image_array.shape} and dtype {image_array.dtype}")

            if image_array.dtype == np.uint16:
                image_array = (image_array / 256).astype(np.uint8)

            # Assuming the image is grayscale, prepare it for display
            image_height, image_width = image_array.shape

            self.image_width = image_width
            self.image_height = image_height

            qimage = QImage(image_array.data, image_width, image_height, image_width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)

            # Clear the previous items in the scene
            self.tifScene.clear()

            # Add new pixmap item to the scene
            self.tifScene.addPixmap(pixmap)
            # self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            # Set the index box text
            self.indexBox.setText(str(index))

            # Draw a red point if it exists for this image
            if len(self.points) > 0:
                point_np = umbilicus_xy_at_z(self.points, np.array([index]))
                x, y = point_np[0][0], point_np[0][1]
                size_display = 10
                # Size of point in image coordinates
                size_image = size_display / self.tifView.transform().m11()
                self.tifScene.addEllipse(x - size_image / 2, y - size_image / 2, size_image, size_image, QPen(Qt.red), QBrush(Qt.red))

                # Draw a cone from the point to the bottom of the image with cone angle 45 degrees
                # calculate bottom x point (just bottom of the image)
                cone_line_left_y = self.image_height
                cone_line_right_y = self.image_height
                cone_line_left_x = x + (cone_line_left_y - y) * np.tan(np.deg2rad(45))
                cone_line_right_x = x - (cone_line_right_y - y) * np.tan(np.deg2rad(45))
                self.tifScene.addLine(x, y, cone_line_left_x, cone_line_left_y, QPen(Qt.green))
                self.tifScene.addLine(x, y, cone_line_right_x, cone_line_right_y, QPen(Qt.green))
                self.tifScene.addLine(cone_line_left_x, cone_line_left_y, cone_line_right_x, cone_line_right_y, QPen(Qt.green))

            # Draw green starting point on the z index image
            if hasattr(self, "xField") and hasattr(self, "yField") and hasattr(self, "zField") and self.zField.text() == str(index):
                x = int(self.xField.text())
                y = int(self.yField.text())
                # Size of point unchanged in display no matter the zoom
                size_display = 10
                # Size of point in image coordinates
                size_image = size_display / self.tifView.transform().m11()
                self.tifScene.addEllipse(x - size_image / 2, y - size_image / 2, size_image, size_image, QPen(Qt.green), QBrush(Qt.green))

            if self.display_points: 
                # Draw Pointclouds from .ply files
                self.draw_pointcloud(index)
            if self.display_instances_points:
                self.draw_instances_pointcloud(index)
            
            self.old_index = index
        else:
            print(f"Index {index} out of range")
            # white placeholder image
            pixmap = QPixmap(400, 400)
            pixmap.fill(Qt.white)  # White background
            self.tifScene.clear()
            self.tifScene.addPixmap(pixmap)

    def loadZarrImage(self, index):
        if self.zarr_volume is None:
            # ome-zarr volume
            self.zarr_volume = zarr.open(self.Config.get("original_2d_tiffs", "r"))
            self.zarr_volume = self.zarr_volume[0]

        image_array = self.zarr_volume[index]
        return image_array

    def loadTifImage(self, index):
        imagePath = os.path.join(self.Config.get("downsampled_2d_tiffs", ""), self.tifImages[index])

        # Use tifffile to read the TIFF image
        with tifffile.TiffFile(imagePath) as tif:
            image_array = tif.asarray()
        
        return image_array

    def draw_pointcloud(self, index):
        if index != self.old_index or self.points_pc is None:
            # Find all .ply files in the directory
            pointcloud_path = os.path.join(self.Config["surface_points_path"], "point_cloud_verso")
            ply_files = [f for f in os.listdir(pointcloud_path) if f.endswith('.ply')]
            adjusted_tiff_index = 500 + self.currentTifIndex
            z_index_cell = adjusted_tiff_index // 200
            # pad to 6 digits
            z_index_cell_string = str(z_index_cell).zfill(3)
            # filter all cell_yxz_*_{z_index_cell_string}_*.ply files
            ply_files = [f for f in ply_files if f.split("_")[4] == z_index_cell_string+".ply"]
            self.points_pc = []
            nr_valid_points = 0
            for ply_file in tqdm(ply_files, desc="Loading .ply files"):
                # load the ply files
                pcd = o3d.io.read_point_cloud(os.path.join(pointcloud_path, ply_file))
                points_pcd = np.array(pcd.points)
                # find the points in the cell that are +- 1 in z away from the current z index
                valid_points = points_pcd[np.where(np.abs(points_pcd[:, 1] - adjusted_tiff_index) <= 1)]
                nr_valid_points += len(valid_points)
                self.points_pc.append(valid_points)

        # Optimize Drawing
        graphics_group = QGraphicsItemGroup()
        for pointcloud in tqdm(self.points_pc, desc="Drawing Points"):
            for point in pointcloud:
                y, z, x = point
                ellipse = QGraphicsEllipseItem(x-500, y-500, 2, 2)
                ellipse.setPen(QPen(Qt.blue))  # Set outline color
                ellipse.setBrush(QBrush(Qt.blue))  # Set fill color
                graphics_group.addToGroup(ellipse)

        self.tifScene.addItem(graphics_group)

    def draw_instances_pointcloud(self, index):
        if index != self.old_index or self.points_instances_pc is None:
            # Find all .ply files in the directory
            pointcloud_path = os.path.join(self.Config["surface_points_path"], "point_cloud_colorized_verso_subvolume_blocks")
            tar_files = [f for f in os.listdir(pointcloud_path) if f.endswith('.tar') or f.endswith('.7z')]
            print(f"Found {len(tar_files)} tar/7z files")
            adjusted_tiff_index = 500 + self.currentTifIndex
            tar_files = [f for f in tar_files if (s_i := (int(f.split("_")[1]) * 4)) >= adjusted_tiff_index and s_i <= adjusted_tiff_index + 200]
            print(f"Found {len(tar_files)} tar files")
            self.points_instances_pc = []
            nr_valid_points = 0
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_ply_files = []

                for tar_file in tqdm(tar_files, desc="Extracting .tar/.7z files"):
                    tar_path = os.path.join(pointcloud_path, tar_file)
                    tar_subdir = os.path.join(temp_dir, os.path.splitext(tar_file)[0])  # Unique subdir for each archive
                    os.makedirs(tar_subdir, exist_ok=True)

                    try:
                        if tar_path.endswith('.tar'):
                            with tarfile.open(tar_path, 'r') as archive:
                                ply_members = [m for m in archive.getmembers() if m.name.endswith('.ply')]
                                archive.extractall(path=tar_subdir, members=ply_members)
                                extracted_ply_files.extend([os.path.join(tar_subdir, m.name) for m in ply_members])
                        elif tar_path.endswith('.7z'):
                            with py7zr.SevenZipFile(tar_path, 'r') as archive:
                                archive.extractall(tar_subdir)  # Extract directly into the subdir
                                extracted_ply_files.extend([os.path.join(tar_subdir, f) for f in os.listdir(tar_subdir) if f.endswith('.ply')])
                    except Exception as e:
                        print(f"Failed to extract {tar_path}: {e}")
                    
                for ply_file in tqdm(extracted_ply_files, desc="Loading .ply files"):
                    pcd = o3d.io.read_point_cloud(ply_file)
                    points_pcd = np.asarray(pcd.points) * 4
                    valid_points = points_pcd[np.abs(points_pcd[:, 1] - adjusted_tiff_index) <= 1]
                    nr_valid_points += len(valid_points)
                    self.points_instances_pc.append(valid_points)

        # Optimize Drawing
        graphics_group = QGraphicsItemGroup()
        for pointcloud in tqdm(self.points_instances_pc, desc="Drawing Points"):
            for point in pointcloud:
                y, z, x = point
                ellipse = QGraphicsEllipseItem(x-500, y-500, 2, 2)
                ellipse.setPen(QPen(Qt.red))  # Set outline color
                ellipse.setBrush(QBrush(Qt.red))  # Set fill color
                graphics_group.addToGroup(ellipse)

        self.tifScene.addItem(graphics_group)

    def jumpToTifIndex(self):
        index = int(self.indexBox.text())
        self.loadSlice(index)

    def toggleDisplayPoints(self):
        self.display_points = self.displayPointsCheckbox.isChecked()
        self.loadSlice(self.currentTifIndex)

    def toggleDisplayInstancesPoints(self):
        self.display_instances_points = self.displayInstancesPointsCheckbox.isChecked()
        self.loadSlice(self.currentTifIndex)

    def onTifMousePress(self, event):
        # print(f"Clicked position in image coordinates: ({int(scenePos.x())}, {int(scenePos.y())}, {self.currentTifIndex})")
        # Set starting point if in selection mode
        if self.isSelectingStartingPoint:
            scenePos = self.tifView.mapToScene(event.pos())
            
            self.xField.setText(str(int(scenePos.x())))
            self.yField.setText(str(int(scenePos.y())))
            self.zField.setText(str(self.currentTifIndex))

            self.loadSlice(self.currentTifIndex)

    def incrementIndex(self):
        self.incrementing = True
        step_size = int(self.stepSizeBox.text())
        self.currentTifIndex = min((self.currentTifIndex + step_size), len(self.tifImages) - 1)

    def decrementIndex(self):
        self.incrementing = False
        step_size = int(self.stepSizeBox.text())
        self.currentTifIndex = max((self.currentTifIndex - step_size), 0)

    def keyPressEvent(self, event: QKeyEvent):
        # "A" "D" keys for navigation
        if event.key() == Qt.Key_D:
            self.incrementIndex()
        elif event.key() == Qt.Key_A:
            self.decrementIndex()
        self.loadSlice(self.currentTifIndex)

    def openConfigWindow(self):
        self.configWindow = ConfigWindow(self)
        self.configWindow.show()

    def loadConfig(self):
        try:
            with open("config.json", "r") as file:
                self.Config = json.load(file)
                # Restore the state of UI components using self.Config
        except FileNotFoundError:
            self.Config = {}

    def showHelp(self):
        helpText = "ThaumatoAnakalyptor Help\n\n" \
                   "There are the following shortcuts:\n\n" \
                   "- Use 'A' and 'D' to switch between TIFF layers.\n" \
                   "- Click on the TIFF to place a point.\n" \
                   "If you run into out-of-memory issues on your GPU, try to reduce the number of threads and the batch size." 
        QMessageBox.information(self, "Help", helpText)

    def addCollapsibleSection(self, layout, title):
        box = CollapsibleBox(title)
        button1 = QPushButton('Button 1')
        button1.clicked.connect(self.printHi)  # Connect the button click to printHi method

        button2 = QPushButton('Button 2')
        button2.clicked.connect(self.doTQDM)

        button3 = QPushButton('Button 3')

        box.add_widget(button1)
        box.add_widget(button2)
        box.add_widget(button3)

        layout.addWidget(box)

    def printHi(self):
        print(self.Config)

    def doTQDM(self):
        for i in tqdm(range(10)):
            time.sleep(1)

    def addVolumeProcessing(self, layout):
        # Main Collapsible Box for Volume Preprocessing
        volumeBox = CollapsibleBox("Volume Preprocessing")

        # Generate Grid Cells Area
        generateGridBox = CollapsibleBox("Generate Grid Cells")
        self.addGenerateGridCellsArea(generateGridBox)
        volumeBox.add_widget(generateGridBox)

        # Umbilicus Area
        umbilicusBox = CollapsibleBox("Umbilicus")
        self.addUmbilicusArea(umbilicusBox)
        volumeBox.add_widget(umbilicusBox)

        # Pointcloud Area
        pointcloudBox = CollapsibleBox("Pointcloud")
        self.addPointcloudArea(pointcloudBox)
        volumeBox.add_widget(pointcloudBox)

        # Instances Area
        instancesBox = CollapsibleBox("Instances")
        self.addInstancesArea(instancesBox)
        volumeBox.add_widget(instancesBox)


        layout.addWidget(volumeBox)

    def addUmbilicusArea(self, box):
        label = QLabel("Umbilicus")
        # Create an info button
        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(self.showUmbilicusInfo)        

        button = QPushButton("Generate")
        button.clicked.connect(self.openUmbilicusWindow)
        box.add_widget(label)
        box.add_widget(infoButton)
        box.add_widget(button)

    def showUmbilicusInfo(self):
        QMessageBox.information(self, "Umbilicus Information",
                                "If you already have an umbilicus generated, load it with 'Load'. Place the umbilicus points in the center of the scroll. Save your work with 'Save' before closing. Make sure to have the appropriate paths set in the Config.")

    def openUmbilicusWindow(self):
        if self.Config.get("downsampled_2d_tiffs", None) and os.path.exists(self.Config["downsampled_2d_tiffs"]):
            self.umbilicusWindow = UmbilicusWindow(self.Config["downsampled_2d_tiffs"], abs(self.Config["downsample_factor"]))
            self.umbilicusWindow.resize(self.width(), self.height())  # Adjust width and height as needed
            self.umbilicusWindow.show()

    def addGenerateGridCellsArea(self, box):
        label = QLabel("Generate Grid Cells")

        self.computeGridCellsButton = QPushButton("Compute")
        self.stopGridCellsButton = QPushButton("Stop")
        self.stopGridCellsButton.setEnabled(False)

        self.computeGridCellsButton.clicked.connect(self.computeGridCells)
        self.stopGridCellsButton.clicked.connect(self.stopGridCells)

        box.add_widget(label)
        box.add_widget(self.computeGridCellsButton)
        box.add_widget(self.stopGridCellsButton)

        # Connect the signal to the slot method
        self.gridCellsComputationDone.connect(self.onGridCellsComputationDone)

    def computeGridCells(self):
        command = [
                "python3", "-m", "ThaumatoAnakalyptor.generate_half_sized_grid", 
                "--input_directory", str(self.Config["original_2d_tiffs"]), 
                "--output_directory", str(self.Config["downsampled_2d_tiffs"]), 
                "--downsample_factor", str(abs(self.Config["downsample_factor"]))
            ]
        
        def run():
            self.process_grid_cells = subprocess.Popen(command)
            self.process_grid_cells.wait()  # Wait for the process to complete
            print("Done Waiting")
            self.gridCellsComputationDone.emit()  # Emit the signal on completion
    
        # Start the monitoring thread
        thread = threading.Thread(target=run)

        thread.start()
        self.computeGridCellsButton.setEnabled(False)
        self.stopGridCellsButton.setEnabled(True)

    def stopGridCells(self):
        if self.process_grid_cells and self.process_grid_cells.poll() is None:  # Check if process is running
            print("Stopping process...")
            self.process_grid_cells.kill()  # or self.process.kill() for a more forceful termination
            self.process_grid_cells = None
            self.gridCellsComputationDone.emit()  # Update UI immediately
        else:
            print("No process to stop")

    def onGridCellsComputationDone(self):
        self.computeGridCellsButton.setEnabled(True)
        self.stopGridCellsButton.setEnabled(False)
        print("Computation process stopped.")

    def addPointcloudArea(self, box):
        label = QLabel("Pointcloud")

        self.recomputeCheckbox = QCheckBox("Recompute")
        self.computePointcloudButton = QPushButton("Compute")
        self.stopPointcloudButton = QPushButton("Stop")
        self.stopPointcloudButton.setEnabled(False)

        self.computePointcloudButton.clicked.connect(self.computePointcloud)
        self.stopPointcloudButton.clicked.connect(self.stopPointcloud)

        box.add_widget(label)
        box.add_widget(self.recomputeCheckbox)
        box.add_widget(self.computePointcloudButton)
        box.add_widget(self.stopPointcloudButton)

        # Connect the signal to the slot method
        self.pointcloudComputationDone.connect(self.onPointcloudComputationDone)

    def computePointcloud(self):
        try:
            command = [
                "python3", "-m", "ThaumatoAnakalyptor.grid_to_pointcloud", 
                "--base_path", "", 
                "--volume_subpath", self.Config["downsampled_3d_grids"], 
                "--disk_load_save", "", "", 
                "--pointcloud_subpath", os.path.join(self.Config["surface_points_path"], "point_cloud"), 
                "--num_threads", str(self.Config["num_threads"]), 
                "--gpus", str(self.Config["gpus"])
            ]
            
            if self.recomputeCheckbox.isChecked():
                command += ["--recompute"]

            def run():
                self.process_pointcloud = subprocess.Popen(command)
                self.process_pointcloud.wait()  # Wait for the process to complete
                self.pointcloudComputationDone.emit()  # Emit the signal on completion
        
            # Start the monitoring thread
            thread = threading.Thread(target=run)

            thread.start()
            self.computePointcloudButton.setEnabled(False)
            self.stopPointcloudButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computePointcloudButton.setEnabled(True)
            self.stopPointcloudButton.setEnabled(False)

    def stopPointcloud(self):
        if self.process_pointcloud and self.process_pointcloud.poll() is None:  # Check if process is running
            self.process_pointcloud.kill()  # or self.process.kill() for a more forceful termination
            self.process_pointcloud = None
            self.pointcloudComputationDone.emit()  # Update UI immediately

    def onPointcloudComputationDone(self):
        self.computePointcloudButton.setEnabled(True)
        self.stopPointcloudButton.setEnabled(False)
        print("Computation process stopped.")

    def addInstancesArea(self, box):
        label = QLabel("Instances")
        # self.addFieldWithLabel(box, "Batch Size:", "Enter Batch Size", "batchSizeField", value=4)
        self.computeInstancesButton = QPushButton("Compute")
        self.stopInstancesButton = QPushButton("Stop")
        self.stopInstancesButton.setEnabled(False)

        self.computeInstancesButton.clicked.connect(self.computeInstances)
        self.stopInstancesButton.clicked.connect(self.stopInstances)

        box.add_widget(label)
        box.add_widget(self.computeInstancesButton)
        box.add_widget(self.stopInstancesButton)

        # Connect the signal to the slot method
        self.pointcloudComputationDone.connect(self.onPointcloudComputationDone)

    def computeInstances(self):
        try:
            batch_size = int(self.Config["batch_size"])

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.pointcloud_to_instances", 
                "--path", self.Config["surface_points_path"], 
                "--dest", self.Config["surface_points_path"], 
                "--umbilicus_path", self.Config["umbilicus_path"], 
                "--main_drive", "",
                "--alternative_ply_drives", "", "", 
                "--max_umbilicus_dist", "-1",
                "--batch_size", str(batch_size),
                "--gpus", str(self.Config["gpus"])
            ]

            def run():
                self.process_instances = subprocess.Popen(command)
                self.process_instances.wait()  # Wait for the process to complete
                self.pointcloudComputationDone.emit() # Emit the signal on completion

            # Start the monitoring thread
            thread = threading.Thread(target=run)

            thread.start()
            self.computeInstancesButton.setEnabled(False)
            self.stopInstancesButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeInstancesButton.setEnabled(True)
            self.stopInstancesButton.setEnabled(False)

    def stopInstances(self):
        if self.process_instances and self.process_instances.poll() is None:
            self.process_instances.kill()
            self.process_instances = None
            self.pointcloudComputationDone.emit()

    def onInstancesComputationDone(self):
        self.computeInstancesButton.setEnabled(True)
        self.stopInstancesButton.setEnabled(False)
        print("Computation process stopped.")

    def addMeshGeneration(self, layout):
        # Main Collapsible Box for Mesh Generation
        volumeBox = CollapsibleBox("Mesh Generation")

        # Generate Grid Cells Area
        stitchSheetBox = CollapsibleBox("Stitch Sheet")
        self.addStitchSheetArea(stitchSheetBox)
        volumeBox.add_widget(stitchSheetBox)

        # Meshing Area
        meshingBox = CollapsibleBox("Meshing")
        self.addMeshingArea(meshingBox)
        volumeBox.add_widget(meshingBox)

        # Swap Volume Area
        swapVolumeBox = CollapsibleBox("Swap Volume")
        self.addSwapVolumeArea(swapVolumeBox)
        volumeBox.add_widget(swapVolumeBox)

        layout.addWidget(volumeBox)

    def addFieldWithLabel(self, box, labelText, placeholderText, fieldAttribute, value=None):
        # Create a widget to hold the label and field, ensuring proper layout within the collapsible box
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Create and add the label
        label = QLabel(labelText)
        layout.addWidget(label)
        
        # Create, configure, and add the field
        field = QLineEdit()
        field.setPlaceholderText(placeholderText)
        if value:
            field.setText(str(value))
        layout.addWidget(field)
        
        # Add the composite widget to the box
        box.add_widget(widget)
        
        # Store a reference to the field using setattr for future access
        setattr(self, fieldAttribute, field)

    def toggleStartingPointSelection(self):
        self.isSelectingStartingPoint = not self.isSelectingStartingPoint
        if self.isSelectingStartingPoint:
            self.setCursor(Qt.CrossCursor)  # Change cursor to crosshair
        else:
            self.setCursor(Qt.ArrowCursor)  # Reset cursor to default

    def addStartingPointArea(self, box):
        # Starting Point button
        self.startingPointField = QPushButton("Select Starting Point")
        self.startingPointField.clicked.connect(self.toggleStartingPointSelection)
        box.add_widget(self.startingPointField)

        # Title for the starting point section
        title_label = QLabel("Starting Point:")
        box.add_widget(title_label)


        # Horizontal layout to hold the x, y, and z fields
        starting_point_layout = QHBoxLayout()

        # X coordinate field
        self.xField = QLineEdit()
        self.xField.setPlaceholderText("x")
        starting_point_layout.addWidget(QLabel("x:"))
        starting_point_layout.addWidget(self.xField)

        # Y coordinate field
        self.yField = QLineEdit()
        self.yField.setPlaceholderText("y")
        starting_point_layout.addWidget(QLabel("y:"))
        starting_point_layout.addWidget(self.yField)

        # Z coordinate field
        self.zField = QLineEdit()
        self.zField.setPlaceholderText("z")
        starting_point_layout.addWidget(QLabel("z:"))
        starting_point_layout.addWidget(self.zField)

        # Create a container widget to add the layout to
        starting_point_widget = QWidget()
        starting_point_widget.setLayout(starting_point_layout)

        # Add the starting point widget to the box
        box.add_widget(starting_point_widget)


    def setStitchSheetDefaultValues(self):
        self.sheetKRangeStartField.setText("-1")
        self.sheetKRangeEndField.setText("1")
        self.sheetZRangeStartField.setText("0")
        self.sheetZRangeEndField.setText("40000")
        self.windingDirectionField.setText("1")
        self.minStepsField.setText("16")
        self.minEndStepsField.setText("4")
        self.maxNrWalksField.setText("30000")
        self.walkAggregationThresholdField.setText("5")

    def showWindingDirectionInfo(self):
        QMessageBox.information(self, "Winidng Direction Information",
                                "Set the winding direction based on your Scrolls orientation. Scroll 1 is oriented with -1, Scroll 3 with 1.")
        
    def addStitchSheetArea(self, box):
        label = QLabel("Stitch Sheet")
        box.add_widget(label)

        self.addStartingPointArea(box)

        # Sheet K range fields
        self.addFieldWithLabel(box, "Sheet K Range Start:", "Enter start for K range", "sheetKRangeStartField")
        self.addFieldWithLabel(box, "Sheet K Range End:", "Enter end for K range", "sheetKRangeEndField")

        # Sheet Z range fields
        self.addFieldWithLabel(box, "Sheet Z Range Start:", "Enter start for Z range", "sheetZRangeStartField")
        self.addFieldWithLabel(box, "Sheet Z Range End:", "Enter end for Z range", "sheetZRangeEndField")

        # Winding direction field
        # Create an info button
        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(self.showWindingDirectionInfo)
        box.add_widget(infoButton) 
        self.addFieldWithLabel(box, "Winding Direction:", "Enter winding direction", "windingDirectionField")

        # Other parameter fields
        self.addFieldWithLabel(box, "Min Steps:", "Enter minimum steps", "minStepsField")
        self.addFieldWithLabel(box, "Min End Steps:", "Enter minimum end steps", "minEndStepsField")
        self.addFieldWithLabel(box, "Max Nr Walks:", "Enter maximum number of walks", "maxNrWalksField")
        self.addFieldWithLabel(box, "Walk Aggregation Threshold:", "Enter walk aggregation threshold", "walkAggregationThresholdField")


        self.recomputeStitchSheetCheckbox = QCheckBox("Recompute")
        self.continueSegmentationCheckbox = QCheckBox("Continue Segmentation")
        self.computeStitchSheetButton = QPushButton("Compute")
        self.stopStitchSheetButton = QPushButton("Stop")
        self.stopStitchSheetButton.setEnabled(False)

        self.defaultValuesButton = QPushButton("Default Values")
        self.defaultValuesButton.clicked.connect(self.setStitchSheetDefaultValues)

        self.computeStitchSheetButton.clicked.connect(self.computeStitchSheet)
        self.stopStitchSheetButton.clicked.connect(self.stopStitchSheet)

        box.add_widget(self.recomputeStitchSheetCheckbox)
        box.add_widget(self.continueSegmentationCheckbox)
        box.add_widget(self.computeStitchSheetButton)
        box.add_widget(self.stopStitchSheetButton)
        box.add_widget(self.defaultValuesButton)

        # Connect the signal to the slot method
        self.stitchSheetComputationDone.connect(self.onStitchSheetComputationDone)

    def stitchSheetScriptComputation(self, overlapp_threshold, start_point, path, recompute, stop_event):
        try:

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.Random_Walks",
                "--path", path,
                "--recompute", str(int(recompute)),
                "--starting_point", str(start_point[0]), str(start_point[1]), str(start_point[2]),
                "--winding_direction", str(int(overlapp_threshold["winding_direction"])),
                "--sheet_z_range", str(int(overlapp_threshold["sheet_z_range"][0])), str(int(overlapp_threshold["sheet_z_range"][1])),
                "--sheet_k_range", str(int(overlapp_threshold["sheet_k_range"][0])), str(int(overlapp_threshold["sheet_k_range"][1])),
                "--walk_aggregation_threshold", str(overlapp_threshold["walk_aggregation_threshold"]),
                "--max_nr_walks", str(overlapp_threshold["max_nr_walks"]),
                "--max_unchanged_walks", str(overlapp_threshold["max_unchanged_walks"]),
                "--continue_segmentation", str(int(overlapp_threshold["continue_walks"])),
                "--min_steps", str(overlapp_threshold["min_steps"]),
                "--min_end_steps", str(overlapp_threshold["min_end_steps"])
            ]
            
            def run():
                self.process_Random_Walks = subprocess.Popen(command)
                self.process_Random_Walks.wait()
                self.stitchSheetComputationDone.emit()  # Emit the signal on completion

            # Start the monitoring thread
            thread = threading.Thread(target=run)

            thread.start()
            self.computeStitchSheetButton.setEnabled(False)
            self.stopStitchSheetButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeStitchSheetButton.setEnabled(True)
            self.stopStitchSheetButton.setEnabled(False)


    def computeStitchSheet(self):
        try:
            # Fetching values from GUI fields
            path = os.path.join(self.Config["surface_points_path"], "point_cloud_colorized_verso_subvolume_blocks")
            start_point = [int(self.xField.text()), int(self.yField.text()), int(self.zField.text())]
            sheet_k_range = (int(self.sheetKRangeStartField.text()), int(self.sheetKRangeEndField.text()))
            sheet_z_range = (int(self.sheetZRangeStartField.text()), int(self.sheetZRangeEndField.text()))
            winding_direction = int(self.windingDirectionField.text())
            min_steps = int(self.minStepsField.text())
            min_end_steps = int(self.minEndStepsField.text())
            max_nr_walks = int(self.maxNrWalksField.text())
            continue_segmentation = self.continueSegmentationCheckbox.isChecked()
            recompute = self.recomputeStitchSheetCheckbox.isChecked()
            walk_aggregation_threshold = int(self.walkAggregationThresholdField.text())


            overlapp_threshold = {"sample_ratio_score": 0.03, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.0005, "score_threshold": 0.005, "fit_sheet": False, "cost_threshold": 17, "cost_percentile": 75, "cost_percentile_threshold": 14, 
                          "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.005,
                          "cost_threshold_prediction": 2.5, "min_prediction_threshold": 0.15, "nr_points_min": 200.0, "nr_points_max": 4000.0, "min_patch_points": 300.0, 
                          "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
                          "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
                          "min_points_winding_switch": 1000, "min_winding_switch_sheet_distance": 3, "max_winding_switch_sheet_distance": 10, "winding_switch_sheet_score_factor": 1.5, "winding_direction": 1.0,
                          "enable_winding_switch": True, "max_same_block_jump_range": 3,
                          "enable_winding_switch_postprocessing": False,
                          "surrounding_patches_size": 3, "max_sheet_clip_distance": 60, "sheet_z_range": (-5000, 400000), "sheet_k_range": (-1, 2), "volume_min_certainty_total_percentage": 0.0, "max_umbilicus_difference": 30,
                          "walk_aggregation_threshold": 100, "walk_aggregation_max_current": -1,
                          "bad_edge_threshold": 3
                          }

            # max_nr_walks = 10000
            max_steps = 101
            # min_steps = 16
            max_tries = 6
            # min_end_steps = 4
            max_unchanged_walks = 30 * max_nr_walks

            overlapp_threshold["sheet_z_range"] = [z_range_ /(200.0 / 50.0) for z_range_ in sheet_z_range]
            overlapp_threshold["sheet_k_range"] = sheet_k_range
            overlapp_threshold["winding_direction"] = winding_direction
            overlapp_threshold["walk_aggregation_threshold"] = walk_aggregation_threshold
            overlapp_threshold["max_nr_walks"] = max_nr_walks
            overlapp_threshold["max_unchanged_walks"] = max_unchanged_walks
            overlapp_threshold["continue_walks"] = continue_segmentation
            overlapp_threshold["max_steps"] = max_steps
            overlapp_threshold["max_tries"] = max_tries
            overlapp_threshold["min_steps"] = min_steps
            overlapp_threshold["min_end_steps"] = min_end_steps

            # Creating an Event object
            self.stop_event = threading.Event()

            # start self.stitchSheetComputation in a new thread
            self.process_stitching = threading.Thread(target=self.stitchSheetScriptComputation, args=(overlapp_threshold, start_point, path, recompute, self.stop_event))
            self.process_stitching.start()

            self.computeStitchSheetButton.setEnabled(False)
            self.stopStitchSheetButton.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeStitchSheetButton.setEnabled(True)
            self.stopStitchSheetButton.setEnabled(False)

    def stopStitchSheet(self):
        # Kill the thread
        self.stop_event.set()
        self.process_stitching.join()

        self.onStitchSheetComputationDone()

    def onStitchSheetComputationDone(self):
        self.computeStitchSheetButton.setEnabled(True)
        self.stopStitchSheetButton.setEnabled(False)
        print("Computation process stopped.")

    def addMeshingArea(self, box):
        label = QLabel("Meshing")
        self.computeMeshingButton = QPushButton("Compute")
        self.stopMeshingButton = QPushButton("Stop")
        self.stopMeshingButton.setEnabled(False)

        self.computeMeshingButton.clicked.connect(self.computeMeshing)
        self.stopMeshingButton.clicked.connect(self.stopMeshing)

        box.add_widget(label)
        box.add_widget(self.computeMeshingButton)
        box.add_widget(self.stopMeshingButton)

        # Connect the signal to the slot method
        self.meshingComputationDone.connect(self.onMeshingComputationDone)

    def computeMeshing(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            path = os.path.join(self.Config["surface_points_path"], "point_cloud_colorized_verso_subvolume_blocks")
            graph = f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}/point_cloud_colorized_verso_subvolume_graph_RW_solved.pkl"

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.graph_to_mesh",
                "--path", path,
                "--graph", graph,
                "--start_point", str(int(starting_point[0])), str(int(starting_point[1])), str(int(starting_point[2])),
                "--scale_factor", f"{abs(self.Config['downsample_factor']):f}"
            ]
            
            def run():
                self.process_meshing = subprocess.Popen(command)
                self.process_meshing.wait()
                self.meshingComputationDone.emit()

            # Start the monitoring thread
            thread = threading.Thread(target=run)

            thread.start()
            self.computeMeshingButton.setEnabled(False)
            self.stopMeshingButton.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeMeshingButton.setEnabled(True)
            self.stopMeshingButton.setEnabled(False)

    def stopMeshing(self):
        if self.process_meshing and self.process_meshing.poll() is None:
            self.process_meshing.kill()
            self.process_meshing = None
            self.meshingComputationDone.emit()

    def onMeshingComputationDone(self):
        self.computeMeshingButton.setEnabled(True)
        self.stopMeshingButton.setEnabled(False)
        print("Computation process stopped.")

    def addSwapVolumeArea(self, box):
        label = QLabel("Swap Volume")
        self.targetVolumeIdField = QLineEdit()
        self.targetVolumeIdField.setPlaceholderText("Target Volume ID")
        self.computeSwapVolumeButton = QPushButton("Compute")
        self.stopSwapVolumeButton = QPushButton("Stop")
        self.stopSwapVolumeButton.setEnabled(False)

        self.computeSwapVolumeButton.clicked.connect(self.computeSwapVolume)
        self.stopSwapVolumeButton.clicked.connect(self.stopSwapVolume)

        box.add_widget(label)
        box.add_widget(self.targetVolumeIdField)
        box.add_widget(self.computeSwapVolumeButton)
        box.add_widget(self.stopSwapVolumeButton)

        # Connect the signal to the slot method
        self.swapVolumeComputationDone.connect(self.onSwapVolumeComputationDone)

    def computeSwapVolume(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            original_volume_id = self.Config.get("original_2d_tiffs", None)
            if original_volume_id is None:
                QMessageBox.critical(self, "Error", f"Please specify the 2D Tiff files path")
                return
            original_volume_id = os.path.basename(original_volume_id)

            target_volume_id = self.targetVolumeIdField.text()
            base_path = self.Config.get("surface_points_path", None)
            if base_path is None:
                QMessageBox.critical(self, "Error", f"Please specify the surface points path")
                return
            obj_path =  os.path.join(base_path, f"{starting_point[0]}_{starting_point[1]}_{starting_point[2]}", "point_cloud_colorized_verso_subvolume_blocks/mesh_flatboi.obj")

            command = [
                "python3", "-m", "ThaumatoAnakalyptor.mesh_transform", 
                "--transform_path", self.Config["surface_points_path"], 
                "--original_volume_id", original_volume_id,
                "--targed_volume_id", target_volume_id,
                "--obj_path", obj_path
            ]

            def run():
                self.process_swap = subprocess.Popen(command)
                self.process_swap.wait()
                self.swapVolumeComputationDone.emit()

            # Start the monitoring thread
            thread = threading.Thread(target=run)
            
            thread.start()
            self.computeSwapVolumeButton.setEnabled(False)
            self.stopSwapVolumeButton.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeSwapVolumeButton.setEnabled(True)
            self.stopSwapVolumeButton.setEnabled(False)

    def stopSwapVolume(self):
        if self.process_swap and self.process_swap.poll() is None:
            self.process_swap.kill()
            self.process_swap = None
            self.swapVolumeComputationDone.emit()

    def onSwapVolumeComputationDone(self):
        self.computeSwapVolumeButton.setEnabled(True)
        self.stopSwapVolumeButton.setEnabled(False)
        print("Computation process stopped.")

    def addRendering(self, layout):        
        # Main Collapsible Box for Rendering
        renderingBox = CollapsibleBox("Rendering")

        # Texturing Area
        self.addTexturingArea(renderingBox)

        layout.addWidget(renderingBox)

    def addTexturingArea(self, box):
        label = QLabel("Texturing")
        self.addFieldWithLabel(box, "Output Format:", "Enter format", "formatRenderField", value="jpg")
        self.computeTexturingButton = QPushButton("Compute")
        self.stopTexturingButton = QPushButton("Stop")
        self.stopTexturingButton.setEnabled(False)

        self.computeTexturingButton.clicked.connect(self.computeTexturing)
        self.stopTexturingButton.clicked.connect(self.stopTexturing)

        box.add_widget(label)
        box.add_widget(self.computeTexturingButton)
        box.add_widget(self.stopTexturingButton)

        # Connect the signal to the slot method
        self.texturingComputationDone.connect(self.onTexturingComputationDone)

    def computeTexturing(self):
        try:
            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            meshes_path = os.path.join(self.Config["surface_points_path"], f"{int(starting_point[0])}_{int(starting_point[1])}_{int(starting_point[2])}", "point_cloud_colorized_verso_subvolume_blocks", "windowed_mesh")
            # Get all *_flatboi.obj files in the meshes_path
            obj_files = [f for f in os.listdir(meshes_path) if f.endswith("_flatboi.obj")]
            print(f"obj_files: {obj_files}")

            for obj in obj_files:
                mesh_path = os.path.join(meshes_path, obj)
                print(mesh_path)
                command_gpu = [
                    "python3", "-m", "ThaumatoAnakalyptor.large_mesh_to_surface",
                    "--input_mesh", mesh_path,
                    "--scroll", self.Config.get("original_render_volume", None),
                    "--gpus", str(self.Config["gpus"]),
                    "--format", self.formatRenderField.text(),
                    "--display"
                ]

                # Prepare the environment variable
                env = os.environ.copy()
                env["MAX_TILE_SIZE"] = "4294967295"  # Set maximum CV io image size for tiff files
                env["OPENCV_IO_MAX_IMAGE_PIXELS"] = "4294967295"  # Set maximum CV io image size for tiff files
                env["CV_IO_MAX_IMAGE_PIXELS"] = "4294967295"  # Set maximum CV io image size for tiff files
                
                def run():
                    # Run the command with the specified environment variables
                    self.process_texturing = subprocess.Popen(command_gpu, env=env)
                    self.process_texturing.wait()

                # Start the monitoring thread
                thread = threading.Thread(target=run)

                thread.start()

            self.texturingComputationDone.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeTexturingButton.setEnabled(True)
            self.stopTexturingButton.setEnabled(False)

    def stopTexturing(self):
        if self.process_texturing and self.process_texturing.poll() is None:
            self.process_texturing.kill()
            self.process_texturing = None
            self.texturingComputationDone.emit()

    def onTexturingComputationDone(self):
        self.computeTexturingButton.setEnabled(True)
        self.stopTexturingButton.setEnabled(False)
        print("Computation process stopped.")

    def addInkDetection(self, layout):
        # Main Collapsible Box for Ink Detection
        renderingBox = CollapsibleBox("Ink Detection")

        # Ink Detection Area
        timesformerBox = CollapsibleBox("TimeSformer Inference")
        self.addInkDetectionArea(timesformerBox)
        renderingBox.add_widget(timesformerBox)

        layout.addWidget(renderingBox)

    def addInkDetectionArea(self, box):
        label = QLabel("Ink Detection")
        self.computeInkDetectionButton = QPushButton("Compute")
        self.stopInkDetectionButton = QPushButton("Stop")
        self.stopInkDetectionButton.setEnabled(False)

        self.computeInkDetectionButton.clicked.connect(self.computeInkDetection)
        self.stopInkDetectionButton.clicked.connect(self.stopInkDetection)

        box.add_widget(label)
        box.add_widget(self.computeInkDetectionButton)
        box.add_widget(self.stopInkDetectionButton)

        # Connect the signal to the slot method
        self.inkDetectionComputationDone.connect(self.onInkDetectionComputationDone)

    def computeInkDetection(self):
        try:
            # Set the path to the virtual environment's Python executable
            python_executable = "/youssefGP/bin/python"  # Python env for the timesformer
            script_name = "/workspace/Vesuvius-Grandprize-Winner/inference_timesformer.py"  # Path to the script to run
            
            # Prepare the environment variable
            env = os.environ.copy()
            env["WANDB_MODE"] = "dryrun"  # Set the WANDB_MODE environment variable

            # Modify the segment_path as per your requirement or dynamically determine it
            segment_path = self.Config.get("surface_points_path", None)
            if segment_path is None:
                QMessageBox.critical(self, "Error", f"Please specify the surface points path")
                return
            segment_path = os.path.join(segment_path, "working")
            model_path = "/workspace/Vesuvius-Grandprize-Winner/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt"

            starting_point = [self.xField.text(), self.yField.text(), self.zField.text()]
            working_directories = [dir_ for dir_ in os.listdir(segment_path) if dir_.startswith(f"working_{starting_point[0]}_{starting_point[1]}_{starting_point[2]}")]
            
            self.computeInkDetectionButton.setEnabled(False)
            self.stopInkDetectionButton.setEnabled(True)

            for segment_id in working_directories:
                out_path = os.path.join(segment_path, segment_id, "predictions")

                # Construct the command with the provided arguments
                command = [
                    python_executable, script_name,
                    "--segment_id", segment_id,
                    "--segment_path", segment_path,
                    "--model_path", model_path,
                    "--out_path", out_path
                ]

                def run():
                    self.process_inkdetection = subprocess.Popen(command, env=env)
                    self.process_inkdetection.wait()
                    self.inkDetectionComputationDone.emit()

                # Start the monitoring thread
                thread = threading.Thread(target=run)

                thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start the script: {e}")
            self.computeInkDetectionButton.setEnabled(True)
            self.stopInkDetectionButton.setEnabled(False)

    def stopInkDetection(self):
        if self.process_inkdetection and self.process_inkdetection.poll() is None:
            self.process_inkdetection.kill()
            self.process_inkdetection = None
            self.inkDetectionComputationDone.emit()

    def onInkDetectionComputationDone(self):
        self.computeInkDetectionButton.setEnabled(True)
        self.stopInkDetectionButton.setEnabled(False)
        print("Computation process stopped.")