### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QMainWindow, QAction, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QHBoxLayout,
                             QLineEdit, QGraphicsView, QGraphicsScene, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QPen, QBrush, QIcon
import tifffile
import zarr
import numpy as np
import os

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

class UmbilicusWindow(QMainWindow):
    def __init__(self, imagePath, scale_factor, parent=None):
        super().__init__(parent)
        self.imagePath = imagePath
        self.scale_factor = scale_factor
        self.currentIndex = 0
        self.incrementing = True
        # Zarr multiscale support
        self.is_zarr = False
        self.zarr_group = None
        self.pyramid_levels = {}
        # current pyramid level (0 = full resolution)
        self.pyr_level = 0
        # cache last loaded slice and level
        self.index_old = -1
        self.last_pyr_level = None
        self.construct_images()
        self.points = {}  # Dictionary to store points as {index: (x, y)}
        self.initUI()
        # set icon
        icon = QIcon("GUI/ThaumatoAnakalyptor.png")
        self.setWindowIcon(icon)

    def initUI(self):
        self.setWindowTitle("Generate Umbilicus")

        # Menu Bar
        menubar = self.menuBar()
        helpMenu = QAction('Help', self)
        helpMenu.triggered.connect(self.showHelp)  # Connect the triggered signal to showHelp method
        menubar.addAction(helpMenu)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Graphics View for Image Display
        self.scene = QGraphicsScene(self)
        self.view = GraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        # Set the focus policy to accept key events
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.setMouseTracking(True)  # Enable mouse tracking if needed

        self.view.mousePressEvent = self.viewMousePressEvent
        layout.addWidget(self.view)

        # Step Size and Index Input Area
        stepSizeLayout = QHBoxLayout()
        self.fileNameLabel = QLabel("File: " + self.images[self.currentIndex])
        stepSizeLabel = QLabel("Step Size:")
        self.stepSizeBox = QLineEdit("100")  # Set default value
        self.stepSizeBox.setFixedWidth(100)  # Adjust width as appropriate
        self.stepSizeBox.returnPressed.connect(lambda: self.view.setFocus())  # Unfocus on Enter
        # Pyramid level selection for zarr multiscale (0 = full res, 1 = half, ...)
        pyrLabel = QLabel("Level:")
        self.pyrLevelBox = QLineEdit(str(self.pyr_level))
        self.pyrLevelBox.setFixedWidth(50)
        self.pyrLevelBox.returnPressed.connect(self.changePyramidLevel)

        indexLabel = QLabel("Index:")
        self.indexBox = QLineEdit()
        self.indexBox.setFixedWidth(100)  # Adjust width as appropriate
        self.indexBox.returnPressed.connect(self.jumpToIndex)  # Jump to index on Enter
        self.indexBox.setText(str(self.currentIndex))

        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.loadPoints)
        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.savePoints)

        stepSizeLayout.addWidget(self.fileNameLabel)
        stepSizeLayout.addWidget(stepSizeLabel)
        stepSizeLayout.addWidget(self.stepSizeBox)
        # add pyramid level controls
        stepSizeLayout.addWidget(pyrLabel)
        stepSizeLayout.addWidget(self.pyrLevelBox)
        stepSizeLayout.addWidget(indexLabel)
        stepSizeLayout.addWidget(self.indexBox)
        stepSizeLayout.addWidget(loadButton)
        stepSizeLayout.addWidget(saveButton)
        layout.addLayout(stepSizeLayout)

        # Load the first image
        self.loadImage(self.currentIndex)

        # Set focus to the QGraphicsView
        self.view.setFocus()

        # Load the first image
        self.loadImage(self.currentIndex)

    def construct_images(self):
        self.images = {}
        if self.imagePath.endswith('.zarr'):
            # ome-zarr multiscale volume
            print("Loading zarr volume")
            self.is_zarr = True
            # open zarr group or array
            zarr_obj = zarr.open(self.imagePath)
            if hasattr(zarr_obj, 'keys'):
                # multiscale group with levels '0', '1', ...
                level_keys = sorted([int(k) for k in zarr_obj.keys()])
                for level in level_keys:
                    self.pyramid_levels[level] = zarr_obj[str(level)]
            else:
                # single-scale array
                self.pyramid_levels[0] = zarr_obj
            # use level 0 for z dimension count
            arr0 = self.pyramid_levels[0]
            z_count = arr0.shape[0]
            for i in range(z_count):
                self.images[i] = f"zarr z-slice {i}"
        else:
            tifs = [f for f in os.listdir(self.imagePath) if f.endswith('.tif')]
            for tif in tifs:
                tif_name = tif.split("_")[-1]
                self.images[int(tif_name[:-4])] = tif

    def showHelp(self):
        helpText = "ThaumatoAnakalyptor Help\n\n" \
                   "If you already have an umbilicus generated, load it with 'Load'. \n" \
                   "Place the umbilicus points in the center of the scroll and when done press 'Save' before closing the window.\n\n" \
                   "There are the following shortcuts:\n\n" \
                   "- Use Mouse Wheel to zoom in and out.\n" \
                   "- Use 'A' and 'D' to switch between TIFF layers.\n" \
                   "- Use 'Ctrl + A' and 'Ctrl + D' to switch between TIFF layers with umbilicus points.\n" \
                   "- Click on the TIFF to place a point.\n" \
                   "- Use Ctrl + Click to automatically switch to the next TIFF.\n"
        QMessageBox.information(self, "Help", helpText)
    
    def changePyramidLevel(self):
        """
        Change the pyramid level for zarr multiscale and reload current image.
        """
        try:
            level = int(self.pyrLevelBox.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid pyramid level.")
            self.pyrLevelBox.setText(str(self.pyr_level))
            return
        if level not in self.pyramid_levels:
            QMessageBox.warning(self, "Warning", f"Pyramid level {level} not available.")
            self.pyrLevelBox.setText(str(self.pyr_level))
            return
        # set new level and reload
        self.pyr_level = level
        self.loadImage(self.currentIndex)

    def loadImage(self, index_original):
        """
        Load and display the image slice at given downsampled index and pyramid level.
        """
        # compute actual slice index
        index_z = index_original * self.scale_factor

        # only proceed if index valid
        if index_z not in self.images:
            print(f"Image at index {index_z} not found in images dictionary.")
            return

        # determine if reload needed (new slice or new pyramid level)
        reload_needed = (self.index_old != index_z) or (self.is_zarr and self.last_pyr_level != self.pyr_level)
        if reload_needed:
            # load image array from source
            if self.is_zarr:
                print(f"Loading zarr volume image at level {self.pyr_level}, index {index_z}")
                arr = self.pyramid_levels.get(self.pyr_level, self.pyramid_levels.get(0))
                scale_xy = 2 ** self.pyr_level
                z_scaled = index_z // scale_xy
                image_array = arr[z_scaled]
            else:
                path = os.path.join(self.imagePath, self.images[index_z])
                with tifffile.TiffFile(path) as tif:
                    image_array = tif.asarray()
            # cache loaded data
            self.image = image_array
            self.index_old = index_z
            self.last_pyr_level = self.pyr_level
        else:
            image_array = self.image

        # convert to numpy array
        image_array = np.array(image_array)
        if image_array.dtype == np.uint16:
            image_array = (image_array / 256).astype(np.uint8)
        if image_array.dtype == np.float16:
            print("float16, experimental")
            image_array = (image_array * 256).astype(np.uint8)

        # upscale XY if using lower resolution pyramid
        if self.is_zarr and self.pyr_level > 0:
            scale_xy = 2 ** self.pyr_level
            image_array = np.repeat(np.repeat(image_array, scale_xy, axis=0), scale_xy, axis=1)

        # prepare for display
        image_height, image_width = image_array.shape
        self.image_width = image_width
        self.image_height = image_height
        qimage = QImage(image_array.data, image_width, image_height, image_width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # update scene
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        # update UI elements
        self.indexBox.setText(str(index_original))
        self.fileNameLabel.setText("File: " + self.images[index_z])

        # draw umbilicus point if exists
        if index_original in self.points:
            x, y = self.points[index_original]
            sceneX = x / self.image_width * self.image_width
            sceneY = y / self.image_height * self.image_height
            size_display = 10
            size_image = size_display / self.view.transform().m11()
            sceneX -= size_image / 2
            sceneY -= size_image / 2
            self.scene.addEllipse(sceneX, sceneY, size_image, size_image, QPen(Qt.red), QBrush(Qt.red))

    def jumpToIndex(self):
        index = int(self.indexBox.text())
        if 0 <= index <= max(self.images.keys()) // self.scale_factor:
            self.currentIndex = index
            self.loadImage(self.currentIndex)
        # Unfocus the index box
        self.view.setFocus()

    def incrementIndex(self):
        self.incrementing = True
        step_size = int(self.stepSizeBox.text())
        self.currentIndex = min((self.currentIndex + step_size), max(self.images.keys()) // self.scale_factor)

    def decrementIndex(self):
        self.incrementing = False
        step_size = int(self.stepSizeBox.text())
        self.currentIndex = max((self.currentIndex - step_size), 0)

    def keyPressEvent(self, event: QKeyEvent):
        # "A" "D" keys for navigation
        if event.key() == Qt.Key_D:
            if event.modifiers() == Qt.ControlModifier:
                # find next image with umbilicus
                next_index = None
                for key in self.points.keys():
                    if key > self.currentIndex and (next_index is None or key < next_index):
                        next_index = key
                if next_index is not None:
                    self.currentIndex = next_index
            else:
                self.incrementIndex()
        elif event.key() == Qt.Key_A:
            if event.modifiers() == Qt.ControlModifier:
                # find previous image with umbilicus
                prev_index = None
                for key in self.points.keys():
                    if key < self.currentIndex and (prev_index is None or key > prev_index):
                        prev_index = key
                if prev_index is not None:
                    self.currentIndex = prev_index
            else:
                self.decrementIndex()
        self.loadImage(self.currentIndex)

    def viewMousePressEvent(self, event):
        if self.scene.itemsBoundingRect().width() > 0:
            scenePos = self.view.mapToScene(event.pos())
            imgRect = self.scene.itemsBoundingRect()
            # Calculate the proportional coordinates
            originalX = int((scenePos.x() / imgRect.width()) * self.image_width)
            originalY = int((scenePos.y() / imgRect.height()) * self.image_height)
            self.points[self.currentIndex] = (originalX, originalY)

            if event.modifiers() == Qt.ControlModifier:
                # Go to next image
                if self.incrementing:
                    self.incrementIndex()
                else:
                    self.decrementIndex()
            self.loadImage(self.currentIndex)

    def loadPoints(self):
        self.points = {}
        umbilicus_name = "umbilicus.txt"
        umbilicus_path = os.path.join(self.imagePath, umbilicus_name)
        if os.path.exists(umbilicus_path):
            with open(umbilicus_path, "r") as file:
                for line in file:
                    y, index, x = map(int, line.strip().split(', '))
                    self.points[index-500] = (x-500, y-500)
            self.loadImage(self.currentIndex)
            print("Points loaded from umbilicus.txt")


    def savePoints(self):
        umbilicus_name = "umbilicus.txt"
        try:
            umbilicus_path = os.path.join(self.imagePath, umbilicus_name)
            print(umbilicus_path)
            point_keys = list(self.points.keys())
            # sort list
            point_keys.sort()
            with open(umbilicus_path, "w") as file:
                for index in point_keys:
                    x, y = self.points[index]
                    file.write(f"{y + 500}, {index + 500}, {x + 500}\n")

            umbilicus_folder = os.path.dirname(self.imagePath) if self.imagePath.endswith('.zarr') else self.imagePath + "_grids"
            umbilicus_path = os.path.join(umbilicus_folder, umbilicus_name)
            print(umbilicus_path)
            with open(umbilicus_path, "w") as file:
                for index in point_keys:
                    x, y = self.points[index]
                    file.write(f"{y + 500}, {index + 500}, {x + 500}\n")
            print("Points saved to umbilicus.txt")
        except Exception as e:
            print(e)
            # Error popup
            QMessageBox.critical(self, "Error", "Could not save points to umbilicus.txt.")

    def resizeEvent(self, event):
        if self.scene.itemsBoundingRect().width() > 0:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
