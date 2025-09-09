### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import (QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QLineEdit, QCheckBox, QMessageBox, QStyle, QDialog, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QIntValidator, QDoubleValidator

import os
import json

class ConfigWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Configuration")
        self.layout = QVBoxLayout(self)

        # Adding specific configuration fields
        self.original_2d_tiffs_field = self.addPathConfigField("Canonical 2D TIFFs", "Enter the path to the folder containing Canonical original 2D TIFF files for the volume", self.parent.Config.get('original_2d_tiffs', ''))
        self.original_render_volume_field = self.addPathConfigField("Canonical Render Volume", "Enter the path to the folder containing Canonical original Render Volume files for the volume. Grid Cells or .zarr", self.parent.Config.get('original_render_volume', ''))
        self.downsample_factor = self.addIntegerField("Downsample Factor", "Enter the downsample factor (whole number) to achieve close to 8um resolution. Negative numbers will allow you to set the 2D and 3D TIFF paths manually and are treated like their positive analogs", str(self.parent.Config.get('downsample_factor', '')))
        self.downsampled_2d_tiffs_field = self.addPathConfigField("Downsampled 2D TIFFs", "Enter the path to the folder containing downsampled 2D TIFF files", self.parent.Config.get('downsampled_2d_tiffs', ''))
        self.downsampled_3d_grids_field = self.addPathConfigField("Downsampled 3D Grid Cells", "Enter the path to the folder containing downsampled 3D Grid Cells", self.parent.Config.get('downsampled_3d_grids', ''))
        self.surface_points_path_field = self.addPathConfigField("Surface Points Path", "Enter the surface points path", self.parent.Config.get('surface_points_path', ''))
        self.addIntegerField("Num Threads", "Enter the number of threads for processing", str(self.parent.Config.get('num_threads', '4')))
        self.addIntegerField("GPUs", "Enter the number of GPUs to use", str(self.parent.Config.get('gpus', '1')))
        self.addIntegerField("Batch Size", "Enter the batch size", str(self.parent.Config.get('batch_size', '4')))
        self.addIntegerField("Num Threads Texturing", "Enter the number of threads for texturing", str(self.parent.Config.get('num_threads_texturing', '4')))

        # Connect the config fields to actions
        self.original_2d_tiffs_field.textChanged.connect(lambda _: self.onDownsampleFactorChanged(self.downsample_factor.text()))

        self.onDownsampleFactorChanged(self.downsample_factor.text())
        self.downsample_factor.textChanged.connect(self.onDownsampleFactorChanged)

        self.downsampled_2d_tiffs_field.textChanged.connect(self.onDownsampled2DTiffsChanged)

        saveButton = QPushButton("Save", self)
        saveButton.clicked.connect(self.saveConfig)
        self.layout.addWidget(saveButton)

    def addPathConfigField(self, label_text, help_text, default_value=''):
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        hbox.addWidget(label)

        field = QLineEdit()
        field.setPlaceholderText(help_text)
        field.setText(default_value)
        field.mousePressEvent = lambda event: self.selectPath(field)
        hbox.addWidget(field)

        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(lambda: QMessageBox.information(self, label_text, help_text))
        infoButton.setFixedSize(20, 20)  # Set a fixed size for uniformity
        hbox.addWidget(infoButton)

        self.layout.addLayout(hbox)
        setattr(self, label_text.replace(" ", "_").lower() + "_field", field)

        return field

    def addIntegerField(self, label_text, help_text, default_value=''):
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        hbox.addWidget(label)

        field = QLineEdit()
        field.setPlaceholderText(help_text)
        field.setText(default_value)
        field.setValidator(QIntValidator())
        hbox.addWidget(field)

        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(lambda: QMessageBox.information(self, label_text, help_text))
        infoButton.setFixedSize(20, 20)  # Set a fixed size for uniformity
        hbox.addWidget(infoButton)

        self.layout.addLayout(hbox)
        setattr(self, label_text.replace(" ", "_").lower() + "_field", field)

        return field

    def addFloatField(self, label_text, help_text, default_value=''):
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        hbox.addWidget(label)

        field = QLineEdit()
        field.setPlaceholderText(help_text)
        field.setText(default_value)
        field.setValidator(QDoubleValidator())
        hbox.addWidget(field)

        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(lambda: QMessageBox.information(self, label_text, help_text))
        infoButton.setFixedSize(20, 20)  # Set a fixed size for uniformity
        hbox.addWidget(infoButton)

        self.layout.addLayout(hbox)
        setattr(self, label_text.replace(" ", "_").lower() + "_field", field)

        return field

    def addFlagField(self, label_text, help_text, default_value=False):
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        hbox.addWidget(label)

        field = QCheckBox()
        field.setText(default_value)
        hbox.addWidget(field)

        infoButton = QPushButton()
        infoButton.setIcon(infoButton.style().standardIcon(QStyle.SP_MessageBoxInformation))
        infoButton.clicked.connect(lambda: QMessageBox.information(self, label_text, help_text))
        infoButton.setFixedSize(20, 20)  # Set a fixed size for uniformity
        hbox.addWidget(infoButton)

        self.layout.addLayout(hbox)
        setattr(self, label_text.replace(" ", "_").lower() + "_field", field)

        return field

    def selectPath(self, lineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory", lineEdit.text() if lineEdit.text() else None)
        if path:
            lineEdit.setText(path)

    def showInfo(self, label_text):
        infoTexts = {
            "Umbilicus Path": "Enter the path to the folder containing 8um downsampled 2D TIFF files."
            # Add info texts for other fields as needed
        }
        QMessageBox.information(self, label_text + " Information", infoTexts.get(label_text, ""))

    def onDownsampleFactorChanged(self, text):
        try:
            factor = int(text)
            self.downsampled_3d_grids_field.setDisabled(True)
            if factor == 1:
                self.downsampled_2d_tiffs_field.setText(self.original_2d_tiffs_field.text())
                self.downsampled_2d_tiffs_field.setDisabled(True)
            elif factor <= 0:
                self.downsampled_3d_grids_field.setDisabled(False)
                self.downsampled_2d_tiffs_field.setEnabled(True)
            else:
                self.downsampled_2d_tiffs_field.setEnabled(True)
                if self.original_2d_tiffs_field.text() == self.downsampled_2d_tiffs_field.text():
                    self.downsampled_2d_tiffs_field.setText(None)

        except ValueError:
            self.downsampled_2d_tiffs_field.setEnabled(True)

    def onDownsampled2DTiffsChanged(self, text):
        if not text or text == "":
            self.downsampled_3d_grids_field.setText(None)
        else:
            # Set the default value for downsampled 3D Grid Cells
            self.downsampled_3d_grids_field.setText(text + "_grids")

    def saveConfig(self):
        config = {
            "original_2d_tiffs": self.original_2d_tiffs_field.text(),
            "original_render_volume": self.original_render_volume_field.text(),
            "downsample_factor": int(self.downsample_factor_field.text()) if self.downsample_factor_field.text() and self.downsample_factor_field.text() != 'None' else None,
            "downsampled_2d_tiffs": self.downsampled_2d_tiffs_field.text(),
            "downsampled_3d_grids": self.downsampled_3d_grids_field.text(),
            "surface_points_path": self.surface_points_path_field.text(),
            "num_threads": int(self.num_threads_field.text()) if self.num_threads_field.text() and self.num_threads_field.text() != 'None' else None,
            "gpus": int(self.gpus_field.text()) if self.gpus_field.text() and self.gpus_field.text() != 'None' else None,
            "batch_size": int(self.batch_size_field.text()) if self.batch_size_field.text() and self.batch_size_field.text() != 'None' else None,
            "num_threads_texturing": int(self.num_threads_texturing_field.text()) if self.num_threads_texturing_field.text() and self.num_threads_texturing_field.text() != 'None' else None
        }

        # Remove empty fields
        for key in list(config.keys()):
            if config[key] is None:
                del config[key]

        # Compute constructed field umbilicus path
        if config.get("downsampled_2d_tiffs", None) is not None:
            config["umbilicus_path"] = os.path.join(config["downsampled_2d_tiffs"], "umbilicus.txt")

        with open("config.json", "w") as file:
            json.dump(config, file)

        self.parent.Config = config
        self.close()
