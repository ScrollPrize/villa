from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel, QDoubleSpinBox

def create_sync_slider_spinbox(label_text, min_val, max_val, default_val, scaleFactor, callback, decimals=2):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * scaleFactor))
        slider.setMaximum(int(max_val * scaleFactor))
        slider.setValue(int(default_val * scaleFactor))
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(decimals)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(1.0)
        slider.valueChanged.connect(lambda val: (spinbox.blockSignals(True), spinbox.setValue(val / scaleFactor),
                                                  spinbox.blockSignals(False), callback()))
        spinbox.valueChanged.connect(lambda val: (slider.blockSignals(True), slider.setValue(int(val * scaleFactor)),
                                                   slider.blockSignals(False), callback()))
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        return container, slider, spinbox